"""Two-class loss for the PCB1 baseline: CE + 0.5 * per-image foreground Dice.

The definition below is part of the experiment identity. Changing any bullet
means the run must be renamed, because the numbers stop being comparable:

* CE runs on logits **upsampled to the label resolution** (512) with
  ``ignore_index=255``, so the 23.83% padding never enters the numerator or the
  denominator. A nearest-neighbour label downsample would also be well-defined,
  but would discard more label detail. This protocol instead keeps the labels
  and ignore mask at their existing resolution.
* Dice uses ``softmax(float(logits))`` -- FP32 even when the forward pass runs
  under AMP -- on the defect channel, and is averaged **only** over images whose
  valid GT contains at least one defect pixel. A normal image has no Dice term;
  CE supervises it and still punishes false positives.
* A batch whose foreground subset is empty contributes ``prob.sum() * 0.0``
  instead of being skipped: the value is 0, the graph stays connected, and
  ``backward()`` on Dice alone yields an exact zero gradient rather than
  ``None``. CE still supervises normal images; adding a Python zero to CE would
  not remove CE's gradient. The connected zero keeps a uniform tensor interface.
* A batch with **no valid pixel at all** is rejected loudly. ``cross_entropy``
  would return NaN (0/0), which poisons the weights one step later.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

NUM_LABELS = 2
IGNORE_INDEX = 255
DEFECT_CLASS = 1


def upsample_logits(
    logits: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """Resize 4-D logits to ``size`` = (H, W).

    ``align_corners=False`` is used everywhere in this project. The 512 canvas
    is a resampling grid, not a set of corner samples; mixing conventions
    between training and evaluation shifts every prediction by a fraction of a
    pixel and is hard to see in a loss curve.
    """
    if logits.ndim != 4:
        raise ValueError(f"expected 4-D logits [B, C, H, W], got shape {tuple(logits.shape)}")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid target size {size}")
    if logits.shape[-2] == height and logits.shape[-1] == width:
        return logits
    return F.interpolate(logits, size=(height, width), mode=mode, align_corners=align_corners)


def _invalid_label_mask(labels: torch.Tensor) -> torch.Tensor:
    """True where a label is neither background, defect, nor ignore."""
    return (labels != 0) & (labels != 1) & (labels != IGNORE_INDEX)


def _validate_loss_parameters(ce_weight: float, dice_weight: float, eps: float) -> tuple[float, float, float]:
    try:
        ce_weight, dice_weight, eps = float(ce_weight), float(dice_weight), float(eps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("loss weights and eps must be finite numbers") from exc
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if (not math.isfinite(ce_weight) or not math.isfinite(dice_weight)
            or ce_weight < 0 or dice_weight < 0 or (ce_weight == 0 and dice_weight == 0)):
        raise ValueError("loss weights must be finite, nonnegative, and not both zero")
    return ce_weight, dice_weight, eps


@dataclass(frozen=True)
class LossTerms:
    """Scalar terms plus the counts needed to explain them."""

    total: torch.Tensor
    ce: torch.Tensor
    dice: torch.Tensor
    positive_images: int
    valid_images: int
    valid_pixels: int
    ignored_pixels: int
    upsampled: bool
    ce_weight: float
    dice_weight: float
    eps: float
    dtype: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": float(self.total.detach()),
            "ce": float(self.ce.detach()),
            "dice": float(self.dice.detach()),
            "positive_images": self.positive_images,
            "valid_images": self.valid_images,
            "valid_pixels": self.valid_pixels,
            "ignored_pixels": self.ignored_pixels,
            "upsampled": self.upsampled,
            "ce_weight": self.ce_weight,
            "dice_weight": self.dice_weight,
            "eps": self.eps,
            "dtype": self.dtype,
        }


def segformer_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ce_weight: float = 1.0,
    dice_weight: float = 0.5,
    eps: float = 1e-6,
    ignore_index: int = IGNORE_INDEX,
    num_labels: int = NUM_LABELS,
    check_labels: bool = True,
) -> LossTerms:
    """``ce_weight * CE + dice_weight * mean(1 - per-image soft Dice)``.

    ``logits``  [B, 2, h, w] float (any precision; Dice is forced to FP32)
    ``labels``  [B, H, W] int64 with values in {0, 1, 255}
    """
    if logits.ndim != 4:
        raise ValueError(f"logits must be 4-D [B, C, H, W], got {tuple(logits.shape)}")
    if not logits.is_floating_point():
        raise ValueError(f"logits must be floating point, got {logits.dtype}")
    if num_labels != NUM_LABELS:
        raise ValueError("this protocol fixes num_labels at 2")
    ce_weight, dice_weight, eps = _validate_loss_parameters(ce_weight, dice_weight, eps)
    if labels.ndim != 3:
        raise ValueError(f"labels must be 3-D [B, H, W], got {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(f"batch mismatch: logits {logits.shape[0]} vs labels {labels.shape[0]}")
    if logits.shape[1] != num_labels:
        raise ValueError(f"expected {num_labels} logit channels, got {logits.shape[1]}")
    if labels.dtype != torch.int64:
        raise ValueError(f"labels must be int64, got {labels.dtype}")
    if ignore_index != IGNORE_INDEX:
        raise ValueError("this protocol fixes ignore_index at 255")
    if check_labels and bool(_invalid_label_mask(labels).any()):
        bad = sorted({int(v) for v in torch.unique(labels)} - {0, 1, IGNORE_INDEX})
        raise ValueError(f"labels contain values outside {{0, 1, 255}}: {bad}")

    target = (int(labels.shape[-2]), int(labels.shape[-1]))
    logits_512 = upsample_logits(logits, target)
    upsampled = logits_512 is not logits

    valid = labels != ignore_index
    valid_pixels = int(valid.sum())
    ignored_pixels = int(labels.numel() - valid_pixels)
    if valid_pixels == 0:
        raise ValueError(
            "batch has no valid pixel (every label is 255); CE would be 0/0 = NaN. "
            "Refusing instead of returning a silent NaN loss."
        )

    # A half-precision mean can overflow at 512x512 even with all-zero logits
    # when the caller has left autocast. Cast explicitly, retaining autograd.
    ce = F.cross_entropy(logits_512.float(), labels, ignore_index=ignore_index, reduction="mean")

    prob = torch.softmax(logits_512.float(), dim=1)[:, DEFECT_CLASS]
    valid_f = valid.to(dtype=prob.dtype)
    gt = ((labels == 1) & valid).to(dtype=prob.dtype)

    prob_sum = (prob * valid_f).sum(dim=(1, 2))
    gt_sum = gt.sum(dim=(1, 2))
    # |P n G| is a per-pixel product summed over pixels, NOT prob_sum * gt_sum.
    # The scalar product would be "total probability times total area", which is
    # a different (and much larger) quantity and would make the loss look fine
    # while optimising something else.
    intersection = (prob * gt).sum(dim=(1, 2))
    has_foreground = gt_sum > 0
    positive_images = int(has_foreground.sum())
    valid_images = int(valid.flatten(1).any(dim=1).sum())

    if positive_images > 0:
        p = prob_sum[has_foreground]
        g = gt_sum[has_foreground]
        inter = intersection[has_foreground]
        per_image = (2.0 * inter + eps) / (p + g + eps)
        dice = (1.0 - per_image).mean()
    else:
        # Keep the graph: value 0, exact zero gradient, never None.
        dice = prob.sum() * 0.0

    total = ce_weight * ce + dice_weight * dice

    return LossTerms(
        total=total,
        ce=ce,
        dice=dice,
        positive_images=positive_images,
        valid_images=valid_images,
        valid_pixels=valid_pixels,
        ignored_pixels=ignored_pixels,
        upsampled=upsampled,
        ce_weight=float(ce_weight),
        dice_weight=float(dice_weight),
        eps=eps,
        dtype=str(logits.dtype),
    )


def loss_definition(*, ce_weight: float = 1.0, dice_weight: float = 0.5,
                    eps: float = 1e-6) -> dict[str, Any]:
    """Definition for the supplied runtime parameters; defaults remain fixed."""
    ce_weight, dice_weight, eps = _validate_loss_parameters(ce_weight, dice_weight, eps)
    return {
        "name": "ce_plus_foreground_soft_dice",
        "total": "ce_weight * CE + dice_weight * mean_over_positive_images(1 - Dice_i)",
        "ce": {
            "op": "torch.nn.functional.cross_entropy",
            "reduction": "mean over pixels with label != ignore_index",
            "ignore_index": IGNORE_INDEX,
            "precision": "logits.float() before cross_entropy, including outside autocast",
            "upsampled_to": "label resolution (512) with bilinear, align_corners=False",
        },
        "dice": {
            "class": DEFECT_CLASS,
            "probability": "softmax(logits.float(), dim=1)[:, 1]",
            "per_image": "(2*sum(p*g) + eps) / (sum(p) + sum(g) + eps)",
            "valid_region": "label != ignore_index",
            "averaged_over": "images whose valid GT contains at least one defect pixel",
            "empty_foreground_batch": "prob.sum() * 0.0 (graph-connected exact zero)",
            "eps": eps,
        },
        "defaults": {"ce_weight": 1.0, "dice_weight": 0.5, "eps": 1e-6},
        "runtime": {"ce_weight": ce_weight, "dice_weight": dice_weight, "eps": eps},
        "rejected": "batch with zero valid pixels",
    }
