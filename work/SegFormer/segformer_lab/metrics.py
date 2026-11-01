"""Original-resolution scoring: one restore function, integer confusion counts.

The only supported way to turn model logits into a full-size prediction is
:func:`restore_defect_probability`:

    logits [2, 128, 128]
      -> bilinear to 512x512, align_corners=False
      -> crop this sample's valid box (drop the 61 padding rows)
      -> bilinear to the original H x W, align_corners=False
      -> softmax over the two channels, keep the defect channel
      -> threshold at >= 0.5

Ground truth is the **original** mask handed out by the Dataset, never the 512
label resized back up. Nearest-neighbour resampling changes boundaries and can
lose whole tiny regions (stage 04 counted 19 lost regions comprising 39 original
pixels in 12 images). Evaluating against a resized GT can hide false negatives.

Scoring accumulates TP/FP/FN/TN as 64-bit integers and divides once at the end.
Averaging per-batch IoU and calling it "global IoU" is a different number,
which can be higher or lower; it is not used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Iterable

import numpy as np
import torch

from .losses import DEFECT_CLASS, NUM_LABELS, upsample_logits

DEFAULT_THRESHOLD = 0.5


def _geometry_integer(geometry: dict[str, Any], name: str) -> int:
    if name not in geometry:
        raise ValueError(f"geometry is missing required dimension {name}")
    value = geometry[name]
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or int(value) != value):
        raise ValueError(f"geometry {name} must be a finite integer, got {value!r}")
    return int(value)


def _checked_threshold(value: float) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold must be finite and within [0, 1]") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")
    return threshold


# ------------------------------------------------------------------ restore ----
def restore_defect_probability(
    logits: torch.Tensor,
    geometry: dict[str, Any],
) -> torch.Tensor:
    """Return the defect probability at the sample's **original** resolution.

    ``logits`` is [2, h, w] for a single sample and may live on any device;
    the result is a CPU float32 tensor of shape ``[orig_h, orig_w]``.
    """
    if logits.ndim != 3:
        raise ValueError(f"expected single-sample logits [C, H, W], got {tuple(logits.shape)}")
    if logits.shape[0] != NUM_LABELS:
        raise ValueError(f"expected {NUM_LABELS} channels, got {logits.shape[0]}")
    if not logits.is_floating_point() or min(logits.shape[-2:]) <= 0:
        raise ValueError("logits must be floating point with nonempty spatial dimensions")

    input_size = _geometry_integer(geometry, "input_size")
    pad_top = _geometry_integer(geometry, "pad_top")
    pad_left = _geometry_integer(geometry, "pad_left")
    new_h = _geometry_integer(geometry, "new_h")
    new_w = _geometry_integer(geometry, "new_w")
    orig_h = _geometry_integer(geometry, "orig_h")
    orig_w = _geometry_integer(geometry, "orig_w")
    if min(input_size, new_h, new_w, orig_h, orig_w) <= 0 or min(pad_top, pad_left) < 0:
        raise ValueError(f"invalid geometry: {geometry}")
    if pad_top + new_h > input_size or pad_left + new_w > input_size:
        raise ValueError(f"geometry does not fit the {input_size} canvas: {geometry}")
    # Older fixtures omit bottom/right; when supplied they must describe the
    # same complete canvas, rather than silently contradict the crop.
    for side, before, length in (("pad_bottom", pad_top, new_h), ("pad_right", pad_left, new_w)):
        if side in geometry:
            after = _geometry_integer(geometry, side)
            if after < 0 or before + length + after != input_size:
                raise ValueError(f"inconsistent {side} for the {input_size} canvas: {geometry}")

    work = logits.detach().to("cpu", torch.float32).unsqueeze(0)
    if not bool(torch.isfinite(work).all()):
        raise ValueError("logits must remain finite in float32")

    # 1) model grid -> 512 canvas (the same 512 the Dataset padded)
    canvas = upsample_logits(work, (input_size, input_size))[0]
    # 2) drop this sample's padding; padding must never be rescaled into the image
    valid = canvas[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    # 3) stretch the valid box back to the original pixel grid
    full = upsample_logits(valid.unsqueeze(0), (orig_h, orig_w))[0]
    # 4) softmax last: resizing logits and resizing probabilities are not the
    #    same function, and the guide fixes the order.
    probability = torch.softmax(full, dim=0)[DEFECT_CLASS]
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("restored probability contains NaN or infinity")
    return probability


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


# -------------------------------------------------------------------- sample ----
@dataclass
class ImageScore:
    image_id: str
    label: str
    split: str
    tp: int
    fp: int
    fn: int
    tn: int
    gt_foreground_pixels: int
    predicted_foreground_pixels: int
    total_pixels: int
    iou: float | None
    iou_na: str
    dice: float | None
    dice_na: str
    precision: float | None
    recall: float | None
    precision_na: str = ""
    recall_na: str = ""
    ignored_pixels: int = 0

    @property
    def predicted_foreground_ratio(self) -> float:
        return self.predicted_foreground_pixels / float(self.total_pixels)

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "label": self.label,
            "split": self.split,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "gt_foreground_pixels": self.gt_foreground_pixels,
            "predicted_foreground_pixels": self.predicted_foreground_pixels,
            "total_pixels": self.total_pixels,
            "ignored_pixels": self.ignored_pixels,
            "predicted_foreground_ratio": round(self.predicted_foreground_ratio, 10),
            "iou": None if self.iou is None else round(self.iou, 10),
            "iou_na": self.iou_na,
            "dice": None if self.dice is None else round(self.dice, 10),
            "dice_na": self.dice_na,
            "precision": None if self.precision is None else round(self.precision, 10),
            "recall": None if self.recall is None else round(self.recall, 10),
            "precision_na": self.precision_na,
            "recall_na": self.recall_na,
        }


@dataclass
class OriginalSpaceEvaluator:
    """Accumulate original-resolution confusion counts over any batch layout."""

    threshold: float = DEFAULT_THRESHOLD
    accept_flipped: bool = False
    tp: np.int64 = field(default=np.int64(0))
    fp: np.int64 = field(default=np.int64(0))
    fn: np.int64 = field(default=np.int64(0))
    tn: np.int64 = field(default=np.int64(0))
    scores: list[ImageScore] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.threshold = _checked_threshold(self.threshold)

    def add_sample(
        self,
        logits: torch.Tensor,
        original_mask: np.ndarray,
        geometry: dict[str, Any],
        *,
        image_id: str,
        label: str,
        split: str = "val",
        flipped: bool = False,
    ) -> ImageScore:
        threshold = _checked_threshold(self.threshold)
        if label not in {"normal", "anomaly"}:
            raise ValueError(f"label must be normal or anomaly, got {label!r}")
        if flipped and not self.accept_flipped:
            raise ValueError(
                f"refusing to score a flipped sample ({image_id}): the baseline evaluation "
                "protocol requires flip='none'. Explicit accept_flipped=True is a protocol variant; "
                "its original mask must share the logits' transformed coordinates."
            )
        mask = np.asarray(original_mask)
        if mask.ndim != 2:
            raise ValueError(f"original mask must be 2-D, got {mask.shape} for {image_id}")
        expected = (_geometry_integer(geometry, "orig_h"), _geometry_integer(geometry, "orig_w"))
        if mask.shape != expected:
            raise ValueError(f"original mask {mask.shape} does not match geometry {expected} for {image_id}")
        if mask.dtype.kind not in "biuf" or not np.isin(mask, (0, 1, 255)).all():
            raise ValueError(f"original mask must contain only discrete 0/1/255 values for {image_id}")
        valid = mask != 255
        valid_pixels = int(valid.sum())
        if valid_pixels == 0:
            raise ValueError(f"original mask has no valid pixels for {image_id}")

        prob = restore_defect_probability(logits, geometry)
        if tuple(prob.shape) != expected:
            raise ValueError(f"restored probability {tuple(prob.shape)} != {expected}")
        if not bool(torch.isfinite(prob).all()):
            raise ValueError(f"restored probability is not finite for {image_id}")
        pred = prob.numpy() >= threshold
        gt = mask == 1

        tp = int((pred & gt & valid).sum())
        fp = int((pred & ~gt & valid).sum())
        fn = int((~pred & gt & valid).sum())
        tn = int((~pred & ~gt & valid).sum())
        total = tp + fp + fn + tn
        if total != valid_pixels:
            raise ValueError(f"confusion counts {total} != valid pixels {valid_pixels} for {image_id}")

        self.tp += np.int64(tp)
        self.fp += np.int64(fp)
        self.fn += np.int64(fn)
        self.tn += np.int64(tn)

        gt_fg = tp + fn
        pred_fg = tp + fp
        iou_den = tp + fp + fn
        dice_den = 2 * tp + fp + fn
        iou = _safe_div(tp, iou_den)
        dice = _safe_div(2 * tp, dice_den)
        reason = "GT and prediction are both empty" if iou_den == 0 else ""
        score = ImageScore(
            image_id=image_id,
            label=label,
            split=split,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            gt_foreground_pixels=gt_fg,
            predicted_foreground_pixels=pred_fg,
            total_pixels=total,
            iou=iou,
            iou_na=reason,
            dice=dice,
            dice_na=reason,
            precision=_safe_div(tp, pred_fg),
            recall=_safe_div(tp, gt_fg),
            precision_na="prediction contains no foreground pixel" if pred_fg == 0 else "",
            recall_na="ground truth contains no foreground pixel" if gt_fg == 0 else "",
            ignored_pixels=int(mask.size) - valid_pixels,
        )
        self.scores.append(score)
        return score

    # -- aggregate ----------------------------------------------------------
    def result(self) -> dict[str, Any]:
        tp, fp, fn, tn = int(self.tp), int(self.fp), int(self.fn), int(self.tn)
        iou_den = tp + fp + fn
        dice_den = 2 * tp + fp + fn
        precision_den = tp + fp
        recall_den = tp + fn

        na: dict[str, str] = {}
        global_iou = _safe_div(tp, iou_den)
        if global_iou is None:
            na["foreground_iou"] = "no foreground pixel in ground truth or prediction"
        global_dice = _safe_div(2 * tp, dice_den)
        if global_dice is None:
            na["foreground_dice"] = "no foreground pixel in ground truth or prediction"
        global_precision = _safe_div(tp, precision_den)
        if global_precision is None:
            na["precision"] = "model predicted no foreground pixel at all"
        global_recall = _safe_div(tp, recall_den)
        if global_recall is None:
            na["recall"] = "ground truth has no foreground pixel in this set"

        normal = [s for s in self.scores if s.label == "normal"]
        anomaly = [s for s in self.scores if s.label == "anomaly"]

        normal_fp = int(sum(s.fp for s in normal))
        normal_total = int(sum(s.total_pixels for s in normal))
        normal_ratios = [s.predicted_foreground_ratio for s in normal]

        def _percentile(values: Iterable[float], q: float) -> float | None:
            arr = np.asarray(list(values), dtype=np.float64)
            if arr.size == 0:
                return None
            return round(float(np.percentile(arr, q)), 10)

        anomaly_macro = {}
        for name in ("iou", "dice", "precision", "recall"):
            values = [getattr(s, name) for s in anomaly]
            defined = [v for v in values if v is not None]
            anomaly_macro[name] = {
                "mean": round(float(np.mean(defined)), 10) if defined else None,
                "defined_images": len(defined),
                "na_images": len(values) - len(defined),
                "na_reason": ("no anomaly images" if not values else f"no defined per-image {name}") if not defined else "",
            }

        # normal images whose prediction is non-empty are the false-positive
        # offenders a global average tends to hide
        normal_offenders = sorted(
            (s for s in normal if s.fp > 0),
            key=lambda s: (-s.predicted_foreground_pixels, s.image_id),
        )

        return {
            "threshold": self.threshold,
            "accept_flipped": self.accept_flipped,
            "images": {"total": len(self.scores), "normal": len(normal), "anomaly": len(anomaly)},
            "confusion": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "pixels": tp + fp + fn + tn,
                "accumulator_dtype": "int64",
            },
            "global": {
                "foreground_iou": None if global_iou is None else round(global_iou, 10),
                "foreground_dice": None if global_dice is None else round(global_dice, 10),
                "precision": None if global_precision is None else round(global_precision, 10),
                "recall": None if global_recall is None else round(global_recall, 10),
                "na_reasons": na,
                "aggregation": "sum TP/FP/FN/TN over images, then divide once",
            },
            "normal_images": {
                "count": len(normal),
                "false_positive_pixels": normal_fp,
                "total_pixels": normal_total,
                "fp_pixel_ratio": _safe_div(normal_fp, normal_total),
                "na_reasons": ({"fp_pixel_ratio": "no normal-image valid pixels",
                                "predicted_foreground_ratio": "no normal images"} if not normal else {}),
                "predicted_foreground_ratio": {
                    "median": _percentile(normal_ratios, 50),
                    "p95": _percentile(normal_ratios, 95),
                    "max": round(float(max(normal_ratios)), 10) if normal_ratios else None,
                    "percentile_method": "numpy linear interpolation over per-image ratios",
                },
                "images_with_any_false_positive": len(normal_offenders),
                "worst_offenders": [
                    {
                        "image_id": s.image_id,
                        "predicted_foreground_pixels": s.predicted_foreground_pixels,
                        "predicted_foreground_ratio": round(s.predicted_foreground_ratio, 10),
                    }
                    for s in normal_offenders[:5]
                ],
            },
            "anomaly_images": {
                "count": len(anomaly),
                "macro": anomaly_macro,
                "na_detail": {
                    "iou_or_dice": [s.image_id for s in anomaly if s.iou is None],
                },
            },
            "per_image": [s.as_dict() for s in self.scores],
        }


def score_definition() -> dict[str, Any]:
    """Machine-readable scoring definition stored with every evaluation."""
    return {
        "space": "original",
        "mask_values": "0 background, 1 defect, 255 ignored; other original IDs must first be binarized",
        "valid_pixels": "255 excluded from confusion counts and every foreground-area denominator; all-ignore sample rejected",
        "gt_source": "Dataset original_mask (never the 512 label resized back up)",
        "restore": [
            "bilinear logits -> 512x512, align_corners=False",
            "crop pad_top..pad_top+new_h, pad_left..pad_left+new_w",
            "bilinear valid box -> orig_h x orig_w, align_corners=False",
            "softmax over 2 channels, take channel 1",
            "binarize with prob >= threshold",
        ],
        "threshold": DEFAULT_THRESHOLD,
        "confusion": "int64 TP/FP/FN/TN summed over images, divided once",
        "foreground_iou": "TP / (TP + FP + FN)",
        "foreground_dice": "2TP / (2TP + FP + FN)",
        "precision": "TP / (TP + FP)",
        "recall": "TP / (TP + FN)",
        "no_smoothing": True,
        "zero_denominator": "null plus a reason string",
        "normal_image_fp": "sum(FP) / sum(valid non-255 pixels of normal images)",
        "per_image": "IoU/Dice are null when GT and prediction are both empty",
        "macro": "anomaly images only, averaged over images where the value is defined",
        "rejected": "flipped evaluation samples by default (baseline flip='none')",
        "flip_variant": "accept_flipped=True explicitly opts into a different protocol; original mask must match transformed coordinates",
    }
