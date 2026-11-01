"""Stage 7 acceptance: loss definition and original-space metrics.

Two kinds of evidence, deliberately kept apart:

1. **Fixtures with hand-computed answers.** Every number can be checked on
   paper; if the code and the paper disagree, the code is wrong.
2. **A pipeline probe on six real validation images.** The decode head is still
   randomly initialised, so those numbers only prove that tensors flow end to
   end on real data. They live under ``random_head_pipeline_probe`` and must
   never be quoted as a baseline.

Nothing here trains, steps an optimiser, or touches the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import data, geometry  # noqa: E402
from .config import load_config, resolved_yaml, write_resolved_config  # noqa: E402
from .audit_data import verify_manifests  # noqa: E402
from .download_model import verify_snapshot  # noqa: E402
from .evaluation import evaluation_precision  # noqa: E402
from .losses import loss_definition, segformer_loss, upsample_logits  # noqa: E402
from .metrics import (  # noqa: E402
    OriginalSpaceEvaluator,
    restore_defect_probability,
    score_definition,
)

SCHEMA = "segformer-lab/loss-metrics-check/2"

# Identity captured at stage 06 acceptance; a mismatch means the protocol moved
# and stage 07 must not silently continue.
EXPECTED_MANIFESTS = {
    "train": "f1522eb9b474d0acc1d736a391002f5044153208fa10f51eea3dc7df729f65b0",
    "val": "60a6c6cbf5ed7f5f9e57fb57bb096a5b57db677af5ee73f4fa140ddceaba6191",
    "test": "4bdc594148f208c7067a66a8cb3b06fde53ebc1f4ab6fda84f08f80383061284",
}
EXPECTED_REVIEWED_GROUPS = "220dae5831d2b72502ecd9832a60fd12d349f4b7bb873b5d78e25c657e533ad8"
EXPECTED_PREVIEW_IDS = "4f17789dfecc28023003092dddbe3c4683ac781cb469b483e4180c2ebb8f9718"
EXPECTED_REVISION = "80983a413c30d36a39c20203974ae7807835e2b4"
EXPECTED_WEIGHTS_SHA = "4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc"
EXPECTED_SOURCE_SHA = "8e57fa70638ae7aad076ea9ad9603f4e4a87da54e09d2cb698ea26328beb8650"
PROBABILITY_ATOL = 1e-5


def _array_sha(value: Any) -> str:
    array = value.detach().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _state_sha(model: Any, prefix: str = "") -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(prefix):
            digest.update(name.encode())
            digest.update(_array_sha(tensor).encode())
    return digest.hexdigest()


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    array = values.detach().cpu().numpy().reshape(-1)
    if not array.size:
        return {"count": 0, "min": None, "p95": None, "max": None}
    return {"count": int(array.size), "min": float(array.min()),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def compare_forward_tensors(
    logits_left: torch.Tensor, logits_right: torch.Tensor,
    probability_left: torch.Tensor, probability_right: torch.Tensor,
    *, threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare paired pixels, never the difference of aggregate confusion counts.

    FP32 probability tolerance is fixed *before* observing GPU results. 1e-5
    (~84 machine eps at unit scale) is a diagnostic acceptance budget, not a
    universal cuDNN error bound. Logit differences are reported without a gate:
    softmax is invariant to common offsets in both channels.
    """
    tensors = (logits_left, logits_right, probability_left, probability_right)
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("forward comparison requires float32 tensors")
    if logits_left.shape != logits_right.shape or probability_left.shape != probability_right.shape:
        raise ValueError("paired forward shapes differ")
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError("forward comparison requires finite tensors")
    if not probability_left.numel():
        raise ValueError("forward comparison requires nonempty probabilities")
    if any(bool(((p < 0) | (p > 1)).any()) for p in (probability_left, probability_right)):
        raise ValueError("probabilities must be in [0,1]")
    left, right = probability_left >= threshold, probability_right >= threshold
    changed = left ^ right
    n = int(changed.sum())
    left_margin = (probability_left[changed] - threshold).abs()
    right_margin = (probability_right[changed] - threshold).abs()
    close_margins = bool((left_margin <= PROBABILITY_ATOL).all() and
                         (right_margin <= PROBABILITY_ATOL).all())
    logit_diff = (logits_left - logits_right).abs()
    probability_diff = (probability_left - probability_right).abs()
    max_prob_diff = float(probability_diff.max())
    return {
        "changed_pixels": n, "total_pixels": probability_left.numel(),
        "differing_pixels": n, "differing_ratio": n / probability_left.numel(),
        "max_abs_logit_diff": float(logit_diff.max()),
        "max_abs_prob_diff": max_prob_diff,
        "absolute_probability_diff": _distribution(probability_diff),
        "threshold_crossings": {"negative_to_positive": int((~left & right).sum()),
                                "positive_to_negative": int((left & ~right).sum())},
        "changed_pixel_margin": {"left": _distribution(left_margin), "right": _distribution(right_margin)},
        "near_threshold_pixels": {"left": int(((probability_left-threshold).abs() <= PROBABILITY_ATOL).sum()),
                                  "right": int(((probability_right-threshold).abs() <= PROBABILITY_ATOL).sum())},
        "changed_pixels_within_threshold_tolerance": close_margins,
        "float_close": max_prob_diff <= PROBABILITY_ATOL and close_margins,
        "tolerance": {"dtype": "torch.float32", "probability_atol": PROBABILITY_ATOL,
                      "probability_rtol": 0.0, "machine_epsilon": torch.finfo(torch.float32).eps,
                      "atol_in_machine_eps": PROBABILITY_ATOL / torch.finfo(torch.float32).eps,
                      "policy": "predeclared diagnostic budget; all probability differences and both changed-pixel margins must fit; no changed-ratio gate"},
    }


def same_logits_layout_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Exact global, per-image and tensor/hash identity, excluding timings."""
    keys = ("threshold", "images", "confusion", "global", "normal_images", "anomaly_images", "per_image", "sample_evidence")
    return all(left.get(key) == right.get(key) for key in keys)


def _compare_caches(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if left.keys() != right.keys():
        raise ValueError("forward image identities differ")
    per_image = []
    for image_id, a in left.items():
        b = right[image_id]
        if a["input_sha256"] != b["input_sha256"] or a["mask_sha256"] != b["mask_sha256"] or a["geometry"] != b["geometry"]:
            raise ValueError("forward input/GT/geometry changed between passes")
        per_image.append({"image_id": image_id, **compare_forward_tensors(
            a["logits"], b["logits"], a["probability"], b["probability"])})
    changed = sum(s["changed_pixels"] for s in per_image)
    total = sum(s["total_pixels"] for s in per_image)
    return {"per_image": per_image, "changed_pixels": changed, "differing_pixels": changed,
            "total_pixels": total, "differing_ratio": changed / total,
            "max_abs_logit_diff": max(s["max_abs_logit_diff"] for s in per_image),
            "max_abs_prob_diff": max(s["max_abs_prob_diff"] for s in per_image),
            "float_close": all(s["float_close"] for s in per_image),
            "tolerance": per_image[0]["tolerance"],
            "interpretation": "Batch-dependent floating-point execution is a possible explanation; no specific cuDNN kernel was identified. Paired probability differences and threshold margins are measured above."}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(actual: float | None, expected: float, tol: float = 1e-9) -> bool:
    if actual is None:
        return False
    return abs(float(actual) - float(expected)) <= tol


# ------------------------------------------------------------------ fixtures ----
def binary_logits(pred: np.ndarray, strength: float = 12.0) -> torch.Tensor:
    """Logits whose defect channel is unambiguous where ``pred`` is True."""
    pred = np.asarray(pred, dtype=bool)
    logits = np.empty((2,) + pred.shape, dtype=np.float32)
    logits[0] = np.where(pred, -strength, strength)
    logits[1] = np.where(pred, strength, -strength)
    return torch.from_numpy(logits)


def square_geometry(size: int) -> dict[str, Any]:
    """Identity geometry: no scaling, no padding, so the restore is exact."""
    return {
        "orig_w": int(size),
        "orig_h": int(size),
        "input_size": int(size),
        "pad_left": 0,
        "pad_top": 0,
        "new_w": int(size),
        "new_h": int(size),
        "pad_right": 0,
        "pad_bottom": 0,
    }


def score_fixture(pred: np.ndarray, gt: np.ndarray, label: str = "anomaly") -> dict[str, Any]:
    evaluator = OriginalSpaceEvaluator()
    evaluator.add_sample(
        binary_logits(pred),
        np.asarray(gt, dtype=np.uint8),
        square_geometry(int(gt.shape[0])),
        image_id="fixture",
        label=label,
    )
    return evaluator.result()


def metric_fixtures() -> dict[str, Any]:
    out: dict[str, Any] = {}

    # 1) the guide's 2x2 example, hand-computed:
    #    TP=1 FP=0 FN=1 TN=2 -> IoU 0.5, Dice 2/3, precision 1, recall 0.5
    gt_2x2 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    pred_2x2 = np.array([[0, 1], [0, 0]], dtype=bool)
    r = score_fixture(pred_2x2, gt_2x2)
    out["hand_2x2"] = {
        "confusion": {k: r["confusion"][k] for k in ("tp", "fp", "fn", "tn")},
        "iou": r["global"]["foreground_iou"],
        "dice": r["global"]["foreground_dice"],
        "precision": r["global"]["precision"],
        "recall": r["global"]["recall"],
        "expected": {"tp": 1, "fp": 0, "fn": 1, "tn": 2, "iou": 0.5, "dice": 2.0 / 3.0,
                     "precision": 1.0, "recall": 0.5},
    }

    # 2) perfect prediction on a 4x4 with a 2x3 block
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[1:3, 1:4] = 1
    r = score_fixture(gt.astype(bool), gt)
    out["perfect"] = {
        "confusion": {k: r["confusion"][k] for k in ("tp", "fp", "fn", "tn")},
        "iou": r["global"]["foreground_iou"],
        "dice": r["global"]["foreground_dice"],
        "expected": {"tp": 6, "fp": 0, "fn": 0, "tn": 10, "iou": 1.0, "dice": 1.0},
    }

    # 3) all-background prediction on an image that has defects
    r = score_fixture(np.zeros((4, 4), dtype=bool), gt)
    out["all_background_prediction"] = {
        "iou": r["global"]["foreground_iou"],
        "dice": r["global"]["foreground_dice"],
        "precision": r["global"]["precision"],
        "precision_na": r["global"]["na_reasons"].get("precision", ""),
        "recall": r["global"]["recall"],
        "expected": {"iou": 0.0, "dice": 0.0, "precision": None, "recall": 0.0},
    }

    # 4) all-foreground prediction: recall 1, precision punished
    r = score_fixture(np.ones((4, 4), dtype=bool), gt)
    out["all_foreground_prediction"] = {
        "iou": r["global"]["foreground_iou"],
        "dice": r["global"]["foreground_dice"],
        "precision": r["global"]["precision"],
        "recall": r["global"]["recall"],
        "expected": {"iou": 6.0 / 16.0, "dice": 12.0 / 22.0, "precision": 6.0 / 16.0, "recall": 1.0},
    }

    # 5) false positives on a normal image
    r = score_fixture(np.array([[0, 0, 0, 0]] * 3 + [[1, 1, 1, 0]], dtype=bool),
                      np.zeros((4, 4), dtype=np.uint8), label="normal")
    out["normal_false_positive"] = {
        "iou": r["global"]["foreground_iou"],
        "precision": r["global"]["precision"],
        "recall": r["global"]["recall"],
        "recall_na": r["global"]["na_reasons"].get("recall", ""),
        "per_image_iou": r["per_image"][0]["iou"],
        "fp_pixel_ratio": r["normal_images"]["fp_pixel_ratio"],
        "expected": {"iou": 0.0, "precision": 0.0, "recall": None, "per_image_iou": 0.0,
                     "fp_pixel_ratio": 3.0 / 16.0},
    }

    # 6) normal image with an empty prediction -> per-image IoU must be null,
    #    not 1.0, otherwise 120 easy normals would dominate the macro average
    r = score_fixture(np.zeros((4, 4), dtype=bool), np.zeros((4, 4), dtype=np.uint8), label="normal")
    out["normal_empty_prediction"] = {
        "per_image_iou": r["per_image"][0]["iou"],
        "per_image_dice": r["per_image"][0]["dice"],
        "reason": r["per_image"][0]["iou_na"],
        "dice_na": r["per_image"][0]["dice_na"],
        "precision_na": r["per_image"][0]["precision_na"],
        "recall_na": r["per_image"][0]["recall_na"],
        "fp_pixel_ratio": r["normal_images"]["fp_pixel_ratio"],
    }

    # A predicted foreground at GT255 contributes neither FP nor TP. The
    # remaining three pixels have TP1, FN1, TN1; the denominator is exactly 3.
    r = score_fixture(np.array([[1, 1], [0, 0]], dtype=bool),
                      np.array([[1, 255], [1, 0]], dtype=np.uint8))
    out["original_mask_ignore255"] = {
        "confusion": {key: r["confusion"][key] for key in ("tp", "fp", "fn", "tn")},
        "ignored_pixels": r["per_image"][0]["ignored_pixels"],
        "pixels": r["confusion"]["pixels"], "iou": r["global"]["foreground_iou"],
        "expected": {"tp": 1, "fp": 0, "fn": 1, "tn": 1, "ignored_pixels": 1, "iou": 0.5},
    }
    rejected = None
    try:
        score_fixture(np.zeros((2, 2), dtype=bool), np.full((2, 2), 255, dtype=np.uint8))
    except ValueError as exc:
        rejected = str(exc)
    out["all_ignored_original_mask"] = {"rejected": rejected is not None, "message": rejected}

    # 7) "sum then divide" is not "average per-image IoU". A tiny perfect image
    #    and a large sloppy one must not get equal votes.
    a_gt = np.zeros((10, 10), dtype=np.uint8)
    a_gt[0, 0] = 1
    a_pred = np.zeros((10, 10), dtype=bool)
    a_pred[0, 0] = True          # tp=1, fp=0, fn=0 -> IoU 1.0
    b_gt = np.zeros((10, 10), dtype=np.uint8)
    b_gt[0, 0] = 1
    b_gt[0, 1] = 1               # 2 defect pixels
    b_pred = np.ones((10, 10), dtype=bool)
    b_pred[0, 1] = False         # tp=1, fp=98, fn=1 -> IoU 1/100 = 0.01
    evaluator = OriginalSpaceEvaluator()
    evaluator.add_sample(binary_logits(a_pred), a_gt, square_geometry(10), image_id="a", label="anomaly")
    evaluator.add_sample(binary_logits(b_pred), b_gt, square_geometry(10), image_id="b", label="anomaly")
    r = evaluator.result()
    per_image = [s["iou"] for s in r["per_image"]]
    out["mean_of_per_image_iou_is_not_global"] = {
        "per_image_iou": per_image,
        "mean_of_per_image": float(np.mean(per_image)),
        "global_iou": r["global"]["foreground_iou"],
        "confusion": {k: r["confusion"][k] for k in ("tp", "fp", "fn", "tn")},
        "expected_global": 2.0 / 101.0,
        "expected_mean": 0.5 * (1.0 + 0.01),
    }

    # 8) accumulation must not depend on how the set is chopped into batches
    rng = np.random.default_rng(7)
    samples = []
    for i in range(6):
        gt_i = (rng.random((8, 8)) > 0.75).astype(np.uint8)
        pred_i = gt_i.astype(bool) ^ (rng.random((8, 8)) > 0.7)
        samples.append((pred_i, gt_i, "anomaly" if i % 2 == 0 else "normal"))
    layouts = {}
    for name, order in {
        "sequential": list(range(6)),
        "reversed": list(reversed(range(6))),
        "interleaved": [0, 2, 4, 1, 3, 5],
    }.items():
        ev = OriginalSpaceEvaluator()
        for i in order:
            pred_i, gt_i, label_i = samples[i]
            ev.add_sample(binary_logits(pred_i), gt_i, square_geometry(8), image_id=f"s{i}", label=label_i)
        r = ev.result()
        layouts[name] = {
            "confusion": {k: r["confusion"][k] for k in ("tp", "fp", "fn", "tn")},
            "iou": r["global"]["foreground_iou"],
            "dice": r["global"]["foreground_dice"],
        }
    out["batch_layout_invariance"] = layouts

    return out


def loss_fixtures() -> dict[str, Any]:
    out: dict[str, Any] = {}

    # 1) CE by hand. labels 2x2 with one ignored pixel; every valid pixel gets
    #    p(correct) = 3/4 -> CE = -ln(0.75) = 0.2876820724517809
    ln3 = float(np.log(3.0))
    labels = torch.tensor([[0, 1], [1, 255]], dtype=torch.int64).unsqueeze(0)
    logits = torch.tensor(
        [[[ln3, 0.0], [0.0, 0.0]],
         [[0.0, ln3], [ln3, 0.0]]],
        dtype=torch.float32,
    ).unsqueeze(0)
    terms = segformer_loss(logits, labels, dice_weight=0.0)
    out["ce_hand_computed"] = {
        "ce": float(terms.ce),
        "expected": -float(np.log(0.75)),
        "valid_pixels": terms.valid_pixels,
        "ignored_pixels": terms.ignored_pixels,
        "upsampled": terms.upsampled,
    }

    # 2) Dice by hand on the same logits: sum(p*g)=1.5, sum(p)=1.75, sum(g)=2
    #    Dice = 3/3.75 = 0.8 -> loss 0.2
    terms = segformer_loss(logits, labels, ce_weight=0.0)
    out["dice_hand_computed"] = {
        "dice_loss": float(terms.dice),
        "expected": 0.2,
        "positive_images": terms.positive_images,
        "valid_images": terms.valid_images,
    }

    # 3) a perfect prediction drives the Dice term to ~eps/(1+eps)
    labels_perfect = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64).unsqueeze(0)
    perfect = binary_logits(np.array([[0, 1], [1, 0]], dtype=bool)).unsqueeze(0)
    terms = segformer_loss(perfect, labels_perfect, ce_weight=0.0)
    out["perfect_prediction_dice"] = {"dice_loss": float(terms.dice)}

    # 4) all-normal batch: Dice term must be exactly 0 and still differentiable
    labels_normal = torch.zeros((1, 4, 4), dtype=torch.int64)
    logits_normal = torch.zeros((1, 2, 4, 4), dtype=torch.float32, requires_grad=True)
    terms = segformer_loss(logits_normal, labels_normal)
    terms.total.backward()
    grad = logits_normal.grad
    out["all_normal_batch"] = {
        "dice": float(terms.dice.detach()),
        "dice_is_exactly_zero": float(terms.dice.detach()) == 0.0,
        "positive_images": terms.positive_images,
        "grad_present": grad is not None,
        "grad_finite": bool(torch.isfinite(grad).all()) if grad is not None else False,
        "grad_abs_sum": float(grad.abs().sum()) if grad is not None else None,
        "total": float(terms.total.detach()),
    }

    # 5) changing logits under ignore_index=255 must not move the loss at all
    labels_pad = torch.tensor([[0, 1], [1, 255]], dtype=torch.int64).unsqueeze(0)
    base = logits.clone()
    perturbed = logits.clone()
    perturbed[0, :, 1, 1] = torch.tensor([500.0, -500.0])
    t_base = segformer_loss(base, labels_pad)
    t_pert = segformer_loss(perturbed, labels_pad)
    out["padding_does_not_affect_loss"] = {
        "total_base": float(t_base.total),
        "total_perturbed": float(t_pert.total),
        "abs_diff": abs(float(t_base.total) - float(t_pert.total)),
        "ce_diff": abs(float(t_base.ce) - float(t_pert.ce)),
        "dice_diff": abs(float(t_base.dice) - float(t_pert.dice)),
        "resolution": "logits share the label resolution, so ignore is exact",
    }

    # 6) the real training shape: 128 logits vs 512 labels, change only rows
    #    that fall entirely inside the padding band
    size = 512
    labels_big = torch.full((1, size, size), 255, dtype=torch.int64)
    labels_big[0, 200:260, 100:400] = 0
    labels_big[0, 210:250, 150:350] = 1
    logits_small = torch.zeros((1, 2, 128, 128), dtype=torch.float32)
    logits_small[0, 1, 60:70, 40:80] = 8.0
    logits_small[0, 0, 60:70, 40:80] = -8.0
    changed = logits_small.clone()
    changed[0, 0, 0:10, :] = 200.0   # rows 0..9 of a 128 grid -> canvas rows 0..39
    changed[0, 1, 0:10, :] = -200.0
    t_a = segformer_loss(logits_small, labels_big)
    t_b = segformer_loss(changed, labels_big)
    out["padding_lowres_inner_rows"] = {
        "total_base": float(t_a.total),
        "total_perturbed": float(t_b.total),
        "abs_diff": abs(float(t_a.total) - float(t_b.total)),
        "upsampled": t_a.upsampled,
        "note": "rows 0..9 of a 128 grid map to canvas rows 0..39, fully inside the 61-row pad",
    }

    # 7) a fully ignored batch must be rejected, not silently NaN
    rejected = None
    try:
        segformer_loss(torch.zeros((1, 2, 4, 4)), torch.full((1, 4, 4), 255, dtype=torch.int64))
    except ValueError as exc:
        rejected = str(exc)
    out["all_ignored_batch"] = {"rejected": rejected is not None, "message": rejected}

    # 8) splitting the same four images into 2+2 must match one pass of 4
    torch.manual_seed(3)
    big_labels = torch.zeros((4, 8, 8), dtype=torch.int64)
    for i in range(4):
        big_labels[i, 2:5, 2:6] = 1
    big_logits = torch.randn(4, 2, 8, 8)
    whole = segformer_loss(big_logits, big_labels)
    first = segformer_loss(big_logits[:2], big_labels[:2])
    second = segformer_loss(big_logits[2:], big_labels[2:])
    split_ce = 0.5 * (float(first.ce) + float(second.ce))
    split_dice = 0.5 * (float(first.dice) + float(second.dice))
    out["batch_split_equivalence"] = {
        "whole_ce": float(whole.ce),
        "split_ce": split_ce,
        "whole_dice": float(whole.dice),
        "split_dice": split_dice,
        "ce_diff": abs(float(whole.ce) - split_ce),
        "dice_diff": abs(float(whole.dice) - split_dice),
        "caveat": "equal only because both halves have the same valid-pixel count and both contain foreground",
    }

    return out


def restore_fixtures(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (report section, payload for the figure)."""
    out: dict[str, Any] = {}
    input_size = int(config["input_size"])
    orig_w, orig_h = 1404, 1070
    box = geometry.compute_letterbox(orig_w, orig_h, input_size)
    geo = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "input_size": input_size,
        "pad_left": box.pad_left,
        "pad_top": box.pad_top,
        "new_w": box.new_w,
        "new_h": box.new_h,
        "pad_right": box.pad_right,
        "pad_bottom": box.pad_bottom,
    }

    # --- asymmetric restore: a known block must land where the geometry says
    cy, cx, half = 200, 256, 48
    # background must be strictly negative for the defect channel; equal logits
    # give softmax 0.5, which the >= 0.5 threshold would call foreground
    small = torch.empty((2, 128, 128))
    small[0] = 10.0
    small[1] = -10.0
    r0, r1 = (cy - half) // 4, (cy + half) // 4
    c0, c1 = (cx - half) // 4, (cx + half) // 4
    small[1, r0:r1, c0:c1] = 10.0
    small[0, r0:r1, c0:c1] = -10.0

    prob = restore_defect_probability(small, geo)
    mask = (prob >= 0.5).numpy()
    ys, xs = np.nonzero(mask)
    centroid_correct = (float(ys.mean()), float(xs.mean()))

    # the wrong implementation the guide warns about: resize the whole canvas
    wrong = upsample_logits(small.unsqueeze(0), (input_size, input_size))[0]
    wrong_full = upsample_logits(wrong.unsqueeze(0), (orig_h, orig_w))[0]
    wrong_mask = (torch.softmax(wrong_full, dim=0)[1] >= 0.5).numpy()
    wys, wxs = np.nonzero(wrong_mask)
    centroid_wrong = (float(wys.mean()), float(wxs.mean()))

    expected_y = (cy - box.pad_top) / box.new_h * orig_h
    expected_x = (cx - box.pad_left) / box.new_w * orig_w
    out["asymmetric_restore"] = {
        "geometry": geo,
        "centroid_correct": [round(centroid_correct[0], 3), round(centroid_correct[1], 3)],
        "centroid_expected": [round(expected_y, 3), round(expected_x, 3)],
        "centroid_wrong_no_crop": [round(centroid_wrong[0], 3), round(centroid_wrong[1], 3)],
        "correct_error_px": round(abs(centroid_correct[0] - expected_y), 3),
        "wrong_error_px": round(abs(centroid_wrong[0] - expected_y), 3),
        "prob_shape": list(prob.shape),
    }

    # --- a single original pixel that nearest-neighbour sampling throws away
    dropped = None
    for y in range(520, 560):
        for x in range(700, 740):
            one = np.zeros((orig_h, orig_w), dtype=np.uint8)
            one[y, x] = 1
            label512 = geometry.letterbox_label(one, box, int(config["ignore_index"]))
            if int((label512 == 1).sum()) == 0:
                dropped = (y, x, one, label512)
                break
        if dropped:
            break
    if dropped is None:
        raise SystemExit("could not find a single pixel that disappears at 512; fixture invalid")

    y, x, one, label512 = dropped
    background = torch.full((2, 128, 128), -12.0)
    background[0] = 12.0

    evaluator = OriginalSpaceEvaluator()
    evaluator.add_sample(background, one, geo, image_id="single_pixel", label="anomaly")
    original = evaluator.result()

    # the cheat: resize the 512 label back up and score against that
    from PIL import Image as PILImage

    valid = label512[box.pad_top:box.pad_top + box.new_h, box.pad_left:box.pad_left + box.new_w]
    upscaled_gt = np.asarray(
        PILImage.fromarray(valid, mode="L").resize((orig_w, orig_h), PILImage.NEAREST),
        dtype=np.uint8,
    )
    evaluator2 = OriginalSpaceEvaluator()
    evaluator2.add_sample(background, upscaled_gt, geo, image_id="single_pixel_upscaled_gt", label="anomaly")
    shrunk = evaluator2.result()

    out["single_pixel_dropped_at_512"] = {
        "pixel": {"y": int(y), "x": int(x)},
        "label512_positive_pixels": int((label512 == 1).sum()),
        "original_gt": {
            "fn": original["confusion"]["fn"],
            "tp": original["confusion"]["tp"],
            "iou": original["global"]["foreground_iou"],
            "dice": original["global"]["foreground_dice"],
        },
        "upscaled_shrunk_gt": {
            "fn": shrunk["confusion"]["fn"],
            "tp": shrunk["confusion"]["tp"],
            "iou": shrunk["global"]["foreground_iou"],
            "iou_is_na": shrunk["per_image"][0]["iou_na"],
        },
        "conclusion": "original GT records the miss as FN; the resized label silently reports NA",
    }

    # --- figure: the two restore paths side by side
    return out, {"geo": geo, "small": small, "prob": prob, "centroid_correct": centroid_correct,
                 "centroid_wrong": centroid_wrong, "expected": (expected_y, expected_x),
                 "dropped": {"y": y, "x": x, "mask": one, "label512": label512},
                 "box": box}


def render_restore_figure(payload: dict[str, Any], out_path: Path) -> None:
    box = payload["box"]
    prob = payload["prob"].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)
    axes[0].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axes[0].axhline(payload["expected"][0], color="cyan", lw=1.2, ls="--", label="expected y")
    axes[0].axhline(payload["centroid_correct"][0], color="lime", lw=1.0, ls="-", label="restored centroid")
    axes[0].axhline(payload["centroid_wrong"][0], color="red", lw=1.0, ls=":", label="whole-canvas resize")
    axes[0].set_title("restore: crop padding, then upscale\n(cyan = expected, green = code, red = wrong path)",
                      fontsize=9)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("original y")

    valid_top, valid_left = box.pad_top, box.pad_left
    axes[1].imshow(payload["small"][1].numpy(), cmap="viridis")
    axes[1].add_patch(Rectangle((valid_left / 4, valid_top / 4), box.new_w / 4, box.new_h / 4,
                                fill=False, edgecolor="cyan", lw=1.2, ls="--"))
    axes[1].set_title("128x128 logits; cyan = valid box in model grid\n(rest is padding, 61/512 of the canvas)",
                      fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# -------------------------------------------------------------------- identity ----
def identity_checks(config: dict[str, Any]) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[1]
    audit = verify_manifests(config)
    acceptances = {stage: json.loads((project / "reports" / f"stage{stage}-acceptance.json").read_text(encoding="utf-8"))
                   for stage in ("05", "06")}
    manifests = {}
    ok = True
    for split in ("train", "val", "test"):
        path = Path(config["paths"][f"data.{split}_manifest"])
        digest = _sha256(path)
        expected = EXPECTED_MANIFESTS[split]
        manifests[split] = {"path": str(path), "sha256": digest, "matches_acceptance": digest == expected}
        ok = ok and digest == expected

    meta_path = Path(config["paths"]["data.split_meta"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_matches = all(
        meta["manifests"][split]["sha256"] == manifests[split]["sha256"] for split in ("train", "val", "test")
    )

    groups_path = Path(config["paths"]["data.split_meta"]).parent / "reviewed_groups.json"
    groups_sha = _sha256(groups_path) if groups_path.is_file() else ""
    preview_path = Path(config["paths"]["logging.fixed_preview_manifest"])
    preview_sha = _sha256(preview_path)

    model_dir = Path(config["paths"]["model.pretrained_dir"])
    source_path = model_dir / "source.json"
    portable_path = project / "reports" / "model_source.json"
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    weights = next(f for f in source["files"] if f["path"] == "pytorch_model.bin")
    snapshot_problems = verify_snapshot(model_dir, portable)
    source_sha = _sha256(source_path)
    portable_sha = _sha256(portable_path)
    acceptance_matches = True
    for stage, record in acceptances.items():
        anchor = record["data_identity"]
        snap = record["snapshot"] if stage == "05" else record["snapshot_identity"]
        acceptance_matches = acceptance_matches and (
            record.get("full_stage_verified", stage == "05") is True
            and anchor["manifest_sha256"] == EXPECTED_MANIFESTS
            and anchor["reviewed_groups_sha256"] == EXPECTED_REVIEWED_GROUPS
            and anchor["fixed_val_preview_sha256"] == EXPECTED_PREVIEW_IDS
            and snap["revision"] == EXPECTED_REVISION
            and snap["source_sha256"] == EXPECTED_SOURCE_SHA
            and snap["files"] == portable["files"])
    snapshot_verified = not snapshot_problems and source_sha == portable_sha == EXPECTED_SOURCE_SHA

    return {
        "manifests": manifests,
        "split_meta": {"path": str(meta_path), "schema": meta.get("schema"), "matches_manifests": meta_matches},
        "reviewed_groups": {"sha256": groups_sha, "matches": groups_sha == EXPECTED_REVIEWED_GROUPS},
        "preview_ids": {"sha256": preview_sha, "matches": preview_sha == EXPECTED_PREVIEW_IDS},
        "snapshot": {
            "model_id": source.get("model_id"),
            "revision": source.get("revision"),
            "weights_sha256": weights.get("sha256"),
            "revision_matches": source.get("revision") == EXPECTED_REVISION,
            "weights_match": weights.get("sha256") == EXPECTED_WEIGHTS_SHA and snapshot_verified,
            "source_sha256": source_sha,
            "portable_source_sha256": portable_sha,
            "verified_actual_bytes_offline": snapshot_verified,
            "problems": snapshot_problems,
        },
        "audit_data_verifier": audit,
        "acceptance_anchors_match": acceptance_matches,
        "acceptance_records": {stage: {"path": str(project / "reports" / f"stage{stage}-acceptance.json"),
                                       "sha256": _sha256(project / "reports" / f"stage{stage}-acceptance.json")}
                               for stage in acceptances},
        "ok": bool(ok and audit["ok"] and acceptance_matches and snapshot_verified
                   and meta_matches and groups_sha == EXPECTED_REVIEWED_GROUPS
                   and preview_sha == EXPECTED_PREVIEW_IDS
                   and source.get("revision") == EXPECTED_REVISION
                   and weights.get("sha256") == EXPECTED_WEIGHTS_SHA),
    }


# --------------------------------------------------------------- pipeline probe ----
def _run_probe(
    dataset: Any,
    indices: list[int],
    label_of: dict[str, str],
    model: Any,
    device: str,
    *,
    batch_size: int,
    cache: dict[str, Any] | None,
    reuse: bool,
) -> tuple[dict[str, Any], list[float]]:
    """Score the given indices at one batch size, optionally from cached logits.

    ``reuse=True`` feeds the **same** logits through a different batch layout.
    That isolates the accumulator from the floating-point differences between
    two real forward passes, which are not bit-identical on GPU.
    """
    evaluator = OriginalSpaceEvaluator(threshold=0.5)
    forward_ms: list[float] = []
    evidence: list[dict[str, Any]] = []
    no_grad_verified: list[bool] = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        items = [dataset[i] for i in chunk]
        if reuse:
            if cache is None:
                raise ValueError("reuse requires a logits cache")
            logits_list = [cache[item["image_id"]]["logits"] for item in items]
        else:
            batch = data.collate_pcb1(items)
            pixels = batch["pixel_values"].to(device, non_blocking=False)
            started = time.perf_counter()
            with torch.no_grad():
                logits = model(pixel_values=pixels).logits
                no_grad_verified.append(not torch.is_grad_enabled() and not logits.requires_grad and not model.training)
            if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
                raise ValueError("probe requires finite float32 logits (AMP is disabled)")
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            forward_ms.append(round((time.perf_counter() - started) * 1000.0, 2))
            logits_cpu = logits.float().cpu()
            logits_list = [logits_cpu[i] for i in range(len(items))]
        for item, tensor in zip(items, logits_list):
            probability = restore_defect_probability(tensor, item["geometry"])
            info = {"image_id": item["image_id"], "input_sha256": _array_sha(item["pixel_values"]),
                    "mask_sha256": _array_sha(item["original_mask"]), "geometry": item["geometry"],
                    "logits_sha256": _array_sha(tensor), "probability_sha256": _array_sha(probability),
                    "prediction_sha256": _array_sha(probability >= 0.5)}
            if cache is not None:
                if reuse:
                    original = {key: value for key, value in cache[item["image_id"]].items()
                                if key not in ("logits", "probability")}
                    if original != info:
                        raise ValueError("cached-logit reuse input/GT/restore identity changed")
                else:
                    cache[item["image_id"]] = {**info, "logits": tensor, "probability": probability}
            evidence.append(info)
            evaluator.add_sample(
                tensor,
                item["original_mask"],
                item["geometry"],
                image_id=item["image_id"],
                label=label_of[item["image_id"]],
                split="val",
                flipped=bool(item["flipped"]),
            )
    result = evaluator.result()
    result["sample_evidence"] = evidence
    result["runtime"] = {"model_training": bool(model.training), "real_forward_batches": len(no_grad_verified),
                         "all_eval_no_grad": all(no_grad_verified) if no_grad_verified else None,
                         "mode": "cached logits" if reuse else "eval + torch.no_grad; FP32; no autocast"}
    return result, forward_ms


def pipeline_probe(config: dict[str, Any], device: str) -> dict[str, Any]:
    # Also protect direct callers, including evaluation invoked inside training
    # autocast. The outer checker context records its own startup settings.
    with evaluation_precision(config, device=device) as precision:
        return _pipeline_probe(config, device, precision)


def _pipeline_probe(config: dict[str, Any], device: str, precision: dict[str, Any]) -> dict[str, Any]:
    from .modeling import load_all

    if config["raw"]["eval"].get("batch_size") != 1:
        raise ValueError("formal eval.batch_size must be 1")
    seed = int(config["raw"]["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    preview_path = Path(config["paths"]["logging.fixed_preview_manifest"])
    wanted = [line.strip() for line in preview_path.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.strip().startswith("#")]

    dataset = data.PCB1Dataset(config, "val", flip="none", return_original_mask=True)
    index_of = {row["image"]: i for i, row in enumerate(dataset.rows)}
    missing = [item for item in wanted if item not in index_of]
    if missing:
        raise SystemExit(f"preview ids not present in val: {missing}")
    indices = [index_of[item] for item in wanted]
    label_of = {row["image"]: row["label"] for row in dataset.rows}

    model_dir = Path(config["paths"]["model.pretrained_dir"])
    source_path = model_dir / "source.json"
    source_before = _sha256(source_path)
    # Reset directly before model creation: fixtures and Dataset setup cannot
    # consume the random-head seed implicitly.
    torch.manual_seed(seed)
    loaded = load_all(model_dir, num_labels=2, ignore_index=255)
    model = loaded.model.to(device).eval()
    head_before = _state_sha(model, "decode_head.")
    state_before = _state_sha(model)

    cache: dict[str, Any] = {}
    real_b2, ms_b2 = _run_probe(dataset, indices, label_of, model, device,
                                batch_size=2, cache=cache, reuse=False)
    # same logits, different batch layout: the accumulator must not care
    reuse_b1, _ = _run_probe(dataset, indices, label_of, model, device,
                             batch_size=1, cache=cache, reuse=True)
    # a second real forward at batch 1: GPU kernels may differ from batch 2
    cache_b1: dict[str, Any] = {}
    real_b1, ms_b1 = _run_probe(dataset, indices, label_of, model, device,
                                batch_size=1, cache=cache_b1, reuse=False)
    repeat_cache: dict[str, Any] = {}
    repeat_b1, ms_repeat_b1 = _run_probe(dataset, indices, label_of, model, device,
                                       batch_size=1, cache=repeat_cache, reuse=False)
    comparison = _compare_caches(cache, cache_b1)
    repeat_comparison = _compare_caches(cache_b1, repeat_cache)
    head_after = _state_sha(model, "decode_head.")
    state_after = _state_sha(model)
    source_after = _sha256(source_path)
    snapshot_problems = verify_snapshot(model_dir, json.loads(
        (Path(__file__).resolve().parents[1] / "reports" / "model_source.json").read_text(encoding="utf-8")))

    def _pack(result: dict[str, Any], ms: list[float]) -> dict[str, Any]:
        return {
            "confusion": result["confusion"],
            "global": result["global"],
            "normal_images": result["normal_images"],
            "anomaly_images": result["anomaly_images"],
            "per_image": result["per_image"],
            "forward_ms": ms,
            "sample_evidence": result["sample_evidence"],
            "runtime": result["runtime"],
        }

    return {
        "warning": "decode head is randomly initialised; these numbers are a pipeline probe, NOT a baseline",
        "not_a_baseline": True,
        "status": "executed",
        "device": device,
        "evaluation_precision": precision,
        "seed": seed,
        "formal_eval_batch_size": 1,
        "diagnostic_batch_sizes": [2, 1],
        "model_identity": {"same_instance_for_all_runs": True, "model_training": model.training,
                           "decode_head_sha256_before": head_before, "decode_head_sha256_after": head_after,
                           "state_sha256_before": state_before, "state_sha256_after": state_after,
                           "state_unchanged": state_before == state_after and head_before == head_after,
                           "all_real_forwards_eval_no_grad": all(run["runtime"]["all_eval_no_grad"]
                                                                  for run in (real_b2, real_b1, repeat_b1))},
        "images": wanted,
        "runs": {
            "real_forward_batch_2": _pack(real_b2, ms_b2),
            "same_logits_batch_1": _pack(reuse_b1, []),
            "real_forward_batch_1": _pack(real_b1, ms_b1),
            "real_forward_batch_1_repeat": _pack(repeat_b1, ms_repeat_b1),
        },
        "same_logits_layout_equal": same_logits_layout_equal(real_b2, reuse_b1),
        "two_forward_passes": comparison,
        "batch_1_repeat": repeat_comparison,
        "checkpoint_identity": {
            "revision": loaded.snapshot.get("revision"),
            "encoder_check_ok": loaded.encoder_check.get("ok"),
            "source_sha256_before": source_before,
            "source_sha256_after": source_after,
            "snapshot_unchanged": source_before == source_after == EXPECTED_SOURCE_SHA and not snapshot_problems,
            "post_probe_snapshot_problems": snapshot_problems,
        },
    }


# ------------------------------------------------------------------------ main ----
def provenance_paths(config: dict[str, Any], *, include_probe_data: bool) -> list[Path]:
    project = Path(__file__).resolve().parents[1]
    files = list((project / "segformer_lab").glob("*.py"))
    files += [project / "tests" / name for name in ("test_metrics_check.py", "test_losses_metrics.py", "test_evaluation.py")]
    files += [Path(config["path"]), project / "reports" / "model_source.json"]
    files += [project / "reports" / f"stage{stage}-acceptance.json" for stage in ("05", "06")]
    files += [Path(config["paths"][name]) for name in
              ("data.train_manifest", "data.val_manifest", "data.test_manifest", "data.split_meta", "logging.fixed_preview_manifest")]
    files += [Path(config["paths"]["data.split_meta"]).parent / name
              for name in ("reviewed_groups.json", "official_2cls_highshot.csv")]
    files += [Path(config["paths"]["model.pretrained_dir"]) / "source.json"]
    if include_probe_data:
        wanted = {line.strip() for line in Path(config["paths"]["logging.fixed_preview_manifest"]).read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.strip().startswith("#")}
        for row in data.read_manifest(config["paths"]["data.val_manifest"]):
            if row["image"] in wanted:
                files.append(Path(config["data_root"]) / row["image"])
                if row["mask"].strip():
                    files.append(Path(config["data_root"]) / row["mask"])
    return sorted(set(path.resolve() for path in files))


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stage 7: loss and original-space metrics check")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--report", default="reports/loss_metrics_check.json")
    parser.add_argument("--figure-dir", default="")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    config = load_config(args.config)
    with evaluation_precision(config, device=args.device) as precision:
        return _check(config, args, started, precision)


def _check(config: dict[str, Any], args: argparse.Namespace, started: float,
           precision: dict[str, Any]) -> int:
    if config["raw"]["eval"].get("batch_size") != 1:
        raise SystemExit("stage 7 requires formal eval.batch_size=1; refusing probe")
    if config["raw"]["eval"].get("threshold") != 0.5:
        raise SystemExit("stage 7 requires eval.threshold=0.5")
    print("stage 7: loss and metrics check")
    print(f"  config : {args.config}")
    print("  eval precision: IEEE FP32; cuDNN TF32=false, matmul TF32=false, autocast=false")

    identity = identity_checks(config)
    if not identity["ok"]:
        raise SystemExit("stage 7 identity preflight failed; refusing to read samples or load model")
    print(f"  identity: manifests_ok={identity['ok']} revision={identity['snapshot']['revision'][:12]}")
    tracked_files = provenance_paths(config, include_probe_data=not args.skip_model)
    before_files = [_file_record(path) for path in tracked_files]

    metrics = metric_fixtures()
    losses = loss_fixtures()
    restore, payload = restore_fixtures(config)

    figure_dir = Path(args.figure_dir) if args.figure_dir else (
        Path(config["raw"]["experiment"]["output_root"]) / "loss_metrics_check"
    )
    figure_path = figure_dir / "restore_asymmetric.png"
    render_restore_figure(payload, figure_path)
    print(f"  figure : {figure_path}")

    probe = None
    if not args.skip_model:
        probe = pipeline_probe(config, args.device)
        print(f"  probe  : same_logits_layout_equal={probe['same_logits_layout_equal']} "
              f"differing_pixels={probe['two_forward_passes']['differing_pixels']}")

    # ------------------------------------------------------------------ checks
    checks: dict[str, bool | None] = {}
    checks["identity_manifests"] = all(v["matches_acceptance"] for v in identity["manifests"].values())
    checks["identity_split_meta"] = identity["split_meta"]["matches_manifests"]
    checks["identity_reviewed_groups"] = identity["reviewed_groups"]["matches"]
    checks["identity_preview_ids"] = identity["preview_ids"]["matches"]
    checks["identity_snapshot"] = identity["snapshot"]["revision_matches"] and identity["snapshot"]["weights_match"]
    checks["identity_audit_data_verifier"] = identity["audit_data_verifier"]["ok"]
    checks["identity_acceptance_anchors"] = identity["acceptance_anchors_match"]
    checks["formal_eval_batch_size_one"] = config["raw"]["eval"]["batch_size"] == 1

    f = metrics["hand_2x2"]
    checks["hand_2x2_confusion"] = (
        f["confusion"]["tp"] == 1 and f["confusion"]["fp"] == 0
        and f["confusion"]["fn"] == 1 and f["confusion"]["tn"] == 2
    )
    checks["hand_2x2_iou"] = _close(f["iou"], 0.5)
    checks["hand_2x2_dice"] = _close(f["dice"], 2.0 / 3.0)
    checks["hand_2x2_precision_recall"] = _close(f["precision"], 1.0) and _close(f["recall"], 0.5)

    checks["perfect_iou_dice"] = _close(metrics["perfect"]["iou"], 1.0) and _close(metrics["perfect"]["dice"], 1.0)
    checks["all_background_prediction"] = (
        _close(metrics["all_background_prediction"]["iou"], 0.0)
        and _close(metrics["all_background_prediction"]["dice"], 0.0)
        and metrics["all_background_prediction"]["precision"] is None
        and _close(metrics["all_background_prediction"]["recall"], 0.0)
    )
    checks["all_foreground_prediction"] = (
        _close(metrics["all_foreground_prediction"]["iou"], 6.0 / 16.0)
        and _close(metrics["all_foreground_prediction"]["dice"], 12.0 / 22.0)
        and _close(metrics["all_foreground_prediction"]["recall"], 1.0)
    )
    checks["normal_false_positive"] = (
        _close(metrics["normal_false_positive"]["precision"], 0.0)
        and metrics["normal_false_positive"]["recall"] is None
        and _close(metrics["normal_false_positive"]["fp_pixel_ratio"], 3.0 / 16.0)
    )
    checks["normal_empty_prediction_is_na"] = (
        metrics["normal_empty_prediction"]["per_image_iou"] is None
        and metrics["normal_empty_prediction"]["per_image_dice"] is None
        and metrics["normal_empty_prediction"]["fp_pixel_ratio"] == 0.0
        and bool(metrics["normal_empty_prediction"]["reason"])
        and bool(metrics["normal_empty_prediction"]["dice_na"])
        and bool(metrics["normal_empty_prediction"]["precision_na"])
        and bool(metrics["normal_empty_prediction"]["recall_na"])
    )
    ignored = metrics["original_mask_ignore255"]
    checks["original_gt255_excluded"] = (ignored["confusion"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 1}
                                         and ignored["pixels"] == 3 and ignored["ignored_pixels"] == 1
                                         and _close(ignored["iou"], 0.5))
    checks["all_ignored_original_gt_rejected"] = metrics["all_ignored_original_mask"]["rejected"]
    m = metrics["mean_of_per_image_iou_is_not_global"]
    checks["global_is_not_mean_of_images"] = (
        _close(m["global_iou"], 2.0 / 101.0, 1e-9) and abs(m["mean_of_per_image"] - m["global_iou"]) > 0.3
    )
    layouts = metrics["batch_layout_invariance"]
    checks["batch_layout_invariance"] = (
        layouts["sequential"]["confusion"] == layouts["reversed"]["confusion"] == layouts["interleaved"]["confusion"]
        and layouts["sequential"]["iou"] == layouts["reversed"]["iou"] == layouts["interleaved"]["iou"]
    )

    c = losses["ce_hand_computed"]
    checks["ce_hand_computed"] = _close(c["ce"], c["expected"], 1e-7) and c["valid_pixels"] == 3 and c["ignored_pixels"] == 1
    checks["dice_hand_computed"] = _close(losses["dice_hand_computed"]["dice_loss"], 0.2, 1e-6)
    checks["perfect_prediction_dice_zero"] = losses["perfect_prediction_dice"]["dice_loss"] <= 1e-6
    n = losses["all_normal_batch"]
    checks["all_normal_dice_is_zero_and_backward"] = (
        n["dice_is_exactly_zero"] and n["positive_images"] == 0 and n["grad_present"] and n["grad_finite"]
    )
    checks["padding_does_not_affect_loss"] = losses["padding_does_not_affect_loss"]["abs_diff"] == 0.0
    checks["padding_lowres_inner_rows"] = losses["padding_lowres_inner_rows"]["abs_diff"] <= 1e-7
    checks["all_ignored_batch_rejected"] = losses["all_ignored_batch"]["rejected"]
    checks["batch_split_equivalence"] = (
        losses["batch_split_equivalence"]["ce_diff"] <= 1e-6
        and losses["batch_split_equivalence"]["dice_diff"] <= 1e-6
    )

    r = restore["asymmetric_restore"]
    checks["restore_centroid_matches_geometry"] = r["correct_error_px"] <= 3.0 and r["prob_shape"] == [1070, 1404]
    checks["restore_without_crop_is_wrong"] = r["wrong_error_px"] >= 20.0
    s = restore["single_pixel_dropped_at_512"]
    checks["single_pixel_scores_as_fn"] = (
        s["label512_positive_pixels"] == 0
        and s["original_gt"]["fn"] == 1
        and _close(s["original_gt"]["iou"], 0.0)
    )
    checks["shrunk_gt_would_hide_the_miss"] = (
        s["upscaled_shrunk_gt"]["fn"] == 0 and s["upscaled_shrunk_gt"]["iou"] is None
    )

    if probe is not None:
        checks["probe_real_data_ran"] = True
        checks["probe_accumulation_independent_of_layout"] = probe["same_logits_layout_equal"] is True
        checks["probe_two_passes_float_close"] = probe["two_forward_passes"]["float_close"]
        checks["probe_batch_one_repeat_float_close"] = probe["batch_1_repeat"]["float_close"]
        checks["probe_encoder_loaded"] = probe["checkpoint_identity"]["encoder_check_ok"] is True
        checks["probe_model_unchanged"] = probe["model_identity"]["state_unchanged"]
        checks["probe_eval_no_grad"] = probe["model_identity"]["all_real_forwards_eval_no_grad"]
        checks["probe_snapshot_unchanged"] = probe["checkpoint_identity"]["snapshot_unchanged"]
    else:
        for name in ("probe_real_data_ran", "probe_accumulation_independent_of_layout", "probe_two_passes_float_close",
                     "probe_batch_one_repeat_float_close", "probe_encoder_loaded", "probe_model_unchanged",
                     "probe_eval_no_grad", "probe_snapshot_unchanged"):
            checks[name] = None

    after_files = [_file_record(path) for path in tracked_files]
    checks["provenance_files_unchanged"] = before_files == after_files
    fixtures_ok = all(value is True for name, value in checks.items() if not name.startswith("probe_"))
    ok = fixtures_ok and probe is not None and all(value is True for value in checks.values())
    loss = config["raw"]["loss"]
    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "fixtures_only" if args.skip_model else "loss_metrics_with_random_head_pipeline_probe",
        "fixtures_ok": fixtures_ok,
        "full_stage_verified": bool(ok and probe is not None),
        "not_a_baseline": True,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": args.device,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "runtime_protocol": {"formal_eval_batch_size": config["raw"]["eval"]["batch_size"],
                             "evaluation_precision": precision,
                             "threshold": config["raw"]["eval"]["threshold"],
                             "seed": int(config["raw"]["experiment"]["seed"])},
        "provenance": {"files": before_files, "files_after": after_files,
                       "unchanged": before_files == after_files,
                       "resolved_config_sha256": hashlib.sha256(resolved_yaml(config).encode("utf-8")).hexdigest(),
                       "restore_figure": _file_record(figure_path)},
        "identity": identity,
        "loss_definition": loss_definition(ce_weight=float(loss["ce_weight"]),
                                           dice_weight=float(loss["foreground_dice_weight"]),
                                           eps=float(loss["dice_epsilon"])),
        "score_definition": score_definition(),
        "loss_fixtures": losses,
        "metric_fixtures": metrics,
        "restore_fixtures": restore,
        "random_head_pipeline_probe": probe if probe is not None else {
            "status": "skipped", "reason": "--skip-model; fixtures-only diagnostics cannot verify the full stage",
            "not_a_baseline": True},
        "checks": checks,
        "ok": ok,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    resolved = report_path.parent / "loss_metrics_resolved_config.yaml"
    write_resolved_config(config, resolved)

    failed = [name for name, value in checks.items() if value is False]
    skipped = [name for name, value in checks.items() if value is None]
    print()
    print(f"  checks passed: {sum(value is True for value in checks.values())}/{len(checks)}; skipped={len(skipped)}")
    if failed:
        print(f"  FAILED: {failed}")
        return 1
    if args.skip_model:
        print("  RESULT: FIXTURES ONLY; full_stage_verified=false, ok=false; model checks SKIPPED")
        print(f"  report: {report_path.resolve()}")
        return 0
    print("  RESULT: ok")
    print(f"  report: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
