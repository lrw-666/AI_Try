"""Stage 09: can the model learn these labels on 8 fixed training images?

This is a debugging experiment. It answers exactly one question -- "is the
label / geometry / loss / optimiser path wired correctly?" -- and says nothing
about generalisation. The output is tagged ``not_a_baseline`` and the metrics are
``train-subset`` numbers.

Design points that matter:

* the model is rebuilt from the pinned MiT-B0 snapshot, so nothing leaks in from
  the stage-08 smoke run;
* update 0 is scored *before* any optimiser step, which is the only honest
  baseline for "did it learn";
* evaluation happens every 50 successful updates on the same 8 images, in
  original space, B1, IEEE FP32 -- the same path stage 07 validated;
* at 100 updates a mandatory check runs. If nothing has been learned the run
  stops with a reason instead of burning the remaining budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from . import data, paths  # noqa: E402
from .config import load_config, write_resolved_config  # noqa: E402
from .losses import loss_definition  # noqa: E402
from .metrics import restore_defect_probability, score_definition  # noqa: E402
from .modeling import load_all  # noqa: E402
from .smoke import provenance, _labels_of, _eval_summary, sha256_file  # noqa: E402
from .training import (  # noqa: E402
    evaluate_dataset,
    parameter_digest,
    run_training,
    save_run,
    set_seed,
    trainable_parameter_summary,
)

SCHEMA = "segformer-lab/overfit/1"
NOT_A_BASELINE = True


def _iou(result: dict[str, Any]) -> float:
    return float(result["global"]["foreground_iou"] or 0.0)


def _learning_verdict(current: dict[str, Any], baseline: dict[str, Any], min_gain: float) -> dict[str, Any]:
    iou = _iou(current)
    base = _iou(baseline)
    tp = int(current["confusion"]["tp"])
    anomaly_macro = current["anomaly_images"]["macro"]["iou"]["mean"]
    return {
        "iou": round(iou, 8),
        "baseline_iou": round(base, 8),
        "iou_gain": round(iou - base, 8),
        "true_positive_pixels": tp,
        "anomaly_macro_iou": anomaly_macro,
        "min_iou_gain": min_gain,
        "learning": bool(tp > 0 and (iou - base) >= min_gain),
    }


def _render_sheet(samples: list[dict[str, Any]], data_root: Path, out_path: Path,
                  threshold: float, tag: str) -> str:
    rows = len(samples)
    figure, axes = plt.subplots(rows, 3, figsize=(9.2, 1.85 * rows), dpi=95)
    if rows == 1:
        axes = np.asarray([axes])
    for row, sample in enumerate(samples):
        image_id = sample["image_id"]
        probability = restore_defect_probability(sample["logits"], sample["geometry"]).numpy()
        with Image.open(data_root / image_id) as handle:
            handle.load()
            rgb = np.asarray(handle.convert("RGB"))
        gt = np.asarray(sample["original_mask"])
        pred = probability >= threshold
        tp = int((pred & (gt > 0)).sum())
        fp = int((pred & (gt == 0)).sum())
        fn = int((~pred & (gt > 0)).sum())
        inter = tp
        union = tp + fp + fn
        iou = None if union == 0 else inter / union

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_ylabel(f"{Path(image_id).stem}\n{sample['label']}", fontsize=7)
        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(rgb)
        overlay = np.ma.masked_where(probability < threshold, probability)
        axes[row, 2].imshow(overlay, cmap="autumn", alpha=0.45, vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("IoU NA" if iou is None else f"IoU {iou:.3f}", fontsize=7)
        for column in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        if row == 0:
            axes[row, 0].set_title("source", fontsize=8)
            axes[row, 1].set_title("original GT", fontsize=8)
    figure.suptitle(f"{tag} | threshold {threshold} | same 8 training images, same display scale", fontsize=9)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path)
    plt.close(figure)
    return out_path.name


def _render_zoom(sample: dict[str, Any], data_root: Path, out_path: Path, threshold: float,
                 tag: str) -> str:
    """Full-frame context above actual source/GT/probability/overlay crops."""
    probability = restore_defect_probability(sample["logits"], sample["geometry"]).numpy()
    with Image.open(data_root / sample["image_id"]) as handle:
        handle.load()
        rgb = np.asarray(handle.convert("RGB"))
    gt = np.asarray(sample["original_mask"])
    ys, xs = np.nonzero(gt > 0)
    if len(ys) == 0:
        y0, y1, x0, x1 = 0, gt.shape[0], 0, gt.shape[1]
    else:
        pad_y = max(24, int(0.06 * gt.shape[0]))
        pad_x = max(24, int(0.06 * gt.shape[1]))
        y0, y1 = max(0, int(ys.min()) - pad_y), min(gt.shape[0], int(ys.max()) + 1 + pad_y)
        x0, x1 = max(0, int(xs.min()) - pad_x), min(gt.shape[1], int(xs.max()) + 1 + pad_x)

    figure, axes = plt.subplots(2, 4, figsize=(11.5, 6.0), dpi=100)
    for row, region in enumerate((np.s_[:, :], np.s_[y0:y1, x0:x1])):
        frame, mask, prob = rgb[region], gt[region], probability[region]
        axes[row, 0].imshow(frame)
        axes[row, 0].set_title("source" if row == 0 else f"ROI x={x0}:{x1}, y={y0}:{y1}", fontsize=8)
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 1].set_title(f"GT: {int((mask > 0).sum())} px", fontsize=8)
        axes[row, 2].imshow(prob, cmap="magma", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("defect probability (0-1)", fontsize=8)
        axes[row, 3].imshow(frame)
        overlay = np.ma.masked_where(prob < threshold, prob)
        axes[row, 3].imshow(overlay, cmap="autumn", alpha=0.5, vmin=0.0, vmax=1.0)
        axes[row, 3].set_title(f"pred >= {threshold}", fontsize=8)
    for column in (0, 3):
        axes[0, column].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                fill=False, edgecolor="cyan", lw=1.0, ls="--"))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"{tag} | {Path(sample['image_id']).stem} | cyan ROI enlarged in bottom row", fontsize=9)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path)
    plt.close(figure)
    return out_path.name


def do_train(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = load_config(args.config)
    out_dir = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite an existing run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if config["raw"]["eval"].get("precision") != "ieee_fp32":
        raise SystemExit("eval.precision must be 'ieee_fp32'")
    if config["raw"]["train"].get("flip") != "none":
        raise SystemExit("stage 09 requires train.flip: none")
    amp = bool(config["raw"]["train"]["amp"])
    target = int(config["raw"]["train"]["max_steps"])
    attempt_cap = int(config["raw"]["overfit"]["max_attempts"])
    eval_every = int(config["raw"]["overfit"]["eval_every"])
    check_steps = [int(v) for v in config["raw"]["overfit"]["check_steps"]]
    min_gain = float(config["raw"]["overfit"]["min_iou_gain"])
    if target > attempt_cap:
        raise SystemExit("max_steps cannot exceed max_attempts")

    seed = int(config["raw"]["experiment"]["seed"])
    rng_record = set_seed(seed)
    device = args.device

    dataset = data.PCB1Dataset(config, "train", flip="none", return_original_mask=True)
    if len(dataset) != 8:
        raise SystemExit(f"training subset must be 8 images, got {len(dataset)}")
    labels = [row["label"] for row in dataset.rows]
    if labels.count("normal") != 4 or labels.count("anomaly") != 4:
        raise SystemExit(f"subset composition is {labels}")
    label_of = _labels_of(dataset)
    loader = data.build_dataloader(dataset, batch_size=int(config["raw"]["train"]["batch_size"]),
                                   shuffle=True, num_workers=0, drop_last=False)

    print(f"  run     : {out_dir}")
    print(f"  subset  : {len(dataset)} images ({labels.count('normal')}N+{labels.count('anomaly')}A), flip=none")
    loaded = load_all(Path(config["paths"]["model.pretrained_dir"]), num_labels=2, ignore_index=255)
    if not loaded.encoder_check.get("ok"):
        raise SystemExit("encoder weights failed the checkpoint comparison")
    model = loaded.model.to(device)
    params = trainable_parameter_summary(model)
    initial_digest = parameter_digest(model)

    # ---- update 0: the only honest baseline for "did it learn" --------------
    print("  eval    : update 0")
    baseline = evaluate_dataset(model, dataset, config, device, label_of=label_of, keep_probabilities=True)
    (out_dir / "eval").mkdir(parents=True, exist_ok=True)
    (out_dir / "eval" / "update_000.json").write_text(
        json.dumps(_eval_summary(baseline), indent=2, ensure_ascii=False), encoding="utf-8")
    _render_sheet(baseline["samples"], dataset.data_root, out_dir / "eval" / "sheet_update_000.png",
                  float(config["raw"]["eval"]["threshold"]), "overfit | update 0")
    smallest = min((s for s in baseline["samples"] if s["label"] == "anomaly"),
                   key=lambda s: int((s["original_mask"] > 0).sum()), default=None)
    if smallest is not None:
        _render_zoom(smallest, dataset.data_root, out_dir / "eval" / "zoom_smallest_update_000.png",
                     float(config["raw"]["eval"]["threshold"]), "overfit | update 0")

    curve: list[dict[str, Any]] = [{"update": 0, **_eval_summary(baseline)}]
    verdicts: dict[str, Any] = {}
    stop_reason: list[str] = []

    def on_update(current_model: Any, outcome: Any) -> None:
        if outcome.successful == 0 or outcome.successful % eval_every != 0:
            return
        print(f"  eval    : update {outcome.successful}")
        result = evaluate_dataset(current_model, dataset, config, device,
                                  label_of=label_of, keep_probabilities=True)
        summary = _eval_summary(result)
        (out_dir / "eval" / f"update_{outcome.successful:03d}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _render_sheet(result["samples"], dataset.data_root,
                      out_dir / "eval" / f"sheet_update_{outcome.successful:03d}.png",
                      float(config["raw"]["eval"]["threshold"]), f"overfit | update {outcome.successful}")
        if smallest is not None:
            again = next((s for s in result["samples"] if s["image_id"] == smallest["image_id"]), None)
            if again is not None:
                _render_zoom(again, dataset.data_root,
                             out_dir / "eval" / f"zoom_smallest_update_{outcome.successful:03d}.png",
                             float(config["raw"]["eval"]["threshold"]),
                             f"overfit | update {outcome.successful}")
        curve.append({"update": outcome.successful, **summary})
        if outcome.successful in check_steps:
            verdict = _learning_verdict(result, baseline, min_gain)
            verdicts[str(outcome.successful)] = verdict
            print(f"    learning at {outcome.successful}: {verdict}")

    def should_stop(current_model: Any, outcome: Any) -> str | None:
        verdict = verdicts.get(str(outcome.successful))
        if verdict is not None and not verdict["learning"] and outcome.successful == check_steps[0]:
            return (f"no learning at {outcome.successful} updates "
                    f"(tp={verdict['true_positive_pixels']}, iou_gain={verdict['iou_gain']:.4f}); "
                    "stop and diagnose instead of running to the cap")
        return None

    with (out_dir / "updates.jsonl").open("w", encoding="utf-8") as log_handle:
        outcome, optimizer, scaler, optimizer_info = run_training(
            model, loader, config, device,
            target_updates=target, max_attempts=attempt_cap,
            lr=float(config["raw"]["train"]["learning_rate"]),
            weight_decay=float(config["raw"]["train"]["weight_decay"]),
            grad_clip=float(config["raw"]["train"]["grad_clip_norm"]),
            amp=amp, accumulation=int(config["raw"]["train"]["gradient_accumulation_steps"]),
            on_update=on_update, should_stop=should_stop, log_handle=log_handle,
        )

    final = evaluate_dataset(model, dataset, config, device, label_of=label_of, keep_probabilities=True)
    if not curve or curve[-1]["update"] != outcome.successful:
        (out_dir / "eval" / f"update_{outcome.successful:03d}.json").write_text(
            json.dumps(_eval_summary(final), indent=2, ensure_ascii=False), encoding="utf-8")
        _render_sheet(final["samples"], dataset.data_root,
                      out_dir / "eval" / f"sheet_update_{outcome.successful:03d}.png",
                      float(config["raw"]["eval"]["threshold"]), f"overfit | update {outcome.successful}")
        curve.append({"update": outcome.successful, **_eval_summary(final)})

    manifest = save_run(
        out_dir, model, loaded.processor, config,
        optimizer=optimizer, scaler=scaler,
        successful_updates=outcome.successful, attempts=outcome.attempts, skipped=outcome.skipped,
        extra={"stage": "09", "not_a_baseline": NOT_A_BASELINE},
    )
    write_resolved_config(config, out_dir / "resolved_config.yaml")

    curve_path = out_dir / "evaluation_curve.json"
    curve_path.write_text(json.dumps({
        "not_a_baseline": True,
        "scope": "train-subset (the same 8 training images)",
        "threshold": float(config["raw"]["eval"]["threshold"]),
        "points": [{"update": point["update"],
                    "global_iou": point["global"]["foreground_iou"],
                    "global_dice": point["global"]["foreground_dice"],
                    "tp": point["confusion"]["tp"], "fp": point["confusion"]["fp"],
                    "fn": point["confusion"]["fn"],
                    "anomaly_macro_iou": point["anomaly_images"]["macro"]["iou"]["mean"],
                    "normal_fp_ratio": point["normal_images"]["fp_pixel_ratio"]}
                   for point in curve],
        "verdicts": verdicts,
        "baseline_update": 0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "schema": SCHEMA,
        "stage": "09",
        "not_a_baseline": NOT_A_BASELINE,
        "statement": "8-image overfit debugging run; original-space scores on the training "
                     "subset, not a generalisation result",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "config": {"path": str(Path(config["path"]).resolve()), "sha256": sha256_file(Path(config["path"]))},
        "seed": rng_record,
        "device": device,
        "train_subset": {
            "ids_file": str(config["paths"]["data.train_ids_file"]),
            "ids_sha256": sha256_file(Path(config["paths"]["data.train_ids_file"])),
            "images": [row["image"] for row in dataset.rows],
            "labels": labels,
        },
        "parameters": params,
        "optimizer": optimizer_info,
        "initial_parameter_digest": initial_digest,
        "loss": loss_definition(ce_weight=float(config["raw"]["loss"]["ce_weight"]),
                                dice_weight=float(config["raw"]["loss"]["foreground_dice_weight"]),
                                eps=float(config["raw"]["loss"]["dice_epsilon"])),
        "score_definition": score_definition(),
        "training": {
            "successful_updates": outcome.successful,
            "attempts": outcome.attempts,
            "skipped": outcome.skipped,
            "stopped_reason": outcome.stopped_reason,
            "parameter_evidence": outcome.parameter_evidence,
            "timing": outcome.timing,
            "memory": outcome.memory,
        },
        "baseline_update_0": _eval_summary(baseline),
        "final": _eval_summary(final),
        "verdicts": verdicts,
        "evaluation_curve": str(curve_path),
        "artifacts": manifest,
        "encoder_check": {"matched_exact": loaded.encoder_check.get("matched_exact"),
                          "checkpoint_tensors": loaded.encoder_check.get("checkpoint_tensor_count"),
                          "max_abs_diff": loaded.encoder_check.get("max_abs_diff")},
        "provenance": provenance(config),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  updates : {outcome.successful} successful / {outcome.attempts} attempts, "
          f"skipped={outcome.skipped}")
    print(f"  stop    : {outcome.stopped_reason}")
    print(f"  timing  : median {outcome.timing['median_step_ms']} ms/update, "
          f"peak reserved {outcome.memory['gpu'].get('peak_reserved_mib')} MiB")
    print(f"  iou     : update0={_iou(baseline):.4f} -> final={_iou(final):.4f} "
          f"(train subset, not a baseline)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stage 09: 8-image overfit check")
    parser.add_argument("--config", default="configs/overfit.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return do_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
