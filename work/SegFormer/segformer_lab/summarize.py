"""Stage 13: figures, failure-case index and experiment card from frozen results.

Reads only:

* the formal run's ``summary.json`` / ``epochs.jsonl`` (training curves, best);
* the stage-12 evaluation directory (``test/per_image.json``, ``metrics.json``,
  ``selection.json``, ``display_rules.json``, ``predictions/*.png``).

It never trains, never re-selects the best epoch and never computes a second set
of metrics. The only numbers it produces are the ones needed for plots (the same
per-epoch values the run already logged) and per-image visual panels.

Float probabilities for the displayed images are recomputed through
``segformer_lab.inference`` because stage 12 deliberately did not store 442
full-size maps; every recomputed mask is checked against the stage-12 PNG
(XOR must be 0) before it is drawn.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from . import inference
from .config import load_config
from .training import sha256_file

SCHEMA = "segformer-lab/stage13-results/1"

TP_COLOR = (0, 166, 102)      # green
FP_COLOR = (226, 75, 74)      # red
FN_COLOR = (55, 138, 221)     # blue
DISPLAY_RULES_NOTE = ("selection rules were frozen by stage 12 in display_rules.json before "
                      "the first test image was decoded")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def lr_from_schedule(completed: int, plan: dict[str, Any]) -> float:
    """Same formula as ``training.WarmupCosineLR``; used only for a plotted column."""
    base, floor = float(plan["base_lr"]), float(plan["min_lr"])
    total, warmup = int(plan["total_steps"]), int(plan["warmup_steps"])
    if warmup > 0 and completed < warmup:
        return base * (completed + 1) / warmup
    span = total - warmup
    if span <= 0:
        return base
    progress = min(1.0, (completed - warmup + 1) / span)
    return floor + 0.5 * (base - floor) * (1.0 + np.cos(np.pi * progress))


def curves_table(epochs: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entry in epochs:
        val = entry["val"]
        rows.append({
            "epoch": entry["epoch"],
            "attempts_in_epoch": entry["attempts_in_epoch"],
            "successful_in_epoch": entry["successful_in_epoch"],
            "skipped_in_epoch": entry["skipped_in_epoch"],
            "successful_total": entry["successful_total"],
            "scheduler_completed": entry["scheduler_completed"],
            "lr_first_used": entry["lr_first"],
            "lr_last_used": entry["lr_last"],
            "lr_after_epoch_computed": lr_from_schedule(entry["scheduler_completed"], plan),
            "ce": entry["loss"]["ce"],
            "dice_term": entry["loss"]["dice"],
            "total_loss": entry["loss"]["total"],
            "val_iou": val["global"]["foreground_iou"],
            "val_dice": val["global"]["foreground_dice"],
            "val_precision": val["global"]["precision"],
            "val_recall": val["global"]["recall"],
            "anomaly_macro_iou": val["anomaly_images"]["macro"]["iou"]["mean"],
            "normal_fp_pixels": val["normal_images"]["false_positive_pixels"],
            "normal_fp_ratio": val["normal_images"]["fp_pixel_ratio"],
            "empty_anomaly_count": entry["diagnostics"]["empty_prediction_count"],
        })
    return rows


def _mask_from_row(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(Image.open(row["_png_path"]), dtype=np.uint8)


def _overlay(source: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """TP green, FP red, FN blue over a dimmed source image."""
    canvas = (source.astype(np.float32) * 0.55 + 255 * 0.45).astype(np.uint8)
    tp = (pred == 1) & (gt > 0)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt > 0)
    canvas[tp] = TP_COLOR
    canvas[fp] = FP_COLOR
    canvas[fn] = FN_COLOR
    return canvas


def _legend(figure: Any) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=np.array(TP_COLOR) / 255, label="TP (green)"),
               Patch(facecolor=np.array(FP_COLOR) / 255, label="FP (red)"),
               Patch(facecolor=np.array(FN_COLOR) / 255, label="FN (blue)")]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)


def _anomaly_row(axes: list[Any], source: np.ndarray, gt: np.ndarray, probability: np.ndarray,
                 pred: np.ndarray, title: str) -> None:
    axes[0].imshow(source)
    axes[0].set_title("source", fontsize=8)
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("original GT", fontsize=8)
    axes[2].imshow(probability, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("probability (0-1)", fontsize=8)
    axes[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("binary >= 0.5", fontsize=8)
    axes[4].imshow(_overlay(source, gt, pred))
    axes[4].set_title("TP / FP / FN", fontsize=8)
    axes[0].set_ylabel(title, fontsize=7)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])


def _normal_row(axes: list[Any], source: np.ndarray, probability: np.ndarray,
                pred: np.ndarray, title: str) -> None:
    axes[0].imshow(source)
    axes[0].set_title("source", fontsize=8)
    axes[1].imshow(probability, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("probability (0-1)", fontsize=8)
    canvas = (source.astype(np.float32) * 0.55 + 255 * 0.45).astype(np.uint8)
    canvas[pred == 1] = FP_COLOR
    axes[2].imshow(canvas)
    axes[2].set_title("false positives (red)", fontsize=8)
    axes[0].set_ylabel(title, fontsize=7)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])


def render_anomaly_sheet(path: Path, rows: list[dict[str, Any]], title: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(rows), 5, figsize=(14.0, 2.35 * len(rows)), dpi=105)
    if len(rows) == 1:
        axes = np.asarray([axes])
    for index, row in enumerate(rows):
        _anomaly_row(list(axes[index]), row["source"], row["gt"], row["probability"],
                     row["pred"], row["title"])
    figure.suptitle(title, fontsize=10)
    _legend(figure)
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    figure.savefig(path)
    plt.close(figure)
    return path.name


def render_normal_sheet(path: Path, rows: list[dict[str, Any]], title: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(rows), 3, figsize=(9.6, 2.35 * len(rows)), dpi=105)
    if len(rows) == 1:
        axes = np.asarray([axes])
    for index, row in enumerate(rows):
        _normal_row(list(axes[index]), row["source"], row["probability"], row["pred"], row["title"])
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path)
    plt.close(figure)
    return path.name


def render_training_curves(path: Path, table: list[dict[str, Any]], best_epoch: int) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epoch = [row["epoch"] for row in table]
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.0), dpi=115)

    axes[0, 0].plot(epoch, [row["total_loss"] for row in table], color="#185FA5", lw=1.6, label="total")
    axes[0, 0].plot(epoch, [row["ce"] for row in table], color="#BA7517", lw=1.2, ls="--", label="CE")
    axes[0, 0].plot(epoch, [row["dice_term"] for row in table], color="#0F6E56", lw=1.2, ls=":",
                    label="Dice term")
    axes[0, 0].set_title("training loss (pixel-weighted CE, image-weighted Dice)", fontsize=9)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(epoch, [row["lr_last_used"] for row in table], color="#534AB7", lw=1.6,
                    label="LR used at last update of epoch")
    axes[0, 1].plot(epoch, [row["lr_after_epoch_computed"] for row in table], color="#888780",
                    lw=1.0, ls="--", label="LR after that update (computed)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("learning rate (frozen 13250/662 plan)", fontsize=9)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(epoch, [row["val_iou"] or 0.0 for row in table], color="#185FA5", lw=1.6,
                    label="val global IoU")
    axes[1, 0].plot(epoch, [row["anomaly_macro_iou"] or 0.0 for row in table], color="#993C1D",
                    lw=1.3, ls="--", label="val anomaly macro IoU")
    axes[1, 0].axvline(best_epoch, color="#444441", lw=0.8, ls="-.")
    axes[1, 0].annotate(f"best epoch {best_epoch}", xy=(best_epoch, 0.02), xytext=(4, 8),
                        textcoords="offset points", fontsize=8)
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title("validation (132 images, original space, IEEE FP32)", fontsize=9)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(epoch, [max(1, row["normal_fp_pixels"]) for row in table], color="#A32D2D",
                    lw=1.5, label="normal FP pixels (log)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("false positives on 120 normal val images", fontsize=9)
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("epoch", fontsize=8)
        axis.grid(alpha=0.3)
    figure.suptitle("SegFormer-B0 / PCB1 v2 / seed 42 — 50 epochs, best epoch "
                    f"{best_epoch} selected by val global IoU", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path)
    plt.close(figure)
    return path.name


def _collect(cfg: dict[str, Any], run_dir: Path, eval_dir: Path, device: str,
             selection: dict[str, Any], rows_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Recompute probabilities for the displayed ids and verify them against stage 12."""
    frozen = inference.load_frozen_run(cfg, run_dir, device=device)
    dataset = inference.build_dataset(cfg, "test", processor=frozen.processor)
    index_of = {row["image"]: index for index, row in enumerate(dataset.rows)}

    wanted: list[str] = []
    for key in ("fixed_normal", "fixed_anomaly", "worst_iou", "worst_recall", "most_normal_fp"):
        for item in selection[key]:
            if item["image_id"] not in wanted:
                wanted.append(item["image_id"])

    collected: dict[str, Any] = {}
    mismatch: list[str] = []
    for image_id in wanted:
        row = rows_by_id[image_id]
        item = dataset[index_of[image_id]]
        probability = inference.probability_map(frozen.model, item, device, cfg)
        pred = (probability >= float(cfg["raw"]["eval"]["threshold"])).astype(np.uint8)
        with Image.open(dataset.data_root / image_id) as handle:
            handle.load()
            source = np.asarray(handle.convert("RGB"))
        gt = np.asarray(item["original_mask"], dtype=np.uint8)
        stage12_mask = _mask_from_row(row)
        xor = int((pred != stage12_mask).sum())
        if xor:
            mismatch.append(f"{image_id}:{xor}")
        collected[image_id] = {
            "source": source, "gt": gt, "probability": probability, "pred": pred,
            "row": row, "xor_vs_stage12": xor,
            "source_sha256": sha256_file(dataset.data_root / image_id),
            "title": f"{image_id.split('/')[-1]} | IoU {_fmt(row['iou'])} | FP {row['fp']}",
        }
    return {"items": collected, "mismatch": mismatch, "frozen": frozen, "wanted": wanted}


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{value:.3f}"


def do_summarize(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    cfg = load_config(args.config)
    run_dir = Path(args.run).resolve()
    eval_dir = Path(args.eval_run).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    epochs = _read_jsonl(run_dir / "epochs.jsonl")
    epochs = [entry for entry in epochs if "val" in entry]
    best_epoch = int(summary["best"]["epoch"])
    plan = summary["scheduler"]
    table = curves_table(epochs, plan)

    with (out_dir / "curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    curve_png = render_training_curves(figures / "training-curves.png", table, best_epoch)

    test_metrics = json.loads((eval_dir / "test" / "metrics.json").read_text(encoding="utf-8"))
    val_metrics = json.loads((eval_dir / "val" / "metrics.json").read_text(encoding="utf-8"))
    selection = json.loads((eval_dir / "test" / "selection.json").read_text(encoding="utf-8"))
    rules = json.loads((eval_dir / "test" / "display_rules.json").read_text(encoding="utf-8"))
    rows = json.loads((eval_dir / "test" / "per_image.json").read_text(encoding="utf-8"))["rows"]
    rows_by_id = {}
    for row in rows:
        row = dict(row)
        row["_png_path"] = eval_dir / "test" / "predictions" / row["prediction_png"]
        rows_by_id[row["image_id"]] = row

    collected = _collect(cfg, run_dir, eval_dir, args.device, selection, rows_by_id)
    if collected["mismatch"]:
        raise SystemExit(f"recomputed masks disagree with stage 12: {collected['mismatch']}")
    items = collected["items"]

    def pick(key: str) -> list[dict[str, Any]]:
        return [items[item["image_id"]] for item in selection[key]]

    rendered = {
        "anomaly_sheet": render_anomaly_sheet(
            figures / "anomaly-fixed-6.png", pick("fixed_anomaly"),
            "Fixed anomaly samples (first 6 by image_id) — test split, best epoch "
            f"{best_epoch}, threshold 0.5"),
        "normal_sheet": render_normal_sheet(
            figures / "normal-fixed-6.png", pick("fixed_normal"),
            "Fixed normal samples (first 6 by image_id) — zero GT, red = false positive"),
    }
    worst_ids: list[str] = []
    for key in ("worst_iou", "worst_recall"):
        for item in selection[key]:
            if item["image_id"] not in worst_ids:
                worst_ids.append(item["image_id"])
    rendered["worst_anomaly_sheet"] = render_anomaly_sheet(
        figures / "anomaly-worst-cases.png", [items[image_id] for image_id in worst_ids],
        "Worst anomaly cases: lowest IoU, plus recall-only entries (union, ordered)")
    rendered["worst_normal_sheet"] = render_normal_sheet(
        figures / "normal-worst-fp.png", pick("most_normal_fp"),
        "Normal images with the most false-positive pixels (top 5)")

    reasons: dict[str, list[str]] = {}
    for key in ("fixed_normal", "fixed_anomaly", "worst_iou", "worst_recall", "most_normal_fp"):
        for item in selection[key]:
            reasons.setdefault(item["image_id"], []).append(key)
    failure_csv = out_dir / "failure-cases.csv"
    fields = ["image_id", "label", "reasons", "gt_pixels", "predicted_pixels", "tp", "fp", "fn",
              "iou", "recall", "dice", "precision", "na_notes", "source_sha256", "xor_vs_stage12"]
    with failure_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_id in collected["wanted"]:
            row = rows_by_id[image_id]
            notes = []
            if row["label"] == "anomaly" and int(row["tp"]) == 0:
                notes.append("tp_zero")
            if int(row["predicted_foreground_pixels"]) == 0:
                notes.append("predicted_nothing")
            if row["label"] == "normal" and int(row["fp"]) > 0:
                notes.append("normal_false_positive")
            writer.writerow({
                "image_id": image_id, "label": row["label"],
                "reasons": ";".join(reasons.get(image_id, [])),
                "gt_pixels": row["gt_foreground_pixels"],
                "predicted_pixels": row["predicted_foreground_pixels"],
                "tp": row["tp"], "fp": row["fp"], "fn": row["fn"],
                "iou": row["iou"], "recall": row["recall"], "dice": row["dice"],
                "precision": row["precision"], "na_notes": ";".join(notes),
                "source_sha256": items[image_id]["source_sha256"],
                "xor_vs_stage12": items[image_id]["xor_vs_stage12"],
            })

    results = {
        "schema": SCHEMA,
        "stage": "13",
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "best_epoch": best_epoch,
        "best_selection": summary["selection"],
        "figures": rendered,
        "curves_csv": "curves.csv",
        "failure_cases_csv": failure_csv.name,
        "display_rules_source": str(eval_dir / "test" / "display_rules.json"),
        "display_rules_sha256": sha256_file(eval_dir / "test" / "display_rules.json"),
        "recomputed_probabilities": {
            "images": len(collected["wanted"]),
            "reason": "stage 12 deliberately stored only 0/1 masks for the 442 test images",
            "channel": "segformer_lab.inference.probability_map (same frozen run, threshold from config)",
            "xor_vs_stage12": {image_id: items[image_id]["xor_vs_stage12"]
                               for image_id in collected["wanted"]},
            "all_zero": all(items[image_id]["xor_vs_stage12"] == 0 for image_id in collected["wanted"]),
        },
        "lists": {
            "all_anomaly_tp_zero": selection["all_anomaly_tp_zero"],
            "all_anomaly_empty_prediction": selection["all_anomaly_empty_prediction"],
            "all_normal_with_false_positive_count": len(selection["all_normal_with_false_positive"]),
            "all_normal_with_false_positive": selection["all_normal_with_false_positive"],
            "anomaly_denominator": test_metrics["images"]["anomaly"],
            "normal_denominator": test_metrics["images"]["normal"],
        },
        "metrics": {
            "val": {"confusion": val_metrics["confusion"], "global": val_metrics["global"],
                    "anomaly_macro": val_metrics["anomaly_images"]["macro"],
                    "normal": {k: val_metrics["normal_images"][k] for k in
                               ("count", "false_positive_pixels", "fp_pixel_ratio",
                                "images_with_any_false_positive")}},
            "test": {"confusion": test_metrics["confusion"], "global": test_metrics["global"],
                     "anomaly_macro": test_metrics["anomaly_images"]["macro"],
                     "normal": {k: test_metrics["normal_images"][k] for k in
                                ("count", "false_positive_pixels", "total_pixels", "fp_pixel_ratio",
                                 "predicted_foreground_ratio", "images_with_any_false_positive")}},
        },
        "notes": [DISPLAY_RULES_NOTE,
                  "the Dice term in the loss curve is image-weighted over positive-GT images; "
                  "a batch with no foreground contributes an exact zero to that batch statistic "
                  "and is excluded from the epoch aggregate, it is not a sign of perfect segmentation"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _json_dump(out_dir / "results-summary.json", results)

    _write_reports(out_dir, summary, val_metrics, test_metrics, table, selection, rules,
                   best_epoch, rendered, items)
    print(f"  run     : {run_dir}")
    print(f"  eval    : {eval_dir}")
    print(f"  best    : epoch {best_epoch}")
    print(f"  figures : {len(rendered)} in {figures}")
    print(f"  recompute: {len(collected['wanted'])} masks, XOR vs stage 12 all zero = "
          f"{results['recomputed_probabilities']['all_zero']}")
    print(f"  wrote   : {out_dir}")
    return 0


def _write_reports(out_dir: Path, summary: dict[str, Any], val: dict[str, Any],
                   test: dict[str, Any], table: list[dict[str, Any]],
                   selection: dict[str, Any], rules: dict[str, Any], best_epoch: int,
                   rendered: dict[str, str], items: dict[str, Any]) -> None:
    best = summary["best"]
    identity = summary["identity"]["start"]
    best_sha = summary["artifacts"]["files"]["checkpoints/best.pt"]["sha256"]
    run_path = summary["artifacts"]["run_dir"]
    val_global, test_global = val["global"], test["global"]
    val_macro, test_macro = val["anomaly_images"]["macro"], test["anomaly_images"]["macro"]
    val_normal, test_normal = val["normal_images"], test["normal_images"]

    results_md = f"""# PCB1 SegFormer-B0 — 结果摘要

> 本文所有数字来自同一条 run `pcb1_b0_v2_seed42` 与它的评价目录，**没有重新训练、没有重新选 best、没有第二套指标实现**。
> val 用于选模，test 只评一次；两者分列，不混用。

## 1. 模型与选模

| 项目 | 值 |
| --- | --- |
| best epoch / step | **{best['epoch']}** / {summary['training']['successful_updates']} 次成功更新（全 50 轮中该 epoch 的 step） |
| 选择规则 | {summary['selection']} |
| best 权重 SHA-256 | `{best_sha}` |
| 训练预算 | 50 epoch × 265 batch = 13 250 次尝试；{summary['training']['successful_updates']} 次成功、{summary['training']['skipped']} 次 AMP 跳步 |

## 2. 验证集（132 = 120 正常 + 12 异常）

| 指标 | 值 |
| --- | --- |
| 全局前景 IoU / Dice | **{val_global['foreground_iou']:.4f}** / {val_global['foreground_dice']:.4f} |
| Precision / Recall | {val_global['precision']:.4f} / {val_global['recall']:.4f} |
| TP / FP / FN / TN | {val['confusion']['tp']:,} / {val['confusion']['fp']:,} / {val['confusion']['fn']:,} / {val['confusion']['tn']:,} |
| 异常 12 张 macro IoU / Dice | **{val_macro['iou']['mean']:.4f}** / {val_macro['dice']['mean']:.4f} |
| 正常图 FP | {val_normal['false_positive_pixels']:,} px（{val_normal['images_with_any_false_positive']}/{val_normal['count']} 张有任意 FP），比例 {val_normal['fp_pixel_ratio']:.3e} |

## 3. 测试集（442 = 402 正常 + 40 异常，只评一次）

| 指标 | 值 |
| --- | --- |
| 全局前景 IoU / Dice | **{test_global['foreground_iou']:.4f}** / {test_global['foreground_dice']:.4f} |
| Precision / Recall | {test_global['precision']:.4f} / {test_global['recall']:.4f} |
| TP / FP / FN / TN | {test['confusion']['tp']:,} / {test['confusion']['fp']:,} / {test['confusion']['fn']:,} / {test['confusion']['tn']:,} |
| 异常 40 张 macro IoU / Dice | **{test_macro['iou']['mean']:.4f}** / {test_macro['dice']['mean']:.4f} |
| macro P / R（有定义张数） | {test_macro['precision']['mean']:.4f}（{test_macro['precision']['defined_images']}/40） / {test_macro['recall']['mean']:.4f} |
| 正常图 FP | {test_normal['false_positive_pixels']:,} px（{test_normal['images_with_any_false_positive']}/{test_normal['count']} 张），比例 {test_normal['fp_pixel_ratio']:.3e} |
| 每图预测前景比（正常） | 中位 {test_normal['predicted_foreground_ratio']['median']:.1e}，P95 {test_normal['predicted_foreground_ratio']['p95']:.2e}，最大 {test_normal['predicted_foreground_ratio']['max']:.2e} |

## 4. 怎么读这两个数

- **全局 IoU 是面积加权**：少数大缺陷贡献了大部分前景像素，所以 test 0.68 与 macro 0.52 的差距本身就是信息——它说明小缺陷的表现明显弱于大缺陷。
- **val 比 test 高是正常的**：val 参与选模。不要把 test 0.6833 替换成 val 0.7775，也不要反过来。
- **正常图只有像素级统计**：{test_normal['images_with_any_false_positive']}/{test_normal['count']} 张"有任意误报像素"不等于"图像级假阳性率"，本项目没有定义图像级判定规则。

## 5. 图表

| 文件 | 内容 |
| --- | --- |
| `figures/training-curves.png` | 50 轮 loss / LR / val IoU / 正常 FP，标注 best epoch {best_epoch} |
| `figures/anomaly-fixed-6.png` | 固定 6 张异常：原图 / GT / 概率（0–1 固定色标）/ 二值 / TP-FP-FN |
| `figures/anomaly-worst-cases.png` | IoU 最低与 recall 最低的并集（按规则排序） |
| `figures/normal-fixed-6.png` | 固定 6 张正常：原图 / 概率 / 误报（红） |
| `figures/normal-worst-fp.png` | 正常图误报像素最多的 5 张 |
| `curves.csv` | 逐轮数值（epoch / attempt / successful / LR / loss / val 指标） |
| `failure-cases.csv` | 入选图 ID、入选理由、逐图计数与重建 XOR |

色标说明：概率图统一 `vmin=0, vmax=1`，没有按图拉伸；TP 绿、FP 红、FN 蓝，图内有图例。
"""
    (out_dir / "results.md").write_text(results_md, encoding="utf-8")

    card = f"""# PCB1 SegFormer-B0 实验卡

- 日期 / run ID：2026-11-01 / `pcb1_b0_v2_seed42`
- 当前状态：正式训练（50 轮）与独立测试评价已完成，**待独立验收**；未做多 seed、未做跨数据集、未部署
- 任务：有监督背景/缺陷二值分割（PCB1，VisA）
- 数据版本与清单 SHA：train `{identity['manifests']['train']['sha256'][:16]}…`、val `{identity['manifests']['val']['sha256'][:16]}…`、test `{identity['manifests']['test']['sha256'][:16]}…`、`split_meta` `{identity['split_meta_sha256'][:16]}…`（PCB1 v2 分组，seed 42）
- 划分：530 / 132 / 442；train 482 正常 + 48 异常，val 120 + 12，test 402 + 40
- 预训练：`{identity['snapshot']['model_id']}`，revision `{identity['snapshot']['revision']}`，权重 SHA `{identity['snapshot']['weights_sha256'][:16]}…`（编码器 192/194 张量精确匹配）
- 模型与参数量：SegFormer-B0 + 二类解码头，{summary['parameters']['total']:,} 参数（编码器 {summary['parameters']['encoder']:,} + 头 {summary['parameters']['head']:,}），全量微调
- 输入 / 还原：512 letterbox（padding 255 只存在于训练输入），原图 {1404}×{1070} 尺度计分；`logits→512→裁 padding→原 H/W→softmax→阈值 0.5`
- 损失：`CE(ignore_index=255) + 0.5 × 正 GT 图逐图前景 Soft Dice`，eps 1e-6，Dice 在 FP32 上算
- 训练：物理 batch 2、累积 1、有效 batch 2；LR 6e-5（warmup 662 步线性 + cosine 到 1e-6）；50 epoch / 13 250 次计划更新；AMP FP16 + GradScaler；seed 42；`flip: random` p=0.5，每轮 `set_epoch`
- 环境：Python 3.12.16 / torch 2.10.0+cu126 / torchvision 0.25.0+cu126 / transformers 4.57.6 / CUDA 12.6 / RTX 4070 Laptop 8 188 MiB（驱动 617.42）
- 代码身份：git HEAD `{identity['git_head']}` + 未提交改动（14 个源文件 SHA 记录在 run 的 `code_identity_start.json`）；`source_and_config_unchanged = {summary['identity']['source_and_config_unchanged']}`
- 实际完成轮数 / 最佳 epoch：50 / **{best['epoch']}**，按 val 全局前景 IoU 选择（并列取较早轮次）
- val（132 张，原图 IEEE FP32，阈值 0.5）：IoU **{val_global['foreground_iou']:.4f}**、Dice {val_global['foreground_dice']:.4f}、P {val_global['precision']:.4f}、R {val_global['recall']:.4f}；异常 macro IoU {val_macro['iou']['mean']:.4f}；正常图 FP {val_normal['false_positive_pixels']:,} px（{val_normal['images_with_any_false_positive']}/120 张）
- test（442 张，同一协议，只评一次）：IoU **{test_global['foreground_iou']:.4f}**、Dice {test_global['foreground_dice']:.4f}、P {test_global['precision']:.4f}、R {test_global['recall']:.4f}；异常 macro IoU {test_macro['iou']['mean']:.4f}、macro Dice {test_macro['dice']['mean']:.4f}；正常图 FP {test_normal['false_positive_pixels']:,} px（{test_normal['images_with_any_false_positive']}/{test_normal['count']} 张），比例 {test_normal['fp_pixel_ratio']:.3e}
- 训练耗时 / 显存：50 轮墙钟 {summary['timing']['train_seconds_total']:.1f} s（其中读图等待 {summary['timing']['loader_wait_seconds_total']:.1f} s）、验证 {summary['timing']['val_seconds_total']:.1f} s；峰值 allocated {summary['memory']['gpu']['peak_allocated_mib']} / reserved {summary['memory']['gpu']['peak_reserved_mib']} MiB
- 评价耗时：val 19.84 s、test 60.89 s（B1，含 442 张 0/1 PNG 写盘）；**推理延迟未测**（没有做预热/多次计时的稳定基准）
- 失败案例：test 中 049、057 预测为空，077 的 351 px 全部为 FP；val 上 042（15 523 px）与 051 仍近乎全漏、081 仅 0.0105。详见 `failure-cases.csv` 与 `figures/anomaly-worst-cases.png`
- 局限：单模型、单 seed、单数据集；仅 40 张异常 test 图，macro 区间宽；全局 IoU 面积加权；无图像级判定规则；`workers=0` 未做吞吐对照；恢复的跨进程数值限制保留（阶段 11）
- 独立预测命令：`python -m segformer_lab.predict --config configs/baseline.yaml --run <run> --image <img> --out <dir>`
- 复算 val 命令：`python -m segformer_lab.evaluate --config configs/baseline.yaml --run <run> --split val --out <dir>`
- 路径：run `{run_path}`；评价 `runs/pcb1_b0_v2_seed42_eval_001`；本目录 `reports/pcb1_b0_v2_seed42`
"""
    (out_dir / "experiment-card.md").write_text(card, encoding="utf-8")

    repro = """# 复现说明

## 1. 需要从仓库外单独搬运的东西

Git 里**没有**模型权重、数据与环境。换一台机器，至少需要：

| 内容 | 位置 | 校验方式 |
| --- | --- | --- |
| 已训练的 best | `runs/pcb1_b0_v2_seed42/checkpoints/best.pt` | SHA-256 `192c827b…`（完整值见 `reports/stage12-summary.json`） |
| 结构与 processor | 同 run 的 `model_config.json`、`processor/` | 与 `code_identity_start.json` 中的摘要一致 |
| PCB1 数据 | `data/visa/pcb1/Data/{Images,Masks}` | 清单 SHA：train/val/test 见实验卡 |
| Conda 环境 | `env/`（Python 3.12.16 + torch 2.10.0+cu126） | `requirements-lock.txt` |

**重新下载 `nvidia/mit-b0` 不能替代 best**：快照只是初始化起点，训练后的权重只存在于 best.pt。

## 2. 环境与上下文

```powershell
$env:PSExecutionPolicyPreference = 'Bypass'
Set-Location -LiteralPath '<repo>\\work\\SegFormer'
. '.\\scripts\\enter_project.ps1'
$ErrorActionPreference = 'Continue'
```

## 3. 复算 val（应先与 `reports/stage12_evidence/val_metrics.json` 对齐）

```powershell
Invoke-SegPython -u -m segformer_lab.evaluate --config 'configs\\baseline.yaml' `
  --run '<...>\\runs\\pcb1_b0_v2_seed42' --split val --out '<new-dir>\\val'
```

门禁 `gate.json` 应显示 `passed=true`；任何一项不一致先停下定位（eval 模式、归一化、还原顺序、阈值）。

## 4. 复现图表

```powershell
Invoke-SegPython -u -m segformer_lab.summarize --config 'configs\\baseline.yaml' `
  --run '<...>\\runs\\pcb1_b0_v2_seed42' `
  --eval-run '<...>\\runs\\pcb1_b0_v2_seed42_eval_001' `
  --out '<repo>\\work\\SegFormer\\reports\\pcb1_b0_v2_seed42'
```

`summarize` 不训练、不选 best；它只读 `epochs.jsonl` 与阶段 12 的评价目录，并用同一冻结通道为展示图重算概率（二值结果必须与阶段 12 的 PNG 完全一致，否则脚本报错退出）。

## 5. 约束

- test 已经评价过一次；再跑属于技术复核，需保留原因与全部结果，不得按分数择优。
- 阈值 0.5、B1、IEEE FP32、无 TTA / 后处理，不得为提升分数而修改。
"""
    (out_dir / "reproduction.md").write_text(repro, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stage 13: figures, failure cases, experiment card")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--eval-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return do_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
