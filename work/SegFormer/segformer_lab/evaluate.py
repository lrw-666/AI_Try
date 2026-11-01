"""Stage 12: independent evaluation of a frozen run.

Two jobs, one entry point:

* ``--split val``  recompute all 132 held-out images and prove the frozen best
  reproduces the training run's stored epoch-39 numbers, field by field, then
  write ``gate.json``;
* ``--split test`` evaluate all 442 official test images exactly once under the
  frozen protocol, but only if a passing ``--val-gate`` is supplied and its
  identity (checkpoint, config, manifests, threshold, model structure) matches.

Protocol is fixed and not negotiated here: B1, ``model.eval()`` +
``inference_mode``, IEEE FP32 with TF32/autocast disabled, original-size GT,
threshold ``>= 0.5``, no augmentation, no post-processing, no TTA. Probabilities
are never accumulated in memory for a whole split; each image is scored, written
as a 0/1 PNG and released.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import torch
from PIL import Image

from . import paths
from .config import load_config
from .inference import FrozenRun, build_dataset, load_frozen_run, save_binary_mask, stream_split
from .training import sha256_file

SCHEMA = "segformer-lab/eval-metrics/1"
GATE_SCHEMA = "segformer-lab/eval-gate/1"

EXPECTED = {"val": (132, 120, 12), "test": (442, 402, 40)}

# Fixed before any test image is read; stage 13 consumes exactly this rule.
DISPLAY_RULES = {
    "fixed_normal": {"k": 6, "rule": "test split, normal rows sorted by image_id ascending, first 6"},
    "fixed_anomaly": {"k": 6, "rule": "test split, anomaly rows sorted by image_id ascending, first 6"},
    "worst_iou": {"k": 5, "rule": "anomaly rows sorted by iou ascending (null treated as -1), ties by image_id"},
    "worst_recall": {"k": 5, "rule": "anomaly rows sorted by recall ascending (null treated as -1), ties by image_id"},
    "most_normal_fp": {"k": 5, "rule": "normal rows sorted by false-positive pixels descending, ties by image_id"},
    "notes": "selection lists may overlap; every row keeps its own reason. Rules were written before "
             "the first test image was decoded and are hashed into evaluation_meta.json.",
}

# Declared before the comparison; the same values stage 11 used where relevant.
GATE_TOLERANCE = {
    "integers": "exact",
    "na_strings": "exact",
    "floats": "abs diff <= 1e-9",
    "preview_probability_atol": 1e-5,
    "preview_xor_pixels": 0,
}


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _git_state() -> dict[str, Any]:
    import subprocess

    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=str(paths.PROJECT_ROOT),
                                  capture_output=True, text=True, timeout=30).stdout
        except Exception as exc:  # pragma: no cover
            return f"<error: {exc}>"

    return {"head": run("rev-parse", "HEAD").strip(),
            "status_porcelain": run("status", "--porcelain").strip(),
            "diff_sha256": sha256_file_text(run("diff"))}


def sha256_file_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _code_identity() -> dict[str, str]:
    files = [
        "segformer_lab/inference.py", "segformer_lab/evaluate.py", "segformer_lab/predict.py",
        "segformer_lab/training.py", "segformer_lab/data.py", "segformer_lab/geometry.py",
        "segformer_lab/metrics.py", "segformer_lab/evaluation.py", "segformer_lab/modeling.py",
        "segformer_lab/config.py", "segformer_lab/losses.py",
    ]
    out: dict[str, str] = {}
    for name in files:
        path = paths.PROJECT_ROOT / name
        out[name] = sha256_file(path) if path.is_file() else ""
    return out


def _manifest_of(root: Path, exclude: set[str]) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in exclude:
            continue
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"root": str(root.resolve()), "files": files, "file_count": len(files),
            "total_bytes": sum(item["bytes"] for item in files.values())}


def _precision_text(result: dict[str, Any]) -> dict[str, Any]:
    return result["evaluation"]["precision_record"]


def display_selection(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    """Apply DISPLAY_RULES to finished rows (rules themselves were frozen earlier)."""
    normal = [row for row in rows if row["label"] == "normal"]
    anomaly = [row for row in rows if row["label"] == "anomaly"]
    normal_sorted = sorted(normal, key=lambda r: r["image_id"])
    anomaly_sorted = sorted(anomaly, key=lambda r: r["image_id"])

    def value(row: dict[str, Any], field: str) -> float:
        value = row.get(field)
        return -1.0 if value is None else float(value)

    worst_iou = sorted(anomaly, key=lambda r: (value(r, "iou"), r["image_id"]))[:5]
    worst_recall = sorted(anomaly, key=lambda r: (value(r, "recall"), r["image_id"]))[:5]
    most_fp = sorted(normal, key=lambda r: (-int(r["fp"]), r["image_id"]))[:5]

    def pick(rows_in: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
        return [{"image_id": row["image_id"], "label": row["label"], "reason": reason,
                 "iou": row.get("iou"), "recall": row.get("recall"), "fp": row["fp"]}
                for row in rows_in]

    return {
        "split": split,
        "fixed_normal": pick(normal_sorted[:6], "fixed_normal"),
        "fixed_anomaly": pick(anomaly_sorted[:6], "fixed_anomaly"),
        "worst_iou": pick(worst_iou, "worst_iou"),
        "worst_recall": pick(worst_recall, "worst_recall"),
        "most_normal_fp": pick(most_fp, "most_normal_fp"),
        "all_anomaly_tp_zero": sorted(row["image_id"] for row in anomaly if int(row["tp"]) == 0),
        "all_anomaly_empty_prediction": sorted(row["image_id"] for row in anomaly
                                               if int(row["predicted_foreground_pixels"]) == 0),
        "all_normal_with_false_positive": sorted(row["image_id"] for row in normal if int(row["fp"]) > 0),
        "counts": {"normal": len(normal), "anomaly": len(anomaly)},
        "distinct_tp_zero_and_empty": True,
    }


def _gate_compare(fresh: dict[str, Any], saved: dict[str, Any]) -> dict[str, Any]:
    """Compare a recomputed val result with the run's stored epoch record."""
    checks: list[dict[str, Any]] = []

    def exact(name: str, left: Any, right: Any) -> None:
        checks.append({"name": name, "kind": "integer/na", "left": left, "right": right,
                       "ok": left == right})

    def close(name: str, left: Any, right: Any, tol: float = 1e-9) -> None:
        if left is None or right is None:
            checks.append({"name": name, "kind": "float", "left": left, "right": right,
                           "ok": left is None and right is None})
            return
        diff = abs(float(left) - float(right))
        checks.append({"name": name, "kind": "float", "left": float(left), "right": float(right),
                       "abs_diff": diff, "ok": diff <= tol})

    exact("images.total", fresh["images"]["total"], saved["images"]["total"])
    exact("images.normal", fresh["images"]["normal"], saved["images"]["normal"])
    exact("images.anomaly", fresh["images"]["anomaly"], saved["images"]["anomaly"])
    for key in ("tp", "fp", "fn", "tn", "pixels"):
        exact(f"confusion.{key}", int(fresh["confusion"][key]), int(saved["confusion"][key]))
    for key in ("foreground_iou", "foreground_dice", "precision", "recall"):
        close(f"global.{key}", fresh["global"].get(key), saved["global"].get(key))
    exact("global.na_reasons", fresh["global"].get("na_reasons", {}),
          saved["global"].get("na_reasons", {}))

    fresh_rows = {row["image_id"]: row for row in fresh["per_image"]}
    saved_rows = {row["image_id"]: row for row in saved["per_image"]}
    exact("per_image.ids", sorted(fresh_rows), sorted(saved_rows))
    integer_fields = ("total_pixels", "gt_foreground_pixels", "predicted_foreground_pixels",
                      "tp", "fp", "fn", "tn")
    float_fields = ("iou", "dice", "precision", "recall", "predicted_foreground_ratio")
    na_fields = ("iou_na", "dice_na", "precision_na", "recall_na")
    mismatched = []
    for image_id in sorted(set(fresh_rows) & set(saved_rows)):
        left, right = fresh_rows[image_id], saved_rows[image_id]
        for field in integer_fields:
            if int(left[field]) != int(right[field]):
                mismatched.append(f"{image_id}:{field}")
        for field in float_fields:
            a, b = left.get(field), right.get(field)
            if (a is None) != (b is None) or (a is not None and abs(float(a) - float(b)) > 1e-9):
                mismatched.append(f"{image_id}:{field}")
        for field in na_fields:
            if left.get(field) != right.get(field):
                mismatched.append(f"{image_id}:{field}")
    checks.append({"name": "per_image.fields", "kind": "integer/na/float", "ok": not mismatched,
                   "mismatched": mismatched[:20], "mismatched_count": len(mismatched)})

    for metric in ("iou", "dice", "precision", "recall"):
        exact(f"macro.{metric}.defined_images",
              fresh["anomaly_images"]["macro"][metric]["defined_images"],
              saved["anomaly_images"]["macro"][metric]["defined_images"])
        close(f"macro.{metric}.mean",
              fresh["anomaly_images"]["macro"][metric]["mean"],
              saved["anomaly_images"]["macro"][metric]["mean"])
    exact("anomaly.count", fresh["anomaly_images"]["count"], saved["anomaly_images"]["count"])
    exact("normal.count", fresh["normal_images"]["count"], saved["normal_images"]["count"])
    exact("normal.false_positive_pixels", int(fresh["normal_images"]["false_positive_pixels"]),
          int(saved["normal_images"]["false_positive_pixels"]))
    close("normal.fp_pixel_ratio", fresh["normal_images"]["fp_pixel_ratio"],
          saved["normal_images"]["fp_pixel_ratio"])
    exact("normal.images_with_any_false_positive",
          int(fresh["normal_images"]["images_with_any_false_positive"]),
          int(saved["normal_images"]["images_with_any_false_positive"]))

    failed = [check["name"] for check in checks if not check["ok"]]
    return {"schema": GATE_SCHEMA, "tolerance": GATE_TOLERANCE, "checks": checks,
            "failed": failed, "passed": not failed}


def _verify_val_gate(gate_path: Path, frozen: FrozenRun, config: dict[str, Any]) -> dict[str, Any]:
    if not gate_path.is_file():
        raise SystemExit(f"val gate not found: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema") != GATE_SCHEMA:
        raise SystemExit("val gate has an unexpected schema")
    if not gate.get("passed"):
        raise SystemExit(f"val gate did not pass: {gate.get('failed')}")
    for key in ("best_sha256", "config_sha256"):
        if gate["identity"][key] != frozen.identity[key]:
            raise SystemExit(f"val gate {key} does not match the frozen run")
    for split in ("train", "val", "test"):
        if gate["identity"]["manifests"][split]["sha256"] != \
                frozen.identity["manifests"][split]["sha256"]:
            raise SystemExit(f"val gate {split} manifest does not match the frozen run")
    if gate["identity"]["split_meta_sha256"] != frozen.identity["split_meta_sha256"]:
        raise SystemExit("val gate split_meta does not match the frozen run")
    if abs(float(gate["threshold"]) - float(config["raw"]["eval"]["threshold"])) > 0:
        raise SystemExit("val gate threshold differs from the configured threshold")
    if gate["identity"]["model_config_sha256"] != frozen.identity["model_config_sha256"]:
        raise SystemExit("val gate model structure does not match the frozen run")
    return gate


def do_evaluate(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = load_config(args.config)
    threshold = float(config["raw"]["eval"]["threshold"])
    split = args.split
    if split not in EXPECTED:
        raise SystemExit("--split must be val or test")

    out_dir = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite an existing evaluation directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen = load_frozen_run(config, args.run, device=args.device)
    dataset = build_dataset(config, split, processor=frozen.processor)
    expected_total, expected_normal, expected_anomaly = EXPECTED[split]
    labels = [row["label"] for row in dataset.rows]
    observed = (len(dataset), labels.count("normal"), labels.count("anomaly"))
    if observed != (expected_total, expected_normal, expected_anomaly):
        raise SystemExit(f"{split} split is {observed}, protocol expects "
                         f"{(expected_total, expected_normal, expected_anomaly)}")

    gate = None
    if split == "test":
        if not args.val_gate:
            raise SystemExit("test evaluation requires --val-gate from a passing val run")
        gate = _verify_val_gate(Path(args.val_gate).resolve(), frozen, config)

    rules_path = out_dir / "display_rules.json"
    _json_dump(rules_path, {"split": split, "rules": DISPLAY_RULES,
                            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "written_before_inference": True})

    mask_root = out_dir / "predictions"
    display_root = out_dir / "predictions_display" if args.display_masks else None
    rows: list[dict[str, Any]] = []
    write_seconds = 0.0

    def on_sample(index: int, item: dict[str, Any], probability: np.ndarray) -> None:
        nonlocal write_seconds
        tick = time.perf_counter()
        mask = (probability >= threshold).astype(np.uint8)
        relative = Path(item["image_id"]).with_suffix(".png")
        save_binary_mask(mask_root / relative, mask)
        if display_root is not None:
            display = display_root / relative
            display.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(display, format="PNG")
        rows.append({"image_id": item["image_id"], "prediction_png": relative.as_posix(),
                     "height": int(mask.shape[0]), "width": int(mask.shape[1])})
        write_seconds += time.perf_counter() - tick

    keep_ids = set()
    preview_path = frozen.run_dir / "pre_save_probabilities.npz"
    preview_keys: set[str] = set()
    if split == "val" and preview_path.is_file():
        preview_keys = set(np.load(preview_path).files)
        keep_ids = {row["image"] for row in dataset.rows
                    if row["image"].replace("/", "_") in preview_keys}

    result, kept = stream_split(frozen, dataset, threshold=threshold, on_sample=on_sample,
                                keep_ids=keep_ids)

    prediction_index = {row["image_id"]: row for row in rows}
    per_image = []
    for row in result["per_image"]:
        extra = prediction_index[row["image_id"]]
        per_image.append({**row, "prediction_png": extra["prediction_png"],
                          "height": extra["height"], "width": extra["width"]})
    seen = {row["image_id"] for row in per_image}
    missing = sorted({row["image"] for row in dataset.rows} - seen)
    duplicates = len(per_image) - len(seen)

    metrics = {
        "schema": SCHEMA,
        "stage": "12",
        "split": split,
        "run_dir": str(frozen.run_dir),
        "best_epoch": frozen.identity["best_epoch"],
        "best_global_step": frozen.identity["best_global_step"],
        "images": result["images"],
        "confusion": result["confusion"],
        "global": result["global"],
        "anomaly_images": result["anomaly_images"],
        "normal_images": result["normal_images"],
        "evaluation": result["evaluation"],
        "coverage": {"rows": len(per_image), "missing": missing, "duplicates": duplicates},
        "note": "integer counts are exact; NA means the denominator was zero, no epsilon is added; "
                "normal-image statistics are pixel-level, not an image-level detection rate",
    }
    _json_dump(out_dir / "metrics.json", metrics)
    _json_dump(out_dir / "per_image.json", {"split": split, "rows": per_image})

    csv_fields = ["image_id", "label", "split", "height", "width", "total_pixels",
                  "gt_foreground_pixels", "predicted_foreground_pixels",
                  "predicted_foreground_ratio", "tp", "fp", "fn", "tn",
                  "iou", "dice", "precision", "recall",
                  "iou_na", "dice_na", "precision_na", "recall_na", "prediction_png"]
    with (out_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in per_image:
            writer.writerow(row)

    selection = display_selection(per_image, split)
    _json_dump(out_dir / "selection.json", selection)

    gate_block = None
    preview_block = None
    exit_code = 0
    if split == "val":
        saved_entry = next(entry for entry in
                           json.loads((frozen.run_dir / "summary.json").read_text(encoding="utf-8"))["epochs"]
                           if entry["epoch"] == frozen.identity["best_epoch"])
        comparison = _gate_compare(
            {"images": result["images"], "confusion": result["confusion"], "global": result["global"],
             "per_image": per_image, "anomaly_images": result["anomaly_images"],
             "normal_images": result["normal_images"]},
            saved_entry["val"])
        if preview_keys:
            stored = np.load(preview_path)
            diffs = []
            xor_total = 0
            for key in sorted(preview_keys):
                fresh = kept[key]
                old = stored[key]
                diffs.append(float(np.abs(old - fresh).max()))
                xor_total += int(((old >= threshold) != (fresh >= threshold)).sum())
            preview_block = {
                "images": len(preview_keys),
                "max_abs_probability_diff": max(diffs) if diffs else None,
                "xor_pixels": xor_total,
                "total_pixels": int(sum(stored[key].size for key in preview_keys)),
                "atol": GATE_TOLERANCE["preview_probability_atol"],
                "within_tolerance": bool(diffs) and max(diffs) <= GATE_TOLERANCE["preview_probability_atol"],
                "xor_ok": xor_total == 0,
            }
            comparison["checks"].append({"name": "preview.probability", "kind": "float",
                                         "ok": preview_block["within_tolerance"] and preview_block["xor_ok"],
                                         "detail": preview_block})
        comparison["failed"] = [check["name"] for check in comparison["checks"] if not check["ok"]]
        comparison["passed"] = not comparison["failed"]
        comparison.update({"identity": frozen.identity, "threshold": threshold,
                           "saved_record": {"epoch": saved_entry["epoch"],
                                            "source": "run/summary.json epochs[best]"},
                           "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        _json_dump(out_dir / "gate.json", comparison)
        gate_block = {"passed": comparison["passed"], "failed": comparison["failed"],
                      "path": str(out_dir / "gate.json")}
        exit_code = 0 if comparison["passed"] else 1

    meta = {
        "schema": "segformer-lab/eval-meta/1",
        "stage": "12",
        "command": {"argv": sys.argv, "module": "segformer_lab.evaluate"},
        "process": {"pid": os.getpid(), "python": sys.version.split()[0],
                    "platform": platform.platform()},
        "versions": {"torch": torch.__version__, "cuda_build": torch.version.cuda,
                     "numpy": np.__version__},
        "device": args.device,
        "split": split,
        "threshold": threshold,
        "precision": _precision_text(result),
        "restore_order": "logits -> bilinear 512 (align_corners=False) -> crop padding -> "
                         "bilinear original H/W -> channel softmax -> threshold >= 0.5",
        "gt_definition": "original-size binary mask; normal images get an all-zero mask; "
                         "training-only padding 255 never appears here",
        "display_rules_sha256": sha256_file(rules_path),
        "identity": frozen.identity,
        "val_gate": gate_block,
        "val_gate_input": str(Path(args.val_gate).resolve()) if args.val_gate else None,
        "code": _code_identity(),
        "git": _git_state(),
        "timings": {"elapsed_seconds": round(time.perf_counter() - started, 3),
                    "png_write_seconds": round(write_seconds, 3)},
        "outputs": {"predictions": "predictions/ (uint8 0/1, original size)",
                    "display_masks": "predictions_display/ (0/255 display only)" if display_root else None},
    }
    _json_dump(out_dir / "evaluation_meta.json", meta)
    manifest = _manifest_of(out_dir, exclude={"artifact_manifest.json"})
    _json_dump(out_dir / "artifact_manifest.json", manifest)

    print(f"  split   : {split}, {len(per_image)} images "
          f"({result['images']['normal']}N+{result['images']['anomaly']}A)")
    print(f"  global  : IoU {result['global']['foreground_iou']} Dice "
          f"{result['global']['foreground_dice']} P {result['global']['precision']} "
          f"R {result['global']['recall']}")
    print(f"  macro   : IoU {result['anomaly_images']['macro']['iou']['mean']} "
          f"(defined {result['anomaly_images']['macro']['iou']['defined_images']})")
    print(f"  normal  : FP {result['normal_images']['false_positive_pixels']} px, "
          f"{result['normal_images']['images_with_any_false_positive']}/"
          f"{result['normal_images']['count']} images")
    if gate_block is not None:
        print(f"  val gate: passed={gate_block['passed']} failed={gate_block['failed']}")
    if preview_block is not None:
        print(f"  preview : max|dprob|={preview_block['max_abs_probability_diff']} "
              f"XOR={preview_block['xor_pixels']}")
    print(f"  wrote   : {out_dir}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stage 12: frozen-run evaluation (val gate, test)")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--run", required=True, help="trained run directory holding best.pt")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-gate", default="", help="gate.json from a passing val evaluation")
    parser.add_argument("--display-masks", action="store_true",
                        help="also write a 0/255 display copy (never used for scoring)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return do_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
