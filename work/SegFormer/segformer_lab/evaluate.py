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
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import torch
from PIL import Image

from . import data, paths, pilot
from .config import load_config
from .inference import (FrozenRun, build_dataset, guard_output_directory, load_frozen_run,
                        save_binary_mask, stream_split)
from .training import sha256_file

SCHEMA = "segformer-lab/eval-metrics/1"
GATE_SCHEMA = "segformer-lab/eval-gate/2"

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
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
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
    files = sorted((paths.PROJECT_ROOT / "segformer_lab").glob("*.py"))
    files += sorted((paths.PROJECT_ROOT / "scripts").glob("*.ps1"))
    return {path.relative_to(paths.PROJECT_ROOT).as_posix(): sha256_file(path) for path in files}


def _runtime_info(device: str) -> dict[str, Any]:
    from importlib.metadata import version

    info = {"executable": sys.executable, "python": sys.version,
            "platform": platform.platform(), "device": str(torch.device(device)),
            "versions": {name: version(name) for name in
                         ("torch", "transformers", "numpy", "Pillow", "matplotlib", "PyYAML")},
            "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if torch.device(device).type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(torch.device(device))
    return info


def _data_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Bind original image AND mask bytes using the frozen extraction inventory.

    Reading metadata here never opens test image/GT files. Their actual bytes
    are checked only after a valid gate has been established.
    """
    meta_path = Path(config["paths"]["data.split_meta"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    root = Path(config["data_root"]).resolve()
    inventory_path = root / "extraction_manifest.json"
    if sha256_file(inventory_path) != meta["extraction"]["manifest_sha256"]:
        raise SystemExit("data extraction inventory differs from frozen split metadata")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    indexed = {item["path"]: item for item in inventory["files"]}
    if len(indexed) != len(inventory["files"]):
        raise SystemExit("duplicate paths in data extraction inventory")
    official = Path(meta["sources"]["official_csv"]["copy"])
    official_sha = sha256_file(official)
    if official_sha != meta["sources"]["official_csv"]["sha256"]:
        raise SystemExit("official split CSV differs from frozen split metadata")
    splits = {}
    for split in ("val", "test"):
        rows = data.read_manifest(config["paths"][f"data.{split}_manifest"])
        ids = [row["image"] for row in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"duplicate image IDs in {split} manifest")
        records = []
        for row in rows:
            if row["label"] not in {"normal", "anomaly"} or bool(row["mask"].strip()) != (row["label"] == "anomaly"):
                raise SystemExit(f"invalid label/mask contract: {row['image']}")
            for kind in ("image", "mask"):
                name = row[kind].strip()
                if not name:
                    continue
                actual = (root / name).resolve()
                if not actual.is_relative_to(root) or Path(name).is_absolute() or ".." in Path(name).parts:
                    raise SystemExit(f"data path escapes source root: {name}")
                expected = indexed.get(name)
                if expected is None:
                    raise SystemExit(f"data path missing from extraction inventory: {name}")
                records.append({"image_id": row["image"], "label": row["label"], "kind": kind,
                                "path": name, "bytes": int(expected["size"]), "sha256": expected["sha256"]})
        splits[split] = records
    return {"data_root": str(root), "extraction_inventory": {
                "path": str(inventory_path), "sha256": sha256_file(inventory_path)},
            "official_csv": {"path": str(official), "sha256": official_sha}, "splits": splits}


def _verify_data_files(identity: dict[str, Any], split: str) -> None:
    for record in identity["splits"][split]:
        path = Path(identity["data_root"]) / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise SystemExit(f"{split} {record['kind']} bytes differ from frozen inventory: {record['path']}")


def _protocol_identity(config: dict[str, Any]) -> dict[str, Any]:
    requested = config["raw"]["eval"]
    if requested.get("precision") != "ieee_fp32" or requested.get("batch_size") != 1 or \
            requested.get("threshold") != 0.5 or requested.get("postprocess") != "none" or \
            config["raw"]["data"].get("workers") != 0:
        raise SystemExit("evaluation requires frozen IEEE FP32/B1/threshold0.5/no-postprocess protocol")
    return {"batch_size": 1, "workers": 0, "precision": "ieee_fp32", "autocast": False,
            "cudnn_allow_tf32": False, "matmul_allow_tf32": False, "threshold": 0.5,
            "flip": "none", "postprocess": "none", "tta": False,
            "restore": "logits->bilinear512(align_corners=False)->crop->bilinear_original->softmax1",
            "original_gt": "nonzero merged to 1; normal zero mask", "tolerance": GATE_TOLERANCE}


def _evaluation_binding(frozen: FrozenRun, config: dict[str, Any]) -> dict[str, Any]:
    return {"identity": frozen.identity, "code": _code_identity(),
            "data": _data_identity(config), "protocol": _protocol_identity(config),
            "runtime": _runtime_info(frozen.device)}


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
                       "abs_diff": diff, "ok": math.isfinite(diff) and diff <= tol})

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
    exact("per_image.row_count", len(fresh["per_image"]), len(saved["per_image"]))
    exact("per_image.unique_ids", len(fresh_rows) == len(fresh["per_image"])
          and len(saved_rows) == len(saved["per_image"]), True)
    exact("per_image.ids", sorted(fresh_rows), sorted(saved_rows))
    integer_fields = ("total_pixels", "gt_foreground_pixels", "predicted_foreground_pixels",
                      "tp", "fp", "fn", "tn")
    float_fields = ("iou", "dice", "precision", "recall", "predicted_foreground_ratio")
    na_fields = ("iou_na", "dice_na", "precision_na", "recall_na")
    mismatched = []
    for image_id in sorted(set(fresh_rows) & set(saved_rows)):
        left, right = fresh_rows[image_id], saved_rows[image_id]
        for field in ("label", "split", "ignored_pixels"):
            if left.get(field) != right.get(field):
                mismatched.append({"image_id": image_id, "field": field,
                                   "fresh": left.get(field), "saved": right.get(field)})
        # Older epoch records lack dimensions; frozen original-file hashes bind
        # those dimensions. Compare them directly whenever saved by the producer.
        for field in ("height", "width"):
            if field in right and left.get(field) != right[field]:
                mismatched.append({"image_id": image_id, "field": field,
                                   "fresh": left.get(field), "saved": right[field]})
        for field in integer_fields:
            if int(left[field]) != int(right[field]):
                mismatched.append(f"{image_id}:{field}")
        for field in float_fields:
            a, b = left.get(field), right.get(field)
            if (a is None) != (b is None) or (a is not None and
                    (not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(float(a) - float(b)) > 1e-9)):
                mismatched.append(f"{image_id}:{field}")
        for field in na_fields:
            if left.get(field) != right.get(field):
                mismatched.append(f"{image_id}:{field}")
    checks.append({"name": "per_image.fields", "kind": "integer/na/float", "ok": not mismatched,
                   "mismatched": mismatched, "mismatched_count": len(mismatched)})

    for metric in ("iou", "dice", "precision", "recall"):
        exact(f"macro.{metric}.defined_images",
              fresh["anomaly_images"]["macro"][metric]["defined_images"],
              saved["anomaly_images"]["macro"][metric]["defined_images"])
        close(f"macro.{metric}.mean",
              fresh["anomaly_images"]["macro"][metric]["mean"],
              saved["anomaly_images"]["macro"][metric]["mean"])
        for field in ("na_images", "na_reason"):
            exact(f"macro.{metric}.{field}", fresh["anomaly_images"]["macro"][metric].get(field),
                  saved["anomaly_images"]["macro"][metric].get(field))
    exact("anomaly.count", fresh["anomaly_images"]["count"], saved["anomaly_images"]["count"])
    exact("normal.count", fresh["normal_images"]["count"], saved["normal_images"]["count"])
    exact("normal.false_positive_pixels", int(fresh["normal_images"]["false_positive_pixels"]),
          int(saved["normal_images"]["false_positive_pixels"]))
    close("normal.fp_pixel_ratio", fresh["normal_images"]["fp_pixel_ratio"],
          saved["normal_images"]["fp_pixel_ratio"])
    exact("normal.images_with_any_false_positive",
          int(fresh["normal_images"]["images_with_any_false_positive"]),
          int(saved["normal_images"]["images_with_any_false_positive"]))

    def compare_tree(name: str, left: Any, right: Any) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            exact(name + ".keys", sorted(left), sorted(right))
            for key in sorted(set(left) & set(right)):
                compare_tree(name + "." + key, left[key], right[key])
        elif isinstance(left, list) and isinstance(right, list):
            exact(name + ".count", len(left), len(right))
            for index, (a, b) in enumerate(zip(left, right)):
                compare_tree(f"{name}[{index}]", a, b)
        elif isinstance(left, float) or isinstance(right, float):
            close(name, left, right)
        else:
            exact(name, left, right)

    compare_tree("normal.all_fields", fresh["normal_images"], saved["normal_images"])

    failed = [check["name"] for check in checks if not check["ok"]]
    return {"schema": GATE_SCHEMA, "tolerance": GATE_TOLERANCE, "checks": checks,
            "failed": failed, "passed": not failed}


def _verify_val_gate(gate_path: Path, frozen: FrozenRun, config: dict[str, Any]) -> dict[str, Any]:
    if not gate_path.is_file():
        raise SystemExit(f"val gate not found: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema") != GATE_SCHEMA:
        raise SystemExit("val gate has an unexpected schema")
    if gate.get("passed") is not True or gate.get("failed") != []:
        raise SystemExit(f"val gate did not pass: {gate.get('failed')}")
    checks = gate.get("checks")
    required = {"images.total", "images.normal", "images.anomaly", "per_image.fields",
                "per_image.ids", "per_image.unique_ids", "preview.probability", "coverage"}
    if not isinstance(checks, list) or not checks or any(check.get("ok") is not True for check in checks) or \
            not required <= {check.get("name") for check in checks}:
        raise SystemExit("val gate is missing complete passing validation/preview/coverage checks")
    count_checks = {check["name"]: check for check in checks if check.get("name") in
                    {"images.total", "images.normal", "images.anomaly"}}
    if any(count_checks[f"images.{name}"].get("left") != expected or
           count_checks[f"images.{name}"].get("right") != expected
           for name, expected in zip(("total", "normal", "anomaly"), EXPECTED["val"])):
        raise SystemExit("val gate does not cover the complete 132-image validation split")
    if gate.get("identity") != frozen.identity or gate.get("threshold") != 0.5:
        raise SystemExit("val gate frozen input identity/threshold differs")
    if "binding" not in gate or gate["binding"] != _evaluation_binding(frozen, config):
        raise SystemExit("val gate code/processor/data/runtime/protocol binding differs")
    _verify_data_files(gate["binding"]["data"], "val")
    return gate


def do_evaluate(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    code_before = _code_identity()
    config = load_config(args.config)
    _protocol_identity(config)
    threshold = float(config["raw"]["eval"]["threshold"])
    split = args.split
    if split not in EXPECTED:
        raise SystemExit("--split must be val or test")

    out_dir = guard_output_directory(args.out, args.run, config)

    frozen = load_frozen_run(config, args.run, device=args.device)
    gate = None
    if split == "test":
        if not args.val_gate:
            raise SystemExit("test evaluation requires --val-gate from a passing val run")
        gate = _verify_val_gate(Path(args.val_gate).resolve(), frozen, config)
    binding = _evaluation_binding(frozen, config)
    if binding["code"] != code_before:
        raise SystemExit("source code changed during initialization")
    # For test this is deliberately AFTER the gate, and before opening a sample.
    _verify_data_files(binding["data"], split)
    dataset = build_dataset(config, split, processor=frozen.processor)
    expected_total, expected_normal, expected_anomaly = EXPECTED[split]
    labels = [row["label"] for row in dataset.rows]
    observed = (len(dataset), labels.count("normal"), labels.count("anomaly"))
    if observed != (expected_total, expected_normal, expected_anomaly):
        raise SystemExit(f"{split} split is {observed}, protocol expects "
                         f"{(expected_total, expected_normal, expected_anomaly)}")

    ids = [row["image"] for row in dataset.rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate image IDs in evaluation split")
    out_dir.mkdir(parents=True, exist_ok=False)

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
    if split == "val":
        fixed_path = Path(config["paths"]["logging.fixed_preview_manifest"])
        fixed_ids = [line.strip() for line in fixed_path.read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
        if len(fixed_ids) != 6 or len(set(fixed_ids)) != 6 or not set(fixed_ids) <= set(ids):
            raise SystemExit("validation requires all six fixed preview IDs")
        keep_ids = set(fixed_ids)
        preview_keys = {name.replace("/", "_") for name in fixed_ids}
        with np.load(preview_path, allow_pickle=False) as stored:
            if len(stored.files) != 6 or set(stored.files) != preview_keys:
                raise SystemExit("saved preview probabilities do not match the six fixed IDs")

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
    if missing or duplicates or len(per_image) != expected_total or [row["image_id"] for row in per_image] != ids:
        raise SystemExit("evaluation coverage/order differs from the complete split manifest")

    # Freeze evidence over the whole operation, including original source bytes.
    _verify_data_files(binding["data"], split)
    code_after = _code_identity()
    if code_after != code_before or _evaluation_binding(frozen, config) != binding or \
            _manifest_of(frozen.run_dir, exclude=set())["files"] != frozen.identity["source_files"]:
        raise SystemExit("code, metadata or original training run changed during evaluation")

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
        with np.load(preview_path, allow_pickle=False) as stored:
            preview_block = pilot.compare_probabilities(stored, kept, threshold=threshold,
                                                        atol=GATE_TOLERANCE["preview_probability_atol"])
        preview_block.update({"images": len(kept), "xor_pixels": preview_block["xor_pixels_total"]})
        comparison["checks"].extend([
            {"name": "preview.probability", "kind": "float", "ok": preview_block["ok"], "detail": preview_block},
            {"name": "coverage", "kind": "integer", "ok": not missing and duplicates == 0 and len(per_image) == 132}])
        comparison["checks"].append({"name": "source.stable", "kind": "identity",
                                      "ok": _code_identity() == code_before and
                                      _manifest_of(frozen.run_dir, exclude=set())["files"] == frozen.identity["source_files"]})
        comparison["failed"] = [check["name"] for check in comparison["checks"] if not check["ok"]]
        comparison["passed"] = not comparison["failed"]
        comparison.update({"identity": frozen.identity, "threshold": threshold, "binding": binding,
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
        "process": {"pid": os.getpid(), **_runtime_info(args.device)},
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
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
        "input_data": binding["data"],
        "protocol": binding["protocol"],
        "technical_rerun_reason": getattr(args, "rerun_reason", None),
        "val_gate": gate_block,
        "val_gate_input": str(Path(args.val_gate).resolve()) if args.val_gate else None,
        "code": code_before,
        "code_after": code_after,
        "code_and_original_run_unchanged": True,
        "val_gate_input_sha256": sha256_file(Path(args.val_gate)) if args.val_gate else None,
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
    parser.add_argument("--rerun-reason", default=None, help="reason for a required technical re-verification")
    args = parser.parse_args(argv)
    return do_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
