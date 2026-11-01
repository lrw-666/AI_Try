"""Freeze the 8-image training subset shared by stage 08 and stage 09.

Why a separate command instead of an inline ``np.random.choice`` in the trainer:
both stages must train on *exactly* the same eight images, and a future reader
must be able to recompute the choice without re-running any GPU code. The IDs
file is the single source of truth; this module only writes and verifies it.

Selection rule (deterministic, no val/test involved):

* anomaly: sort the 48 train anomalies by original foreground pixels (ties by
  image path) and take ranks 0 %, 33 %, 66 %, 100 % -- four very different
  defect sizes, from a 0.08 % scratch to a 12.7 % area;
* normal: ``random.Random(seed).sample`` over the sorted train-normal list.

The written file is what ``data.PCB1Dataset`` consumes through
``data.train_ids_file``; nothing downstream re-samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import geometry, paths
from .config import load_config

SCHEMA = "segformer-lab/overfit-subset/1"
ID_FILE = "manifests/overfit_train_ids.txt"
DETAIL_FILE = "reports/overfit_subset.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pick_anomalies(rows: list[dict[str, str]], audit: dict[str, dict[str, str]], count: int = 4) -> list[dict[str, str]]:
    """Ranks spread over the foreground-size distribution, inclusive of both ends."""
    scored = []
    for row in rows:
        record = audit.get(row["image"])
        if record is None:
            raise SystemExit(f"audit row missing for {row['image']}; rerun stage 04 first")
        scored.append((int(record["fg_px"]), row["image"], row))
    scored.sort()
    if len(scored) < count:
        raise SystemExit("not enough anomalies in train")
    # 0 %, 33 %, 66 %, 100 % of the sorted list, de-duplicated
    ranks = sorted({round(i * (len(scored) - 1) / (count - 1)) for i in range(count)})
    return [scored[rank][2] for rank in ranks]


def describe(config: dict[str, Any], row: dict[str, str], audit: dict[str, dict[str, str]]) -> dict[str, Any]:
    data_root: Path = config["data_root"]
    image_path = data_root / row["image"]
    record = audit[row["image"]]
    entry: dict[str, Any] = {
        "image": row["image"],
        "split": row["split"],
        "label": row["label"],
        "image_sha256": sha256_file(image_path),
        "image_size_bytes": image_path.stat().st_size,
        "width": int(record["width"]),
        "height": int(record["height"]),
        "original_foreground_pixels": int(record["fg_px"]),
        "original_foreground_ratio": float(record["fg_ratio"]),
        "foreground_pixels_512": int(record["fg_px_512"]),
        "mask": row["mask"].strip(),
    }
    if row["mask"].strip():
        mask_path = data_root / row["mask"]
        entry["mask_sha256"] = sha256_file(mask_path)
        entry["mask_size_bytes"] = mask_path.stat().st_size
        entry["mask_values"] = record["mask_values"]
    else:
        entry["mask_sha256"] = ""
        entry["mask_size_bytes"] = 0
        entry["mask_values"] = ""

    # Recompute the 512 foreground count instead of trusting the audit file:
    # this is the number the trainer actually optimises against.
    with Image.open(image_path) as handle:
        handle.load()
        width, height = handle.size
        if row["mask"].strip():
            with Image.open(data_root / row["mask"]) as mask_handle:
                raw = np.asarray(mask_handle)
            binary = geometry.binarize_ids(raw)
        else:
            binary = np.zeros((height, width), dtype=np.uint8)
    box = geometry.compute_letterbox(width, height, int(config["input_size"]))
    label512 = geometry.letterbox_label(binary, box, int(config["ignore_index"]))
    entry["recomputed_foreground_pixels_512"] = int((label512 == 1).sum())
    entry["letterbox"] = {"new_w": box.new_w, "new_h": box.new_h,
                          "pad_left": box.pad_left, "pad_top": box.pad_top}
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="freeze the 4 normal + 4 anomaly training subset")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--ids", default=ID_FILE)
    parser.add_argument("--detail", default=DETAIL_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    config = load_config(args.config)
    train_path = Path(config["paths"]["data.train_manifest"])
    val_path = Path(config["paths"]["data.val_manifest"])
    test_path = Path(config["paths"]["data.test_manifest"])
    rows = read_rows(train_path)

    audit_path = paths.PROJECT_ROOT / "reports/data_audit/per_image.csv"
    if not audit_path.is_file():
        raise SystemExit(f"missing {audit_path}; run stage 04 audit first")
    audit = {row["image"]: row for row in read_rows(audit_path) if row["split"] == "train"}

    ids_path = Path(args.ids) if Path(args.ids).is_absolute() else paths.PROJECT_ROOT / args.ids
    if args.verify_only:
        if not ids_path.is_file():
            raise SystemExit(f"ids file not found: {ids_path}")
        chosen = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_image = {row["image"]: row for row in rows}
        missing = [item for item in chosen if item not in by_image]
        if missing:
            raise SystemExit(f"{len(missing)} frozen ids are no longer in train: {missing}")
        selected = [by_image[item] for item in chosen]
        print(f"verify-only: {len(selected)} ids still present in train")
    else:
        normals = [row for row in rows if row["label"] == "normal"]
        anomalies = [row for row in rows if row["label"] == "anomaly"]
        if len(normals) != 482 or len(anomalies) != 48:
            print(f"  note: train composition is {len(normals)} normal / {len(anomalies)} anomaly")

        normal_pick = random.Random(args.seed).sample(sorted(normals, key=lambda r: r["image"]), 4)
        anomaly_pick = pick_anomalies(anomalies, audit, 4)
        selected = sorted(normal_pick, key=lambda r: r["image"]) + anomaly_pick

        labels = [row["label"] for row in selected]
        if labels.count("normal") != 4 or labels.count("anomaly") != 4:
            raise SystemExit(f"subset must be 4 normal + 4 anomaly, got {labels}")
        if len({row["image"] for row in selected}) != 8:
            raise SystemExit("duplicate ids in the subset")

        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text("\n".join(row["image"] for row in selected) + "\n", encoding="utf-8")
        print(f"  wrote {ids_path}")

    # Never let a val/test id slip into the training subset.
    allowed = {row["image"] for row in rows}
    for split_name, split_path in (("val", val_path), ("test", test_path)):
        other = {row["image"] for row in read_rows(split_path)}
        overlap = allowed & other
        if overlap:
            raise SystemExit(f"train and {split_name} overlap on {len(overlap)} images")

    detail = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection": {
            "seed": args.seed,
            "normal_rule": "random.Random(seed).sample over train normals sorted by image path",
            "anomaly_rule": "ranks 0/33/66/100 % of train anomalies sorted by original foreground pixels",
            "uses_val_or_test": False,
        },
        "ids": {
            "path": str(ids_path.resolve()),
            "sha256": sha256_file(ids_path),
            "lines": len(selected),
        },
        "manifests": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(read_rows(path))}
            for name, path in (("train", train_path), ("val", val_path), ("test", test_path))
        },
        "snapshot": {
            "dir": str(Path(config["paths"]["model.pretrained_dir"]).resolve()),
            "source_sha256": sha256_file(Path(config["paths"]["model.pretrained_dir"]) / "source.json"),
            "weights_sha256": sha256_file(Path(config["paths"]["model.pretrained_dir"]) / "pytorch_model.bin"),
        },
        "config": {"path": str(Path(config["path"]).resolve()), "sha256": sha256_file(Path(config["path"]))},
        "samples": [describe(config, row, audit) for row in selected],
    }
    detail_path = Path(args.detail) if Path(args.detail).is_absolute() else paths.PROJECT_ROOT / args.detail
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  subset : {len(selected)} images "
          f"({sum(1 for r in selected if r['label'] == 'normal')} normal / "
          f"{sum(1 for r in selected if r['label'] == 'anomaly')} anomaly)")
    for entry in detail["samples"]:
        print(f"    {entry['label']:7s} {entry['image'].split('/')[-1]:9s} "
              f"fg={entry['original_foreground_pixels']:>7d} px ({entry['original_foreground_ratio']*100:6.3f}%) "
              f"512={entry['recomputed_foreground_pixels_512']:>6d}")
    print(f"  ids sha256: {detail['ids']['sha256']}")
    print(f"  detail    : {detail_path.resolve()}  ({time.perf_counter() - started:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
