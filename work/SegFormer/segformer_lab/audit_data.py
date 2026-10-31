"""Stage 4 data audit: decode everything, measure labels, find duplicates.

What this module answers, in order
----------------------------------
1. Are the manifests still the ones the split protocol produced? (SHA-256)
2. Can every image and every anomaly mask be decoded *completely* (not just its
   header), and do image/mask sizes and modes agree?
3. What values do the official masks store, how large is the defect area, and
   does a normal image really have no mask?
4. Are there byte-identical files, and which cross-split pairs are suspiciously
   similar? Similarity is reported as a review lead only; nothing is re-split.
5. After the letterbox transform to 512, how much foreground survives, and does
   any original connected region disappear? Padding is 255 and is excluded from those
   numbers.

Memory behaviour: one image is decoded at a time and released before the next.
The only per-image state kept in memory is a row of numbers for the CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import __version__, geometry, paths
from .config import load_config, write_resolved_config
from .prepare_data import OfficialRow, check_group_splits, read_groups

SCHEMA = "segformer-lab/data-audit/2"
CHUNK = 1024 * 1024

# Near-duplicate search: every cross-split pair is compared on a 64x64 grayscale
# thumbnail (Pearson correlation). A 64-bit dHash is recorded as supporting
# evidence only, because on PCB photographs it is far too permissive: a loose
# dHash threshold flags ~90k pairs and tells the reviewer nothing.
DHASH_MAX = 4  # only used to label how close a surviving candidate is
THUMB_SIZE = 64
CORR_MIN = 0.985
MAD_MAX = 6.0


# --------------------------------------------------------------------- utils ----
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _percent(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "p10": round(_percent(values, 0.10), 6),
        "median": round(statistics.median(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "p90": round(_percent(values, 0.90), 6),
        "max": round(max(values), 6),
    }


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _thumb(image: Image.Image) -> np.ndarray:
    """64x64 grayscale thumbnail used for the decisive similarity check."""
    small = image.convert("L").resize((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR)
    return np.asarray(small, dtype=np.float32)


def _similarity(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return (Pearson correlation, mean absolute difference in 0..255)."""
    a = left.reshape(-1) - left.mean()
    b = right.reshape(-1) - right.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    corr = 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)
    mad = float(np.abs(left - right).mean())
    return corr, mad


def _normalized_matrix(records: list[dict[str, Any]], thumbs: dict[str, np.ndarray]) -> np.ndarray:
    """Row-normalised thumbnails, so a dot product is a Pearson correlation."""
    matrix = np.stack([thumbs[r["image"]].reshape(-1) for r in records]).astype(np.float32)
    matrix -= matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


# ------------------------------------------------------------------ manifests ----
def verify_manifests(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compare the manifests on disk with the identities recorded at split time."""
    meta_path = cfg["paths"]["data.split_meta"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "split_meta": str(meta_path),
        "split_meta_sha256": sha256_file(meta_path),
        "manifests": {},
        "ok": True,
    }
    for name in ("train", "val", "test"):
        path = cfg["paths"][f"data.{name}_manifest"]
        actual = sha256_file(path)
        expected = meta["manifests"][name]["sha256"]
        match = actual == expected
        report["manifests"][name] = {
            "path": str(path),
            "sha256": actual,
            "expected": expected,
            "match": match,
            "rows": meta["manifests"][name]["rows"],
        }
        report["ok"] = report["ok"] and match

    official_copy = cfg["paths"]["data.split_meta"].parent / "official_2cls_highshot.csv"
    expected_official = meta["sources"]["official_csv"]["sha256"]
    actual_official = sha256_file(official_copy)
    report["official_csv"] = {
        "path": str(official_copy),
        "sha256": actual_official,
        "expected": expected_official,
        "match": actual_official == expected_official,
    }
    report["ok"] = report["ok"] and report["official_csv"]["match"]
    report["protocol"] = meta.get("protocol", {"name": "pcb1-highshot", "version": 1, "grouped": False})
    group_source = meta.get("sources", {}).get("reviewed_groups")
    group_report: dict[str, Any] = {"applicable": bool(group_source), "ok": True}
    if group_source:
        try:
            group_path = Path(group_source["path"])
            if not group_path.is_absolute():
                group_path = meta_path.parent / group_path
            actual_group_sha = sha256_file(group_path)
            with official_copy.open(encoding="utf-8", newline="") as handle:
                official_rows = [OfficialRow(**row) for row in csv.DictReader(handle)]
            groups = read_groups(group_path, official_rows, meta.get("category", "pcb1"))
            derived = {
                split: [OfficialRow(**row) for row in _read_manifest(cfg["paths"][f"data.{split}_manifest"])]
                for split in ("train", "val", "test")
            }
            group_report.update(check_group_splits(derived, groups))
            group_report.update({"path": str(group_path), "sha256": actual_group_sha,
                                 "expected_sha256": group_source["sha256"],
                                 "source_match": actual_group_sha == group_source["sha256"]})
            group_report["ok"] = group_report["ok"] and group_report["source_match"]
        except (OSError, ValueError, KeyError, SystemExit) as exc:
            group_report.update({"ok": False, "error": str(exc)})
    elif report["protocol"].get("grouped"):
        group_report.update({"ok": False, "error": "grouped protocol has no reviewed group source"})
    report["reviewed_groups"] = group_report
    report["ok"] = report["ok"] and group_report["ok"]
    return report


# ----------------------------------------------------------------- audit body ----
def audit(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    data_root: Path = cfg["data_root"]
    input_size: int = cfg["input_size"]
    ignore_index: int = cfg["ignore_index"]

    manifest_check = verify_manifests(cfg)
    print(f"  manifests verified: {manifest_check['ok']} "
          f"(meta sha {manifest_check['split_meta_sha256'][:12]}…)")

    rows: list[dict[str, Any]] = []
    per_split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in ("train", "val", "test"):
        for row in _read_manifest(cfg["paths"][f"data.{split}_manifest"]):
            row["split"] = split
            per_split_rows[split].append(row)
            rows.append(row)

    decode_failures: list[dict[str, str]] = []
    size_counter: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    mask_modes: Counter[str] = Counter()
    mask_size_mismatch: list[dict[str, str]] = []
    raw_value_sets: Counter[str] = Counter()
    raw_value_union: set[int] = set()
    normal_with_mask_file: list[str] = []
    normal_label_failures: list[str] = []
    anomaly_without_mask: list[str] = []
    empty_foreground: list[str] = []
    fg_ratio_by_split: dict[str, list[float]] = defaultdict(list)
    fg_ratio_512_by_split: dict[str, list[float]] = defaultdict(list)
    images_without_surviving_foreground: list[str] = []
    component_effects: list[dict[str, Any]] = []
    padding_fractions: list[float] = []
    thumbs: dict[str, np.ndarray] = {}

    total = len(rows)
    for index, row in enumerate(rows, start=1):
        image_rel = row["image"]
        mask_rel = row["mask"].strip()
        image_path = data_root / image_rel
        record: dict[str, Any] = {
            "split": row["split"],
            "label": row["label"],
            "image": image_rel,
            "mask": mask_rel,
            "file_sha256": "",
            "dhash": "",
            "width": "",
            "height": "",
            "mode": "",
            "decode_ok": 0,
            "mask_ok": "" if not mask_rel else 0,
            "fg_px": 0,
            "fg_ratio": 0.0,
            "mask_values": "",
            "components": 0,
            "largest_component": 0,
            "fg_px_512": 0,
            "fg_ratio_512": 0.0,
            "components_512": 0,
            "largest_component_512": 0,
            "original_components_retained": 0,
            "original_components_lost": 0,
            "split_original_components": 0,
            "merged_output_components": 0,
        }

        if not image_path.is_file():
            decode_failures.append({"image": image_rel, "error": "missing file"})
            rows[index - 1] = record
            continue

        try:
            record["file_sha256"] = sha256_file(image_path)
            with Image.open(image_path) as image:
                image.load()  # force a full decode, not just the header
                width, height = image.size
                mode = image.mode
                record.update(width=width, height=height, mode=mode, decode_ok=1)
                record["dhash"] = f"{geometry.dhash64(image):016x}"
                thumbs[image_rel] = _thumb(image)
        except Exception as exc:  # noqa: BLE001 - report per-file, keep going
            decode_failures.append({"image": image_rel, "error": f"{type(exc).__name__}: {exc}"})
            rows[index - 1] = record
            continue

        size_counter[f"{width}x{height}"] += 1
        image_modes[mode] += 1
        box = geometry.compute_letterbox(width, height, input_size)
        padding_fractions.append(box.padding_pixels / (input_size * input_size))

        if row["label"] == "normal":
            if mask_rel:
                normal_with_mask_file.append(image_rel)
            # Zero label in original size; letterbox keeps 0 inside the valid box.
            label_512 = geometry.letterbox_label(
                np.zeros((height, width), dtype=np.uint8), box, ignore_index
            )
            top, left, bottom, right = box.crop_box
            inside = label_512[top:bottom, left:right]
            if int(inside.max()) != 0 or int(label_512[0, 0]) != ignore_index:
                normal_label_failures.append(image_rel)
            record["fg_px"] = 0
            record["fg_ratio"] = 0.0
            record["components"] = 0
            record["fg_px_512"] = 0
            record["fg_ratio_512"] = 0.0
            record["components_512"] = 0
        else:
            if not mask_rel:
                anomaly_without_mask.append(image_rel)
                rows[index - 1] = record
                continue
            mask_path = data_root / mask_rel
            if not mask_path.is_file():
                decode_failures.append({"image": mask_rel, "error": "mask missing"})
                rows[index - 1] = record
                continue
            try:
                with Image.open(mask_path) as mask_image:
                    mask_image.load()
                    mask_modes[mask_image.mode] += 1
                    mask_array = np.asarray(mask_image)
                    if mask_array.ndim != 2:
                        raise ValueError(f"mask is not single channel: shape {mask_array.shape}")
                    if (mask_array.shape[1], mask_array.shape[0]) != (width, height):
                        mask_size_mismatch.append(
                            {"mask": mask_rel, "image": image_rel,
                             "mask_size": f"{mask_array.shape[1]}x{mask_array.shape[0]}",
                             "image_size": f"{width}x{height}"}
                        )
                        rows[index - 1] = record
                        continue
            except Exception as exc:  # noqa: BLE001
                decode_failures.append({"image": mask_rel, "error": f"{type(exc).__name__}: {exc}"})
                rows[index - 1] = record
                continue

            values = np.unique(mask_array)
            raw_value_union.update(int(v) for v in values)
            raw_value_sets["{" + ",".join(str(int(v)) for v in values) + "}"] += 1
            binary = geometry.binarize_ids(mask_array)
            fg_px = int(binary.sum())
            fg_ratio = fg_px / (width * height)
            retention = geometry.component_retention(binary, box)
            components = retention["components_before"]
            largest = retention["largest_component_before"]
            if fg_px == 0:
                empty_foreground.append(image_rel)

            label_512 = geometry.letterbox_label(binary, box, ignore_index)
            top, left, bottom, right = box.crop_box
            valid = label_512[top:bottom, left:right]
            fg_px_512 = int((valid == 1).sum())
            components_512 = retention["components_after"]
            largest_512 = retention["largest_component_after"]
            if fg_px_512 == 0:
                images_without_surviving_foreground.append(image_rel)
            component_effects.append({"image": image_rel, "split": row["split"], **retention})

            record.update(
                mask_ok=1,
                fg_px=fg_px,
                fg_ratio=round(fg_ratio, 8),
                mask_values=";".join(str(int(v)) for v in values),
                components=components,
                largest_component=largest,
                fg_px_512=fg_px_512,
                fg_ratio_512=round(fg_px_512 / box.valid_area, 8),
                components_512=components_512,
                largest_component_512=largest_512,
                original_components_retained=retention["retained_components"],
                original_components_lost=len(retention["lost_components"]),
                split_original_components=retention["split_original_components"],
                merged_output_components=retention["merged_output_components"],
            )
            fg_ratio_by_split[row["split"]].append(fg_ratio * 100.0)
            fg_ratio_512_by_split[row["split"]].append(fg_px_512 / box.valid_area * 100.0)

        record["scale"] = round(box.scale, 6)
        record["new_w"] = box.new_w
        record["new_h"] = box.new_h
        record["pad_left"] = box.pad_left
        record["pad_top"] = box.pad_top
        record["pad_right"] = box.pad_right
        record["pad_bottom"] = box.pad_bottom
        rows[index - 1] = record

        if index % 200 == 0:
            print(f"  ... {index}/{total} images", flush=True)

    # ---------------------------------------------------------- duplicates ----
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        if record["decode_ok"]:
            by_hash[record["file_sha256"]].append(record)
    exact_groups = [
        {
            "sha256": digest,
            "count": len(members),
            "splits": sorted({m["split"] for m in members}),
            "images": [{"split": m["split"], "image": m["image"]} for m in members],
        }
        for digest, members in by_hash.items()
        if len(members) > 1
    ]
    exact_cross_split = [g for g in exact_groups if len(g["splits"]) > 1]

    near_pairs: list[dict[str, Any]] = []
    dhash_histogram: Counter[int] = Counter()
    considered = 0
    correlations: list[float] = []
    decode_rows = [r for r in rows if r["decode_ok"] and r["dhash"]]
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in decode_rows:
        by_split[record["split"]].append(record)
    for left, right in (("train", "test"), ("train", "val"), ("val", "test")):
        left_rows = by_split[left]
        right_rows = by_split[right]
        if not left_rows or not right_rows:
            continue
        left_matrix = _normalized_matrix(left_rows, thumbs)
        right_matrix = _normalized_matrix(right_rows, thumbs)
        corr = left_matrix @ right_matrix.T                    # Pearson correlation
        distances = geometry.hamming_matrix(
            [int(r["dhash"], 16) for r in left_rows],
            [int(r["dhash"], 16) for r in right_rows],
        )
        for level, count in zip(*np.unique(distances, return_counts=True)):
            if int(level) <= 8:
                dhash_histogram[int(level)] += int(count)
        considered += int(corr.size)
        correlations.extend(float(v) for v in corr.reshape(-1)[::37])
        for i, j in np.argwhere(corr >= CORR_MIN):
            _, mad = _similarity(thumbs[left_rows[i]["image"]], thumbs[right_rows[j]["image"]])
            if mad > MAD_MAX:
                continue
            near_pairs.append(
                {
                    "pair": f"{left}-{right}",
                    "correlation": round(float(corr[i, j]), 6),
                    "mean_abs_diff": round(mad, 3),
                    "dhash_distance": int(distances[i, j]),
                    "left": {"split": left, "label": left_rows[i]["label"], "image": left_rows[i]["image"]},
                    "right": {"split": right, "label": right_rows[j]["label"], "image": right_rows[j]["image"]},
                }
            )
    near_pairs.sort(key=lambda item: (-item["correlation"], item["left"]["image"], item["right"]["image"]))

    # ------------------------------------------------------------- summary ----
    anomaly_rows = [r for r in rows if r["label"] == "anomaly" and r["decode_ok"]]
    normal_rows = [r for r in rows if r["label"] == "normal" and r["decode_ok"]]
    count_decreased = [r for r in component_effects if r["components_after"] < r["components_before"]]
    count_increased = [r for r in component_effects if r["components_after"] > r["components_before"]]
    lost_original = [r for r in component_effects if r["lost_components"]]
    counts = {
        split: {
            "normal": sum(1 for r in per_split_rows[split] if r["label"] == "normal"),
            "anomaly": sum(1 for r in per_split_rows[split] if r["label"] == "anomaly"),
            "total": len(per_split_rows[split]),
        }
        for split in ("train", "val", "test")
    }

    checks = {
        "manifests_match_split_meta": manifest_check["ok"],
        "reviewed_groups_stay_in_one_split": manifest_check["reviewed_groups"]["ok"],
        "all_images_decoded": len(decode_failures) == 0,
        "single_image_size": len(size_counter) == 1,
        "images_rgb": set(image_modes) <= {"RGB"},
        "masks_single_channel_L": set(mask_modes) <= {"L"},
        "mask_sizes_match": not mask_size_mismatch,
        "all_anomalies_have_mask": not anomaly_without_mask,
        "no_empty_foreground": not empty_foreground,
        "normals_have_no_mask_field": not normal_with_mask_file,
        "normal_512_labels_zero_inside_ignore_outside": not normal_label_failures,
        "foreground_survives_512": not images_without_surviving_foreground,
        "counts_match_protocol": counts == {
            "train": {"normal": 482, "anomaly": 48, "total": 530},
            "val": {"normal": 120, "anomaly": 12, "total": 132},
            "test": {"normal": 402, "anomaly": 40, "total": 442},
        },
    }

    summary = {
        "schema": SCHEMA,
        "package_version": __version__,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "config": str(cfg["path"]),
        "data_root": str(data_root),
        "input_size": input_size,
        "ignore_index": ignore_index,
        "counts": counts,
        "manifest_check": manifest_check,
        "decode": {
            "images_total": len(rows),
            "images_ok": sum(1 for r in rows if r["decode_ok"]),
            "masks_total": sum(1 for r in rows if r["mask"]),
            "masks_ok": sum(1 for r in rows if r["mask"] and r.get("mask_ok") == 1),
            "failures": decode_failures,
        },
        "image_sizes": dict(size_counter),
        "image_modes": dict(image_modes),
        "mask_modes": dict(mask_modes),
        "mask_size_mismatch": mask_size_mismatch,
        "mask_raw_value_sets": dict(raw_value_sets),
        "mask_raw_value_union": sorted(raw_value_union),
        "anomaly_foreground_percent": {split: _stats(values) for split, values in fg_ratio_by_split.items()},
        "anomaly_foreground_percent_512": {split: _stats(values) for split, values in fg_ratio_512_by_split.items()},
        "empty_foreground": empty_foreground,
        "anomaly_without_mask": anomaly_without_mask,
        "normal_with_mask_field": normal_with_mask_file,
        "letterbox": {
            "padding_fraction_percent": _stats([v * 100.0 for v in padding_fractions]),
            "example": next(({k: r[k] for k in ("width", "height", "scale", "new_w", "new_h",
                                                "pad_left", "pad_top", "pad_right", "pad_bottom")}
                             for r in rows if r["decode_ok"]), None),
        },
        "transform_512": {
            "foreground_lost_images": len(images_without_surviving_foreground),
            "foreground_lost_examples": images_without_surviving_foreground[:10],
            "component_tracking_method": (
                "Original 4-connected regions receive int32 ids and are sampled with the same "
                "Pillow NEAREST resize as the binary label, excluding padding. Retained means "
                "at least one sampled output pixel keeps the original id. Split/merge counts "
                "come from original-id/output-component overlaps, not net component counts. "
                "Every output foreground pixel inherits an original id; count increases are "
                "not newly created defect instances."
            ),
            "component_count_decreased_images": len(count_decreased),
            "component_count_increased_images": len(count_increased),
            "component_count_unchanged_images": len(component_effects) - len(count_decreased) - len(count_increased),
            "component_count_decrease_examples": count_decreased[:10],
            "original_components_total": sum(r["components_before"] for r in component_effects),
            "original_components_retained": sum(r["retained_components"] for r in component_effects),
            "original_components_lost": sum(len(r["lost_components"]) for r in component_effects),
            "images_with_lost_original_components": len(lost_original),
            "lost_original_component_examples": lost_original[:10],
            "split_original_components": sum(r["split_original_components"] for r in component_effects),
            "merged_output_components": sum(r["merged_output_components"] for r in component_effects),
            "area_ratio_percent": _stats(
                [100.0 * r["fg_ratio_512"] / r["fg_ratio"] for r in anomaly_rows if r["fg_ratio"] > 0]
            ),
        },
        "duplicates": {
            "exact_groups": len(exact_groups),
            "exact_cross_split_groups": len(exact_cross_split),
            "exact_group_examples": exact_groups[:10],
        },
        "near_duplicates": {
            "method": (
                "every cross-split pair compared on a 64x64 grayscale thumbnail: "
                f"Pearson correlation >= {CORR_MIN} and mean absolute difference <= {MAD_MAX}; "
                "dHash distance is reported per candidate as supporting evidence"
            ),
            "pairs_compared": considered,
            "correlation_stats_sampled_every_37th": _stats(correlations),
            "dhash_distance_histogram_le8": {str(k): v for k, v in sorted(dhash_histogram.items())},
            "candidate_pairs": len(near_pairs),
            "by_pair": dict(Counter(item["pair"] for item in near_pairs)),
            "closest": near_pairs[:50],
            "note": "review leads only; the split is never changed automatically",
        },
        "normal_zero_mask_rows": len(normal_rows),
        "normal_label_failures": normal_label_failures[:10],
        "anomaly_rows": len(anomaly_rows),
        "checks": checks,
        "ok": all(checks.values()),
        "elapsed_seconds": None,
    }
    summary["elapsed_seconds"] = round(time.perf_counter() - start, 2)

    # -------------------------------------------------------------- outputs ----
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "duplicates_exact.json").write_text(json.dumps(exact_groups, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "near_duplicates.json").write_text(json.dumps(near_pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "component_retention.json").write_text(
        json.dumps(component_effects, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fieldnames = [
        "split", "label", "image", "mask", "width", "height", "mode", "decode_ok", "mask_ok",
        "file_sha256", "dhash", "fg_px", "fg_ratio", "mask_values", "components", "largest_component",
        "fg_px_512", "fg_ratio_512", "components_512", "largest_component_512",
        "original_components_retained", "original_components_lost", "split_original_components", "merged_output_components",
        "scale", "new_w", "new_h", "pad_left", "pad_top", "pad_right", "pad_bottom",
    ]
    with (out_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in rows:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    write_resolved_config(cfg, out_dir / "resolved_config.yaml")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    """Short human-readable digest; the JSON stays the source of truth."""
    lines = ["# Data audit summary", ""]
    lines.append(f"- created: {summary['created_at']}")
    lines.append(f"- data root: `{summary['data_root']}`")
    lines.append(f"- input size: {summary['input_size']}, ignore index: {summary['ignore_index']}")
    lines.append(f"- overall: **{'ok' if summary['ok'] else 'FAILED'}** in {summary['elapsed_seconds']} s")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| split | normal | anomaly | total |")
    lines.append("| --- | ---: | ---: | ---: |")
    for split, counts in summary["counts"].items():
        lines.append(f"| {split} | {counts['normal']} | {counts['anomaly']} | {counts['total']} |")
    lines.append("")
    lines.append("## Decode")
    lines.append("")
    lines.append(f"- image sizes: `{summary['image_sizes']}`")
    lines.append(f"- image modes: `{summary['image_modes']}`")
    lines.append(f"- mask modes: `{summary['mask_modes']}`")
    lines.append(f"- decode failures: {len(summary['decode']['failures'])}")
    lines.append("")
    lines.append("## Mask semantics")
    lines.append("")
    lines.append(f"- raw value union: `{summary['mask_raw_value_union']}`")
    lines.append(f"- raw value sets: `{summary['mask_raw_value_sets']}`")
    lines.append("")
    lines.append("## Anomaly foreground (% of image)")
    lines.append("")
    lines.append("| split | min | p10 | median | mean | p90 | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for split, stats in summary["anomaly_foreground_percent"].items():
        lines.append(
            f"| {split} | {stats['min']} | {stats['p10']} | {stats['median']} | "
            f"{stats['mean']} | {stats['p90']} | {stats['max']} |"
        )
    lines.append("")
    lines.append("## 512 letterbox")
    lines.append("")
    box = summary["letterbox"]["example"]
    lines.append(f"- padding fraction: {summary['letterbox']['padding_fraction_percent']['median']}% (median)")
    if box:
        lines.append(
            f"- example {box['width']}x{box['height']} -> {box['new_w']}x{box['new_h']} at scale "
            f"{box['scale']}, pads L{box['pad_left']} T{box['pad_top']} R{box['pad_right']} B{box['pad_bottom']}"
        )
    transform = summary["transform_512"]
    lines.append(f"- anomalies whose 512 foreground became empty: {transform['foreground_lost_images']}")
    lines.append(f"- images with decreased/increased/unchanged component counts: "
                 f"{transform['component_count_decreased_images']}/"
                 f"{transform['component_count_increased_images']}/"
                 f"{transform['component_count_unchanged_images']}")
    lines.append(f"- original components retained/lost: {transform['original_components_retained']}/"
                 f"{transform['original_components_lost']} of {transform['original_components_total']}; "
                 f"images with an actual lost region: {transform['images_with_lost_original_components']}")
    lines.append(f"- original components split: {transform['split_original_components']}; "
                 f"output components merging original ids: {transform['merged_output_components']}")
    area = transform["area_ratio_percent"]
    if area.get("count"):
        lines.append(f"- 512/original foreground ratio: median {area['median']}% (min {area['min']}%, max {area['max']}%)")
    lines.append("")
    lines.append("## Duplicates")
    lines.append("")
    lines.append(f"- exact duplicate groups: {summary['duplicates']['exact_groups']} "
                 f"(cross-split: {summary['duplicates']['exact_cross_split_groups']})")
    near = summary["near_duplicates"]
    lines.append(f"- near-duplicate method: {near['method']}")
    lines.append(f"- dHash distance histogram (<=8): `{near['dhash_distance_histogram_le8']}`")
    lines.append(f"- cross-split pairs compared: {near['pairs_compared']}, "
                 f"candidates: {near['candidate_pairs']} {near['by_pair']}")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    for name, value in summary["checks"].items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — {name}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--out", default="reports/data_audit")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = paths.resolve(args.out)
    print("SegFormer stage 4: data audit")
    print(f"  config : {cfg['path']}")
    print(f"  data   : {cfg['data_root']}")
    print(f"  output : {out_dir}")

    summary = audit(cfg, out_dir)
    print("")
    print(f"  images decoded : {summary['decode']['images_total']} (failures {len(summary['decode']['failures'])})")
    print(f"  mask values    : {summary['mask_raw_value_union']}")
    print(f"  exact dups     : {summary['duplicates']['exact_groups']} groups, "
          f"cross-split {summary['duplicates']['exact_cross_split_groups']}")
    print(f"  near dups      : {summary['near_duplicates']['candidate_pairs']} candidates")
    print(f"  512 lost fg    : {summary['transform_512']['foreground_lost_images']} images")
    print(f"  checks ok      : {summary['ok']}")
    print(f"  summary        : {out_dir / 'summary.json'}")
    if not summary["ok"]:
        failed = [name for name, value in summary["checks"].items() if not value]
        print(f"  RESULT: FAILED checks: {failed}", file=sys.stderr)
        return 1
    print("  RESULT: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
