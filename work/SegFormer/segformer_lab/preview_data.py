"""Stage 4 preview: render fixed, human-checkable samples.

Each selected image becomes one five-tile figure:

    original RGB | GT binary | red overlay | 512 letterboxed input | 512 label (255 = ignore)

The overlay is what catches real mistakes: an image/mask mismatch, a transposed
pair or a flipped mask is invisible in a black-and-white mask alone.

Selection is deterministic and independent of model results: normals are taken
at evenly spaced positions of the sorted path list, anomalies at evenly spaced
positions of the sorted foreground-ratio list, so small and large defects are
both represented. For the validation split the chosen ids are written to
``manifests/val_preview_ids.txt`` and reused afterwards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from . import geometry, paths  # noqa: E402
from .config import load_config, write_resolved_config  # noqa: E402


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def even_indices(count: int, wanted: int) -> list[int]:
    """Evenly spaced positions over ``range(count)``, inclusive of both ends."""
    if wanted <= 0 or count <= 0:
        return []
    if wanted >= count:
        return list(range(count))
    if wanted == 1:
        return [count // 2]
    return sorted({int(round(i * (count - 1) / (wanted - 1))) for i in range(wanted)})


def mask_foreground_ratio(data_root: Path, mask_rel: str) -> tuple[int, float]:
    with Image.open(data_root / mask_rel) as mask_image:
        mask_image.load()
        array = np.asarray(mask_image)
    binary = geometry.binarize_ids(array)
    foreground = int(binary.sum())
    ratio = foreground / binary.size
    return foreground, ratio


def select_rows(
    rows: list[dict[str, str]],
    data_root: Path,
    normal_count: int,
    anomaly_count: int,
) -> list[dict[str, str]]:
    normals = sorted([row for row in rows if row["label"] == "normal"], key=lambda r: r["image"])
    anomalies = [row for row in rows if row["label"] == "anomaly"]
    scored = []
    for row in anomalies:
        foreground, ratio = mask_foreground_ratio(data_root, row["mask"])
        scored.append((ratio, foreground, row))
    scored.sort(key=lambda item: (item[0], item[2]["image"]))
    anomalies_sorted = [item[2] for item in scored]

    picked = [normals[i] for i in even_indices(len(normals), normal_count)]
    picked += [anomalies_sorted[i] for i in even_indices(len(anomalies_sorted), anomaly_count)]
    return picked


def select_with_ids(
    rows: list[dict[str, str]],
    ids_path: Path,
    normal_count: int | None,
    anomaly_count: int | None,
) -> list[dict[str, str]] | None:
    """Reuse frozen ids; an invalid file must never trigger silent reselection."""
    if not ids_path.is_file():
        return None
    wanted = []
    for line in ids_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            wanted.append(line)
    by_image = {row["image"]: row for row in rows}
    if not wanted or len(wanted) != len(set(wanted)) or any(item not in by_image for item in wanted):
        raise ValueError(f"invalid fixed preview ids (empty, duplicate or outside split): {ids_path}")
    picked = [by_image[item] for item in wanted]
    for label, expected in (("normal", normal_count), ("anomaly", anomaly_count)):
        if expected is not None and sum(r["label"] == label for r in picked) != expected:
            raise ValueError(f"requested {label} count conflicts with fixed preview ids: {ids_path}")
    return picked


def write_ids(path: Path, rows: list[dict[str, str]], split: str) -> None:
    normal = sum(1 for r in rows if r["label"] == "normal")
    anomaly = sum(1 for r in rows if r["label"] == "anomaly")
    header = [
        "# Fixed preview samples for the SegFormer PCB1 baseline.",
        f"# split={split} normal={normal} anomaly={anomaly}",
        "# Deterministic selection; do not edit after the first preview run.",
    ]
    lines = header + [row["image"] for row in rows]
    if path.exists():
        if select_with_ids(rows, path, normal, anomaly) != rows:
            raise ValueError(f"refusing to replace fixed preview ids: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _label_to_rgb(label_512: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label_512.shape, 3), dtype=np.uint8)
    rgb[label_512 == 1] = (235, 56, 56)
    rgb[label_512 == 255] = (115, 115, 115)
    return rgb


def render_sample(
    row: dict[str, str],
    data_root: Path,
    input_size: int,
    ignore_index: int,
    pad_rgb: tuple[int, int, int],
    out_path: Path,
) -> dict[str, Any]:
    image_path = data_root / row["image"]
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        rgb_image = image.convert("RGB")

    if row["mask"]:
        with Image.open(data_root / row["mask"]) as mask_image:
            mask_image.load()
            mask_array = np.asarray(mask_image)
        binary = geometry.binarize_ids(mask_array)
        raw_values = [int(v) for v in np.unique(mask_array)]
    else:
        binary = np.zeros((height, width), dtype=np.uint8)
        raw_values = []

    box = geometry.compute_letterbox(width, height, input_size)
    image_512 = geometry.letterbox_image(rgb_image, box, pad_rgb)
    label_512 = geometry.letterbox_label(binary, box, ignore_index)

    foreground = int(binary.sum())
    ratio = foreground / binary.size
    top, left, bottom, right = box.crop_box

    figure, axes = plt.subplots(1, 5, figsize=(17.0, 3.9), dpi=150)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("0.7")

    axes[0].imshow(rgb_image)
    axes[0].set_title(f"original {width}x{height}", fontsize=9)

    axes[1].imshow(binary, cmap=ListedColormap([(0.05, 0.05, 0.05), (0.92, 0.22, 0.22)]), vmin=0, vmax=1,
                   interpolation="nearest")
    if row["mask"]:
        axes[1].set_title(f"GT binary (raw ids {raw_values})\n{foreground} px, {ratio * 100:.3f}%", fontsize=9)
    else:
        axes[1].set_title("constructed zero mask\n(no mask file for normal)", fontsize=9)

    axes[2].imshow(rgb_image)
    axes[2].imshow(np.ma.masked_where(~binary.astype(bool), binary), cmap=ListedColormap([(0.95, 0.15, 0.15)]),
                   alpha=0.45, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("red overlay (+ zoom inset)", fontsize=9)

    # A 0.1% defect is invisible at page scale, so small defects get a zoom inset;
    # without it "does the mask cover the real scratch" cannot be answered.
    rgb_array = np.asarray(rgb_image)
    if binary.any():
        rows_any = np.flatnonzero(binary.any(axis=1))
        cols_any = np.flatnonzero(binary.any(axis=0))
        y0, y1 = int(rows_any[0]), int(rows_any[-1])
        x0, x1 = int(cols_any[0]), int(cols_any[-1])
        box_w, box_h = x1 - x0 + 1, y1 - y0 + 1
        if box_w < width * 0.35 and box_h < height * 0.35:
            margin = int(max(16, 0.35 * max(box_w, box_h)))
            cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
            cx1, cy1 = min(width, x1 + margin + 1), min(height, y1 + margin + 1)
            axes[2].add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0,
                                        fill=False, edgecolor="yellow", linewidth=0.9))
            inset = axes[2].inset_axes([0.60, 0.04, 0.38, 0.38])
            inset.imshow(rgb_array[cy0:cy1, cx0:cx1])
            inset.imshow(
                np.ma.masked_where(~binary[cy0:cy1, cx0:cx1].astype(bool), binary[cy0:cy1, cx0:cx1]),
                cmap=ListedColormap([(0.95, 0.15, 0.15)]), alpha=0.45, vmin=0, vmax=1, interpolation="nearest",
            )
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title("zoom", fontsize=6, pad=1)
            for spine in inset.spines.values():
                spine.set_color("yellow")

    axes[3].imshow(image_512)
    axes[3].add_patch(Rectangle((left, top), right - left - 1, bottom - top - 1,
                                fill=False, edgecolor="cyan", linewidth=1.0, linestyle="--"))
    axes[3].set_title(
        f"512 input, scale {box.scale:.4f}\nvalid {box.new_w}x{box.new_h}, pad L{box.pad_left} T{box.pad_top} "
        f"R{box.pad_right} B{box.pad_bottom}",
        fontsize=8,
    )

    axes[4].imshow(_label_to_rgb(label_512), interpolation="nearest")
    axes[4].add_patch(Rectangle((left, top), right - left - 1, bottom - top - 1,
                                fill=False, edgecolor="cyan", linewidth=1.0, linestyle="--"))
    axes[4].set_title("512 label: red=1, black=0, grey=255 ignore", fontsize=8)

    figure.suptitle(
        f"{row['split']} / {row['label']} / {Path(row['image']).stem}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)

    return {
        "split": row["split"],
        "label": row["label"],
        "image": row["image"],
        "mask": row["mask"],
        "original_width": width,
        "original_height": height,
        "mask_raw_values": raw_values,
        "foreground_px": foreground,
        "foreground_ratio": round(ratio, 8),
        "input_size": input_size,
        "scale": round(box.scale, 6),
        "new_w": box.new_w,
        "new_h": box.new_h,
        "pad": {"left": box.pad_left, "top": box.pad_top, "right": box.pad_right, "bottom": box.pad_bottom},
        "panel": str(out_path),
    }


def render_contact_sheet(samples: list[dict[str, Any]], data_root: Path, out_path: Path,
                         input_size: int, ignore_index: int) -> None:
    """3 columns (original | overlay | 512 label) x N rows, for quick scrolling."""
    rows = len(samples)
    figure, axes = plt.subplots(rows, 3, figsize=(11.0, 3.0 * rows), dpi=110, squeeze=False)
    for index, sample in enumerate(samples):
        with Image.open(data_root / sample["image"]) as image:
            image.load()
            rgb_image = image.convert("RGB")
        mask_path = sample["mask"]
        if mask_path:
            with Image.open(data_root / mask_path) as mask_image:
                mask_image.load()
                binary = geometry.binarize_ids(np.asarray(mask_image))
        else:
            binary = np.zeros((sample["original_height"], sample["original_width"]), dtype=np.uint8)
        box = geometry.compute_letterbox(sample["original_width"], sample["original_height"], input_size)
        label_512 = geometry.letterbox_label(binary, box, ignore_index)

        axes[index][0].imshow(rgb_image)
        axes[index][0].set_title(f"{sample['split']} {sample['label']} {Path(sample['image']).stem}", fontsize=8)
        axes[index][1].imshow(rgb_image)
        axes[index][1].imshow(np.ma.masked_where(~binary.astype(bool), binary),
                              cmap=ListedColormap([(0.95, 0.15, 0.15)]), alpha=0.45, vmin=0, vmax=1)
        axes[index][1].set_title(f"overlay {sample['foreground_ratio'] * 100:.3f}%", fontsize=8)
        axes[index][2].imshow(_label_to_rgb(label_512), interpolation="nearest")
        axes[index][2].set_title("512 label (grey=ignore)", fontsize=8)
        for axis in axes[index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=110)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--normal", type=int, default=None, help="new selection count (default 6); frozen ids take precedence")
    parser.add_argument("--anomaly", type=int, default=None, help="new selection count (default 6); frozen ids take precedence")
    parser.add_argument("--out", default="")
    parser.add_argument("--ids-file", default="", help="optional fixed id list; defaults to logging.fixed_preview_manifest for val")
    parser.add_argument("--write-ids", action="store_true", help="write the selected ids to the ids file")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_root: Path = cfg["data_root"]
    manifest = cfg["paths"][f"data.{args.split}_manifest"]
    rows = read_manifest(manifest)

    ids_path = paths.resolve(args.ids_file) if args.ids_file else cfg["paths"]["logging.fixed_preview_manifest"]
    picked = None
    reused_fixed_ids = False
    if args.split == "val" and ids_path:
        picked = select_with_ids(rows, ids_path, args.normal, args.anomaly)
        if picked is not None:
            reused_fixed_ids = True
            print(f"  selection : fixed ids from {ids_path}")
    if picked is None:
        picked = select_rows(rows, data_root, args.normal if args.normal is not None else 6,
                             args.anomaly if args.anomaly is not None else 6)
        if args.split == "val" and args.write_ids and ids_path:
            write_ids(ids_path, picked, args.split)
            print(f"  selection : computed, ids written to {ids_path}")

    out_dir = paths.resolve(args.out) if args.out else (cfg["paths"]["experiment.output_root"] / "data_preview" / args.split)
    print(f"SegFormer stage 4: preview ({args.split})")
    print(f"  config : {cfg['path']}")
    print(f"  output : {out_dir}")
    print(f"  samples: {len(picked)}")

    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    for row in picked:
        stem = Path(row["image"]).stem
        panel = out_dir / f"{args.split}_{row['label']}_{stem}_panel.png"
        sample = render_sample(row, data_root, cfg["input_size"], cfg["ignore_index"], cfg["image_pad_rgb"], panel)
        samples.append(sample)
        print(f"  rendered {panel.name} fg={sample['foreground_px']} ({sample['foreground_ratio'] * 100:.3f}%)")

    sheet = out_dir / f"{args.split}_contact_sheet.png"
    render_contact_sheet(samples, data_root, sheet, cfg["input_size"], cfg["ignore_index"])

    index = {
        "split": args.split,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "config": str(cfg["path"]),
        "config_sha256": hashlib.sha256(cfg["path"].read_bytes()).hexdigest(),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split_meta_sha256": hashlib.sha256(cfg["paths"]["data.split_meta"].read_bytes()).hexdigest(),
        "data_root": str(data_root),
        "selection_rule": "frozen ids in listed order" if reused_fixed_ids else (
            "normals: evenly spaced positions over the path-sorted list; "
            "anomalies: evenly spaced positions over the foreground-ratio-sorted list"
        ),
        "ids_file": str(ids_path) if (args.split == "val" and ids_path) else None,
        "samples": samples,
        "contact_sheet": str(sheet),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    write_resolved_config(cfg, out_dir / "resolved_config.yaml")
    print(f"  contact sheet: {sheet}")
    print(f"  RESULT: ok ({index['elapsed_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
