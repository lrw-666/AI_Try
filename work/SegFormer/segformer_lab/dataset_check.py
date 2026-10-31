"""Stage 6 acceptance: prove the Dataset returns what the trainer will assume.

Every check below answers a question that would otherwise only fail deep inside
training: are the shapes and dtypes right, is the padding really 255, does a
random flip keep image and mask aligned, was the processor applied exactly once,
does re-reading the same index give the same tensor, and does a real batch still
produce ``[B, 2, 128, 128]`` logits in eval/no_grad mode?

Figures and JSON are written separately: small JSON stays in the repository,
PNG figures go to the resource root.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import data, paths  # noqa: E402
from .config import load_config, write_resolved_config  # noqa: E402
from .modeling import load_all  # noqa: E402

SCHEMA = "segformer-lab/dataset-check/2"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifests(config: dict[str, Any]) -> dict[str, Any]:
    """The manifests used here must be the ones the split protocol produced."""
    from .audit_data import verify_manifests as audit_verify
    return audit_verify(config)


def _label_check(batch: dict[str, Any]) -> dict[str, Any]:
    labels = batch["labels"]
    geom = batch["geometry"][0]
    top, bottom = geom["padding_rows"]
    unique = sorted({int(v) for v in torch.unique(labels)})
    pad_top = labels[:, :top, :]
    pad_bottom = labels[:, bottom:, :]
    valid = labels[:, top:bottom, :]
    per_item = []
    for index, box in enumerate(batch["geometry"]):
        width, height = box["orig_w"], box["orig_h"]
        size = box["input_size"]
        scale = min(size / width, size / height)
        nw, nh = round(width * scale), round(height * scale)
        left, upper = (size - nw) // 2, (size - nh) // 2
        expected = dict(input_size=size, scale=scale, new_w=nw, new_h=nh,
                        pad_left=left, pad_top=upper, pad_right=size-nw-left,
                        pad_bottom=size-nh-upper, valid_area=nw*nh,
                        padding_rows=[upper,upper+nh], ignore_index=255)
        exact = all(box.get(key) == value for key, value in expected.items())
        outside = torch.ones_like(labels[index], dtype=torch.bool)
        outside[upper:upper+nh, left:left+nw] = False
        per_item.append({"index": index, "geometry_exact": exact,
                         "padding_exact": bool((labels[index][outside] == 255).all()),
                         "valid_binary": bool(torch.all((labels[index][~outside] == 0) |
                                                         (labels[index][~outside] == 1)))})
    return {
        "geometry_all_exact": all(item["geometry_exact"] for item in per_item),
        "per_item": per_item,
        "all_padding_exact": all(item["padding_exact"] for item in per_item),
        "all_valid_binary": all(item["valid_binary"] for item in per_item),
        "dtype": str(labels.dtype),
        "shape": list(labels.shape),
        "unique": unique,
        "unique_subset_0_1_255": set(unique) <= {0, 1, 255},
        "padding_rows": [top, bottom],
        "padding_rows_all_255": bool((pad_top == 255).all() and (pad_bottom == 255).all()),
        "padding_pixel_count": int((labels == 255).sum()),
        "valid_shape": list(valid.shape),
        "valid_values": sorted({int(v) for v in torch.unique(valid)}),
        "valid_contains_no_ignore": bool(255 not in {int(v) for v in torch.unique(valid)}),
    }


def _pixel_check(batch: dict[str, Any]) -> dict[str, Any]:
    pixel_values = batch["pixel_values"]
    return {
        "dtype": str(pixel_values.dtype),
        "shape": list(pixel_values.shape),
        "device": str(pixel_values.device),
        "finite": bool(torch.isfinite(pixel_values).all()),
        "min": float(pixel_values.min()),
        "max": float(pixel_values.max()),
        "mean": float(pixel_values.mean()),
        "std": float(pixel_values.std()),
    }


def _render_sample(dataset: data.PCB1Dataset, index: int, out_path: Path) -> dict[str, Any]:
    prepared = dataset.prepared_sample(index)
    pixel_values = dataset.processor(images=prepared["image_512"], return_tensors="pt")["pixel_values"][0]
    label = prepared["label_512"]
    geom = prepared["geometry"]

    with Image.open(dataset.data_root / prepared["image_id"]) as handle:
        handle.load()
        original = handle.convert("RGB")

    figure, axes = plt.subplots(1, 5, figsize=(17.0, 3.9), dpi=140)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    axes[0].imshow(original)
    axes[0].set_title(f"source RGB {geom['orig_w']}x{geom['orig_h']} (unaugmented)", fontsize=9)

    binary = prepared["binary_original"]
    source_binary = np.fliplr(binary) if prepared["flipped"] else binary
    axes[1].imshow(source_binary, cmap=ListedColormap([(0.05, 0.05, 0.05), (0.92, 0.22, 0.22)]),
                   vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"source GT (unaugmented, {int(binary.sum())} px)", fontsize=9)

    axes[2].imshow(prepared["image_512"])
    axes[2].add_patch(Rectangle((geom["pad_left"], geom["pad_top"]), geom["new_w"] - 1, geom["new_h"] - 1,
                                fill=False, edgecolor="cyan", linewidth=1.0, linestyle="--"))
    axes[2].set_title(f"512 input flipped={prepared['flipped']}\nvalid {geom['new_w']}x{geom['new_h']} "
                      f"pads L{geom['pad_left']} T{geom['pad_top']} R{geom['pad_right']} B{geom['pad_bottom']}",
                      fontsize=8)

    axes[3].imshow(label, cmap=ListedColormap([(0.0, 0.0, 0.0), (0.92, 0.22, 0.22), (0.45, 0.45, 0.45)]),
                   vmin=0, vmax=2, interpolation="nearest")
    axes[3].set_title("512 label: red=1, black=0, grey=255", fontsize=9)

    axes[4].imshow(prepared["image_512"])
    axes[4].imshow(np.ma.masked_where(label != 1, label), cmap=ListedColormap([(0.95, 0.15, 0.15)]),
                   alpha=0.5, vmin=0, vmax=1, interpolation="nearest")
    axes[4].set_title("512 overlay", fontsize=9)
    if np.any(binary):
        ys, xs = np.nonzero(binary)
        margin = 35
        x0, x1 = max(0, int(xs.min())-margin), min(binary.shape[1], int(xs.max())+margin+1)
        y0, y1 = max(0, int(ys.min())-margin), min(binary.shape[0], int(ys.max())+margin+1)
        augmented_rgb = np.asarray(original)
        if prepared["flipped"]:
            augmented_rgb = np.fliplr(augmented_rgb)
        inset = axes[4].inset_axes([0.62, 0.61, 0.36, 0.36])
        inset.imshow(augmented_rgb[y0:y1, x0:x1])
        crop = binary[y0:y1, x0:x1]
        inset.imshow(np.ma.masked_where(crop != 1, crop),
                     cmap=ListedColormap([(0.95, 0.15, 0.15)]), alpha=0.5, vmin=0, vmax=1)
        inset.set(xticks=[], yticks=[], title="source-size augmented GT zoom")
        inset.title.set_fontsize(6)

    figure.suptitle(
        f"{dataset.split} / {prepared['label']} / {Path(prepared['image_id']).stem} "
        f"/ flip={prepared['flipped']}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
    geometry_check = _label_check({"labels": torch.from_numpy(label.astype(np.int64))[None],
                                   "geometry": [geom]})
    outside = np.ones(label.shape, dtype=bool)
    outside[geom["pad_top"]:geom["pad_top"]+geom["new_h"],
            geom["pad_left"]:geom["pad_left"]+geom["new_w"]] = False
    checks = {
        "geometry_exact": geometry_check["geometry_all_exact"],
        "padding_exact": geometry_check["all_padding_exact"],
        "valid_binary": geometry_check["all_valid_binary"],
        "image_padding_rgb_exact": bool(np.all(prepared["image_512"][outside] == dataset.pad_rgb)),
        "original_mask_uint8_binary": binary.dtype == np.uint8 and set(np.unique(binary)) <= {0, 1},
        "original_mask_shape": binary.shape == (geom["orig_h"], geom["orig_w"]),
        "pixel_values_float32_finite": pixel_values.dtype == torch.float32 and bool(torch.isfinite(pixel_values).all()),
        "pixel_values_shape": list(pixel_values.shape) == [3, geom["input_size"], geom["input_size"]],
        "normalization_matches_formula": float(np.abs(pixel_values.numpy() -
                dataset.normalization_reference(prepared["image_512"])).max()) <= 1e-6,
    }
    return {
        "image_id": prepared["image_id"],
        "mask_id": prepared["mask_id"],
        "coordinate_space": {"source_rgb": "source_file", "source_gt": "source_file",
                             "input_label_overlay": "augmented_letterbox",
                             "zoom": "augmented_original_resolution"},
        "label": prepared["label"],
        "flipped": prepared["flipped"],
        "geometry": geom,
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values_dtype": str(pixel_values.dtype),
        "label_unique": sorted({int(v) for v in np.unique(label)}),
        "foreground_px_512": int((label == 1).sum()),
        "figure": str(out_path),
        "figure_sha256": _sha256(out_path),
        "checks": checks,
        "source_image_sha256": _sha256(dataset.data_root / prepared["image_id"]),
        "source_mask_sha256": _sha256(dataset.data_root / prepared["mask_id"]) if prepared["mask_id"] else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--report", default="reports/dataset_check.json")
    parser.add_argument("--figure-dir", default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-model", action="store_true", help="skip the eval forward (dataset checks only)")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    config = load_config(args.config)
    figure_dir = Path(args.figure_dir).expanduser().resolve() if args.figure_dir else (paths.RUNS_DIR / "dataset_check")

    print("SegFormer stage 6: dataset check")
    print(f"  config : {config['path']}")
    print(f"  data   : {config['data_root']}")
    print(f"  figures: {figure_dir}")

    identity = verify_manifests(config)
    if not identity["ok"]:
        raise SystemExit("manifest/group verification failed; refusing to read samples")
    print(f"  manifests verified: {identity['ok']} "
          f"(train {identity['manifests']['train']['sha256'][:12]}…)")

    from .download_model import download
    model_source_path = paths.PROJECT_ROOT / "reports" / "model_source.json"
    expected_model = json.loads(model_source_path.read_text(encoding="utf-8"))
    expected_model = expected_model.get("source", expected_model)
    model_identity = download(expected_model["model_id"], config["paths"]["model.pretrained_dir"],
                              revision=expected_model["revision"], verify_only=True,
                              source_record=model_source_path)
    if not model_identity["ok"]:
        raise SystemExit(f"snapshot verification failed: {model_identity['problems']}")
    data.close_processor_cache()
    processor = data.build_processor(str(config["paths"]["model.pretrained_dir"]))
    train_ds = data.PCB1Dataset(config, "train", processor, flip="random", seed=args.seed)
    val_ds = data.PCB1Dataset(config, "val", processor)
    test_ds = data.PCB1Dataset(config, "test", processor)
    train_plain = data.PCB1Dataset(config, "train", processor, flip="none")
    train_force = data.PCB1Dataset(config, "train", processor, flip="force")
    print(f"  sizes  : train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")
    print(f"  Dataset processor factory misses: {data.processor_cache_info().misses}")

    train_loader = data.build_dataloader(train_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    batch = next(iter(train_loader))
    preview_path = config["paths"]["logging.fixed_preview_manifest"]
    preview_ids = [line.strip() for line in preview_path.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.strip().startswith("#")]
    by_id = {row["image"]: i for i, row in enumerate(val_ds.rows)}
    if len(preview_ids) != 6 or len(set(preview_ids)) != 6 or any(item not in by_id for item in preview_ids):
        raise SystemExit("fixed validation preview must contain six distinct validation ids")
    preview_indices = [by_id[item] for item in preview_ids]
    normal_index = next(i for i in preview_indices if val_ds.rows[i]["label"] == "normal")
    anomaly_index = next(i for i in preview_indices if val_ds.rows[i]["label"] == "anomaly")
    val_batch = data.collate_pcb1([val_ds[normal_index], val_ds[anomaly_index]])

    pixel = _pixel_check(batch)
    label = _label_check(batch)
    geom = batch["geometry"][0]

    # -- repeated reads ------------------------------------------------------
    first = train_plain[0]
    second = train_plain[0]
    repeat = {
        "pixel_values_equal": bool(torch.equal(first["pixel_values"], second["pixel_values"])),
        "labels_equal": bool(torch.equal(first["labels"], second["labels"])),
        "original_mask_equal": bool(np.array_equal(first["original_mask"], second["original_mask"])),
    }
    # Two independent passes over the same loader must agree as well.
    pass_a = next(iter(data.build_dataloader(train_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed)))
    pass_b = next(iter(data.build_dataloader(train_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed)))
    repeat["loader_pass_equal"] = bool(torch.equal(pass_a["pixel_values"], pass_b["pixel_values"]))
    val_first, val_second = val_ds[anomaly_index], val_ds[anomaly_index]
    repeat["val_pixel_values_equal"] = torch.equal(val_first["pixel_values"], val_second["pixel_values"])
    repeat["val_labels_equal"] = torch.equal(val_first["labels"], val_second["labels"])
    repeat["val_original_mask_equal"] = np.array_equal(val_first["original_mask"], val_second["original_mask"])

    # -- synchronized flip ---------------------------------------------------
    plain = train_plain.prepared_sample(0)
    forced = train_force.prepared_sample(0)
    pixel_plain = train_plain[0]["pixel_values"]
    pixel_forced = train_force[0]["pixel_values"]
    # torch.fliplr flips the second dimension, which is the *height* of a CHW
    # tensor; a horizontal flip means dim=2 (width). Using fliplr here would
    # compare a vertical mirror and silently look "almost right" on symmetric
    # images.
    flipped_pixels = torch.flip(pixel_plain, dims=(2,))
    image_lsb = int(np.abs(plain["image_512"].astype(np.int16) - np.fliplr(forced["image_512"]).astype(np.int16)).max())
    flip_check = {
        "plain_flipped_flag": plain["flipped"],
        "force_flipped_flag": forced["flipped"],
        "label_flip_equal": bool(np.array_equal(forced["label_512"], np.fliplr(plain["label_512"]))),
        "image_uint8_max_abs_diff": image_lsb,
        "image_flip_max_abs_diff": float((flipped_pixels - pixel_forced).abs().max()),
        "original_mask_flip_equal": bool(
            np.array_equal(forced["binary_original"], np.fliplr(plain["binary_original"]))
        ),
        "val_random_rejected": False,
        "uint8_threshold_lsb": 1,
        "tensor_threshold": float(1 / 255 / min(processor.image_std) + 1e-6),
        "label_comparison": "exact pixel equality",
        "sample_image_id": plain["image_id"],
        "sample_mask_id": plain["mask_id"],
    }
    try:
        data.PCB1Dataset(config, "val", processor, flip="random")
    except ValueError:
        flip_check["val_random_rejected"] = True
    for split in ("val", "test"):
        for mode in ("random", "force"):
            key = f"{split}_{mode}_rejected"
            flip_check[key] = False
            try:
                data.PCB1Dataset(config, split, processor, flip=mode)
            except ValueError:
                flip_check[key] = True

    # -- processor applied exactly once -------------------------------------
    normalization = {
        "max_abs_diff": float(
            np.abs(pixel_plain.numpy() - train_plain.normalization_reference(plain["image_512"])).max()
        ),
        "image_mean": list(map(float, np.asarray(processor.image_mean).ravel())),
        "image_std": list(map(float, np.asarray(processor.image_std).ravel())),
        "rescale_factor": float(getattr(processor, "rescale_factor", 0.0)),
        "do_resize": bool(processor.do_resize),
        "do_reduce_labels": bool(processor.do_reduce_labels),
    }

    # -- normal / anomaly label semantics -----------------------------------
    normal_sample = val_ds.prepared_sample(normal_index)
    anomaly_sample = val_ds.prepared_sample(anomaly_index)
    top, bottom = normal_sample["geometry"]["padding_rows"]
    normal_valid = normal_sample["label_512"][top:bottom, :]
    anomaly_valid = anomaly_sample["label_512"][top:bottom, :]
    semantics = {
        "normal_image_id": normal_sample["image_id"],
        "normal_valid_all_zero": bool((normal_valid == 0).all()),
        "normal_padding_all_255": bool(
            (normal_sample["label_512"][:top, :] == 255).all()
            and (normal_sample["label_512"][bottom:, :] == 255).all()
        ),
        "anomaly_image_id": anomaly_sample["image_id"],
        "anomaly_valid_foreground_px": int((anomaly_valid == 1).sum()),
        "anomaly_valid_nonempty": bool((anomaly_valid == 1).sum() > 0),
        "anomaly_raw_ids_before_merge": sorted({int(v) for v in np.unique(
            np.asarray(Image.open(val_ds.data_root / anomaly_sample["mask_id"]))
        )}),
    }

    # -- original mask kept as a per-sample list ----------------------------
    original = {
        "type": type(batch["original_mask"]).__name__,
        "count": len(batch["original_mask"]),
        "shapes": [list(item.shape) for item in batch["original_mask"]],
        "dtypes": sorted({str(item.dtype) for item in batch["original_mask"]}),
        "values": sorted({int(v) for item in batch["original_mask"] for v in np.unique(item)}),
        "geometry_is_list": isinstance(batch["geometry"], list),
        "image_id_is_list": isinstance(batch["image_id"], list),
        "mask_id_is_list": isinstance(batch["mask_id"], list),
        "flipped_is_list": isinstance(batch["flipped"], list),
    }

    # -- no CUDA / no model inside __getitem__ ------------------------------
    source = inspect.getsource(data.PCB1Dataset.__getitem__) + inspect.getsource(data.PCB1Dataset.prepared_sample)
    isolation = {
        "getitem_device": str(first["pixel_values"].device),
        "getitem_mentions_cuda": "cuda" in source.lower(),
        "getitem_mentions_model": "load_all" in source or "from_pretrained" in source,
        "processor_in_getitem": "self.processor(" in inspect.getsource(data.PCB1Dataset.__getitem__),
        "processor_builds": data.processor_cache_info().misses,
        "processor_factory_misses": data.processor_cache_info().misses,
        "processor_factory_scope": "Dataset build_processor factory only; excludes modeling.load_all",
        "processor_shared_identity": all(ds.processor is processor for ds in
                                         (train_ds, val_ds, test_ds, train_plain, train_force)),
    }

    # -- one real eval forward ----------------------------------------------
    forward: dict[str, Any] = {"status": "skipped", "reason": "--skip-model"}
    if not args.skip_model:
        loaded = load_all(config["paths"]["model.pretrained_dir"], num_labels=config["num_labels"],
                          ignore_index=config["ignore_index"])
        model = loaded.model
        if not loaded.encoder_check["ok"]:
            raise SystemExit("encoder check failed; refusing to run the forward")
        device = torch.device("cuda")
        model.to(device).eval()
        with torch.no_grad():
            pixel_values = val_batch["pixel_values"].to(device)
            labels = val_batch["labels"].to(device)
            logits = model(pixel_values=pixel_values).logits
        forward = {
            "status": "executed",
            "mode": "model.eval() + torch.no_grad()",
            "optimizer": None,
            "batch_size": int(pixel_values.shape[0]),
            "pixel_values_shape": list(pixel_values.shape),
            "labels_shape": list(labels.shape),
            "label_unique": sorted({int(v) for v in torch.unique(labels)}),
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "device": torch.cuda.get_device_name(0),
            "image_ids": val_batch["image_id"],
            "model_training": bool(model.training),
            "logits_requires_grad": bool(logits.requires_grad),
        }
        print(f"  eval forward : logits {forward['logits_shape']} finite={forward['logits_finite']}")

    # -- figures -------------------------------------------------------------
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    train_normal_index = next(i for i, row in enumerate(train_ds.rows) if row["label"] == "normal")
    train_anomaly_index = next(i for i, row in enumerate(train_ds.rows) if row["label"] == "anomaly")
    for tag, dataset, index in (
        ("train_normal", train_plain, train_normal_index),
        ("train_anomaly", train_plain, train_anomaly_index),
        ("train_anomaly_forced_flip", train_force, train_anomaly_index),
    ):
        out_path = figure_dir / f"{tag}_{Path(dataset.rows[index]['image']).stem}.png"
        figures.append(_render_sample(dataset, index, out_path))
        print(f"  figure       : {out_path.name}")
    validation_previews = []
    for index in preview_indices:
        row = val_ds.rows[index]
        path = figure_dir / f"val_{row['label']}_{Path(row['image']).stem}.png"
        validation_previews.append(_render_sample(val_ds, index, path))

    provenance_paths = {config["path"], preview_path, model_source_path,
                        config["paths"]["model.pretrained_dir"] / "source.json",
                        config["paths"]["data.split_meta"]}
    provenance_paths.update(Path(__file__).parent.glob("*.py"))
    provenance_paths.update(config["paths"]["model.pretrained_dir"] / item["path"]
                            for item in model_identity["source"]["files"])
    provenance_paths.update(config["paths"][f"data.{name}_manifest"] for name in data.SPLITS)
    for key in ("official_csv", "reviewed_groups"):
        if identity.get(key, {}).get("path"):
            provenance_paths.add(Path(identity[key]["path"]))
    if config["paths"].get("data.train_ids_file"):
        provenance_paths.add(config["paths"]["data.train_ids_file"])
    selected = []
    selected_rows = list(train_ds.rows[:args.batch_size]) + [train_plain.rows[0]]
    selected_rows += [train_plain.rows[train_normal_index], train_plain.rows[train_anomaly_index]]
    selected_rows += [val_ds.rows[i] for i in preview_indices]
    for row in selected_rows:
        for kind in ("image", "mask"):
            if row[kind]:
                provenance_paths.add(config["data_root"] / row[kind])
        selected.append({"image_id": row["image"], "mask_id": row["mask"], "label": row["label"]})
    provenance_paths.update(Path(item["figure"]) for item in figures + validation_previews)
    provenance = {"selected_samples": selected, "files": [
        {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(provenance_paths, key=str)]}

    checks = {
        "manifests_match_split_meta": identity["ok"],
        "pixel_values_dtype_float32": pixel["dtype"] == "torch.float32",
        "pixel_values_finite": pixel["finite"],
        "pixel_values_shape_b3_512": pixel["shape"][1:] == [3, 512, 512] and pixel["shape"][0] == args.batch_size,
        "labels_dtype_int64": label["dtype"] == "torch.int64",
        "labels_shape_b_512": label["shape"][1:] == [512, 512] and label["shape"][0] == args.batch_size,
        "labels_subset_0_1_255": label["unique_subset_0_1_255"],
        "padding_rows_all_255": label["padding_rows_all_255"],
        "valid_area_512x390": [geom["new_w"], geom["new_h"]] == [512, 390],
        "valid_has_no_ignore": label["valid_contains_no_ignore"],
        "geometry_all_exact": label["geometry_all_exact"],
        "all_padding_exact": label["all_padding_exact"],
        "all_valid_binary": label["all_valid_binary"],
        "repeat_read_identical": all(repeat.values()),
        "flip_label_aligned": flip_check["label_flip_equal"],
        "flip_image_aligned": flip_check["image_uint8_max_abs_diff"] <= 1,
        "flip_tensor_aligned": flip_check["image_flip_max_abs_diff"] <= flip_check["tensor_threshold"],
        "flip_flags_correct": not plain["flipped"] and forced["flipped"],
        "flip_original_mask_aligned": flip_check["original_mask_flip_equal"],
        "val_random_rejected": flip_check["val_random_rejected"],
        "evaluation_augmentation_rejected": all(flip_check[f"{split}_{mode}_rejected"]
                                                for split in ("val", "test") for mode in ("random", "force")),
        "normalization_matches_formula": normalization["max_abs_diff"] <= 1e-6,
        "normal_valid_all_zero": semantics["normal_valid_all_zero"],
        "normal_padding_all_255": semantics["normal_padding_all_255"],
        "anomaly_valid_nonempty": semantics["anomaly_valid_nonempty"],
        "original_mask_is_list": original["type"] == "list" and original["count"] == args.batch_size,
        "original_mask_original_size": all(
            shape == [batch["geometry"][i]["orig_h"], batch["geometry"][i]["orig_w"]]
            for i, shape in enumerate(original["shapes"])
        ),
        "original_mask_binary": set(original["values"]) <= {0, 1},
        "original_mask_uint8": original["dtypes"] == ["uint8"],
        "metadata_is_list": all(original[key] for key in
                                ("geometry_is_list", "image_id_is_list", "mask_id_is_list", "flipped_is_list")),
        "getitem_cpu_tensor": isolation["getitem_device"] == "cpu",
        "getitem_no_cuda": not isolation["getitem_mentions_cuda"],
        "getitem_no_model_load": not isolation["getitem_mentions_model"],
        "processor_built_once": isolation["processor_factory_misses"] == 1 and isolation["processor_shared_identity"],
        "eval_forward_shape": (None if args.skip_model else forward["logits_shape"] == [2, 2, 128, 128]),
        "eval_forward_finite": (None if args.skip_model else forward["logits_finite"]),
        "eval_forward_eval_no_grad": (None if args.skip_model else
                                       not forward["model_training"] and not forward["logits_requires_grad"]),
    }
    validation_checks = [_label_check(val_batch)]
    val_pixel, val_labels = _pixel_check(val_batch), _label_check(val_batch)
    checks["val_batch_contract"] = (val_pixel["finite"] and val_pixel["dtype"] == "torch.float32" and
        val_pixel["shape"] == [2, 3, 512, 512] and val_labels["dtype"] == "torch.int64" and
        val_labels["shape"] == [2, 512, 512] and
        all(item["geometry_all_exact"] and item["all_padding_exact"] and item["all_valid_binary"]
            for item in validation_checks))
    checks["diagnostic_sample_contracts"] = all(all(item["checks"].values())
                                              for item in figures + validation_previews)
    data_checks_ok = all(value for key, value in checks.items() if not key.startswith("eval_forward"))
    full_stage_verified = not args.skip_model and all(checks.values())

    report = {
        "schema": SCHEMA,
        "scope": "dataset_only" if args.skip_model else "full_stage",
        "data_checks_ok": data_checks_ok,
        "full_stage_verified": full_stage_verified,
        "model_identity": model_identity,
        "provenance": provenance,
        "validation_previews": validation_previews,
        "validation_batch": {"image_ids": val_batch["image_id"], "pixel_values": val_pixel,
                             "labels": val_labels},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "config": str(config["path"]),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "workers": int(config["raw"]["data"].get("workers", 0)),
        "identity": identity,
        "dataset_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "pixel_values": pixel,
        "labels": label,
        "geometry_first_sample": geom,
        "repeat_read": repeat,
        "flip": flip_check,
        "normalization": normalization,
        "label_semantics": semantics,
        "original_mask": original,
        "isolation": isolation,
        "eval_forward": forward,
        "figures": figures,
        "checks": checks,
        "ok": full_stage_verified,
        "elapsed_seconds": None,
    }
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)

    report_path = paths.ensure_parent(args.report)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_resolved_config(config, report_path.parent / "resolved_config.yaml")

    print("")
    for name, value in checks.items():
        print(f"  {'SKIPPED' if value is None else 'PASS' if value else 'FAIL'} — {name}")
    print(f"  report       : {report_path}")
    print(f"  elapsed      : {report['elapsed_seconds']} s")
    if args.skip_model and data_checks_ok:
        print("  RESULT: DATASET ONLY — data checks passed; full stage NOT verified")
        return 0
    if not report["ok"]:
        failed = [name for name, value in checks.items() if value is False]
        print(f"  RESULT: FAILED {failed}", file=sys.stderr)
        return 1
    print("  RESULT: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
