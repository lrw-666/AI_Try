"""PCB1 Dataset, collate function and DataLoader for the SegFormer baseline.

Design in one paragraph
-----------------------
A sample is built from the manifest row only: open the image, build its mask
(zero mask at original size for a normal image; ``>0 -> 1`` merge of the stored
defect ids for an anomaly, done **before** padding), optionally flip image and
mask together, then letterbox both with the shared geometry module. The uint8
RGB canvas goes to the image processor exactly once (rescale + normalize) with
resizing and label reduction disabled; the label becomes ``int64`` on its own
path. Nothing here touches CUDA or the model, so the same class can be imported
by a DataLoader worker.

Why the merge happens before padding
------------------------------------
After letterboxing, the padding is 255. If the merge ran afterwards, ``>0``
would turn every padding pixel into class 1. The order encoded here is:
raw ids -> 0/1 -> flip -> nearest resize -> pad 255.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import geometry, paths
from .config import load_config

SPLITS = ("train", "val", "test")
FLIP_MODES = ("none", "random", "force")


# --------------------------------------------------------------------- helpers ----
def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _processor_statistics(processor: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate the preprocessing contract, including injected processors."""
    for flag, expected in (("do_resize", False), ("do_reduce_labels", False),
                           ("do_rescale", True), ("do_normalize", True)):
        actual = getattr(processor, flag, None)
        if actual is None or actual != expected:
            raise ValueError(f"processor.{flag} is {actual}, expected {expected}")
    statistics = []
    for name in ("image_mean", "image_std"):
        try:
            array = np.asarray(getattr(processor, name, None), dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(f"processor.{name} must contain three finite values") from error
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(f"processor.{name} must contain three finite values")
        if name == "image_std" and not (array > 0).all():
            raise ValueError("processor.image_std must be strictly positive")
        statistics.append(array.reshape(3, 1, 1))
    return statistics[0], statistics[1]


@lru_cache(maxsize=4)
def build_processor(model_dir: str) -> Any:
    """Build (and cache) the image processor once per snapshot directory.

    ``do_resize=False`` keeps our letterbox as the only resize step;
    ``do_reduce_labels=False`` keeps background 0. Both are re-asserted here so
    a future config edit cannot silently re-enable them.
    """
    from transformers import SegformerImageProcessor

    processor = SegformerImageProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        do_resize=False,
        do_reduce_labels=False,
        do_rescale=True,
        do_normalize=True,
    )
    _processor_statistics(processor)
    return processor


def flip_decision(mode: str, seed: int, epoch: int, index: int, probability: float) -> bool:
    """Deterministic per-sample flip decision.

    Deterministic means: the same (seed, epoch, index) always gives the same
    answer, so a rerun of the same epoch reproduces the same augmentation, while
    a different epoch gets a different pattern.
    """
    if mode == "none":
        return False
    if mode == "force":
        return True
    if mode != "random":
        raise ValueError(f"unknown flip mode: {mode!r}")
    rng = np.random.default_rng([int(seed), int(epoch), int(index)])
    return bool(rng.random() < float(probability))


# --------------------------------------------------------------------- dataset ----
class PCB1Dataset(Dataset):
    """One VisA PCB1 sample per item.

    Parameters
    ----------
    config:
        Result of :func:`segformer_lab.config.load_config`.
    split:
        ``train`` / ``val`` / ``test``; selects the manifest.
    flip:
        ``none`` (no augmentation), ``random`` (training) or ``force``
        (verification path that must produce a flipped sample). Random flipping
        and forced flipping are refused for val/test.
    seed:
        Defaults to ``config['raw']['experiment']['seed']``, or 42 when absent.
    epoch:
        Initial augmentation epoch; callers must use :meth:`set_epoch` before
        each new epoch. Iterating a DataLoader does not advance it.
    ids_file:
        Optional text file with one manifest ``image`` path per line; used by
        the smoke/overfit configurations. Every id must belong to the split.
        Training defaults to config's resolved ``data.train_ids_file`` when
        omitted; val/test never inherit the training subset.
    return_original_mask:
        Keep the augmented-direction, original-resolution binary GT. The
        collate function keeps it as a per-sample list; it is never stacked.
    """

    def __init__(
        self,
        config: dict[str, Any],
        split: str,
        processor: Any | None = None,
        *,
        flip: str | None = None,
        seed: int | None = None,
        epoch: int = 0,
        ids_file: str | Path | None = None,
        return_original_mask: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if flip is None:
            flip = "random" if split == "train" else "none"
        if flip not in FLIP_MODES:
            raise ValueError(f"flip must be one of {FLIP_MODES}, got {flip!r}")
        if split != "train" and flip != "none":
            raise ValueError(f"augmentation is disabled for split={split!r}; use flip='none'")

        self.config = config
        self.split = split
        self.flip_mode = flip
        if seed is None:
            seed = config["raw"].get("experiment", {}).get("seed", 42)
        self.seed = _nonnegative_int(seed, "seed")
        self.epoch = _nonnegative_int(epoch, "epoch")
        self.return_original_mask = bool(return_original_mask)

        self.data_root: Path = config["data_root"]
        self.input_size: int = config["input_size"]
        self.ignore_index: int = config["ignore_index"]
        self.pad_rgb: tuple[int, int, int] = config["image_pad_rgb"]
        self.flip_probability: float = float(config["raw"]["data"].get("horizontal_flip_p", 0.5))
        if not 0.0 <= self.flip_probability <= 1.0:
            raise ValueError("horizontal_flip_p must be finite and in [0, 1]")

        manifest_path = config["paths"][f"data.{split}_manifest"]
        rows = read_manifest(manifest_path)
        if ids_file is None and split == "train":
            ids_file = config["paths"].get("data.train_ids_file")
        if ids_file is not None:
            rows = self._filter_by_ids(rows, Path(ids_file))
        self.rows: list[dict[str, str]] = rows

        self.processor = processor if processor is not None else build_processor(str(config["paths"]["model.pretrained_dir"]))
        # A guard for the "never load the model here" rule: the dataset only
        # needs the processor's normalisation constants, read once.
        self.image_mean, self.image_std = _processor_statistics(self.processor)

    # -- manifest filtering -------------------------------------------------
    def _filter_by_ids(self, rows: list[dict[str, str]], ids_path: Path) -> list[dict[str, str]]:
        wanted = []
        for line in ids_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                wanted.append(line)
        if not wanted:
            raise ValueError(f"ids file must select at least one image: {ids_path}")
        by_image = {row["image"]: row for row in rows}
        missing = [item for item in wanted if item not in by_image]
        if missing:
            raise ValueError(f"{len(missing)} id(s) from {ids_path} are not in split {self.split!r}: {missing[:5]}")
        if len(set(wanted)) != len(wanted):
            raise ValueError(f"duplicate ids in {ids_path}")
        return [by_image[item] for item in wanted]

    # -- public API ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation pattern for a new epoch (training only)."""
        self.epoch = _nonnegative_int(epoch, "epoch")

    def image_ids(self) -> list[str]:
        return [row["image"] for row in self.rows]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.prepared_sample(index)
        pixel_values = self.processor(images=sample["image_512"], return_tensors="pt")["pixel_values"][0]
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.float()

        item: dict[str, Any] = {
            "pixel_values": pixel_values,
            "labels": torch.from_numpy(sample["label_512"].astype(np.int64)),
            "image_id": sample["image_id"],
            "mask_id": sample["mask_id"],
            "flipped": sample["flipped"],
            "geometry": sample["geometry"],
        }
        if self.return_original_mask:
            item["original_mask"] = sample["binary_original"]
        return item

    # -- diagnostics --------------------------------------------------------
    def prepared_sample(self, index: int) -> dict[str, Any]:
        """Everything before tensor conversion, as NumPy arrays.

        Kept public so the stage-06 check and the figures can verify the exact
        same code path instead of re-implementing flip/letterbox logic.
        """
        row = self.rows[index]
        image_path = self.data_root / row["image"]
        with Image.open(image_path) as handle:
            handle.load()
            image = handle.convert("RGB")
            width, height = image.size

        if row["label"] == "normal":
            if row["mask"].strip():
                raise ValueError(f"normal row must not carry a mask: {row['image']}")
            binary = np.zeros((height, width), dtype=np.uint8)
            mask_id = ""
        else:
            if not row["mask"].strip():
                raise ValueError(f"anomaly row without mask: {row['image']}")
            with Image.open(self.data_root / row["mask"]) as handle:
                handle.load()
                raw = np.asarray(handle)
            if raw.ndim != 2:
                raise ValueError(f"mask is not single channel: {row['mask']} shape={raw.shape}")
            if (raw.shape[1], raw.shape[0]) != (width, height):
                raise ValueError(
                    f"mask size {raw.shape[1]}x{raw.shape[0]} does not match image {width}x{height}: {row['mask']}"
                )
            # Merge before any padding: 1..4 -> 1, background stays 0.
            binary = geometry.binarize_ids(raw)
            mask_id = row["mask"]

        flipped = flip_decision(self.flip_mode, self.seed, self.epoch, index, self.flip_probability)
        if flipped:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            binary = np.ascontiguousarray(binary[:, ::-1])

        box = geometry.compute_letterbox(width, height, self.input_size)
        image_512 = geometry.letterbox_image(image, box, self.pad_rgb)
        label_512 = geometry.letterbox_label(binary, box, self.ignore_index)

        return {
            "image_id": row["image"],
            "mask_id": mask_id,
            "label": row["label"],
            "flipped": bool(flipped),
            "image_512": image_512,
            "label_512": label_512,
            "binary_original": binary,
            "geometry": {
                "orig_w": int(width),
                "orig_h": int(height),
                "input_size": int(self.input_size),
                "scale": float(box.scale),
                "new_w": int(box.new_w),
                "new_h": int(box.new_h),
                "pad_left": int(box.pad_left),
                "pad_top": int(box.pad_top),
                "pad_right": int(box.pad_right),
                "pad_bottom": int(box.pad_bottom),
                "valid_area": int(box.valid_area),
                "padding_rows": [int(box.pad_top), int(box.pad_top + box.new_h)],
                "ignore_index": int(self.ignore_index),
            },
        }

    def normalization_reference(self, image_512: np.ndarray) -> np.ndarray:
        """Expected processor output for a uint8 HWC canvas.

        ``(x / 255 - mean) / std`` in CHW float32; used by the acceptance check
        to prove the processor was applied once, not twice.
        """
        array = image_512.astype(np.float32) / 255.0
        chw = np.transpose(array, (2, 0, 1))
        return ((chw - self.image_mean) / self.image_std).astype(np.float32)


# --------------------------------------------------------------------- collate ----
def collate_pcb1(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors, keep everything original-resolution as a Python list.

    ``original_mask`` deliberately stays a list: masks are original size and a
    future category may have a different size, so stacking them would fail (or
    silently pad). The same applies to geometry dicts and image ids.
    """
    if not batch:
        raise ValueError("empty batch")
    has_original = ["original_mask" in item for item in batch]
    if any(has_original) and not all(has_original):
        raise ValueError("batch must consistently include or omit original_mask")
    out: dict[str, Any] = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "image_id": [item["image_id"] for item in batch],
        "mask_id": [item["mask_id"] for item in batch],
        "flipped": [bool(item["flipped"]) for item in batch],
        "geometry": [item["geometry"] for item in batch],
    }
    if all(has_original):
        out["original_mask"] = [item["original_mask"] for item in batch]
    return out


# ------------------------------------------------------------------ dataloader ----
def build_dataloader(
    dataset: PCB1Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
    num_workers: int | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """Build a CPU loader; omitted shuffle seed follows the Dataset's seed."""
    generator = torch.Generator()
    generator.manual_seed(_nonnegative_int(dataset.seed if seed is None else seed, "seed"))
    workers = dataset.config["raw"]["data"].get("workers", 0) if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        collate_fn=collate_pcb1,
        drop_last=bool(drop_last),
        pin_memory=False,
        generator=generator,
    )


def iterate_batches(loader: DataLoader) -> Iterator[dict[str, Any]]:
    """Thin alias that documents the intended usage in stage 06 checks."""
    for batch in loader:
        yield batch


def close_processor_cache() -> None:
    """Drop the cached processor (useful in tests that patch the snapshot)."""
    build_processor.cache_clear()


def processor_cache_info() -> Any:
    return build_processor.cache_info()


def load_dataset_from_path(config_path: str | Path, split: str, **kwargs: Any) -> PCB1Dataset:
    """Convenience for scripts: load the config first, then the dataset."""
    config = load_config(config_path)
    return PCB1Dataset(config, split, **kwargs)


def manifest_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Hash the manifests used by this dataset, for the acceptance record."""
    import hashlib

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    identity = {}
    for split in SPLITS:
        path = config["paths"][f"data.{split}_manifest"]
        identity[split] = {"path": str(path), "sha256": sha256(path), "rows": len(read_manifest(path))}
    identity["split_meta"] = {
        "path": str(config["paths"]["data.split_meta"]),
        "sha256": sha256(config["paths"]["data.split_meta"]),
    }
    return identity


def paths_of(config_path: str | Path) -> Path:
    """Small helper so callers do not re-import paths."""
    return paths.resolve(config_path)
