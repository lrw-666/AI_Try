"""Geometry helpers shared by the audit, the preview and (later) the Dataset.

The single most important convention here is the letterbox transform:

* the whole image is scaled by one factor so the *longer* side fits
  ``input_size``; nothing is cropped,
* the scaled image is centred and padded with ``image_pad_rgb``,
* labels are resized with **nearest neighbour** and padded with
  ``ignore_index`` (255), so padding is never learned as background.

Keeping this in one module means the audit's "what will the network see"
calculation and the training Dataset cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Letterbox:
    """Result of fitting a ``width x height`` image into ``input_size``."""

    width: int
    height: int
    input_size: int
    scale: float
    new_w: int
    new_h: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    @property
    def valid_area(self) -> int:
        return self.new_w * self.new_h

    @property
    def padding_pixels(self) -> int:
        return self.input_size * self.input_size - self.valid_area

    @property
    def crop_box(self) -> tuple[int, int, int, int]:
        """Slice of the 512 canvas that holds real pixels: (top, left, bottom, right)."""
        return (self.pad_top, self.pad_left, self.pad_top + self.new_h, self.pad_left + self.new_w)


def compute_letterbox(width: int, height: int, input_size: int) -> Letterbox:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    if input_size <= 0:
        raise ValueError(f"invalid input_size: {input_size}")

    scale = min(input_size / width, input_size / height)
    new_w = max(1, min(input_size, int(round(width * scale))))
    new_h = max(1, min(input_size, int(round(height * scale))))
    pad_w = input_size - new_w
    pad_h = input_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    return Letterbox(
        width=width,
        height=height,
        input_size=input_size,
        scale=scale,
        new_w=new_w,
        new_h=new_h,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_w - pad_left,
        pad_bottom=pad_h - pad_top,
    )


def binarize_ids(mask: np.ndarray) -> np.ndarray:
    """Merge every non-zero defect id into class 1 (background stays 0).

    The official masks store ids such as 1..4. Training uses two classes, so the
    merge happens in memory; the mask files on disk are never rewritten.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    return (mask > 0).astype(np.uint8)


def letterbox_image(image: Image.Image, box: Letterbox, pad_rgb: tuple[int, int, int]) -> np.ndarray:
    """Scale an RGB image and centre-pad it; returns uint8 HxWx3."""
    resized = image.convert("RGB").resize((box.new_w, box.new_h), Image.BILINEAR)
    canvas = np.empty((box.input_size, box.input_size, 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(pad_rgb, dtype=np.uint8)
    canvas[box.pad_top:box.pad_top + box.new_h, box.pad_left:box.pad_left + box.new_w] = np.asarray(resized)
    return canvas


def letterbox_label(binary: np.ndarray, box: Letterbox, ignore_index: int = 255) -> np.ndarray:
    """Nearest-neighbour resize a 0/1 label and pad it with ignore_index."""
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    resized = Image.fromarray(binary, mode="L").resize((box.new_w, box.new_h), Image.NEAREST)
    canvas = np.full((box.input_size, box.input_size), int(ignore_index), dtype=np.uint8)
    canvas[box.pad_top:box.pad_top + box.new_h, box.pad_left:box.pad_left + box.new_w] = np.asarray(resized)
    return canvas


def _label_components(binary: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Label 4-connected foreground; area list entry i belongs to label i + 1.

    Implemented with row runs plus union-find: for a 1404x1070 mask with a
    handful of defects this is milliseconds, and it needs no SciPy/OpenCV.
    """
    if binary.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {binary.shape}")
    mask = binary.astype(bool, copy=False)
    height, width = mask.shape

    parent: list[int] = []
    run_length: list[int] = []
    runs: list[tuple[int, int, int, int]] = []

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    previous: list[tuple[int, int, int]] = []
    for y in range(height):
        row = mask[y]
        if not row.any():
            previous = []
            continue
        padded = np.concatenate(([0], row.astype(np.int8, copy=False), [0]))
        delta = np.diff(padded)
        starts = np.flatnonzero(delta == 1)
        ends = np.flatnonzero(delta == -1)
        current: list[tuple[int, int, int]] = []
        cursor = 0
        for start, end in zip(starts, ends):
            node = len(parent)
            parent.append(node)
            run_length.append(int(end - start))
            while cursor < len(previous) and previous[cursor][1] <= start:
                cursor += 1
            probe = cursor
            while probe < len(previous) and previous[probe][0] < end:
                union(node, previous[probe][2])
                probe += 1
            current.append((int(start), int(end), node))
            runs.append((y, int(start), int(end), node))
        previous = current

    labels = np.zeros(mask.shape, dtype=np.int32)
    if not parent:
        return labels, []

    totals: dict[int, int] = {}
    for node, length in enumerate(run_length):
        root = find(node)
        totals[root] = totals.get(root, 0) + length
    ids = {root: index for index, root in enumerate(totals, start=1)}
    for y, start, end, node in runs:
        labels[y, start:end] = ids[find(node)]
    return labels, list(totals.values())


def count_components(binary: np.ndarray) -> tuple[int, int]:
    """Count 4-connected foreground components and return (count, largest area)."""
    _, areas = _label_components(binary)
    return len(areas), max(areas, default=0)


def component_retention(binary: np.ndarray, box: Letterbox) -> dict[str, Any]:
    """Track each original 4-connected component through the label's NEAREST resize.

    Component identities use int32, independently of the 0/1/255 training label.
    Sampling this identity map with the same Pillow NEAREST coordinates tells us
    whether at least one original pixel survives. Comparing counts alone cannot
    answer that: components can merge, split, or disappear at the same time.
    The valid box only is measured, so padding is never a component.

    A split means one original id reaches multiple output components; a merge
    means one output component contains multiple original ids. These are overlap
    counts, not a claim that connected regions equal semantic defect instances.
    """
    if binary.shape != (box.height, box.width):
        raise ValueError(f"mask shape {binary.shape} does not match {box.width}x{box.height}")
    original_ids, original_areas = _label_components(binary)
    resized_ids = np.asarray(
        Image.fromarray(original_ids).resize((box.new_w, box.new_h), Image.NEAREST)
    )
    output_ids, output_areas = _label_components(resized_ids > 0)
    surviving_ids = set(int(value) for value in np.unique(resized_ids) if value > 0)
    lost = [
        {"component_id": index, "original_area": area}
        for index, area in enumerate(original_areas, start=1)
        if index not in surviving_ids
    ]
    foreground = resized_ids > 0
    overlaps = np.unique(
        np.column_stack((resized_ids[foreground], output_ids[foreground])), axis=0
    )
    original_degrees = np.bincount(overlaps[:, 0], minlength=len(original_areas) + 1)
    output_degrees = np.bincount(overlaps[:, 1], minlength=len(output_areas) + 1)
    return {
        "components_before": len(original_areas),
        "components_after": len(output_areas),
        "largest_component_before": max(original_areas, default=0),
        "largest_component_after": max(output_areas, default=0),
        "retained_components": len(surviving_ids),
        "lost_components": lost,
        "split_original_components": int((original_degrees > 1).sum()),
        "merged_output_components": int((output_degrees > 1).sum()),
    }


def dhash64(image: Image.Image, hash_size: int = 8) -> int:
    """64-bit difference hash: robust to resampling/compression, not to crops."""
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    pixels = np.asarray(small, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    packed = np.packbits(bits.reshape(-1))
    return int.from_bytes(packed.tobytes(), "big")


def hamming_matrix(left: list[int], right: list[int]) -> np.ndarray:
    """Pairwise Hamming distance between two lists of 64-bit hashes."""
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.uint16)
    a = np.asarray(left, dtype=np.uint64)
    b = np.asarray(right, dtype=np.uint64)
    xor = np.bitwise_xor(a[:, None], b[None, :])
    bits = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1)
    return bits.sum(axis=1, dtype=np.uint16).reshape(len(left), len(right))
