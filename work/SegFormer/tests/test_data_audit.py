"""Unit tests for the stage 4 geometry, selection and config logic.

These tests do not read the dataset: they pin down the transform that the audit
and the future Dataset must agree on, plus the deterministic sample selection.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from segformer_lab.config import ConfigError, load_config
from segformer_lab.geometry import (
    binarize_ids,
    compute_letterbox,
    count_components,
    dhash64,
    hamming_matrix,
    letterbox_image,
    letterbox_label,
)
from segformer_lab import geometry, paths
from segformer_lab.preview_data import even_indices


class LetterboxTests(unittest.TestCase):
    def test_pcb1_image(self) -> None:
        box = compute_letterbox(1404, 1070, 512)
        self.assertEqual((box.new_w, box.new_h), (512, 390))
        self.assertEqual((box.pad_left, box.pad_top, box.pad_right, box.pad_bottom), (0, 61, 0, 61))
        self.assertEqual(box.valid_area, 512 * 390)
        self.assertEqual(box.padding_pixels, 512 * 512 - 512 * 390)

    def test_square_image_has_no_padding(self) -> None:
        box = compute_letterbox(800, 800, 512)
        self.assertEqual((box.new_w, box.new_h), (512, 512))
        self.assertEqual(box.padding_pixels, 0)

    def test_portrait_image_pads_horizontally(self) -> None:
        box = compute_letterbox(500, 1000, 512)
        self.assertEqual((box.new_w, box.new_h), (256, 512))
        self.assertEqual(box.pad_left + box.pad_right, 256)
        self.assertEqual(box.pad_top + box.pad_bottom, 0)

    def test_padding_never_exceeds_canvas(self) -> None:
        for width, height in ((1, 1000), (9999, 3), (512, 512), (1404, 1070)):
            box = compute_letterbox(width, height, 512)
            self.assertLessEqual(box.new_w, 512)
            self.assertLessEqual(box.new_h, 512)
            self.assertEqual(box.pad_left + box.new_w + box.pad_right, 512)
            self.assertEqual(box.pad_top + box.new_h + box.pad_bottom, 512)


class LabelTests(unittest.TestCase):
    def test_binarize_merges_all_non_zero_ids(self) -> None:
        mask = np.array([[0, 2], [4, 255]], dtype=np.uint8)
        self.assertEqual(binarize_ids(mask).tolist(), [[0, 1], [1, 1]])

    def test_letterbox_label_uses_255_for_padding(self) -> None:
        box = compute_letterbox(1404, 1070, 512)
        binary = np.ones((1070, 1404), dtype=np.uint8)
        label = letterbox_label(binary, box, ignore_index=255)
        self.assertEqual(label.shape, (512, 512))
        self.assertEqual(int(label[0, 0]), 255)          # top padding row
        self.assertEqual(int(label[511, 511]), 255)      # bottom padding row
        self.assertEqual(int(label[200, 200]), 1)        # inside the valid box
        top, left, bottom, right = box.crop_box
        self.assertTrue(set(np.unique(label[top:bottom, left:right]).tolist()) <= {0, 1})

    def test_letterbox_label_keeps_zero(self) -> None:
        box = compute_letterbox(1404, 1070, 512)
        label = letterbox_label(np.zeros((1070, 1404), dtype=np.uint8), box)
        self.assertEqual(set(np.unique(label).tolist()), {0, 255})

    def test_letterbox_image_shape_and_pad_colour(self) -> None:
        box = compute_letterbox(1404, 1070, 512)
        image = Image.new("RGB", (1404, 1070), (10, 20, 30))
        canvas = letterbox_image(image, box, (124, 116, 104))
        self.assertEqual(canvas.shape, (512, 512, 3))
        self.assertEqual(canvas.dtype, np.uint8)
        self.assertEqual(tuple(canvas[0, 0].tolist()), (124, 116, 104))
        self.assertEqual(tuple(canvas[200, 200].tolist()), (10, 20, 30))


class ComponentTests(unittest.TestCase):
    def test_single_blob(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:10, 5:12] = 1
        self.assertEqual(count_components(mask), (1, 35))

    def test_two_separated_blobs(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:4, 2:4] = 1
        mask[10:13, 10:15] = 1
        self.assertEqual(count_components(mask), (2, 15))

    def test_diagonal_touch_is_not_connected(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2, 2] = 1
        mask[3, 3] = 1
        self.assertEqual(count_components(mask)[0], 2)

    def test_u_shape_is_one_component(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2] = 1
        mask[2:8, 6] = 1
        mask[7, 2:7] = 1
        count, largest = count_components(mask)
        self.assertEqual(count, 1)
        # rows 2..6 contribute 2 px each (10), row 7 contributes 5 -> 15
        self.assertEqual(largest, 15)

    def test_empty_mask(self) -> None:
        self.assertEqual(count_components(np.zeros((5, 5), dtype=np.uint8)), (0, 0))


class ComponentRetentionTests(unittest.TestCase):
    def test_count_decrease_can_be_a_merge_without_any_lost_component(self) -> None:
        # 8 -> 4 NEAREST samples coordinates 1, 3, 5, 7 on each axis.
        # The two original points at x=1 and x=3 become adjacent output pixels.
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1, 1] = mask[1, 3] = 1
        measure = getattr(geometry, "component_retention", None)
        self.assertTrue(callable(measure), "per-original-component retention must be measured")
        stats = measure(mask, compute_letterbox(8, 8, 4))
        self.assertEqual(stats["components_before"], 2)
        self.assertEqual(stats["components_after"], 1)
        self.assertEqual(stats["retained_components"], 2)
        self.assertEqual(stats["lost_components"], [])
        self.assertEqual(stats["split_original_components"], 0)
        self.assertEqual(stats["merged_output_components"], 1)

    def test_disappearance_can_be_hidden_by_another_component_splitting(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        # One U-shaped component has two tips on the sampled row y=1, joined
        # only on unsampled row y=2. It splits into two output components.
        mask[1, 1] = mask[1, 5] = 1
        mask[2, 1:6] = 1
        # A separate 1-pixel component on an unsampled row disappears.
        mask[6, 7] = 1
        measure = getattr(geometry, "component_retention", None)
        self.assertTrue(callable(measure), "per-original-component retention must be measured")
        stats = measure(mask, compute_letterbox(8, 8, 4))
        self.assertEqual(stats["components_before"], 2)
        self.assertEqual(stats["components_after"], 2)
        self.assertEqual(stats["retained_components"], 1)
        self.assertEqual(stats["lost_components"], [{"component_id": 2, "original_area": 1}])
        self.assertEqual(stats["split_original_components"], 1)
        self.assertEqual(stats["merged_output_components"], 0)

    def test_component_identity_is_not_truncated_to_uint8_or_confused_with_padding(self) -> None:
        mask = np.zeros((33, 65), dtype=np.uint8)
        mask[::2, ::2] = 1  # 17 * 33 = 561 separate original components.
        measure = getattr(geometry, "component_retention", None)
        self.assertTrue(callable(measure), "per-original-component retention must be measured")
        stats = measure(mask, compute_letterbox(65, 33, 65))
        self.assertEqual(stats["components_before"], 561)
        self.assertEqual(stats["components_after"], 561)
        self.assertEqual(stats["retained_components"], 561)
        self.assertEqual(stats["lost_components"], [])
        self.assertEqual(stats["split_original_components"], 0)
        self.assertEqual(stats["merged_output_components"], 0)


class HashTests(unittest.TestCase):
    def _image(self, seed: int) -> Image.Image:
        rng = np.random.default_rng(seed)
        return Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))

    def test_same_image_same_hash(self) -> None:
        image = self._image(1)
        self.assertEqual(dhash64(image), dhash64(image.copy()))

    def test_different_images_differ(self) -> None:
        self.assertNotEqual(dhash64(self._image(1)), dhash64(self._image(2)))

    def test_hamming_matrix_diagonal_is_zero(self) -> None:
        hashes = [dhash64(self._image(i)) for i in range(4)]
        distances = hamming_matrix(hashes, hashes)
        self.assertTrue(np.all(np.diag(distances) == 0))

    def test_hamming_matrix_matches_manual_popcount(self) -> None:
        left = [0b1010, 0b1111]
        right = [0b0010]
        # 1010 ^ 0010 = 1000 -> 1 bit; 1111 ^ 0010 = 1101 -> 3 bits
        self.assertEqual(hamming_matrix(left, right).tolist(), [[1], [3]])


class SelectionTests(unittest.TestCase):
    def test_even_indices_covers_both_ends(self) -> None:
        self.assertEqual(even_indices(120, 3), [0, 60, 119])
        self.assertEqual(even_indices(48, 6), [0, 9, 19, 28, 38, 47])

    def test_even_indices_when_fewer_items_than_wanted(self) -> None:
        self.assertEqual(even_indices(3, 6), [0, 1, 2])

    def test_even_indices_is_unique_and_sorted(self) -> None:
        indices = even_indices(12, 6)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertEqual(len(indices), 6)


class ConfigTests(unittest.TestCase):
    def test_baseline_config_loads(self) -> None:
        cfg = load_config("configs/baseline.yaml")
        self.assertEqual(cfg["input_size"], 512)
        self.assertEqual(cfg["ignore_index"], 255)
        self.assertEqual(cfg["num_labels"], 2)
        for key in ("data.root", "data.train_manifest", "experiment.output_root"):
            self.assertTrue(Path(cfg["paths"][key]).is_absolute())

    def test_protocol_drift_is_rejected(self) -> None:
        raw = yaml.safe_load((paths.PROJECT_ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
        raw["data"]["input_size"] = 256
        with tempfile.TemporaryDirectory() as tmp:
            drifted = Path(tmp) / "drifted.yaml"
            drifted.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(drifted)

    def test_missing_section_is_rejected(self) -> None:
        raw = yaml.safe_load((paths.PROJECT_ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
        raw.pop("loss")
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.yaml"
            broken.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(broken)


if __name__ == "__main__":
    unittest.main()
