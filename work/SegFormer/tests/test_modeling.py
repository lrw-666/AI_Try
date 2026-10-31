"""Unit tests for the stage 5 download/loading helpers.

The expensive, GPU-bound parts (actual loading of the 14 MB checkpoint and the
forward/backward) are exercised by `python -m segformer_lab.model_check`, not by
unit tests. What is tested here are the decisions that are easy to get silently
wrong: which missing keys are acceptable, whether a Git LFS pointer would fool
the downloader, and whether the synthetic label really encodes the 0/1/255
convention the rest of the project relies on.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from segformer_lab.download_model import is_lfs_pointer, sha256_file, verify_snapshot
from segformer_lab.model_check import PAD_ROWS, make_batch
from segformer_lab.modeling import classify_loading_keys


class LoadingKeyTests(unittest.TestCase):
    def _info(self, missing, mismatched, unexpected):
        return {"missing_keys": missing, "mismatched_keys": mismatched, "unexpected_keys": unexpected}

    def test_head_only_missing_is_acceptable(self) -> None:
        info = classify_loading_keys(
            self._info(
                missing=["decode_head.classifier.weight", "decode_head.classifier.bias"],
                mismatched=[["decode_head.classifier.weight", [2, 256, 1, 1], [1000, 256, 1, 1]]],
                unexpected=["classifier.weight", "classifier.bias"],
            )
        )
        self.assertTrue(info["encoder_fully_loaded"])
        self.assertEqual(len(info["missing_head_keys"]), 2)
        self.assertEqual(len(info["missing_other_keys"]), 0)
        self.assertEqual(len(info["mismatched_head_keys"]), 1)
        self.assertEqual(len(info["mismatched_other_keys"]), 0)

    def test_missing_encoder_key_is_not_acceptable(self) -> None:
        info = classify_loading_keys(
            self._info(missing=["segformer.encoder.layer.0.norm1.weight"], mismatched=[], unexpected=[])
        )
        self.assertFalse(info["encoder_fully_loaded"])
        self.assertEqual(info["missing_other_keys"], ["segformer.encoder.layer.0.norm1.weight"])

    def test_mismatched_encoder_key_is_not_acceptable(self) -> None:
        info = classify_loading_keys(
            self._info(
                missing=[],
                mismatched=[["segformer.encoder.patch_embeddings.1.proj.weight", [64, 32, 3, 3], [64, 32, 1, 1]]],
                unexpected=[],
            )
        )
        self.assertFalse(info["encoder_fully_loaded"])
        self.assertEqual(len(info["mismatched_other_keys"]), 1)

    def test_unexpected_keys_do_not_break_encoder(self) -> None:
        info = classify_loading_keys(self._info([], [], ["classifier.weight"]))
        self.assertTrue(info["encoder_fully_loaded"])


class LfsPointerTests(unittest.TestCase):
    def test_pointer_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.bin"
            path.write_bytes(
                b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n"
            )
            self.assertTrue(is_lfs_pointer(path))

    def test_binary_weights_are_not_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.bin"
            path.write_bytes(b"\x00\x01\x02" * 2048)
            self.assertFalse(is_lfs_pointer(path))


class SnapshotVerificationTests(unittest.TestCase):
    def _make(self, root: Path) -> dict:
        weights = root / "pytorch_model.bin"
        weights.write_bytes(b"x" * 2048)
        config = root / "config.json"
        config.write_text("{}", encoding="utf-8")
        pre = root / "preprocessor_config.json"
        pre.write_text("{}", encoding="utf-8")
        source = {
            "schema": "segformer-lab/model-source/1",
            "model_id": "nvidia/mit-b0",
            "revision": "a" * 40,
            "weights": "pytorch_model.bin",
            "total_bytes": 2052,
            "files": [
                {"path": "pytorch_model.bin", "size_bytes": 2048, "sha256": sha256_file(weights)},
                {"path": "config.json", "size_bytes": 2, "sha256": sha256_file(config)},
                {"path": "preprocessor_config.json", "size_bytes": 2, "sha256": sha256_file(pre)},
            ]
        }
        (root / "source.json").write_text(json.dumps(source), encoding="utf-8")
        return source

    def test_intact_snapshot_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make(root)
            self.assertEqual(verify_snapshot(root, source), [])

    def test_missing_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make(root)
            (root / "pytorch_model.bin").unlink()
            problems = verify_snapshot(root, source)
            self.assertTrue(any("missing file" in p for p in problems))

    def test_size_change_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make(root)
            (root / "pytorch_model.bin").write_bytes(b"y" * 4096)
            problems = verify_snapshot(root, source)
            self.assertTrue(any("size changed" in p for p in problems))

    def test_hash_change_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make(root)
            (root / "pytorch_model.bin").write_bytes(b"z" * 2048)
            problems = verify_snapshot(root, source)
            self.assertTrue(any("sha256 changed" in p for p in problems))


class SyntheticBatchTests(unittest.TestCase):
    def test_shapes_and_label_values(self) -> None:
        images, labels = make_batch(2, 512)
        self.assertEqual(tuple(images.shape), (2, 3, 512, 512))
        self.assertEqual(tuple(labels.shape), (2, 512, 512))
        self.assertEqual(sorted({int(v) for v in labels.unique()}), [0, 1, 255])

    def test_padding_rows_are_255(self) -> None:
        _, labels = make_batch(1, 512)
        self.assertTrue(bool((labels[:, :PAD_ROWS, :] == 255).all()))
        self.assertTrue(bool((labels[:, 512 - PAD_ROWS:, :] == 255).all()))
        self.assertEqual(int((labels == 255).sum()), 2 * PAD_ROWS * 512)

    def test_valid_region_has_no_ignore_value(self) -> None:
        _, labels = make_batch(1, 512)
        valid = labels[:, PAD_ROWS:512 - PAD_ROWS, :]
        self.assertNotIn(255, {int(v) for v in valid.unique()})
        self.assertGreater(int((valid == 1).sum()), 0)

    def test_batch_is_deterministic(self) -> None:
        a_images, a_labels = make_batch(2, 256, seed=7)
        b_images, b_labels = make_batch(2, 256, seed=7)
        self.assertTrue(bool((a_images == b_images).all()))
        self.assertTrue(bool((a_labels == b_labels).all()))


if __name__ == "__main__":
    unittest.main()
