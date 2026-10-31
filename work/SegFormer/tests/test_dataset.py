"""Unit tests for the stage 6 Dataset, collate and flip policy.

The real acceptance run reads the 1104-image manifests through
`python -m segformer_lab.dataset_check`. These tests build a two-image toy set
on disk instead, so the rules that are easy to break silently stay pinned:
normal images get a constructed zero mask, defect ids are merged **before**
padding, a forced flip keeps image and mask aligned, and the collate function
never stacks original-resolution masks.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from segformer_lab.data import PCB1Dataset, build_dataloader, collate_pcb1, flip_decision


class StubProcessor:
    """Minimal stand-in for SegformerImageProcessor (same maths, no weights)."""

    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    do_resize = False
    do_reduce_labels = False
    do_rescale = True
    do_normalize = True
    rescale_factor = 1.0 / 255.0

    def __call__(self, images, return_tensors: str = "pt"):
        array = np.asarray(images, dtype=np.float32) / 255.0
        chw = np.transpose(array, (2, 0, 1))
        mean = np.asarray(self.image_mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(self.image_std, dtype=np.float32).reshape(3, 1, 1)
        return {"pixel_values": torch.from_numpy(((chw - mean) / std)[None])}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object", "split", "label", "image", "mask"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _gradient(width: int, height: int, reverse: bool = False) -> np.ndarray:
    """Non-symmetric test image: a vertical flip cannot be mistaken for a horizontal one."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[..., 0] = (np.arange(width, dtype=np.int32)[None, :] * 3 % 256).astype(np.uint8)
    canvas[..., 1] = (np.arange(height, dtype=np.int32)[:, None] * 5 % 256).astype(np.uint8)
    canvas[..., 2] = 40
    if reverse:
        canvas = canvas[:, ::-1].copy()
    return canvas


def _make_toy_root(tmp: Path, width: int = 40, height: int = 30) -> tuple[Path, dict]:
    root = tmp / "data"
    (root / "pcb1" / "Data" / "Images" / "Normal").mkdir(parents=True)
    (root / "pcb1" / "Data" / "Images" / "Anomaly").mkdir(parents=True)
    (root / "pcb1" / "Data" / "Masks" / "Anomaly").mkdir(parents=True)

    Image.fromarray(_gradient(width, height)).save(root / "pcb1" / "Data" / "Images" / "Normal" / "0001.JPG")
    Image.fromarray(_gradient(width, height, reverse=True)).save(
        root / "pcb1" / "Data" / "Images" / "Anomaly" / "0002.JPG"
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:14, 15:20] = 2
    mask[20:22, 5:7] = 4
    Image.fromarray(mask, mode="L").save(root / "pcb1" / "Data" / "Masks" / "Anomaly" / "0002.png")

    rows = [
        {"object": "pcb1", "split": "train", "label": "normal",
         "image": "pcb1/Data/Images/Normal/0001.JPG", "mask": ""},
        {"object": "pcb1", "split": "train", "label": "anomaly",
         "image": "pcb1/Data/Images/Anomaly/0002.JPG", "mask": "pcb1/Data/Masks/Anomaly/0002.png"},
    ]
    manifests = tmp / "manifests"
    manifests.mkdir()
    _write_csv(manifests / "train.csv", rows)
    _write_csv(manifests / "val.csv", rows)
    _write_csv(manifests / "test.csv", rows)

    config = {
        "data_root": root,
        "input_size": 16,
        "ignore_index": 255,
        "image_pad_rgb": (124, 116, 104),
        "num_labels": 2,
        "raw": {"data": {"horizontal_flip_p": 0.5, "workers": 0}},
        "paths": {
            "data.train_manifest": manifests / "train.csv",
            "data.val_manifest": manifests / "val.csv",
            "data.test_manifest": manifests / "test.csv",
        },
    }
    return root, config


class FlipDecisionTests(unittest.TestCase):
    def test_none_and_force(self) -> None:
        self.assertFalse(flip_decision("none", 42, 0, 0, 0.5))
        self.assertTrue(flip_decision("force", 42, 0, 0, 0.5))

    def test_random_is_deterministic(self) -> None:
        first = [flip_decision("random", 42, 3, i, 0.5) for i in range(20)]
        second = [flip_decision("random", 42, 3, i, 0.5) for i in range(20)]
        self.assertEqual(first, second)
        self.assertTrue(any(first) and not all(first), "with p=0.5 both outcomes should appear")

    def test_epoch_changes_pattern(self) -> None:
        epoch0 = [flip_decision("random", 42, 0, i, 0.5) for i in range(32)]
        epoch1 = [flip_decision("random", 42, 1, i, 0.5) for i in range(32)]
        self.assertNotEqual(epoch0, epoch1)

    def test_probability_bounds(self) -> None:
        self.assertFalse(any(flip_decision("random", 7, 0, i, 0.0) for i in range(16)))
        self.assertTrue(all(flip_decision("random", 7, 0, i, 1.0) for i in range(16)))

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            flip_decision("sometimes", 42, 0, 0, 0.5)


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root, self.config = _make_toy_root(Path(self.tmp.name))
        self.processor = StubProcessor()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _dataset(self, split: str = "train", **kwargs) -> PCB1Dataset:
        kwargs.setdefault("processor", self.processor)
        return PCB1Dataset(self.config, split, **kwargs)

    def test_normal_has_zero_mask_and_geometry(self) -> None:
        dataset = self._dataset(flip="none")
        item = dataset[0]
        self.assertEqual(item["image_id"], "pcb1/Data/Images/Normal/0001.JPG")
        self.assertEqual(item["mask_id"], "")
        self.assertEqual(tuple(item["pixel_values"].shape), (3, 16, 16))
        self.assertEqual(item["pixel_values"].dtype, torch.float32)
        self.assertEqual(item["labels"].dtype, torch.int64)
        self.assertEqual(tuple(item["labels"].shape), (16, 16))
        # 40x30 -> 16x12, two padding rows top and bottom
        self.assertEqual(item["geometry"]["new_w"], 16)
        self.assertEqual(item["geometry"]["new_h"], 12)
        self.assertEqual(item["geometry"]["pad_top"], 2)
        labels = item["labels"].numpy()
        self.assertTrue((labels[:2] == 255).all())
        self.assertTrue((labels[-2:] == 255).all())
        self.assertTrue((labels[2:-2] == 0).all())

    def test_anomaly_ids_merged_before_padding(self) -> None:
        dataset = self._dataset(flip="none")
        item = dataset[1]
        labels = item["labels"].numpy()
        self.assertEqual(sorted(np.unique(labels).tolist()), [0, 1, 255])
        self.assertGreater(int((labels == 1).sum()), 0)
        self.assertEqual(int((labels == 255).sum()), 2 * 2 * 16)
        # the original-resolution mask stays binary and unletterboxed
        original = item["original_mask"]
        self.assertEqual(original.shape, (30, 40))
        self.assertEqual(sorted(np.unique(original).tolist()), [0, 1])
        self.assertEqual(int(original.sum()), 20 + 4)

    def test_force_flip_keeps_image_and_mask_aligned(self) -> None:
        plain_dataset = self._dataset(flip="none")
        forced_dataset = self._dataset(flip="force")
        plain_item = plain_dataset[1]
        forced_item = forced_dataset[1]
        plain = plain_dataset.prepared_sample(1)
        forced = forced_dataset.prepared_sample(1)

        self.assertFalse(plain_item["flipped"])
        self.assertTrue(forced_item["flipped"])
        self.assertTrue(np.array_equal(forced["binary_original"], np.fliplr(plain["binary_original"])))
        # labels are 2-D [H, W]: horizontal flip is dim 1
        self.assertTrue(torch.equal(forced_item["labels"], torch.flip(plain_item["labels"], dims=(1,))))
        diff = np.abs(
            plain["image_512"].astype(np.int16) - np.fliplr(forced["image_512"]).astype(np.int16)
        ).max()
        self.assertLessEqual(int(diff), 1, "horizontal flip must survive letterboxing to within one LSB")

    def test_wrong_axis_would_not_match(self) -> None:
        """Guards the earlier bug: flipping the height axis must not look like a horizontal flip."""
        plain = self._dataset(flip="none")[1]
        forced = self._dataset(flip="force")[1]
        vertical = torch.flip(plain["labels"], dims=(0,))
        self.assertFalse(torch.equal(vertical, forced["labels"]))

    def test_random_flip_rejected_for_validation(self) -> None:
        with self.assertRaises(ValueError):
            PCB1Dataset(self.config, "val", self.processor, flip="random")

    def test_invalid_split_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PCB1Dataset(self.config, "dev", self.processor)

    def test_evaluation_rejects_all_flips(self) -> None:
        for split in ("val", "test"):
            self.assertFalse(self._dataset(split)[1]["flipped"])
            for mode in ("random", "force"):
                with self.subTest(split=split, mode=mode), self.assertRaises(ValueError):
                    self._dataset(split, flip=mode)

    def test_configured_train_ids_are_used_only_for_train(self) -> None:
        ids = Path(self.tmp.name) / "configured.txt"
        ids.write_text("pcb1/Data/Images/Anomaly/0002.JPG\n", encoding="utf-8")
        self.config["paths"]["data.train_ids_file"] = ids
        self.assertEqual(self._dataset().image_ids(), ["pcb1/Data/Images/Anomaly/0002.JPG"])
        for split in ("val", "test"):
            self.assertEqual(len(self._dataset(split)), 2)

    def test_explicit_ids_override_configured_train_ids(self) -> None:
        self.config["paths"]["data.train_ids_file"] = Path(self.tmp.name) / "missing.txt"
        ids = Path(self.tmp.name) / "explicit.txt"
        ids.write_text("pcb1/Data/Images/Normal/0001.JPG\n", encoding="utf-8")
        self.assertEqual(self._dataset(ids_file=ids).image_ids(), ["pcb1/Data/Images/Normal/0001.JPG"])

    def test_empty_ids_file_is_rejected(self) -> None:
        ids = Path(self.tmp.name) / "empty.txt"
        ids.write_text("# no selected images\n\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._dataset(ids_file=ids)

    def test_default_seed_comes_from_config_with_explicit_override(self) -> None:
        self.assertEqual(self._dataset().seed, 42)
        self.config["raw"]["experiment"] = {"seed": 19}
        self.assertEqual(self._dataset().seed, 19)
        self.assertEqual(self._dataset(seed=7).seed, 7)

    def test_loader_default_shuffle_seed_follows_dataset(self) -> None:
        dataset = self._dataset(seed=19)
        # Enough sampler indices to distinguish permutation seeds reliably.
        dataset.rows = dataset.rows * 6
        default = build_dataloader(dataset, batch_size=2, shuffle=True)
        explicit = build_dataloader(dataset, batch_size=2, shuffle=True, seed=19)
        self.assertEqual(list(default.sampler), list(explicit.sampler))

    def test_loader_rejects_negative_seed(self) -> None:
        with self.assertRaises(ValueError):
            build_dataloader(self._dataset(), batch_size=2, shuffle=True, seed=-1)

    def test_invalid_flip_probability_rejected_at_construction(self) -> None:
        for probability in (-0.1, 1.1, float("nan"), float("inf")):
            self.config["raw"]["data"]["horizontal_flip_p"] = probability
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                self._dataset(flip="none")

    def test_negative_seed_and_epoch_rejected_at_construction(self) -> None:
        for kwargs in ({"seed": -1}, {"epoch": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self._dataset(flip="none", **kwargs)

    def test_set_epoch_rejects_negative_without_changing_state(self) -> None:
        dataset = self._dataset(epoch=3)
        with self.assertRaises(ValueError):
            dataset.set_epoch(-1)
        self.assertEqual(dataset.epoch, 3)

    def test_custom_processor_protocol_flags_are_checked(self) -> None:
        for flag, wrong in (("do_resize", True), ("do_reduce_labels", True),
                            ("do_rescale", False), ("do_normalize", False)):
            processor = StubProcessor()
            setattr(processor, flag, wrong)
            with self.subTest(flag=flag), self.assertRaises((ValueError, RuntimeError)):
                self._dataset(processor=processor)

    def test_custom_processor_statistics_are_checked(self) -> None:
        for name, value in (("image_mean", [0.5, 0.5]),
                            ("image_mean", [0.5, float("nan"), 0.5]),
                            ("image_std", [1, 0, 1]),
                            ("image_std", [1, -1, 1]),
                            ("image_std", [1, float("inf"), 1])):
            processor = StubProcessor()
            setattr(processor, name, value)
            with self.subTest(name=name, value=value), self.assertRaises((ValueError, RuntimeError)):
                self._dataset(processor=processor)

    def test_numpy_processor_statistics_are_supported(self) -> None:
        self.processor.image_mean = np.asarray(self.processor.image_mean)
        self.processor.image_std = np.asarray(self.processor.image_std)
        self.assertTrue(torch.isfinite(self._dataset()[1]["pixel_values"]).all())

    def test_ids_file_filters_and_validates(self) -> None:
        ids = Path(self.tmp.name) / "ids.txt"
        ids.write_text("# comment\npcb1/Data/Images/Anomaly/0002.JPG\n", encoding="utf-8")
        dataset = self._dataset(flip="none", ids_file=ids)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["image_id"], "pcb1/Data/Images/Anomaly/0002.JPG")

        bad = Path(self.tmp.name) / "bad.txt"
        bad.write_text("pcb1/Data/Images/Normal/9999.JPG\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._dataset(flip="none", ids_file=bad)


class CollateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root, self.config = _make_toy_root(Path(self.tmp.name))
        self.processor = StubProcessor()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_collate_stacks_tensors_and_keeps_lists(self) -> None:
        dataset = PCB1Dataset(self.config, "train", self.processor, flip="none")
        batch = collate_pcb1([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["pixel_values"].shape), (2, 3, 16, 16))
        self.assertEqual(batch["pixel_values"].dtype, torch.float32)
        self.assertEqual(tuple(batch["labels"].shape), (2, 16, 16))
        self.assertEqual(batch["labels"].dtype, torch.int64)
        self.assertIsInstance(batch["original_mask"], list)
        self.assertEqual(len(batch["original_mask"]), 2)
        self.assertEqual(batch["original_mask"][0].shape, (30, 40))
        self.assertIsInstance(batch["geometry"], list)
        self.assertIsInstance(batch["image_id"], list)
        self.assertIsInstance(batch["flipped"], list)

    def test_empty_batch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            collate_pcb1([])

    def test_mixed_original_mask_presence_is_rejected_in_both_orders(self) -> None:
        dataset = PCB1Dataset(self.config, "train", self.processor, flip="none")
        with_mask = dataset[0]
        without_mask = dataset[1]
        del without_mask["original_mask"]
        for batch in ([without_mask, with_mask], [with_mask, without_mask]):
            with self.subTest(first_has_mask="original_mask" in batch[0]), self.assertRaises(ValueError):
                collate_pcb1(batch)


if __name__ == "__main__":
    unittest.main()
