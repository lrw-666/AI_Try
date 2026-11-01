"""CPU regressions for the stage-12 evaluation/ prediction channel.

These pin the parts that can silently mis-report:

* the display-selection rules (ordering, tie-breaking and the difference between
  "TP == 0" and "predicted nothing");
* the val gate comparator (integers/NA exact, floats within 1e-9);
* the 0/1 mask writer and the refusal of non-binary arrays;
* ``prepare_image`` reproducing the dataset preprocessing bit for bit;
* the frozen-run loader refusing an incomplete run directory.
"""
import json
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from segformer_lab import evaluate, inference, predict


def _row(image_id, label, *, tp=0, fp=0, fn=0, tn=100, gt=0, pred=0, iou=None,
         recall=None, dice=None, precision=None):
    return {
        "image_id": image_id, "label": label, "split": "val",
        "total_pixels": tp + fp + fn + tn, "gt_foreground_pixels": gt,
        "predicted_foreground_pixels": pred, "predicted_foreground_ratio": pred / max(1, tp + fp + fn + tn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": iou, "dice": dice, "precision": precision, "recall": recall,
        "iou_na": "NA" if iou is None else "", "dice_na": "NA" if dice is None else "",
        "precision_na": "NA" if precision is None else "", "recall_na": "NA" if recall is None else "",
    }


class SelectionTests(unittest.TestCase):
    def test_rules_are_ordered_and_ties_break_on_image_id(self):
        rows = [
            _row("b/Anomaly/2.JPG", "anomaly", tp=0, fn=10, gt=10, pred=0, iou=0.0, recall=0.0),
            _row("b/Anomaly/1.JPG", "anomaly", tp=0, fn=10, gt=10, pred=5, iou=0.0, recall=0.0),
            _row("b/Anomaly/3.JPG", "anomaly", tp=5, fn=5, gt=10, pred=5, iou=0.5, recall=0.5),
            _row("b/Normal/2.JPG", "normal", fp=7, tn=93),
            _row("b/Normal/1.JPG", "normal", fp=7, tn=93),
            _row("b/Normal/3.JPG", "normal", fp=0, tn=100),
        ]
        selection = evaluate.display_selection(rows, "val")
        self.assertEqual([item["image_id"] for item in selection["worst_iou"]][:2],
                         ["b/Anomaly/1.JPG", "b/Anomaly/2.JPG"])
        self.assertEqual([item["image_id"] for item in selection["most_normal_fp"]][:2],
                         ["b/Normal/1.JPG", "b/Normal/2.JPG"])
        self.assertEqual(selection["fixed_anomaly"][0]["image_id"], "b/Anomaly/1.JPG")
        # TP == 0 is not the same statement as "predicted nothing".
        self.assertEqual(selection["all_anomaly_tp_zero"], ["b/Anomaly/1.JPG", "b/Anomaly/2.JPG"])
        self.assertEqual(selection["all_anomaly_empty_prediction"], ["b/Anomaly/2.JPG"])
        self.assertEqual(selection["all_normal_with_false_positive"], ["b/Normal/1.JPG", "b/Normal/2.JPG"])

    def test_counts_and_reasons_are_kept(self):
        rows = [_row("a/Normal/1.JPG", "normal", fp=0, tn=10), _row("a/Anomaly/1.JPG", "anomaly", tp=1, gt=4)]
        selection = evaluate.display_selection(rows, "test")
        self.assertEqual(selection["counts"], {"normal": 1, "anomaly": 1})
        self.assertTrue(all(item["reason"] for item in selection["fixed_anomaly"]))


class GateTests(unittest.TestCase):
    def build(self):
        rows = [_row("a/Anomaly/1.JPG", "anomaly", tp=2, fp=1, fn=2, gt=4, pred=3,
                     iou=0.4, dice=0.5714, precision=0.6667, recall=0.5),
                _row("a/Normal/1.JPG", "normal", fp=0, tn=10)]
        return {
            "images": {"total": 2, "normal": 1, "anomaly": 1},
            "confusion": {"tp": 2, "fp": 1, "fn": 2, "tn": 10, "pixels": 15},
            "global": {"foreground_iou": 0.4, "foreground_dice": 0.5714,
                       "precision": 0.6667, "recall": 0.5, "na_reasons": {}},
            "per_image": rows,
            "anomaly_images": {"count": 1, "macro": {
                "iou": {"mean": 0.4, "defined_images": 1, "na_images": 0},
                "dice": {"mean": 0.5714, "defined_images": 1, "na_images": 0},
                "precision": {"mean": 0.6667, "defined_images": 1, "na_images": 0},
                "recall": {"mean": 0.5, "defined_images": 1, "na_images": 0}}},
            "normal_images": {"count": 1, "false_positive_pixels": 0, "fp_pixel_ratio": 0.0,
                              "images_with_any_false_positive": 0},
        }

    def test_identical_records_pass(self):
        block = self.build()
        gate = evaluate._gate_compare(block, self.build())
        self.assertTrue(gate["passed"], gate["failed"])

    def test_integer_difference_fails(self):
        left, right = self.build(), self.build()
        left["per_image"][0]["tp"] = 3
        gate = evaluate._gate_compare(left, right)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failed"], ["per_image.fields"])

    def test_float_drift_beyond_tolerance_fails(self):
        left, right = self.build(), self.build()
        left["global"]["foreground_iou"] = 0.4 + 1e-6
        gate = evaluate._gate_compare(left, right)
        self.assertFalse(gate["passed"])
        self.assertIn("global.foreground_iou", gate["failed"])

    def test_na_structure_difference_fails(self):
        left, right = self.build(), self.build()
        left["per_image"][0]["iou_na"] = "NA"
        gate = evaluate._gate_compare(left, right)
        self.assertFalse(gate["passed"])

    def test_duplicate_rows_cannot_be_hidden_by_image_id_dictionary(self):
        fresh, saved = self.build(), self.build()
        fresh["per_image"].append(copy.deepcopy(fresh["per_image"][0]))
        self.assertFalse(evaluate._gate_compare(fresh, saved)["passed"])

    def test_per_image_label_and_nonfinite_value_are_rejected(self):
        for field, value in (("label", "normal"), ("iou", float("nan"))):
            with self.subTest(field=field):
                fresh, saved = self.build(), self.build()
                fresh["per_image"][0][field] = value
                self.assertFalse(evaluate._gate_compare(fresh, saved)["passed"])

    def test_normal_percentiles_and_macro_na_are_compared(self):
        for section, field, value in (("normal_images", "predicted_foreground_ratio", {"p95": 0.5}),
                                      ("anomaly_images", "macro", {"iou": {"na_images": 1}})):
            with self.subTest(section=section):
                fresh, saved = self.build(), self.build()
                if section == "normal_images":
                    fresh[section][field] = value
                    saved[section][field] = {"p95": 0.0}
                else:
                    fresh[section][field]["iou"]["na_images"] = 1
                self.assertFalse(evaluate._gate_compare(fresh, saved)["passed"])

    def test_gate_file_must_pass_and_match_identity(self):
        frozen = SimpleNamespace(identity={
            "best_sha256": "abc", "config_sha256": "cfg",
            "manifests": {split: {"sha256": split} for split in ("train", "val", "test")},
            "split_meta_sha256": "meta", "model_config_sha256": "modelcfg"})
        config = {"raw": {"eval": {"threshold": 0.5}}}
        binding = {"code": {"inference.py": "source-a"}, "data": {"splits": {}},
                   "protocol": {"precision": "ieee_fp32"}, "runtime": {"torch": "same"}}
        names = ("images.total", "images.normal", "images.anomaly", "per_image.fields",
                 "per_image.ids", "per_image.unique_ids", "preview.probability", "coverage")
        checks = [{"name": name, "ok": True} for name in names]
        for check, count in zip(checks[:3], (132, 120, 12)):
            check.update(left=count, right=count)
        good = {"schema": evaluate.GATE_SCHEMA, "passed": True, "failed": [],
                "identity": {"best_sha256": "abc", "config_sha256": "cfg",
                             "manifests": frozen.identity["manifests"],
                             "split_meta_sha256": "meta", "model_config_sha256": "modelcfg"},
                "threshold": 0.5, "checks": checks, "binding": binding}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(evaluate, "_evaluation_binding", return_value=copy.deepcopy(binding)), \
                patch.object(evaluate, "_verify_data_files"):
            path = Path(tmp) / "gate.json"
            path.write_text(json.dumps(good), encoding="utf-8")
            self.assertEqual(evaluate._verify_val_gate(path, frozen, config)["passed"], True)

            bad = json.loads(json.dumps(good))
            bad["passed"] = False
            bad["failed"] = ["confusion.tp"]
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(SystemExit):
                evaluate._verify_val_gate(path, frozen, config)

            for section in ("code", "data", "protocol", "runtime"):
                with self.subTest(binding=section):
                    stale = copy.deepcopy(good)
                    stale["binding"][section] = {"changed": True}
                    path.write_text(json.dumps(stale), encoding="utf-8")
                    with self.assertRaises(SystemExit):
                        evaluate._verify_val_gate(path, frozen, config)

            other = json.loads(json.dumps(good))
            other["identity"]["best_sha256"] = "different"
            path.write_text(json.dumps(other), encoding="utf-8")
            with self.assertRaises(SystemExit):
                evaluate._verify_val_gate(path, frozen, config)

    def test_stale_processor_and_unbound_legacy_gates_are_rejected(self):
        frozen = SimpleNamespace(identity={
            "best_sha256": "abc", "config_sha256": "cfg", "processor_sha256": "current",
            "manifests": {split: {"sha256": split} for split in ("train", "val", "test")},
            "split_meta_sha256": "meta", "model_config_sha256": "modelcfg"})
        gate = {"schema": evaluate.GATE_SCHEMA, "passed": True, "failed": [],
                "identity": {**frozen.identity, "processor_sha256": "stale"}, "threshold": 0.5}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaises(SystemExit):
                evaluate._verify_val_gate(path, frozen, {"raw": {"eval": {"threshold": 0.5}}})


class OutputGuardTests(unittest.TestCase):
    def test_predict_refuses_even_an_empty_existing_output_before_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "source.png"
            Image.new("RGB", (2, 3)).save(image)
            output = Path(tmp) / "existing"
            output.mkdir()
            args = SimpleNamespace(config="configs/baseline.yaml", image=str(image),
                                   out=str(output), run=str(Path(tmp) / "run"), device="cpu")
            with patch.object(predict, "load_frozen_run", side_effect=AssertionError("loaded before guard")):
                with self.assertRaises(SystemExit):
                    predict.do_predict(args)

    def test_evaluate_refuses_a_new_output_inside_the_source_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            args = SimpleNamespace(config="configs/baseline.yaml", out=str(run / "evaluation"),
                                   run=str(run), device="cpu", split="val")
            with patch.object(evaluate, "load_frozen_run", side_effect=AssertionError("loaded before guard")):
                with self.assertRaises(SystemExit):
                    evaluate.do_evaluate(args)
            self.assertFalse((run / "evaluation").exists())

    def test_test_gate_is_checked_before_dataset_or_test_bytes_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = Path(tmp) / "bad_gate.json"
            gate.write_text("{}", encoding="utf-8")
            args = SimpleNamespace(config="configs/baseline.yaml", out=str(Path(tmp) / "new"),
                                   run=str(Path(tmp) / "run"), device="cpu", split="test", val_gate=str(gate))
            with patch.object(evaluate, "load_frozen_run", return_value=SimpleNamespace()), \
                    patch.object(evaluate, "build_dataset", side_effect=AssertionError("read test dataset")), \
                    patch.object(evaluate, "_verify_data_files", side_effect=AssertionError("read test bytes")):
                with self.assertRaises(SystemExit):
                    evaluate.do_evaluate(args)
            self.assertFalse((Path(tmp) / "new").exists())


class DataIdentityTests(unittest.TestCase):
    def test_image_and_gt_bytes_are_both_checked_against_frozen_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, mask = root / "image.jpg", root / "mask.png"
            image.write_bytes(b"original image")
            mask.write_bytes(b"original gt")
            records = [{"path": path.name, "kind": kind, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for kind, path in (("image", image), ("mask", mask))]
            identity = {"data_root": str(root), "splits": {"test": records}}
            evaluate._verify_data_files(identity, "test")
            for path, original in ((image, b"original image"), (mask, b"original gt")):
                with self.subTest(kind=path.name):
                    path.write_bytes(b"modified content")
                    with self.assertRaises(SystemExit):
                        evaluate._verify_data_files(identity, "test")
                    path.write_bytes(original)

    def test_preview_nan_shape_missing_keys_fail_closed(self):
        from segformer_lab import pilot

        fresh = {"preview": np.zeros((3, 4), dtype=np.float32)}
        for stored in ({}, {"preview": np.full((3, 4), np.nan)},
                       {"preview": np.zeros((4, 3), dtype=np.float32)}):
            with self.subTest(stored=list(stored)):
                with self.assertRaises(ValueError):
                    pilot.compare_probabilities(stored, fresh, threshold=0.5)

class MaskTests(unittest.TestCase):
    def test_writes_uint8_mask_and_rejects_non_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "mask.png"
            mask = np.zeros((10, 12), dtype=np.uint8)
            mask[3:5, 4:6] = 1
            inference.save_binary_mask(path, mask)
            with Image.open(path) as handle:
                array = np.asarray(handle)
            self.assertEqual(array.shape, (10, 12))
            np.testing.assert_array_equal(array, mask)
            with self.assertRaises(ValueError):
                inference.save_binary_mask(Path(tmp) / "bad.png", (mask * 255).astype(np.uint8))
            with self.assertRaises(ValueError):
                inference.save_binary_mask(Path(tmp) / "bad2.png", mask.astype(np.float32))


class FrozenRunGuardTests(unittest.TestCase):
    def test_incomplete_run_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"path": Path(tmp) / "config.yaml", "paths": {}}
            with self.assertRaises(inference.FrozenRunError):
                inference.load_frozen_run(config, Path(tmp) / "missing_run", device="cpu")

    def test_missing_preview_is_refused_before_checkpoint_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            for name in ("checkpoints/best.pt", "processor/preprocessor_config.json", "model_config.json",
                         "summary.json", "resolved_config.yaml", "artifact_manifest.json"):
                path = run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"placeholder")
            with patch.object(torch, "load", side_effect=AssertionError("deserialized incomplete bundle")):
                with self.assertRaisesRegex(inference.FrozenRunError, "pre_save_probabilities"):
                    inference.load_frozen_run({"path": Path(tmp) / "config.yaml", "paths": {}}, run, device="cpu")


class NoGroundTruthPredictionTests(unittest.TestCase):
    def test_image_only_cli_writes_original_size_binary_and_float_probability(self):
        class Processor:
            def __call__(self, *, images, return_tensors):
                return {"pixel_values": torch.zeros((1, 3, 512, 512), dtype=torch.float32)}

        class Model(torch.nn.Module):
            def forward(self, *, pixel_values):
                # Equal logits produce exactly 0.5, so >=0.5 must export ones.
                return SimpleNamespace(logits=torch.zeros((1, 2, 2, 2), dtype=torch.float32))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            image = root / "image.png"
            Image.new("RGB", (2, 3), color=(10, 20, 30)).save(image)
            frozen = SimpleNamespace(model=Model().eval(), processor=Processor(), run_dir=run,
                                     identity={"source_files": {}, "best_sha256": "unit-fixture", "best_epoch": 39})
            args = SimpleNamespace(config="configs/baseline.yaml", image=str(image), out=str(root / "out"),
                                   run=str(run), device="cpu")
            with patch.object(predict, "load_frozen_run", return_value=frozen), \
                    patch("segformer_lab.data.PCB1Dataset", side_effect=AssertionError("GT dataset accessed")):
                self.assertEqual(predict.do_predict(args), 0)
            with Image.open(root / "out" / "mask.png") as handle:
                np.testing.assert_array_equal(np.asarray(handle), np.ones((3, 2), dtype=np.uint8))
            with np.load(root / "out" / "probability.npz") as saved:
                np.testing.assert_array_equal(saved["probability"], np.full((3, 2), 0.5, dtype=np.float32))
                self.assertEqual(saved["probability"].dtype, np.float32)
            meta = json.loads((root / "out" / "prediction_meta.json").read_text(encoding="utf-8"))
            self.assertFalse(meta["image_uses_gt"])
            self.assertEqual(meta["source_size"], [3, 2])
            self.assertTrue((root / "out" / "heatmap.png").is_file())
            self.assertTrue((root / "out" / "overlay.png").is_file())


class PreprocessEquivalenceTests(unittest.TestCase):
    """prepare_image must equal the dataset's own preprocessing for the same file."""

    def test_matches_dataset_sample(self):
        from segformer_lab.config import load_config
        from segformer_lab import data

        config = load_config("configs/baseline.yaml")
        dataset = data.PCB1Dataset(config, "val", flip="none", return_original_mask=True,
                                   ids_file=Path(config["paths"]["logging.fixed_preview_manifest"]))
        item = dataset[0]
        prepared = inference.prepare_image(dataset.processor,
                                           dataset.data_root / item["image_id"], config)
        self.assertTrue(torch.equal(prepared["pixel_values"], item["pixel_values"]))
        for key in ("orig_w", "orig_h", "new_w", "new_h", "pad_left", "pad_top"):
            self.assertEqual(prepared["geometry"][key], item["geometry"][key])
        self.assertEqual(prepared["source_size"], [item["geometry"]["orig_h"], item["geometry"]["orig_w"]])


if __name__ == "__main__":
    unittest.main()
