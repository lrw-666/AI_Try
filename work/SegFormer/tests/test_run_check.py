"""Read-only reload checker: tiny CPU fixtures, no model construction."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from segformer_lab import run_check as check


class ArtifactTests(unittest.TestCase):
    def bundle(self, root):
        checkpoint = root / "checkpoints" / "model_state.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"frozen weights")
        entry = {"bytes": checkpoint.stat().st_size,
                 "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        manifest = {"run_dir": str(root), "files": {"checkpoints/model_state.pt": entry},
                    "model_state_sha256": entry["sha256"]}
        (root / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_every_listed_byte_is_checked_and_summary_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.bundle(root)
            self.assertEqual(check.verify_artifacts(root, {"artifacts": manifest})["files_checked"], 1)
            (root / "checkpoints/model_state.pt").write_bytes(b"changed weight")
            with self.assertRaisesRegex(ValueError, "identity"):
                check.verify_artifacts(root, {"artifacts": manifest})
            with self.assertRaisesRegex(ValueError, "summary.artifacts"):
                check.verify_artifacts(root, {"artifacts": {}})

    def test_manifest_path_traversal_is_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../outside", "/absolute", "checkpoints/../../outside"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "run boundary"):
                    check.run_file(root, name)

    def test_report_and_figures_cannot_overwrite_source_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            for out, figures in ((run / "report.json", None),
                                 (root / "report.json", run / "preview")):
                with self.assertRaisesRegex(ValueError, "source run"):
                    check.validate_outputs(run, out, figures)
            figures = root / "figures"
            figures.mkdir()
            (figures / "old.png").write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "empty"):
                check.validate_outputs(run, root / "report.json", figures)
            self.assertEqual((figures / "old.png").read_bytes(), b"keep")

    def test_error_report_cannot_overwrite_a_protected_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, source = root / "run", root / "source"
            run.mkdir()
            source.mkdir()
            target = source / "config.json"
            target.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protected"):
                check.validate_outputs(run, target, None, (source,))
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")


class ModelConfigTests(unittest.TestCase):
    def test_integer_label_keys_survive_the_torch_to_json_boundary(self):
        state_config = {"id2label": {0: "background", 1: "defect"}, "num_labels": 2}
        json_config = {"id2label": {"0": "background", "1": "defect"}, "num_labels": 2}
        check.verify_model_config(state_config, json_config)
        self.assertEqual(list(state_config["id2label"]), [0, 1])

    def test_changed_model_configuration_value_is_still_rejected(self):
        state_config = {"id2label": {0: "background", 1: "defect"}, "num_labels": 2}
        for changed in ({"id2label": {"0": "background", "1": "wrong"}, "num_labels": 2},
                        {"id2label": {"0": "background", "1": "defect"}, "num_labels": 3}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "training model configuration differs"):
                check.verify_model_config(state_config, changed)


class EvaluationComparisonTests(unittest.TestCase):
    def fixture(self, split="train"):
        return {"confusion": {"tp": 1, "fp": 0, "fn": 2, "tn": 3},
                "global": {"foreground_iou": 1 / 3, "na_reasons": {}},
                "images": {"total": 1}, "normal_images": {"count": 0},
                "anomaly_images": {"count": 1, "macro": {"mean": 1 / 3}},
                "per_image": [{"image_id": "a", "split": split, "label": "anomaly",
                               "tp": 1, "iou": 1 / 3, "precision_na": ""}],
                "evaluation": {"batch_size": 1, "threshold": 0.5,
                               "forward_ms": [1.0], "median_forward_ms": 1.0}}

    def test_timing_only_is_excluded(self):
        old, fresh = self.fixture(), self.fixture()
        fresh["evaluation"]["forward_ms"] = [42.0]
        fresh["evaluation"]["median_forward_ms"] = 42.0
        self.assertTrue(check.compare_evaluations(old, fresh, "train")["identical"])

    def test_all_metrics_na_and_per_image_counts_are_checked(self):
        old = self.fixture()
        for section, key, value in (("global", "foreground_iou", 0.5),
                                    ("normal_images", "count", 1),
                                    ("evaluation", "threshold", 0.4)):
            fresh = copy.deepcopy(old)
            fresh[section][key] = value
            self.assertFalse(check.compare_evaluations(old, fresh, "train")["identical"])
        for key, value in (("tp", 2), ("precision_na", "changed"), ("iou", 0.5)):
            fresh = copy.deepcopy(old)
            fresh["per_image"][0][key] = value
            self.assertFalse(check.compare_evaluations(old, fresh, "train")["identical"])

    def test_legacy_split_is_explicitly_corrected_without_editing_input(self):
        old, fresh = self.fixture("val"), self.fixture("train")
        result = check.compare_evaluations(old, fresh, "train")
        self.assertTrue(result["identical"])
        self.assertEqual(result["metadata_correction"][0],
                         {"image_id": "a", "stored_split": "val", "actual_split": "train"})
        self.assertEqual(old["per_image"][0]["split"], "val")
        with self.assertRaisesRegex(ValueError, "fresh.*split"):
            check.compare_evaluations(old, old, "train")

    def test_missing_probability_evidence_is_never_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            result = check.compare_probabilities(Path(directory) / "absent.npz", [], 0.5)
        self.assertEqual(result["status"], "not_available")
        self.assertIsNone(result["max_abs_probability_diff"])
        self.assertIsNone(result["xor_pixels_total"])
        self.assertIn("counts", result["scope"])


if __name__ == "__main__":
    unittest.main()
