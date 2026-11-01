"""Small CPU checks for the evidence helper; no training or dataset decoding."""
import unittest
from pathlib import Path

import numpy as np
import torch

from segformer_lab import resume_check as rc


class EvidenceChecks(unittest.TestCase):
    def test_state_equal_checks_nested_tensor_dtype_keys_and_numpy(self):
        original = {"model": torch.tensor([1.0]), "rng": ("MT", np.array([1, 2], dtype="uint32"))}
        self.assertTrue(rc.state_equal(original, {"model": original["model"].clone(),
                                                 "rng": ("MT", original["rng"][1].copy())}))
        self.assertFalse(rc.state_equal(original, {**original, "extra": 1}))
        self.assertFalse(rc.state_equal(original, {**original, "model": torch.tensor([1.0], dtype=torch.float64)}))
        self.assertFalse(rc.state_equal(original, {**original, "rng": ("MT", np.array([1, 2], dtype="int64"))}))
        self.assertFalse(rc.state_equal({"groups": [{"lr": 1.0}]}, {"groups": [{"lr": 2.0}]}))

    def test_boundary_requires_frozen_epoch_one_last(self):
        good = {"kind": "last", "epoch_completed": 1, "next_epoch": 2}
        rc.validate_checkpoint(good, boundary=True)
        for changed in ({"kind": "best"}, {"epoch_completed": 3, "next_epoch": 4}, {"next_epoch": 3}):
            with self.assertRaises(ValueError):
                rc.validate_checkpoint({**good, **changed}, boundary=True)

    def test_continuation_requires_actual_final_last_path_and_epoch(self):
        good = {"kind": "last", "epoch_completed": 3, "next_epoch": 4}
        rc.validate_checkpoint(good, boundary=False, path=Path("run/checkpoints/last.pt"), final_epoch=3)
        for payload, path in (({**good, "kind": "best"}, Path("run/checkpoints/last.pt")),
                              (good, Path("run/checkpoints/best.pt")),
                              ({**good, "epoch_completed": 1, "next_epoch": 2}, Path("run/checkpoints/last.pt"))):
            with self.assertRaises(ValueError):
                rc.validate_checkpoint(payload, boundary=False, path=path, final_epoch=3)

    def test_probability_limit_cannot_turn_a_large_difference_into_success(self):
        self.assertFalse(rc.compare_probabilities(torch.tensor([0.1]), torch.tensor([0.1001]))["within_tolerance"])
        self.assertFalse(rc.compare_probabilities(torch.tensor([0.499999]), torch.tensor([0.500001]))["within_tolerance"])
        self.assertTrue(rc.compare_probabilities(torch.tensor([0.1]), torch.tensor([0.100001]))["within_tolerance"])

    def test_probability_comparison_refuses_nonfinite_or_shape_mismatch(self):
        for right in (torch.tensor([float("nan")]), torch.tensor([[0.1]])):
            with self.assertRaises(ValueError):
                rc.compare_probabilities(torch.tensor([0.1]), right)

    def test_output_guard_refuses_existing_files_and_run_descendants(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            with self.assertRaises(ValueError):
                rc.protect_output(run / "new.json", [run])
            existing = Path(directory) / "old.json"
            existing.write_text("original")
            with self.assertRaises(ValueError):
                rc.protect_output(existing, [run])
            self.assertEqual(existing.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
