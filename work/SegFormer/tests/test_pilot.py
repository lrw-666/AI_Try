"""CPU regressions for the stage-10 pilot control flow; no data and no CUDA work.

These pin the three things a pilot can silently get wrong:

* the warmup+cosine arithmetic and its off-by-one boundary;
* the schedule advancing on real optimizer updates only;
* per-epoch loss aggregation using pixels/images as weights instead of a mean
  of batch means.
"""
import copy
import json
import random
import tempfile
import unittest
import copy
from pathlib import Path

import numpy as np

import torch
from torch import nn

from segformer_lab import pilot, training


class CheckpointConfigurationTests(unittest.TestCase):
    def test_transformers_constructor_preserves_saved_configuration(self):
        payload = {"model_config": {"model_type": "segformer",
                                    "id2label": {0: "background", 1: "defect"},
                                    "label2id": {"background": 0, "defect": 1}}}
        before = copy.deepcopy(payload)
        config = pilot._checkpoint_config(payload)
        self.assertEqual(config.num_labels, 2)
        self.assertEqual(payload, before)


class WarmupCosineTests(unittest.TestCase):
    def build(self, base=6e-5, minimum=1e-6, total=795, warmup=40):
        return training.WarmupCosineLR(base_lr=base, min_lr=minimum,
                                       total_steps=total, warmup_steps=warmup)

    def test_warmup_reaches_base_exactly_and_never_exceeds_it(self):
        schedule = self.build()
        self.assertAlmostEqual(schedule.lr_at(0), 6e-5 / 40, places=15)
        self.assertAlmostEqual(schedule.lr_at(38), 6e-5 * 39 / 40, places=15)
        self.assertAlmostEqual(schedule.lr_at(39), 6e-5, places=15)
        # float32/float64 rounding can push the exact boundary a few ulp above
        # base_lr; the contract is "never meaningfully above", not "bitwise <=".
        self.assertLessEqual(max(schedule.lr_at(k) for k in range(0, 46)), 6e-5 * (1 + 1e-12))

    def test_cosine_decays_monotonically_and_ends_at_min_lr(self):
        schedule = self.build()
        values = [schedule.lr_at(k) for k in range(40, 795)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[-1], 1e-6, places=15)
        self.assertGreater(values[0], 5.9e-5)

    def test_single_epoch_budget_also_works(self):
        schedule = self.build(total=265, warmup=13)
        self.assertAlmostEqual(schedule.lr_at(12), 6e-5, places=15)
        self.assertAlmostEqual(schedule.lr_at(264), 1e-6, places=15)

    def test_state_dict_round_trip_keeps_the_step(self):
        schedule = self.build()
        schedule.completed = 123
        restored = training.WarmupCosineLR(**{k: v for k, v in schedule.state_dict().items()})
        self.assertEqual(restored.completed, 123)
        self.assertEqual(restored.lr_at(200), schedule.lr_at(200))

    def test_rejects_inconsistent_budgets(self):
        with self.assertRaises(ValueError):
            training.WarmupCosineLR(base_lr=1e-4, min_lr=1e-3, total_steps=10, warmup_steps=2)
        with self.assertRaises(ValueError):
            training.WarmupCosineLR(base_lr=1e-4, min_lr=1e-6, total_steps=10, warmup_steps=11)


class ScheduleAdvanceTests(unittest.TestCase):
    def setUp(self):
        self.parameter = nn.Parameter(torch.zeros(1))
        self.optimizer = torch.optim.AdamW([self.parameter], lr=1e-4)
        self.schedule = training.WarmupCosineLR(base_lr=1e-4, min_lr=1e-6,
                                                total_steps=10, warmup_steps=2)
        self.schedule.apply(self.optimizer)

    def test_skip_does_not_advance_the_schedule_or_the_lr(self):
        before = self.optimizer.param_groups[0]["lr"]
        advanced = pilot.advance_schedule(self.schedule, {"skipped": True}, self.optimizer)
        self.assertFalse(advanced)
        self.assertEqual(self.schedule.completed, 0)
        self.assertEqual(self.optimizer.param_groups[0]["lr"], before)

    def test_real_update_advances_by_exactly_one(self):
        advanced = pilot.advance_schedule(self.schedule, {"skipped": False}, self.optimizer)
        self.assertTrue(advanced)
        self.assertEqual(self.schedule.completed, 1)
        self.assertEqual(self.optimizer.param_groups[0]["lr"], self.schedule.lr_at(1))


class LossAggregationTests(unittest.TestCase):
    def test_ce_is_weighted_by_valid_pixels_not_by_batch(self):
        tally = pilot._LossTally()
        tally.add(ce=1.0, dice=0.0, valid_pixels=300_000, positive_images=0)
        tally.add(ce=5.0, dice=0.0, valid_pixels=100_000, positive_images=0)
        result = tally.as_dict(ce_weight=1.0, dice_weight=0.5)
        self.assertAlmostEqual(result["ce"], 2.0, places=12)
        self.assertNotAlmostEqual(result["ce"], 3.0, places=12)
        self.assertEqual(result["valid_pixels"], 400_000)

    def test_dice_is_weighted_by_positive_images_only(self):
        tally = pilot._LossTally()
        tally.add(ce=0.0, dice=0.2, valid_pixels=10, positive_images=1)
        tally.add(ce=0.0, dice=0.0, valid_pixels=10, positive_images=0)
        tally.add(ce=0.0, dice=0.8, valid_pixels=10, positive_images=1)
        result = tally.as_dict(ce_weight=1.0, dice_weight=0.5)
        self.assertAlmostEqual(result["dice"], 0.5, places=12)
        self.assertEqual(result["positive_images"], 2)

    def test_total_uses_the_shared_weights(self):
        tally = pilot._LossTally()
        tally.add(ce=0.4, dice=0.5, valid_pixels=100, positive_images=2)
        result = tally.as_dict(ce_weight=1.0, dice_weight=0.5)
        self.assertAlmostEqual(result["total"], 0.4 + 0.25, places=12)


class ConfigGuardTests(unittest.TestCase):
    def base(self):
        return {
            "raw": {
                "train": {"flip": "random", "amp": True, "gradient_accumulation_steps": 1,
                          "epochs": 3, "grad_clip_norm": 1.0},
                "eval": {"precision": "ieee_fp32"},
                "data": {"workers": 0},
            },
            "paths": {"data.train_ids_file": None},
        }

    def test_clean_pilot_config_passes(self):
        self.assertEqual(pilot.config_failures(self.base()), [])

    def test_each_boundary_is_enforced(self):
        cases = [
            (("raw", "train", "flip"), "none"),
            (("raw", "eval", "precision"), "fp16"),
            (("raw", "train", "amp"), False),
            (("paths", "data.train_ids_file"), "manifests/overfit_train_ids.txt"),
            (("raw", "train", "gradient_accumulation_steps"), 2),
            (("raw", "data", "workers"), 4),
            (("raw", "train", "epochs"), 0),
            (("raw", "train", "epochs"), 4),
            (("raw", "train", "epochs"), 50),
            (("raw", "train", "grad_clip_norm"), 0.0),
        ]
        for path, value in cases:
            config = self.base()
            target = config
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(field=".".join(path)):
                self.assertEqual(len(pilot.config_failures(config)), 1)


class ReloadGateTests(unittest.TestCase):
    def test_probability_gate_rejects_nan_even_after_a_clean_sample(self):
        fresh = {"a": np.zeros((2, 2)), "b": np.zeros((2, 2))}
        stored = {"a": fresh["a"], "b": np.full((2, 2), np.nan)}
        with self.assertRaisesRegex(ValueError, "finite"):
            pilot.compare_probabilities(stored, fresh, threshold=0.5)

    def test_probability_gate_rejects_broadcast_shape_and_extra_keys(self):
        for stored in ({"a": np.zeros((1, 2, 2))},
                       {"a": np.zeros((2, 2)), "extra": np.zeros((2, 2))}):
            with self.subTest(stored=list(stored)):
                with self.assertRaises(ValueError):
                    pilot.compare_probabilities(stored, {"a": np.zeros((2, 2))}, threshold=0.5)

    def test_probability_gate_rejects_threshold_crossing_within_atol(self):
        result = pilot.compare_probabilities({"a": np.array([[0.499999]])},
                                             {"a": np.array([[0.500001]])}, threshold=0.5)
        self.assertFalse(result["ok"])
        self.assertEqual(result["xor_pixels_total"], 1)

    def test_probability_gate_accepts_identical_original_size_maps(self):
        values = {"a": np.array([[0.0, 0.5], [0.9, 1.0]])}
        result = pilot.compare_probabilities(values, values, threshold=0.5)
        self.assertTrue(result["ok"])
        self.assertEqual(result["total_pixels"], 4)

    def test_metrics_gate_detects_per_image_change_with_same_global(self):
        saved = {"global": {"foreground_iou": 0.5}, "confusion": {"tp": 1},
                 "per_image": [{"image_id": "a", "tp": 1}]}
        fresh = copy.deepcopy(saved)
        fresh["per_image"][0]["tp"] = 2
        self.assertFalse(pilot.compare_metrics(saved, fresh)["ok"])

    def test_inventory_rejects_modified_bytes_and_conflicting_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "weights.pt"
            item.write_bytes(b"abc")
            files = {"weights.pt": {"bytes": 3, "sha256": pilot.sha256_file(item)}}
            (root / "artifact_manifest.json").write_text(json.dumps({"files": files}))
            summary = {"artifacts": {"files": copy.deepcopy(files)}}
            self.assertTrue(pilot.verify_inventory(root, summary)["ok"])
            summary["artifacts"]["files"]["weights.pt"]["sha256"] = "wrong"
            with self.assertRaises(ValueError):
                pilot.verify_inventory(root, summary)
            summary["artifacts"]["files"] = files
            item.write_bytes(b"xyz")
            with self.assertRaises(ValueError):
                pilot.verify_inventory(root, summary)

    def test_identity_gate_detects_manifest_drift(self):
        start = {"code": {}, "config": {"sha256": "a"}, "manifests": {"test": "a"},
                 "split_meta_sha256": "a", "preview_ids_sha256": "a",
                 "git": {"head": "a", "diff_sha256": "a"}}
        end = copy.deepcopy(start)
        end["manifests"]["test"] = "b"
        self.assertFalse(pilot.identity_unchanged(start, end))

    def test_val_summary_preserves_global_na_reasons(self):
        result = {"images": {}, "confusion": dict.fromkeys(("tp", "fp", "fn", "tn", "pixels"), 0),
                  "global": {"na_reasons": {"precision": "empty"}}, "normal_images": {},
                  "anomaly_images": {"count": 0, "macro": {}}, "evaluation": {}, "per_image": []}
        self.assertEqual(pilot._val_summary(result)["na_reasons"], {"precision": "empty"})

    def checkpoint_fixture(self):
        val = {"global": {"foreground_iou": 0.5}, "anomaly_images": {"macro": {"iou": {"mean": 0.4}}}}
        best = {"epoch": 1, "iou": 0.5}
        identity = {"label": "start", "git": {"head": "h", "diff_sha256": "d"}}
        schedule = {"base_lr": 1e-4, "min_lr": 1e-6, "total_steps": 2, "warmup_steps": 1, "completed": 2}
        entry = {"epoch": 1, "successful_total": 2, "attempts_total": 3, "skipped_total": 1,
                 "scheduler_completed": 2, "best_so_far": best, "val": val}
        summary = {"best": best, "epochs": [entry], "training": {"epochs_completed": 1},
                   "scheduler": schedule, "device": "cpu", "optimizer": {"weight_decay": 0.01,
                   "decay_tensors": 1, "no_decay_tensors": 1, "betas": [0.9, 0.999], "eps": 1e-8}}
        states = {index: {"step": torch.tensor(2.0), "exp_avg": torch.zeros(1),
                          "exp_avg_sq": torch.zeros(1)} for index in (0, 1)}
        groups = [{"params": [index], "lr": 1e-6, "betas": (0.9, 0.999),
                   "eps": 1e-8, "weight_decay": decay} for index, decay in ((0, 0.01), (1, 0.0))]
        payload = {"schema": pilot.CHECKPOINT_SCHEMA, "kind": "best", "epoch_completed": 1,
                   "global_step": 2, "attempts": 3, "skipped": 1, "best": best,
                   "metrics": {"epoch": 1, "val": val, "iou": 0.5, "macro_iou": 0.4},
                   "identity": {"label": "start"}, "git_head": "h", "git_diff_sha256": "d",
                   "scheduler": schedule, "lr": 1e-6, "model": {"weight": torch.ones(1)},
                   "optimizer": {"param_groups": groups, "state": states},
                   "grad_scaler": {"scale": 1024.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                                   "growth_interval": 2000, "_growth_tracker": 1},
                   "rng": {"python": random.Random(0).getstate(), "numpy": np.random.RandomState(0).get_state(),
                           "torch": torch.get_rng_state(), "cuda": []}}
        return payload, summary, identity

    def test_checkpoint_gate_checks_every_optimizer_parameter_step(self):
        payload, summary, identity = self.checkpoint_fixture()
        self.assertTrue(pilot.verify_checkpoint(payload, "best", summary, identity, parameter_count=2)["ok"])
        payload["optimizer"]["state"][1]["step"] = torch.tensor(1.0)
        with self.assertRaisesRegex(ValueError, "optimizer step"):
            pilot.verify_checkpoint(payload, "best", summary, identity, parameter_count=2)

    def test_checkpoint_gate_rejects_missing_rng_and_nonfinite_model(self):
        for kind in ("rng", "weights"):
            payload, summary, identity = self.checkpoint_fixture()
            if kind == "rng":
                del payload["rng"]["numpy"]
            else:
                payload["model"]["weight"].fill_(float("inf"))
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    pilot.verify_checkpoint(payload, "best", summary, identity, parameter_count=2)

    def test_checkpoint_gate_rejects_optimizer_group_drift(self):
        payload, summary, identity = self.checkpoint_fixture()
        payload["optimizer"]["param_groups"][0]["weight_decay"] = 0.0
        with self.assertRaisesRegex(ValueError, "optimizer group"):
            pilot.verify_checkpoint(payload, "best", summary, identity, parameter_count=2)


if __name__ == "__main__":
    unittest.main()
