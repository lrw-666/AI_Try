"""Small CPU acceptance gates; no training, dataset decode, or CUDA allocation."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from segformer_lab import full_check


class FullCheckTests(unittest.TestCase):
    def fixture(self):
        val = {"global": {"foreground_iou": 0.5},
               "anomaly_images": {"macro": {"iou": {"mean": 0.4}}}}
        best = {"epoch": 1, "iou": 0.5}
        identity = {"label": "start", "git": {"head": "h", "diff_sha256": "d"}}
        scheduler = {"base_lr": 1e-4, "min_lr": 1e-6, "total_steps": 2,
                     "warmup_steps": 1, "completed": 2}
        entry = {"epoch": 1, "successful_total": 2, "attempts_total": 3,
                 "skipped_total": 1, "scheduler_completed": 2, "best_so_far": best,
                 "val": val, "diagnostics": {"empty_prediction_count": 0}}
        summary = {"best": best, "epochs": [entry], "device": "cpu",
                   "training": {"epochs_completed": 1}, "scheduler": scheduler,
                   "optimizer": {"decay_tensors": 1, "no_decay_tensors": 1,
                                 "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
                   "data": {"batch_size": 2, "workers": 0, "shuffle": True,
                            "drop_last": False, "batches_per_epoch": 3}}
        groups = [{"params": [i], "lr": 1e-6, "weight_decay": decay,
                   "betas": (0.9, 0.999), "eps": 1e-8}
                  for i, decay in ((0, 0.01), (1, 0.0))]
        payload = {"schema": full_check.CHECKPOINT_SCHEMA, "stage": "11", "kind": "last",
                   "epoch_completed": 1, "next_epoch": 2,
                   "step": {"successful": 2, "attempts": 3, "skipped": 1},
                   "metrics": {"epoch": 1, "val": val, "diagnostics": entry["diagnostics"]},
                   "best": best, "identity": {"label": "start"}, "git": identity["git"],
                   "scheduler": scheduler, "scheduler_plan": {"total_steps": 2, "warmup_steps": 1},
                   "lr": 1e-6, "model": {"weight": torch.ones(1)},
                   "optimizer": {"param_groups": groups, "state": {
                       i: {"step": torch.tensor(2.), "exp_avg": torch.zeros(1),
                           "exp_avg_sq": torch.zeros(1)} for i in (0, 1)}},
                   "optimizer_groups": [{"weight_decay": g["weight_decay"], "lr": g["lr"],
                                         "tensors": len(g["params"])} for g in groups],
                   "grad_scaler": {"scale": 1024., "growth_factor": 2., "backoff_factor": .5,
                                   "growth_interval": 2000, "_growth_tracker": 1},
                   "rng": {"python": random.Random(0).getstate(),
                           "numpy": np.random.RandomState(0).get_state(),
                           "torch": torch.get_rng_state(), "cuda": []},
                   "loader": {"generator_state": torch.Generator().get_state(), "batch_size": 2,
                              "num_workers": 0, "shuffle": True, "drop_last": False, "dataset_epoch": 1},
                   "resume_supported": True, "resume_scope": "complete_epoch_boundary"}
        return payload, summary, identity

    def check(self, payload, summary, identity):
        return full_check.verify_checkpoint(payload, "last", summary, identity,
                                            parameter_shapes={0: (1,), 1: (1,)})

    def test_full_schema_dict_counters_do_not_mutate_saved_payload(self):
        payload, summary, identity = self.fixture()
        original_metrics = copy.deepcopy(payload["metrics"])
        result = self.check(payload, summary, identity)
        self.assertTrue(result["ok"])
        self.assertEqual(result["step"], payload["step"])
        self.assertEqual(payload["metrics"], original_metrics)
        self.assertEqual(payload["schema"], full_check.CHECKPOINT_SCHEMA)

    def test_rejects_inconsistent_full_boundary(self):
        mutations = [lambda p: p.update(next_epoch=3),
                     lambda p: p["step"].update(attempts=4),
                     lambda p: p["loader"].update(dataset_epoch=0),
                     lambda p: p["loader"].update(generator_state=torch.ones(2)),
                     lambda p: p["scheduler_plan"].update(total_steps=9),
                     lambda p: p["optimizer_groups"][0].update(tensors=99)]
        for mutate in mutations:
            payload, summary, identity = self.fixture()
            mutate(payload)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.check(payload, summary, identity)

    def test_rejects_half_model_and_wrong_optimizer_moment_shape(self):
        for fault in ("half_model", "half_moment", "moment_shape", "late_step"):
            payload, summary, identity = self.fixture()
            if fault == "half_model":
                payload["model"]["weight"] = torch.ones(1, dtype=torch.float16)
            else:
                state = payload["optimizer"]["state"][1]
                state[{"half_moment": "exp_avg", "moment_shape": "exp_avg_sq",
                       "late_step": "step"}[fault]] = {
                           "half_moment": torch.zeros(1, dtype=torch.float16),
                           "moment_shape": torch.zeros(2), "late_step": torch.tensor(1.)}[fault]
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.check(payload, summary, identity)

    def test_report_path_cannot_overwrite_input_or_unrelated_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            report = root / "report.json"
            full_check.guard_report_path(run, report, protected=[root / "source.py"])
            for target in (run / "check.json", root / "source.py", root / "source.py" / "check.json"):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    full_check.guard_report_path(run, target, protected=[root / "source.py"])
            report.write_text(json.dumps({"schema": "unrelated"}))
            with self.assertRaises(ValueError):
                full_check.guard_report_path(run, report, protected=[])

    def test_observed_forward_rejects_cpu_even_when_report_claims_cuda(self):
        model = torch.nn.Linear(2, 2).eval()
        observer = full_check.ForwardObserver(model)
        try:
            with torch.inference_mode(), self.assertRaisesRegex(ValueError, "CUDA"):
                model(torch.ones(1, 2))
        finally:
            observer.close()


if __name__ == "__main__":
    unittest.main()
