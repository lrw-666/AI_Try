"""CPU regressions for the stage-11 formal entry point.

Pins the numbers the hand-off fixes in prose: 50 x 265 = 13250 planned updates,
warmup max(1, int(round(13250*0.05))) = 662 (ties-to-even), cosine span 12588,
first update 6e-5/662, 662nd update 6e-5, 13250th update 1e-6. Also pins the
formal/fixture guard split and the stage-10 boundary that must not move.
"""
import unittest
from copy import deepcopy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from segformer_lab import pilot, train_full
from segformer_lab.config import load_config
from segformer_lab.training import WarmupCosineLR


class BudgetTests(unittest.TestCase):
    def test_formal_plan_matches_the_declared_protocol(self):
        plan = train_full.planned_schedule(50, 265, 0.05)
        self.assertEqual(plan["total_steps"], 13250)
        self.assertEqual(plan["warmup_steps"], 662)     # 662.5 -> 662 (ties-to-even)
        self.assertEqual(plan["cosine_span"], 12588)

    def test_first_step_lr_is_the_documented_value(self):
        schedule = WarmupCosineLR(base_lr=6e-5, min_lr=1e-6, total_steps=13250, warmup_steps=662)
        self.assertAlmostEqual(schedule.lr_at(0), 6e-5 / 662, places=20)
        self.assertAlmostEqual(schedule.lr_at(0), 9.06344410876133e-08, places=20)

    def test_warmup_reaches_base_on_the_662nd_update(self):
        schedule = WarmupCosineLR(base_lr=6e-5, min_lr=1e-6, total_steps=13250, warmup_steps=662)
        self.assertAlmostEqual(schedule.lr_at(661), 6e-5, places=18)
        self.assertLess(schedule.lr_at(660), 6e-5)

    def test_cosine_is_monotone_and_ends_at_the_floor(self):
        schedule = WarmupCosineLR(base_lr=6e-5, min_lr=1e-6, total_steps=13250, warmup_steps=662)
        values = [schedule.lr_at(k) for k in range(662, 13250)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[-1], 1e-6, places=18)
        # 1e-6 is the cosine end point, not a warmup floor: early warmup sits below it.
        self.assertLess(schedule.lr_at(0), 1e-6)

    def test_apply_then_record_matches_the_used_lr(self):
        schedule = WarmupCosineLR(base_lr=6e-5, min_lr=1e-6, total_steps=13250, warmup_steps=662)
        optimizer = __import__("torch").optim.AdamW([__import__("torch").nn.Parameter(
            __import__("torch").zeros(1))], lr=1e-3)
        schedule.completed = 100
        self.assertAlmostEqual(schedule.apply(optimizer), schedule.lr_at(100), places=20)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], schedule.lr_at(100), places=20)


class GuardTests(unittest.TestCase):
    def test_baseline_config_passes_the_formal_guards(self):
        config = load_config("configs/baseline.yaml")
        kind, failures = train_full.config_failures(config)
        self.assertEqual(kind, "formal")
        self.assertEqual(failures, [])

    def test_fixture_config_passes_only_the_fixture_guards(self):
        config = load_config("configs/resume_fixture.yaml")
        kind, failures = train_full.config_failures(config)
        self.assertEqual(kind, "fixture")
        self.assertEqual(failures, [])
        self.assertNotEqual(train_full.formal_failures(config), [])

    def test_formal_rejects_a_subset_and_a_wrong_epoch_budget(self):
        config = deepcopy(load_config("configs/baseline.yaml"))
        config["paths"]["data.train_ids_file"] = "manifests/overfit_train_ids.txt"
        config["raw"]["train"]["epochs"] = 3
        failures = train_full.formal_failures(config)
        self.assertIn("data.train_ids_file must be null", failures)
        self.assertIn("train.epochs must be 50", failures)

    def test_formal_rejects_implicit_scheduler_or_precision_fields(self):
        config = deepcopy(load_config("configs/baseline.yaml"))
        config["raw"]["train"]["lr_scheduler"] = "none"
        config["raw"]["train"]["precision"] = "fp32"
        config["raw"]["eval"]["precision"] = "fp16"
        failures = train_full.formal_failures(config)
        self.assertIn("train.lr_scheduler must be warmup_cosine", failures)
        self.assertIn("train.precision must be fp16_amp", failures)
        self.assertIn("eval.precision must be 'ieee_fp32'", failures)

    def test_pilot_still_owns_three_epochs(self):
        for epochs in (1, 4, 50):
            config = {"raw": {"train": {"flip": "random", "amp": True,
                                        "gradient_accumulation_steps": 1,
                                        "epochs": epochs, "grad_clip_norm": 1.0},
                              "eval": {"precision": "ieee_fp32"},
                              "data": {"workers": 0}},
                      "paths": {"data.train_ids_file": None}}
            with self.subTest(epochs=epochs):
                self.assertEqual(len(pilot.config_failures(config)), 1)

    def test_unknown_stage_kind_is_rejected(self):
        config = deepcopy(load_config("configs/baseline.yaml"))
        config["raw"]["stage"]["kind"] = "production"
        kind, failures = train_full.config_failures(config)
        self.assertEqual(kind, "production")
        self.assertTrue(failures)


class ResumeSafetyTests(unittest.TestCase):
    """Exercise rejection before a checkpoint can mutate a run or RNG."""

    def helper(self, name):
        value = getattr(train_full, name, None)
        self.assertTrue(callable(value), f"missing resume safety helper: {name}")
        return value

    def test_same_basename_in_another_parent_is_not_the_same_run(self):
        validate = self.helper("_validate_resume_path")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "one" / "run"
            checkpoint = root / "two" / "run" / "checkpoints" / "last.pt"
            with self.assertRaises(SystemExit):
                validate(target, checkpoint)
            with self.assertRaises(SystemExit):
                validate(target, target / "checkpoints" / "best.pt")
            validate(target, target / "checkpoints" / "last.pt")

    def write_logs(self, root, *, event=True):
        epochs = [{"epoch": 1, "attempts_total": 2, "successful_total": 1,
                   "skipped_total": 1, "scheduler_completed": 1}]
        if event:
            epochs.append({"event": "resume", "from_epoch": 1, "next_epoch": 2})
        updates = [{"epoch": 1, "attempt": 1, "skipped": False, "success_index": 1},
                   {"epoch": 1, "attempt": 2, "skipped": True, "success_index": None}]
        for name, rows in (("epochs", epochs), ("updates", updates)):
            (root / f"{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return {"epoch_completed": 1, "step": {"attempts": 2, "successful": 1, "skipped": 1}}

    def test_second_resume_filters_event_rows_from_epoch_history(self):
        validate = self.helper("_validated_resume_logs")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self.write_logs(root)
            rows = validate(root, payload)
            self.assertEqual([row["epoch"] for row in rows], [1])

    def test_uncommitted_updates_and_partial_json_fail_without_rewriting(self):
        validate = self.helper("_validated_resume_logs")
        for suffix in ('{"epoch":2,"attempt":3,"skipped":false}\n', '{"epoch":2'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = self.write_logs(root)
                path = root / "updates.jsonl"
                original = path.read_bytes() + suffix.encode()
                path.write_bytes(original)
                with self.assertRaises(SystemExit):
                    validate(root, payload)
                self.assertEqual(path.read_bytes(), original)

    def test_missing_committed_epoch_or_duplicated_attempt_refuses_resume(self):
        validate = self.helper("_validated_resume_logs")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self.write_logs(root)
            (root / "epochs.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(SystemExit):
                validate(root, payload)
            self.write_logs(root)
            path = root / "updates.jsonl"
            path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
            with self.assertRaises(SystemExit):
                validate(root, payload)

    def test_saved_processor_is_bound_to_all_datasets(self):
        build = self.helper("_make_datasets")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            manifest.write_text("image,mask,label\nnormal.jpg,,normal\n", encoding="utf-8")
            ids = root / "preview.txt"
            ids.write_text("normal.jpg\n", encoding="utf-8")
            config = deepcopy(load_config("configs/baseline.yaml"))
            config["paths"].update({"data.train_manifest": manifest, "data.val_manifest": manifest,
                                    "logging.fixed_preview_manifest": ids,
                                    "model.pretrained_dir": root / "must-not-be-opened"})
            processor = SimpleNamespace(do_resize=False, do_reduce_labels=False,
                                        do_rescale=True, do_normalize=True,
                                        image_mean=[0.1, 0.2, 0.3], image_std=[1.0, 1.0, 1.0])
            datasets = build(config, processor)
            self.assertEqual(len(datasets), 3)
            for dataset in datasets:
                self.assertIs(dataset.processor, processor)
                np.testing.assert_array_equal(dataset.image_mean[:, 0, 0],
                                              np.asarray([0.1, 0.2, 0.3], dtype=np.float32))

    def test_identity_rejects_changed_eval_ids_processor_or_code_paths(self):
        validate = self.helper("_validate_resume_identity")
        identity = {"config": {"sha256": "cfg", "path": "config.yaml"},
                    "manifests": {split: {"sha256": split} for split in ("train", "val", "test")},
                    "split_meta_sha256": "meta", "preview_ids_sha256": "preview",
                    "train_ids_file": None, "eval_ids_file": {"sha256": "eval"},
                    "snapshot": {"revision": "rev", "weights_sha256": "weight"},
                    "snapshot_files": {"config.json": "config", "pytorch_model.bin": "weight"},
                    "snapshot_source_sha256": "source", "saved_processor": {"preprocessor_config.json": "p"},
                    "resolved_paths": {"data.root": "root"}, "code": {"paths.py": "paths"},
                    "resolved_config": {"seed": 42},
                    "versions": {"python": "same"}}
        validate(identity, deepcopy(identity))
        for key in ("eval_ids_file", "saved_processor", "resolved_paths", "code", "manifests"):
            changed = deepcopy(identity)
            changed[key] = {"changed": True}
            with self.subTest(key=key), self.assertRaises(SystemExit):
                validate(identity, changed)

    def payload(self):
        # Hand-counted fixture: 12 images, B2, epoch 1 => 6 attempts, 5 updates.
        config = deepcopy(load_config("configs/resume_fixture.yaml"))
        config["raw"]["train"]["amp"] = False  # CPU guard fixture; no forward or CUDA.
        rng = {"python": __import__("random").getstate(), "numpy": np.random.get_state(),
               "torch": torch.get_rng_state(), "cuda": []}
        lr = WarmupCosineLR(base_lr=6e-5, min_lr=1e-6, total_steps=18,
                            warmup_steps=1, completed=5).lr_at(5)
        payload = {"schema": train_full.CHECKPOINT_SCHEMA, "stage": "11", "kind": "last",
                   "resume_scope": "complete_epoch_boundary", "resume_supported": True,
                   "epoch_completed": 1, "next_epoch": 2,
                   "step": {"attempts": 6, "successful": 5, "skipped": 1},
                   "scheduler": {"base_lr": 6e-5, "min_lr": 1e-6, "total_steps": 18,
                                 "warmup_steps": 1, "completed": 5},
                   "scheduler_plan": {"total_steps": 18, "warmup_steps": 1},
                   "lr": lr,
                   "best": {"epoch": 1, "iou": 0.1}, "metrics": {"epoch": 1},
                   "model_config": {"id2label": {0: "background", 1: "defect"}},
                   "model": {"weight": torch.ones(1)},
                   "optimizer": {"state": {0: {"step": torch.tensor(5.),
                                                "exp_avg": torch.zeros(1),
                                                "exp_avg_sq": torch.zeros(1)}},
                                 "param_groups": [{"params": [0], "lr": lr}]}, "grad_scaler": {},
                   "loader": {"batch_size": 2, "shuffle": True, "num_workers": 0,
                              "drop_last": False, "dataset_epoch": 1,
                              "generator_state": torch.Generator().manual_seed(42).get_state()},
                   "rng": rng, "runtime": {"device_type": "cpu", "cuda_device_count": 0}}
        return config, payload

    def test_optimizer_slots_cannot_silently_disappear(self):
        config, payload = self.payload()
        validate = self.helper("_validate_resume_payload")
        validate(payload, config, "cpu", batches_per_epoch=6)
        missing_state = deepcopy(payload)
        missing_state["optimizer"]["state"] = {}
        missing_moment = deepcopy(payload)
        del missing_moment["optimizer"]["state"][0]["exp_avg"]
        wrong_step = deepcopy(payload)
        wrong_step["optimizer"]["state"][0]["step"] = torch.tensor(4.)
        for changed in (missing_state, missing_moment, wrong_step):
            with self.assertRaises(SystemExit):
                validate(changed, config, "cpu", batches_per_epoch=6)

    def test_corrupt_budget_counters_loader_and_rng_are_rejected(self):
        validate = self.helper("_validate_resume_payload")
        config, payload = self.payload()
        # The literal LR is independently calculated for completed=5 / 18.
        schedule = WarmupCosineLR(base_lr=6e-5, min_lr=1e-6, total_steps=18, warmup_steps=1, completed=5)
        payload["lr"] = schedule.lr_at(5)
        validate(payload, config, "cpu", batches_per_epoch=6)
        corruptions = [("next_epoch", 3), ("step", {"attempts": 6, "successful": 6, "skipped": 1}),
                       ("scheduler_plan", {"total_steps": 12, "warmup_steps": 1}),
                       ("rng", {"torch": torch.get_rng_state()}),
                       ("loader", {**payload["loader"], "shuffle": False}),
                       ("model", {"weight": torch.tensor([float("nan")])})]
        for key, value in corruptions:
            changed = deepcopy(payload)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(SystemExit):
                validate(changed, config, "cpu", batches_per_epoch=6)

    def test_real_cpu_checkpoint_restores_optimizer_rng_and_preserves_config(self):
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        from segformer_lab.training import build_optimizer
        import random

        config = deepcopy(load_config("configs/resume_fixture.yaml"))
        config["raw"]["train"]["amp"] = False
        model = SegformerForSemanticSegmentation(SegformerConfig(
            hidden_sizes=[4, 8, 12, 16], depths=[1, 1, 1, 1],
            num_attention_heads=[1, 1, 1, 1], mlp_ratios=[2, 2, 2, 2],
            decoder_hidden_size=8, id2label={0: "background", 1: "defect"}))
        optimizer, _ = build_optimizer(model, 6e-5, 0.01)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        for _ in range(5):
            optimizer.step()
        schedule = WarmupCosineLR(base_lr=6e-5, min_lr=1e-6, total_steps=18, warmup_steps=1, completed=5)
        schedule.apply(optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        dataset = torch.utils.data.TensorDataset(torch.zeros(12, 1))
        dataset.epoch = 1
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,
                                             generator=torch.Generator().manual_seed(42))
        list(loader)
        identity = {key: {} for key in ("config", "manifests", "train_ids_file", "eval_ids_file",
                    "snapshot", "snapshot_files", "saved_processor", "resolved_paths", "resolved_config", "code", "versions")}
        identity.update(split_meta_sha256="meta", preview_ids_sha256="preview",
                        snapshot_source_sha256="source")
        with tempfile.TemporaryDirectory() as temporary, patch("torch.cuda.is_available", return_value=False):
            root = Path(temporary)
            checkpoint = train_full.save_checkpoint(
                root, "last", model=model, optimizer=optimizer, scaler=scaler, schedule=schedule,
                loader=loader, dataset=dataset, epoch_completed=1, successful=5, attempts=6,
                skipped=1, best={"epoch": 1, "iou": 0.1}, metrics={"epoch": 1}, identities=identity)
            payload = train_full.read_checkpoint(checkpoint)
            model_config_before = deepcopy(payload["model_config"])
            random.random(), np.random.random(), torch.rand(1)
            restored = train_full.restore_checkpoint(payload, config, "cpu", identities=identity,
                                                       batches_per_epoch=6)
            self.assertEqual(payload["model_config"], model_config_before)
            for name, value in restored["model"].state_dict().items():
                self.assertTrue(torch.equal(value, payload["model"][name]), name)
            self.assertEqual(restored["optimizer"].state_dict()["param_groups"],
                             payload["optimizer"]["param_groups"])
            for index, state in restored["optimizer"].state_dict()["state"].items():
                for key, value in state.items():
                    self.assertTrue(torch.equal(value, payload["optimizer"]["state"][index][key]))
            self.assertEqual(random.getstate(), payload["rng"]["python"])
            actual_numpy = np.random.get_state()
            self.assertEqual(actual_numpy[0], payload["rng"]["numpy"][0])
            np.testing.assert_array_equal(actual_numpy[1], payload["rng"]["numpy"][1])
            self.assertEqual(actual_numpy[2:], payload["rng"]["numpy"][2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), payload["rng"]["torch"]))
            self.assertTrue(torch.equal(restored["loader_state"]["generator_state"], loader.generator.get_state()))

    def test_final_artifact_inventories_never_hash_themselves(self):
        inventory = self.helper("_artifact_inventory")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("summary.json", "artifact_manifest.json", "weights.pt"):
                (root / name).write_text("old")
            inner = inventory(root, include_summary=False)
            outer = inventory(root, include_summary=True)
            self.assertEqual(list(inner["files"]), ["weights.pt"])
            self.assertEqual(list(outer["files"]), ["summary.json", "weights.pt"])


if __name__ == "__main__":
    unittest.main()
