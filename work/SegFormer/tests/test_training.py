"""Small CPU regressions for the training loop; no datasets or CUDA work."""
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from segformer_lab import training


class _ChosenGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, gradient):
        ctx.gradient = gradient
        return value.clone()

    @staticmethod
    def backward(ctx, incoming):
        return torch.full_like(incoming, ctx.gradient), None


class TinyModel(nn.Module):
    """Two small trainable tensors with the real trainer's probe names."""
    def __init__(self, gradients=(1.0,)):
        super().__init__()
        self.segformer = nn.Module()
        self.segformer.encoder = nn.Module()
        self.segformer.encoder.patch_embeddings = nn.ModuleList([nn.Module()])
        self.segformer.encoder.patch_embeddings[0].proj = nn.Linear(2, 1, bias=False)
        self.decode_head = nn.Module()
        self.decode_head.linear_c = nn.ModuleList([nn.Module()])
        self.decode_head.linear_c[0].proj = nn.Linear(2, 1, bias=False)
        for parameter in self.parameters():
            nn.init.constant_(parameter, 0.1)
        self.gradients = gradients
        self.calls = 0

    def forward(self, pixel_values):
        gradient = self.gradients[min(self.calls, len(self.gradients) - 1)]
        self.calls += 1
        enc = _ChosenGradient.apply(self.get_parameter(training.PROBE_ENCODER), gradient).sum()
        head = _ChosenGradient.apply(self.get_parameter(training.PROBE_HEAD), gradient).sum()
        positive = (enc + head).expand(pixel_values.shape[0], 2, 2)
        return SimpleNamespace(logits=torch.stack((-positive, positive), dim=1))


class TrainingStepTests(unittest.TestCase):
    config = {"raw": {"loss": {"ce_weight": 1.0, "foreground_dice_weight": 0.5,
                                 "dice_epsilon": 1e-6}}}

    def run_cpu(self, model, *, target=1, cap=10, amp=True, on_update=None, init_scale=2.0):
        # These four CUDA calls only measure/synchronise an actual GPU run. The
        # optimiser, loss, backward, clipping, and scaler below remain real.
        cpu_scaler = torch.amp.GradScaler
        build_optimizer = training.build_optimizer

        def capture_optimizer(*args):
            self.optimizer, info = build_optimizer(*args)
            return self.optimizer, info

        batch = {"pixel_values": torch.zeros(1, 3, 2, 2),
                 "labels": torch.ones(1, 2, 2, dtype=torch.long),
                 "image_id": ["pcb1/Anomaly/tiny.png"]}
        with ExitStack() as stack:
            for name in ("reset_peak_memory_stats", "synchronize"):
                stack.enter_context(patch.object(torch.cuda, name, return_value=None))
            stack.enter_context(patch.object(torch.cuda, "current_device", return_value=0))
            stack.enter_context(patch.object(training, "gpu_memory", return_value={}))
            stack.enter_context(patch.object(training, "build_optimizer", side_effect=capture_optimizer))
            stack.enter_context(patch.object(torch, "autocast", return_value=nullcontext()))
            stack.enter_context(patch.object(torch.amp, "GradScaler",
                side_effect=lambda device, enabled: cpu_scaler("cpu", enabled=enabled, init_scale=init_scale)))
            return training.run_training(model, [batch], self.config, "cpu",
                target_updates=target, max_attempts=cap, lr=0.01, weight_decay=0.0,
                grad_clip=1.0, amp=amp, on_update=on_update)

    def test_amp_infinite_gradient_skips_without_mutating_weights_or_adam_state(self):
        model = TinyModel((1.0, float("inf"), 1.0))
        outcome, optimizer, scaler, _ = self.run_cpu(model, target=2)
        self.assertEqual((outcome.successful, outcome.attempts, outcome.skipped), (2, 3, 1))
        skipped = outcome.records[1].as_dict()
        self.assertIs(skipped.get("optimizer_step_executed"), False)
        self.assertIs(skipped.get("parameters_unchanged"), True)
        self.assertIs(skipped.get("optimizer_steps_unchanged"), True)
        self.assertEqual((skipped["scale_before"], skipped["scale_after"]), (2.0, 1.0))
        self.assertTrue(np.isfinite(skipped["total"]))
        self.assertTrue(all(float(state["step"]) == 2 for state in optimizer.state.values()))
        self.assertEqual(scaler.get_scale(), 1.0)

    def test_finite_gradient_elements_with_overflowing_norm_stop_before_optimizer(self):
        model = TinyModel((1e30,))
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        with self.assertRaisesRegex(RuntimeError, "gradient norm"):
            self.run_cpu(model)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 5e29), rtol=0, atol=0)
        self.assertEqual(len(self.optimizer.state), 0)

    def test_overflow_created_by_unscale_is_not_a_scaler_confirmed_skip(self):
        model = TinyModel((3e38,))
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        with self.assertRaisesRegex(RuntimeError, "non-finite gradient"):
            self.run_cpu(model, init_scale=0.5)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))
        self.assertEqual(len(self.optimizer.state), 0)

    def test_disabled_amp_rejects_nonfinite_gradients_before_optimizer(self):
        model = TinyModel((float("inf"),))
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        with self.assertRaisesRegex(RuntimeError, "non-finite gradient"):
            self.run_cpu(model, amp=False)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))
        self.assertEqual(len(self.optimizer.state), 0)

    def test_a_success_resets_consecutive_skip_count(self):
        model = TinyModel((float("inf"), 1.0, float("inf"), 1.0, float("inf"), 1.0))
        outcome, optimizer, _, _ = self.run_cpu(model, target=3)
        self.assertEqual((outcome.successful, outcome.attempts, outcome.skipped), (3, 6, 3))
        self.assertTrue(all(float(state["step"]) == 3 for state in optimizer.state.values()))

    def test_three_consecutive_skips_stop_without_a_fourth_batch(self):
        model = TinyModel((float("inf"), float("inf"), float("inf"), 1.0))
        with self.assertRaisesRegex(RuntimeError, "three consecutive"):
            self.run_cpu(model)
        self.assertEqual(model.calls, 3)
        for parameter in model.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.1), rtol=0, atol=0)

    def test_finite_updates_still_match_two_adam_updates(self):
        model = TinyModel()
        outcome, optimizer, _, _ = self.run_cpu(model, target=2)
        self.assertEqual((outcome.successful, outcome.attempts, outcome.skipped), (2, 2, 0))
        self.assertTrue(all(record.as_dict().get("optimizer_step_executed") is True for record in outcome.records))
        self.assertTrue(all(float(state["step"]) == 2 for state in optimizer.state.values()))
        for parameter in model.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.08), rtol=0, atol=1e-7)

    def test_first_step_with_unmoved_probes_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "probe|parameter.*update"):
            self.run_cpu(TinyModel((0.0,)))

    def test_nonfinite_parameter_after_step_stops_before_evaluation_callback(self):
        build_optimizer = training.build_optimizer
        callback_calls = []

        def poison_after_step(model, lr, weight_decay):
            optimizer, info = build_optimizer(model, lr, weight_decay)
            def poison(*args):
                with torch.no_grad():
                    next(model.parameters()).fill_(float("nan"))
            optimizer.register_step_post_hook(poison)
            return optimizer, info

        with patch.object(training, "build_optimizer", side_effect=poison_after_step):
            with self.assertRaisesRegex(RuntimeError, "non-finite parameter"):
                self.run_cpu(TinyModel(), on_update=lambda *args: callback_calls.append(True))
        self.assertEqual(callback_calls, [])


class TinyDataset:
    split = "train"

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"pixel_values": torch.zeros(3, 2, 2), "image_id": "pcb1/Normal/tiny.png",
                "original_mask": np.zeros((2, 2), dtype=np.uint8), "flipped": False,
                "geometry": {"orig_w": 2, "orig_h": 2, "input_size": 2,
                             "pad_left": 0, "pad_top": 0, "new_w": 2, "new_h": 2,
                             "pad_right": 0, "pad_bottom": 0}}


class EvaluationModeTests(unittest.TestCase):
    config = {"raw": {"eval": {"threshold": 0.5, "precision": "ieee_fp32"}}}

    def test_evaluation_restores_callers_mode_and_uses_the_actual_split(self):
        for mode in (False, True):
            with self.subTest(training=mode):
                model = TinyModel().train(mode)
                result = training.evaluate_dataset(model, TinyDataset(), self.config, "cpu")
                self.assertEqual(model.training, mode)
                self.assertEqual(result["per_image"][0]["split"], "train")

    def test_evaluation_failure_restores_callers_mode(self):
        class BrokenDataset(TinyDataset):
            def __getitem__(self, index):
                raise RuntimeError("broken sample")
        for mode in (False, True):
            with self.subTest(training=mode):
                model = TinyModel().train(mode)
                with self.assertRaisesRegex(RuntimeError, "broken sample"):
                    training.evaluate_dataset(model, BrokenDataset(), self.config, "cpu")
                self.assertEqual(model.training, mode)


if __name__ == "__main__":
    unittest.main()
