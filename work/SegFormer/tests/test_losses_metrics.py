"""Unit tests for the stage 7 loss and original-space metrics.

The acceptance run (`python -m segformer_lab.metrics_check`) repeats the same
fixtures on real data and on GPU. These tests keep the rules that are easy to
break silently pinned on CPU in a second:

* Dice is ``2*sum(p*g)`` over pixels, not ``2*sum(p)*sum(g)``;
* a normal image contributes no Dice term, but does keep CE;
* a fully ignored batch is rejected instead of returning NaN;
* padding never enters the loss numerator or denominator;
* confusion counts are summed before dividing, so batch layout cannot change
  the global IoU;
* the original-resolution restore drops padding *before* upscaling, and the
  ground truth is the original mask -- not the 512 label resized back up.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from segformer_lab import geometry
from segformer_lab.losses import loss_definition, segformer_loss, upsample_logits
from segformer_lab.metrics import (
    OriginalSpaceEvaluator,
    restore_defect_probability,
    score_definition,
)


def binary_logits(pred: np.ndarray, strength: float = 12.0) -> torch.Tensor:
    pred = np.asarray(pred, dtype=bool)
    logits = np.empty((2,) + pred.shape, dtype=np.float32)
    logits[0] = np.where(pred, -strength, strength)
    logits[1] = np.where(pred, strength, -strength)
    return torch.from_numpy(logits)


def square_geometry(size: int) -> dict:
    return {
        "orig_w": size,
        "orig_h": size,
        "input_size": size,
        "pad_left": 0,
        "pad_top": 0,
        "new_w": size,
        "new_h": size,
        "pad_right": 0,
        "pad_bottom": 0,
    }


class LossDefinitionTests(unittest.TestCase):
    def test_ce_matches_hand_computation(self) -> None:
        ln3 = float(np.log(3.0))
        labels = torch.tensor([[0, 1], [1, 255]], dtype=torch.int64).unsqueeze(0)
        logits = torch.tensor(
            [[[ln3, 0.0], [0.0, 0.0]], [[0.0, ln3], [ln3, 0.0]]], dtype=torch.float32
        ).unsqueeze(0)
        terms = segformer_loss(logits, labels, dice_weight=0.0)
        self.assertAlmostEqual(float(terms.ce), -float(np.log(0.75)), places=6)
        self.assertEqual(terms.valid_pixels, 3)
        self.assertEqual(terms.ignored_pixels, 1)
        self.assertFalse(terms.upsampled)

    def test_dice_uses_pixel_intersection_not_scalar_product(self) -> None:
        # sum(p*g)=1.5, sum(p)=1.75, sum(g)=2 -> Dice 0.8 -> loss 0.2.
        # The wrong formula (2*sum(p)*sum(g)) would give 7/3.75 > 1.
        ln3 = float(np.log(3.0))
        labels = torch.tensor([[0, 1], [1, 255]], dtype=torch.int64).unsqueeze(0)
        logits = torch.tensor(
            [[[ln3, 0.0], [0.0, 0.0]], [[0.0, ln3], [ln3, 0.0]]], dtype=torch.float32
        ).unsqueeze(0)
        terms = segformer_loss(logits, labels, ce_weight=0.0)
        self.assertAlmostEqual(float(terms.dice), 0.2, places=5)
        self.assertEqual(terms.positive_images, 1)

    def test_perfect_prediction_has_near_zero_dice(self) -> None:
        labels = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64).unsqueeze(0)
        perfect = binary_logits(np.array([[0, 1], [1, 0]], dtype=bool)).unsqueeze(0)
        terms = segformer_loss(perfect, labels, ce_weight=0.0)
        self.assertLessEqual(float(terms.dice), 1e-6)

    def test_all_normal_batch_returns_graph_connected_zero(self) -> None:
        labels = torch.zeros((1, 4, 4), dtype=torch.int64)
        logits = torch.zeros((1, 2, 4, 4), dtype=torch.float32, requires_grad=True)
        terms = segformer_loss(logits, labels)
        self.assertEqual(float(terms.dice.detach()), 0.0)
        self.assertEqual(terms.positive_images, 0)
        terms.total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        # CE still supervises the normal image, so the gradient must be non-zero
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_padding_logits_do_not_move_the_loss(self) -> None:
        labels = torch.tensor([[0, 1], [1, 255]], dtype=torch.int64).unsqueeze(0)
        logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        logits[0, 1, 0, 1] = 2.0
        perturbed = logits.clone()
        perturbed[0, :, 1, 1] = torch.tensor([500.0, -500.0])
        a = segformer_loss(logits, labels)
        b = segformer_loss(perturbed, labels)
        self.assertEqual(float(a.total), float(b.total))

    def test_low_resolution_padding_rows_do_not_leak(self) -> None:
        labels = torch.full((1, 512, 512), 255, dtype=torch.int64)
        labels[0, 200:260, 100:400] = 0
        labels[0, 210:250, 150:350] = 1
        logits = torch.zeros((1, 2, 128, 128), dtype=torch.float32)
        logits[0, 1, 60:70, 40:80] = 8.0
        changed = logits.clone()
        changed[0, 0, 0:10, :] = 200.0
        changed[0, 1, 0:10, :] = -200.0
        a = segformer_loss(logits, labels)
        b = segformer_loss(changed, labels)
        self.assertTrue(a.upsampled)
        self.assertLessEqual(abs(float(a.total) - float(b.total)), 1e-7)

    def test_fully_ignored_batch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            segformer_loss(torch.zeros((1, 2, 4, 4)), torch.full((1, 4, 4), 255, dtype=torch.int64))

    def test_invalid_label_values_are_rejected(self) -> None:
        labels = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64).unsqueeze(0)
        with self.assertRaises(ValueError):
            segformer_loss(torch.zeros((1, 2, 2, 2)), labels)

    def test_upsample_is_skipped_when_sizes_match(self) -> None:
        logits = torch.zeros((1, 2, 8, 8))
        self.assertIs(upsample_logits(logits, (8, 8)), logits)
        self.assertEqual(tuple(upsample_logits(logits, (16, 16)).shape), (1, 2, 16, 16))

    def test_definition_is_explicit(self) -> None:
        definition = loss_definition()
        self.assertEqual(definition["dice"]["class"], 1)
        self.assertIn("ignore_index", definition["ce"])
        self.assertEqual(score_definition()["threshold"], 0.5)


class MetricTests(unittest.TestCase):
    def _score(self, pred: np.ndarray, gt: np.ndarray, label: str = "anomaly") -> dict:
        evaluator = OriginalSpaceEvaluator()
        evaluator.add_sample(
            binary_logits(pred),
            np.asarray(gt, dtype=np.uint8),
            square_geometry(int(gt.shape[0])),
            image_id="t",
            label=label,
        )
        return evaluator.result()

    def test_hand_computed_2x2(self) -> None:
        gt = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        pred = np.array([[0, 1], [0, 0]], dtype=bool)
        r = self._score(pred, gt)
        self.assertEqual(
            (r["confusion"]["tp"], r["confusion"]["fp"], r["confusion"]["fn"], r["confusion"]["tn"]),
            (1, 0, 1, 2),
        )
        self.assertAlmostEqual(r["global"]["foreground_iou"], 0.5, places=9)
        self.assertAlmostEqual(r["global"]["foreground_dice"], 2.0 / 3.0, places=9)
        self.assertAlmostEqual(r["global"]["precision"], 1.0, places=9)
        self.assertAlmostEqual(r["global"]["recall"], 0.5, places=9)

    def test_all_background_prediction_scores_zero(self) -> None:
        gt = np.zeros((4, 4), dtype=np.uint8)
        gt[1:3, 1:4] = 1
        r = self._score(np.zeros((4, 4), dtype=bool), gt)
        self.assertEqual(r["global"]["foreground_iou"], 0.0)
        self.assertEqual(r["global"]["foreground_dice"], 0.0)
        self.assertIsNone(r["global"]["precision"])
        self.assertIn("precision", r["global"]["na_reasons"])

    def test_normal_false_positive_counts(self) -> None:
        pred = np.zeros((4, 4), dtype=bool)
        pred[3, :3] = True
        r = self._score(pred, np.zeros((4, 4), dtype=np.uint8), label="normal")
        self.assertEqual(r["global"]["precision"], 0.0)
        self.assertIsNone(r["global"]["recall"])
        self.assertAlmostEqual(r["normal_images"]["fp_pixel_ratio"], 3.0 / 16.0, places=9)

    def test_normal_empty_prediction_is_na_not_one(self) -> None:
        r = self._score(np.zeros((4, 4), dtype=bool), np.zeros((4, 4), dtype=np.uint8), label="normal")
        self.assertIsNone(r["per_image"][0]["iou"])
        self.assertIsNone(r["per_image"][0]["dice"])
        self.assertIn("both empty", r["per_image"][0]["iou_na"])

    def test_global_is_not_mean_of_images(self) -> None:
        a_gt = np.zeros((10, 10), dtype=np.uint8)
        a_gt[0, 0] = 1
        a_pred = np.zeros((10, 10), dtype=bool)
        a_pred[0, 0] = True
        b_gt = np.zeros((10, 10), dtype=np.uint8)
        b_gt[0, 0:2] = 1
        b_pred = np.ones((10, 10), dtype=bool)
        b_pred[0, 1] = False
        evaluator = OriginalSpaceEvaluator()
        for name, pred, gt in (("a", a_pred, a_gt), ("b", b_pred, b_gt)):
            evaluator.add_sample(binary_logits(pred), gt, square_geometry(10), image_id=name, label="anomaly")
        r = evaluator.result()
        per_image = [s["iou"] for s in r["per_image"]]
        self.assertAlmostEqual(float(np.mean(per_image)), 0.505, places=9)
        self.assertAlmostEqual(r["global"]["foreground_iou"], 2.0 / 101.0, places=9)

    def test_batch_layout_cannot_change_the_result(self) -> None:
        rng = np.random.default_rng(11)
        samples = []
        for _ in range(5):
            gt = (rng.random((8, 8)) > 0.7).astype(np.uint8)
            pred = gt.astype(bool) ^ (rng.random((8, 8)) > 0.6)
            samples.append((pred, gt))
        results = []
        for order in (list(range(5)), list(reversed(range(5))), [2, 0, 4, 1, 3]):
            evaluator = OriginalSpaceEvaluator()
            for i in order:
                pred, gt = samples[i]
                evaluator.add_sample(binary_logits(pred), gt, square_geometry(8),
                                     image_id=f"s{i}", label="anomaly")
            results.append(evaluator.result()["confusion"])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])

    def test_flipped_evaluation_sample_is_rejected(self) -> None:
        evaluator = OriginalSpaceEvaluator()
        with self.assertRaises(ValueError):
            evaluator.add_sample(
                binary_logits(np.zeros((4, 4), dtype=bool)),
                np.zeros((4, 4), dtype=np.uint8),
                square_geometry(4),
                image_id="f",
                label="normal",
                flipped=True,
            )

    def test_mask_size_must_match_geometry(self) -> None:
        evaluator = OriginalSpaceEvaluator()
        with self.assertRaises(ValueError):
            evaluator.add_sample(
                binary_logits(np.zeros((4, 4), dtype=bool)),
                np.zeros((5, 5), dtype=np.uint8),
                square_geometry(4),
                image_id="m",
                label="normal",
            )


class RestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.box = geometry.compute_letterbox(1404, 1070, 512)
        self.geo = {
            "orig_w": 1404,
            "orig_h": 1070,
            "input_size": 512,
            "pad_left": self.box.pad_left,
            "pad_top": self.box.pad_top,
            "new_w": self.box.new_w,
            "new_h": self.box.new_h,
            "pad_right": self.box.pad_right,
            "pad_bottom": self.box.pad_bottom,
        }

    def _block_logits(self, cy: int, cx: int, half: int = 40) -> torch.Tensor:
        logits = torch.empty((2, 128, 128))
        logits[0] = 10.0
        logits[1] = -10.0
        logits[1, (cy - half) // 4:(cy + half) // 4, (cx - half) // 4:(cx + half) // 4] = 10.0
        logits[0, (cy - half) // 4:(cy + half) // 4, (cx - half) // 4:(cx + half) // 4] = -10.0
        return logits

    def test_restore_shape_is_original(self) -> None:
        prob = restore_defect_probability(self._block_logits(200, 256), self.geo)
        self.assertEqual(tuple(prob.shape), (1070, 1404))
        self.assertEqual(prob.dtype, torch.float32)

    def test_padding_is_dropped_before_upscaling(self) -> None:
        cy, cx = 200, 256
        prob = restore_defect_probability(self._block_logits(cy, cx), self.geo)
        ys, xs = np.nonzero((prob >= 0.5).numpy())
        expected_y = (cy - self.box.pad_top) / self.box.new_h * 1070
        expected_x = (cx - self.box.pad_left) / self.box.new_w * 1404
        self.assertLessEqual(abs(float(ys.mean()) - expected_y), 3.0)
        self.assertLessEqual(abs(float(xs.mean()) - expected_x), 3.0)

    def test_skipping_the_crop_shifts_the_prediction(self) -> None:
        cy = 200
        prob = restore_defect_probability(self._block_logits(cy, 256), self.geo)
        correct = float(np.nonzero((prob >= 0.5).numpy())[0].mean())
        wrong_small = upsample_logits(self._block_logits(cy, 256).unsqueeze(0), (512, 512))[0]
        wrong_full = upsample_logits(wrong_small.unsqueeze(0), (1070, 1404))[0]
        wrong = float(np.nonzero((torch.softmax(wrong_full, dim=0)[1] >= 0.5).numpy())[0].mean())
        self.assertGreater(abs(wrong - correct), 20.0)

    def test_single_original_pixel_that_512_sampling_drops(self) -> None:
        dropped = None
        for y in range(520, 560):
            for x in range(700, 740):
                one = np.zeros((1070, 1404), dtype=np.uint8)
                one[y, x] = 1
                label512 = geometry.letterbox_label(one, self.box, 255)
                if int((label512 == 1).sum()) == 0:
                    dropped = (y, x, one, label512)
                    break
            if dropped:
                break
        self.assertIsNotNone(dropped, "fixture needs a pixel that the 512 grid skips")
        y, x, one, label512 = dropped

        background = torch.empty((2, 128, 128))
        background[0] = 12.0
        background[1] = -12.0

        evaluator = OriginalSpaceEvaluator()
        evaluator.add_sample(background, one, self.geo, image_id="p", label="anomaly")
        original = evaluator.result()
        self.assertEqual(original["confusion"]["fn"], 1)
        self.assertEqual(original["confusion"]["tp"], 0)
        self.assertEqual(original["global"]["foreground_iou"], 0.0)

        valid = label512[self.box.pad_top:self.box.pad_top + self.box.new_h,
                         self.box.pad_left:self.box.pad_left + self.box.new_w]
        from PIL import Image as PILImage

        upscaled = np.asarray(PILImage.fromarray(valid, mode="L").resize((1404, 1070), PILImage.NEAREST))
        evaluator2 = OriginalSpaceEvaluator()
        evaluator2.add_sample(background, upscaled, self.geo, image_id="p2", label="anomaly")
        shrunk = evaluator2.result()
        self.assertEqual(shrunk["confusion"]["fn"], 0)
        self.assertIsNone(shrunk["global"]["foreground_iou"])


if __name__ == "__main__":
    unittest.main()
