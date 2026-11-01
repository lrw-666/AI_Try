"""CPU boundary regressions for the stage-07 loss and scoring contract."""
from __future__ import annotations

import unittest
import warnings

import numpy as np
import torch

from segformer_lab.losses import loss_definition, segformer_loss
from segformer_lab.metrics import OriginalSpaceEvaluator, restore_defect_probability


def geometry(size=2):
    return dict(input_size=size, orig_h=size, orig_w=size, new_h=size,
                new_w=size, pad_top=0, pad_left=0, pad_bottom=0, pad_right=0)


class LossAcceptanceTests(unittest.TestCase):
    def test_half_logits_without_autocast_have_finite_ce_and_gradient(self):
        logits = torch.zeros((1, 2, 128, 128), dtype=torch.float16, requires_grad=True)
        terms = segformer_loss(logits, torch.zeros((1, 512, 512), dtype=torch.int64))
        self.assertAlmostEqual(float(terms.ce.detach()), float(np.log(2)), places=5)
        terms.total.backward()
        self.assertTrue(bool(logits.grad.isfinite().all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_loss_input_contract(self):
        logits = torch.zeros((1, 2, 2, 2))
        labels = torch.zeros((1, 2, 2), dtype=torch.int64)
        for dtype in (torch.float32, torch.int32, torch.bool):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                segformer_loss(logits, labels.to(dtype))
        with self.subTest(logits='integer'), self.assertRaises(ValueError):
            segformer_loss(logits.to(torch.int64), labels)
        with self.subTest(num_labels=3), self.assertRaises(ValueError):
            segformer_loss(torch.zeros((1, 3, 2, 2)), labels, num_labels=3)
        for kwargs in (dict(eps=0), dict(eps=-1), dict(eps=float('nan')),
                       dict(eps=float('inf')), dict(ce_weight=-1),
                       dict(dice_weight=float('inf')), dict(ce_weight=float('nan')),
                       dict(ce_weight=0, dice_weight=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                segformer_loss(logits, labels, **kwargs)

    def test_valid_images_and_zero_dice_graph(self):
        logits = torch.zeros((2, 2, 2, 2), requires_grad=True)
        labels = torch.zeros((2, 2, 2), dtype=torch.int64)
        labels[1] = 255
        terms = segformer_loss(logits, labels)
        self.assertEqual(terms.valid_images, 1)
        terms.dice.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(float(logits.grad.abs().sum()), 0)

    def test_two_positive_images_are_averaged_without_normal_image(self):
        logits = torch.zeros((3, 2, 2, 2))
        labels = torch.tensor([[[1, 0], [0, 0]], [[1, 1], [1, 1]], [[0, 0], [0, 0]]])
        terms = segformer_loss(logits, labels)
        expected = ((1-(1+1e-6)/(3+1e-6)) + (1-(4+1e-6)/(6+1e-6))) / 2
        self.assertAlmostEqual(float(terms.dice), expected, places=6)

    def test_definition_records_runtime_parameters(self):
        definition = loss_definition(ce_weight=2, dice_weight=0.25, eps=1e-4)
        self.assertEqual(definition['runtime'], dict(ce_weight=2.0, dice_weight=0.25, eps=1e-4))
        self.assertEqual(definition['dice']['eps'], 1e-4)


class MetricAcceptanceTests(unittest.TestCase):
    def add(self, evaluator, mask, logits=None, **kwargs):
        if logits is None:
            logits = torch.stack((torch.full((2, 2), 12.), torch.full((2, 2), -12.)))
        return evaluator.add_sample(logits, np.asarray(mask), geometry(), image_id='a',
                                    label=kwargs.pop('label', 'normal'), **kwargs)

    def test_ignored_pixels_are_excluded_from_confusion_and_normal_area(self):
        evaluator = OriginalSpaceEvaluator()
        logits = torch.zeros((2, 2, 2))
        logits[1] = 12
        score = self.add(evaluator, [[0, 255], [0, 255]], logits)
        result = evaluator.result()
        self.assertEqual((score.tp, score.fp, score.fn, score.tn), (0, 2, 0, 0))
        self.assertEqual(score.total_pixels, 2)
        self.assertEqual(result['normal_images']['fp_pixel_ratio'], 1)
        self.assertEqual(result['normal_images']['predicted_foreground_ratio']['p95'], 1)
        self.assertEqual(result['confusion']['pixels'], 2)

    def test_mask_validation_and_all_ignore_are_atomic(self):
        for mask in ([[255, 255], [255, 255]], [[0, 2], [0, 0]],
                     [[0., .5], [0., 0.]], [[0., float('nan')], [0., 0.]]):
            evaluator = OriginalSpaceEvaluator()
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.add(evaluator, mask)
            self.assertEqual(evaluator.result()['images']['total'], 0)
            self.assertEqual(int(evaluator.tp + evaluator.fp + evaluator.fn + evaluator.tn), 0)

    def test_threshold_logits_and_label_validation(self):
        for threshold in (-.1, 1.1, float('nan'), float('inf')):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                evaluator = OriginalSpaceEvaluator(threshold=threshold)
                self.add(evaluator, [[0, 0], [0, 0]])
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.add(OriginalSpaceEvaluator(), [[0, 0], [0, 0]], torch.full((2, 2, 2), value))
        with self.assertRaises(ValueError):
            self.add(OriginalSpaceEvaluator(), [[0, 0], [0, 0]], label='other')

    def test_geometry_validation(self):
        for changes in (dict(pad_top=-1), dict(pad_left=-1), dict(pad_bottom=-1),
                        dict(pad_right=1), dict(new_h=3), dict(orig_h=0),
                        dict(input_size=0), dict(new_h=1.5), dict(input_size=float('nan'))):
            geo = geometry(); geo.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                restore_defect_probability(torch.zeros((2, 2, 2)), geo)
        geo = geometry(); del geo['orig_h']
        with self.assertRaises(ValueError):
            restore_defect_probability(torch.zeros((2, 2, 2)), geo)

    def test_integer_accumulation_past_int32_and_per_image_percentiles(self):
        evaluator = OriginalSpaceEvaluator(fp=np.int64(2**31))
        for pixels in (0, 1, 4):
            logits = torch.stack((torch.full((2, 2), 12.), torch.full((2, 2), -12.)))
            logits[0].view(-1)[:pixels] = -12
            logits[1].view(-1)[:pixels] = 12
            self.add(evaluator, [[0, 0], [0, 0]], logits)
        result = evaluator.result()
        self.assertIsInstance(evaluator.fp, np.int64)
        self.assertEqual(result['confusion']['fp'], 2**31 + 5)
        normal = result['normal_images']
        self.assertAlmostEqual(normal['fp_pixel_ratio'], 5 / 12)
        self.assertEqual(normal['predicted_foreground_ratio']['median'], .25)
        self.assertEqual(normal['predicted_foreground_ratio']['p95'], .925)

    def test_na_reasons_and_no_numpy_torch_mixing_warning(self):
        evaluator = OriginalSpaceEvaluator()
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            score = self.add(evaluator, [[0, 0], [0, 0]])
        record = score.as_dict()
        self.assertTrue(record['precision_na'])
        self.assertTrue(record['recall_na'])
        result = evaluator.result()
        self.assertTrue(result['anomaly_images']['macro']['iou']['na_reason'])
        empty = OriginalSpaceEvaluator().result()
        self.assertTrue(empty['normal_images']['na_reasons']['fp_pixel_ratio'])
        self.assertTrue(empty['normal_images']['na_reasons']['predicted_foreground_ratio'])


if __name__ == '__main__':
    unittest.main()
