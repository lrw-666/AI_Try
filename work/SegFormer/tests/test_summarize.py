"""CPU regressions for the stage-13 summary helpers (no plotting backend needed)."""
import unittest

import numpy as np

from segformer_lab import summarize
from segformer_lab.training import WarmupCosineLR


class ScheduleMirrorTests(unittest.TestCase):
    plan = {"base_lr": 6e-5, "min_lr": 1e-6, "total_steps": 13250, "warmup_steps": 662}

    def test_matches_the_shared_scheduler(self):
        schedule = WarmupCosineLR(base_lr=6e-5, min_lr=1e-6, total_steps=13250, warmup_steps=662)
        for completed in (0, 1, 661, 662, 663, 5000, 13249):
            with self.subTest(completed=completed):
                self.assertAlmostEqual(summarize.lr_from_schedule(completed, self.plan),
                                       schedule.lr_at(completed), places=15)

    def test_first_and_last_values_are_the_documented_ones(self):
        self.assertAlmostEqual(summarize.lr_from_schedule(0, self.plan), 9.06344410876133e-08, places=20)
        self.assertAlmostEqual(summarize.lr_from_schedule(661, self.plan), 6e-5, places=18)
        self.assertAlmostEqual(summarize.lr_from_schedule(13249, self.plan), 1e-6, places=18)


class CurvesTableTests(unittest.TestCase):
    def test_maps_every_column_without_inventing_numbers(self):
        entry = {
            "epoch": 3, "attempts_in_epoch": 265, "successful_in_epoch": 263, "skipped_in_epoch": 2,
            "successful_total": 793, "scheduler_completed": 793,
            "lr_first": 4.8e-5, "lr_last": 6.0e-5,
            "loss": {"ce": 0.5, "dice": 0.9, "total": 0.95},
            "val": {"global": {"foreground_iou": 0.4, "foreground_dice": 0.5,
                               "precision": None, "recall": 0.3},
                    "anomaly_images": {"macro": {"iou": {"mean": 0.2}}},
                    "normal_images": {"false_positive_pixels": 12, "fp_pixel_ratio": 1e-5}},
            "diagnostics": {"empty_prediction_count": 7},
        }
        table = summarize.curves_table([entry], {"base_lr": 6e-5, "min_lr": 1e-6,
                                                 "total_steps": 13250, "warmup_steps": 662})
        row = table[0]
        self.assertEqual(row["epoch"], 3)
        self.assertEqual(row["skipped_in_epoch"], 2)
        self.assertEqual(row["scheduler_completed"], 793)
        self.assertEqual(row["ce"], 0.5)
        self.assertEqual(row["val_iou"], 0.4)
        self.assertIsNone(row["val_precision"])
        self.assertEqual(row["empty_anomaly_count"], 7)
        self.assertAlmostEqual(row["lr_after_epoch_computed"],
                               summarize.lr_from_schedule(793, {"base_lr": 6e-5, "min_lr": 1e-6,
                                                                "total_steps": 13250,
                                                                "warmup_steps": 662}), places=18)


class OverlayTests(unittest.TestCase):
    def test_tp_fp_fn_colours(self):
        source = np.zeros((2, 3, 3), dtype=np.uint8)
        gt = np.zeros((2, 3), dtype=np.uint8)
        pred = np.zeros((2, 3), dtype=np.uint8)
        gt[0, 0] = 1
        gt[1, 1] = 1
        pred[0, 0] = 1     # TP
        pred[0, 1] = 1     # FP
        canvas = summarize._overlay(source, gt, pred)
        self.assertEqual(tuple(canvas[0, 0]), summarize.TP_COLOR)
        self.assertEqual(tuple(canvas[0, 1]), summarize.FP_COLOR)
        self.assertEqual(tuple(canvas[1, 1]), summarize.FN_COLOR)
        # untouched background: source*0.55 + 255*0.45, truncated to uint8
        self.assertEqual(tuple(canvas[1, 2]), (114, 114, 114))


if __name__ == "__main__":
    unittest.main()
