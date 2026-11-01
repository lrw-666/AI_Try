"""Stage-07 checker regressions: CPU only, generated files use temp dirs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from segformer_lab import metrics_check as check


class ForwardComparisonTests(unittest.TestCase):
    def compare(self, left, right):
        a = torch.tensor([left], dtype=torch.float32)
        b = torch.tensor([right], dtype=torch.float32)
        logits = torch.zeros(2, 1, len(left), dtype=torch.float32)
        return check.compare_forward_tensors(logits, logits, a, b)

    def test_opposite_flips_do_not_cancel_when_confusion_is_equal(self):
        result = self.compare([0.500001, 0.499999], [0.499999, 0.500001])
        self.assertEqual(result['changed_pixels'], 2)
        self.assertEqual(result['threshold_crossings'], {'negative_to_positive': 1, 'positive_to_negative': 1})
        self.assertTrue(result['float_close'])

    def test_positive_gt_flip_is_counted_once_not_once_each_for_tp_and_fn(self):
        self.assertEqual(self.compare([0.499999], [0.500001])['changed_pixels'], 1)

    def test_equal_aggregate_counts_cannot_pass_large_probability_changes(self):
        result = self.compare([0.9, 0.1], [0.1, 0.9])
        self.assertEqual(result['changed_pixels'], 2)
        self.assertFalse(result['float_close'])
        self.assertFalse(result['changed_pixels_within_threshold_tolerance'])

    def test_identical_predictions_do_not_hide_large_probability_changes(self):
        self.assertFalse(self.compare([0.9], [0.6])['float_close'])

    def test_no_changes_have_null_margin_statistics(self):
        result = self.compare([0.1, 0.9], [0.1, 0.9])
        self.assertEqual(result['changed_pixels'], 0)
        self.assertIsNone(result['changed_pixel_margin']['left']['max'])
        self.assertEqual(result['max_abs_prob_diff'], 0.0)
        self.assertTrue(result['float_close'])

    def test_nonfinite_outputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.compare([float('nan')], [0.5])

    def test_fp32_protocol_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'float32'):
            check.compare_forward_tensors(torch.zeros(2, 1, 1), torch.zeros(2, 1, 1),
                                          torch.zeros(1, 1, dtype=torch.float16), torch.zeros(1, 1))


class AcceptanceTests(unittest.TestCase):
    def test_baseline_declares_ieee_fp32_evaluation_without_disabling_training_amp(self):
        config = check.load_config('configs/baseline.yaml')
        self.assertEqual(config['raw']['eval']['precision'], 'ieee_fp32')
        self.assertTrue(config['raw']['train']['amp'])

    def test_metric_fixtures_include_ignore_and_na_reasons(self):
        fixtures = check.metric_fixtures()
        self.assertEqual(fixtures['original_mask_ignore255']['confusion'], {'tp': 1, 'fp': 0, 'fn': 1, 'tn': 1})
        self.assertEqual(fixtures['original_mask_ignore255']['ignored_pixels'], 1)
        self.assertTrue(fixtures['all_ignored_original_mask']['rejected'])
        self.assertTrue(fixtures['normal_empty_prediction']['precision_na'])
        self.assertTrue(fixtures['normal_empty_prediction']['recall_na'])

    def test_skip_model_is_fixtures_only_with_null_checks_and_no_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(check, 'pipeline_probe', side_effect=AssertionError('model forbidden')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                code = check.main(['--skip-model', '--device', 'cpu',
                                   '--report', str(root / 'report.json'), '--figure-dir', str(root / 'figures')])
            report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(code, 0)
            self.assertEqual(report['scope'], 'fixtures_only')
            self.assertTrue(report['fixtures_ok'])
            self.assertFalse(report['ok'])
            self.assertFalse(report['full_stage_verified'])
            self.assertTrue(report['not_a_baseline'])
            self.assertIsNone(report['checks']['probe_real_data_ran'])
            self.assertEqual(report['random_head_pipeline_probe']['status'], 'skipped')
            self.assertIn('FIXTURES ONLY', output.getvalue())
            self.assertTrue(report['provenance']['files'])
            self.assertEqual(report['runtime_protocol']['evaluation_precision']['policy'], 'ieee_fp32')
            self.assertFalse(report['environment']['cudnn_allow_tf32'])
            self.assertFalse(report['environment']['matmul_allow_tf32'])
            self.assertTrue(any(item['path'].endswith('evaluation.py') for item in report['provenance']['files']))

    def test_bad_identity_stops_before_fixture_or_model_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(check, 'identity_checks', return_value={'ok': False}), \
                 patch.object(check, 'metric_fixtures', side_effect=AssertionError('fixtures forbidden')):
                with self.assertRaisesRegex(SystemExit, 'identity'):
                    check.main(['--skip-model', '--report', str(Path(directory)/'report.json'),
                                '--figure-dir', str(Path(directory)/'figures')])

    def test_eval_batch_size_must_be_one_before_probe(self):
        config = check.load_config('configs/baseline.yaml')
        config['raw']['eval']['batch_size'] = 2
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(check, 'load_config', return_value=config), \
                 patch.object(check, 'pipeline_probe', side_effect=AssertionError('model forbidden')):
                with self.assertRaisesRegex(SystemExit, 'eval.batch_size'):
                    check.main(['--skip-model', '--report', str(Path(directory)/'report.json'),
                                '--figure-dir', str(Path(directory)/'figures')])

    def test_same_logits_check_compares_per_image_and_prediction_identity(self):
        left = {'confusion': {'tp': 1, 'fp': 1}, 'per_image': [{'image_id': 'a', 'tp': 1}],
                'sample_evidence': [{'image_id': 'a', 'prediction_sha256': 'one'}]}
        right = dict(left, per_image=[{'image_id': 'a', 'tp': 0}])
        self.assertFalse(check.same_logits_layout_equal(left, right))
        right = dict(left, sample_evidence=[{'image_id': 'a', 'prediction_sha256': 'two'}])
        self.assertFalse(check.same_logits_layout_equal(left, right))

    def test_cpu_stub_probe_uses_one_seeded_eval_model_and_paired_caches(self):
        config = check.load_config('configs/baseline.yaml')
        wanted = [line.strip() for line in Path(config['paths']['logging.fixed_preview_manifest']).read_text().splitlines()
                  if line.strip() and not line.startswith('#')]

        class Dataset:
            rows = [{'image': name, 'label': 'normal' if '/Normal/' in name else 'anomaly'} for name in wanted]

            def __getitem__(self, index):
                return {'image_id': wanted[index], 'mask_id': '', 'flipped': False,
                        'pixel_values': torch.ones(3, 2, 2), 'labels': torch.zeros(2, 2, dtype=torch.int64),
                        'original_mask': np.zeros((2, 2), dtype=np.uint8), 'geometry': check.square_geometry(2)}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decode_head = torch.nn.Conv2d(3, 2, 1)
                self.calls = []

            def forward(self, pixel_values):
                self.calls.append((self.training, torch.is_grad_enabled(), len(pixel_values)))
                return SimpleNamespace(logits=self.decode_head(pixel_values))

        created = []

        def load(*args, **kwargs):
            model = Model()
            created.append(model)
            return SimpleNamespace(model=model, snapshot={'revision': check.EXPECTED_REVISION}, encoder_check={'ok': True})

        with patch.object(check.data, 'PCB1Dataset', return_value=Dataset()), \
             patch('segformer_lab.modeling.load_all', side_effect=load):
            result = check.pipeline_probe(config, 'cpu')
        self.assertEqual(len(created), 1)
        self.assertEqual([n for _, _, n in created[0].calls], [2, 2, 2] + [1]*12)
        self.assertTrue(all(not train and not grad for train, grad, _ in created[0].calls))
        self.assertTrue(result['same_logits_layout_equal'])
        self.assertTrue(result['model_identity']['state_unchanged'])
        self.assertEqual(result['two_forward_passes']['changed_pixels'], 0)
        self.assertEqual(result['batch_1_repeat']['changed_pixels'], 0)
        self.assertTrue(result['two_forward_passes']['float_close'])
        self.assertEqual(len(result['two_forward_passes']['per_image']), 6)
        self.assertEqual(result['seed'], 42)

    def test_offline_snapshot_failure_is_part_of_identity_preflight(self):
        config = check.load_config('configs/baseline.yaml')
        with patch.object(check, 'verify_snapshot', return_value=['sha256 changed: weights']):
            result = check.identity_checks(config)
        self.assertFalse(result['ok'])
        self.assertFalse(result['snapshot']['verified_actual_bytes_offline'])

    def test_identity_reuses_group_and_official_csv_audit(self):
        config = check.load_config('configs/baseline.yaml')
        sentinel = {'ok': False, 'reviewed_groups': {'ok': False}}
        with patch.object(check, 'verify_manifests', return_value=sentinel):
            result = check.identity_checks(config)
        self.assertIs(result['audit_data_verifier'], sentinel)
        self.assertFalse(result['ok'])


if __name__ == '__main__':
    unittest.main()
