"""CPU regressions for stage-five acceptance gates (no GPU or network)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from segformer_lab import modeling, model_check


class ModelAcceptanceTests(unittest.TestCase):
    def test_rejects_changed_encoder_even_below_old_tolerance(self):
        model = torch.nn.Module()
        model.segformer = torch.nn.Module()
        model.segformer.encoder = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.segformer.encoder.weight.fill_(0.0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.save({'encoder.weight': torch.tensor([[1e-7]]),
                        'classifier.weight': torch.zeros(1), 'classifier.bias': torch.zeros(1)},
                       root / 'pytorch_model.bin')
            self.assertFalse(modeling.verify_encoder_weights(model, root)['ok'])

    def test_unused_checkpoint_keys_are_complete_and_rejected(self):
        model = torch.nn.Module()
        model.segformer = torch.nn.Module()
        model.segformer.encoder = torch.nn.Linear(1, 1, bias=False)
        state = {'encoder.weight': model.segformer.encoder.weight.detach().clone()}
        state.update({f'unknown.{i}': torch.zeros(1) for i in range(12)})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.save(state, root / 'pytorch_model.bin')
            result = modeling.verify_encoder_weights(model, root)
            self.assertFalse(result['ok'])
            self.assertEqual(len(result['checkpoint_tensors_not_used']), 12)

    def test_loading_unknown_unexpected_is_rejected(self):
        result = modeling.classify_loading_keys({'unexpected_keys': ['encoder.extra']})
        self.assertFalse(result['encoder_fully_loaded'])

    def test_load_rejects_missing_source_before_transformers(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'source.json'):
                modeling.load_all(Path(temp))

    def test_load_rejects_wrong_pinned_identity_before_transformers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'source.json').write_text(json.dumps({'model_id': 'other/model', 'revision': 'main'}))
            with self.assertRaises(ValueError):
                modeling.load_all(root)

    def test_256_label_uses_scaled_letterbox_and_has_defect(self):
        _, labels = model_check.make_batch(1, 256)
        self.assertEqual(int((labels == 255).sum()), 61 * 256)
        self.assertTrue(bool((labels[:, :30] == 255).all()))
        self.assertTrue(bool((labels[:, 225:] == 255).all()))
        self.assertGreater(int((labels == 1).sum()), 0)

    def test_input_size_override_cannot_bypass_fixed_config(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(torch.cuda, 'is_available', return_value=False), \
                patch.object(model_check, 'load_all', side_effect=AssertionError('must reject before loading')):
            with self.assertRaisesRegex(SystemExit, 'input-size'):
                model_check.main(['--input-size', '256', '--loading-report', str(Path(temp) / 'loading.json'),
                                  '--check-report', str(Path(temp) / 'check.json')])

    def test_runtime_gate_rejects_bad_shape_nonfinite_loss_and_missing_gradient(self):
        forward = {'input_shape': [2, 3, 512, 512], 'label_shape': [2, 512, 512],
                   'logits_shape': [2, 2, 128, 128], 'upsampled_shape': [2, 2, 512, 512],
                   'label_unique': [0, 1, 255], 'ce_full_padded': 0.5, 'ce_valid_region': 0.5,
                   'ce_difference': 0.0}
        grads = [{'exists': True, 'grad_finite': True, 'grad_nonzero': True}]
        step = [{'changed': True, 'max_abs_delta': 0.01, 'mean_abs_delta': 0.001}]
        for key, value in [('logits_shape', [2, 3, 128, 128]), ('ce_full_padded', float('nan'))]:
            broken = copy.deepcopy(forward)
            broken[key] = value
            self.assertFalse(all(model_check.runtime_checks(broken, grads, step, True, 512).values()))
        self.assertFalse(all(model_check.runtime_checks(forward, [{'exists': False}], step, True, 512).values()))

    def test_load_rejects_nonbinary_head_or_wrong_processor(self):
        loaded = SimpleNamespace(
            processor_info={'do_resize': False, 'do_reduce_labels': False, 'do_rescale': True, 'do_normalize': True},
            param_summary={'classifier_head': {'found': True, 'out_channels': 3}, 'encoder_trainable': True,
                           'total_params': 4, 'trainable_params': 4},
            encoder_check={'ok': True}, loading_info={'encoder_fully_loaded': True})
        self.assertFalse(all(modeling.loading_checks(loaded).values()))
        loaded.param_summary['classifier_head']['out_channels'] = 2
        self.assertTrue(all(modeling.loading_checks(loaded).values()))
        loaded.processor_info['do_reduce_labels'] = True
        self.assertFalse(all(modeling.loading_checks(loaded).values()))

    def test_manifest_hash_failure_is_rejected_before_processor(self):
        from segformer_lab.download_model import PINNED_REVISION
        with tempfile.TemporaryDirectory() as temp, patch.object(
                modeling, 'load_processor', side_effect=AssertionError('invalid manifest reached processor')):
            root = Path(temp)
            source = {'schema': 'segformer-lab/model-source/1', 'model_id': 'nvidia/mit-b0',
                      'revision': PINNED_REVISION, 'weights': 'pytorch_model.bin',
                      'files': [{'path': 'pytorch_model.bin', 'size_bytes': 2048, 'sha256': '0' * 64}]}
            (root / 'pytorch_model.bin').write_bytes(b'x' * 2048)
            (root / 'source.json').write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'verification failed'):
                modeling.load_all(root)


if __name__ == '__main__':
    unittest.main()
