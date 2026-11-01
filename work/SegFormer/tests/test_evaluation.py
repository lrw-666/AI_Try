"""IEEE FP32 evaluation policy: CPU tests, no CUDA tensors or model loads."""
import unittest

import torch

from segformer_lab.evaluation import evaluation_precision


class EvaluationPrecisionTests(unittest.TestCase):
    config = {'raw': {'eval': {'precision': 'ieee_fp32'}, 'train': {'amp': True}}}

    def test_disables_tf32_and_outer_autocast_then_restores_both(self):
        before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        try:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            with torch.autocast('cpu', dtype=torch.bfloat16):
                with evaluation_precision(self.config, device='cpu') as record:
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertFalse(torch.is_autocast_enabled('cpu'))
                    self.assertEqual(record['api'], 'torch2.10 allow_tf32 (legacy API only)')
                    self.assertTrue(record['before']['cudnn_allow_tf32'])
                    self.assertTrue(record['before']['matmul_allow_tf32'])
                    result = torch.ones(2, 2) @ torch.ones(2, 2)
                    self.assertEqual(result.dtype, torch.float32)
                self.assertTrue(torch.is_autocast_enabled('cpu'))
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(record['restored_after_exit'])
            self.assertTrue(self.config['raw']['train']['amp'])
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = before

    def test_backend_state_is_restored_after_failure(self):
        before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            with evaluation_precision(self.config, device='cpu'):
                raise RuntimeError('failed')
        self.assertEqual(before, (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32))

    def test_missing_or_other_precision_policy_fails_without_mutation(self):
        before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        for raw in ({}, {'precision': 'tf32'}, {'precision': None}):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, 'eval.precision'):
                with evaluation_precision({'raw': {'eval': raw}}, device='cpu'):
                    self.fail('invalid policy entered')
        self.assertEqual(before, (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32))


if __name__ == '__main__':
    unittest.main()
