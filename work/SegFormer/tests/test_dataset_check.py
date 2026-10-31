"""Acceptance evidence regressions; all generated evidence lives in temp dirs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from matplotlib.axes import Axes

from segformer_lab import dataset_check as check, data
from tests.test_dataset import _make_toy_root, StubProcessor


class AcceptanceTests(unittest.TestCase):
    def test_skip_model_never_claims_full_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = check.main(['--skip-model', '--report', str(root / 'report.json'),
                                   '--figure-dir', str(root / 'figures')])
            report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(code, 0)
            self.assertFalse(report['ok'], 'dataset-only must not certify the whole stage')
            self.assertTrue(report['data_checks_ok'])
            self.assertFalse(report['full_stage_verified'])
            self.assertIsNone(report['checks']['eval_forward_shape'])
            self.assertEqual(report['eval_forward']['status'], 'skipped')
            self.assertIn('DATASET ONLY', output.getvalue())
            self.assertEqual(len(report['validation_previews']), 6)
            self.assertTrue(all(f['mask_id'] is not None for f in report['figures']))
            self.assertTrue(report['provenance']['files'])
            self.assertTrue(all(item['checks']['geometry_exact'] for item in report['validation_previews']))

    def test_forced_figure_source_gt_matches_source_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = _make_toy_root(root)
            plain = data.PCB1Dataset(config, 'train', StubProcessor(), flip='none')
            forced = data.PCB1Dataset(config, 'train', StubProcessor(), flip='force')
            arrays = []
            original = Axes.imshow
            def capture(axis, array, *args, **kwargs):
                arrays.append(np.asarray(array).copy())
                return original(axis, array, *args, **kwargs)
            with patch.object(Axes, 'imshow', capture):
                result = check._render_sample(forced, 1, root / 'figure.png')
            np.testing.assert_array_equal(arrays[1], plain.prepared_sample(1)['binary_original'])
            self.assertEqual(result['coordinate_space']['source_gt'], 'source_file')
            self.assertEqual(result['mask_id'], forced.rows[1]['mask'])

    def test_geometry_checks_every_item_not_only_first(self):
        geometry = dict(orig_w=1404, orig_h=1070, input_size=512, scale=512/1404,
                        new_w=512, new_h=390, pad_left=0, pad_right=0, pad_top=61,
                        pad_bottom=61, valid_area=512*390, padding_rows=[61,451], ignore_index=255)
        labels = torch.full((2,512,512), 255, dtype=torch.int64)
        labels[:,61:451] = 0
        batch = {'labels': labels, 'geometry': [geometry, dict(geometry, pad_bottom=60)]}
        result = check._label_check(batch)
        self.assertFalse(result.get('geometry_all_exact', True), 'second geometry drift must fail')

    def test_bad_manifest_is_rejected_before_processor_or_sample_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(check, 'verify_manifests', return_value={'ok': False}), \
                 patch.object(data, 'build_processor', side_effect=AssertionError('processor must not build')):
                with self.assertRaisesRegex(SystemExit, 'refusing to read samples'):
                    check.main(['--skip-model', '--report', str(Path(directory)/'report.json'),
                                '--figure-dir', str(Path(directory)/'figures')])

    def test_manifest_verifier_preserves_reviewed_group_checks(self):
        sentinel = {'ok': False, 'reviewed_groups': {'source_match': False}}
        with patch('segformer_lab.audit_data.verify_manifests', return_value=sentinel):
            self.assertIs(check.verify_manifests(check.load_config('configs/baseline.yaml')), sentinel)


if __name__ == '__main__':
    unittest.main()
