"""The audit checks group constraints independently of saved success flags."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from segformer_lab import paths
from segformer_lab.audit_data import verify_manifests


class AuditGroupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('train.csv', 'val.csv', 'test.csv', 'split_meta.json',
                     'official_2cls_highshot.csv', 'reviewed_groups.json'):
            shutil.copyfile(paths.MANIFESTS_DIR / name, self.root / name)
        self.cfg = {'paths': {'data.split_meta': self.root / 'split_meta.json', **{
            f'data.{split}_manifest': self.root / f'{split}.csv' for split in ('train', 'val', 'test')
        }}}

    def test_changed_group_source_is_rejected(self):
        self.assertTrue(verify_manifests(self.cfg)['ok'])
        with (self.root / 'reviewed_groups.json').open('a', encoding='utf-8') as handle:
            handle.write(' ')
        self.assertFalse(verify_manifests(self.cfg)['ok'])

    def test_hash_consistent_cross_split_group_is_rejected(self):
        meta = json.loads((self.root / 'split_meta.json').read_text(encoding='utf-8'))
        for split in ('train', 'val'):
            target = self.root / f'{split}.csv'
            shutil.copyfile(paths.MANIFESTS_DIR / 'archive/pcb1_v1' / f'{split}.csv', target)
            meta['manifests'][split]['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        (self.root / 'split_meta.json').write_text(json.dumps(meta), encoding='utf-8')
        report = verify_manifests(self.cfg)
        self.assertTrue(all(x['match'] for x in report['manifests'].values()))
        self.assertFalse(report['ok'])
        self.assertEqual(len(report['reviewed_groups']['cross_split_groups']), 1)


if __name__ == '__main__':
    unittest.main()
