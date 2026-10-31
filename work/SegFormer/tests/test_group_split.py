"""Reviewed near-duplicate groups must stay within one derived split."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from segformer_lab import paths, prepare_data
from segformer_lab.prepare_data import OfficialRow, _write_manifest, build_split


class ReviewedGroupSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="segformer-group-split-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.read_groups = getattr(prepare_data, "read_groups", None)
        self.split_groups = getattr(prepare_data, "build_group_split", None)
        self.check_groups = getattr(prepare_data, "check_group_splits", None)
        self.assertTrue(callable(self.read_groups), "reviewed group membership must be validated")
        self.assertTrue(callable(self.split_groups), "reviewed groups need indivisible train/val assignment")
        self.assertTrue(callable(self.check_groups), "group overlap must be auditable after splitting")

    def rows(self, normal: int = 8, anomaly: int = 4) -> list[OfficialRow]:
        rows = [OfficialRow("pcb1", "train", "normal", f"pcb1/n{i}.JPG", "") for i in range(normal)]
        rows += [OfficialRow("pcb1", "train", "anomaly", f"pcb1/a{i}.JPG", f"pcb1/a{i}.png")
                 for i in range(anomaly)]
        rows += [OfficialRow("pcb1", "test", "normal", "pcb1/test.JPG", "")]
        return rows

    def groups_file(self, groups: list[dict]) -> Path:
        target = self.root / "reviewed_groups.json"
        target.write_text(json.dumps({"schema": "segformer-lab/reviewed-groups/1", "groups": groups}),
                          encoding="utf-8")
        return target

    def test_group_members_stay_together_with_exact_quotas(self) -> None:
        rows = self.rows()
        groups = self.read_groups(self.groups_file([
            {"id": "normal-pair", "label": "normal", "members": ["pcb1/n0.JPG", "pcb1/n1.JPG"]},
            {"id": "anomaly-pair", "label": "anomaly", "members": ["pcb1/a0.JPG", "pcb1/a1.JPG"]},
        ]), rows)
        derived, algorithm = self.split_groups(rows, "pcb1", 42, 3, 2, groups)
        membership = {row.image: name for name, members in derived.items() for row in members}
        self.assertEqual(membership["pcb1/n0.JPG"], membership["pcb1/n1.JPG"])
        self.assertEqual(membership["pcb1/a0.JPG"], membership["pcb1/a1.JPG"])
        self.assertEqual(sum(r.label == "normal" for r in derived["val"]), 3)
        self.assertEqual(sum(r.label == "anomaly" for r in derived["val"]), 2)
        self.assertEqual(algorithm["protocol"]["version"], 2)
        self.assertTrue(self.check_groups(derived, groups)["ok"])

    def test_row_and_group_order_do_not_change_assignment(self) -> None:
        rows = self.rows()
        groups = self.read_groups(self.groups_file([
            {"id": "pair-1", "label": "normal", "members": ["pcb1/n0.JPG", "pcb1/n1.JPG"]},
            {"id": "pair-2", "label": "normal", "members": ["pcb1/n4.JPG", "pcb1/n5.JPG"]},
        ]), rows)
        first, _ = self.split_groups(rows, "pcb1", 42, 3, 2, groups)
        second, _ = self.split_groups(list(reversed(rows)), "pcb1", 42, 3, 2, list(reversed(groups)))
        self.assertEqual(first, second)

    def test_official_test_manifest_bytes_do_not_change(self) -> None:
        rows = self.rows()
        groups = self.read_groups(self.groups_file([
            {"id": "pair", "label": "normal", "members": ["pcb1/n0.JPG", "pcb1/n1.JPG"]},
        ]), rows)
        original, _ = build_split(rows, "pcb1", 42, 3, 2)
        grouped, _ = self.split_groups(rows, "pcb1", 42, 3, 2, groups)
        _write_manifest(self.root / "test-v1.csv", original["test"], "test")
        _write_manifest(self.root / "test-v2.csv", grouped["test"], "test")
        self.assertEqual((self.root / "test-v1.csv").read_bytes(),
                         b"object,split,label,image,mask\npcb1,test,normal,pcb1/test.JPG,\n")
        self.assertEqual((self.root / "test-v1.csv").read_bytes(), (self.root / "test-v2.csv").read_bytes())

    def test_invalid_or_ambiguous_group_members_are_rejected(self) -> None:
        invalid = [
            [{"id": "unknown", "label": "normal", "members": ["pcb1/n0.JPG", "pcb1/missing.JPG"]}],
            [{"id": "test", "label": "normal", "members": ["pcb1/n0.JPG", "pcb1/test.JPG"]}],
            [{"id": "mixed", "label": "normal", "members": ["pcb1/n0.JPG", "pcb1/a0.JPG"]}],
            [{"id": "duplicate", "label": "normal", "members": ["pcb1/n0.JPG", "pcb1/n0.JPG"]}],
            [{"id": "one", "label": "normal", "members": ["pcb1/n0.JPG", "pcb1/n1.JPG"]},
             {"id": "two", "label": "normal", "members": ["pcb1/n1.JPG", "pcb1/n2.JPG"]}],
        ]
        for groups in invalid:
            with self.subTest(groups=groups), self.assertRaises(SystemExit):
                self.read_groups(self.groups_file(groups), self.rows())

    def test_indivisible_groups_cannot_silently_change_validation_quota(self) -> None:
        rows = self.rows(normal=2, anomaly=0)
        groups = self.read_groups(self.groups_file([
            {"id": "pair", "label": "normal", "members": ["pcb1/n0.JPG", "pcb1/n1.JPG"]},
        ]), rows)
        with self.assertRaises(SystemExit):
            self.split_groups(rows, "pcb1", 42, 1, 0, groups)

    def test_group_check_rejects_cross_split_or_missing_members(self) -> None:
        groups = [{"id": "pair", "label": "normal", "members": ["pcb1/n0.JPG", "pcb1/n1.JPG"]}]
        rows = self.rows()
        crossed = {"train": [rows[0]], "val": [rows[1]], "test": [rows[-1]]}
        report = self.check_groups(crossed, groups)
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["cross_split_groups"]), 1)
        missing = {"train": [rows[0]], "val": [], "test": [rows[-1]]}
        self.assertFalse(self.check_groups(missing, groups)["ok"])

    def test_ungrouped_protocol_keeps_recorded_v1_csv_bytes(self) -> None:
        official = paths.MANIFESTS_DIR / "official_2cls_highshot.csv"
        with official.open(encoding="utf-8", newline="") as handle:
            rows = [OfficialRow(**row) for row in csv.DictReader(handle)]
        derived, _ = build_split(rows, "pcb1", 42, 120, 12)
        expected = {
            "train": "b26077266a59efdf45306bd9de0ce726aca52db4d7d50ea3ae284eaa92130d53",
            "val": "7335d8c82b8997bffa06af59283f1ade7ac18949bfdec14973cd054f3161bb9c",
            "test": "4bdc594148f208c7067a66a8cb3b06fde53ebc1f4ab6fda84f08f80383061284",
        }
        for split, members in derived.items():
            output = self.root / f"{split}.csv"
            _write_manifest(output, members, split)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), expected[split])


if __name__ == "__main__":
    unittest.main()
