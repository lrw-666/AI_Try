"""Unit tests for stage 3 data preparation.

These tests do not touch the dataset archive or any image file: they pin down
the two parts that are easy to get wrong and hard to notice later, namely the
archive path guard and the deterministic split.
"""

from __future__ import annotations

import csv
import hashlib
import io
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from segformer_lab.prepare_data import (
    COLUMNS,
    OfficialRow,
    _write_manifest,
    build_split,
    extract_category,
    main,
    safe_member_target,
)


class PreparationWriteSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="segformer-write-safety-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def archive(self, entries):
        path = self.root / "fixture.tar"
        with tarfile.open(path, "w") as handle:
            for name, content in entries:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                handle.addfile(member, io.BytesIO(content))
        return path

    def test_rewrite_identical_does_not_authorize_different_content(self):
        source = self.archive([("pcb1/a", b"archive")])
        target = self.root / "out/pcb1/a"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"user copy")
        with self.assertRaises(SystemExit):
            extract_category(source, "pcb1", self.root / "out", rewrite_identical=True)
        self.assertEqual(target.read_bytes(), b"user copy")

    def test_regular_archive_part_filename_is_not_used_as_scratch(self):
        source = self.archive([("pcb1/a.part", b"legitimate member"), ("pcb1/a", b"second member")])
        extract_category(source, "pcb1", self.root / "out")
        self.assertTrue((self.root / "out/pcb1/a.part").is_file(), "legitimate .part member was removed")
        self.assertEqual((self.root / "out/pcb1/a.part").read_bytes(), b"legitimate member")
        self.assertEqual((self.root / "out/pcb1/a").read_bytes(), b"second member")

    def test_conflicting_existing_manifest_is_preserved(self):
        target = self.root / "train.csv"
        target.write_bytes(b"existing user manifest")
        with self.assertRaises(SystemExit):
            _write_manifest(target, [], "train")
        self.assertEqual(target.read_bytes(), b"existing user manifest")

    def test_cli_rejects_different_protocol_before_accessing_sources(self):
        for extra in [("--val-normal", "119", "--val-anomaly", "13"), ("--seed", "43")]:
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as raised:
                main(["--archive", str(self.root / "absent.tar"), "--source-zip", str(self.root / "absent.zip"), *extra])
            self.assertEqual(raised.exception.code, 2)

    def test_failed_audit_does_not_publish_or_replace_manifests(self):
        out = self.root / "manifests"
        out.mkdir()
        for name in ["train.csv", "val.csv", "test.csv", "split_meta.json"]:
            (out / name).write_bytes(b"preserved result")
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(COLUMNS)
        for split, label, count in [("train", "normal", 602), ("train", "anomaly", 60), ("test", "normal", 6)]:
            for i in range(count):
                image = f"pcb1/{split}/{label}/{i}.JPG"
                writer.writerow(["pcb1", split, label, image, image + ".png" if label == "anomaly" else ""])
        raw = text.getvalue().encode()
        source_zip = self.root / "source.zip"
        with zipfile.ZipFile(source_zip, "w") as handle:
            handle.writestr("spot-diff-main/split_csv/2cls_highshot.csv", raw)
        source_tar = self.archive([])
        with redirect_stdout(io.StringIO()):
            code = main(["--archive", str(source_tar), "--source-zip", str(source_zip),
                         "--data-root", str(self.root / "missing-data"), "--out", str(out),
                         "--expect-csv-sha256", hashlib.sha256(raw).hexdigest(), "--skip-extraction"])
        self.assertEqual(code, 1)
        for name in ["train.csv", "val.csv", "test.csv", "split_meta.json"]:
            self.assertEqual((out / name).read_bytes(), b"preserved result", name)


class SafeMemberTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="segformer-safety-"))

    def test_accepts_relative_member(self) -> None:
        target = safe_member_target(self.root, "pcb1/Data/Images/Normal/0001.JPG")
        self.assertTrue(str(target).startswith(str(self.root.resolve())))
        self.assertEqual(target.name, "0001.JPG")

    def test_rejects_absolute_path(self) -> None:
        with self.assertRaises(ValueError):
            safe_member_target(self.root, "/etc/passwd")

    def test_rejects_windows_drive(self) -> None:
        with self.assertRaises(ValueError):
            safe_member_target(self.root, "C:/Windows/system32/cmd.exe")

    def test_rejects_parent_traversal(self) -> None:
        with self.assertRaises(ValueError):
            safe_member_target(self.root, "pcb1/../../evil.png")

    def test_rejects_backslash_traversal(self) -> None:
        with self.assertRaises(ValueError):
            safe_member_target(self.root, "pcb1\\..\\..\\evil.png")

    def test_rejects_alternate_data_stream(self) -> None:
        with self.assertRaises(ValueError):
            safe_member_target(self.root, "pcb1/Data/inject:hidden.png")


class SplitTests(unittest.TestCase):
    def _rows(self, normal: int, anomaly: int) -> list[OfficialRow]:
        rows = [
            OfficialRow("pcb1", "train", "normal", f"pcb1/Data/Images/Normal/{i:04d}.JPG", "")
            for i in range(normal)
        ]
        rows += [
            OfficialRow(
                "pcb1",
                "train",
                "anomaly",
                f"pcb1/Data/Images/Anomaly/{i:04d}.JPG",
                f"pcb1/Data/Masks/Anomaly/{i:04d}.png",
            )
            for i in range(anomaly)
        ]
        rows += [
            OfficialRow("pcb1", "test", "normal", f"pcb1/Data/Images/Normal/T{i:04d}.JPG", "")
            for i in range(6)
        ]
        return rows

    def test_negative_quota_is_rejected(self):
        with self.assertRaises(SystemExit):
            build_split(self._rows(20, 4), "pcb1", 42, -1, 1)

    def test_input_row_order_does_not_change_membership(self):
        rows = self._rows(20, 4)
        baseline, _ = build_split(rows, "pcb1", 42, 5, 2)
        shuffled = list(reversed(rows))
        actual, _ = build_split(shuffled, "pcb1", 42, 5, 2)
        self.assertEqual(actual, baseline)

    def test_quota_and_disjoint(self) -> None:
        derived, _ = build_split(self._rows(20, 4), "pcb1", seed=42, val_normal=5, val_anomaly=2)
        self.assertEqual(len(derived["val"]), 7)
        self.assertEqual(len(derived["train"]), 17)
        self.assertEqual(len(derived["test"]), 6)
        self.assertEqual(sum(1 for r in derived["val"] if r.label == "normal"), 5)
        self.assertEqual(sum(1 for r in derived["val"] if r.label == "anomaly"), 2)
        sets = {name: {r.image for r in rows} for name, rows in derived.items()}
        self.assertEqual(sets["train"] & sets["val"], set())
        self.assertEqual(sets["train"] & sets["test"], set())
        self.assertEqual(sets["val"] & sets["test"], set())

    def test_is_deterministic(self) -> None:
        first, _ = build_split(self._rows(20, 4), "pcb1", seed=42, val_normal=5, val_anomaly=2)
        second, _ = build_split(self._rows(20, 4), "pcb1", seed=42, val_normal=5, val_anomaly=2)
        self.assertEqual([r.image for r in first["val"]], [r.image for r in second["val"]])
        self.assertEqual([r.image for r in first["train"]], [r.image for r in second["train"]])

    def test_matches_reference_permutation(self) -> None:
        """The documented algorithm must be reproduced exactly, not approximately."""
        rows = self._rows(20, 4)
        derived, _ = build_split(rows, "pcb1", seed=42, val_normal=5, val_anomaly=2)
        rng = np.random.default_rng(42)
        normal = sorted([r for r in rows if r.split == "train" and r.label == "normal"], key=lambda r: r.image)
        anomaly = sorted([r for r in rows if r.split == "train" and r.label == "anomaly"], key=lambda r: r.image)
        expected_val = {normal[i].image for i in rng.permutation(len(normal))[:5]}
        expected_val |= {anomaly[i].image for i in rng.permutation(len(anomaly))[:2]}
        self.assertEqual({r.image for r in derived["val"]}, expected_val)

    def test_manifest_rows_are_sorted_and_mask_empty_for_normal(self) -> None:
        derived, _ = build_split(self._rows(20, 4), "pcb1", seed=42, val_normal=5, val_anomaly=2)
        with tempfile.TemporaryDirectory() as tmp:
            info = _write_manifest(Path(tmp) / "val.csv", derived["val"], "val")
            with open(info["path"], encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(tuple(rows[0]), COLUMNS)
        body = rows[1:]
        self.assertEqual([r[3] for r in body], sorted(r[3] for r in body))
        for row in body:
            self.assertEqual(row[1], "val")
            if row[2] == "normal":
                self.assertEqual(row[4], "")
            else:
                self.assertTrue(row[4].endswith(".png"))

    def test_quota_exceeding_available_rows_fails(self) -> None:
        with self.assertRaises(SystemExit):
            build_split(self._rows(3, 1), "pcb1", seed=42, val_normal=5, val_anomaly=2)


if __name__ == "__main__":
    unittest.main()
