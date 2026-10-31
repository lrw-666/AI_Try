"""Protect the fixed validation panel identity across repeated CLI calls."""

import tempfile
import unittest
from pathlib import Path

from segformer_lab.preview_data import select_with_ids, write_ids


class FixedPreviewIdsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ids.txt"
        self.rows = [
            {"image": "normal/a.JPG", "label": "normal"},
            {"image": "anomaly/b.JPG", "label": "anomaly"},
        ]

    def test_existing_ids_are_reused_without_cli_counts(self):
        write_ids(self.path, self.rows, "val")
        self.assertEqual(select_with_ids(self.rows, self.path, None, None), self.rows)

    def test_explicit_count_conflict_fails_without_reselection(self):
        write_ids(self.path, self.rows, "val")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "fixed preview"):
            select_with_ids(self.rows, self.path, 6, 6)
        self.assertEqual(self.path.read_bytes(), before)

    def test_duplicate_and_out_of_split_ids_fail(self):
        for content in ("normal/a.JPG\nnormal/a.JPG\n", "normal/missing.JPG\n", "# empty\n"):
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "fixed preview"):
                    select_with_ids(self.rows, self.path, None, None)

    def test_existing_file_cannot_be_replaced(self):
        write_ids(self.path, self.rows, "val")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "fixed preview"):
            write_ids(self.path, list(reversed(self.rows)), "val")
        self.assertEqual(self.path.read_bytes(), before)
        write_ids(self.path, self.rows, "val")
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
