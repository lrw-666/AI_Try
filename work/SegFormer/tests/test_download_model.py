"""Offline identity and corruption regressions; Hub calls are never live."""
import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from segformer_lab import download_model as dm

COMMIT = "80983a413c30d36a39c20203974ae7807835e2b4"


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = {"schema": dm.SCHEMA, "model_id": dm.MODEL_ID,
                       "revision": COMMIT, "requested_revision": "main",
                       "weights": "pytorch_model.bin", "files": []}
        for name, content in [("config.json", b"{}"), ("preprocessor_config.json", b"{}"),
                              ("pytorch_model.bin", b"weights" * 300)]:
            (self.root / name).write_bytes(content)
            self.source["files"].append({"path": name, "size_bytes": len(content),
                                          "sha256": hashlib.sha256(content).hexdigest()})
        self.write_source()

    def write_source(self):
        (self.root / "source.json").write_text(json.dumps(self.source), encoding="utf-8")

    def test_complete_snapshot_passes(self):
        self.assertEqual(dm.verify_snapshot(self.root, self.source), [])

    def test_rejects_malformed_identity_and_manifest(self):
        variants = [{"files": []}, {"schema": "wrong"}, {"model_id": ""},
                    {"revision": "main"}, {"weights": "other.bin"},
                    {"files": self.source["files"][:-1]},
                    {"files": self.source["files"] + [self.source["files"][0]]}]
        for changes in variants:
            with self.subTest(changes=changes):
                source = dict(self.source, **changes)
                self.assertTrue(dm.verify_snapshot(self.root, source))

    def test_rejects_unsafe_paths_and_invalid_hashes(self):
        for field, value in [("path", "../outside.bin"), ("path", "C:/outside.bin"),
                             ("path", "config.json:stream"), ("sha256", "bad"),
                             ("size_bytes", -1)]:
            source = copy.deepcopy(self.source)
            source["files"][0][field] = value
            with self.subTest(field=field, value=value):
                self.assertTrue(dm.verify_snapshot(self.root, source))

    def test_rejects_unrecorded_shadow_weights_and_other_files(self):
        for name in ["model.safetensors", "extra.txt"]:
            with self.subTest(name=name):
                (self.root / name).write_bytes(b"unexpected")
                self.assertTrue(dm.verify_snapshot(self.root, self.source))
                (self.root / name).unlink()

    def test_rejects_lfs_pointer_even_with_matching_hash(self):
        data = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n"
        (self.root / "pytorch_model.bin").write_bytes(data)
        self.source["files"][-1].update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.assertTrue(dm.verify_snapshot(self.root, self.source))

    def test_existing_snapshot_reused_without_network(self):
        with patch.object(dm, "resolve_revision", side_effect=AssertionError("network forbidden")):
            self.assertEqual(dm.download(dm.MODEL_ID, self.root)["mode"], "reused")
            self.assertEqual(dm.download(dm.MODEL_ID, self.root, revision=COMMIT)["mode"], "reused")

    def test_existing_identity_changes_and_corruption_fail_without_network_or_write(self):
        original = (self.root / "source.json").read_bytes()
        with patch.object(dm, "resolve_revision", side_effect=AssertionError("network forbidden")):
            for model_id, revision in [("other/model", None), (dm.MODEL_ID, "a" * 40),
                                       (dm.MODEL_ID, "new-branch")]:
                with self.subTest(model_id=model_id, revision=revision), self.assertRaises(SystemExit):
                    dm.download(model_id, self.root, revision=revision)
            (self.root / "pytorch_model.bin").write_bytes(b"corruption")
            with self.assertRaises(SystemExit):
                dm.download(dm.MODEL_ID, self.root)
        self.assertEqual((self.root / "source.json").read_bytes(), original)

    def test_cli_never_logs_proxy_credentials(self):
        output = io.StringIO()
        with patch.dict(os.environ), redirect_stdout(output):
            result = dm.main(["--verify-only", "--out", str(self.root),
                              "--proxy", "http://private-user:secret-password@localhost:12345"])
        self.assertEqual(result, 0)
        for text in ["private-user", "secret-password", "localhost:12345"]:
            self.assertNotIn(text, output.getvalue())

    def test_explicit_proxy_clears_all_proxy_and_matches_no_proxy(self):
        with patch.dict(os.environ, {"ALL_PROXY": "stale", "all_proxy": "stale",
                                     "NO_PROXY": "example.invalid", "no_proxy": "other.invalid"}):
            dm.apply_proxy("http://localhost:12345")
            self.assertTrue("ALL_PROXY" not in os.environ)
            self.assertTrue("all_proxy" not in os.environ)
            self.assertEqual(os.environ["NO_PROXY"], os.environ["no_proxy"])

    def test_source_record_restores_exact_commit_and_hashes(self):
        record = self.root.parent / (self.root.name + "-source.json")
        record.write_text(json.dumps({"source": self.source}), encoding="utf-8")
        self.addCleanup(record.unlink)
        destination = self.root / "new"
        destination.mkdir()

        def fetch(**kwargs):
            self.assertEqual(kwargs["revision"], COMMIT)
            self.assertEqual(set(kwargs["allow_patterns"]), {i["path"] for i in self.source["files"]})
            for item in self.source["files"]:
                (destination / item["path"]).write_bytes((self.root / item["path"]).read_bytes())

        with patch.object(dm, "resolve_revision", side_effect=AssertionError("resolution forbidden")), \
                patch("huggingface_hub.snapshot_download", side_effect=fetch):
            result = dm.download(dm.MODEL_ID, destination, source_record=record)
        self.assertTrue(result["ok"])
        self.assertEqual(result["source"]["files"], self.source["files"])

    def test_source_record_hash_mismatch_never_creates_source(self):
        record = self.root.parent / (self.root.name + "-source.json")
        record.write_text(json.dumps(self.source), encoding="utf-8")
        self.addCleanup(record.unlink)
        destination = self.root / "new"
        destination.mkdir()

        def fetch(**kwargs):
            for item in self.source["files"]:
                (destination / item["path"]).write_bytes(b"wrong")

        with patch.object(dm, "resolve_revision", side_effect=AssertionError("resolution forbidden")), \
                patch("huggingface_hub.snapshot_download", side_effect=fetch), self.assertRaises(SystemExit):
            dm.download(dm.MODEL_ID, destination, source_record=record)
        self.assertFalse((destination / "source.json").exists())

    def test_new_directory_defaults_to_pinned_commit(self):
        destination = self.root / "new"
        destination.mkdir()

        def fetch(**kwargs):
            self.assertEqual(kwargs["revision"], COMMIT)
            for item in self.source["files"]:
                (destination / item["path"]).write_bytes((self.root / item["path"]).read_bytes())

        with patch.object(dm, "resolve_revision", side_effect=AssertionError("resolution forbidden")), \
                patch("huggingface_hub.snapshot_download", side_effect=fetch):
            self.assertTrue(dm.download(dm.MODEL_ID, destination)["ok"])

    def test_explicit_different_sha_cannot_hide_in_requested_revision(self):
        self.source["requested_revision"] = "a" * 40
        self.write_source()
        with patch.object(dm, "resolve_revision", side_effect=AssertionError("network forbidden")), \
                self.assertRaises(SystemExit):
            dm.download(dm.MODEL_ID, self.root, revision="a" * 40)

    def test_failed_hub_request_does_not_echo_proxy_credentials(self):
        destination = self.root / "new"
        destination.mkdir()
        with patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("secret-proxy-url")):
            with self.assertRaises(SystemExit) as failure:
                dm.download(dm.MODEL_ID, destination)
        self.assertNotIn("secret-proxy-url", str(failure.exception))

    def test_existing_manifest_cannot_be_reblessed_against_source_record(self):
        record = self.root.parent / (self.root.name + "-source.json")
        record.write_text(json.dumps(self.source), encoding="utf-8")
        self.addCleanup(record.unlink)
        content = b"replaced weights"
        (self.root / "pytorch_model.bin").write_bytes(content)
        self.source["files"][-1].update(size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        self.write_source()
        with patch.object(dm, "resolve_revision", side_effect=AssertionError("network forbidden")), \
                self.assertRaises(SystemExit):
            dm.download(dm.MODEL_ID, self.root, source_record=record)


if __name__ == "__main__":
    unittest.main()
