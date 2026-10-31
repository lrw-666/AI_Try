"""Regression checks for false-positive environment acceptance.

Replace external inventory boundaries rather than uninstalling real packages or
altering the user's Python environment. The acceptance decision stays real.
"""
import copy
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from segformer_lab import env_check


class EnvironmentAcceptanceTests(unittest.TestCase):
    def collect(self, *, bad_package=None, wrong_prefix=False,
                user_site=False, cache_error=False):
        prefix = env_check.paths.ENV_DIR
        packages = {
            name: {"version": "test", "file": str(prefix / "Lib/site-packages" / name / "__init__.py")}
            for name in env_check.PROBE_MODULES
        }
        if bad_package:
            packages[bad_package] = {"import_error": "ImportError: simulated missing dependency"}
        effective = {"temp_dir": str(env_check.paths.TMP_DIR), "outside_resource_root": []}
        if cache_error:
            effective["hf_cache_error"] = "ImportError: cannot resolve required cache"
        with ExitStack() as stack:
            stack.enter_context(patch.object(env_check, "_package_facts", return_value=copy.deepcopy(packages)))
            stack.enter_context(patch.object(env_check, "_effective_paths", return_value=effective))
            stack.enter_context(patch.object(env_check, "_cuda_device_facts", return_value={"available": True, "devices": []}))
            stack.enter_context(patch.object(env_check.site, "ENABLE_USER_SITE", user_site))
            if wrong_prefix:
                stack.enter_context(patch.object(sys, "prefix", str(prefix.parent / "unrelated_env")))
            return env_check.collect(probe="none")

    def test_available_dependencies_and_project_prefix_pass(self):
        self.assertEqual(self.collect()["status"], "ok")

    def test_missing_required_package_cannot_pass(self):
        report = self.collect(bad_package="torchvision")
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("torchvision" in message for message in report["errors"]))

    def test_user_site_enabled_cannot_pass(self):
        report = self.collect(user_site=True)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("user" in message.lower() for message in report["errors"]))

    def test_wrong_prefix_cannot_pass_even_with_no_external_site_packages(self):
        # Package files remain under the actual sys.prefix in this scenario,
        # so the previous relative import check alone would not catch it.
        with patch.object(env_check, "_import_isolation", return_value={
            "ok": True, "outside_prefix": [], "packages_checked": 7,
        }), patch.object(env_check, "_site_paths_outside_prefix", return_value=[]):
            report = self.collect(wrong_prefix=True)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("prefix" in message.lower() for message in report["errors"]))

    def test_failed_required_cache_inventory_cannot_pass(self):
        report = self.collect(cache_error=True)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("cache" in message.lower() for message in report["errors"]))


if __name__ == "__main__":
    unittest.main()
