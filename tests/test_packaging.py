"""Checks specific to the installed distribution, not numerical algorithms."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from precision_validator.common import ASSET_ROOT, ROOT, ValidationError, digest, read
from precision_validator.profiles import profiles
from precision_validator.planning import toolkit_inventory
from precision_validator.resources import EXAMPLES, export_example


class PackagingTests(unittest.TestCase):
    def test_installed_core_does_not_require_training_frameworks(self):
        requirements = importlib.metadata.requires("precision-validator")
        self.assertEqual([r for r in requirements if 'extra ==' not in r], ["numpy>=1.24"])

    def test_import_and_env_do_not_import_torch_or_mutate_environment(self):
        code = '''
import json, os, sys
before = dict(os.environ)
from precision_validator.environment import inspect_environment
from precision_validator.api import prepare_generic, compare
row = inspect_environment()
assert "torch" not in sys.modules
assert before == dict(os.environ)
assert not row["gpu_checked"]
print(json.dumps(row))
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=tempfile.gettempdir(),
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "environment_inventory")

    def test_bundled_resources_have_no_cwd_dependency(self):
        self.assertEqual(len(profiles()), 6)
        inventory = toolkit_inventory()
        self.assertIn("precision_validator/_assets/runtime-lock.json", inventory)
        self.assertIn("precision_validator/_assets/policies/qwen-noise-v1.json", inventory)
        for name, expected in inventory.items():
            self.assertEqual(digest(ROOT / name), expected)
        self.assertTrue((ASSET_ROOT / "tests/test_protocol.py").is_file())

    def test_export_all_templates_with_local_helpers_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, spec in EXAMPLES.items():
                root = Path(temporary) / name
                result = export_example(name, root)
                self.assertEqual(result["status"], "example_created")
                manifest = read(root / "template-manifest.json")
                self.assertEqual(set(manifest["files"]), set(spec["files"]))
                for file in spec["files"]:
                    self.assertEqual(digest(root / file), manifest["files"][file])
                if name.startswith("slime-"):
                    code = (root / "slime_parallel.py").read_text()
                    self.assertNotIn("/root/Megatron-LM", code)
                    self.assertIn("with_name('megatron_parallel.py')", code)

    def test_export_does_not_overwrite_or_follow_dangling_output_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "existing"
            root.mkdir()
            sentinel = root / "user.txt"
            sentinel.write_text("keep")
            with self.assertRaises(ValidationError):
                export_example("torch", root)
            self.assertEqual(sentinel.read_text(), "keep")
            link = Path(temporary) / "link"
            link.symlink_to(Path(temporary) / "absent")
            with self.assertRaises(ValidationError):
                export_example("torch", link)
            self.assertFalse((Path(temporary) / "absent").exists())

    def test_candidate_frameworks_win_over_stale_site_packages(self):
        from precision_validator.runner import environment
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, megatron, installed = [root / name for name in ("candidate", "megatron", "site-packages")]
            for parent, module, marker in [(source, "slime", "candidate"), (megatron, "megatron", "candidate"),
                                           (installed, "slime", "stale"), (installed, "megatron", "stale")]:
                path = parent / module
                path.mkdir(parents=True)
                (path / "__init__.py").write_text(f"marker = {marker!r}\n")
            library = root / "memory-saver.so"
            library.touch()
            plan = {"source": str(source), "megatron_path": str(megatron), "profile": {"environment": {}},
                    "output": str(root), "plan_sha256": "0" * 64}
            with patch("precision_validator.runner.ROOT", installed), patch(
                    "precision_validator.runner.subprocess.check_output", return_value=str(library)):
                env = environment(plan, "cp1")
            env.pop("LD_PRELOAD", None)  # The empty fixture represents a path, not a shared library.
            code = "import slime, megatron; assert slime.marker == megatron.marker == 'candidate'"
            result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_pinned_doctor_respects_explicit_megatron_environment(self):
        from precision_validator.runtime import doctor
        lock = read(ASSET_ROOT / "runtime-lock.json")
        with patch.dict(os.environ, {"MEGATRON_PATH": "/test/explicit-megatron"}), patch(
                "precision_validator.runtime.runtime_fingerprint", return_value={"versions": lock["versions"]}) as capture:
            result = doctor(cpu_only=True)
        capture.assert_called_once_with("/test/explicit-megatron")
        self.assertFalse(result["gpu_checked"])


if __name__ == "__main__":
    unittest.main()
