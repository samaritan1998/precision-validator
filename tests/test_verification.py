"""Successful process exit cannot substitute for executed packaging tests."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    'verify_wheel', Path(__file__).resolve().parents[1] / 'scripts/verify_wheel.py')
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class VerificationTests(unittest.TestCase):
    def test_actual_empty_unittest_discovery_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            result = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                                     '-s', temporary, '-v'],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Older supported Python versions exit zero for empty discovery;
            # newer versions may return 5. The log must reject both outcomes.
            self.assertIn(result.returncode, (0, 5), result.stdout)
            self.assertIn('Ran 0 tests', result.stdout)
            path = folder / 'unittest.log'; path.write_text(result.stdout)
            with self.assertRaisesRegex(RuntimeError, 'did not execute'):
                VERIFY.check_packaging_log(path)

    def test_optional_skips_are_explicit_and_native_skips_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'unittest.log'
            path.write_text('Ran 4 tests in 0.001s\n\nOK (skipped=1)\n')
            self.assertEqual(VERIFY.check_packaging_log(path), {'tests_run': 4, 'skipped': 1})
            with self.assertRaisesRegex(RuntimeError, 'skipped tests'):
                VERIFY.check_packaging_log(path, native=True)
            path.write_text('Ran 4 tests in 0.001s\n\nOK\n')
            self.assertEqual(VERIFY.check_packaging_log(path, native=True),
                             {'tests_run': 4, 'skipped': 0})


if __name__ == '__main__':
    unittest.main()
