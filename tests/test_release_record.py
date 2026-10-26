"""A model numerical failure must never become a release qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location('release_record',
    Path(__file__).resolve().parents[1] / 'scripts/write_release_record.py')
RECORD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECORD)


class ReleaseRecordTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.wheel = self.root / 'fixture.whl'
        source = b'__version__ = "test"\n'
        with zipfile.ZipFile(self.wheel, 'w') as z:
            z.writestr('fixture.dist-info/METADATA', 'Version: test\n')
            z.writestr('precision_validator/__init__.py', source)
        self.cpu = self.root / 'cpu'; self.cpu.mkdir()
        self.write(self.cpu/'summary.json', dict(status='passed', wheel_sha256=RECORD.sha(self.wheel),
            checks=[dict(returncode=0)], native_cpu_runtime=True))
        self.write(self.cpu/'installed-files.log', {'precision_validator/__init__.py': hashlib.sha256(source).hexdigest()})
        self.write(self.cpu/'selftest.log', dict(skipped=0, status='selftest_passed'))
        self.coverage = self.root/'required.json'; self.write(self.coverage, {'test_gpu':['case']})
        self.gpu = self.root/'gpu'; (self.gpu/'case').mkdir(parents=True)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value)+'\n')

    def outcome(self, status):
        report = self.gpu/'case/acceptance.json'; self.write(report, dict(status=status))
        self.write(self.gpu/'qualification.json', dict(wheel_sha256=RECORD.sha(self.wheel),
            status='passed' if status == 'passed' else 'failed', device_family='test_gpu', device='fixture',
            cases={'case':dict(status=status, report_sha256=RECORD.sha(report))}))

    def build(self, folders=None):
        return RECORD.build_record(self.wheel,[self.cpu], folders if folders is not None else [self.gpu],self.coverage)

    def test_numerical_failure_is_recorded_without_qualification(self):
        self.outcome('numerical_failure')
        result = self.build()
        self.assertEqual(result['status'], 'numerical_failure')
        self.assertEqual(result['missing_gpu_coverage'], {})
        self.assertEqual(result['numerical_failures'][0]['case'], 'case')

    def test_missing_or_failed_execution_cannot_qualify(self):
        self.assertEqual(self.build([])['status'], 'incomplete')
        self.outcome('execution_failure')
        result = self.build()
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['missing_gpu_coverage'], {'test_gpu':['case']})

    def test_only_unchanged_matching_complete_pass_can_qualify(self):
        self.outcome('passed')
        self.assertEqual(self.build()['status'], 'qualified')
        self.write(self.gpu/'case/acceptance.json', {'status':'numerical_failure'})
        with self.assertRaisesRegex(ValueError, 'report changed'):
            self.build()

    def test_duplicate_execution_cannot_replace_required_cases(self):
        self.outcome('passed')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.build([self.gpu,self.gpu])

    def test_interrupted_matrix_counts_only_finished_hash_matched_cases(self):
        self.outcome('passed')
        record = RECORD.read(self.gpu/'qualification.json'); record['status'] = 'running'
        self.write(self.gpu/'qualification.json', record)
        self.write(self.coverage, {'test_gpu':['case', 'not_finished']})
        result = self.build()
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['missing_gpu_coverage'], {'test_gpu':['not_finished']})
        self.assertEqual(result['evidence'][-1]['matrix_status_at_snapshot'], 'running')


if __name__ == '__main__':
    unittest.main()
