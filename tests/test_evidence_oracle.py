"""Negative controls for the independent trajectory evidence auditor."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'trajectory_oracle', Path(__file__).resolve().parents[1] / 'scripts/audit_trajectory_report.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class EvidenceOracleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parameters = {'small.weight': {'shape': [2]}}
        self.fields = ('gradients', 'updates')
        self.row = {'rank': 0, 'observations': {'loss': {'shape': [], 'values': [1.0]}},
                    'storage': {'files': {}, 'pieces': []}}
        for field in self.fields:
            filename = field + '.f32'
            np.asarray([0.25, -0.5], dtype='<f4').tofile(self.root / filename)
            self.row['storage']['files'][filename] = {'bytes': 8, 'sha256': oracle.digest(self.root / filename)}
            self.row['storage']['pieces'].append(dict(name='small.weight', shape=[2], start=0, end=2,
                elements=2, file=filename, offset_bytes=0, field=field))
        self.save()

    def save(self):
        (self.root / 'rank0.json').write_text(json.dumps(self.row))

    def assemble(self):
        return oracle.assemble(self.root, 1, self.parameters, self.fields)

    def test_complete_and_zero_reference_metrics(self):
        data, observations, _ = self.assemble()
        self.assertEqual(len(data), 2)
        np.testing.assert_array_equal(observations['loss'], [1.0])
        zeros = np.zeros(2)
        self.assertEqual(oracle.metrics(zeros, zeros)['relative_l2'], 0.0)
        self.assertIsNone(oracle.metrics(zeros, np.ones(2))['relative_l2'])

    def test_raw_tensor_tampering_rejected(self):
        (self.root / 'gradients.f32').write_bytes(b'\0' * 8)
        with self.assertRaises(AssertionError):
            self.assemble()

    def test_missing_fragment_rejected(self):
        self.row['storage']['pieces'][0].update(end=1, elements=1)
        self.save()
        with self.assertRaises(AssertionError):
            self.assemble()

    def test_overlapping_fragment_rejected(self):
        self.row['storage']['pieces'].append(dict(self.row['storage']['pieces'][0]))
        self.save()
        with self.assertRaises(AssertionError):
            self.assemble()

    def test_wrong_report_metric_rejected(self):
        actual = oracle.metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        published = dict(actual, max_abs=0.0)
        with self.assertRaises(AssertionError):
            oracle.check_metrics(actual, published)

    def test_nonfinite_observation_rejected(self):
        self.row['observations']['loss']['values'] = [float('nan')]
        self.save()
        with self.assertRaises(AssertionError):
            self.assemble()


if __name__ == '__main__':
    unittest.main()
