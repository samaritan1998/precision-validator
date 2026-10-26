"""Bounded report reader: alignment, rejection and file descriptor cleanup."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from precision_validator.storage import aligned_chunks


class StreamingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = np.arange(12, dtype='<f4')
        self.candidate = self.reference + np.arange(12, dtype='<f4') / 8
        self.reference.tofile(self.root / 'reference.f32')
        self.candidate[:5].tofile(self.root / 'candidate-a.f32')
        self.candidate[5:].tofile(self.root / 'candidate-b.f32')
        piece = lambda file, start, end, offset=0: {'file': file, 'shape': [3, 4], 'start': start, 'end': end, 'offset_bytes': offset}
        self.left = {'weight': [piece('reference.f32', 0, 12)]}
        self.right = {'weight': [piece('candidate-a.f32', 0, 5), piece('candidate-b.f32', 5, 12)]}

    def readers(self):
        return (aligned_chunks,)

    def test_different_boundaries_and_file_offsets_preserve_all_values(self):
        # A candidate split occurs in the middle of a canonical matrix row.
        for reader in self.readers():
            values = list(reader(self.root, self.left, self.root, self.right))
            np.testing.assert_array_equal(np.concatenate([v[1] for v in values]), self.candidate)
            np.testing.assert_array_equal(np.concatenate([v[2] for v in values]), self.reference)

    def test_truncation_is_rejected(self):
        (self.root / 'candidate-b.f32').write_bytes(b'')
        for reader in self.readers():
            with self.assertRaisesRegex(Exception, 'Truncated raw tensor'):
                list(reader(self.root, self.left, self.root, self.right))

    def test_missing_interval_is_rejected(self):
        self.right['weight'][1]['start'] = 6
        for reader in self.readers():
            with self.assertRaisesRegex(Exception, 'Invalid shard intersection'):
                list(reader(self.root, self.left, self.root, self.right))

    def test_parent_escape_is_rejected(self):
        self.right['weight'][0]['file'] = '../outside.f32'
        for reader in self.readers():
            with self.assertRaises(Exception):
                list(reader(self.root, self.left, self.root, self.right))

    def test_file_handles_are_bounded_and_closed_on_early_exit(self):
        opened = []
        original = Path.open
        def tracked(path, *args, **kwargs):
            stream = original(path, *args, **kwargs)
            opened.append(stream)
            self.assertLessEqual(sum(not f.closed for f in opened), 33)
            return stream
        pieces = {}
        for i in range(40):
            name = f'{i}.f32';np.array([i], dtype='<f4').tofile(self.root/name)
            pieces[name] = [{'file': name, 'shape': [1], 'start': 0, 'end': 1, 'offset_bytes': 0}]
        with patch.object(Path, 'open', tracked):
            generator = aligned_chunks(self.root, pieces, self.root, pieces)
            for _ in range(39):next(generator)
            self.assertEqual(sum(not f.closed for f in opened), 32)
            generator.close()
        self.assertTrue(all(f.closed for f in opened))


if __name__ == '__main__':
    unittest.main()
