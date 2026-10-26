"""Real file publication races observed during the first 256K image run."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from precision_validator.common import ValidationError
from precision_validator.runner import raw_data_bytes, check_raw_budget


class LiveStorageTests(unittest.TestCase):
    def test_rename_after_listing_preserves_size_without_false_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / "updates.rank2.f32.partial"
            final = partial.with_suffix("")
            partial.write_bytes(b"\0" * 32)
            original = Path.glob

            def publish_during_scan(folder, pattern):
                listed = list(original(folder, pattern))
                partial.replace(final)
                yield from listed

            with patch.object(Path, "glob", publish_during_scan):
                self.assertEqual(check_raw_budget(root, 32), 32)
            self.assertEqual(raw_data_bytes(root), 32)

    def test_two_directory_names_for_one_published_shard_are_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / "gradients.rank0.f32.partial"
            final = partial.with_suffix("")
            partial.write_bytes(b"\0" * 32)

            def changing_listing(folder, pattern):
                yield partial
                partial.replace(final)
                yield final

            with patch.object(Path, "glob", changing_listing):
                self.assertEqual(check_raw_budget(root, 32), 32)

    def test_publication_race_does_not_hide_a_budget_overrun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partial = root / "updates.rank7.f32.partial"
            final = partial.with_suffix("")
            partial.write_bytes(b"\0" * 36)

            def stale_listing(folder, pattern):
                partial.replace(final)
                yield partial

            with patch.object(Path, "glob", stale_listing), self.assertRaisesRegex(ValidationError, "budget exceeded"):
                check_raw_budget(root, 32)

    def test_other_io_errors_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "gradients.rank0.f32.partial"
            path.write_bytes(b"\0" * 4)
            original = Path.stat

            def failed_stat(p, *args, **kwargs):
                if p == path:
                    raise PermissionError("storage permission denied")
                return original(p, *args, **kwargs)

            with patch.object(Path, "stat", failed_stat), self.assertRaises(PermissionError):
                raw_data_bytes(root)

    def test_finished_and_in_progress_shards_both_count_towards_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gradients.rank0.f32").write_bytes(b"\0" * 20)
            (root / "updates.rank0.f32.partial").write_bytes(b"\0" * 12)
            (root / "train.log").write_text("not raw data")
            self.assertEqual(check_raw_budget(root, 32), 32)
            with self.assertRaises(ValidationError):
                check_raw_budget(root, 31)


if __name__ == "__main__":
    unittest.main()
