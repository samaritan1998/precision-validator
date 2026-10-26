"""Runs in the training image; skipped on a host without torch."""
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires the pinned image's torch")
class ObserverTests(unittest.TestCase):
    def test_cpu_chunk_writer_preserves_full_gradient_and_actual_adam_delta(self):
        import numpy as np
        import torch
        from precision_validator.slime_adapter import cpu_chunks
        from precision_validator.storage import ShardWriter
        torch.manual_seed(1234)
        parameter = torch.nn.Parameter(torch.randn(17, 9, dtype=torch.float32))
        optimizer = torch.optim.AdamW([parameter], lr=2e-5, betas=(.9, .98), weight_decay=.1)
        initial = parameter.detach().clone()
        inputs = torch.randn(3, 17)
        loss = (inputs @ parameter).square().mean()
        loss.backward()
        gradient = parameter.grad.detach().clone()
        piece = {"name": "weight", "shape": [17, 9], "start": 0, "end": 153}
        with tempfile.TemporaryDirectory() as tmp:
            writer = ShardWriter(tmp, 0, 153 * 8)
            writer.append("gradients", piece, cpu_chunks(parameter.grad))
            torch.nn.utils.clip_grad_norm_([parameter], .1)
            optimizer.step()
            delta = parameter.detach() - initial
            writer.append("updates", piece, cpu_chunks(delta))
            writer.close()
            np.testing.assert_array_equal(np.fromfile(Path(tmp) / "gradients.rank0.f32", dtype="<f4"),
                                          gradient.numpy().ravel())
            np.testing.assert_array_equal(np.fromfile(Path(tmp) / "updates.rank0.f32", dtype="<f4"),
                                          delta.numpy().ravel())
            self.assertTrue(float(delta.norm()) > 0)
            self.assertEqual(optimizer.state[parameter]["exp_avg"].dtype, torch.float32)

    def test_bf16_hash_and_native_master_states(self):
        import torch
        from precision_validator.slime_adapter import parameter_hash, master_parameter
        from precision_validator.common import ValidationError
        parameter = torch.tensor([1, 2, 3], dtype=torch.bfloat16)
        initial = parameter_hash(parameter)
        parameter[1] += 1
        self.assertNotEqual(initial, parameter_hash(parameter))
        master = torch.tensor([1.001, 2.002, 3.003], dtype=torch.float32)
        optimizer = SimpleNamespace(state={parameter: {"master_param": master}})
        self.assertIs(master_parameter(optimizer, parameter), master)
        optimizer.state[parameter]["master_param"] = master.bfloat16()
        with self.assertRaises(ValidationError):
            master_parameter(optimizer, parameter)

    def test_stream_budget_fails_before_overrun(self):
        import numpy as np
        from precision_validator.common import ValidationError
        from precision_validator.storage import ShardWriter
        with tempfile.TemporaryDirectory() as tmp:
            writer = ShardWriter(tmp, 0, 4)
            with self.assertRaises(ValidationError):
                writer.append("gradients", {"name": "p", "shape": [2], "start": 0, "end": 2}, [np.array([1, 2])])
            for stream in writer.files.values():
                stream.close()


if __name__ == "__main__":
    unittest.main()
