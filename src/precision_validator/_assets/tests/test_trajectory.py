"""Real training, late-step corruptions, and failed checkpoint restores."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from precision_validator.common import ASSET_ROOT, read, save, digest, load_plan, ValidationError
from precision_validator.trajectory import validate_spec
from precision_validator.trajectory_compare import compare_suite


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires the pinned image's torch")
class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ASSET_ROOT / "tests"))
        from run_training_trajectory import run
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "trajectory"
        run(cls.root)
        cls.path = cls.root / "results/plan.json"
        cls.plan = load_plan(cls.path)
        cls.original = {p: p.read_bytes() for p in cls.root.rglob("*") if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tearDown(self):
        for path, data in self.original.items():
            path.write_bytes(data)

    def step_mutation(self, stage, number, fn):
        folder = self.path.parent / stage
        path = folder / "steps" / f"{number:06d}" / "receipt.json"
        row = read(path)
        fn(row, path.parent)
        save(path, row)
        rank = read(folder / "rank0.json")
        rank["step_receipts"][str(path.relative_to(folder))] = digest(path)
        save(folder / "rank0.json", rank)
        final = read(folder / "complete.json")
        final["ranks"]["rank0.json"] = digest(folder / "rank0.json")
        save(folder / "complete.json", final)

    def test_real_three_inputs_adam_states_and_fresh_process_resume(self):
        result = compare_suite(self.path)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["coverage"]["distinct_inputs"], 3)
        self.assertEqual(result["coverage"]["steps_per_stage"], 3)
        for step in result["comparisons"].values():
            self.assertTrue(all(v["max_abs"] == 0 for v in step["resume"].values()))
        rank = read(self.path.parent / "resume/rank0.json")
        self.assertNotEqual(rank["resume"]["producer_pid"], rank["resume"]["consumer_pid"])
        lr = [read(self.path.parent / "reference/steps" / f"{i:06d}" / "receipt.json")["lr_used"] for i in range(1, 4)]
        self.assertEqual(lr, [[.01], [.005], [.0025]])

    def test_bad_late_loss_cannot_hide_behind_good_first_steps(self):
        self.step_mutation("candidate", 3, lambda row, _: row["observations"]["loss"].update(values=[1e6]))
        report = compare_suite(self.path)
        self.assertEqual(report["status"], "numerical_failure")
        self.assertFalse(report["gates"]["step/3/observation/loss"]["passed"])
        self.assertTrue(report["gates"]["step/1/observation/loss"]["passed"])

    def mutate_tensor(self, row, folder, field):
        piece = next(p for p in row["storage"]["pieces"] if p["field"] == field and p["name"] == "head.bias")
        file = folder / piece["file"]
        values = np.fromfile(file, dtype="<f4")
        start, end = piece["offset_bytes"] // 4, piece["offset_bytes"] // 4 + piece["elements"]
        values[start:end] *= -1
        values.tofile(file)
        row["storage"]["files"][file.name]["sha256"] = digest(file)
        if field == "exp_avg":
            import hashlib
            i = list(self.plan["parameters"]).index("head.bias")
            row["state"]["optimizer"]["state"][str(i)][field]["sha256"] = hashlib.sha256(values[start:end].tobytes()).hexdigest()

    def test_late_gradient_sign_flip_fails_local_gate(self):
        self.step_mutation("candidate", 3, lambda r, f: self.mutate_tensor(r, f, "gradients"))
        result = compare_suite(self.path)
        self.assertFalse(result["gates"]["step/3/parameters/gradients/head.bias/rule0"]["passed"])

    def test_momentum_error_fails_even_when_gradients_and_updates_match(self):
        self.step_mutation("candidate", 3, lambda r, f: self.mutate_tensor(r, f, "exp_avg"))
        result = compare_suite(self.path)
        self.assertEqual(result["status"], "numerical_failure")
        self.assertTrue(result["gates"]["step/3/gradients"]["passed"])
        self.assertTrue(result["gates"]["step/3/updates"]["passed"])
        self.assertFalse(result["gates"]["step/3/parameters/exp_avg/head.bias/rule0"]["passed"])

    def test_missing_adam_tensor_is_incomplete_evidence(self):
        self.step_mutation("candidate", 2, lambda r, _: r["storage"].update(pieces=[p for p in r["storage"]["pieces"] if p["field"] != "exp_avg_sq"]))
        with self.assertRaises(ValidationError):
            compare_suite(self.path)

    def test_cursor_skipping_a_sample_is_rejected(self):
        self.step_mutation("candidate", 2, lambda r, _: r["state"]["data_cursor"].update(next_step=4))
        with self.assertRaisesRegex(ValidationError, "cursor"):
            compare_suite(self.path)

    def test_wrong_learning_rate_is_rejected(self):
        self.step_mutation("candidate", 2, lambda r, _: r.update(lr_used=[.1]))
        with self.assertRaisesRegex(ValidationError, "learning rate"):
            compare_suite(self.path)

    def test_truncated_trajectory_is_rejected(self):
        self.step_mutation("candidate", 3, lambda r, _: r["optimizer_step_counters"].update({"head.bias": 2}))
        with self.assertRaisesRegex(ValidationError, "step counter"):
            compare_suite(self.path)

    def objects(self):
        import torch
        from precision_validator._assets.examples.training_trajectory import TokenModel, seed_all
        seed_all(9001)
        model = TokenModel()
        options = dict(self.plan["contract"]["optimizer"]["options"])
        options["betas"] = tuple(options["betas"])
        optimizer = torch.optim.AdamW(model.parameters(), **options)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
        return model, optimizer, scheduler

    def test_restore_rejects_dropped_momentum(self):
        from precision_validator.trajectory_sdk import restore_checkpoint
        model, optimizer, scheduler = self.objects()
        native = optimizer.load_state_dict
        def wrong(state):
            native(state)
            for s in optimizer.state.values():
                s["exp_avg"].zero_()
        with patch.object(optimizer, "load_state_dict", side_effect=wrong), self.assertRaisesRegex(ValidationError, "Restored"):
            restore_checkpoint(self.plan, model, optimizer, scheduler)

    def test_restore_rejects_missing_rng_restore(self):
        from precision_validator.trajectory_sdk import restore_checkpoint
        objects = self.objects()
        with patch("precision_validator.trajectory_sdk.restore_rng"), self.assertRaisesRegex(ValidationError, "Restored"):
            restore_checkpoint(self.plan, *objects)

    def test_restore_rejects_wrong_cursor(self):
        from precision_validator.trajectory_sdk import restore_checkpoint
        path = self.path.parent / "resume/checkpoint.json"
        row = read(path)
        row["cursor"]["next_step"] = 1
        save(path, row)
        with self.assertRaisesRegex(ValidationError, "cursor"):
            restore_checkpoint(self.plan, *self.objects())

    def test_restore_rejects_bad_checkpoint_bytes(self):
        from precision_validator.trajectory_sdk import restore_checkpoint
        path = self.path.parent / "resume/checkpoint.pt"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValidationError, "checksum"):
            restore_checkpoint(self.plan, *self.objects())

    def test_strict_state_gates_cannot_omit_parameters(self):
        spec = read(self.root / "spec.json")
        spec["trajectory"]["state_acceptance"]["parameters"][0]["pattern"] = "head.*"
        with self.assertRaisesRegex(ValidationError, "coverage"):
            validate_spec(spec)

    def test_unsupported_distributed_and_mixed_precision_do_not_claim_coverage(self):
        original = read(self.root / "spec.json")
        for mutation in (lambda s: s["configurations"]["candidate"].update(world_size=2),
                         lambda s: s["contract"]["precision"].update(parameters="bfloat16")):
            spec = copy.deepcopy(original)
            mutation(spec)
            with self.assertRaises(ValidationError):
                validate_spec(spec)

    def test_zero_length_or_external_resume_boundary_is_rejected(self):
        for boundary in (0, 3, True):
            spec = read(self.root / "spec.json")
            spec["trajectory"]["resume_after_step"] = boundary
            with self.assertRaisesRegex(ValidationError, "boundary"):
                validate_spec(spec)
