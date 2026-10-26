import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from precision_validator.common import ASSET_ROOT, ValidationError, read, save, digest, object_hash, load_plan
from precision_validator.protocol import prepare, STAGES, verify_inputs
from precision_validator.sdk import RankWriter
from precision_validator.protocol_compare import Stage, compare_suite, compare_regression, cleanup_passed
from precision_validator.protocol_runner import run_stage
from precision_validator.telemetry import Telemetry


def specification(root, policy=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "input.json").write_text('{"example": "protocol_test"}')
    (root / "train.py").write_text('raise SystemExit(0)\n')
    return {"schema_version": 1, "name": "architecture-neutral-test", "scope": "integration_test",
        "artifacts": {"code": str(root / "train.py"), "input": str(root / "input.json")},
        "parameters": {"arbitrary.encoder.kernel": {"shape": [2, 3], "group": "encoder"}, "scalar_gain": {"shape": []}},
        "configurations": {"reference": {"world_size": 1, "device": "cpu"}, "candidate": {"world_size": 4, "device": "cpu"}},
        "command": ["{python}", str(root / "train.py"), "--plan", "{plan}", "--stage", "{stage}"],
        "contract": {"seed": 123, "gradient_point": "after_reduction_before_clipping",
            "update_point": "optimizer_parameter_after_minus_before", "optimizer_steps": 1,
            "optimizer_initial_state": "fresh", "optimizer": {"name": "SGD", "options": {"lr": 0.1}},
            "objective": {"normalization": "sum_over_declared_samples"}, "precision": {"parameters": "float32"}},
        "observables": {"loss": [], "custom.activation": [2]}, "required_checks": ["native_backward"],
        "policy": {f: {"repeat_relative_ceiling": 0.01, "target_relative_ceiling": 0.02, "noise_multiplier": 2,
                       "noise_slack": 0.001} for f in ("gradients", "updates")} if policy else None}


@patch("precision_validator.sdk.runtime_info", new=lambda: {"python": "synthetic", "packages_sha256": "a" * 64})
def fixture(root, scale=1.0005, policy=True):
    root = Path(root)
    spec = specification(root / "inputs", policy)
    save(root / "spec.json", spec)
    prepare(root / "spec.json", root / "results", budget_gib=0.01, retain_raw=True)
    path = root / "results/plan.json"
    plan = load_plan(path)
    baseline = np.arange(1, 8, dtype=np.float32)
    for stage in STAGES:
        count = plan["configurations"][stage]["world_size"]
        ranks = {}
        for rank in range(count):
            writer = RankWriter(plan, stage, rank)
            for name, info in plan["parameters"].items():
                values = baseline[:6] if info["shape"] else baseline[6:]
                writer.initial_parameter(name, "float32", hashlib.sha256(values.tobytes()).hexdigest())
                if count == 1:
                    start, end = 0, len(values)
                elif info["shape"]:
                    start, end = ((0, 2), (2, 6), (0, 0), (0, 0))[rank]
                else:
                    start, end = (0, 1) if rank == 2 else (0, 0)
                if start != end:
                    factor = scale if stage == "candidate" else 1.0
                    for field, weight in (("gradients", 1), ("updates", 0.1)):
                        writer.append(field, name, start, end, [values[start:end] * factor * weight])
            if rank == 0:
                writer.observe("loss", 1.0)
                writer.observe("custom.activation", [2, 3])
            writer.finish(execution={"backward_completed": True, "gradients_reduced": True, "optimizer_succeeded": True,
                "optimizer_steps": 1, "teardown_completed": True}, precision=plan["contract"]["precision"],
                device={"kind": "cpu", "name": "test"}, checks={"native_backward": True})
            ranks[f"rank{rank}.json"] = digest(writer.folder / f"rank{rank}.json")
        folder = Path(plan["output"]) / stage
        save(folder / "process.json", {"returncode": 0})
        save(folder / "complete.json", {"status": "stage_complete", "stage": stage, "plan_sha256": plan["plan_sha256"],
             "ranks": ranks, "process_sha256": digest(folder / "process.json")})
    return path


def mutate(plan_path, stage, rank, callback):
    folder = plan_path.parent / stage
    path = folder / f"rank{rank}.json"
    row = read(path)
    callback(row)
    save(path, row)
    final = read(folder / "complete.json")
    final["ranks"][path.name] = digest(path)
    save(folder / "complete.json", final)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = fixture(self.root / "suite")
        self.plan = load_plan(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_different_rank_counts_shard_shapes_and_scalar_parameters(self):
        result = compare_suite(self.path)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["comparisons"]["candidate"]["gradients"]["elements"], 7)
        self.assertIn("encoder", result["comparisons"]["candidate"]["gradients"]["groups"])
        self.assertIn("绝对 L2", (self.path.parent / "report.md").read_text())

    def test_without_policy_reports_metrics_and_never_passes_or_cleans(self):
        path = fixture(self.root / "no-policy", policy=False)
        result = compare_suite(path)
        self.assertEqual(result["status"], "report_only")
        self.assertTrue(list(path.parent.glob("*/*.f32")))
        with self.assertRaises(ValidationError):
            cleanup_passed(load_plan(path), result)

    def test_numerical_failure_preserves_raw(self):
        path = fixture(self.root / "bad", scale=1.5)
        result = compare_suite(path)
        self.assertEqual(result["status"], "numerical_failure")
        self.assertTrue(result["raw_retained"])

    def test_missing_rank_rejected(self):
        (self.path.parent / "candidate/rank3.json").unlink()
        with self.assertRaises(FileNotFoundError):
            compare_suite(self.path)

    def test_overlapping_ownership_rejected(self):
        def change(row):
            for piece in row["storage"]["pieces"]:
                piece.update(start=1, end=5)
        mutate(self.path, "candidate", 1, change)
        with self.assertRaisesRegex(ValidationError, "overlapping"):
            compare_suite(self.path)

    def test_initial_parameters_and_buffers_must_match(self):
        mutate(self.path, "candidate", 0, lambda r: r["initial_auxiliary"].update(buffer={"sha256": "b" * 64}))
        with self.assertRaisesRegex(ValidationError, "Initial model/buffer"):
            compare_suite(self.path)

    def test_parameter_catalog_cannot_be_deduced_from_only_captured_tensors(self):
        mutate(self.path, "candidate", 2, lambda r: r["storage"].update(files={}, pieces=[]))
        with self.assertRaisesRegex(ValidationError, "Coverage"):
            compare_suite(self.path)

    def test_missing_adapter_check_or_update_rejected(self):
        mutate(self.path, "candidate", 0, lambda r: r["checks"].update(native_backward=False))
        with self.assertRaisesRegex(ValidationError, "Required adapter check"):
            compare_suite(self.path)

    def test_precision_drift_rejected(self):
        mutate(self.path, "candidate", 0, lambda r: r["precision"].update(parameters="bfloat16"))
        with self.assertRaisesRegex(ValidationError, "Precision"):
            compare_suite(self.path)

    def test_incorrect_input_or_normalization_rejected(self):
        mutate(self.path, "candidate", 0, lambda r: r["contract"]["objective"].update(normalization="different"))
        with self.assertRaisesRegex(ValidationError, "contract mismatch"):
            compare_suite(self.path)

    def test_raw_hash_mismatch_rejected(self):
        path = self.path.parent / "candidate/gradients.rank0.f32"
        path.write_bytes(b"\0" * path.stat().st_size)
        with self.assertRaisesRegex(ValidationError, "checksum"):
            compare_suite(self.path)

    def test_nonfinite_with_updated_checksum_rejected(self):
        path = self.path.parent / "candidate/gradients.rank0.f32"
        with path.open("r+b") as f:
            f.write(np.array([np.nan], dtype="<f4").tobytes())
        mutate(self.path, "candidate", 0, lambda r: r["storage"]["files"][path.name].update(sha256=digest(path)))
        with self.assertRaisesRegex(ValidationError, "Nonfinite"):
            compare_suite(self.path)

    def test_process_failure_cannot_be_hidden_by_complete_tensor_receipts(self):
        folder = self.path.parent / "candidate"
        save(folder / "process.json", {"returncode": 1})
        final = read(folder / "complete.json")
        final["process_sha256"] = digest(folder / "process.json")
        save(folder / "complete.json", final)
        with self.assertRaisesRegex(ValidationError, "Process"):
            compare_suite(self.path)

    def test_adapter_success_without_evidence_is_incomplete(self):
        root = self.root / "empty"
        spec = specification(root / "inputs")
        save(root / "spec.json", spec)
        prepare(root / "spec.json", root / "results", budget_gib=0.01)
        with self.assertRaises(FileNotFoundError):
            run_stage(root / "results/plan.json", "reference")
        self.assertFalse((root / "results/reference/complete.json").exists())
        self.assertEqual(read(root / "results/reference/status.json")["status"], "execution_failure")

    def test_input_changes_stop_launch(self):
        Path(self.plan["artifacts"]["input"]["path"]).write_text("changed")
        with self.assertRaisesRegex(ValidationError, "File changed"):
            verify_inputs(self.plan)

    def test_runtime_drift_is_not_accepted_as_numeric_noise(self):
        for rank in range(4):
            mutate(self.path, "candidate", rank, lambda r: r["runtime"].update(packages_sha256="b" * 64))
        with self.assertRaisesRegex(ValidationError, "Runtime changed"):
            compare_suite(self.path)

    def test_python_launcher_overrides_are_rejected_before_execution(self):
        spec = specification(self.root / "reserved")
        spec["environment"] = {"PYTHONPATH": "/untracked"}
        save(self.root / "reserved/spec.json", spec)
        with self.assertRaisesRegex(ValidationError, "launcher variables"):
            prepare(self.root / "reserved/spec.json", self.root / "reserved/results")

    def test_full_model_scope_requires_weight_identity(self):
        spec = specification(self.root / "full-model")
        spec["scope"] = "full_model_single_step"
        save(self.root / "full-model/spec.json", spec)
        with self.assertRaisesRegex(ValidationError, "weight files"):
            prepare(self.root / "full-model/spec.json", self.root / "full-model/results")

    def test_cleanup_owned_data_only_and_idempotent(self):
        report = compare_suite(self.path)
        sentinel = self.path.parent / "do-not-delete.bin"
        sentinel.write_text("keep")
        self.assertEqual(cleanup_passed(self.plan, report), 7 * 4 * 2 * 3)
        self.assertEqual(cleanup_passed(self.plan, report), 0)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_regression_rejects_common_mode_errors(self):
        second = fixture(self.root / "second")
        for stage in STAGES:
            folder = second.parent / stage
            for rank in range(load_plan(second)["configurations"][stage]["world_size"]):
                for field in ("gradients", "updates"):
                    path = folder / f"{field}.rank{rank}.f32"
                    if path.exists():
                        (np.fromfile(path, dtype="<f4") * 2).tofile(path)
                        mutate(second, stage, rank, lambda r, p=path: r["storage"]["files"][p.name].update(sha256=digest(p)))
        self.assertEqual(compare_suite(second)["status"], "passed")
        self.assertEqual(compare_regression(self.path, second)["status"], "numerical_failure")


class TelemetryTests(unittest.TestCase):
    def test_slow_telemetry_does_not_replace_required_rank_exit(self):
        entered, release = threading.Event(), threading.Event()
        def slow():
            entered.set()
            release.wait(5)
        telemetry = Telemetry(slow, interval=0.001).start()
        self.assertTrue(entered.wait(2))
        try:
            receipt = telemetry.stop(timeout=0.001)
            self.assertFalse(receipt["stopped"])
            self.assertEqual(receipt["scope"], "optional_progress_only")
        finally:
            release.set()
            telemetry.thread.join(timeout=2)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires image torch")
class TorchIntegrationTests(unittest.TestCase):
    def test_two_architectures_run_real_backward_and_actual_adam_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            for architecture in ("mlp", "attention"):
                folder = Path(temp) / architecture
                result = subprocess.run([sys.executable, str(ASSET_ROOT / "examples/torch_models.py"), "make", "--model", architecture,
                                         "--output", str(folder), "--example-thresholds"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                prepare(folder / "spec.json", folder / "results", budget_gib=0.01, retain_raw=True)
                for stage in STAGES:
                    run_stage(folder / "results/plan.json", stage)
                report = compare_suite(folder / "results/plan.json")
                self.assertEqual(report["status"], "passed", report["gates"])
                self.assertGreater(report["comparisons"]["candidate"]["gradients"]["elements"], 50)
                self.assertEqual(report["comparisons"]["repeat"]["gradients"]["max_abs"], 0)
