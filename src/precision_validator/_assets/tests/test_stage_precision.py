import copy
import importlib.util
from pathlib import Path
import runpy
import tempfile
import unittest

from precision_validator.common import ASSET_ROOT, ValidationError, load_plan, save, read, digest
from precision_validator.precision import FP32_STORAGE, stage_precision
from precision_validator.protocol import prepare, validate_spec
from precision_validator.protocol_compare import compare_suite
from test_protocol import specification, mutate


def mixed_spec(root):
    spec = specification(root, policy=False)
    spec["configurations"]["candidate"]["world_size"] = 1
    del spec["contract"]["precision"]
    spec["contract"]["precision_by_stage"] = {
        "reference": {**FP32_STORAGE, "compute": "float32"},
        "candidate": {**FP32_STORAGE, "compute": "bfloat16_autocast"}}
    spec["contract"]["precision_comparison"] = {
        "allowed_changes": ["compute"], "initial_weights_alignment": "identical_fp32"}
    return spec


class StagePrecisionContractTests(unittest.TestCase):
    def test_exact_allowed_changes_and_storage_alignment_required(self):
        with tempfile.TemporaryDirectory() as path:
            spec = mixed_spec(path)
            validate_spec(spec)
            self.assertEqual(stage_precision(spec, "repeat"), stage_precision(spec, "reference"))
            bad = []
            value = copy.deepcopy(spec); value["contract"]["precision_comparison"]["allowed_changes"] = []; bad.append(value)
            value = copy.deepcopy(spec); value["contract"]["precision_comparison"]["allowed_changes"] += ["parameters"]; bad.append(value)
            value = copy.deepcopy(spec); value["contract"]["precision_comparison"]["initial_weights_alignment"] = "rounded_bf16"; bad.append(value)
            value = copy.deepcopy(spec); value["contract"]["precision_by_stage"]["candidate"]["parameters"] = "bfloat16"; bad.append(value)
            value = copy.deepcopy(spec); value["contract"]["precision_by_stage"]["candidate"]["compute"] = "float16"; bad.append(value)
            value = copy.deepcopy(spec); value["contract"]["precision"] = FP32_STORAGE; bad.append(value)
            value = copy.deepcopy(spec); value["configurations"]["candidate"]["world_size"] = 2; bad.append(value)
            for value in bad:
                with self.subTest(value=value["contract"]), self.assertRaises(ValidationError):
                    validate_spec(value)


class CausalInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normalize = staticmethod(runpy.run_path(str(ASSET_ROOT / "examples/hf_causal_lm/inputs.py"))["normalize_batch"])

    def test_masks_apply_to_shifted_target_not_prediction_position(self):
        result = self.normalize({"input_ids": [[1, 2, 3, 4], [5, 6, 0, 0]],
            "attention_mask": [[1, 1, 1, 1], [1, 1, 0, 0]],
            "loss_mask": [[0, 0, 1, 1], [0, 1, 0, 0]],
            "labels": [[-100, -100, 7, 4], [-100, 6, -100, -100]]}, 8)
        self.assertEqual(result["labels"], [[-100, -100, 7, 4], [-100, 6, -100, -100]])
        self.assertEqual(result["supervised_tokens"], 3)
        self.assertEqual(result, self.normalize(result, 8))

    def test_legacy_sequence_is_one_unpadded_sample(self):
        row = self.normalize({"input_ids": [1, 2, 3]}, 8)
        self.assertEqual(row["labels"], [[-100, 2, 3]])
        self.assertEqual(row["supervised_tokens"], 2)

    def test_malformed_masks_cannot_change_training_objective_silently(self):
        cases = [
            {"input_ids": [[1, 2], [3]]}, {"input_ids": [True, 2]},
            {"input_ids": [1, 8]}, {"input_ids": [1]},
            {"input_ids": [1, 2, 3], "attention_mask": [0, 1, 1]},
            {"input_ids": [1, 2, 3], "attention_mask": [1, 0, 1]},
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 0], "loss_mask": [0, 1, 1]},
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 0], "labels": [-100, 2, 3]},
            {"input_ids": [1, 2], "loss_mask": [0, 0]},
            {"input_ids": [1, 2], "loss_mask": [0, .5]},
            {"input_ids": [1, 2], "labels": [-100, -1]},
            {"input_ids": [1, 2], "labels": [-100, -100]},
            {"input_ids": [1, 2], "supervised_tokens": 2},
            {"input_ids": [1, 2], "position_ids": [0, 1]},
            {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0], "sample_positions": [2]},
        ]
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ValidationError):
                self.normalize(row, 8)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires pinned image torch")
class NativeStagePrecisionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.spec = mixed_spec(self.root / "input")
        self.spec["parameters"] = {"weight": {"shape": [2, 3]}, "bias": {"shape": [2]}}
        self.spec["observables"] = {"loss": [], "custom.activation": [2]}
        self.spec["required_checks"] = []
        save(self.root / "spec.json", self.spec)
        prepare(self.root / "spec.json", self.root / "result", budget_gib=.01, retain_raw=True)
        self.path = self.root / "result/plan.json"
        self.plan = load_plan(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def execute(self, stage, *, apply_autocast=True, backward_in_autocast=False, change_weights=False):
        from precision_validator.sdk import capture_torch_step
        torch = self.torch
        torch.manual_seed(73)
        model = torch.nn.Linear(3, 2).float()
        if change_weights:
            with torch.no_grad():
                model.weight[0, 0] += .01
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        enable = stage == "candidate" and apply_autocast
        def backward():
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enable):
                output = model(torch.tensor([[.3, .7, -.4]]))
                loss = output.float().square().mean()
                if backward_in_autocast:
                    loss.backward()
            if not backward_in_autocast:
                loss.backward()
            return {"loss": loss.detach(), "custom.activation": output.detach()[0].float()}
        capture_torch_step(self.path, stage, model, optimizer, backward)
        folder = self.path.parent / stage
        save(folder / "process.json", {"returncode": 0})
        save(folder / "complete.json", {"status": "stage_complete", "stage": stage,
            "plan_sha256": self.plan["plan_sha256"], "ranks": {"rank0.json": digest(folder / "rank0.json")},
            "process_sha256": digest(folder / "process.json")})

    def test_real_fp32_vs_bf16_compute_is_reported_without_inventing_thresholds(self):
        for stage in ("reference", "repeat", "candidate"):
            self.execute(stage)
        result = compare_suite(self.path)
        self.assertEqual(result["status"], "report_only")
        self.assertEqual(result["comparisons"]["repeat"]["gradients"]["max_abs"], 0)
        self.assertGreater(result["comparisons"]["candidate"]["gradients"]["max_abs"], 0)
        row = read(self.path.parent / "candidate/rank0.json")
        self.assertEqual(row["precision_evidence"]["modules"][""]["output_dtype"], "torch.bfloat16")

    def test_mislabeled_bf16_execution_fails_at_native_operation(self):
        with self.assertRaisesRegex(ValidationError, "autocast was not applied"):
            self.execute("candidate", apply_autocast=False)

    def test_backward_must_run_outside_autocast(self):
        with self.assertRaisesRegex(ValidationError, "backward outside autocast"):
            self.execute("candidate", backward_in_autocast=True)

    def test_stage_precision_does_not_permit_different_initial_weights(self):
        for stage in ("reference", "repeat", "candidate"):
            self.execute(stage, change_weights=stage == "candidate")
        with self.assertRaisesRegex(ValidationError, "Initial model"):
            compare_suite(self.path)

    def test_missing_native_precision_evidence_is_rejected(self):
        for stage in ("reference", "repeat", "candidate"):
            self.execute(stage)
        mutate(self.path, "candidate", 0, lambda row: row.pop("precision_evidence"))
        with self.assertRaisesRegex(ValidationError, "Missing native precision"):
            compare_suite(self.path)

    def test_repeat_cannot_silently_use_candidate_precision(self):
        for stage in ("reference", "repeat", "candidate"):
            self.execute(stage)
        mutate(self.path, "repeat", 0, lambda row: row["precision"].update(compute="bfloat16_autocast"))
        with self.assertRaisesRegex(ValidationError, "Precision contract mismatch"):
            compare_suite(self.path)
