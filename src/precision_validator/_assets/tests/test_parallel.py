import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import test_protocol as fixtures
from precision_validator.acceptance import evaluate_limit
from precision_validator.common import ValidationError, digest, load_plan, read, save
from precision_validator.metrics import Metrics
from precision_validator.parallel import CONVENTION, parse_layout, prepare_parallel, validate_layout, validate_rank_groups
from precision_validator.protocol import validate_spec, STAGES
from precision_validator.protocol_compare import compare_suite, compare_regression


def strict_policy(spec):
    limit = {"relative_l2": 0.05, "max_abs": 0.01, "near_zero_rms": 1e-10, "near_zero_max_abs": 1e-10}
    spec["acceptance"] = {
        category: [{"pattern": "*", "gradients": copy.deepcopy(limit), "updates": copy.deepcopy(limit)}]
        for category in ("parameters", "groups")}
    spec["acceptance"]["observations"] = {key: copy.deepcopy(limit) for key in spec["observables"]}
    return spec


def strict_fixture(path):
    original = fixtures.specification
    with patch.object(fixtures, "specification", side_effect=lambda *a, **k: strict_policy(original(*a, **k))):
        return fixtures.fixture(path, scale=1.0)


def flip_small_parameter(path):
    plan = load_plan(path)
    for stage in STAGES:
        for rank in range(plan["configurations"][stage]["world_size"]):
            def mutate(row, stage=stage):
                for piece in row["storage"]["pieces"]:
                    if piece["name"] != "scalar_gain":
                        continue
                    value = 1e-4 if piece["field"] == "gradients" else 1e-5
                    raw = path.parent / stage / piece["file"]
                    with raw.open("r+b") as stream:
                        stream.seek(piece["offset_bytes"])
                        stream.write(np.array([-value if stage == "candidate" else value], dtype="<f4").tobytes())
                    row["storage"]["files"][piece["file"]]["sha256"] = digest(raw)
            fixtures.mutate(path, stage, rank, mutate)


class StrictAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_sign_flip_hidden_by_global_norm_is_rejected(self):
        path = strict_fixture(self.root / "local-error")
        flip_small_parameter(path)
        report = compare_suite(path)
        self.assertEqual(report["status"], "numerical_failure")
        self.assertTrue(report["gates"]["gradients"]["passed"])
        self.assertFalse(report["gates"]["parameters/gradients/scalar_gain/rule0"]["passed"])
        self.assertFalse(report["gates"]["groups/updates/scalar_gain/rule0"]["passed"])
        self.assertEqual(report["comparisons"]["candidate"]["gradients"]["parameters"]["scalar_gain"]["relative_l2"], 2)
        self.assertTrue(report["raw_retained"])

    def test_loss_divergence_is_rejected_even_with_identical_vectors(self):
        path = strict_fixture(self.root / "loss")
        fixtures.mutate(path, "candidate", 0, lambda row: row["observations"]["loss"].update(values=[1e6]))
        report = compare_suite(path)
        self.assertEqual(report["status"], "numerical_failure")
        self.assertTrue(report["gates"]["gradients"]["passed"])
        self.assertFalse(report["gates"]["observation/loss"]["passed"])

    def test_hidden_output_and_reference_repeat_are_also_gated(self):
        path = strict_fixture(self.root / "hidden")
        fixtures.mutate(path, "repeat", 0, lambda row: row["observations"]["custom.activation"].update(values=[20, 30]))
        report = compare_suite(path)
        self.assertEqual(report["status"], "numerical_failure")
        self.assertFalse(report["gates"]["observation/custom.activation"]["passed"])

    def test_near_zero_uses_explicit_absolute_bound(self):
        limit = {"relative_l2": 0.01, "max_abs": 1e-4, "near_zero_rms": 1e-9, "near_zero_max_abs": 1e-10}
        same = Metrics().add([0], [0]).result()
        tiny = Metrics().add([1e-11], [0]).result()
        wrong = Metrics().add([1e-8], [0]).result()
        self.assertTrue(evaluate_limit(same, tiny, limit)["passed"])
        self.assertFalse(evaluate_limit(same, wrong, limit)["passed"])
        self.assertIsNone(tiny["relative_l2"])

    def test_uncovered_names_typo_and_missing_observation_fail_preparation(self):
        for mutation in (
            lambda s: s["acceptance"]["observations"].pop("loss"),
            lambda s: s["acceptance"]["parameters"][0].update(pattern="encoder.typo.*"),
            lambda s: s["acceptance"]["parameters"][0].update(pattern="arbitrary.*"),
            lambda s: s["acceptance"]["groups"][0]["gradients"].update(near_zero_max_abs=1),
        ):
            spec = strict_policy(fixtures.specification(self.root / "inputs"))
            mutation(spec)
            with self.assertRaises(ValidationError):
                validate_spec(spec)

    def test_extra_critical_rule_is_additive_not_an_override(self):
        spec = strict_policy(fixtures.specification(self.root / "additive"))
        rule = copy.deepcopy(spec["acceptance"]["parameters"][0])
        rule["pattern"] = "scalar_gain"
        rule["gradients"]["relative_l2"] = 0
        spec["acceptance"]["parameters"].append(rule)
        validate_spec(spec)

    def test_regress_checks_shared_forward_error(self):
        good = strict_fixture(self.root / "good")
        bad = strict_fixture(self.root / "bad")
        for stage in STAGES:
            fixtures.mutate(bad, stage, 0, lambda row: row["observations"]["loss"].update(values=[1e6]))
        self.assertEqual(compare_suite(bad)["status"], "passed")
        report = compare_regression(good, bad)
        self.assertEqual(report["status"], "numerical_failure")
        self.assertFalse(report["gates"]["candidate/observation/loss"]["passed"])

    def test_legacy_report_is_explicitly_global_only(self):
        path = fixtures.fixture(self.root / "legacy", scale=1.0)
        report = compare_suite(path)
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["acceptance_coverage"]["strict"])


def synthetic_groups(layout, world):
    sizes = validate_layout(layout, world)
    rows = [{"rank": rank, "native_parallelism": {"convention": CONVENTION, "degrees": sizes, "groups": {}}} for rank in range(world)]
    for axes in (("tp", "cp", "dp", "pp"), ("etp", "ep", "edp", "pp")):
        stride = 1
        for axis in axes:
            size = sizes[axis]
            for rank in range(world):
                start = rank - (rank // stride % size) * stride
                rows[rank]["native_parallelism"]["groups"][axis] = [start + i * stride for i in range(size)]
            stride *= size
    return rows


class ParallelContractTests(unittest.TestCase):
    def test_ep_is_not_multiplied_into_dense_world(self):
        layout = parse_layout("tp=2,pp=2,cp=2,ep=4,etp=1", 16)
        self.assertEqual(layout["dp"], 2)
        self.assertEqual(validate_layout(layout, 16)["edp"], 2)
        validate_rank_groups({"world_size": 16, "parallelism": layout}, synthetic_groups(layout, 16))

    def test_invalid_world_and_duplicate_fields_are_rejected(self):
        for text in ("tp=3,pp=2,cp=1,ep=1,etp=1", "tp=2,tp=1,pp=1,cp=1,ep=1,etp=1", "tp=0,pp=1,cp=1,ep=1,etp=1"):
            with self.assertRaises(ValidationError):
                parse_layout(text, 8)

    def test_native_group_size_and_membership_must_match(self):
        layout = parse_layout("tp=2,pp=1,cp=1,ep=1,etp=1", 4)
        config = {"world_size": 4, "parallelism": layout}
        rows = synthetic_groups(layout, 4)
        validate_rank_groups(config, rows)
        rows[1]["native_parallelism"]["groups"]["tp"] = [1, 2]
        with self.assertRaises(ValidationError):
            validate_rank_groups(config, rows)

    def test_missing_receipts_cannot_certify_declared_tp(self):
        config = {"world_size": 2, "parallelism": parse_layout("tp=2,pp=1,cp=1,ep=1,etp=1", 2)}
        with self.assertRaises(ValidationError):
            validate_rank_groups(config, [{"rank": 0}, {"rank": 1}])

    def test_prepare_explicit_tp1_vs_tp2_freezes_all_contracts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = strict_policy(fixtures.specification(root / "inputs"))
            spec["parallel_adapter"] = {"convention": CONVENTION, "framework": "test-only", "supported_axes": ["tp", "dp"]}
            spec["contract"]["objective"].update(global_samples=8, supervised_units=24)
            save(root / "spec.json", spec)
            result = prepare_parallel(root / "spec.json", root / "results", "tp=1,pp=1,cp=1,ep=1,etp=1",
                                      "tp=2,pp=1,cp=1,ep=1,etp=1", 8, budget_gib=0.01)
            plan = load_plan(result["plan"])
            self.assertEqual(plan["configurations"]["reference"]["parallelism"]["dp"], 8)
            self.assertEqual(plan["configurations"]["candidate"]["parallelism"]["dp"], 4)
            self.assertEqual(plan["configurations"]["repeat"], plan["configurations"]["reference"])
            self.assertEqual(plan["parallel_validation"]["changed_axes"], ["tp", "dp"])
            self.assertEqual(plan["acceptance"], spec["acceptance"])

    def test_parallel_plan_without_complete_gates_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = fixtures.specification(Path(temp))
            spec["parallel_adapter"] = {"convention": CONVENTION, "framework": "test-only", "supported_axes": ["tp", "dp"]}
            for config in spec["configurations"].values():
                config["parallelism"] = parse_layout("tp=1,pp=1,cp=1,ep=1,etp=1", config["world_size"])
            with self.assertRaisesRegex(ValidationError, "frozen global"):
                validate_spec(spec)
