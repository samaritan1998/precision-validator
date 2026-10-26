import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from precision_validator.common import ASSET_ROOT, STAGES, ValidationError, digest, object_hash, read, save, load_plan
from precision_validator.profiles import profiles, get_profile, contribution_count, select_sample, topology, launch_args
from precision_validator.storage import ShardWriter, index_pieces, aligned_chunks, verify_storage
from precision_validator.metrics import Metrics, evaluate
from precision_validator.compare import Stage, compare_suite, compare_regression, cleanup_passed
from precision_validator.runner import classify_failure


def fixture(root, target_scale=1.001, repeat_scale=1.001, cleanup=False):
    """Small synthetic evidence for CPU contract tests, never GPU acceptance."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    profile = get_profile("q38-b200-256k")
    profile.update(trainable_elements=32, layers=1, hidden_size=32, sequence_length=64,
                   image_validation_status="synthetic_cpu_contract_fixture")
    source = {"slime/backends/megatron_utils/model.py": "a" * 64}
    plan = {"schema_version": 1, "adapter": "slime-qwen-v1", "profile": profile,
        "output": str(root), "source_inventory": {"commit": "a" * 40, "files": source},
        "input_proof": {"sample_sha256": "b" * 64, "supervised_tokens": 8},
        "runtime_lock": read(ASSET_ROOT / "runtime-lock.json"), "sample_copies_per_step": 4,
        "policy": read(ASSET_ROOT / "policies/qwen-noise-v1.json"), "cleanup_on_pass": cleanup,
        "model_inventory": {"sha256": "c" * 64}, "seed": 1234, "learning_rate": 2e-5, "toolkit_files": {}}
    plan["plan_sha256"] = object_hash(plan)
    save(root / "plan.json", plan)
    baseline = np.arange(1, 33, dtype=np.float32)
    for stage in STAGES:
        folder = root / stage
        folder.mkdir()
        layout = topology(profile, stage)
        cp, dp = layout["cp"], layout["dp"]
        environment = {"status": "environment_passed", "gpu_checked": True,
            "numerical_environment": profile["environment"],
            "runtime": {"versions": plan["runtime_lock"]["versions"], "megatron_sha256": "d" * 64},
            "devices": [{"index": i, "name": "NVIDIA B200", "capability": [10, 0]} for i in range(8)]}
        save(folder / "environment.json", environment)
        save(folder / "process.json", {"returncode": 0})
        rank_hashes = {}
        bounds = np.cumsum([0, 1, 2, 3, 4, 5, 6, 7, 4]).tolist() if stage == "cp_target" else list(range(0, 33, 4))
        for rank in range(8):
            piece_rank = 7 - rank if stage == "cp_target" else rank
            start, end = bounds[piece_rank:piece_rank + 2]
            piece = {"name": "language_model.weight", "shape": [32], "start": start, "end": end}
            scale = 1 if stage == "cp1" else repeat_scale if stage == "cp1_repeat" else target_scale
            writer = ShardWriter(folder, rank, 1024)
            writer.append("gradients", piece, [baseline[start:end] * scale])
            writer.append("updates", piece, [baseline[start:end] * scale * 1e-5])
            storage = writer.close()
            native = {}
            for kind in ("cp_model", "cp_step"):
                path = folder / "native" / f"{kind}-rank{rank}-pid123.json"
                save(path, {"rank": rank, "kind": kind})
                native[str(path.relative_to(folder))] = digest(path)
            fla = {"mode": "two-forward", "capability": [10, 0], "configs": [
                {"kernel": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "calls": 2,
                 "config": "BV: 32, num_warps: 2, num_stages: 2", "best_config": "BV: 32, num_warps: 2, num_stages: 2"},
                {"kernel": "chunk_fwd_kernel_o", "calls": 2, "config": "BK: 64, BV: 64, num_warps: 4, num_stages: 3",
                 "best_config": "BK: 64, BV: 64, num_warps: 4, num_stages: 3"}]}
            dp_rank = rank % dp
            count = 4 // dp
            row = {"rank": rank, "stage": stage, "plan_sha256": plan["plan_sha256"], "topology": layout,
                "optimizer_step_executed": True, "distributed_groups_destroyed": True,
                "gradient_stage": "after_prepare_grads_before_clipping", "sample_sha256": "b" * 64,
                "optimizer_observation": {"captured": True, "optimizer_step_calls": 1, "optimizer_success": True},
                "optimizer_dtypes": [{k: "torch.float32" for k in ("gradient", "master", "exp_avg", "exp_avg_sq")}],
                "grad_norm": 1.0, "entry": {name: {"relative_path": next(iter(source)), "sha256": "a" * 64}
                    for name in ("initialize", "setup_model_and_optimizer", "train_one_step")},
                "native": {"optimizer_updated": True, "grad_sync": {"calls": 1, "completed": 1}, "files": native, "fla": fla},
                "initial_parameters": {"language_model.weight": {"shape": [32], "dtype": "torch.bfloat16", "sha256": "e" * 64}},
                "schedule": {"dp_rank": dp_rank, "indices": list(range(dp_rank * count, (dp_rank + 1) * count)),
                    "microbatches": count, "batches": [[i] for i in range(count)], "supervised_tokens": 8},
                "storage": storage, "losses": {"loss": 1.0} if rank == 0 else {},
                "forward_samples": {"1": {"positions": list(range(64)), "features": list(range(32)),
                    "values": [[float(p + f) * scale for f in range(32)] for p in range(64)]}} if rank == 0 else {},
                "timings": {"train_step_wall_seconds": 1.0}, "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048}
            path = folder / f"rank{rank}.json"
            save(path, row)
            rank_hashes[path.name] = digest(path)
        save(folder / "complete.json", {"status": "stage_complete", "stage": stage, "plan_sha256": plan["plan_sha256"],
            "ranks": rank_hashes, "environment_sha256": digest(folder / "environment.json"), "process_sha256": digest(folder / "process.json")})
    return plan


def mutate_rank(root, stage, rank, change):
    folder = Path(root) / stage
    path = folder / f"rank{rank}.json"
    row = read(path)
    change(row)
    save(path, row)
    final = read(folder / "complete.json")
    final["ranks"][path.name] = digest(path)
    save(folder / "complete.json", final)


class ProfilesTests(unittest.TestCase):
    def test_six_topologies_and_matched_contributions(self):
        self.assertEqual(len(profiles()), 6)
        for name in profiles():
            p = get_profile(name)
            copies = contribution_count(p)
            for t in p["topologies"].values():
                self.assertEqual(8, t["tp"] * t["pp"] * t["cp"] * t["dp"])
                self.assertEqual(copies % t["dp"], 0)
            if name.startswith("q38-") and "-b200-" not in name:
                self.assertEqual(p["topologies"]["reference"]["pp"], 8)

    def test_profiles_have_no_absolute_experiment_paths_or_fp8(self):
        for name in profiles():
            p = get_profile(name)
            self.assertNotIn("/mnt/", json.dumps(p))
            self.assertNotIn("--fp8-format", p["argv"])
            self.assertEqual(p["environment"]["SLIME_FLA_CONFIG_CONTROL"], "two-forward")
            self.assertEqual(p["environment"]["SLIME_QWENVL_EXTERNAL_SFT_LOSS"], "0")
            plan = {"profile": p, "model_path": "/model", "input_path": "/run/input.json",
                    "sample_copies_per_step": contribution_count(p), "learning_rate": 2e-5, "seed": 1234}
            for stage in STAGES:
                argv = launch_args(plan, stage)
                self.assertIn("--accumulate-allreduce-grads-in-fp32", argv)
                self.assertEqual(argv[argv.index("--pipeline-model-parallel-size") + 1], str(topology(p, stage)["pp"]))
                self.assertEqual(argv[argv.index("--lr-warmup-init") + 1], "2e-05")

    def test_real_input_requires_length_mask_and_provenance(self):
        sample = {"token_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1], "metadata": {"source": "unit-test"}}
        selected, proof = select_sample({"samples": [sample]}, 4)
        self.assertEqual(proof["supervised_tokens"], 2)
        self.assertEqual(selected, sample)
        for bad in ({**sample, "loss_mask": [1, 0, 0, 1]}, {**sample, "token_ids": [1, 2]},
                    {**sample, "metadata": {}}, {**sample, "loss_mask": [0, 0, 0, 0]}):
            with self.assertRaises(ValidationError):
                select_sample({"samples": [bad]}, 4)


class MetricTests(unittest.TestCase):
    def test_absolute_and_relative_metrics(self):
        result = Metrics().add([2, 1], [1, 1]).result()
        self.assertEqual(result["max_abs"], 1)
        self.assertEqual(result["absolute_l2"], 1)
        self.assertAlmostEqual(result["rmse"], 1 / math.sqrt(2))
        self.assertAlmostEqual(result["relative_l2"], 1 / math.sqrt(2))

    def test_nonfinite_and_zero_reference(self):
        with self.assertRaises(ValidationError):
            Metrics().add([np.nan], [0])
        with self.assertRaises(ValidationError):
            Metrics().add([1], [np.inf])
        result = Metrics().add([1], [0]).result()
        self.assertIsNone(result["relative_l2"])
        self.assertFalse(evaluate(result, result, read(ASSET_ROOT / "policies/qwen-noise-v1.json")["gradients"])["passed"])

    def test_repeat_noise_is_bounded_and_never_auto_loosened(self):
        policy = read(ASSET_ROOT / "policies/qwen-noise-v1.json")["gradients"]
        good = Metrics().add([1.001], [1]).result()
        bad = Metrics().add([2.0], [1]).result()
        self.assertTrue(evaluate(good, good, policy)["passed"])
        self.assertFalse(evaluate(bad, bad, policy)["passed"])
        self.assertFalse(evaluate(good, bad, policy)["passed"])

    def test_absolute_gate_when_explicitly_configured(self):
        policy = read(ASSET_ROOT / "policies/qwen-noise-v1.json")["gradients"]
        policy["max_abs_ceiling"] = 0.01
        value = Metrics().add([100.1], [100]).result()
        self.assertFalse(evaluate(value, value, policy)["passed"])

    def test_failure_categories(self):
        self.assertEqual(classify_failure("torch.OutOfMemoryError: CUDA out of memory"), "oom")
        self.assertEqual(classify_failure("NCCL watchdog timed out"), "communication_failure")
        self.assertEqual(classify_failure("No space left on device"), "storage_failure")
        self.assertEqual(classify_failure("Nonfinite gradients"), "numerical_failure")


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "run"
        self.plan = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_cross_rank_and_different_shard_boundaries_pass(self):
        result = compare_suite(self.root / "plan.json")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["comparisons"]["cp_target"]["gradients"]["elements"], 32)
        self.assertIn("最大绝对误差", (self.root / "report.md").read_text())
        self.assertTrue(result["raw_retained"])

    def test_missing_rank_is_incomplete(self):
        (self.root / "cp_target/rank7.json").unlink()
        with self.assertRaises(FileNotFoundError):
            compare_suite(self.root / "plan.json")

    def test_missing_parameter_is_rejected(self):
        mutate_rank(self.root, "cp_target", 0, lambda r: r["storage"]["pieces"].pop())
        with self.assertRaises(ValidationError):
            Stage(self.plan, "cp_target")

    def test_overlapping_parameter_ownership_is_rejected(self):
        def change(row):
            for piece in row["storage"]["pieces"]:
                piece.update(start=0, end=4)
        mutate_rank(self.root, "cp_target", 0, change)
        with self.assertRaises(ValidationError):
            Stage(self.plan, "cp_target")

    def test_wrong_dtype_or_optimizer_skip_is_rejected(self):
        mutate_rank(self.root, "cp_target", 1, lambda r: r["optimizer_dtypes"][0].update(master="torch.bfloat16"))
        with self.assertRaises(ValidationError):
            Stage(self.plan, "cp_target")

    def test_native_update_skip_is_rejected(self):
        mutate_rank(self.root, "cp_target", 0, lambda r: r["optimizer_observation"].update(optimizer_success=False))
        with self.assertRaises(ValidationError):
            Stage(self.plan, "cp_target")

    def test_different_initial_weight_is_rejected(self):
        for rank in range(8):
            mutate_rank(self.root, "cp_target", rank, lambda r: r["initial_parameters"]["language_model.weight"].update(sha256="f" * 64))
        with self.assertRaisesRegex(ValidationError, "Initial model weights differ"):
            compare_suite(self.root / "plan.json")

    def test_tampered_raw_file_is_rejected(self):
        with (self.root / "cp_target/gradients.rank0.f32").open("r+b") as stream:
            stream.write(b"\0\0\0\0")
        with self.assertRaisesRegex(ValidationError, "checksum"):
            compare_suite(self.root / "plan.json")

    def test_nonfinite_raw_tensor_with_valid_checksum_is_rejected(self):
        path = self.root / "cp_target/gradients.rank0.f32"
        with path.open("r+b") as stream:
            stream.write(np.array([np.nan], dtype="<f4").tobytes())
        mutate_rank(self.root, "cp_target", 0, lambda r: r["storage"]["files"][path.name].update(sha256=digest(path)))
        with self.assertRaisesRegex(ValidationError, "Nonfinite"):
            compare_suite(self.root / "plan.json")

    def test_duplicate_dp_copy_is_rejected(self):
        mutate_rank(self.root, "cp_target", 0, lambda r: r["schedule"].update(indices=[0, 1, 2, 2]))
        with self.assertRaises(ValidationError):
            Stage(self.plan, "cp_target")

    def test_wrong_device_is_not_numeric_acceptance(self):
        path = self.root / "cp_target/environment.json"
        env = read(path)
        env["devices"][0]["name"] = "NVIDIA H100"
        save(path, env)
        final = read(self.root / "cp_target/complete.json")
        final["environment_sha256"] = digest(path)
        save(self.root / "cp_target/complete.json", final)
        with self.assertRaisesRegex(ValidationError, "Wrong device"):
            compare_suite(self.root / "plan.json")

    def test_cleanup_only_accepted_raw_files(self):
        report = compare_suite(self.root / "plan.json")
        sentinel = self.root / "unrelated.bin"
        sentinel.write_text("keep")
        removed = cleanup_passed(self.plan, report)
        self.assertEqual(removed, 3 * 32 * 8)
        self.assertTrue(sentinel.exists())
        self.assertTrue((self.root / "cp_target/rank0.json").exists())
        self.assertFalse(list(self.root.glob("*/*.f32")))
        self.assertEqual(cleanup_passed(self.plan, report), 0)
        self.assertEqual(cleanup_passed(self.plan, report), 0)

    def test_default_success_cleanup(self):
        other = Path(self.temp.name) / "autoclean"
        fixture(other, cleanup=True)
        report = compare_suite(other / "plan.json")
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["raw_retained"])
        self.assertFalse(list(other.glob("*/*.f32")))

    def test_cleanup_refuses_symlink(self):
        report = compare_suite(self.root / "plan.json")
        path = self.root / "cp_target/gradients.rank0.f32"
        saved = self.root / "outside-target"
        path.rename(saved)
        path.symlink_to(saved)
        with self.assertRaises(ValidationError):
            cleanup_passed(self.plan, report)
        self.assertTrue(saved.exists())

    def test_cleanup_refuses_failure(self):
        with self.assertRaises(ValidationError):
            cleanup_passed(self.plan, {"status": "numerical_failure", "plan_sha256": self.plan["plan_sha256"]})

    def test_plan_tampering_is_rejected(self):
        changed = read(self.root / "plan.json")
        changed["policy"]["gradients"]["target_relative_ceiling"] = 100
        save(self.root / "plan.json", changed)
        with self.assertRaisesRegex(ValidationError, "checksum"):
            load_plan(self.root / "plan.json")

    def test_numerical_failure_retains_raw(self):
        other = Path(self.temp.name) / "bad"
        fixture(other, target_scale=1.3)
        result = compare_suite(other / "plan.json")
        self.assertEqual(result["status"], "numerical_failure")
        self.assertTrue(list(other.glob("*/*.f32")))

    def test_regression_catches_common_mode_error(self):
        other = Path(self.temp.name) / "candidate"
        fixture(other, target_scale=1.001)
        # Candidate CP1 and target both wrong in the same direction: CP-only
        # consistency is insufficient, same-topology regression must reject it.
        for stage in STAGES:
            for rank in range(8):
                folder = other / stage
                for field in ("gradients", "updates"):
                    path = folder / f"{field}.rank{rank}.f32"
                    values = np.fromfile(path, dtype="<f4") * 2
                    values.tofile(path)
                    mutate_rank(other, stage, rank, lambda r, p=path: r["storage"]["files"][p.name].update(sha256=digest(p)))
        self.assertEqual(compare_suite(other / "plan.json")["status"], "passed")
        self.assertEqual(compare_regression(self.root / "plan.json", other / "plan.json")["status"], "numerical_failure")


if __name__ == "__main__":
    unittest.main()
