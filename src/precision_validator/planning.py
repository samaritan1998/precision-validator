import math
import argparse
import runpy
from pathlib import Path
from .common import ROOT, ASSET_ROOT, STAGES, save, read, digest, require, object_hash, source_inventory, model_inventory, now
from .profiles import get_profile, contribution_count, select_sample, launch_args


def toolkit_inventory():
    paths = [*ROOT.glob("precision_validator/*.py"), *ASSET_ROOT.glob("profiles/*.json"),
             *ASSET_ROOT.glob("policies/*.json"), ASSET_ROOT / "runtime-lock.json"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def validate_native_profiles(plan):
    """Validate the rendered CLI against the candidate's CPU-only profile contract."""
    file = Path(plan["source"]) / "slime_plugins/models/qwen_cp/profiles.py"
    require(file.is_file(), "Candidate does not implement the qwen_cp profile contract")
    native = runpy.run_path(str(file))
    checks = []
    for stage in STAGES:
        parser = argparse.ArgumentParser(add_help=False)
        for field in ("tensor-model-parallel-size", "pipeline-model-parallel-size", "context-parallel-size",
                      "expert-model-parallel-size", "expert-tensor-parallel-size", "recompute-num-layers",
                      "micro-batch-size", "seq-length"):
            parser.add_argument("--" + field, type=int)
        for field in ("attention-dropout", "hidden-dropout"):
            parser.add_argument("--" + field, type=float)
        for field in ("qwen-cp-profile", "custom-megatron-init-path", "qwen-gdn-backend", "megatron-to-hf-mode",
                      "qkv-format", "loss-type", "optimizer", "recompute-granularity", "recompute-method",
                      "cp-comm-type", "moe-token-dispatcher-type", "moe-router-dtype"):
            parser.add_argument("--" + field)
        for field in ("bf16", "calculate-per-token-loss", "optimizer-cpu-offload", "use-distributed-optimizer",
                      "use-precision-aware-optimizer", "accumulate-allreduce-grads-in-fp32"):
            parser.add_argument("--" + field, action="store_true")
        args, _ = parser.parse_known_args(launch_args(plan, stage))
        result = native["validate_args"](args)
        require(result.name == plan["profile"]["name"] and args.accumulate_allreduce_grads_in_fp32, "Candidate profile validation failed")
        checks.append({"stage": stage, "pp": args.pipeline_model_parallel_size, "cp": args.context_parallel_size,
                       "ep": args.expert_model_parallel_size})
    return {"native_profiles_sha256": digest(file), "checks": checks}


def prepare(args):
    profile = get_profile(args.profile)
    output = Path(args.output).resolve()
    require(not output.exists(), "Output already exists; use a new run directory")
    # Resolve all inputs before creating the output directory.
    source = Path(args.source).resolve()
    model = Path(args.model).resolve()
    inputs = Path(args.input).resolve()
    candidate = source_inventory(source)
    print("Hashing model shards (once per plan)...", flush=True)
    weights = model_inventory(model)
    sample, sample_proof = select_sample(read(inputs), profile["sequence_length"], args.sample_index)
    require(max(sample["token_ids"]) < read(model / "config.json").get("text_config", read(model / "config.json")).get("vocab_size", 248320),
            "Input token ID exceeds vocabulary")
    copies = contribution_count(profile)
    expected_bytes = 3 * profile["trainable_elements"] * 8
    require(args.disk_budget_gib * 2**30 >= expected_bytes * 1.1,
            f"Need at least {math.ceil(expected_bytes * 1.1 / 2**30)} GiB for three full gradient/update stages")
    policy = read(args.policy) if args.policy else read(ASSET_ROOT / "policies/qwen-noise-v1.json")
    for field in ("gradients", "updates"):
        p = policy[field]
        for key in ("repeat_relative_ceiling", "target_relative_ceiling", "noise_multiplier", "noise_slack"):
            require(type(p[key]) in (int, float) and math.isfinite(p[key]) and p[key] >= 0, f"Invalid policy {field}.{key}")
        for key in ("max_abs_ceiling", "rmse_ceiling"):
            require(p.get(key) is None or (math.isfinite(p[key]) and p[key] >= 0), f"Invalid policy {key}")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "input.json", {"samples": [sample]})
    plan = {"schema_version": 1, "adapter": "slime-qwen-v1", "created_utc": now(),
        "profile": profile, "source": str(source), "source_inventory": candidate,
        "model_path": str(model), "model_inventory": weights,
        "input_path": str(output / "input.json"), "input_sha256": digest(output / "input.json"),
        "input_proof": {**sample_proof, "original_input_sha256": digest(inputs)},
        "sample_copies_per_step": copies, "unique_samples": 1, "optimizer_steps": 1,
        "learning_rate": 2e-5, "seed": 1234, "policy": policy,
        "runtime_lock": read(ASSET_ROOT / "runtime-lock.json"), "toolkit_files": toolkit_inventory(),
        "disk_budget_bytes": int(args.disk_budget_gib * 2**30), "estimated_raw_bytes": expected_bytes,
        "stage_timeout_seconds": args.timeout_seconds, "cleanup_on_pass": not args.retain_raw,
        "output": str(output), "megatron_path": str(Path(args.megatron).resolve()),
        "production_model_controls": True, "external_diagnostic_patches": False}
    require(plan["stage_timeout_seconds"] > 0, "Timeout must be positive")
    plan["native_profile_preflight"] = validate_native_profiles(plan)
    plan["plan_sha256"] = object_hash(plan)
    save(output / "plan.json", plan)
    save(output / "launch-preview.json", {stage: launch_args(plan, stage) for stage in STAGES})
    return {"plan": str(output / "plan.json"), "profile": profile["name"],
            "unique_samples": 1, "sample_copies_per_step": copies,
            "raw_budget_gib": args.disk_budget_gib, "estimated_raw_gib": round(expected_bytes / 2**30, 1),
            "stages": list(STAGES), "note": "Each stage can run independently on one 8-GPU node"}
