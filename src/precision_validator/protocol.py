"""Model-independent, immutable experiment contracts and raw tensor protocol."""
import copy
import math
from pathlib import Path
import re
from .common import ROOT, require, read, save, digest, object_hash, now, verify_files

STAGES = ("reference", "repeat", "candidate")
FIELDS = ("gradients", "updates")


def validate_policy(policy):
    if policy is None:
        return  # Metrics without acceptance; never borrow a model's thresholds.
    require(set(policy) == set(FIELDS), "Policy must declare gradients and updates")
    for field in FIELDS:
        row = policy[field]
        for key in ("repeat_relative_ceiling", "target_relative_ceiling", "noise_multiplier", "noise_slack"):
            require(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0,
                    f"Invalid policy {field}.{key}")
        for key in ("max_abs_ceiling", "rmse_ceiling"):
            value = row.get(key)
            require(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0), f"Invalid policy {field}.{key}")


def validate_spec(spec):
    require(spec.get("schema_version") == 1, "Unsupported experiment spec")
    require(isinstance(spec.get("name"), str) and spec["name"], "Experiment name is required")
    require(spec.get("scope") in ("integration_test", "full_model_single_step"), "Declare experiment scope")
    parameters = spec.get("parameters")
    require(isinstance(parameters, dict) and parameters, "Declare the complete canonical trainable parameter catalog")
    for name, info in parameters.items():
        require(isinstance(name, str) and name and isinstance(info, dict), "Invalid parameter catalog")
        shape = info.get("shape")
        require(isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape), "Invalid parameter shape")
        require(isinstance(info.get("group", name), str), "Invalid parameter group")
    require(set(spec.get("configurations", {})) == {"reference", "candidate"}, "Specify reference and candidate configurations; repeat is derived")
    for config in spec["configurations"].values():
        require(type(config.get("world_size")) is int and config["world_size"] > 0, "Invalid world size")
        require(config.get("device") in ("cpu", "cuda"), "Device must be cpu or cuda")
    require(isinstance(spec.get("command"), list) and spec["command"] and all(isinstance(x, str) and x for x in spec["command"]), "Command must be an argv list, never a shell string")
    require("{plan}" in spec["command"] and "{stage}" in spec["command"], "Command must accept {plan} and {stage}")
    for arg in spec["command"]:
        for key in re.findall(r"\{([^{}]+)\}", arg):
            require(key in ("plan", "stage", "world_size", "python", "output"), f"Unknown command placeholder: {key}")
    contract = spec.get("contract", {})
    require(contract.get("gradient_point") == "after_reduction_before_clipping", "Declare reduced, unscaled, pre-clipping gradients")
    require(contract.get("update_point") == "optimizer_parameter_after_minus_before", "Use actual optimizer parameter deltas")
    require(contract.get("optimizer_steps") == 1 and contract.get("optimizer_initial_state") == "fresh", "This protocol observes one step from fresh optimizer state")
    require(isinstance(contract.get("optimizer"), dict) and contract["optimizer"].get("name"), "Declare optimizer and hyperparameters")
    require(isinstance(contract.get("objective"), dict) and contract["objective"].get("normalization"), "Declare objective, sample contributions and normalization")
    require(type(contract.get("seed")) is int, "Declare initialization and input seed")
    from .precision import validate_precision
    validate_precision(spec)
    observations = spec.get("observables", {})
    require(isinstance(observations, dict) and "loss" in observations, "Declare at least a scalar loss observable")
    require(observations["loss"] == [], "Loss must be a scalar")
    for shape in observations.values():
        require(isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape), "Invalid observable shape")
    checks = spec.get("required_checks", [])
    require(isinstance(checks, list) and len(set(checks)) == len(checks) and all(isinstance(k, str) for k in checks), "Invalid required checks")
    artifacts = spec.get("artifacts", {})
    require(isinstance(artifacts, dict) and "code" in artifacts and "input" in artifacts, "Freeze code and input artifacts")
    require(spec["scope"] != "full_model_single_step" or "weights" in artifacts, "Full-model runs must freeze weight files")
    env = spec.get("environment", {})
    require(isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), "Environment values must be strings")
    require(not set(env).intersection({"PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "PYTHONOPTIMIZE"}),
            "Python launcher variables are managed by the runner; use an explicit adapter script path")
    require(env.get("NVIDIA_DISABLE_REQUIRE", "").lower() not in ("true", "1"), "Do not disable NVIDIA requirements")
    validate_policy(spec.get("policy"))
    from .acceptance import validate_acceptance
    from .parallel import validate_parallel_spec
    validate_acceptance(spec)
    validate_parallel_spec(spec)


def inventory(path):
    path = Path(path).resolve()
    require(path.exists(), f"Missing artifact: {path}")
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    require(files, f"Empty artifact: {path}")
    root = path.parent if path.is_file() else path
    entries = {}
    for file in files:
        require(not file.is_symlink() and file.resolve().is_relative_to(root), "Artifact symlinks are unsupported; freeze regular files")
        entries[str(file.relative_to(root))] = {"sha256": digest(file), "bytes": file.stat().st_size}
    return {"path": str(path), "root": str(root), "files": entries, "sha256": object_hash(entries)}


def toolkit_files():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "precision_validator").rglob("*.py"))}


def prepare(spec_path, output, timeout=14400, budget_gib=1024, retain_raw=False):
    spec = read(spec_path)
    validate_spec(spec)
    output = Path(output).resolve()
    require(not output.exists(), "Output exists; prepare a new immutable experiment")
    require(type(timeout) is int and timeout > 0 and math.isfinite(budget_gib) and budget_gib > 0, "Invalid timeout or disk budget")
    artifacts = {name: inventory(path) for name, path in spec["artifacts"].items()}
    for row in artifacts.values():
        require(not output.is_relative_to(Path(row["path"])), "Output cannot be inside an immutable input artifact")
    count = sum(math.prod(p["shape"]) for p in spec["parameters"].values())
    estimated = count * 4 * len(FIELDS) * len(STAGES)
    peak = estimated * 1.5  # Reserve a pre-step optimizer snapshot while writing actual deltas.
    require(budget_gib * 2**30 >= peak * 1.1, "Raw-data budget too small for complete gradients, updates and temporary snapshots")
    configurations = {"reference": spec["configurations"]["reference"], "repeat": copy.deepcopy(spec["configurations"]["reference"]),
                      "candidate": spec["configurations"]["candidate"]}
    plan = {"schema_version": 2, "adapter": "tensor-protocol-v1", "created_utc": now(), "name": spec["name"],
        "scope": spec["scope"], "parameters": spec["parameters"], "trainable_elements": count,
        "configurations": configurations, "command": spec["command"], "contract": spec["contract"],
        "observables": spec["observables"], "required_checks": spec.get("required_checks", []),
        "artifacts": artifacts, "environment": spec.get("environment", {}), "policy": spec.get("policy"),
        "output": str(output), "toolkit_files": toolkit_files(), "stage_timeout_seconds": timeout,
        "disk_budget_bytes": int(budget_gib * 2**30), "estimated_raw_bytes": estimated,
        "estimated_peak_raw_bytes": int(peak), "cleanup_on_pass": not retain_raw}
    if "acceptance" in spec:
        plan["acceptance"] = spec["acceptance"]
    if "parallel_adapter" in spec:
        plan["parallel_adapter"] = spec["parallel_adapter"]
        plan["parallel_validation"] = {
            "changed_axes": [key for key in ("tp", "pp", "cp", "ep", "etp", "dp")
                             if configurations["reference"]["parallelism"][key] != configurations["candidate"]["parallelism"][key]],
            "scope": "only_the_declared_reference_and_candidate_topologies",
            "reference_repeat_required": True,
        }
    plan["plan_sha256"] = object_hash(plan)
    output.mkdir(parents=True)
    save(output / "plan.json", plan)
    save(output / "spec.json", spec)
    return {"plan": str(output / "plan.json"), "adapter": plan["adapter"], "stages": list(STAGES),
            "status": "prepared", "acceptance_enabled": plan["policy"] is not None, "estimated_raw_bytes": estimated}


def verify_inputs(plan):
    verify_files(ROOT, plan["toolkit_files"])
    for artifact in plan["artifacts"].values():
        verify_files(artifact["root"], artifact["files"])
        path = Path(artifact["path"])
        if path.is_dir():
            names = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
            require(names == set(artifact["files"]), "Artifact file set changed")


def input_identity(plan):
    return object_hash({name: artifact["sha256"] for name, artifact in plan["artifacts"].items()})


def identity(plan, stage, rank):
    return {"protocol": "tensor-protocol-v1", "plan_sha256": plan["plan_sha256"], "stage": stage,
            "rank": rank, "configuration": plan["configurations"][stage], "input_identity": input_identity(plan),
            "contract": plan["contract"]}
