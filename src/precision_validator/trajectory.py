"""Separate, strict multi-step protocol; the original one-step contract stays intact."""
import copy
from fnmatch import fnmatchcase
import math
from pathlib import Path

from .common import read, save, require, object_hash, now
from .protocol import validate_spec as validate_single_step, inventory, toolkit_files
from .acceptance import group_for, validate_limit

STAGES = ("reference", "repeat", "candidate", "resume")
STATE_FIELDS = ("parameters", "exp_avg", "exp_avg_sq")
FIELDS = ("gradients", "updates", *STATE_FIELDS)
PRECISION = {k: "float32" for k in ("parameters", "gradients", "optimizer_parameters", "optimizer_state")}


def validate_spec(spec):
    require(spec.get("scope") == "integration_test", "Trajectory v1 currently supports integration tests only")
    contract = spec.get("contract", {})
    count = contract.get("optimizer_steps")
    require(type(count) is int and count >= 2, "Trajectory requires at least two optimizer steps")
    require(contract.get("optimizer_initial_state") == "fresh", "The complete trajectory must start fresh")
    require(contract.get("precision") == PRECISION, "Trajectory v1 requires uniform FP32")
    require(contract.get("optimizer", {}).get("name") in ("Adam", "AdamW"), "Trajectory v1 supports native Adam/AdamW")
    require(contract["optimizer"].get("options", {}).get("amsgrad", False) is False, "AMSGrad state is not supported")
    trajectory = spec.get("trajectory", {})
    require(set(trajectory) == {"steps", "resume_after_step", "scheduler", "state_acceptance"}, "Invalid trajectory fields")
    boundary = trajectory["resume_after_step"]
    require(type(boundary) is int and 0 < boundary < count, "Resume boundary must be inside the trajectory")
    steps = trajectory["steps"]
    require(isinstance(steps, list) and len(steps) == count, "Input schedule must cover every step")
    for step in steps:
        require(set(step) == {"input_artifact", "objective", "observables"}, "Declare each step's frozen input, objective and observables")
        require(step["input_artifact"] in spec["artifacts"] and step["input_artifact"] != "code", "Missing step input artifact")
        require(isinstance(step["objective"], dict) and step["objective"].get("normalization"), "Missing step objective")
        require(set(step["observables"]) == set(spec["observables"]), "Step observable names must match the declared gate catalog")
    scheduler = trajectory["scheduler"]
    require(isinstance(scheduler, dict) and set(scheduler) == {"name", "options"} and
            scheduler["name"] in ("StepLR", "ExponentialLR", "CosineAnnealingLR") and isinstance(scheduler["options"], dict),
            "Declare a supported native scheduler")
    require(spec.get("acceptance") is not None and spec.get("policy") is not None, "Trajectory acceptance requires strict local and forward gates")
    for cfg in spec["configurations"].values():
        require(cfg.get("world_size") == 1 and "parallelism" not in cfg, "Trajectory v1 is single-process; distributed trajectories require a native adapter")
    require("parallel_adapter" not in spec, "Trajectory v1 does not certify distributed training")
    for step in steps:
        base = copy.deepcopy(spec)
        base["contract"]["optimizer_steps"] = 1
        base["contract"]["objective"] = step["objective"]
        base["observables"] = step["observables"]
        validate_single_step(base)
    rules = trajectory["state_acceptance"]
    require(isinstance(rules, dict) and set(rules) == {"parameters", "groups"}, "Declare parameter and group state gates")
    for category, names in (("parameters", set(spec["parameters"])),
                            ("groups", {group_for(k, v) for k, v in spec["parameters"].items()})):
        covered = set()
        require(isinstance(rules[category], list) and rules[category], "Missing state gate rules")
        for rule in rules[category]:
            require(isinstance(rule, dict) and set(rule) == {"pattern", *STATE_FIELDS} and
                    isinstance(rule["pattern"], str) and rule["pattern"], "Invalid state rule")
            matched = {name for name in names if fnmatchcase(name, rule["pattern"])}
            require(matched, "State rule matches no parameter/group")
            covered |= matched
            for field in STATE_FIELDS:
                validate_limit(rule[field])
        require(covered == names, "Incomplete optimizer/parameter state gate coverage")


def prepare(spec_path, output, timeout=3600, budget_gib=16):
    spec = read(spec_path)
    validate_spec(spec)
    output = Path(output).resolve()
    require(not output.exists(), "Output exists; prepare a new immutable experiment")
    require(type(timeout) is int and timeout > 0 and math.isfinite(budget_gib) and budget_gib > 0, "Invalid resource budget")
    artifacts = {name: inventory(path) for name, path in spec["artifacts"].items()}
    require(all(not output.is_relative_to(Path(a["path"])) for a in artifacts.values()), "Output cannot be inside an input artifact")
    n = sum(math.prod(v["shape"]) for v in spec["parameters"].values())
    estimated = n * 4 * len(FIELDS) * spec["contract"]["optimizer_steps"] * len(STAGES)
    peak = estimated + n * 4 * 5 * len(STAGES) + 16 * 2**20
    require(budget_gib * 2**30 >= peak * 1.1, "Insufficient trajectory raw/checkpoint budget")
    configs = copy.deepcopy(spec["configurations"])
    configs["repeat"] = copy.deepcopy(configs["reference"])
    configs["resume"] = copy.deepcopy(configs["candidate"])
    plan = {"schema_version": 3, "adapter": "torch-trajectory-v1", "name": spec["name"], "created_utc": now(),
        "scope": spec["scope"], "parameters": spec["parameters"], "trainable_elements": n,
        "configurations": configs, "command": spec["command"], "contract": spec["contract"],
        "observables": spec["observables"], "acceptance": spec["acceptance"], "policy": spec["policy"],
        "trajectory": spec["trajectory"], "required_checks": spec.get("required_checks", []),
        "artifacts": artifacts, "environment": spec.get("environment", {}), "toolkit_files": toolkit_files(),
        "output": str(output), "stage_timeout_seconds": timeout, "disk_budget_bytes": int(budget_gib * 2**30),
        "estimated_raw_bytes": estimated, "estimated_peak_raw_bytes": peak, "cleanup_on_pass": False}
    plan["plan_sha256"] = object_hash(plan)
    output.mkdir(parents=True)
    save(output / "spec.json", spec)
    save(output / "plan.json", plan)
    return {"status": "prepared", "plan": str(output / "plan.json"), "stages": list(STAGES), "estimated_raw_bytes": estimated}


def identity(plan, stage):
    return {"protocol": "torch-trajectory-v1", "plan_sha256": plan["plan_sha256"], "stage": stage, "rank": 0,
            "configuration": plan["configurations"][stage], "contract": plan["contract"]}


def stored_bytes(folder):
    total, inodes = 0, set()
    for path in Path(folder).rglob("*"):
        if not path.name.endswith((".f32", ".pt", ".partial")):
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            final = path.with_suffix("") if path.suffix == ".partial" else path
            try:
                stat = final.stat()
            except FileNotFoundError:
                continue  # Observer may have removed its temporary before-snapshot.
        key = (stat.st_dev, stat.st_ino)
        if key not in inodes:
            total += stat.st_size
            inodes.add(key)
    return total
