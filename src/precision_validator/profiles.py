"""Versioned profiles and topology-independent single-sample semantics."""
import math
from pathlib import Path
from .common import ROOT, ASSET_ROOT, read, require, object_hash


def profiles():
    return {p.stem: read(p) for p in sorted((ASSET_ROOT / "profiles").glob("*.json"))}


def get_profile(name):
    result = profiles()
    require(name in result, f"Unknown profile {name}; choose from {', '.join(result)}")
    profile = result[name]
    require(profile["name"] == name and profile["world_size"] == 8, "Invalid profile")
    for layout in profile["topologies"].values():
        require(layout["tp"] == layout["etp"] == 1, "V1 supports TP1 / ETP1")
        require(layout["tp"] * layout["pp"] * layout["cp"] * layout["dp"] == 8, "Invalid topology")
        require((layout["dp"] * layout["cp"]) % layout["ep"] == 0, "Invalid EP group")
    return profile


def topology(profile, stage):
    return profile["topologies"]["target" if stage == "cp_target" else "reference"]


def contribution_count(profile):
    # Same unique input and total contributions, even if DP differs.
    return math.lcm(*(t["dp"] for t in profile["topologies"].values()))


def select_sample(inputs, sequence_length, index=None):
    samples = inputs.get("samples")
    require(isinstance(samples, list) and samples, "Expected a nonempty samples array")
    if index is None:
        index = max(range(len(samples)), key=lambda i: (len(samples[i]["token_ids"]), -i))
    require(type(index) is int and 0 <= index < len(samples), "Invalid sample index")
    sample = samples[index]
    tokens = sample.get("token_ids")
    mask = sample.get("loss_mask")
    require(isinstance(tokens, list) and len(tokens) == sequence_length,
            f"Expected one real {sequence_length}-token sample; padding/truncation is not allowed")
    require(all(type(x) is int and x >= 0 for x in tokens), "Token IDs must be nonnegative integers")
    require(isinstance(mask, list) and len(mask) == len(tokens), "An explicit full-length loss_mask is required")
    require(all(type(x) is int and x in (0, 1) for x in mask) and sum(mask) > 0,
            "loss_mask must contain binary values and supervised tokens")
    require(mask[0] == 0, "First token cannot be a causal-LM target")
    require(bool(sample.get("metadata")), "Real-input provenance metadata is required")
    selected = {"token_ids": tokens, "loss_mask": mask, "metadata": sample["metadata"]}
    return selected, {"source_sample_count": len(samples), "source_sample_index": index,
        "sample_sha256": object_hash(selected), "unique_samples": 1, "unique_tokens": len(tokens),
        "supervised_tokens": sum(mask), "response_length": len(mask) - mask.index(1)}


def replace(argv, flag, values):
    if flag in argv:
        index = argv.index(flag)
        stop = index + 1
        while stop < len(argv) and not argv[stop].startswith("--"):
            stop += 1
        argv[index:stop] = [flag, *map(str, values)]
    else:
        argv += [flag, *map(str, values)]


def launch_args(plan, stage):
    profile = plan["profile"]
    layout = topology(profile, stage)
    argv = [str(value).replace("{MODEL}", plan["model_path"]).replace("{INPUTS}", plan["input_path"])
            for value in profile["argv"]]
    for flag, field in (("--tensor-model-parallel-size", "tp"), ("--pipeline-model-parallel-size", "pp"),
                        ("--context-parallel-size", "cp"), ("--expert-model-parallel-size", "ep"),
                        ("--expert-tensor-parallel-size", "etp")):
        replace(argv, flag, [layout[field]])
    for flag in ("--global-batch-size", "--rollout-batch-size"):
        replace(argv, flag, [plan["sample_copies_per_step"]])
    replace(argv, "--lr-warmup-init", [plan["learning_rate"]])
    replace(argv, "--lr", [plan["learning_rate"]])
    replace(argv, "--seed", [plan["seed"]])
    return argv
