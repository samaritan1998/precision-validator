"""Explicit dense/expert topology contracts; adapters own native execution."""
import copy
import itertools
from pathlib import Path

from .common import read, require, save

AXES = ("tp", "pp", "cp", "ep", "etp", "dp")
CONVENTION = "megatron_dense_expert_v1"


def validate_layout(layout, world):
    require(isinstance(layout, dict) and set(layout) == set(AXES), "Declare TP/PP/CP/EP/ETP/DP explicitly")
    require(all(type(layout[k]) is int and layout[k] > 0 for k in AXES), "Parallel degrees must be positive integers")
    require(world == layout["tp"] * layout["pp"] * layout["cp"] * layout["dp"], "Dense world size must equal TP*PP*CP*DP")
    expert_partition = layout["etp"] * layout["ep"] * layout["pp"]
    require(world % expert_partition == 0, "Expert world size must be divisible by ETP*EP*PP")
    return {**layout, "edp": world // expert_partition}


def parse_layout(value, world):
    result = {}
    for item in value.split(","):
        key, sep, raw = item.strip().partition("=")
        require(sep and key in AXES and key not in result, "Use unique tp,pp,cp,ep,etp,dp assignments")
        require(raw.isdecimal(), "Parallel degrees must be integers")
        result[key] = int(raw)
    require(set(result) >= set(AXES) - {"dp"}, "Specify tp,pp,cp,ep,etp; only dp may be inferred")
    partition = result["tp"] * result["pp"] * result["cp"]
    require(partition > 0 and world % partition == 0, "Cannot infer integral DP")
    result.setdefault("dp", world // partition)
    validate_layout(result, world)
    return result


def validate_parallel_spec(spec):
    configs = spec["configurations"]
    present = ["parallelism" in cfg for cfg in configs.values()]
    if not any(present):
        require("parallel_adapter" not in spec, "Parallel adapter requires both topology configurations")
        return
    require(all(present), "Both configurations must declare parallelism")
    capability = spec.get("parallel_adapter", {})
    require(set(capability) == {"convention", "framework", "supported_axes"}
            and capability["convention"] == CONVENTION
            and isinstance(capability["framework"], str) and capability["framework"],
            "Declare the external parallel_adapter and topology convention")
    axes = capability["supported_axes"]
    require(isinstance(axes, list) and len(set(axes)) == len(axes) and set(axes) <= set(AXES), "Invalid adapter supported_axes")
    for cfg in configs.values():
        validate_layout(cfg["parallelism"], cfg["world_size"])
        require(all(value == 1 or axis in axes for axis, value in cfg["parallelism"].items()),
                "Topology requires an axis unsupported by the declared native adapter")
    require(spec.get("policy") is not None and spec.get("acceptance") is not None,
            "Parallel acceptance requires frozen global, parameter, group and observation gates")
    objective = spec["contract"]["objective"]
    require(type(objective.get("global_samples")) is int and objective["global_samples"] > 0,
            "Declare invariant global_samples; topology changes must not alter the batch")
    require(type(objective.get("supervised_units")) is int and objective["supervised_units"] > 0,
            "Declare invariant supervised_units and the loss normalization")


def validate_rank_groups(config, rows):
    """Validate native process-group membership, not just copied degree flags."""
    if "parallelism" not in config:
        return
    world = config["world_size"]
    sizes = validate_layout(config["parallelism"], world)
    ranks = {row["rank"]: row for row in rows}
    require(set(ranks) == set(range(world)), "Native topology rank coverage differs")
    for rank, row in ranks.items():
        receipt = row.get("native_parallelism", {})
        require(receipt.get("convention") == CONVENTION and receipt.get("degrees") == sizes,
                "Native parallel degrees differ from the plan")
        groups = receipt.get("groups", {})
        require(set(groups) == set(sizes), "Missing native parallel process groups")
        for axis, size in sizes.items():
            group = groups[axis]
            require(isinstance(group, list) and len(group) == size and rank in group
                    and all(type(x) is int and x in ranks for x in group) and len(set(group)) == size,
                    f"Invalid native {axis} process group")
    for axis, size in sizes.items():
        groups = {tuple(sorted(row["native_parallelism"]["groups"][axis])) for row in rows}
        require(len(groups) == world // size, f"Native {axis} groups do not partition all ranks")
        for group in groups:
            require(all(set(ranks[rank]["native_parallelism"]["groups"][axis]) == set(group) for rank in group),
                    f"Native {axis} membership is not reciprocal")
    # Dense and expert grids use different DP degrees. EP is not an additional
    # multiplicative dimension of the dense TP*PP*CP*DP grid.
    for axes in (("tp", "pp", "cp", "dp"), ("etp", "pp", "ep", "edp")):
        for row in rows:
            groups = row["native_parallelism"]["groups"]
            for left, right in itertools.combinations(axes, 2):
                require(set(groups[left]) & set(groups[right]) == {row["rank"]},
                        "Native topology axes overlap beyond their owner rank")


def prepare_parallel(spec_path, output, reference, candidate, world_size,
                     reference_world_size=None, timeout=14400, budget_gib=1024, retain_raw=False):
    from .protocol import prepare
    require(type(world_size) is int and world_size > 0, "Invalid candidate world size")
    ref_world = reference_world_size if reference_world_size is not None else world_size
    require(type(ref_world) is int and ref_world > 0, "Invalid reference world size")
    spec = copy.deepcopy(read(spec_path))
    for stage, text, world in (("reference", reference, ref_world), ("candidate", candidate, world_size)):
        spec["configurations"][stage].update(world_size=world, parallelism=parse_layout(text, world))
    require(spec["configurations"]["reference"]["parallelism"] != spec["configurations"]["candidate"]["parallelism"],
            "Reference and candidate topologies must differ")
    output = Path(output).resolve()
    # Save the rendered input outside the new immutable output, without replacing
    # a previous experiment or modifying the supplied adapter specification.
    rendered = output.with_name(output.name + ".spec.json")
    require(not output.exists() and not rendered.exists(), "Use a new topology experiment path")
    from .protocol import validate_spec
    validate_spec(spec)
    save(rendered, spec)
    return prepare(rendered, output, timeout, budget_gib, retain_raw)
