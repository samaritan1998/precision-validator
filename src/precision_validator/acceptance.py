"""Frozen per-parameter, group and observable gates, including near-zero data."""
from fnmatch import fnmatchcase
import math

from .common import require

FIELDS = ("gradients", "updates")


def group_for(name, info):
    return info.get("group", name.rsplit(".", 1)[0])


def validate_limit(limit):
    keys = {"relative_l2", "max_abs", "near_zero_rms", "near_zero_max_abs"}
    require(isinstance(limit, dict) and keys <= set(limit) <= keys | {"rmse"}, "Invalid strict gate fields")
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in limit.values()),
            "Strict gates require finite nonnegative bounds")
    require(limit["near_zero_max_abs"] <= limit["max_abs"], "Near-zero absolute bound cannot relax max_abs")


def validate_acceptance(spec):
    gates = spec.get("acceptance")
    if gates is None:
        return
    require(spec.get("policy") is not None, "Strict gates require global gradient/update policy too")
    require(isinstance(gates, dict) and set(gates) == {"parameters", "groups", "observations"}, "Declare all strict gate categories")
    catalogs = {"parameters": set(spec["parameters"]),
                "groups": {group_for(name, info) for name, info in spec["parameters"].items()}}
    for kind, names in catalogs.items():
        rules = gates[kind]
        require(isinstance(rules, list) and rules, f"Missing {kind} acceptance rules")
        covered = set()
        for rule in rules:
            require(isinstance(rule, dict) and set(rule) == {"pattern", *FIELDS}
                    and isinstance(rule["pattern"], str) and rule["pattern"], "Invalid local rule")
            matched = {name for name in names if fnmatchcase(name, rule["pattern"])}
            require(matched, f"Unmatched {kind} pattern: {rule['pattern']}")
            covered |= matched
            for field in FIELDS:
                validate_limit(rule[field])
        require(covered == names, f"Uncovered {kind} acceptance: {sorted(names - covered)}")
    observations = gates["observations"]
    require(isinstance(observations, dict) and set(observations) == set(spec["observables"]),
            "Every declared observation, including loss, requires a frozen gate")
    for limit in observations.values():
        validate_limit(limit)


def evaluate_limit(repeat, candidate, limit):
    reasons, details = [], {}
    for stage, value in (("repeat", repeat), ("candidate", candidate)):
        reference_rms = math.sqrt(value["reference2"] / value["elements"])
        near_zero = reference_rms <= limit["near_zero_rms"]
        max_abs = limit["near_zero_max_abs"] if near_zero else limit["max_abs"]
        if value["max_abs"] > max_abs:
            reasons.append(f"{stage} max_abs {value['max_abs']} exceeds {max_abs}")
        if not near_zero and (value["relative_l2"] is None or value["relative_l2"] > limit["relative_l2"]):
            reasons.append(f"{stage} relative_l2 exceeds {limit['relative_l2']}")
        if "rmse" in limit and value["rmse"] > limit["rmse"]:
            reasons.append(f"{stage} RMSE exceeds {limit['rmse']}")
        details[stage] = {"reference_rms": reference_rms, "mode": "near_zero_absolute" if near_zero else "absolute_and_relative",
                          "max_abs_limit": max_abs}
    return {"passed": not reasons, "relative_limit": limit["relative_l2"], "policy": limit, "details": details, "reasons": reasons}


def strict_gates(comparisons, acceptance):
    if acceptance is None:
        return {}
    gates = {}
    for category in ("parameters", "groups"):
        for field in FIELDS:
            ref = comparisons["repeat"][field][category]
            actual = comparisons["candidate"][field][category]
            require(ref.keys() == actual.keys(), "Local comparison coverage differs")
            for index, rule in enumerate(acceptance[category]):
                for name in sorted(actual):
                    if fnmatchcase(name, rule["pattern"]):
                        gates[f"{category}/{field}/{name}/rule{index}"] = evaluate_limit(ref[name], actual[name], rule[field])
    for name, limit in acceptance["observations"].items():
        key = "observation/" + name
        gates[key] = evaluate_limit(comparisons["repeat"][key], comparisons["candidate"][key], limit)
    return gates
