"""Adapter dispatch. Model-specific assumptions stay behind these boundaries."""
from importlib import import_module
from .common import require

ADAPTERS = {
    "native-trajectory-v1": {"schema": 4, "stages": ("reference", "repeat", "candidate", "resume"),
        "run": ("protocol_runner", "run_stage"), "compare": ("native_trajectory_compare", "compare_suite"),
        "cleanup": ("native_trajectory_compare", "cleanup_passed"), "regress": ("native_trajectory_compare", "compare_regression")},
    "slime-qwen-v1": {"schema": 1, "stages": ("cp1", "cp1_repeat", "cp_target"),
        "run": ("runner", "run_stage"), "compare": ("compare", "compare_suite"),
        "cleanup": ("compare", "cleanup_passed"), "regress": ("compare", "compare_regression")},
    "tensor-protocol-v1": {"schema": 2, "stages": ("reference", "repeat", "candidate"),
        "run": ("protocol_runner", "run_stage"), "compare": ("protocol_compare", "compare_suite"),
        "cleanup": ("protocol_compare", "cleanup_passed"), "regress": ("protocol_compare", "compare_regression")},
    "torch-trajectory-v1": {"schema": 3, "stages": ("reference", "repeat", "candidate", "resume"),
        "run": ("protocol_runner", "run_stage"), "compare": ("trajectory_compare", "compare_suite"),
        "cleanup": ("trajectory_compare", "cleanup_passed"), "regress": ("trajectory_compare", "compare_regression")},
}


def adapter_for(plan):
    adapter = ADAPTERS.get(plan.get("adapter"))
    require(adapter is not None and plan.get("schema_version") == adapter["schema"], "Unsupported plan adapter/schema")
    return adapter


def dispatch(plan, action, *args):
    module, function = adapter_for(plan)[action]
    return getattr(import_module("." + module, __package__), function)(*args)
