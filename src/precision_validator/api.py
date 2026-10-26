"""Small Python API using the same contracts and comparison lock as the CLI."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

from .adapters import adapter_for, dispatch
from .common import load_plan, read


def prepare_generic(spec, output, *, timeout_seconds=14400, disk_budget_gib=1024, retain_raw=False):
    from .protocol import prepare
    return prepare(spec, output, timeout_seconds, disk_budget_gib, retain_raw)


def run_stage(plan_path, stage):
    plan = load_plan(plan_path)
    return dispatch(plan, "run", plan_path, stage)


@contextmanager
def _comparison_lock(plan):
    with (Path(plan["output"]) / ".compare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def compare(plan_path, *, retain_raw=False):
    """Return the full report; configuration/evidence errors raise ValidationError."""
    plan = load_plan(plan_path)
    with _comparison_lock(plan):
        return dispatch(plan, "compare", plan_path, retain_raw)


def regress(reference_plan, candidate_plan):
    plan = load_plan(candidate_plan)
    with _comparison_lock(plan):
        return dispatch(plan, "regress", reference_plan, candidate_plan)


def status(plan_path):
    plan = load_plan(plan_path)
    root = Path(plan["output"])
    return {
        "name": plan.get("name", plan.get("profile", {}).get("name")),
        "stages": {stage: read(root / stage / "status.json") if (root / stage / "status.json").is_file()
                   else {"status": "not_started"} for stage in adapter_for(plan)["stages"]},
        "acceptance": read(root / "acceptance.json")["status"] if (root / "acceptance.json").is_file() else "not_compared",
    }
