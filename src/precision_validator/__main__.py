import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import unittest

from . import __version__
from .common import ASSET_ROOT, STAGES, ValidationError, load_plan, read, save, now
from .profiles import profiles, get_profile
from .adapters import ADAPTERS, adapter_for, dispatch


def parser():
    cli = argparse.ArgumentParser(prog="precisionctl", description="Model-independent gradient and optimizer-update verification")
    cli.add_argument("--version", action="version", version=__version__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("profiles", help="List bundled Qwen model/GPU profiles")
    commands.add_parser("adapters", help="List training/evidence adapters")
    commands.add_parser("env", help="List dependencies without importing Torch, accessing CUDA or changing the environment")
    commands.add_parser("examples", help="List exportable adapter templates and their runtime requirements")
    from .resources import EXAMPLES
    init = commands.add_parser("init", help="Export one adapter template to a new directory")
    init.add_argument("--adapter", choices=sorted(EXAMPLES), required=True)
    init.add_argument("--output", required=True)
    generic = commands.add_parser("prepare-generic", help="Freeze a model-independent experiment specification")
    generic.add_argument("--spec", required=True)
    generic.add_argument("--output", required=True)
    generic.add_argument("--timeout-seconds", type=int, default=14400)
    generic.add_argument("--disk-budget-gib", type=float, default=1024)
    generic.add_argument("--retain-raw", action="store_true")
    trajectory = commands.add_parser("prepare-trajectory", help="Freeze single-process FP32 multi-step Adam and fresh-process resume validation")
    trajectory.add_argument("--spec", required=True)
    trajectory.add_argument("--output", required=True)
    trajectory.add_argument("--timeout-seconds", type=int, default=3600)
    trajectory.add_argument("--disk-budget-gib", type=float, default=16)
    native_trajectory = commands.add_parser("prepare-native-trajectory", help="Freeze native distributed multi-step Adam and process-restart validation")
    native_trajectory.add_argument("--spec", required=True)
    native_trajectory.add_argument("--output", required=True)
    native_trajectory.add_argument("--timeout-seconds", type=int, default=1800)
    native_trajectory.add_argument("--disk-budget-gib", type=float, default=1)
    parallel = commands.add_parser("prepare-parallel", help="Freeze native TP/PP/CP/EP configurations with strict numerical gates")
    parallel.add_argument("--spec", required=True)
    parallel.add_argument("--reference", required=True, help="tp=1,pp=1,cp=1,ep=1,etp=1; dp may be inferred")
    parallel.add_argument("--candidate", required=True)
    parallel.add_argument("--world-size", type=int, required=True)
    parallel.add_argument("--reference-world-size", type=int)
    parallel.add_argument("--output", required=True)
    parallel.add_argument("--timeout-seconds", type=int, default=14400)
    parallel.add_argument("--disk-budget-gib", type=float, default=1024)
    parallel.add_argument("--retain-raw", action="store_true")
    smoke = commands.add_parser("gpu-smoke", help="Short eight-GPU infrastructure check; not full-model precision acceptance")
    smoke.add_argument("--profile", choices=sorted(profiles()), required=True)
    smoke.add_argument("--output", required=True, help="New directory for small smoke receipts")
    doctor = commands.add_parser("doctor", help="Check pinned runtime, driver access, GPU type and CUDA execution")
    doctor.add_argument("--profile", choices=sorted(profiles()))
    doctor.add_argument("--megatron", default=os.environ.get("MEGATRON_PATH", "/root/Megatron-LM"))
    doctor.add_argument("--cpu-only", action="store_true", help="Check image dependencies only; never emits GPU passed")
    prepare = commands.add_parser("prepare", help="Freeze source, weights, sample, policy and three independent stages")
    prepare.add_argument("--profile", choices=sorted(profiles()), required=True)
    prepare.add_argument("--source", required=True, help="Clean Slime checkout containing slime/ and slime_plugins/")
    prepare.add_argument("--model", required=True, help="HF safetensors directory")
    prepare.add_argument("--input", required=True, help="JSON with samples[{token_ids,loss_mask,metadata}]")
    prepare.add_argument("--sample-index", type=int)
    prepare.add_argument("--output", required=True, help="New directory at its final mounted path")
    prepare.add_argument("--megatron", default=os.environ.get("MEGATRON_PATH", "/root/Megatron-LM"))
    prepare.add_argument("--policy")
    prepare.add_argument("--disk-budget-gib", type=float, default=1024)
    prepare.add_argument("--timeout-seconds", type=int, default=14400)
    prepare.add_argument("--retain-raw", action="store_true", help="Retain successful raw tensors for code regression")
    run = commands.add_parser("run", help="Run one independent stage through the plan's adapter")
    run.add_argument("--plan", required=True)
    run.add_argument("--stage", choices=sorted({s for a in ADAPTERS.values() for s in a["stages"]}), required=True)
    compare = commands.add_parser("compare", help="Compare all three stages on CPU")
    compare.add_argument("--plan", required=True)
    compare.add_argument("--retain-raw", action="store_true")
    regress = commands.add_parser("regress", help="Compare retained known-good and candidate suites")
    regress.add_argument("--reference-plan", required=True)
    regress.add_argument("--candidate-plan", required=True)
    status = commands.add_parser("status", help="Show stage status; running never means precision passed")
    status.add_argument("--plan", required=True)
    cleanup = commands.add_parser("cleanup", help="Remove only this plan's accepted raw shards")
    cleanup.add_argument("--plan", required=True)
    selftest = commands.add_parser("selftest", help="CPU contract tests; reports missing optional-runtime skips explicitly")
    selftest.add_argument("--require-all", action="store_true", help="Fail if any test skips because a native runtime is missing")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = None
        if args.command == "env":
            from .environment import inspect_environment
            result = inspect_environment()
        elif args.command == "examples":
            from .resources import list_examples
            result = list_examples()
        elif args.command == "init":
            from .resources import export_example
            result = export_example(args.adapter, args.output)
        elif args.command == "adapters":
            result = [{"name": name, "schema": a["schema"], "stages": a["stages"]} for name, a in ADAPTERS.items()]
        elif args.command == "prepare-generic":
            from .protocol import prepare
            result = prepare(args.spec, args.output, args.timeout_seconds, args.disk_budget_gib, args.retain_raw)
        elif args.command == "prepare-parallel":
            from .parallel import prepare_parallel
            result = prepare_parallel(args.spec, args.output, args.reference, args.candidate,
                args.world_size, args.reference_world_size, args.timeout_seconds, args.disk_budget_gib, args.retain_raw)
        elif args.command == "prepare-native-trajectory":
            from .native_trajectory import prepare
            result = prepare(args.spec, args.output, args.timeout_seconds, args.disk_budget_gib)
        elif args.command == "prepare-trajectory":
            from .trajectory import prepare
            result = prepare(args.spec, args.output, args.timeout_seconds, args.disk_budget_gib)
        elif args.command == "profiles":
            result = [{"name": p["name"], "model": p["model"], "device": p["device"],
                       "topologies": p["topologies"], "image_validation_status": p["image_validation_status"]}
                      for p in profiles().values()]
        elif args.command == "doctor":
            from .runtime import doctor
            import os
            profile = get_profile(args.profile) if args.profile else None
            if profile:
                for key, value in profile["environment"].items():
                    os.environ.setdefault(key, value)
            result = doctor(profile, args.megatron, args.cpu_only)
        elif args.command == "gpu-smoke":
            from .gpu_smoke import launch
            result = launch(args.profile, args.output)
        elif args.command == "prepare":
            from .planning import prepare
            result = prepare(args)
        elif args.command == "run":
            from .api import run_stage
            result = run_stage(args.plan, args.stage)
        elif args.command in ("compare", "regress", "cleanup"):
            plan_path = args.candidate_plan if args.command == "regress" else args.plan
            plan = load_plan(plan_path)
            with (Path(plan["output"]) / ".compare.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.command == "compare":
                    result = dispatch(plan, "compare", args.plan, args.retain_raw)
                    result = {k: result[k] for k in ("status", "name", "profile", "commit", "gates", "raw_retained") if k in result}
                elif args.command == "regress":
                    report = dispatch(plan, "regress", args.reference_plan, args.candidate_plan)
                    result = {k: report[k] for k in ("status", "name", "profile", "commit", "reference_commit", "gates") if k in report}
                else:
                    report = read(Path(plan["output"]) / "acceptance.json")
                    removed = dispatch(plan, "cleanup", plan, report)
                    report["raw_retained"] = False
                    save(Path(plan["output"]) / "acceptance.json", report)
                    result = {"status": "cleaned", "bytes_removed": removed}
        elif args.command == "status":
            from .api import status
            result = status(args.plan)
        elif args.command == "selftest":
            suite = unittest.defaultTestLoader.discover(str(ASSET_ROOT / "tests"), pattern="test_*.py")
            tests = unittest.TextTestRunner(verbosity=2).run(suite)
            ok = tests.wasSuccessful() and (not args.require_all or not tests.skipped)
            print(json.dumps({"status": "selftest_failed" if not ok else "selftest_partial" if tests.skipped else "selftest_passed",
                              "tests_run": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors),
                              "skipped": len(tests.skipped), "gpu_acceptance": False}))
            return 0 if ok else 1
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 2 if isinstance(result, dict) and result.get("status", "").endswith(("failure", "oom")) else 0
    except (RuntimeError, ImportError, OSError, KeyError, ValueError) as error:
        result = {"status": "incomplete_evidence" if args.command in ("compare", "regress", "cleanup") else "configuration_or_environment_failure",
                  "error": str(error), "utc": now()}
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
