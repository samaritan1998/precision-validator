"""Run an adapter's argv and seal its evidence only after successful exit."""
import fcntl
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
from .common import ROOT, load_plan, read, save, require, digest, now
from .protocol import STAGES, verify_inputs
from .runner import stop_process, raw_data_bytes, classify_failure
from .adapters import adapter_for


def run_stage(plan_path, stage):
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path)
    stages = adapter_for(plan)["stages"]
    trajectory = plan["adapter"] in ("torch-trajectory-v1", "native-trajectory-v1")
    require(stage in stages and plan_path.parent == Path(plan["output"]), "Invalid stage or moved plan")
    folder = plan_path.parent / stage
    folder.mkdir(exist_ok=True)
    lock = (folder / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not [p for p in folder.iterdir() if p.name != ".lock"], "Stage exists; use a new plan for retry")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    started, process, handlers = time.monotonic(), None, {}
    mapping = {"python": sys.executable, "plan": str(plan_path), "stage": stage,
               "world_size": str(plan["configurations"][stage]["world_size"]), "output": str(folder)}
    command = [arg.format_map(mapping) for arg in plan["command"]]
    def stored_bytes():
        if trajectory:
            from .trajectory import stored_bytes
            return stored_bytes(folder)
        return raw_data_bytes(folder)
    try:
        save(folder / "status.json", {"status": "preflight", "utc": now(), "stage": stage})
        verify_inputs(plan)
        require(shutil.disk_usage(folder).free > plan["estimated_peak_raw_bytes"] / len(stages) * 1.1, "Insufficient disk for stage and temporary snapshots")
        env = dict(os.environ)
        env.update(plan["environment"])
        require(env.get("NVIDIA_DISABLE_REQUIRE", "").lower() not in ("true", "1"), "Do not disable NVIDIA compatibility checks")
        env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), plan["environment"].get("PYTHONPATH"))))
        def interrupted(signum, frame):
            raise InterruptedError(f"Received signal {signum}")
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupted)
        with (folder / "train.log").open("x") as log:
            process = subprocess.Popen(command, env=env, cwd=folder, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(folder / "status.json", {"status": "running", "utc": now(), "stage": stage, "command": command})
            while process.poll() is None:
                require(time.monotonic() - started < plan["stage_timeout_seconds"], "Stage timeout")
                require(stored_bytes() <= plan["disk_budget_bytes"] // len(stages), "Stage raw-data budget exceeded")
                require(shutil.disk_usage(folder).free > min(2**30, plan["disk_budget_bytes"] // 10), "Insufficient disk reserve")
                time.sleep(0.2)
        save(folder / "process.json", {"returncode": process.returncode, "command": command, "seconds": time.monotonic() - started, "utc": now()})
        require(process.returncode == 0, "Adapter process failed; see train.log")
        require(stored_bytes() <= plan["disk_budget_bytes"] // len(stages), "Stage raw-data budget exceeded")
        require(not list(folder.rglob("*.partial")), "Adapter left unfinished raw snapshots")
        verify_inputs(plan)
        ranks = {f"rank{i}.json": digest(folder / f"rank{i}.json") for i in range(plan["configurations"][stage]["world_size"])}
        require({p.name for p in folder.glob("rank*.json")} == set(ranks), "Unexpected rank receipt")
        final = {"status": "stage_complete", "stage": stage, "plan_sha256": plan["plan_sha256"], "ranks": ranks,
                 "process_sha256": digest(folder / "process.json"), "finished_utc": now()}
        save(folder / "complete.json", final)
        if plan['adapter'] == 'native-trajectory-v1':
            from .native_trajectory_compare import Stage
        elif trajectory:
            from .trajectory_compare import Stage
        else:
            from .protocol_compare import Stage
        Stage(plan, stage)
        save(folder / "status.json", final)
        return final
    except BaseException as error:
        if process is not None:
            stop_process(process)
            save(folder / "process.json", {"returncode": process.returncode, "command": command,
                 "seconds": time.monotonic() - started, "termination_reason": str(error), "utc": now()})
        (folder / "complete.json").unlink(missing_ok=True)
        save(folder / "status.json", {"status": classify_failure(str(error)), "error": str(error), "utc": now()})
        raise
    finally:
        for sig, previous in handlers.items():
            signal.signal(sig, previous)
        lock.close()
