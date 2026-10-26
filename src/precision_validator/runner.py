"""Launch independently schedulable stages; no silent retries or auto-patching."""
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
from .common import ROOT, STAGES, now, save, read, digest, require, load_plan, verify_files, git


def environment(plan, stage):
    env = dict(os.environ)
    env.update(plan["profile"]["environment"])
    env.update({"SLIME_ROOT": plan["source"], "MEGATRON_PATH": plan["megatron_path"],
        # ROOT is site-packages after pip installation. Explicit candidate
        # checkouts must precede any older Slime/Megatron installed there.
        "PYTHONPATH": os.pathsep.join((plan["source"], plan["megatron_path"], str(ROOT))),
        "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled", "NCCL_DEBUG": "WARN", "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_MODULE_LOADING": "EAGER", "CUDA_MODULE_DATA_LOADING": "EAGER",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "65536", "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
        "SLIME_FLA_CONFIG_LOG_DIR": str(Path(plan["output"]) / stage / "native"),
        "FULL_JOB_NAME": "precision-" + plan["plan_sha256"][:12] + "-" + stage})
    for name in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
        if name in env:
            env[name] = env[name].lstrip("=")
    for name in ("PYTHONOPTIMIZE", "ENABLE_ROUTING_REPLAY"):
        require(not env.get(name), f"Unsupported inherited control: {name}")
    command = [sys.executable, "-c", "from torch_memory_saver.utils import get_binary_path_from_package; print(get_binary_path_from_package('torch_memory_saver_hook_mode_preload'))"]
    library = subprocess.check_output(command, env=env, text=True, timeout=120).strip()
    require(Path(library).is_file(), "torch-memory-saver preload library not found")
    env["LD_PRELOAD"] = ":".join(dict.fromkeys([library, *filter(None, env.get("LD_PRELOAD", "").split(":"))]))
    return env


def classify_failure(log):
    text = log.lower()
    if "out of memory" in text or "outofmemoryerror" in text:
        return "oom"
    if "disk quota" in text or "no space left" in text or "raw-data budget" in text:
        return "storage_failure"
    if "nccl" in text and any(word in text for word in ("timeout", "timed out", "error")):
        return "communication_failure"
    if "nonfinite" in text or "found inf" in text:
        return "numerical_failure"
    return "execution_failure"


def stop_process(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def raw_data_bytes(folder):
    """Count live shards while writers atomically publish .partial files.

    Directory enumeration and stat are separate operations. A writer can rename
    a listed partial between them, or both names can appear in one enumeration.
    Follow that rename and count each inode once. A missing file is checked by
    the final artifact verifier; other I/O errors must still abort the run.
    """
    sizes = {}
    for path in folder.glob("*.f32*"):
        try:
            info = path.stat()
        except FileNotFoundError:
            if not path.name.endswith(".f32.partial"):
                continue
            try:
                info = path.with_suffix("").stat()
            except FileNotFoundError:
                continue
        identity = (info.st_dev, info.st_ino)
        sizes[identity] = max(sizes.get(identity, 0), info.st_size)
    return sum(sizes.values())


def check_raw_budget(folder, limit):
    size = raw_data_bytes(folder)
    require(size <= limit, "Stage raw-data budget exceeded")
    return size


def run_stage(plan_path, stage):
    require(stage in STAGES, "Unknown stage")
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path)
    require(plan_path.parent == Path(plan["output"]), "Plan moved; prepare it at its final mounted path")
    folder = plan_path.parent / stage
    folder.mkdir(exist_ok=True)
    lock = (folder / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not [p for p in folder.iterdir() if p.name != ".lock"],
            "Stage already has evidence; use a new plan for retries")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    started = time.monotonic()
    process = None
    old_signals = {}
    try:
        save(folder / "status.json", {"status": "preflight", "stage": stage, "utc": now()})
        verify_files(ROOT, plan["toolkit_files"])
        verify_files(plan["source"], plan["source_inventory"]["files"])
        require(git(plan["source"], "rev-parse", "HEAD") == plan["source_inventory"]["commit"], "Candidate commit changed")
        require(not git(plan["source"], "status", "--porcelain", "--", "."), "Source worktree changed")
        require(digest(plan["input_path"]) == plan["input_sha256"], "Input changed")
        print("PREFLIGHT: verifying immutable model files", flush=True)
        verify_files(plan["model_path"], plan["model_inventory"]["files"])
        needed = plan["profile"]["trainable_elements"] * 8
        require(shutil.disk_usage(folder).free > needed * 1.1, "Insufficient free disk for this stage")
        model_stat = {name: (Path(plan["model_path"]) / name).stat().st_mtime_ns for name in plan["model_inventory"]["files"]}
        env = environment(plan, stage)
        check = subprocess.run([sys.executable, "-m", "precision_validator", "doctor", "--profile", plan["profile"]["name"],
                "--megatron", plan["megatron_path"]], env=env, capture_output=True, text=True, timeout=180)
        (folder / "preflight.log").write_text(check.stdout + check.stderr)
        require(check.returncode == 0, "GPU/runtime preflight failed; see preflight.log")
        proof = json.loads(check.stdout.splitlines()[-1])
        save(folder / "environment.json", proof)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8", "--max-restarts=0",
                   "-m", "precision_validator.slime_adapter", "--plan", str(plan_path), "--stage", stage]
        save(folder / "status.json", {"status": "running", "stage": stage, "utc": now(), "command": command})
        def interrupted(signum, frame):
            raise InterruptedError(f"Received signal {signum}")
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_signals[sig] = signal.signal(sig, interrupted)
        with (folder / "train.log").open("x") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                       cwd=plan["source"], start_new_session=True)
            while process.poll() is None:
                require(time.monotonic() - started < plan["stage_timeout_seconds"], "Stage timeout")
                check_raw_budget(folder, plan["disk_budget_bytes"] // 3)
                require(shutil.disk_usage(folder).free > 2**30, "Disk quota reserve exhausted")
                time.sleep(5)
        receipt = {"returncode": process.returncode, "seconds": time.monotonic() - started,
                   "command": command, "finished_utc": now()}
        save(folder / "process.json", receipt)
        if process.returncode:
            with (folder / "train.log").open("rb") as stream:
                stream.seek(max(0, (folder / "train.log").stat().st_size - 200000))
                failure = classify_failure(stream.read().decode(errors="replace"))
            save(folder / "status.json", {"status": failure, "utc": now(), "returncode": process.returncode})
            return {"status": failure, "folder": str(folder)}
        check_raw_budget(folder, plan["disk_budget_bytes"] // 3)
        verify_files(plan["source"], plan["source_inventory"]["files"])
        require(model_stat == {name: (Path(plan["model_path"]) / name).stat().st_mtime_ns
                              for name in plan["model_inventory"]["files"]}, "Model changed during execution")
        ranks = {}
        for rank in range(8):
            path = folder / f"rank{rank}.json"
            row = read(path)
            require(row["rank"] == rank and row["stage"] == stage and row["plan_sha256"] == plan["plan_sha256"], "Rank identity mismatch")
            require(row["distributed_groups_destroyed"] and row["optimizer_step_executed"], "Incomplete rank receipt")
            ranks[path.name] = digest(path)
        final = {"status": "stage_complete", "comparison_pending": True, "stage": stage,
            "plan_sha256": plan["plan_sha256"], "finished_utc": now(), "ranks": ranks,
            "environment_sha256": digest(folder / "environment.json"), "process_sha256": digest(folder / "process.json")}
        save(folder / "complete.json", final)
        save(folder / "status.json", final)
        return final
    except BaseException as error:
        if process is not None:
            stop_process(process)
            save(folder / "process.json", {"returncode": process.returncode, "seconds": time.monotonic() - started,
                 "command": command, "finished_utc": now(), "termination_reason": str(error)})
        status = "environment_failure" if process is None else ("interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else classify_failure(str(error)))
        save(folder / "status.json", {"status": status, "error": str(error), "utc": now()})
        raise
    finally:
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
        lock.close()
