"""Observe native FP32 Adam trajectories and restore an owned checkpoint in a fresh process."""
import os
from pathlib import Path
import platform
import random
import time

import numpy as np

from .common import load_plan, require, read, save, digest, object_hash
from .sdk import runtime_info, tensor_digest, torch_chunks
from .storage import ShardWriter
from .trajectory import FIELDS, PRECISION, STAGES, identity, stored_bytes


def describe(value):
    import torch
    if isinstance(value, torch.Tensor):
        require(not value.is_floating_point() or torch.isfinite(value).all().item(), "Nonfinite state tensor")
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_digest(value)}
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(v) for v in value]
    require(value is None or type(value) in (bool, str, int, float), "Unsupported checkpoint metadata")
    object_hash(value)  # Reject NaN and infinity in scalar metadata too.
    return value


def rng_state():
    import torch
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state((state["numpy"][0], np.array(state["numpy"][1], dtype=np.uint32), *state["numpy"][2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def state_signature(model, optimizer, scheduler, cursor):
    return {"model": describe(model.state_dict()), "optimizer": describe(optimizer.state_dict()),
            "scheduler": describe(scheduler.state_dict()), "rng": describe(rng_state()), "data_cursor": cursor}


def cursor_for(plan, completed):
    steps = plan["trajectory"]["steps"][:completed]
    return {"next_step": completed + 1, "consumed_inputs": [plan["artifacts"][s["input_artifact"]]["sha256"] for s in steps]}


def restore_checkpoint(plan, model, optimizer, scheduler):
    import torch
    folder = Path(plan["output"]) / "resume"
    record = read(folder / "checkpoint.json")
    checkpoint = folder / "checkpoint.pt"
    require(record["plan_sha256"] == plan["plan_sha256"] and record["stage"] == "resume" and
            record["sha256"] == digest(checkpoint) and record["bytes"] == checkpoint.stat().st_size,
            "Checkpoint identity/checksum differs")
    boundary = plan["trajectory"]["resume_after_step"]
    require(record["cursor"] == cursor_for(plan, boundary), "Checkpoint data cursor differs from frozen schedule")
    require(record["producer_pid"] != os.getpid(), "Resume must run in a fresh worker process")
    # Load only an owned, hash-verified state_dict checkpoint, never arbitrary Python objects.
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(blob["plan_sha256"] == plan["plan_sha256"] and blob["cursor"] == record["cursor"], "Checkpoint payload cursor/plan mismatch")
    model.load_state_dict(blob["model"], strict=True)
    # Scheduler already exists; restore it before optimizer parameter-group learning rates.
    scheduler.load_state_dict(blob["scheduler"])
    optimizer.load_state_dict(blob["optimizer"])
    restore_rng(blob["rng"])
    restored = state_signature(model, optimizer, scheduler, blob["cursor"])
    require(restored == record["state"], "Restored model/optimizer/scheduler/RNG/cursor differs from saved state")
    return record, {"checkpoint_sha256": record["sha256"], "producer_pid": record["producer_pid"],
                    "consumer_pid": os.getpid(), "restored_state": restored}


def capture_torch_trajectory(plan_path, stage, model, optimizer, scheduler, backward, *, resume=False):
    """backward(step, frozen_step) returns observations, input_sha256, objective, checks.

    The callback owns forward/backward and objective normalization only. This helper
    owns actual optimizer/scheduler steps and complete FP32 evidence. In the resume
    stage its first invocation writes the prefix and checkpoint, then returns;
    the adapter must invoke it with resume=True in a new process with new objects.
    """
    import torch
    plan = load_plan(plan_path)
    require(plan["adapter"] == "torch-trajectory-v1" and stage in STAGES, "Invalid trajectory plan/stage")
    require(not resume or stage == "resume", "Only the resume stage loads a checkpoint")
    cfg, contract = plan["configurations"][stage], plan["contract"]
    require(cfg["world_size"] == 1 and contract["precision"] == PRECISION, "Unsupported trajectory execution")
    require(type(optimizer) in (torch.optim.Adam, torch.optim.AdamW) and
            type(optimizer).__name__ == contract["optimizer"]["name"], "Native optimizer differs from contract")
    require(not optimizer.state, "Construct a fresh optimizer before capture or restore")
    sched_spec = plan["trajectory"]["scheduler"]
    require(type(scheduler) is getattr(torch.optim.lr_scheduler, sched_spec["name"]) and scheduler.optimizer is optimizer,
            "Native scheduler differs from contract")
    for key, value in sched_spec["options"].items():
        require(describe(getattr(scheduler, key)) == value, f"Scheduler option differs: {key}")
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    require(list(params) == list(plan["parameters"]), "Ordered parameter catalog differs")
    require(len(optimizer.param_groups) == 1 and [id(p) for p in optimizer.param_groups[0]["params"]] ==
            [id(p) for p in params.values()], "Trajectory v1 requires one ordered, complete optimizer group")
    for key, value in contract["optimizer"].get("options", {}).items():
        require(describe(optimizer.param_groups[0].get(key)) == value, f"Optimizer option differs: {key}")
    require(not optimizer.param_groups[0].get("amsgrad"), "AMSGrad state is not supported")
    require(scheduler.last_epoch == 0, "Scheduler must start before its first training step")
    for name, p in params.items():
        require(p.dtype == torch.float32 and list(p.shape) == plan["parameters"][name]["shape"] and
                p.device.type == cfg["device"] and p.grad is None, "Parameter precision/shape/device/gradient differs")
    require(len({p.device for p in params.values()}) == 1, "Trajectory helper requires a single device")
    folder = Path(plan["output"]) / stage
    folder.mkdir(exist_ok=True)
    started = time.monotonic()
    boundary, count = plan["trajectory"]["resume_after_step"], contract["optimizer_steps"]
    initial = state_signature(model, optimizer, scheduler, cursor_for(plan, 0))
    if resume:
        checkpoint, resumed = restore_checkpoint(plan, model, optimizer, scheduler)
        initial = checkpoint["initial"]
        first, stop = boundary + 1, count
    else:
        require(not (folder / "steps").exists(), "Trajectory already started")
        first, stop = 1, boundary if stage == "resume" else count
        resumed = None
    budget = plan["disk_budget_bytes"] // len(STAGES)
    for step in range(first, stop + 1):
        frozen = plan["trajectory"]["steps"][step - 1]
        part = folder / "steps" / f"{step:06d}"
        part.mkdir(parents=True, exist_ok=False)
        optimizer.zero_grad(set_to_none=True)
        before_state = state_signature(model, optimizer, scheduler, cursor_for(plan, step - 1))
        before_file = part / "before.f32.partial"
        with before_file.open("xb") as stream:
            for p in params.values():
                for chunk in torch_chunks(p):
                    stream.write(chunk.astype("<f4", copy=False).tobytes())
        lr_used = [group["lr"] for group in optimizer.param_groups]
        output = backward(step, frozen)
        require(set(output) == {"observations", "input_sha256", "objective", "checks"}, "Callback must identify actual input/objective and observations")
        require(output["input_sha256"] == plan["artifacts"][frozen["input_artifact"]]["sha256"] and
                output["objective"] == frozen["objective"], "Actual input/objective differs from step schedule")
        require(all(output["checks"].get(k) is True for k in plan["required_checks"]), "Required native step check missing")
        require({n: describe(p) for n, p in model.state_dict().items() if n in params} ==
                {n: v for n, v in before_state["model"].items() if n in params}, "Backward callback changed parameters")
        require(describe(optimizer.state_dict()) == before_state["optimizer"] and
                describe(scheduler.state_dict()) == before_state["scheduler"], "Backward callback changed optimizer/scheduler")
        writer = ShardWriter(part, 0, budget - stored_bytes(folder), fields=FIELDS)
        def append(field, name, tensor):
            require(tensor.dtype == torch.float32 and tensor.shape == params[name].shape, "Observed FP32 state shape/dtype differs")
            writer.append(field, {"name": name, "shape": list(tensor.shape), "start": 0, "end": tensor.numel()}, torch_chunks(tensor))
        observations = {}
        require(output["observations"].keys() == frozen["observables"].keys(), "Observation catalog differs")
        for name, value in output["observations"].items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value, dtype=np.float64)
            require(list(array.shape) == frozen["observables"][name] and np.isfinite(array).all(), "Nonfinite/mis-shaped observation")
            observations[name] = {"shape": list(array.shape), "values": array.reshape(-1).tolist()}
        for name, p in params.items():
            require(p.grad is not None and not p.grad.is_sparse, "Missing/sparse gradient requires native integration")
            append("gradients", name, p.grad)
        optimizer.step()
        with before_file.open("rb") as stream:
            for name, p in params.items():
                def deltas(parameter=p):
                    for chunk in torch_chunks(parameter):
                        raw = stream.read(chunk.size * 4)
                        require(len(raw) == chunk.size * 4, "Truncated pre-step snapshot")
                        yield chunk - np.frombuffer(raw, dtype="<f4")
                writer.append("updates", {"name": name, "shape": list(p.shape), "start": 0, "end": p.numel()}, deltas())
                state = optimizer.state[p]
                require(set(state) == {"step", "exp_avg", "exp_avg_sq"} and float(state["step"]) == step,
                        "Adam state catalog/step counter differs")
                for field, value in (("parameters", p), ("exp_avg", state["exp_avg"]), ("exp_avg_sq", state["exp_avg_sq"])):
                    append(field, name, value)
        before_file.unlink()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        state = state_signature(model, optimizer, scheduler, cursor_for(plan, step))
        row = {"plan_sha256": plan["plan_sha256"], "stage": stage, "step": step,
            "input_sha256": output["input_sha256"], "objective": output["objective"], "observations": observations,
            "checks": output["checks"], "storage": writer.close(), "state": state, "lr_used": lr_used,
            "optimizer_step_counters": {name: float(optimizer.state[p]["step"]) for name, p in params.items()}}
        save(part / "receipt.json", row)
        require(stored_bytes(folder) <= budget, "Trajectory raw/checkpoint budget exceeded")
    if stage == "resume" and not resume:
        checkpoint_file = folder / "checkpoint.pt"
        blob = {"plan_sha256": plan["plan_sha256"], "cursor": cursor_for(plan, stop), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state()}
        with checkpoint_file.with_suffix(".pt.partial").open("xb") as stream:
            torch.save(blob, stream)
        checkpoint_file.with_suffix(".pt.partial").replace(checkpoint_file)
        record = {"plan_sha256": plan["plan_sha256"], "stage": stage, "cursor": blob["cursor"], "initial": initial,
                  "producer_pid": os.getpid(), "bytes": checkpoint_file.stat().st_size, "sha256": digest(checkpoint_file),
                  "state": state_signature(model, optimizer, scheduler, blob["cursor"])}
        save(folder / "checkpoint.json", record)
        require(stored_bytes(folder) <= budget, "Checkpoint exceeds trajectory budget")
        return {"status": "checkpoint_written", "completed_steps": stop}
    device = {"kind": cfg["device"], "name": platform.processor() or "CPU"}
    if cfg["device"] == "cuda":
        device.update(name=torch.cuda.get_device_name(next(iter(params.values())).device))
    row = {**identity(plan, stage), "initial": initial, "precision": PRECISION, "runtime": runtime_info(), "device": device,
           "environment": {k: os.environ.get(k) for k in plan["environment"]}, "resume": resumed,
           "execution": {"optimizer_steps": count, "scheduler_steps": count, "teardown_completed": True},
           "step_receipts": {f"steps/{i:06d}/receipt.json": digest(folder / "steps" / f"{i:06d}" / "receipt.json") for i in range(1, count + 1)},
           "timings": {"worker_seconds": time.monotonic() - started}}
    require(not (folder / "rank0.json").exists(), "Trajectory receipt already exists")
    save(folder / "rank0.json", row)
    return {"status": "trajectory_written", "completed_steps": count}
