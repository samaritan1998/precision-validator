"""Training adapters publish canonical evidence without importing Slime or Qwen."""
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import time
from contextlib import nullcontext
import numpy as np
from .common import load_plan, require, save, object_hash
from .protocol import identity
from .storage import ShardWriter, CHUNK_ELEMENTS
from .precision import stage_precision, FP32_STORAGE, NativePrecisionObserver


def runtime_info():
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    numeric_env = ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                   "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_ALGO", "NCCL_PROTO", "TRITON_CACHE_AUTOTUNING")
    result = {"python": platform.python_version(), "packages_sha256": object_hash(versions),
              "numerical_environment": {k: os.environ.get(k) for k in numeric_env}}
    try:
        import torch
    except ImportError:
        return result
    result["torch"] = {"version": str(torch.__version__), "cuda": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    return result


class RankWriter:
    """Write unique global parameter slices; DP replicas must elect an owner.

    start/end index the flattened *global* parameter, not the rank-local tensor.
    Initial digests name complete canonical parameters; sharded adapters must
    construct those digests consistently (a shard digest is not interchangeable).
    """
    def __init__(self, plan, stage, rank):
        self.plan, self.stage, self.rank = plan, stage, rank
        require(stage in plan["configurations"] and type(rank) is int and rank in range(plan["configurations"][stage]["world_size"]), "Invalid stage/rank")
        self.folder = Path(plan["output"]) / stage
        self.folder.mkdir(parents=True, exist_ok=True)
        self.storage = ShardWriter(self.folder, rank, plan["disk_budget_bytes"] // 3)
        self.initial, self.auxiliary, self.observations = {}, {}, {}

    def initial_parameter(self, name, dtype, sha256):
        require(name in self.plan["parameters"] and name not in self.initial, "Unknown/duplicate initial parameter")
        require(isinstance(dtype, str) and dtype and isinstance(sha256, str) and len(sha256) == 64, "Invalid initial parameter proof")
        self.initial[name] = {"shape": self.plan["parameters"][name]["shape"], "dtype": dtype, "sha256": sha256}

    def append(self, field, name, start, end, chunks):
        require(name in self.plan["parameters"], "Unknown canonical parameter")
        shape = self.plan["parameters"][name]["shape"]
        require(0 <= start < end <= math.prod(shape), "Slice outside global parameter")
        self.storage.append(field, {"name": name, "shape": shape, "start": start, "end": end}, chunks)

    def observe(self, name, values):
        require(name in self.plan["observables"] and name not in self.observations, "Unknown/duplicate observation")
        array = np.asarray(values, dtype=np.float64)
        require(list(array.shape) == self.plan["observables"][name] and np.isfinite(array).all(), "Nonfinite/mis-shaped observation")
        self.observations[name] = {"shape": list(array.shape), "values": array.reshape(-1).tolist()}

    def finish(self, *, execution, precision, device, checks=None, timings=None, native_parallelism=None, precision_evidence=None):
        # Execution is provided at the native observation points by the adapter.
        # The launcher independently records process exit; this cannot mark a run passed.
        row = {**identity(self.plan, self.stage, self.rank), "storage": self.storage.close(),
            "initial_parameters": self.initial, "initial_auxiliary": self.auxiliary,
            "observations": self.observations, "execution": execution, "precision": precision,
            "runtime": runtime_info(), "device": device, "checks": checks or {}, "timings": timings or {},
            "environment": {k: os.environ.get(k) for k in self.plan["environment"]}}
        if native_parallelism is not None:
            row["native_parallelism"] = native_parallelism
        if precision_evidence is not None:
            row["precision_evidence"] = precision_evidence
        path = self.folder / f"rank{self.rank}.json"
        require(not path.exists(), "Rank receipt already exists")
        save(path, row)
        return row


def tensor_digest(value):
    import torch
    h = hashlib.sha256()
    flat = value.detach().reshape(-1)
    for start in range(0, flat.numel(), CHUNK_ELEMENTS):
        h.update(flat[start:start + CHUNK_ELEMENTS].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def torch_chunks(value):
    import torch
    flat = value.detach().reshape(-1)
    for start in range(0, flat.numel(), CHUNK_ELEMENTS):
        yield flat[start:start + CHUNK_ELEMENTS].to(device="cpu", dtype=torch.float32).numpy()


def capture_torch_step(plan_path, stage, model, optimizer, backward, *, checks=None):
    """Single-process FP32 PyTorch integration, independent of model architecture.

    backward() performs forward/backward only and returns named observations,
    including scalar loss. It must normalize all accumulated contributions.
    For distributed/master-weight frameworks use RankWriter at native hooks.
    No optimizer formula is reimplemented. Before-values spill to owned storage.
    """
    import torch
    plan = load_plan(plan_path)
    config = plan["configurations"][stage]
    precision = stage_precision(plan, stage)
    require({k: precision.get(k) for k in FP32_STORAGE} == FP32_STORAGE,
            "FP32 observer requires FP32 parameters, gradients and optimizer state")
    require(config["world_size"] == 1, "capture_torch_step is single-process; use RankWriter for distributed training")
    require(type(optimizer) in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD), "Observer supports standard Adam/AdamW/SGD; integrate other optimizers with RankWriter")
    require(type(optimizer).__name__ == plan["contract"]["optimizer"]["name"], "Optimizer differs from contract")
    require(not optimizer.state, "Optimizer must start with fresh state")
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    require(set(params) == set(plan["parameters"]), "Trainable parameter catalog differs")
    require({id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in params.values()}, "Optimizer/model parameter ownership differs")
    for group in optimizer.param_groups:
        for key, value in plan["contract"]["optimizer"].get("options", {}).items():
            actual = group.get(key)
            require((list(actual) if isinstance(actual, tuple) else actual) == value, f"Optimizer option differs: {key}")
    writer = RankWriter(plan, stage, 0)
    started = time.monotonic()
    before_path = writer.folder / "optimizer-before.rank0.f32.partial"
    initial = {}
    with before_path.open("xb") as before:
        for name, p in params.items():
            require(p.dtype == torch.float32 and list(p.shape) == plan["parameters"][name]["shape"], "FP32 helper parameter mismatch")
            initial[name] = tensor_digest(p)
            writer.initial_parameter(name, str(p.dtype), initial[name])
            for chunk in torch_chunks(p):
                before.write(chunk.astype("<f4", copy=False).tobytes())
    writer.auxiliary = {name: {"shape": list(b.shape), "dtype": str(b.dtype), "sha256": tensor_digest(b)} for name, b in model.named_buffers()}
    require(all(p.grad is None for p in params.values()), "Unexpected preexisting gradients")
    observer = (NativePrecisionObserver(model, config["device"], precision)
                if "precision_by_stage" in plan["contract"] else nullcontext())
    with observer:
        observations = backward()
    require(set(observations) == set(plan["observables"]), "Observation catalog differs")
    for name, p in params.items():
        require(tensor_digest(p) == initial[name], "Backward callback changed parameters before observation")
        require(p.grad is not None and p.grad.dtype == torch.float32 and not p.grad.is_sparse, "Missing/sparse/non-FP32 gradient; integrate its semantics explicitly")
        writer.append("gradients", name, 0, p.numel(), torch_chunks(p.grad))
    optimizer.step()
    with before_path.open("rb") as before:
        for name, p in params.items():
            def deltas(parameter=p):
                for after in torch_chunks(parameter):
                    raw = before.read(after.size * 4)
                    require(len(raw) == after.size * 4, "Truncated before-parameter snapshot")
                    yield after - np.frombuffer(raw, dtype="<f4")
            writer.append("updates", name, 0, p.numel(), deltas())
    before_path.unlink()  # Only the snapshot created exclusively by this observer.
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                require(value.dtype == torch.float32 and torch.isfinite(value).all().item(), "Invalid optimizer state")
    for name, value in observations.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        writer.observe(name, value)
    kinds = {p.device.type for p in params.values()}
    require(kinds == {config["device"]}, "Model device differs from configuration")
    device = {"kind": config["device"], "name": platform.processor() or "CPU"}
    if config["device"] == "cuda":
        indices = {p.device.index for p in params.values()}
        require(len(indices) == 1, "Single-process helper expects a single CUDA device")
        index = indices.pop()
        device.update(name=torch.cuda.get_device_name(index), capability=list(torch.cuda.get_device_capability(index)))
    return writer.finish(execution={"backward_completed": True, "gradients_reduced": True, "optimizer_steps": 1,
                         "optimizer_succeeded": True, "teardown_completed": True}, precision=precision,
                         device=device, checks=checks, timings={"observed_step_seconds": time.monotonic() - started},
                         precision_evidence=observer.evidence if isinstance(observer, NativePrecisionObserver) else None)
