"""Short infrastructure smoke test. It is never full-model numerical acceptance."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
from .common import save, read, require, digest, now
from .profiles import get_profile


def launch(profile_name, output):
    from .runtime import doctor
    from .runner import stop_process
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    profile = get_profile(profile_name)
    for key, value in profile["environment"].items():
        os.environ.setdefault(key, value)
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    proof = doctor(profile)
    save(folder / "environment.json", proof)
    process = None
    def interrupted(signum, frame):
        raise InterruptedError(f"Smoke interrupted by signal {signum}")
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8", "--max-restarts=0",
                   "-m", "precision_validator.gpu_smoke", "--worker", "--output", str(folder)]
        with (folder / "smoke.log").open("x") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=600)
        require(code == 0, f"GPU smoke process exited {code}; see smoke.log")
        ranks = {}
        for rank in range(8):
            path = folder / f"rank{rank}.json"
            row = read(path)
            require(row["rank"] == rank and row["status"] == "infrastructure_smoke_passed", "Missing smoke rank receipt")
            ranks[path.name] = digest(path)
        report = {"status": "infrastructure_smoke_passed", "full_model_precision_passed": False,
                  "profile_device": profile["device"], "ranks": ranks, "environment_sha256": digest(folder / "environment.json"),
                  "utc": now(), "scope": "CUDA BF16 GEMM/backward, FP32 Adam states, native FA2 backward and eight-rank NCCL"}
        save(folder / "smoke-complete.json", report)
        return report
    finally:
        if process is not None:
            stop_process(process)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def worker(output):
    import torch
    import torch.distributed as dist
    from flash_attn import flash_attn_func
    # Import the two extension stacks used by the full model on the real driver.
    import transformer_engine.pytorch
    import fla
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(1234)
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    require(dist.get_world_size() == 8, "Smoke requires eight ranks")
    total = torch.tensor(float(rank), dtype=torch.float32, device="cuda")
    dist.all_reduce(total)
    require(total.item() == 28, "NCCL FP32 sum failed")
    weight = torch.nn.Parameter(torch.randn(64, 64, device="cuda", dtype=torch.float32))
    initial = weight.detach().clone()
    optimizer = torch.optim.AdamW([weight], lr=2e-5)
    inputs = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    loss = (inputs @ weight.to(torch.bfloat16)).float().square().mean()
    loss.backward()
    require(weight.grad.dtype == torch.float32 and torch.isfinite(weight.grad).all(), "BF16 GEMM backward failed")
    dist.all_reduce(weight.grad)
    weight.grad.div_(8)
    optimizer.step()
    require(torch.isfinite(weight).all() and not torch.equal(weight, initial), "FP32 master update failed")
    require(all(optimizer.state[weight][k].dtype == torch.float32 for k in ("exp_avg", "exp_avg_sq")), "Adam state dtype failed")
    q, k, v = [torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    output_tensor = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    output_tensor.float().square().mean().backward()
    require(all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v)), "Native FA2 backward failed")
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated()
    dist.barrier()
    dist.destroy_process_group()
    save(Path(output) / f"rank{rank}.json", {"rank": rank, "status": "infrastructure_smoke_passed",
         "full_model_precision_passed": False, "device": torch.cuda.get_device_name(),
         "capability": list(torch.cuda.get_device_capability()), "nccl_sum": 28,
         "loss": float(loss), "master_update_l2": float((weight - initial).norm()),
         "peak_allocated_bytes": allocated, "distributed_groups_destroyed": True, "utc": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--output", required=True)
    worker(parser.parse_args().output)
