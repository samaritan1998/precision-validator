"""Native TP, PP and fixed-route EP examples on CPU/Gloo or CUDA/NCCL.

Each experiment compares an unpartitioned model with a two-rank partition.
TP partitions output rows, PP sends activations and their backward gradient,
EP owns separate experts with a fixed routing input and sums their outputs.
"""
import argparse
import copy
from datetime import timedelta
import os
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.nn import functional as distributed

from precision_validator.common import read, require, save, load_plan
from precision_validator.parallel import CONVENTION, parse_layout, validate_layout
from precision_validator.sdk import RankWriter, tensor_digest, torch_chunks


class PipelineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 6)
        self.second = nn.Linear(6, 4)

    def forward(self, x):
        return self.second(torch.tanh(self.first(x)))


class ExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, x):
        outputs = x.new_zeros((len(x), 4))
        for expert, module in enumerate(self.experts):
            selected = torch.arange(expert, len(x), 2, device=x.device)
            outputs = outputs.index_copy(0, selected, module(x[selected]))
        return outputs


def model_for(kind):
    return nn.Linear(4, 4) if kind == "tp" else PipelineModel() if kind == "pp" else ExpertModel()


def make(output, kind, device="cpu"):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(1979)
    model = model_for(kind)
    save(root / "input.json", {"kind": kind, "x": torch.randn(8, 4).tolist(), "y": torch.randn(8, 4).tolist(),
                               "routing": "row_index_mod_2" if kind == "ep" else None})
    reference = parse_layout("tp=1,pp=1,cp=1,ep=1,etp=1", 1)
    target = dict(reference)
    target[kind] = 2
    if kind == "ep":
        target["dp"] = 2
    validate_layout(target, 2)
    limit = {"relative_l2": 0.003, "max_abs": 2e-5, "near_zero_rms": 1e-9, "near_zero_max_abs": 1e-8}
    acceptance = {category: [{"pattern": "*", **{f: copy.deepcopy(limit) for f in ("gradients", "updates")}}]
                  for category in ("parameters", "groups")}
    acceptance["observations"] = {name: copy.deepcopy(limit) for name in ("loss", "outputs")}
    global_policy = {f: {"repeat_relative_ceiling": 1e-6, "target_relative_ceiling": 0.003,
        "noise_multiplier": 2, "noise_slack": 0.003, "max_abs_ceiling": 2e-5} for f in ("gradients", "updates")}
    spec = {"schema_version": 1, "name": f"{device}-native-{kind}1-vs-{kind}2", "scope": "integration_test",
        "artifacts": {"code": str(Path(__file__).resolve()), "input": str(root / "input.json")},
        "parameters": {name: {"shape": list(p.shape), "group": name.rsplit(".", 1)[0]} for name,p in model.named_parameters()},
        "configurations": {"reference": {"world_size": 1, "device": device, "parallelism": reference},
                           "candidate": {"world_size": 2, "device": device, "parallelism": target}},
        "parallel_adapter": {"convention": CONVENTION, "framework": "native-pytorch-partition-demo", "supported_axes": [kind, "dp"]},
        "command": ["{python}", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "{world_size}",
                    str(Path(__file__).resolve()), "worker", "--plan", "{plan}", "--stage", "{stage}"],
        "contract": {"seed": 1979, "gradient_point": "after_reduction_before_clipping",
            "update_point": "optimizer_parameter_after_minus_before", "optimizer_steps": 1, "optimizer_initial_state": "fresh",
            "optimizer": {"name": "AdamW", "options": {"lr": 0.001, "weight_decay": 0.01, "foreach": False, "fused": False}},
            "objective": {"normalization": "mean_squared_error_over_all_32_outputs", "global_samples": 8, "supervised_units": 32},
            "precision": {"parameters": "float32", "gradients": "float32", "optimizer_parameters": "float32", "optimizer_state": "float32"}},
        "observables": {"loss": [], "outputs": [8, 4]},
        "required_checks": ["native_partition_executed", "full_batch_contributed", "native_adamw_one_step"],
        "environment": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        "policy": global_policy, "acceptance": acceptance}
    save(root / "spec.json", spec)
    return root / "spec.json"


def native_group_receipt(layout, rank, world):
    sizes = validate_layout(layout, world)
    memberships = {r: {} for r in range(world)}
    for axes in (("tp", "cp", "dp", "pp"), ("etp", "ep", "edp", "pp")):
        stride = 1
        for axis in axes:
            for r in range(world):
                start = r - (r // stride % sizes[axis]) * stride
                memberships[r][axis] = tuple(start + i * stride for i in range(sizes[axis]))
            stride *= sizes[axis]
    groups = {}
    for members in sorted({group for row in memberships.values() for group in row.values()}):
        groups[members] = dist.new_group(list(members), backend=dist.get_backend())
    actual = {axis: dist.get_process_group_ranks(groups[members]) for axis,members in memberships[rank].items()}
    return {"convention": CONVENTION, "degrees": sizes, "groups": actual}


def worker(path, stage):
    plan = load_plan(path)
    config = plan["configurations"][stage]
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    require(world == config["world_size"], "Example allocation mismatch")
    device = torch.device(config["device"], int(os.environ["LOCAL_RANK"])) if config["device"] == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan["contract"]["seed"])
    data = read(plan["artifacts"]["input"]["path"])
    kind = data["kind"]
    expected = parse_layout("tp=1,pp=1,cp=1,ep=1,etp=1", 1)
    if world == 2:
        expected[kind] = 2
        if kind == "ep":
            expected["dp"] = 2
    require(world in (1, 2) and config["parallelism"] == expected, "Example implements only one-to-two native axis comparison")
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=2))
    native = native_group_receipt(expected, rank, world)
    full = model_for(kind).to(device).train()
    catalog = dict(full.named_parameters())
    require(set(catalog) == set(plan["parameters"]), "Canonical parameter catalog differs")
    writer = RankWriter(plan, stage, rank)
    for name, p in catalog.items():
        writer.initial_parameter(name, str(p.dtype), tensor_digest(p))
    x, y = torch.tensor(data["x"], device=device), torch.tensor(data["y"], device=device)
    require(x.shape[0] == plan["contract"]["objective"]["global_samples"] and y.numel() == plan["contract"]["objective"]["supervised_units"],
            "Global batch or loss denominator changed")
    mappings = {}
    if world == 1:
        model = full
        mappings = {name: (name, 0, p.numel()) for name,p in model.named_parameters()}
    elif kind == "tp":
        model = nn.Linear(4, 2).to(device)
        with torch.no_grad():
            model.weight.copy_(full.weight[rank * 2:(rank + 1) * 2])
            model.bias.copy_(full.bias[rank * 2:(rank + 1) * 2])
        mappings = {"weight": ("weight", rank * 8, (rank + 1) * 8), "bias": ("bias", rank * 2, (rank + 1) * 2)}
    elif kind == "pp":
        prefix = "first" if rank == 0 else "second"
        model = getattr(full, prefix)
        mappings = {name: (prefix + "." + name, 0, p.numel()) for name,p in model.named_parameters()}
    else:
        require(data["routing"] == "row_index_mod_2", "EP example routing changed")
        model = full.experts[rank]
        mappings = {name: (f"experts.{rank}." + name, 0, p.numel()) for name,p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), **plan["contract"]["optimizer"]["options"])
    before = {name: p.detach().clone() for name,p in model.named_parameters()}
    outputs, loss = None, None
    if world == 1:
        outputs = model(x)
        loss = (outputs - y).square().mean()
        loss.backward()
    elif kind == "tp":
        outputs = torch.cat(distributed.all_gather(model(x)), dim=-1)
        loss = (outputs - y).square().mean()
        # All ranks differentiate the same gathered loss; the autograd collective
        # sums those upstream derivatives, so each loss contributes 1/world.
        (loss / world).backward()
    elif kind == "ep":
        indices = torch.arange(rank, len(x), 2, device=device)
        local = x.new_zeros(y.shape).index_copy(0, indices, model(x[indices]))
        outputs = distributed.all_reduce(local)
        loss = (outputs - y).square().mean()
        (loss / world).backward()
    elif rank == 0:
        activation = torch.tanh(model(x))
        dist.send(activation.detach().contiguous(), dst=1)
        upstream = torch.empty_like(activation)
        dist.recv(upstream, src=1)
        activation.backward(upstream)
    else:
        activation = torch.empty((len(x), 6), device=device)
        dist.recv(activation, src=0)
        activation.requires_grad_(True)
        outputs = model(activation)
        loss = (outputs - y).square().mean()
        loss.backward()
        dist.send(activation.grad.contiguous(), dst=0)
    for local_name, p in model.named_parameters():
        name, start, end = mappings[local_name]
        require(p.grad is not None, "Native parameter has no gradient")
        writer.append("gradients", name, start, end, torch_chunks(p.grad))
    optimizer.step()
    require(len(optimizer.state) == len(list(model.parameters())) and all(float(s["step"]) == 1
            and s["exp_avg"].dtype == s["exp_avg_sq"].dtype == torch.float32 for s in optimizer.state.values()),
            "Native AdamW state or step mismatch")
    for local_name, p in model.named_parameters():
        name, start, end = mappings[local_name]
        writer.append("updates", name, start, end, torch_chunks(p.detach() - before[local_name]))
    output_owner = 1 if kind == "pp" and world == 2 else 0
    if rank == output_owner:
        writer.observe("loss", loss.detach().cpu().numpy())
        writer.observe("outputs", outputs.detach().cpu().numpy())
    dist.barrier()
    dist.destroy_process_group()
    writer.finish(execution={"backward_completed": True, "gradients_reduced": True, "optimizer_succeeded": True,
        "optimizer_steps": 1, "teardown_completed": not dist.is_initialized()},
        precision=plan["contract"]["precision"], device={"kind": device.type, "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"},
        checks={"native_partition_executed": True, "full_batch_contributed": True, "native_adamw_one_step": True},
        native_parallelism=native)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    create = sub.add_parser("make")
    create.add_argument("--kind", choices=("tp", "pp", "ep"), required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run = sub.add_parser("worker")
    run.add_argument("--plan", required=True)
    run.add_argument("--stage", required=True)
    args = cli.parse_args()
    if args.command == "make":
        print(make(args.output, args.kind, args.device))
    else:
        worker(args.plan, args.stage)
