"""Observe the real Slime training step; model arithmetic is owned by the candidate.

Derived from the native-entry full_model_check.py used in the October regression.
There are no external precision, attention, memory or optimizer replacements here.
"""
import argparse
from datetime import timedelta
import hashlib
import inspect
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

import torch
import torch.distributed as dist

from .common import ROOT, read, save, digest, load_plan, require, now, verify_files
from .profiles import topology, launch_args, select_sample
from .storage import ShardWriter, CHUNK_ELEMENTS, index_pieces
from .telemetry import Telemetry


def cpu_chunks(tensor):
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), CHUNK_ELEMENTS):
        yield flat[start:start + CHUNK_ELEMENTS].to(device="cpu", dtype=torch.float32, copy=True).numpy()


def parameter_hash(tensor):
    require(tensor.dtype == torch.bfloat16, "Expected BF16 model parameter")
    value = hashlib.sha256()
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), CHUNK_ELEMENTS):
        value.update(flat[start:start + CHUNK_ELEMENTS].cpu().contiguous().view(torch.uint16).numpy().tobytes())
    return value.hexdigest()


def master_parameter(optimizer, parameter):
    state = optimizer.state[parameter]
    if "master_param" not in state:
        return parameter
    value = optimizer.get_unscaled_state(parameter, "master_param") if hasattr(optimizer, "get_unscaled_state") else state["master_param"]
    if value.dtype == torch.int16:
        require(optimizer.store_param_remainders and parameter.dtype == torch.bfloat16, "Unknown master remainder layout")
        upper = parameter.detach().cpu().view(torch.int16).to(torch.int32) << 16
        return (upper + value.cpu().to(torch.int32)).view(torch.float32)
    require(value.dtype == torch.float32, "Master parameters must be FP32")
    return value


def native_receipts(folder, rank, cp, plan):
    records, files = {}, {}
    for kind in ("cp_model", "cp_step"):
        paths = list((folder / "native").glob(f"{kind}-rank{rank}-pid*.json"))
        require(len(paths) == 1, f"Expected one {kind} receipt for rank {rank}")
        path = paths[0]
        row = read(path)
        require(row["rank"] == rank and row["cp"] == cp and row["profile"] == plan["profile"]["name"], "Native receipt identity mismatch")
        require(row["versions"] == plan["runtime_lock"]["versions"], "Native runtime version mismatch")
        records[kind] = row
        files[str(path.relative_to(folder))] = digest(path)
    before, after = records["cp_model"], records["cp_step"]
    require(before["sources"] == after["sources"], "Native sources changed within a step")
    require(after["optimizer_updated"] is True and after["steps"] == 1, "No native optimizer update")
    require(after["grad_sync"]["calls"] == after["grad_sync"]["completed"] > 0, "Gradient synchronization incomplete")
    # These functions only inspect counters; they never install model controls.
    from slime_plugins.models.qwen_cp.runtime import audit_execution
    audit_execution(after["controls"], before["controls"])
    if plan["profile"]["moe"]:
        require(after["precision"]["shared_gate_reduction"] == "row-sum", "MoE gate precision mode mismatch")
        require(after["precision"]["modules"] and all(p["calls"] > 0 for p in after["precision"]["modules"]), "MoE controls not executed")
    from slime_plugins.models.qwen_fla_precision import snapshot
    fla = snapshot(require_launched=True)
    require(fla["mode"] == "two-forward", "FLA config not pinned")
    expected = {
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": ("BV: 32", "num_warps: 2", "num_stages: 2"),
        "chunk_fwd_kernel_o": ("BK: 64", "BV: 64", "num_warps: 4", "num_stages: 3"),
    }
    require({row["kernel"] for row in fla["configs"]} == set(expected), "FLA kernel coverage mismatch")
    for row in fla["configs"]:
        require(row["calls"] > 0 and all(v in row["config"] and v in row["best_config"] for v in expected[row["kernel"]]), "FLA pin did not execute")
    return {"files": files, "fla": fla, "optimizer_updated": True, "grad_sync": after["grad_sync"],
            "controls": sorted(after["controls"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", required=True, choices=("cp1", "cp1_repeat", "cp_target"))
    cli = parser.parse_args()
    plan = load_plan(cli.plan)
    require(__debug__, "Do not run production Slime with python -O")
    require(not plan["external_diagnostic_patches"], "External diagnostic controls are not allowed")
    verify_files(ROOT, plan["toolkit_files"])
    profile = plan["profile"]
    layout = topology(profile, cli.stage)
    cp, pp, ep = layout["cp"], layout["pp"], layout["ep"]
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    require(int(os.environ["WORLD_SIZE"]) == 8, "Expected one node with eight ranks")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    cache = Path("/tmp") / ("precision-" + plan["plan_sha256"][:12]) / cli.stage / f"rank{rank}"
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    import torch._inductor.config as inductor_config
    inductor_config.compile_threads = 1
    from slime.utils.reloadable_process_group import monkey_patch_torch_dist
    monkey_patch_torch_dist()
    dist.init_process_group("nccl", timeout=timedelta(minutes=180))
    folder = Path(plan["output"]) / cli.stage
    progress = {"phase": "initialize"}
    def heartbeat():
        save(folder / f"heartbeat.rank{rank}.json", {**progress, "utc": now(), "rank": rank,
            "allocated_bytes": torch.cuda.memory_allocated(local_rank), "reserved_bytes": torch.cuda.memory_reserved(local_rank),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(local_rank)})

    telemetry = Telemetry(heartbeat).start()
    sys.argv = [sys.argv[0], *launch_args(plan, cli.stage)]
    from slime.utils.arguments import parse_args
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model import setup_model_and_optimizer, train_one_step
    from slime.backends.megatron_utils.checkpoint import _load_checkpoint_hf
    from megatron.core import mpu
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.utils import get_model_config, unwrap_model
    from slime.backends.megatron_utils.data import DataIterator
    from slime.utils.dp_schedule import build_dp_schedule

    entry = {}
    for name, func in (("initialize", init), ("setup_model_and_optimizer", setup_model_and_optimizer), ("train_one_step", train_one_step)):
        path = Path(inspect.getsourcefile(func)).resolve()
        source = Path(plan["source"]).resolve()
        require(path.is_relative_to(source), f"{name} imported from the wrong source")
        relative = str(path.relative_to(source))
        require(digest(path) == plan["source_inventory"]["files"][relative], f"{name} source hash mismatch")
        entry[name] = {"relative_path": relative, "sha256": digest(path)}
    args = parse_args()
    require(args.calculate_per_token_loss and not getattr(args, "global_batch_tokens", None), "Token-normalized native loss required")
    require(args.global_batch_size == plan["sample_copies_per_step"], "Global batch size changed")
    args.rank, args.world_size = rank, 8
    init(args)  # Installs the candidate's rank controls after CUDA binding.
    progress["phase"] = "build_model"
    model, optimizer, scheduler = setup_model_and_optimizer(args)
    progress["phase"] = "load_hf"
    _load_checkpoint_hf(model, optimizer, args, plan["model_path"])
    module = unwrap_model(model)[0]
    language = getattr(module, "language_model", module)
    layers = len(language.decoder.layers)
    require(layers == profile["layers"] // pp and args.num_layers == profile["layers"], "Layer count mismatch")
    require(args.num_experts == (256 if profile["moe"] else None), "Expert count mismatch")
    require(mpu.get_expert_model_parallel_world_size() == ep, "Expert topology mismatch")
    if profile["moe"]:
        require(args.moe_aux_loss_coeff == 0, "MoE auxiliary loss must be zero for this comparison")
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    attention = [{"layer": a.layer_number, "mode": a.cp_comm_type}
                 for a in language.modules() if isinstance(a, TEDotProductAttention)]
    require(len(attention) == sum(layer.layer_number % 4 == 0 for layer in language.decoder.layers), "Attention coverage mismatch")
    if cp > 1:
        require(all(row["mode"] in ("a2a", ["a2a"]) for row in attention), "Native TE attention did not receive a2a")
    for chunk in model:
        require(chunk.ddp_config.grad_reduce_in_fp32, "Gradient reduction must be FP32")
    names = {p: name for name, p in module.named_parameters()}
    require(all(name.startswith("language_model.") and p.dtype == torch.bfloat16
                for p, name in names.items() if p.requires_grad), "Trainable scope or BF16 model dtype mismatch")

    def global_name(p):
        name = re.sub(r"(language_model.decoder.layers\.)(\d+)",
            lambda match: match[1] + str(int(match[2]) + layers * mpu.get_pipeline_model_parallel_rank()), names[p])
        if getattr(p, "allreduce", True):
            return name
        expert_offset = mpu.get_expert_model_parallel_rank() * (args.num_experts // ep)
        match = re.search(r"\.experts\.(linear_fc[12])\.weight(\d+)$", name)
        if match:
            expert = int(match[2])
            require(0 <= expert < args.num_experts // ep, "Expert index out of range")
            return name[:match.start()] + f".experts.global_experts.{expert_offset + expert}.{match[1]}.weight"
        match = re.search(r"\.experts\.local_experts\.(\d+)\.", name)
        if match:
            expert = int(match[1])
            require(0 <= expert < args.num_experts // ep, "Expert index out of range")
            return name[:match.start()] + f".experts.global_experts.{expert_offset + expert}." + name[match.end():]
        raise RuntimeError(f"Unsupported expert parameter layout: {name}")

    # Hash whole model parameters before training. DP replicas must agree. This
    # permits cross-layout initial-weight comparison without storing initial tensors.
    progress["phase"] = "initial_hashes"
    initial = {global_name(p): {"shape": list(p.shape), "dtype": "torch.bfloat16", "sha256": parameter_hash(p)}
               for p in names if p.requires_grad}
    inners = getattr(optimizer, "chained_optimizers", [optimizer])
    mapping, model_parameters = {}, {}
    for oi, inner in enumerate(inners):
        for p, (gi, pi) in inner.model_param_group_index_map.items():
            owned = inner._get_model_param_range_map(p)["param"]
            key = f"{oi}.{gi}.{pi}"
            mapping[key] = {"name": global_name(p), "shape": list(p.shape), "start": owned.start,
                            "end": owned.end, "elements": owned.end - owned.start, "offset_bytes": 0}
            model_parameters[key] = p
        require(all(group["lr"] == plan["learning_rate"] for group in inner.optimizer.param_groups), "Unexpected learning rate")
        for state in inner.optimizer.state.values():
            require("step" not in state or float(state["step"]) == 0, "Optimizer did not start fresh")
    all_mappings = [None] * 8
    dist.all_gather_object(all_mappings, list(mapping.values()))
    index_pieces([p for rows in all_mappings for p in rows], profile["trainable_elements"])
    del all_mappings

    sample, sample_proof = select_sample(read(plan["input_path"]), profile["sequence_length"], 0)
    require(sample_proof["sample_sha256"] == plan["input_proof"]["sample_sha256"], "Selected sample changed")
    copies = plan["sample_copies_per_step"]
    length = profile["sequence_length"]
    tokens = torch.tensor(sample["token_ids"], dtype=torch.long, device="cuda")
    parts, batches, counts, global_sizes = build_dp_schedule(args, {"dp_size": layout["dp"], "cp_size": cp,
        "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}, [length] * copies,
        global_batch_size=copies, rollout_indices=list(range(copies)), sample_is_multimodal=[False] * copies)
    microbatches = copies // layout["dp"]
    require(counts == [microbatches] and global_sizes == [copies], "Contribution schedule mismatch")
    require(all(len(p) == microbatches for p in parts), "Unbalanced DP contributions")
    require(all(len(b) == microbatches and all(len(mb) == 1 for mb in b) for b in batches), "Unexpected packing")
    dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
    local_ids = parts[dp_rank]
    first = sample["loss_mask"].index(1)
    mask = torch.tensor(sample["loss_mask"][first:], dtype=torch.long, device="cuda")
    rollout = {"tokens": [tokens] * microbatches, "total_lengths": [length] * microbatches,
               "response_lengths": [length - first] * microbatches, "loss_masks": [mask] * microbatches}
    iterator = [DataIterator(rollout, batches[dp_rank])]
    schedule = {"dp_rank": dp_rank, "indices": local_ids, "batches": batches[dp_rank],
                "microbatches": microbatches, "supervised_tokens": int(mask.sum())}

    forward = {}
    wanted = torch.linspace(0, length - 1, 64).round().long()
    features = torch.linspace(0, profile["hidden_size"] - 1, 32).round().long()
    positions = torch.arange(length)
    if cp > 1:
        chunks = positions.chunk(2 * cp)
        cr = mpu.get_context_parallel_rank()
        positions = torch.cat((chunks[cr], chunks[-cr - 1]))
    local = torch.nonzero(torch.isin(positions, wanted)).flatten()
    def observe_layer(number):
        def hook(mod, incoming, output):
            progress.update(phase="forward_backward", layer=number)
            if dp_rank != 0 or str(number) in forward:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach().reshape(-1, profile["hidden_size"])
            require(hidden.shape[0] == len(positions), "Forward sampling sequence layout mismatch")
            sampled = hidden[local.to(hidden.device)[:, None], features.to(hidden.device)[None, :]].float().cpu()
            require(torch.isfinite(sampled).all(), "Nonfinite sampled hidden states")
            forward[str(number)] = {"positions": positions[local].tolist(), "features": features.tolist(), "values": sampled.tolist()}
        return hook
    for number, layer in enumerate(language.decoder.layers, 1 + layers * mpu.get_pipeline_model_parallel_rank()):
        layer.register_forward_hook(observe_layer(number))

    writer = ShardWriter(folder, rank, plan["disk_budget_bytes"] // 3)
    before, gradient_dtypes = {}, {}
    observed = {"captured": False, "gradient_capture_seconds": 0.0, "optimizer_step_calls": 0,
                "optimizer_wall_seconds": 0.0, "optimizer_success": False, "clip_calls": []}
    original_prepare, original_step = optimizer.prepare_grads, optimizer.step
    train_started = 0.0
    def prepare_and_capture():
        found_inf = original_prepare()
        require(not found_inf, "Found inf in native optimizer gradients")
        if observed["captured"]:
            return found_inf
        observed["forward_backward_reduce_seconds"] = time.monotonic() - train_started
        progress["phase"] = "capture_gradients"
        capture_started = time.monotonic()
        for oi, inner in enumerate(inners):
            for gi, group in enumerate(inner.optimizer.param_groups):
                for pi, p in enumerate(group["params"]):
                    key = f"{oi}.{gi}.{pi}"
                    if key not in mapping:
                        continue
                    grad = getattr(p, "decoupled_grad", None)
                    if grad is None:
                        grad = p.grad
                    require(grad is not None and grad.dtype == torch.float32, f"Expected FP32 gradient: {key}")
                    gradient_dtypes[key] = str(grad.dtype)
                    writer.append("gradients", mapping[key], cpu_chunks(grad))
                    initial_master = master_parameter(inner.optimizer, p).detach().to(device="cpu", dtype=torch.float32, copy=True)
                    piece = mapping[key]
                    model_flat = model_parameters[key].detach().reshape(-1)
                    flat = initial_master.reshape(-1)
                    require(flat.numel() == piece["elements"], "Optimizer shard length mismatch")
                    for offset in range(0, flat.numel(), CHUNK_ELEMENTS):
                        actual = flat[offset:offset + CHUNK_ELEMENTS]
                        expected = model_flat[piece["start"] + offset:piece["start"] + offset + actual.numel()].float().cpu()
                        require(torch.equal(actual, expected), "Initial optimizer master differs from the loaded BF16 model")
                    before[key] = initial_master
        require(set(before) == set(mapping), "Not all optimizer shards observed")
        observed["captured"] = True
        observed["gradient_capture_seconds"] += time.monotonic() - capture_started
        return found_inf
    optimizer.prepare_grads = prepare_and_capture
    def measured_step(*a, **kw):
        start = time.monotonic()
        result = original_step(*a, **kw)
        observed["optimizer_wall_seconds"] += time.monotonic() - start
        observed["optimizer_step_calls"] += 1
        observed["optimizer_success"] = bool(result[0])
        return result
    optimizer.step = measured_step
    for index, inner in enumerate(inners):
        if hasattr(inner, "clip_grad_norm"):
            original_clip = inner.clip_grad_norm
            def measured_clip(*a, _original=original_clip, _index=index, **kw):
                result = _original(*a, **kw)
                observed["clip_calls"].append({"optimizer": _index, "clip_threshold": float(a[0]) if a else float(args.clip_grad),
                                               "reported_norm": float(result)})
                return result
            inner.clip_grad_norm = measured_clip
    config = get_model_config(model[0])
    config.grad_scale_func, config.timers = optimizer.scale_loss, None
    config.finalize_model_grads_func = finalize_model_grads
    for chunk in model:
        chunk.train()
    torch.cuda.reset_peak_memory_stats()
    train_started = time.monotonic()
    progress["phase"] = "train_step"
    losses, grad_norm = train_one_step(args, 0, 0, iterator, model, optimizer, scheduler, counts[0], copies)
    torch.cuda.synchronize()
    train_seconds = time.monotonic() - train_started
    require(observed["captured"] and observed["optimizer_step_calls"] == 1 and observed["optimizer_success"], "One native optimizer update did not complete")
    require(math.isfinite(float(grad_norm)), "Nonfinite native gradient norm")
    progress["phase"] = "capture_updates"
    capture_started = time.monotonic()
    dtypes = []
    for oi, inner in enumerate(inners):
        for gi, group in enumerate(inner.optimizer.param_groups):
            for pi, p in enumerate(group["params"]):
                key = f"{oi}.{gi}.{pi}"
                if key not in mapping:
                    continue
                master = master_parameter(inner.optimizer, p)
                require(master.dtype == torch.float32, "Actual updated master must be FP32")
                state = inner.optimizer.state[p]
                require(all(name in state and state[name].dtype == torch.float32 for name in ("exp_avg", "exp_avg_sq")),
                        "Adam moments must be FP32")
                require("step" not in state or float(state["step"]) == 1, "Adam step count is not one")
                initial_flat = before.pop(key).reshape(-1)
                def delta_chunks():
                    for offset in range(0, master.numel(), CHUNK_ELEMENTS):
                        value = master.detach().reshape(-1)[offset:offset + CHUNK_ELEMENTS].float().cpu()
                        yield (value - initial_flat[offset:offset + value.numel()]).numpy()
                writer.append("updates", mapping[key], delta_chunks())
                dtypes.append({"key": key, "gradient": gradient_dtypes[key], "master": str(master.dtype),
                               "exp_avg": str(state["exp_avg"].dtype), "exp_avg_sq": str(state["exp_avg_sq"].dtype)})
    storage = writer.close()
    update_seconds = time.monotonic() - capture_started
    native = native_receipts(folder, rank, cp, plan)
    verify_files(plan["source"], plan["source_inventory"]["files"])
    telemetry_receipt = telemetry.stop()
    if not telemetry_receipt["stopped"] or telemetry_receipt["last_errors"]:
        print("PRECISION_TELEMETRY_WARNING", rank, telemetry_receipt, flush=True)
    peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    dist.barrier()
    dist.destroy_process_group()
    save(folder / f"rank{rank}.json", {"schema_version": 1, "rank": rank, "stage": cli.stage,
        "plan_sha256": plan["plan_sha256"], "topology": layout, "entry": entry, "initial_parameters": initial,
        "sample_sha256": sample_proof["sample_sha256"], "schedule": schedule, "storage": storage,
        "native": native, "attention_modes": attention, "optimizer_dtypes": dtypes,
        "gradient_stage": "after_prepare_grads_before_clipping", "optimizer_step_executed": True,
        "optimizer_observation": observed, "losses": {key: float(value) for key, value in losses.items()},
        "grad_norm": float(grad_norm), "clip_threshold": float(args.clip_grad),
        "clip_factor_from_norm": min(1.0, float(args.clip_grad) / (float(grad_norm) + 1e-6)) if args.clip_grad > 0 else 1.0,
        "clip_factor_note": "Derived from the native norm and the pinned Megatron clipping epsilon; clip_calls record native execution",
        "forward_samples": forward, "distributed_groups_destroyed": True, "finished_utc": now(),
        "timings": {"train_step_wall_seconds": train_seconds, "forward_backward_reduce_seconds": observed["forward_backward_reduce_seconds"],
                    "gradient_capture_seconds": observed["gradient_capture_seconds"], "update_capture_seconds": update_seconds,
                    "optimizer_wall_seconds_including_gradient_capture": observed["optimizer_wall_seconds"]},
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "telemetry": telemetry_receipt})
    print("PRECISION_STAGE_RANK_COMPLETE", rank, cli.stage, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # Preserve the first per-rank failure even when torchrun terminates peers.
        import traceback
        print("PRECISION_RANK_FAILED", os.environ.get("RANK"), repr(error), flush=True)
        traceback.print_exc()
        raise
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
