"""Observe an unmodified native training step and canonical optimizer shards.

This adapter neither implements a training schedule nor changes model arithmetic.
Enter after DDP and checkpoint loading. It observes prepare_grads before clipping,
and the actual FP32 optimizer parameters immediately after optimizer.step.
"""
import hashlib
import math
import time
from pathlib import Path

import numpy as np

from .common import require
from .megatron_mapping import collective_fragments, append_owned
from .sdk import torch_chunks
from .storage import ShardWriter, index_pieces, verify_storage, CHUNK_ELEMENTS


def adam_counter(row):
    """Read PyTorch's parameter counter or native fused Adam's group counter."""
    optimizer, parameter = row['state_optimizer'], row['state_parameter']
    state = optimizer.state.get(parameter, {})
    if 'step' in state:
        return float(state['step'])
    groups = [g for g in optimizer.param_groups if any(p is parameter for p in g['params'])]
    require(len(groups) == 1, 'Native Adam parameter group ownership differs')
    return float(groups[0].get('step', 0))


def master_parameter(optimizer, parameter):
    """Read native master weights, including TE's signed BF16 remainders."""
    import torch
    state = optimizer.state.get(parameter, {})
    if type(optimizer) in (torch.optim.Adam, torch.optim.AdamW) and parameter.dtype == torch.float32:
        # Hybrid's metadata may alias the actual CPU parameter. After restoring
        # outer state it can instead point to a different tensor; do not mistake
        # that unsynchronized metadata for the parameter Adam will update.
        if 'master_param' in state:
            require(state['master_param'].dtype == torch.float32 and
                    state['master_param'].shape == parameter.shape and
                    torch.equal(state['master_param'].detach().cpu(), parameter.detach().cpu()),
                    'Native CPU master metadata differs from the actual Adam parameter')
        return parameter
    if "master_param" not in state:
        require(parameter.dtype == torch.float32, "BF16 optimizer parameter has no FP32 master")
        return parameter
    value = (optimizer.get_unscaled_state(parameter, "master_param")
             if hasattr(optimizer, "get_unscaled_state") else state["master_param"])
    if value.dtype == torch.int16:
        require(getattr(optimizer, "store_param_remainders", False) and parameter.dtype == torch.bfloat16,
                "Unknown master remainder layout")
        upper = parameter.detach().cpu().view(torch.int16).to(torch.int32) << 16
        value = (upper + value.detach().cpu().to(torch.int32)).view(torch.float32)
    require(value.dtype == torch.float32 and value.shape == parameter.shape,
            "Unexpected native FP32 master shape/dtype")
    return value


def hybrid_owners(optimizer):
    """Resolve actual CPU/GPU sub-optimizers before Hybrid state is materialized.

    Build one identity index per native optimizer, not a parameter-by-parameter
    scan. Do not create states or substitute rounded model values for masters.
    """
    import torch
    mapping = getattr(optimizer, "param_to_inner_param", None)
    if mapping is None:
        return None
    require(getattr(optimizer, "param_update_in_fp32", False), "Hybrid master parameters must be FP32")
    owners = {}
    for inner in optimizer.sub_optimizers:
        for group in inner.param_groups:
            for parameter in group["params"]:
                require(parameter not in owners, "Duplicate hybrid master owner")
                owners[parameter] = inner
    result = {}
    for original, actual in mapping.items():
        require(actual in owners and actual.dtype == torch.float32 and actual.shape == original.shape,
                "Missing or invalid native hybrid master ownership, shape or dtype")
        result[original] = (owners[actual], actual)
    require(len(result) == len(owners) and len({id(p) for _, p in result.values()}) == len(result),
            "Hybrid master mapping is not one-to-one")
    return result


def optimizer_shards(optimizer, mapping):
    """Use native DP ranges; never infer TP/PP/EP ownership from names."""
    names = {p: n for n, p in mapping.parameters.items()}
    result = {}
    for inner in getattr(optimizer, "chained_optimizers", [optimizer]):
        if getattr(inner, "is_stub_optimizer", False):
            continue
        require(hasattr(inner, "model_param_group_index_map"), "Distributed optimizer metadata required")
        native_owners = hybrid_owners(inner.optimizer)
        for parameter, (group, index) in inner.model_param_group_index_map.items():
            require(parameter in names, "Optimizer contains an unmapped model parameter")
            name = names[parameter]
            require(name not in result, "Duplicate native optimizer ownership")
            owned = inner._get_model_param_range_map(parameter)["param"]
            actual = inner.optimizer.param_groups[group]["params"][index]
            require(0 <= owned.start < owned.end <= parameter.numel()
                    and actual.numel() == owned.end - owned.start, "Native optimizer range mismatch")
            if native_owners is not None:
                require(actual in native_owners, "Optimizer shard has no native hybrid master")
                state_optimizer, state_parameter = native_owners[actual]
            else:
                state_optimizer, state_parameter = inner.optimizer, actual
            result[name] = {"parameter": actual, "optimizer": inner.optimizer,
                            "state_optimizer": state_optimizer, "state_parameter": state_parameter,
                            "range": (owned.start, owned.end)}
    return result


def canonical_digests(folder, storages, catalog, dtype):
    """Hash full canonical parameters in bounded chunks, independent of topology."""
    import torch
    require(dtype in (torch.float32, torch.bfloat16), "Unsupported initial parameter dtype")
    pieces = []
    for storage in storages:
        if storage["files"]:
            verify_storage(folder, storage)
        else:
            require(not storage["pieces"], "Initial pieces have no storage")
        pieces += storage["pieces"]
    indexed = index_pieces(pieces, sum(math.prod(v["shape"]) for v in catalog.values()))
    require(set(indexed) == set(catalog), "Initial canonical catalog mismatch")
    result = {}
    for name, rows in indexed.items():
        require(rows[0]["shape"] == catalog[name]["shape"], "Initial canonical shape mismatch")
        h = hashlib.sha256()
        for row in rows:
            for offset in range(0, row["elements"], CHUNK_ELEMENTS):
                count = min(CHUNK_ELEMENTS, row["elements"] - offset)
                value = np.fromfile(Path(folder) / row["file"], dtype="<f4", count=count,
                                    offset=row["offset_bytes"] + offset * 4)
                require(value.size == count and np.isfinite(value).all(), "Invalid initial tensor bytes")
                # The temporary format is FP32; BF16 is exactly representable.
                actual = torch.from_numpy(value).to(dtype)
                require(torch.equal(actual.float(), torch.from_numpy(value)), "Initial storage cast lost information")
                h.update(actual.view(torch.uint8).numpy().tobytes())
        result[name] = h.hexdigest()
    return result


def capture_initial(mapping, writer, dtype, *, storage_alignment=False):
    """Verify replicas and stream complete hashes through temporary shared files.

    Only files created by this call are removed, after all ranks verify hashes.
    Initial scratch is bounded by one FP32 model and is separate from gradients.
    """
    import torch
    import torch.distributed as dist
    catalog = writer.plan["parameters"]
    require(all(p.dtype == dtype for p in mapping.parameters.values()), "Initial model dtype differs")
    fragments, owners = collective_fragments(mapping, mapping.parameters, catalog)
    folder = writer.folder / "initial-scratch"
    folder.mkdir(exist_ok=True)
    temporary = ShardWriter(folder, dist.get_rank(), writer.plan["disk_budget_bytes"] // 3,
                            fields=("parameters",))
    for owner in owners:
        run, value = fragments[owner["fragment"]]
        temporary.append("parameters", {"name": run.name, "shape": list(run.shape),
                         "start": run.start, "end": run.end}, torch_chunks(value))
    local = temporary.close()
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = {"hashes": canonical_digests(folder, gathered, catalog, dtype)}
            if storage_alignment:
                # Both casts must be exact, on tensors captured from the actual
                # initialized model, not only from the input weights file.
                result[0]['aligned'] = {str(t): canonical_digests(folder, gathered, catalog, t)
                                        for t in (torch.float32, torch.bfloat16)}
        except Exception as error:
            result[0] = {"error": str(error)}
    dist.broadcast_object_list(result, src=0)
    require("hashes" in result[0], "Initial canonical hash failed: " + result[0].get("error", "unknown"))
    if dist.get_rank() == 0:
        for name, sha in result[0]["hashes"].items():
            writer.initial_parameter(name, str(dtype), sha)
            if storage_alignment:
                writer.initial[name]['storage_alignment'] = {
                    key: values[name] for key, values in result[0]['aligned'].items()}
    # Every process has received the successful full-coverage result.
    for name in local["files"]:
        (folder / name).unlink()
    dist.barrier()
    if dist.get_rank() == 0:
        folder.rmdir()
    return {"parameters": len(catalog), "canonical_digest_dtype": str(dtype),
            "all_replicas_verified": True, "temporary_files_removed": True}


class NativeStepObserver:
    """Wrap one fresh distributed Adam step, restoring hooks even on failure."""
    def __init__(self, mapping, optimizer, writer, precision):
        import torch
        require(precision == {"parameters": precision.get("parameters"), "gradients": "float32",
                "optimizer_parameters": "float32", "optimizer_state": "float32"}
                and precision["parameters"] in ("float32", "bfloat16"), "Unsupported native precision contract")
        expected = getattr(torch, precision["parameters"])
        require(all(p.dtype == expected for p in mapping.parameters.values()), "Observed model dtype differs")
        self.mapping, self.optimizer, self.writer = mapping, optimizer, writer
        self.catalog, self.precision = writer.plan["parameters"], precision
        self.shards = optimizer_shards(optimizer, mapping)
        self.ranges = {n: row["range"] for n, row in self.shards.items()}
        self.before, self.captured, self.updated = {}, False, False
        self.receipt = {"prepare_calls": 0, "optimizer_step_calls": 0, "capture_seconds": 0.0,
                        "gradient_point": "after_reduction_before_clipping", "optimizer_state_steps": []}
        self.installed = False
        self.step_number = 1

    def _masters(self):
        return {n: master_parameter(row["state_optimizer"], row["state_parameter"]) for n, row in self.shards.items()}

    def _validate_before(self):
        import torch
        for n, row in self.shards.items():
            state = row["state_optimizer"].state.get(row["state_parameter"], {})
            require(adam_counter(row) == 0, "Optimizer did not start fresh")
            for key in ("exp_avg", "exp_avg_sq"):
                require(key not in state or bool(torch.count_nonzero(state[key]) == 0), "Fresh Adam has nonzero moments")

    def __enter__(self):
        import torch
        require(not self.installed and not self.before and not self.captured, "Observer cannot be reused")
        self._validate_before()
        for n, value in self._masters().items():
            first, last = self.ranges[n]
            initial = value.detach().to(device="cpu", dtype=torch.float32, copy=True).reshape(-1)
            expected = self.mapping.parameters[n].detach().reshape(-1)[first:last].float().cpu()
            if self.step_number == 1:
                require(torch.equal(initial, expected), "Initial FP32 master differs from the loaded model")
            else:
                require(torch.equal(initial.to(self.mapping.parameters[n].dtype).float(), expected),
                        "FP32 master does not round to the current model parameter")
            self.before[n] = initial
        self.original_prepare, self.original_step = self.optimizer.prepare_grads, self.optimizer.step
        self.optimizer.prepare_grads, self.optimizer.step = self._prepare, self._step
        self.installed = True
        return self

    def _prepare(self, *args, **kwargs):
        import torch
        result = self.original_prepare(*args, **kwargs)
        self.receipt["prepare_calls"] += 1
        require(not result, "Native optimizer found nonfinite gradients")
        require(not self.updated, "Gradient preparation occurred after update")
        if not self.captured:
            started = time.monotonic()
            values = {}
            for n, row in self.shards.items():
                p = row["parameter"]
                grad = getattr(p, "decoupled_grad", None)
                if grad is None:
                    grad = p.grad
                require(grad is not None and grad.dtype == torch.float32, "Missing/non-FP32 native optimizer gradient")
                values[n] = grad
            fragments, owners = collective_fragments(self.mapping, values, self.catalog, self.ranges)
            append_owned(self.writer, "gradients", fragments, owners)
            self.captured = True
            self.receipt["capture_seconds"] += time.monotonic() - started
        return result

    def _step(self, *args, **kwargs):
        import torch
        require(self.receipt["optimizer_step_calls"] == 0, "Expected exactly one optimizer step")
        self.receipt["optimizer_step_calls"] += 1
        result = self.original_step(*args, **kwargs)
        require(isinstance(result, tuple) and len(result) == 3 and bool(result[0]), "Native optimizer step failed")
        require(self.captured, "Native step did not call prepare_grads")
        started = time.monotonic()
        values = {}
        for n, master in self._masters().items():
            row = self.shards[n]
            state = row["state_optimizer"].state[row["state_parameter"]]
            for key in ("exp_avg", "exp_avg_sq"):
                require(key in state and state[key].dtype == torch.float32 and bool(torch.isfinite(state[key]).all()),
                        "Missing/nonfinite/non-FP32 native Adam moments")
            counter = adam_counter(row)
            require(counter == self.step_number, "Native Adam step counter differs")
            self.receipt["optimizer_state_steps"].append(counter)
            values[n] = master.detach().float().cpu().reshape(-1) - self.before.pop(n)
        fragments, owners = collective_fragments(self.mapping, values, self.catalog, self.ranges)
        append_owned(self.writer, "updates", fragments, owners)
        self.receipt["capture_seconds"] += time.monotonic() - started
        self.updated = True
        return result

    def __exit__(self, exc_type, exc, traceback):
        self.optimizer.prepare_grads, self.optimizer.step = self.original_prepare, self.original_step
        self.installed = False
        self.before.clear()
        if exc_type is None:
            require(self.captured and self.updated and self.receipt["optimizer_step_calls"] == 1,
                    "Native optimizer step was not completely observed")
