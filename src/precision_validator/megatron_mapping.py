"""Canonical tensor slices from native Megatron checkpoint metadata.

This module observes storage views. It does not replace a model operation or
guess sharding from parameter names. Unknown/copying factories fail closed.
The caller supplies gradients after native reduction and actual optimizer
parameter deltas. A distributed optimizer may supply a local flattened range.
"""
from dataclasses import dataclass
import hashlib
import itertools
import math

from .common import require
from .parallel import CONVENTION, validate_layout
from .storage import CHUNK_ELEMENTS, index_pieces


@dataclass(frozen=True)
class Run:
    name: str
    shape: tuple
    start: int
    end: int
    local_start: int

    @property
    def local_end(self):
        return self.local_start + self.end - self.start

    def intersect(self, start, end):
        left, right = max(start, self.local_start), min(end, self.local_end)
        if left >= right:
            return None
        first = self.start + left - self.local_start
        return Run(self.name, self.shape, first, first + right - left, left)


def contiguous_runs(key, shape, offset, local_shape, strides, storage_offset,
                    prepend=0, max_runs=1000000):
    """Map an arbitrary positive-stride local view into global row-major order."""
    shape, offset, local_shape, strides = map(tuple, (shape, offset, local_shape, strides))
    require(len(shape) == len(offset) == len(local_shape) + prepend,
            "Inconsistent native shard dimensions")
    require(len(strides) == len(local_shape) and all(n > 0 for n in shape + local_shape)
            and all(n >= 0 for n in strides) and storage_offset >= 0, "Invalid native storage geometry")
    require(all(0 <= off < size for off, size in zip(offset, shape)), "Native shard offset out of bounds")
    require(all(off + n <= size for off, n, size in zip(offset[prepend:], local_shape, shape[prepend:])),
            "Native shard extends outside global tensor")
    # Split prepended layer/expert axes into independent acceptance parameters.
    name = key + ("::index=" + ",".join(map(str, offset[:prepend])) if prepend else "")
    shape, offset = shape[prepend:], offset[prepend:]
    global_strides = tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))
    suffix, block = len(local_shape), 1
    while suffix and strides[suffix - 1] == block and global_strides[suffix - 1] == block:
        suffix -= 1
        block *= local_shape[suffix]
    count = math.prod(local_shape[:suffix])
    require(count <= max_runs, "Native layout exceeds the bounded mapping size; add an explicit compact adapter")
    base = sum(o * s for o, s in zip(offset, global_strides))
    runs = []
    for coordinates in itertools.product(*(range(n) for n in local_shape[:suffix])):
        start = base + sum(i * s for i, s in zip(coordinates, global_strides))
        local = storage_offset + sum(i * s for i, s in zip(coordinates, strides))
        value = Run(name, shape, start, start + block, local)
        if runs and runs[-1].name == name and runs[-1].end == start and runs[-1].local_end == local:
            previous = runs.pop()
            value = Run(name, shape, previous.start, value.end, previous.local_start)
        runs.append(value)
    return runs


def _leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


class MegatronParameterMap:
    """One local trainable parameter maps to one or more canonical slices."""
    def __init__(self, model):
        from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
        from megatron.core.dist_checkpointing.optimizer import get_param_id_to_sharded_param_map
        self.parameters = dict(model.named_parameters())
        self.parameters = {n: p for n, p in self.parameters.items() if p.requires_grad}
        require(self.parameters, "No trainable Megatron parameters")
        # Native singleton metadata provides global PP layer and EP expert names,
        # and splits GLU halves before their independent TP row partitions.
        state = model.sharded_state_dict(metadata={"singleton_local_shards": True})
        by_index = get_param_id_to_sharded_param_map(state, self.parameters.values())
        require(set(by_index) == set(range(len(self.parameters))),
                "Native checkpoint metadata does not cover every trainable parameter")
        self.runs = {}
        self.factories = 0
        for index, (local_name, parameter) in enumerate(self.parameters.items()):
            require(parameter.is_contiguous(), "Noncontiguous source parameter needs an explicit adapter")
            native = by_index[index]
            require(native.data is parameter, "Native checkpoint parameter identity changed")
            if isinstance(native, ShardedTensorFactory):
                self.factories += 1
                leaves = list(_leaves(native.build()))
            else:
                leaves = [native]
            runs = []
            for leaf in leaves:
                require(isinstance(leaf, ShardedTensor) and leaf.flattened_range is None,
                        "Unsupported native checkpoint factory output")
                value = leaf.data
                require(value is not None and value.dtype == parameter.dtype
                        and value.untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr(),
                        "Copying checkpoint factory needs an explicit canonical mapping adapter")
                runs += contiguous_runs(leaf.key, leaf.global_shape, leaf.global_offset, leaf.local_shape,
                                        value.stride(), value.storage_offset() - parameter.storage_offset(),
                                        leaf.prepend_axis_num)
            cursor = 0
            for run in sorted(runs, key=lambda r: r.local_start):
                require(run.local_start == cursor, f"Missing/duplicated local parameter coverage: {local_name}")
                cursor = run.local_end
            require(cursor == parameter.numel(), f"Incomplete local parameter coverage: {local_name}")
            self.runs[local_name] = runs

    def catalog(self):
        result = {}
        for runs in self.runs.values():
            for run in runs:
                item = {"shape": list(run.shape), "group": run.name.rsplit(".", 1)[0]}
                require(result.setdefault(run.name, item) == item, "Native canonical shapes disagree")
        return result

    def fragments(self, values, ranges=None):
        """Return views of complete local tensors or owned flat optimizer ranges.

        values is keyed by local model parameter name. With ranges, only owned
        parameters appear and values contain exactly end-start elements.
        """
        require(set(values) <= set(self.parameters), "Unknown local optimizer parameters")
        if ranges is None:
            require(set(values) == set(self.parameters), "Missing local model parameters")
            ranges = {n: (0, self.parameters[n].numel()) for n in values}
        require(set(values) == set(ranges), "Optimizer values/ranges differ")
        result = []
        for name, value in values.items():
            first, last = ranges[name]
            require(0 <= first < last <= self.parameters[name].numel(), "Optimizer range outside local parameter")
            require(value.numel() == last - first, "Optimizer range length differs from observed tensor")
            flat = value.detach().reshape(-1)
            for run in self.runs[name]:
                part = run.intersect(first, last)
                if part is not None:
                    result.append((part, flat[part.local_start - first:part.local_end - first]))
        return result


def fragment_metadata(fragments, rank):
    """Hash actual values before ownership election, including all replicas."""
    import torch
    records = []
    for index, (run, value) in enumerate(fragments):
        require(value.dtype in (torch.float32, torch.bfloat16), "Unsupported observed tensor dtype")
        h = hashlib.sha256()
        for start in range(0, value.numel(), CHUNK_ELEMENTS):
            part = value[start:start + CHUNK_ELEMENTS].cpu().contiguous()
            require(bool(torch.isfinite(part).all()), "Nonfinite canonical fragment")
            h.update(part.view(torch.uint8).numpy().tobytes())
        records.append({"name": run.name, "shape": list(run.shape), "start": run.start, "end": run.end,
                        "elements": run.end - run.start, "offset_bytes": 0, "dtype": str(value.dtype),
                        "sha256": h.hexdigest(), "rank": rank, "fragment": index})
    return records


def elect_owners(records, catalog):
    """Require equal replicas, then select exactly one complete global cover."""
    unique, seen = {}, set()
    for row in records:
        require(row["name"] in catalog and row["shape"] == catalog[row["name"]]["shape"], "Canonical catalog mismatch")
        key = row["name"], row["start"], row["end"]
        require((key, row["rank"]) not in seen, "Duplicate local canonical interval")
        seen.add((key, row["rank"]))
        if key in unique:
            previous = unique[key]
            require(row["rank"] != previous["rank"], "Duplicate local canonical interval")
            require((row["dtype"], row["sha256"]) == (previous["dtype"], previous["sha256"]),
                    f"Native parameter replicas disagree: {row['name']}")
            if row["rank"] < previous["rank"]:
                unique[key] = row
        else:
            unique[key] = row
    owners = list(unique.values())
    indexed = index_pieces(owners, sum(math.prod(p["shape"]) for p in catalog.values()))
    require(set(indexed) == set(catalog), "Missing canonical parameters")
    return owners


def collective_fragments(mapping, values, catalog, ranges=None):
    import torch.distributed as dist
    fragments = mapping.fragments(values, ranges)
    local = fragment_metadata(fragments, dist.get_rank())
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    owners = elect_owners([r for rows in gathered for r in rows], catalog)
    return fragments, [r for r in owners if r["rank"] == dist.get_rank()]


def append_owned(writer, field, fragments, owners):
    from .sdk import torch_chunks
    for owner in owners:
        run, value = fragments[owner["fragment"]]
        writer.append(field, run.name, run.start, run.end, torch_chunks(value))


def native_groups(layout):
    """Read groups already created by Megatron; do not synthesize memberships."""
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    sizes = validate_layout(layout, dist.get_world_size())
    groups = {"tp": ps.get_tensor_model_parallel_group(), "pp": ps.get_pipeline_model_parallel_group(),
              "cp": ps.get_context_parallel_group(), "dp": ps.get_data_parallel_group(),
              "etp": ps.get_expert_tensor_parallel_group(), "ep": ps.get_expert_model_parallel_group(),
              "edp": ps.get_expert_data_parallel_group()}
    members = {axis: dist.get_process_group_ranks(group) for axis, group in groups.items()}
    require(all(len(members[axis]) == size for axis, size in sizes.items()), "Native group sizes differ from contract")
    return {"convention": CONVENTION, "degrees": sizes, "groups": members}
