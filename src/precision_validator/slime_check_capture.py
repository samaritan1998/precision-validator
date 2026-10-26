"""Per-rank observation around Slime's actual train_one_step, never a second step."""
import functools
import math
from pathlib import Path
import time

from .common import require, save, object_hash
from .slime_check import load_manifest

_sessions = {}


def session_key(args):
    return (str(args.precision_check_manifest), args.precision_check_stage)


def reached_limit(args):
    current = _sessions.get(session_key(args))
    return current is not None and current['completed'] >= args.precision_check_steps


def value_identity(value):
    """Hash the actual tensors returned by DataIterator before CP slicing."""
    import torch
    from .sdk import tensor_digest
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': tensor_digest(value)}
    if isinstance(value, dict):
        return {k: value_identity(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [value_identity(v) for v in value]
    require(value is None or type(value) in (str, int, float, bool), 'Unsupported input metadata type')
    if type(value) is float:
        require(math.isfinite(value), 'Nonfinite input metadata')
    return value


class InputTrace:
    def __init__(self, iterator):
        self.iterator, self.rows = iterator, []

    def __enter__(self):
        self.original = self.iterator.get_next
        self.iterator.get_next = self.get_next
        return self

    def get_next(self, keys):
        import torch
        batch = self.original(keys)
        require(batch.get('tokens') is not None and batch.get('loss_masks') is not None,
                'SFT trace requires actual tokens and loss_masks')
        size = len(batch['tokens'])
        for i in range(size):
            fields = {}
            for key, values in batch.items():
                # Packed metadata is generated after this boundary. Reject MM
                # rather than pretending the text-only signature covers it.
                if key == 'multimodal_train_inputs':
                    require(values is None or all(v is None for v in values), 'Multimodal SFT is not yet supported')
                    continue
                if values is not None:
                    require(isinstance(values, (list, tuple)) and len(values) == size,
                            f'Unexpected DataIterator sample field: {key}')
                    fields[key] = value_identity(values[i])
            mask = torch.as_tensor(batch['loss_masks'][i])
            require(bool(((mask == 0) | (mask == 1)).all()), 'First bridge requires binary supervision masks')
            self.rows.append({'sha256': object_hash(fields), 'supervised_tokens': int(mask.sum()),
                              'tokens': int(torch.as_tensor(batch['tokens'][i]).numel())})
        return batch

    def __exit__(self, *exc):
        self.iterator.get_next = self.original


def global_inputs(rows, global_batch_size):
    """Replica consistency first, then one copy per dense data-parallel shard."""
    by_dp = {}
    for row in rows:
        value = sorted(row['inputs'], key=lambda r: (r['sha256'], r['tokens'], r['supervised_tokens']))
        require(by_dp.setdefault(row['dp_rank'], value) == value,
                'TP/CP/PP replicas consumed different input contributions')
    samples = sorted((item for local in by_dp.values() for item in local), key=lambda r: (r['sha256'], r['tokens'], r['supervised_tokens']))
    require(len(samples) == global_batch_size and samples, 'Actual global SFT samples differ from step batch size')
    require(sum(s['supervised_tokens'] for s in samples) > 0, 'No supervised tokens in captured step')
    return {'global_samples': len(samples), 'supervised_tokens': sum(s['supervised_tokens'] for s in samples),
            'tokens': sum(s['tokens'] for s in samples), 'contributions_sha256': object_hash(samples)}


def optimizer_options(optimizer):
    keys = ('lr', 'betas', 'eps', 'weight_decay', 'adam_w_mode', 'maximize')
    return sorted({object_hash(value): value for group in optimizer.param_groups
                   for value in [{key: value_identity(group[key]) for key in keys if key in group}]}.values(),
                  key=object_hash)


def observed_step(function, args, rollout_id, step_id, data_iterator, model,
                  optimizer, scheduler, num_microbatches, global_batch_size, **kwargs):
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.utils import unwrap_model
    from .acceptance import validate_acceptance
    from .megatron_mapping import MegatronParameterMap, native_groups
    from .megatron_observer import NativeStepObserver, capture_initial
    from .native_trajectory_sdk import adam_counter
    from .sdk import RankWriter, runtime_info
    from .storage import ShardWriter

    require(len(model) == len(data_iterator) == 1, 'First bridge supports one model chunk, without virtual pipeline')
    require(dist.is_initialized() and torch.cuda.is_available(), 'Observe inside the real CUDA training rank')
    key = session_key(args)
    if key not in _sessions:
        manifest = load_manifest(args.precision_check_manifest)
        rank = dist.get_rank()
        require(dist.get_world_size() == manifest['world_size'], 'Observed world size differs')
        module = unwrap_model(model)[0]
        mapping = MegatronParameterMap(module)
        local = [None] * dist.get_world_size()
        dist.all_gather_object(local, mapping.catalog())
        catalog = {}
        for part in local:
            for name, info in part.items():
                require(catalog.setdefault(name, info) == info, 'Canonical parameter metadata differs across ranks')
        n = sum(math.prod(v['shape']) for v in catalog.values())
        # Three stages, two full FP32 fields per step + initial scratch.
        required_bytes = int(n * 4 * (6 * manifest['steps'] + 2) * 1.1)
        require(required_bytes <= manifest['disk_budget_bytes'],
                f'Full capture needs at least {required_bytes / 2**30:.2f} GiB; increase --precision-check-disk-gib')
        plan = {**manifest, 'parameters': catalog, 'policy': manifest['rules']['policy']}
        if manifest['rules']['acceptance'] is not None:
            plan['acceptance'] = manifest['rules']['acceptance']
            validate_acceptance(plan)
        dtypes = {p.dtype for p in mapping.parameters.values()}
        require(len(dtypes) == 1 and next(iter(dtypes)) in (torch.float32, torch.bfloat16), 'Mixed/unsupported parameter storage')
        precision = {'parameters': str(next(iter(dtypes))).removeprefix('torch.'), 'gradients': 'float32',
                     'optimizer_parameters': 'float32', 'optimizer_state': 'float32'}
        require(all(chunk.ddp_config.grad_reduce_in_fp32 for chunk in model), 'Gradient reduction must be FP32')
        _sessions[key] = {'manifest': manifest, 'plan': plan, 'mapping': mapping, 'precision': precision,
                          'completed': 0, 'optimizer_id': id(optimizer), 'model_id': id(module), 'rank': rank,
                          'groups': native_groups(manifest['configurations'][args.precision_check_stage])}
    state = _sessions[key]
    require(state['optimizer_id'] == id(optimizer) and state['model_id'] == id(unwrap_model(model)[0]),
            'Model or optimizer was replaced within a precision check')
    number = state['completed'] + 1
    require(number <= args.precision_check_steps, 'Requested optimizer-step limit was exceeded')
    manifest, plan, rank = state['manifest'], state['plan'], state['rank']
    folder = Path(manifest['output']) / args.precision_check_stage / 'steps' / f'{number:06d}'
    folder.mkdir(parents=True, exist_ok=True)

    class Writer(RankWriter):
        def __init__(self):
            self.plan, self.stage, self.rank, self.folder = plan, args.precision_check_stage, rank, folder
            self.storage = ShardWriter(folder, rank, manifest['disk_budget_bytes'] // (3 * manifest['steps']))
            self.initial, self.auxiliary, self.observations = {}, {}, {}

    class Observer(NativeStepObserver):
        def _validate_before(self):
            if number == 1:
                super()._validate_before()
            for row in self.shards.values():
                require(adam_counter(row) == number - 1, 'Native optimizer counter differs from captured step')
                if number > 1:
                    native = row['state_optimizer'].state[row['state_parameter']]
                    for name in ('exp_avg', 'exp_avg_sq'):
                        require(name in native and native[name].dtype == torch.float32 and bool(torch.isfinite(native[name]).all()),
                                'Missing/nonfinite/non-FP32 Adam state')

    writer = Writer()
    if number == 1:
        capture_initial(state['mapping'], writer, getattr(torch, state['precision']['parameters']))
        state['initial'] = writer.initial
        # Stateful auxiliary buffers need a canonical map too. Do not silently
        # certify them with parameter-only coverage.
        buffers = list(unwrap_model(model)[0].named_buffers())
        require(not buffers, 'Stateful model buffers need explicit canonical mapping; unsupported by this bridge')
    options = optimizer_options(optimizer)
    lr_before = [float(g['lr']) for g in optimizer.param_groups]
    observer = Observer(state['mapping'], optimizer, writer, state['precision'])
    observer.step_number = number
    started = time.monotonic()
    with InputTrace(data_iterator[0]) as trace, observer:
        result = function(args, rollout_id, step_id, data_iterator, model, optimizer,
                          scheduler, num_microbatches, global_batch_size, **kwargs)
    require(observer.updated and math.isfinite(float(result[1])), 'Update failed or gradient norm is nonfinite')
    local = {'rank': rank, 'dp_rank': ps.get_data_parallel_rank(), 'inputs': trace.rows,
             'last': ps.is_pipeline_last_stage(), 'loss': float(result[0]['loss']) if ps.is_pipeline_last_stage() else None}
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, local)
    objective = global_inputs(rows, global_batch_size)
    finals = [row['loss'] for row in rows if row['last']]
    require(finals and all(math.isfinite(v) and v == finals[0] for v in finals), 'Reduced loss replicas differ')
    if rank == 0:
        writer.observe('loss', finals[0])
    row = {'manifest_sha256': manifest['sha256'], 'stage': args.precision_check_stage, 'step': number,
           'rank': rank, 'rollout_id': rollout_id, 'step_id_in_rollout': step_id,
           'catalog': plan['parameters'], 'initial': state['initial'] if number == 1 else {},
           'storage': writer.storage.close(), 'observations': writer.observations, 'objective': objective,
           'normalization': {'loss_type': args.loss_type, 'calculate_per_token_loss': args.calculate_per_token_loss,
                             'global_batch_size': global_batch_size},
           'optimizer_options': options, 'lr_before': lr_before, 'scheduler_steps': scheduler.num_steps,
           'optimizer_updated': observer.updated, 'observer': observer.receipt,
           'precision': state['precision'], 'native_parallelism': state['groups'],
           'runtime': runtime_info(), 'device': torch.cuda.get_device_name(),
           'elapsed_seconds': time.monotonic() - started}
    path = folder / f'rank{rank}.json'
    require(not path.exists(), 'Step receipt already exists')
    save(path, row)
    dist.barrier()
    state['completed'] = number
    if rank == 0:
        print(f'[precision-check] {args.precision_check_stage}: step {number}/{args.precision_check_steps} captured', flush=True)
    return result
