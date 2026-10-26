"""Observe native distributed Adam and checkpoint it without replacing training math."""
import os
from pathlib import Path
import uuid

from .common import require, save, read, digest, object_hash
from .megatron_observer import NativeStepObserver, optimizer_shards, master_parameter
from .megatron_mapping import collective_fragments, append_owned
from .sdk import RankWriter, runtime_info
from .storage import ShardWriter
from .trajectory_sdk import describe, rng_state, restore_rng, cursor_for
from .native_trajectory import FIELDS, identity


def optimizer_classes(optimizer):
    """Resolve class-valued native options only from the already constructed optimizer."""
    import torch
    result = {}
    for group in optimizer.param_groups:
        for key in ('cpu_optimizer_cls', 'gpu_optimizer_cls'):
            value = group.get(key)
            if value is not None:
                require(isinstance(value, type) and issubclass(value, torch.optim.Optimizer), 'Unexpected native optimizer class option')
                result[value.__module__ + '.' + value.__qualname__] = value
    return result


def class_metadata(value, classes, *, restore=False):
    """Keep checkpoints weights_only-loadable; never import classes from a checkpoint."""
    if isinstance(value, dict):
        if restore and set(value) == {'__native_optimizer_class__'}:
            name = value['__native_optimizer_class__']
            require(name in classes, 'Checkpoint optimizer class is not present in the current runtime')
            return classes[name]
        return {k: class_metadata(v, classes, restore=restore) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(class_metadata(v, classes, restore=restore) for v in value)
    if isinstance(value, type):
        name = value.__module__ + '.' + value.__qualname__
        require(not restore and classes.get(name) is value, 'Undeclared optimizer class metadata')
        return {'__native_optimizer_class__': name}
    return value


def adam_counter(row):
    optimizer, parameter = row['state_optimizer'], row['state_parameter']
    state = optimizer.state.get(parameter, {})
    if 'step' in state:
        return float(state['step'])
    groups = [g for g in optimizer.param_groups if any(p is parameter for p in g['params'])]
    require(len(groups) == 1, 'Native Adam parameter group ownership differs')
    return float(groups[0].get('step', 0))


class NativeTrajectoryObserver(NativeStepObserver):
    def __init__(self, mapping, optimizer, writer, precision, step):
        require(writer.plan['adapter'] == 'native-trajectory-v1', 'Use a native trajectory contract')
        require(type(step) is int and 1 <= step <= writer.plan['contract']['optimizer_steps'], 'Invalid native step')
        super().__init__(mapping, optimizer, writer, precision)
        self.step_number = step

    def _validate_before(self):
        import torch
        if self.step_number == 1:
            super()._validate_before()
        for row in self.shards.values():
            require(adam_counter(row) == self.step_number - 1, 'Native Adam prior step counter differs')
            if self.step_number > 1:
                state = row['state_optimizer'].state[row['state_parameter']]
                for key in ('exp_avg', 'exp_avg_sq'):
                    require(key in state and state[key].dtype == torch.float32 and
                            state[key].shape == row['state_parameter'].shape and bool(torch.isfinite(state[key]).all()),
                            'Missing/nonfinite/non-FP32 accumulated Adam state')

    def _step(self, *args, **kwargs):
        result = super()._step(*args, **kwargs)
        require(all(adam_counter(r) == self.step_number for r in self.shards.values()), 'Native Adam counter did not advance')
        for field in ('master_parameters', 'exp_avg', 'exp_avg_sq'):
            values = (self._masters() if field == 'master_parameters' else
                      {n: r['state_optimizer'].state[r['state_parameter']][field] for n, r in self.shards.items()})
            fragments, owners = collective_fragments(self.mapping, values, self.catalog, self.ranges)
            append_owned(self.writer, field, fragments, owners)
        fragments, owners = collective_fragments(self.mapping, self.mapping.parameters, self.catalog)
        append_owned(self.writer, 'parameters', fragments, owners)
        self.receipt['optimizer_state_steps'] = [adam_counter(r) for r in self.shards.values()]
        return result


class StepWriter(RankWriter):
    def __init__(self, plan, stage, rank, step):
        self.plan, self.stage, self.rank, self.step = plan, stage, rank, step
        self.folder = Path(plan['output']) / stage / 'steps' / f'{step:06d}'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.storage = ShardWriter(self.folder, rank, plan['disk_budget_bytes'] // (4 * plan['contract']['optimizer_steps']), fields=FIELDS)
        self.initial, self.auxiliary, self.observations = {}, {}, {}

    def seal(self, state, lr_used, checks, observer, timings):
        frozen = self.plan['trajectory']['steps'][self.step - 1]
        row = {**identity(self.plan, self.stage, self.rank), 'step': self.step,
               'input_sha256': self.plan['artifacts'][frozen['input_artifact']]['sha256'],
               'objective': frozen['objective'], 'storage': self.storage.close(), 'observations': self.observations,
               'state': state, 'lr_used': lr_used, 'checks': checks, 'observer': observer, 'timings': timings}
        path = self.folder / f'rank{self.rank}.json'
        require(not path.exists(), 'Native step receipt already exists')
        save(path, row)
        return row

    def observe(self, name, values):
        import numpy as np
        expected = self.plan['trajectory']['steps'][self.step - 1]['observables']
        require(name in expected and name not in self.observations, 'Unknown/duplicate step observation')
        value = np.asarray(values, dtype=np.float64)
        require(list(value.shape) == expected[name] and np.isfinite(value).all(), 'Invalid native step observation')
        self.observations[name] = {'shape': list(value.shape), 'values': value.reshape(-1).tolist()}


def native_rng():
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    return {'standard': rng_state(), 'megatron': get_cuda_rng_tracker().get_states()}


def restore_native_rng(value):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    restore_rng(value['standard'])
    get_cuda_rng_tracker().set_states(value['megatron'])


def state_signature(model, mapping, optimizer, scheduler, plan, completed):
    shards = optimizer_shards(optimizer, mapping)
    states = {}
    for name, row in shards.items():
        state = row['state_optimizer'].state.get(row['state_parameter'], {})
        states[name] = {'range': list(row['range']), 'step': adam_counter(row),
                       'master': describe(master_parameter(row['state_optimizer'], row['state_parameter'])),
                       **{k: describe(state[k]) for k in ('exp_avg', 'exp_avg_sq') if k in state}}
    return {'model': describe(model.state_dict()), 'optimizer_shards': states,
            # Megatron adds a redundant group-level step during checkpoint
            # loading (Hybrid normally keeps it per parameter). Counters are
            # compared separately for every actual owned Adam parameter above.
            'optimizer_groups': describe(class_metadata([{k: v for k, v in g.items() if k not in ('params', 'step')}
                                                         for g in optimizer.param_groups], optimizer_classes(optimizer))),
            'scheduler': describe(scheduler.state_dict()), 'rng': describe(native_rng()),
            'data_cursor': cursor_for(plan, completed)}


def save_checkpoint(model, mapping, optimizer, scheduler, plan, rank, initial, process_id):
    """Use native DP-zero optimizer checkpoint interfaces, keeping topology fixed."""
    import torch
    import torch.distributed as dist
    folder = Path(plan['output']) / 'resume'
    boundary = plan['trajectory']['resume_after_step']
    before = state_signature(model, mapping, optimizer, scheduler, plan, boundary)
    inners = getattr(optimizer, 'chained_optimizers', [optimizer])
    blob = {'plan_sha256': plan['plan_sha256'], 'rank': rank, 'configuration': plan['configurations']['resume'],
            'cursor': cursor_for(plan, boundary), 'model': model.state_dict(),
            'optimizer': class_metadata(optimizer.state_dict(), optimizer_classes(optimizer)),
            'parameter_state': [inner.get_parameter_state_dp_zero() for inner in inners],
            'scheduler': scheduler.state_dict(), 'rng': native_rng()}
    path = folder / f'checkpoint.rank{rank}.pt'
    require(not path.exists(), 'Checkpoint already exists')
    with path.with_suffix('.pt.partial').open('xb') as stream:
        torch.save(blob, stream)
    path.with_suffix('.pt.partial').replace(path)
    after = state_signature(model, mapping, optimizer, scheduler, plan, boundary)
    require(before == after, 'Native checkpoint save changed training state')
    record = {**identity(plan, 'resume', rank), 'cursor': blob['cursor'], 'state': after, 'initial': initial,
              'producer_pid': os.getpid(), 'producer_process_id': process_id, 'sha256': digest(path), 'bytes': path.stat().st_size}
    save(folder / f'checkpoint.rank{rank}.json', record)
    dist.barrier()
    return record


def restore_checkpoint(model, mapping, optimizer, scheduler, plan, rank, process_id):
    import torch
    import torch.distributed as dist
    folder = Path(plan['output']) / 'resume'
    path = folder / f'checkpoint.rank{rank}.pt'
    record = read(folder / f'checkpoint.rank{rank}.json')
    require(all(record.get(k) == v for k, v in identity(plan, 'resume', rank).items()), 'Checkpoint identity differs')
    require(digest(path) == record['sha256'] and path.stat().st_size == record['bytes'], 'Checkpoint checksum differs')
    require(record['producer_process_id'] != process_id and record['producer_pid'] != os.getpid(), 'Resume needs a fresh worker')
    require(record['cursor'] == cursor_for(plan, plan['trajectory']['resume_after_step']), 'Checkpoint cursor differs')
    blob = torch.load(path, map_location='cpu', weights_only=True)
    require(blob['plan_sha256'] == plan['plan_sha256'] and blob['rank'] == rank and
            blob['configuration'] == plan['configurations']['resume'] and blob['cursor'] == record['cursor'],
            'Checkpoint payload differs')
    # Native DP-zero restore writes leaf parameter views in place. This is
    # checkpoint loading, outside the training autograd graph.
    with torch.no_grad():
        model.load_state_dict(blob['model'], strict=True)
        optimizer.load_state_dict(class_metadata(blob['optimizer'], optimizer_classes(optimizer), restore=True))
        inners = getattr(optimizer, 'chained_optimizers', [optimizer])
        require(len(inners) == len(blob['parameter_state']), 'Optimizer checkpoint chain differs')
        for inner, state in zip(inners, blob['parameter_state']):
            inner.load_parameter_state_from_dp_zero(state)
            if hasattr(inner.optimizer, 'param_to_inner_param'):
                # DP-zero loading replaces Hybrid's outer state entries. Its
                # native load_state_dict post-hook synchronizes actual CPU
                # masters/sub-optimizers; reading only outer metadata is unsafe.
                inner.optimizer.load_state_dict(inner.optimizer.state_dict())
        # Native Hybrid initialization may perform a dummy update to allocate
        # state. Restore the saved model after its asynchronous copies finish.
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        model.load_state_dict(blob['model'], strict=True)
    scheduler.load_state_dict(blob['scheduler'])
    restore_native_rng(blob['rng'])
    restored = state_signature(model, mapping, optimizer, scheduler, plan, plan['trajectory']['resume_after_step'])
    if restored != record['state']:
        save(folder / f'restore-failure.rank{rank}.json', {'expected': record['state'], 'actual': restored})
    require(restored == record['state'], 'Restored model/master/Adam/scheduler/RNG/cursor differs')
    dist.barrier()
    return record, {'checkpoint_sha256': record['sha256'], 'checkpoint_record_sha256': digest(folder / f'checkpoint.rank{rank}.json'),
                    'producer_pid': record['producer_pid'], 'consumer_pid': os.getpid(),
                    'producer_process_id': record['producer_process_id'], 'consumer_process_id': process_id, 'restored_state': restored}
