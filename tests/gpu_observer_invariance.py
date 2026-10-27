"""Compare native rank-local training state with precision capture disabled/enabled.

Uses the generated Slime matrix fixture. No canonical parameter mapper is used
by this independent recorder: physical rank, native optimizer group and parameter
index must match because each pair has exactly the same parallel configuration.
Hash equality is deliberately stricter than numerical closeness. This test is
separate from model-parity acceptance and cannot turn a failed topology into a pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import zipfile

import gpu_slime_matrix as fixture
from precision_validator.common import digest, now, read, require, save

MODES = ('disabled', 'disabled_repeat', 'enabled')


def instrument(output):
    def install(module, optimizer, training):
        import torch
        import torch.distributed as dist

        def encode(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().contiguous().cpu()
                require(not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all()),
                        'Nonfinite native state in invariance test')
                raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
                return dict(dtype=str(value.dtype), shape=list(value.shape), elements=value.numel(),
                            sha256=hashlib.sha256(raw).hexdigest())
            if isinstance(value, type):
                return {'class_reference': value.__module__+'.'+value.__qualname__}
            if isinstance(value, (torch.dtype, torch.device)):
                return {'torch_metadata': str(value)}
            if isinstance(value, dict):
                require(all(type(key) in (str, int) for key in value), 'Unexpected native state key')
                return {str(key): encode(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            require(value is None or type(value) in (str, int, float, bool),
                    f'Unsupported native state: {type(value).__name__}')
            return value

        def states(gradients=False):
            result = {}
            for i, outer in enumerate(getattr(optimizer, 'chained_optimizers', [optimizer])):
                if getattr(outer, 'is_stub_optimizer', False):
                    continue
                inner = outer.optimizer
                # Inspect the actual owners for CPU-offloaded Adam, including
                # its FP32 parameter, instead of the Hybrid metadata aliases.
                owners = list(inner.sub_optimizers) if hasattr(inner, 'sub_optimizers') else [inner]
                for j, owner in enumerate(owners):
                    for g, group in enumerate(owner.param_groups):
                        prefix = f'{i}/{j}/{g}'
                        if not gradients:
                            result[prefix+'/options'] = encode({k:v for k,v in group.items() if k != 'params'})
                        for p, parameter in enumerate(group['params']):
                            key = f'{prefix}/{p}'
                            if gradients:
                                grad = getattr(parameter, 'decoupled_grad', None)
                                if grad is None:
                                    grad = parameter.grad
                                # CPU offload transfers gradients during the
                                # Hybrid step; the native outer shard remains
                                # the source at prepare_grads for that case.
                                result[key+'/gradient'] = encode(grad)
                            else:
                                result[key+'/parameter'] = encode(parameter)
                                result[key+'/state'] = encode(owner.state.get(parameter, {}))
                if gradients and hasattr(inner, 'sub_optimizers'):
                    for g, group in enumerate(inner.param_groups):
                        for p, parameter in enumerate(group['params']):
                            grad = getattr(parameter, 'decoupled_grad', None)
                            if grad is None:
                                grad = parameter.grad
                            require(grad is not None, 'Missing native source gradient for CPU offload')
                            result[f'{i}/hybrid/{g}/{p}/gradient'] = encode(grad)
            require(result, 'Empty native optimizer record')
            if gradients:
                require(any(row is not None for row in result.values()), 'No native gradients captured')
            return result

        def model_state():
            return {name:encode(p) for name,p in module.named_parameters()}

        initial = model_state()
        rows, prepared = [], []
        original_prepare = optimizer.prepare_grads
        original_step = training.train_one_step

        def prepare(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            require(not result, 'Native prepare_grads found nonfinite gradients')
            prepared.append(states(gradients=True))
            return result

        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            require(len(prepared) == len(rows)+1, 'Expected one gradient preparation per native step')
            rows.append(dict(gradients=prepared[-1], model=model_state(), optimizer=states(), result=encode(result)))
            return result

        optimizer.prepare_grads, training.train_one_step = prepare, step

        def finish():
            optimizer.prepare_grads, training.train_one_step = original_prepare, original_step
            require(len(rows) == 3, 'Expected exactly three native training steps')
            output.mkdir(parents=True, exist_ok=True)
            save(output/f'rank{dist.get_rank()}.json', dict(rank=dist.get_rank(), initial=initial, steps=rows))
        return finish
    return install


def worker(manifest, mode, output):
    fixture.worker(manifest, 'candidate', capture=mode == 'enabled', instrument=instrument(output))


def launch(root, source, wheel, cases):
    import torch
    import precision_validator
    require(torch.cuda.device_count() >= 8, 'Eight allocated GPUs required for combined topology')
    installed = Path(precision_validator.__file__).parent.parent
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('precision_validator/'):
                require((installed/name).read_bytes() == archive.read(name), f'Installed runtime mismatch: {name}')
    root.mkdir(parents=True, exist_ok=True)
    output = root/('execution-'+uuid.uuid4().hex)
    output.mkdir()
    save(root/'latest-execution.json', dict(path=str(output), started_utc=now()))
    record = dict(status='running', created_utc=now(), version=precision_validator.__version__,
                  wheel_sha256=digest(wheel), test_sha256=digest(__file__), fixture_sha256=digest(fixture.__file__),
                  device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  cases={}, comparison='exact_native_state_hashes')
    save(output/'invariance.json', record)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(torch.cuda.device_count())))).split(',')
    env = dict(os.environ)
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','PET_NODE_RANK','PET_NNODES','PET_NPROC_PER_NODE'):
        env.pop(key, None)
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_MAX_CONNECTIONS='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='2', NVTE_ALLOW_NONDETERMINISTIC_ALGO='0')
    for case in cases:
        print('INVARIANCE_CASE_START', case, flush=True)
        folder = output/case
        folder.mkdir()
        subprocess.run([sys.executable, fixture.__file__, 'make', '--case', case, '--output', str(folder),
                        '--source', str(source)], env=dict(env, CUDA_VISIBLE_DEVICES=visible[0]), check=True, timeout=240)
        jobs, pending, free = [], list(MODES), list(visible)
        world = fixture.CASES[case]['world']
        deadline = time.monotonic()+1800
        try:
            while pending or jobs:
                while pending and len(free) >= world:
                    mode = pending.pop(0)
                    allocated, free = free[:world], free[world:]
                    path = folder/mode
                    path.mkdir()
                    stream = (path/'driver.log').open('xb')
                    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(world),
                               __file__, 'worker', '--manifest', str(folder/'manifest.json'), '--mode', mode,
                               '--output', str(path)]
                    process = subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=','.join(allocated)),
                                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    jobs.append((mode, process, stream, allocated))
                for job in list(jobs):
                    mode, process, stream, allocated = job
                    if process.poll() is None:
                        continue
                    save(folder/mode/'process.json', dict(exit_code=process.returncode, finished_utc=now()))
                    require(process.returncode == 0, f'{case}/{mode}: native worker failed; inspect driver.log')
                    stream.close()
                    free.extend(allocated)
                    jobs.remove(job)
                require(time.monotonic() < deadline, 'Observer invariance case timed out')
                if jobs or pending:
                    time.sleep(2)
            failures, tensors = [], 0
            def count(value):
                if isinstance(value, dict):
                    return int('sha256' in value)+sum(count(v) for v in value.values())
                return sum(count(v) for v in value) if isinstance(value, list) else 0
            for rank in range(world):
                baseline = read(folder/'disabled'/f'rank{rank}.json')
                tensors += count(baseline)
                for mode in MODES[1:]:
                    other = read(folder/mode/f'rank{rank}.json')
                    if other != baseline:
                        failures.append(dict(rank=rank, mode=mode))
            record['cases'][case] = dict(status='passed' if not failures else 'failed', ranks=world, steps=3,
                                       tensor_states_compared=tensors, failures=failures)
            print('INVARIANCE_CASE_RESULT', case, record['cases'][case], flush=True)
        except Exception as error:
            record['cases'][case] = dict(status='execution_failure', error=str(error))
            print('INVARIANCE_CASE_ERROR', case, str(error), flush=True)
        finally:
            for _, process, stream, _ in jobs:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=15)
                stream.close()
            save(output/'invariance.json', record)
    record['status'] = 'passed' if all(r['status']=='passed' for r in record['cases'].values()) else 'failed'
    record['finished_utc'] = now()
    save(output/'invariance.json', record)
    require(record['status'] == 'passed', 'Observer invariance verification did not pass')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch','worker'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--mode', choices=MODES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--cases', nargs='+', choices=fixture.CASES, default=['ep_bf16','combined_bf16','cpu_offload_bf16'])
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.manifest, args.mode, args.output)
    else:
        launch(args.output, args.source, args.wheel, args.cases)
