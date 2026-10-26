"""Small-model GPU test of the real patched Slime train loop (not production acceptance)."""
import argparse
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from precision_validator.common import ASSET_ROOT, require, read, save, object_hash, now
from precision_validator.slime_check import code_inventory, seal_stage, load_manifest


def helper():
    sys.path.insert(0, os.environ.get('MEGATRON_PATH', '/root/Megatron-LM'))
    sys.path.insert(0, str(ASSET_ROOT / 'examples'))
    import megatron_parallel
    return megatron_parallel


def make(output, source):
    native = helper()
    import slime_parallel
    spec_path = slime_parallel.make(output / 'inputs', 'tp', source, 'float32')
    spec = read(spec_path)
    rules = {'policy': spec['policy'], 'acceptance': copy.deepcopy(spec['acceptance'])}
    rules['acceptance']['observations'] = {'loss': rules['acceptance']['observations']['loss']}
    reference = dict(tp=1, pp=1, cp=1, ep=1, etp=1, dp=2)
    candidate = dict(tp=2, pp=1, cp=1, ep=1, etp=2, dp=1)
    import precision_validator
    manifest = dict(schema='slime-training-check-v1', created_utc=now(), output=str(output), steps=3, world_size=2,
                    configurations=dict(reference=reference, repeat=reference, candidate=candidate),
                    source=code_inventory(source), toolkit=code_inventory(Path(precision_validator.__file__).parent),
                    rules=rules, disk_budget_bytes=2**30, scope='small_model_slime_rank_bridge_GPU',
                    observables={'loss': []}, stage_timeout_seconds=900)
    manifest['sha256'] = object_hash(manifest)
    save(output / 'manifest.json', manifest)


def worker(path, stage):
    native = helper()
    import numpy as np
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel as DDP, DistributedDataParallelConfig
    from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.training import global_vars
    from precision_validator.megatron_mapping import MegatronParameterMap
    from precision_validator.slime_check_capture import reached_limit
    manifest = load_manifest(path)
    sys.path.insert(0, manifest['source']['root'])
    from slime.backends.megatron_utils.model import train
    from slime.backends.megatron_utils.data import DataIterator
    layout = manifest['configurations'][stage]
    native.initialize(layout)
    data = read(Path(manifest['output']) / 'inputs/input.json')
    module, config = native.model_for('tp', layout, dtype=torch.float32, position_embedding_type='rope', sequence_length=32)
    wrapped = DDP(config, DistributedDataParallelConfig(grad_reduce_in_fp32=True, use_distributed_optimizer=True,
                  overlap_grad_reduce=False, overlap_param_gather=False, check_for_nan_in_grad=True), module)
    mapping = MegatronParameterMap(module)
    weights = np.load(Path(manifest['output']) / 'inputs/weights.npz', allow_pickle=False)
    with torch.no_grad():
        for name, parameter in mapping.parameters.items():
            for run in mapping.runs[name]:
                parameter.reshape(-1)[run.local_start:run.local_end].copy_(
                    torch.from_numpy(weights[run.name].reshape(-1)[run.start:run.end]).to(parameter.device))
    spec = read(Path(manifest['output']) / 'inputs/spec.json')
    options = spec['contract']['optimizer']['options']
    optimizer = get_megatron_optimizer(OptimizerConfig(optimizer='adam', lr=options['lr'], min_lr=options['lr'],
        weight_decay=0., adam_beta1=options['betas'][0], adam_beta2=options['betas'][1], adam_eps=options['eps'],
        clip_grad=0., params_dtype=torch.float32, use_distributed_optimizer=True,
        bf16=False, fp16=False, overlap_param_gather=False, log_num_zeros_in_grad=False), [wrapped])
    scheduler = OptimizerParamScheduler(optimizer, init_lr=options['lr'], max_lr=options['lr'], min_lr=options['lr'],
        lr_warmup_steps=0, lr_decay_steps=100, lr_decay_style='constant', start_wd=0., end_wd=0., wd_incr_steps=100, wd_incr_style='constant')
    args = SimpleNamespace(custom_megatron_before_train_step_hook_path=None, data_pad_size_multiplier=1,
        allgather_cp=False, qkv_format='bshd', position_embedding_type='rope', uses_unsplit_forward=False,
        loss_type='sft_loss', calculate_per_token_loss=True, enable_mtp_training=False, seq_length=32, micro_batch_size=1,
        decoder_seq_length=None, check_for_nan_in_loss_and_grad=True, ci_test=False, rollout_top_p=1., rollout_temperature=1.,
        log_probs_chunk_size=32, recompute_loss_function=False, overlap_grad_reduce=False, align_grad_reduce=False,
        overlap_param_gather=False, align_param_gather=False, reset_optimizer_states=False, manual_gc=False,
        use_distributed_optimizer=True, use_wandb=False, use_tensorboard=False, ci_save_grad_norm=None, ci_load_grad_norm=None,
        precision_check_steps=3, precision_check_stage=stage, precision_check_manifest=str(path))
    global_vars._GLOBAL_ARGS = args
    samples = list(range(ps.get_data_parallel_rank(), 4, layout['dp'])) * 5
    class Iterator(DataIterator):
        def get_next(self, keys):
            return super().get_next([*keys, 'max_seq_lens'])
    rollout = dict(tokens=[torch.tensor(data['tokens'][i], device='cuda') for i in samples],
        total_lengths=[17] * len(samples), response_lengths=[16] * len(samples),
        loss_masks=[torch.tensor(data['loss_masks'][i], device='cuda') for i in samples],
        rollout_mask_sums=[sum(data['loss_masks'][i]) for i in samples], max_seq_lens=[32] * len(samples))
    iterator = Iterator(rollout, [[i] for i in range(len(samples))])
    # Five scheduled steps, but the plugin must execute exactly three.
    train(0, [wrapped], optimizer, scheduler, [iterator], [4 // layout['dp']] * 5, [4] * 5)
    require(reached_limit(args) and iterator.offset == 3 * 4 // layout['dp'] and scheduler.num_steps == 12,
            'Real Slime loop did not stop at the requested optimizer-step boundary')
    ps.destroy_model_parallel()
    dist.destroy_process_group()


def launch(output, source):
    import torch
    require(torch.cuda.device_count() >= 6, 'Parallel reference/repeat/candidate need six allocated GPUs')
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'PET_NODE_RANK', 'PET_NNODES', 'PET_NPROC_PER_NODE'):
        env.pop(key, None)
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_MAX_CONNECTIONS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    subprocess.run([sys.executable, __file__, 'make', '--output', str(output), '--source', str(source)], env=env, check=True)
    jobs = []
    try:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5').split(',')
        for i, stage in enumerate(('reference', 'repeat', 'candidate')):
            folder = output / stage
            folder.mkdir()
            stream = (folder / 'driver.log').open('xb')
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
                       __file__, 'worker', '--manifest', str(output / 'manifest.json'), '--stage', stage]
            process = subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=','.join(visible[i * 2:i * 2 + 2])),
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append((stage, process, stream))
        deadline = time.monotonic() + 900
        while any(p.poll() is None for _, p, _ in jobs):
            codes = {stage: p.poll() for stage, p, _ in jobs}
            save(output / 'progress.json', {'utc': now(), 'codes': codes})
            require(all(code in (0, None) for code in codes.values()), 'GPU bridge worker failed; inspect stage driver.log')
            require(time.monotonic() < deadline, 'GPU bridge test timed out')
            time.sleep(5)
        manifest = load_manifest(output / 'manifest.json')
        for stage, p, stream in jobs:
            require(p.returncode == 0, 'GPU worker failed')
            save(output / stage / 'process.json', {'exit_code': p.returncode, 'timed_out': False, 'finished_utc': now()})
            seal_stage(manifest, stage)
        from precision_validator.slime_check_compare import compare
        report = compare(output / 'manifest.json')
        print('SLIME_BRIDGE_RESULT', report['status'], flush=True)
        require(report['status'] == 'passed', 'GPU numerical gates failed')
    finally:
        for _, p, stream in jobs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=15)
            stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('make', 'worker', 'launch'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--stage')
    cli = parser.parse_args()
    if cli.action == 'make': make(cli.output, cli.source)
    elif cli.action == 'worker': worker(cli.manifest, cli.stage)
    else: launch(cli.output, cli.source)
