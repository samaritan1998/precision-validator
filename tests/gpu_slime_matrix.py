"""Release qualification against an installed wheel and an isolated Slime tree.

All weights and masked examples are synthetic. This suite exercises Slime's
real training loop, not its Ray driver or production model convergence.
"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

from precision_validator.common import ASSET_ROOT, require, read, save, object_hash, now, digest
from precision_validator.slime_check import code_inventory, seal_stage, load_manifest

CASES = {
    'tp_fp32': dict(kind='tp', dtype='float32', world=2, target=dict(tp=2, etp=2)),
    'pp_fp32': dict(kind='pp', dtype='float32', world=2, target=dict(pp=2)),
    'ep_fp32': dict(kind='ep', dtype='float32', world=2, target=dict(ep=2)),
    'tp_bf16': dict(kind='tp', dtype='bfloat16', world=2, target=dict(tp=2, etp=2)),
    'ep_bf16': dict(kind='ep', dtype='bfloat16', world=2, target=dict(ep=2)),
    'cp2_bf16': dict(kind='cp', dtype='bfloat16', world=2, target=dict(cp=2)),
    'cp4_bf16': dict(kind='cp', dtype='bfloat16', world=4, target=dict(cp=4)),
    'tp2_cp2_bf16': dict(kind='cp', dtype='bfloat16', world=4, target=dict(tp=2, etp=2, cp=2)),
    'combined_bf16': dict(kind='combined', dtype='bfloat16', world=8, target=dict(tp=2, etp=2, pp=2, ep=2)),
    'cpu_offload_bf16': dict(kind='tp', dtype='bfloat16', world=2, target=dict(tp=2, etp=2), offload=True),
}


def helper():
    require(os.environ.get('MEGATRON_PATH'), 'Set MEGATRON_PATH to the tested framework checkout')
    sys.path.insert(0, os.environ['MEGATRON_PATH'])
    sys.path.insert(0, str(ASSET_ROOT / 'examples'))
    import megatron_parallel
    return megatron_parallel


def model_for(case, layout):
    import torch
    native = helper()
    dtype = getattr(torch, case['dtype'])
    if case['kind'] != 'cp':
        return native.model_for(case['kind'], layout, dtype=dtype, position_embedding_type='rope', sequence_length=64)
    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend
    config = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4, num_query_groups=4,
        ffn_hidden_size=256, kv_channels=32, tensor_model_parallel_size=layout['tp'], pipeline_model_parallel_size=layout['pp'],
        context_parallel_size=layout['cp'], expert_model_parallel_size=layout['ep'], expert_tensor_parallel_size=layout['etp'],
        sequence_parallel=False, params_dtype=dtype, pipeline_dtype=dtype, bf16=True,
        hidden_dropout=0., attention_dropout=0., add_bias_linear=False, normalization='RMSNorm',
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        gradient_accumulation_fusion=False, bias_activation_fusion=False, bias_dropout_fusion=False,
        masked_softmax_fusion=False, apply_rope_fusion=False, calculate_per_token_loss=True,
        cp_comm_type='a2a', attention_backend=AttnBackend.flash)
    spec = get_gpt_layer_with_transformer_engine_spec()
    module = GPTModel(config, spec, vocab_size=64, max_sequence_length=64,
        pre_process=ps.is_pipeline_first_stage(), post_process=ps.is_pipeline_last_stage(),
        share_embeddings_and_output_weights=False, parallel_output=True, position_embedding_type='rope').cuda().train()
    return module, config


def make(output, source, case_name):
    import numpy as np
    import torch
    native = helper()
    case = CASES[case_name]
    # The native builder creates the full canonical checkpoint on one GPU.
    spec_path = native.make(output / 'inputs', case['kind'] if case['kind'] != 'cp' else 'tp',
        model_factory=lambda kind, layout: model_for(case, layout), weight_dtype=getattr(torch, case['dtype']))
    spec = read(spec_path)
    rng = np.random.default_rng(1979)
    lengths = [17, 21, 25, 29, 33, 37, 41, 45]
    masks = [[0] * prefix + [1] * (length - 1 - prefix) for length, prefix in zip(lengths, [1, 3, 2, 4, 1, 3, 2, 4])]
    save(output / 'inputs/input.json', dict(tokens=[rng.integers(0, 64, n).tolist() for n in lengths],
        loss_masks=masks, supervised_units=sum(map(sum, masks)), source='synthetic_release_fixture'))
    rules = {'policy': spec['policy'], 'acceptance': copy.deepcopy(spec['acceptance'])}
    rules['acceptance']['observations'] = {'loss': rules['acceptance']['observations']['loss']}
    if case['dtype'] == 'bfloat16':
        # Fixed before any release run, inherited from the previous BF16 integration fixture.
        limit = dict(relative_l2=.05, max_abs=.003, near_zero_rms=1e-8, near_zero_max_abs=1e-8)
        for category in ('parameters', 'groups'):
            for row in rules['acceptance'][category]:
                for field in ('gradients', 'updates'):
                    row[field] = copy.deepcopy(limit)
        rules['acceptance']['observations']['loss'] = copy.deepcopy(limit)
        for row in rules['policy'].values():
            row.update(target_relative_ceiling=.02, noise_slack=.02, max_abs_ceiling=.003)
    def layout(overrides):
        value = dict(tp=1, pp=1, cp=1, ep=1, etp=1)
        value.update(overrides)
        value['dp'] = case['world'] // (value['tp'] * value['pp'] * value['cp'])
        return value
    reference, candidate = layout({}), layout(case['target'])
    import precision_validator
    manifest = dict(schema='slime-training-check-v1', created_utc=now(), output=str(output), steps=3,
        world_size=case['world'], configurations=dict(reference=reference, repeat=reference, candidate=candidate),
        source=code_inventory(source), toolkit=code_inventory(Path(precision_validator.__file__).parent),
        rules=rules, disk_budget_bytes=2**30, scope='synthetic_slime_train_loop_GPU',
        observables={'loss': []}, stage_timeout_seconds=1200,
        qualification=dict(case=case_name, definition=case, test_sha256=digest(__file__),
            inputs={p.name:digest(p) for p in (output / 'inputs').iterdir() if p.is_file()}))
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
    from megatron.core.transformer.module import Float16Module
    from megatron.training import global_vars
    from precision_validator.megatron_mapping import MegatronParameterMap
    from precision_validator.slime_check_capture import reached_limit
    manifest = load_manifest(path)
    case = manifest['qualification']['definition']
    require(digest(__file__) == manifest['qualification']['test_sha256'], 'Test changed during execution')
    for name, sha in manifest['qualification']['inputs'].items():
        require(digest(Path(manifest['output']) / 'inputs' / name) == sha, 'Input artifact changed')
    sys.path.insert(0, manifest['source']['root'])
    import slime.backends.megatron_utils.model as training
    from slime.backends.megatron_utils.data import DataIterator
    layout = manifest['configurations'][stage]
    native.initialize(layout)
    data = read(Path(manifest['output']) / 'inputs/input.json')
    module, config = model_for(case, layout)
    dtype = getattr(torch, case['dtype'])
    precision_module = Float16Module(config, module) if dtype == torch.bfloat16 else module
    wrapped = DDP(config, DistributedDataParallelConfig(grad_reduce_in_fp32=True, use_distributed_optimizer=True,
        overlap_grad_reduce=False, overlap_param_gather=False, check_for_nan_in_grad=True), precision_module)
    mapping = MegatronParameterMap(module)
    weights = np.load(Path(manifest['output']) / 'inputs/weights.npz', allow_pickle=False)
    with torch.no_grad():
        for name, parameter in mapping.parameters.items():
            for run in mapping.runs[name]:
                parameter.reshape(-1)[run.local_start:run.local_end].copy_(
                    torch.from_numpy(weights[run.name].reshape(-1)[run.start:run.end]).to(parameter.device))
    spec = read(Path(manifest['output']) / 'inputs/spec.json')
    options = spec['contract']['optimizer']['options']
    backend = dict(use_precision_aware_optimizer=True, optimizer_cpu_offload=True,
        optimizer_offload_fraction=1., overlap_cpu_optimizer_d2h_h2d=True) if case.get('offload') else {}
    optimizer = get_megatron_optimizer(OptimizerConfig(optimizer='adam', lr=options['lr'], min_lr=options['lr'],
        weight_decay=0., adam_beta1=options['betas'][0], adam_beta2=options['betas'][1], adam_eps=options['eps'],
        clip_grad=0., params_dtype=dtype, use_distributed_optimizer=True, bf16=dtype==torch.bfloat16, fp16=False,
        overlap_param_gather=False, log_num_zeros_in_grad=False, **backend), [wrapped])
    if case.get('offload'):
        require(all(type(inner.optimizer).__name__ == 'HybridDeviceOptimizer' and inner.optimizer.offload_fraction == 1.
            and inner.optimizer.cpu_optimizers and inner.optimizer.gpu_optimizer is None
            for inner in getattr(optimizer, 'chained_optimizers', [optimizer])), 'CPU optimizer offload not active')
    scheduler = OptimizerParamScheduler(optimizer, init_lr=options['lr'], max_lr=options['lr'], min_lr=options['lr'],
        lr_warmup_steps=0, lr_decay_steps=100, lr_decay_style='constant', start_wd=0., end_wd=0., wd_incr_steps=100, wd_incr_style='constant')
    args = SimpleNamespace(custom_megatron_before_train_step_hook_path=None, data_pad_size_multiplier=1,
        allgather_cp=False, qkv_format='bshd', position_embedding_type='rope', uses_unsplit_forward=False,
        loss_type='sft_loss', calculate_per_token_loss=True, enable_mtp_training=False, seq_length=64, micro_batch_size=1,
        decoder_seq_length=None, check_for_nan_in_loss_and_grad=True, ci_test=False, rollout_top_p=1., rollout_temperature=1.,
        log_probs_chunk_size=32, recompute_loss_function=False, overlap_grad_reduce=False, align_grad_reduce=False,
        overlap_param_gather=False, align_param_gather=False, reset_optimizer_states=False, manual_gc=False,
        use_distributed_optimizer=True, use_wandb=False, use_tensorboard=False, ci_save_grad_norm=None, ci_load_grad_norm=None,
        precision_check_steps=3, precision_check_stage=stage, precision_check_manifest=str(path))
    global_vars._GLOBAL_ARGS = args
    samples = list(range(ps.get_data_parallel_rank(), len(data['tokens']), layout['dp'])) * 5
    class Iterator(DataIterator):
        def get_next(self, keys):
            return super().get_next([*keys, 'max_seq_lens'])
    rollout = dict(tokens=[torch.tensor(data['tokens'][i], device='cuda') for i in samples],
        total_lengths=[len(data['tokens'][i]) for i in samples],
        response_lengths=[len(data['loss_masks'][i]) for i in samples],
        loss_masks=[torch.tensor(data['loss_masks'][i], device='cuda') for i in samples],
        rollout_mask_sums=[sum(data['loss_masks'][i]) for i in samples], max_seq_lens=[64] * len(samples))
    iterator = Iterator(rollout, [[i] for i in range(len(samples))])
    # Observe the true denominator after native DP x CP reduction; do not change gradients.
    denominators = []
    finalize_name = '_finalize_model_grads' if hasattr(training, '_finalize_model_grads') else 'finalize_model_grads'
    original_finalize = getattr(training, finalize_name)
    def finalize(chunks, num_tokens=None, **kwargs):
        result = original_finalize(chunks, num_tokens=num_tokens, **kwargs)
        require(num_tokens is not None, 'Missing native supervised-token denominator')
        actual = int(num_tokens)
        expected = data['supervised_units'] * layout['cp']
        denominators.append(actual)
        require(actual == expected, f'Native CP-weighted denominator differs: {actual} != {expected}')
        return result
    setattr(training, finalize_name, finalize)
    training.train(0, [wrapped], optimizer, scheduler, [iterator], [8 // layout['dp']] * 5, [8] * 5)
    require(reached_limit(args) and iterator.offset == 3 * 8 // layout['dp'] and scheduler.num_steps == 24,
        'Real Slime loop did not stop at exactly three optimizer steps')
    require(len(denominators) == 3, 'Missing gradient finalization')
    save(Path(manifest['output']) / stage / f'denominator-rank{dist.get_rank()}.json',
        dict(actual=denominators, supervised_units=data['supervised_units'], cp=layout['cp']))
    ps.destroy_model_parallel()
    dist.destroy_process_group()


def launch_case(output, source, case_name, env, visible):
    case = CASES[case_name]
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, __file__, 'make', '--case', case_name, '--output', str(output), '--source', str(source)],
        env=dict(env, CUDA_VISIBLE_DEVICES=visible[0]), check=True, timeout=240)
    jobs, pending = [], list(('reference', 'repeat', 'candidate'))
    free = list(visible)
    deadline = time.monotonic() + 3600
    try:
        while pending or jobs:
            while pending and len(free) >= case['world']:
                stage = pending.pop(0)
                allocated, free = free[:case['world']], free[case['world']:]
                folder = output / stage
                folder.mkdir()
                stream = (folder / 'driver.log').open('xb')
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(case['world']),
                    __file__, 'worker', '--manifest', str(output / 'manifest.json'), '--stage', stage]
                process = subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=','.join(allocated)),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                jobs.append((stage, process, stream, allocated))
            save(output / 'progress.json', {'utc': now(), 'pending': pending, 'codes': {s:p.poll() for s,p,_,_ in jobs}})
            for job in list(jobs):
                stage, process, stream, allocated = job
                if process.poll() is None:
                    continue
                save(output / stage / 'process.json', dict(exit_code=process.returncode, timed_out=False, finished_utc=now()))
                require(process.returncode == 0, f'{case_name}/{stage} failed; inspect driver.log')
                seal_stage(load_manifest(output / 'manifest.json'), stage)
                stream.close()
                free.extend(allocated)
                jobs.remove(job)
            require(time.monotonic() < deadline, 'GPU matrix case timed out')
            if pending or jobs:
                time.sleep(3)
        from precision_validator.slime_check_compare import compare
        report = compare(output / 'manifest.json')
        return dict(status=report['status'], report_sha256=digest(output / 'acceptance.json'),
            manifest_sha256=read(output / 'manifest.json')['sha256'])
    finally:
        for _, p, stream, _ in jobs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=15)
            stream.close()


def launch(output, source, cases, wheel):
    import torch
    import precision_validator
    require(torch.cuda.device_count() >= 8, 'Full release matrix requires eight allocated GPUs')
    output.mkdir(parents=True, exist_ok=True)
    output = output / ('execution-' + uuid.uuid4().hex)
    output.mkdir()
    save(output.parent / 'latest-execution.json', dict(path=str(output), started_utc=now()))
    print('MATRIX_EXECUTION', str(output), flush=True)
    # Verify every installed package byte against the supplied wheel before any run.
    import zipfile
    installed = Path(precision_validator.__file__).parent.parent
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('precision_validator/'):
                require((installed / name).read_bytes() == archive.read(name), f'Installed wheel mismatch: {name}')
    env = dict(os.environ)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'PET_NODE_RANK', 'PET_NNODES', 'PET_NPROC_PER_NODE'):
        env.pop(key, None)
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_MAX_CONNECTIONS='1', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='2', NVTE_ALLOW_NONDETERMINISTIC_ALGO='0')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(torch.cuda.device_count())))).split(',')
    record = dict(status='running', started_utc=now(), wheel_sha256=digest(wheel), version=precision_validator.__version__,
        test_sha256=digest(__file__), device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        source=code_inventory(source), cases={},
        device_family={8: 'ampere', 9: 'hopper', 10: 'blackwell'}.get(torch.cuda.get_device_capability()[0], 'unknown'))
    save(output / 'qualification.json', record)
    for name in cases:
        print('MATRIX_CASE_START', name, flush=True)
        try:
            result = launch_case(output / name, source, name, env, visible)
        except Exception as error:
            result = dict(status='execution_failure', error=str(error))
        record['cases'][name] = result
        save(output / 'qualification.json', record)
        print('MATRIX_CASE_RESULT', name, result, flush=True)
    record['status'] = 'passed' if all(r['status'] == 'passed' for r in record['cases'].values()) else 'failed'
    record['finished_utc'] = now()
    save(output / 'qualification.json', record)
    require(record['status'] == 'passed', 'Release matrix has failures; inspect qualification.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('make', 'worker', 'launch'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--stage')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    cli = parser.parse_args()
    if cli.action == 'make':
        make(cli.output, cli.source, cli.case)
    elif cli.action == 'worker':
        worker(cli.manifest, cli.stage)
    else:
        launch(cli.output, cli.source, cli.cases, cli.wheel)
