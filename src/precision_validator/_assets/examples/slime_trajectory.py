"""Real Slime train_one_step integration with canonical native optimizer evidence.

The small GPT/MoE fixtures exercise the training entry, loss, scheduler and
distributed optimizer. They are not full Qwen or 256K model acceptance.
"""
import argparse
import copy
import inspect
import math
import os
from pathlib import Path
import sys
import random
import subprocess
import uuid
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

# Slime also needs megatron.training, absent from the pip-only core package.
if os.environ.get('MEGATRON_PATH'):
    sys.path.insert(0, str(Path(os.environ['MEGATRON_PATH']).resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import megatron_parallel as native
from precision_validator.common import require, read, save, load_plan, digest
from precision_validator.megatron_mapping import MegatronParameterMap
from precision_validator.megatron_observer import NativeStepObserver, capture_initial
from precision_validator.sdk import RankWriter


from precision_validator.native_trajectory import STATE_FIELDS, STORAGE_ALIGNMENT, identity
from precision_validator.precision import stage_precision
from precision_validator.native_trajectory_sdk import (NativeTrajectoryObserver, StepWriter, state_signature,
    save_checkpoint, restore_checkpoint)
from precision_validator.sdk import runtime_info
import slime_parallel as single


def make(output, kind, source, dtype, optimizer_backend, reference_dtype=None):
    native_kind = 'tp' if kind == 'storage' else kind
    if reference_dtype is not None:
        require(reference_dtype == 'float32' and dtype == 'bfloat16' and optimizer_backend == 'distributed-adam',
                'Storage comparison currently requires FP32/BF16 native distributed Adam')
    require(kind != 'storage' or reference_dtype is not None, 'Storage fixture requires two precision contracts')
    path = single.make(output, native_kind, source, dtype, optimizer_backend=optimizer_backend)
    root = Path(output)
    spec = read(path)
    spec['name'] = f'slime-trajectory-{dtype}-{kind}-{optimizer_backend}'
    spec['artifacts']['code'] = str(Path(__file__).resolve())
    spec['artifacts']['single_step_helper'] = str(Path(__file__).resolve().with_name('slime_parallel.py'))
    spec['command'] = ['{python}', str(Path(__file__).resolve()), 'launch', '--plan', '{plan}', '--stage', '{stage}']
    spec['contract']['optimizer_steps'] = 3
    if kind == 'storage':
        spec['configurations']['candidate'] = copy.deepcopy(spec['configurations']['reference'])
    if reference_dtype is not None:
        candidate = spec['contract'].pop('precision')
        spec['contract']['precision_by_stage'] = {
            'reference': dict(candidate, parameters=reference_dtype), 'candidate': candidate}
        spec['contract']['precision_comparison'] = {
            'allowed_changes': ['parameters'], 'initial_weights_alignment': STORAGE_ALIGNMENT}
        spec['name'] += '-from-' + reference_dtype
    rng = np.random.default_rng(71623)
    steps = []
    for i, (batch, length) in enumerate(((4,17),(2,25),(4,13)), 1):
        masks = [[0] * ((j+i)%4) + [1] * (length-1-(j+i)%4) for j in range(batch)]
        data = {'kind':native_kind,'tokens':rng.integers(0,64,(batch,length)).tolist(),'loss_masks':masks,
                'supervised_units':sum(map(sum,masks)), 'source':'seeded_native_trajectory_integration_fixture'}
        artifact = f'input_step_{i}'
        save(root/f'{artifact}.json',data)
        spec['artifacts'][artifact] = str(root/f'{artifact}.json')
        steps.append({'input_artifact':artifact,'objective':dict(spec['contract']['objective'],
            global_samples=batch,supervised_units=data['supervised_units']),
            'observables':{'loss':[],'sampled_logits':[batch,4,4]}})
    limit = copy.deepcopy(spec['acceptance']['parameters'][0]['gradients'])
    spec['trajectory'] = {'steps':steps,'resume_after_step':1,
        'scheduler':{'name':'MegatronOptimizerParamScheduler','options':{
            'init_lr':0.001,'max_lr':0.001,'min_lr':0.0005,'lr_warmup_steps':0,'lr_decay_steps':10,
            'lr_decay_style':'linear','start_wd':0.0,'end_wd':0.0,'wd_incr_steps':10,'wd_incr_style':'constant'}},
        'state_acceptance':{category:[{'pattern':'*',**{field:copy.deepcopy(limit) for field in STATE_FIELDS}}]
                            for category in ('parameters','groups')}}
    spec['contract']['objective'] = steps[0]['objective']
    save(path,spec)
    return path

def worker(path, stage, phase):
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.transformer.module import Float16Module
    from megatron.training import global_vars
    plan = load_plan(path)
    process_id = uuid.uuid4().hex
    random.seed(plan['contract']['seed']); np.random.seed(plan['contract']['seed'])
    source = Path(plan['artifacts']['slime_source']['path'])
    sys.path.insert(0,str(source))
    from slime.backends.megatron_utils.model import train_one_step
    from slime.backends.megatron_utils.data import DataIterator
    entry = Path(inspect.getsourcefile(train_one_step)).resolve()
    expected = plan['artifacts']['slime_source']['files']['slime/backends/megatron_utils/model.py']['sha256']
    require(entry==source/'slime/backends/megatron_utils/model.py' and digest(entry)==expected, 'Wrong Slime training entry')
    layout = plan['configurations'][stage]['parallelism']
    groups = native.initialize(layout)
    data = read(plan['artifacts']['input']['path'])
    model_config = read(plan['artifacts']['model_config']['path'])
    precision = stage_precision(plan, stage)
    dtype = getattr(torch, precision['parameters'])
    model,config = native.model_for(data['kind'],layout,dtype=dtype,position_embedding_type='rope',sequence_length=32)
    precision_module = Float16Module(config,model) if dtype==torch.bfloat16 else model
    require(all(p.dtype==dtype for p in model.parameters()), 'Native model parameter cast differs')
    wrapped = DDP(config,DistributedDataParallelConfig(grad_reduce_in_fp32=True,use_distributed_optimizer=True,
                  overlap_grad_reduce=False,overlap_param_gather=False,check_for_nan_in_grad=True),precision_module)
    mapping = MegatronParameterMap(model)
    catalog = plan['parameters']
    catalogs = [None]*dist.get_world_size();dist.all_gather_object(catalogs,mapping.catalog())
    merged = {}
    for local in catalogs:
        for name,info in local.items():require(merged.setdefault(name,info)==info,'Canonical catalog disagreement')
    require(merged==catalog,'Canonical Slime parameter catalog differs')
    weights = np.load(plan['artifacts']['weights']['path'],allow_pickle=False)
    with torch.no_grad():
        for name,p in mapping.parameters.items():
            for run in mapping.runs[name]:
                value = torch.from_numpy(weights[run.name].reshape(-1)[run.start:run.end])
                p.reshape(-1)[run.local_start:run.local_end].copy_(value.to(p.device))
                require(torch.equal(p.reshape(-1)[run.local_start:run.local_end].float().cpu(),value),'Initial weight cast changed')
    writer = RankWriter(plan,stage,dist.get_rank())
    initial = capture_initial(mapping,writer,dtype,storage_alignment='precision_by_stage' in plan['contract'])
    options = plan['contract']['optimizer']['options']
    backend = model_config.get('optimizer_backend','distributed-adam')
    require(backend in ('distributed-adam','cpu-offload'), 'Unknown native optimizer backend')
    backend_options = (dict(use_precision_aware_optimizer=True,optimizer_cpu_offload=True,
                           optimizer_offload_fraction=1.0,overlap_cpu_optimizer_d2h_h2d=True)
                       if backend=='cpu-offload' else {})
    optimconfig = OptimizerConfig(optimizer='adam',lr=options['lr'],min_lr=plan['trajectory']['scheduler']['options']['min_lr'],weight_decay=0.,
        adam_beta1=options['betas'][0],adam_beta2=options['betas'][1],adam_eps=options['eps'],
        clip_grad=0.,params_dtype=dtype,use_distributed_optimizer=True,bf16=dtype==torch.bfloat16,fp16=False,
        overlap_param_gather=False,log_num_zeros_in_grad=False,**backend_options)
    optimizer = get_megatron_optimizer(optimconfig,[wrapped])
    inners = getattr(optimizer,'chained_optimizers',[optimizer])
    if backend=='cpu-offload':
        require(all(type(i.optimizer).__name__=='HybridDeviceOptimizer' and i.optimizer.offload_fraction==1.0
                    and i.optimizer.cpu_optimizers and i.optimizer.gpu_optimizer is None for i in inners),
                'Native full CPU optimizer offload was not enabled')
    scheduler = OptimizerParamScheduler(optimizer, **plan['trajectory']['scheduler']['options'])
    args = SimpleNamespace(custom_megatron_before_train_step_hook_path=None,data_pad_size_multiplier=1,
        allgather_cp=False,qkv_format='bshd',position_embedding_type='rope',uses_unsplit_forward=False,
        loss_type='sft_loss',calculate_per_token_loss=True,enable_mtp_training=False,seq_length=32,micro_batch_size=1,
        decoder_seq_length=None,check_for_nan_in_loss_and_grad=True,ci_test=False,rollout_top_p=1.,
        rollout_temperature=1.,log_probs_chunk_size=32,recompute_loss_function=False)
    global_vars._GLOBAL_ARGS = args
    initial_parameters = writer.initial
    initial_state = state_signature(model,mapping,optimizer,scheduler,plan,0)
    initial_record = {'parameters':initial_parameters,'state':initial_state}
    boundary, count = plan['trajectory']['resume_after_step'], plan['contract']['optimizer_steps']
    resumed = None
    if phase == 'suffix':
        record,resumed = restore_checkpoint(model,mapping,optimizer,scheduler,plan,dist.get_rank(),process_id)
        require(record['initial']==initial_record,'Fresh resume initialization differs from prefix')
        first,stop = boundary+1,count
    else:
        first,stop = 1,boundary if phase=='prefix' else count
    for step in range(first,stop+1):
        frozen = plan['trajectory']['steps'][step-1]
        data = read(plan['artifacts'][frozen['input_artifact']]['path'])
        require(data['supervised_units']==frozen['objective']['supervised_units'] and
                len(data['tokens'])==frozen['objective']['global_samples'],'Frozen input objective differs')
        writer = StepWriter(plan,stage,dist.get_rank(),step)
        dp_rank = ps.get_data_parallel_rank()
        batch = frozen['objective']['global_samples']
        samples = list(range(dp_rank,batch,layout['dp']))
        current_sample = {'id':None}
        class Iterator(DataIterator):
            def get_next(self, keys):
                current_sample['id'] = samples[self.offset]
                # BSHD requires padding metadata; add it through the public iterator.
                return super().get_next([*keys,'max_seq_lens'])
        rollout = {'tokens':[torch.tensor(data['tokens'][i],device='cuda') for i in samples],
            'total_lengths':[len(data['tokens'][i]) for i in samples],'response_lengths':[len(data['tokens'][i])-1 for i in samples],
            'loss_masks':[torch.tensor(data['loss_masks'][i],device='cuda') for i in samples],
            'rollout_mask_sums':[sum(data['loss_masks'][i]) for i in samples],
            'max_seq_lens':[32]*len(samples)}
        iterator = [Iterator(rollout,[[i] for i in range(len(samples))])]
        forward_calls, backward_calls, tokens, outputs, routing = [],[],[],[],[]
        handles = []
        for layer in model.decoder.layers:
            handles += [layer.register_forward_hook(lambda *a:forward_calls.append(1)),
                        layer.register_full_backward_hook(lambda *a:backward_calls.append(1))]
            if model_config.get('diagnose_routing') and hasattr(layer.mlp,'router'):
                def router_hook(module,incoming,result,number=layer.layer_number):
                    probabilities,route = result
                    hidden = incoming[0].detach().reshape(-1,config.hidden_size)
                    require(hidden.shape[0]* (layout['tp'] if config.sequence_parallel else 1)==32,
                            'Diagnostic router token layout differs')
                    start = ps.get_tensor_model_parallel_rank()*hidden.shape[0] if config.sequence_parallel else 0
                    routing.append({'layer':number,'sample':current_sample['id'],'start':start,
                        'tp_rank':ps.get_tensor_model_parallel_rank(),'dp_rank':dp_rank,
                        'hidden':hidden.float().cpu().tolist(),'probabilities':probabilities.detach().float().cpu().tolist(),
                        'routing_map':route.detach().cpu().tolist()})
                handles.append(layer.mlp.router.register_forward_hook(router_hook))
        def finalize(chunks,num_tokens=None,**kwargs):
            finalize_model_grads(chunks,num_tokens=num_tokens,**kwargs)
            require(num_tokens is not None and int(num_tokens)==frozen['objective']['supervised_units'],'Slime global supervised denominator differs')
            tokens.append(int(num_tokens))
        config.finalize_model_grads_func = finalize
        config.grad_scale_func = optimizer.scale_loss
        if ps.is_pipeline_last_stage():
            def output_hook(module,args,result):
                value = result[0] if isinstance(result,tuple) else result
                require(list(value.shape[:2])==[32,1],'Slime native logits layout differs')
                outputs.append({'sample':current_sample['id'],'tp_rank':ps.get_tensor_model_parallel_rank(),
                                'values':value.detach()[[0,3,7,11],0,:].float().cpu().tolist()})
            handles.append(model.output_layer.register_forward_hook(output_hook))
        lr_used = [g['lr'] for g in optimizer.param_groups]
        scheduler_options = plan['trajectory']['scheduler']['options']
        consumed = sum(s['objective']['global_samples'] for s in plan['trajectory']['steps'][:step-1])
        expected_lr = scheduler_options['max_lr'] - (scheduler_options['max_lr'] - scheduler_options['min_lr']) * consumed / scheduler_options['lr_decay_steps']
        require(all(abs(lr-expected_lr)<1e-12 for lr in lr_used), 'Native parameter-group learning rate did not follow frozen decay')
        with NativeTrajectoryObserver(mapping,optimizer,writer,precision,step) as observed:
            losses,grad_norm = train_one_step(args,0,step-1,iterator,[wrapped],optimizer,scheduler,len(samples),batch)
        require(tokens==[frozen['objective']['supervised_units']] and len(forward_calls)==len(backward_calls)==len(samples)*(2//layout['pp']),
                'Slime forward/backward/finalization did not fully execute')
        require(math.isfinite(float(grad_norm)) and scheduler.num_steps==sum(s['objective']['global_samples'] for s in plan['trajectory']['steps'][:step]),'Slime gradient norm/scheduler differs')
        require(all(p.grad is None for p in model.parameters()) and all(bool(torch.count_nonzero(p.main_grad)==0)
                for p in model.parameters() if hasattr(p,'main_grad')),'Slime did not clear native gradients')
        rows = [None]*dist.get_world_size()
        dist.all_gather_object(rows,{'last':ps.is_pipeline_last_stage(),'losses':{k:float(v) for k,v in losses.items()},
            'tp_rank':ps.get_tensor_model_parallel_rank(),'dp_rank':dp_rank,'samples':samples,'outputs':outputs})
        if dist.get_rank()==0:
            finals = [r for r in rows if r['last']]
            loss = finals[0]['losses']['loss']
            require(all(r['losses']['loss']==loss for r in finals),'Reduced Slime loss replicas disagree')
            writer.observe('loss',loss)
            joined = []
            for sample in range(batch):
                parts = [p for r in finals for p in r['outputs'] if p['sample']==sample]
                require(sorted(p['tp_rank'] for p in parts)==list(range(layout['tp'])),'Slime sampled logits coverage differs')
                full = np.concatenate([np.asarray(p['values']) for p in sorted(parts,key=lambda p:p['tp_rank'])],axis=-1)
                joined.append(full[:,[0,7,31,63]])
            writer.observe('sampled_logits',np.asarray(joined))
        for handle in handles:handle.remove()
        master_devices = sorted({row['state_parameter'].device.type for row in observed.shards.values()})
        if backend=='cpu-offload':
            require(master_devices==['cpu'] and all(row['state_parameter'].dtype==torch.float32
                    and all(row['state_optimizer'].state[row['state_parameter']][k].device.type=='cpu'
                            for k in ('exp_avg','exp_avg_sq')) for row in observed.shards.values()),
                    'CPU optimizer masters or states are not actually on CPU')
        optimizer_backend = {'name':backend,'classes':[type(i.optimizer).__name__ for i in inners],
                             'actual_master_devices':master_devices,'native_options':backend_options}
        state = state_signature(model,mapping,optimizer,scheduler,plan,step)
        writer.seal(state,lr_used,{key:True for key in plan['required_checks']},observed.receipt,
            {'native_global_tokens':tokens,'local_samples':samples,'optimizer_backend':optimizer_backend,
             'entry_sha256':expected,'factory_parameters':mapping.factories})
        print('NATIVE_TRAJECTORY_STEP',stage,phase,step,dist.get_rank(),flush=True)
    if phase == 'prefix':
        save_checkpoint(model,mapping,optimizer,scheduler,plan,dist.get_rank(),initial_record,process_id)
    rank = dist.get_rank()
    torch.cuda.synchronize();dist.barrier();ps.destroy_model_parallel();dist.destroy_process_group()
    folder = Path(plan['output'])/stage
    exit_row = {'plan_sha256':plan['plan_sha256'],'stage':stage,'rank':rank,'phase':phase,
                'process_id':process_id,'pid':os.getpid(),'teardown_completed':not dist.is_initialized()}
    save(folder/f'exit.{phase}.rank{rank}.json',exit_row)
    if phase != 'prefix':
        save(folder/f'rank{rank}.pending.json',{**identity(plan,stage,rank),
            'execution':{'optimizer_steps':count,'teardown_completed':not dist.is_initialized()},
            'precision':precision,'runtime':runtime_info(),
            'environment':{k:os.environ.get(k) for k in plan['environment']},'native_parallelism':groups,
            'device':{'kind':'cuda','name':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability())},
            'initial_parameters':initial_parameters,'initial_state':initial_state,'resume':resumed,
            'step_receipts':{f'steps/{i:06d}/rank{rank}.json':digest(folder/'steps'/f'{i:06d}'/f'rank{rank}.json')
                             for i in range(1,count+1)}})
    print('SLIME_TRAJECTORY_RANK_COMPLETE',stage,phase,rank,flush=True)


def launch(plan_path,stage):
    plan = load_plan(plan_path)
    world = plan['configurations'][stage]['world_size']
    folder = Path(plan['output'])/stage
    workers=[]
    for phase in (('prefix','suffix') if stage=='resume' else ('continuous',)):
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--max-restarts=0',
                 '--nproc-per-node',str(world),str(Path(__file__).resolve()),'worker',
                 '--plan',str(plan_path),'--stage',stage,'--phase',phase]
        process=subprocess.run(command,timeout=plan['stage_timeout_seconds'])
        require(process.returncode==0,f'Native trajectory {phase} failed')
        workers.append({'phase':phase,'returncode':process.returncode,'ranks':{
            f'exit.{phase}.rank{i}.json':digest(folder/f'exit.{phase}.rank{i}.json') for i in range(world)}})
    save(folder/'workers.json',workers)
    for rank in range(world):
        pending=folder/f'rank{rank}.pending.json'
        row=read(pending);row['workers_sha256']=digest(folder/'workers.json')
        require(not (folder/f'rank{rank}.json').exists(),'Final rank receipt already exists')
        save(folder/f'rank{rank}.json',row)
        pending.unlink()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);sub=cli.add_subparsers(dest='command',required=True)
    create=sub.add_parser('make');create.add_argument('--output',required=True);create.add_argument('--source',required=True)
    create.add_argument('--kind',choices=('tp','pp','ep','combined','storage'),required=True)
    create.add_argument('--dtype',choices=('float32','bfloat16'),required=True)
    create.add_argument('--reference-dtype',choices=('float32',))
    create.add_argument('--optimizer-backend',choices=('distributed-adam','cpu-offload'),default='distributed-adam')
    for cmd in ('worker','launch'):
        run=sub.add_parser(cmd);run.add_argument('--plan',required=True);run.add_argument('--stage',required=True)
        if cmd=='worker':run.add_argument('--phase',choices=('continuous','prefix','suffix'),required=True)
    args=cli.parse_args()
    if args.command=='make':print(make(args.output,args.kind,args.source,args.dtype,args.optimizer_backend,args.reference_dtype))
    elif args.command=='worker':worker(args.plan,args.stage,args.phase)
    else:launch(args.plan,args.stage)
