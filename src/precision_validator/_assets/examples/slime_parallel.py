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


def make(output, kind, source, dtype, diagnose=False, optimizer_backend='distributed-adam'):
    require(optimizer_backend in ('distributed-adam','cpu-offload'), 'Unknown native optimizer backend')
    require(optimizer_backend!='cpu-offload' or dtype=='bfloat16', 'CPU offload fixture requires BF16 storage')
    source = Path(source).resolve()
    require((source/'slime/backends/megatron_utils/model.py').is_file(), 'Slime source is missing')
    def factory(kind, layout):
        return native.model_for(kind,layout,dtype=getattr(torch,dtype),position_embedding_type='rope',sequence_length=32)
    spec_path = native.make(output,kind,model_factory=factory,weight_dtype=getattr(torch,dtype))
    spec = read(spec_path)
    root = Path(output)
    config = read(root/'model.json')
    config.update(parameter_dtype=dtype,sequence_length=32,position_embedding_type='rope',entry='slime.train_one_step',
                  diagnose_routing=diagnose,optimizer_backend=optimizer_backend)
    save(root/'model.json',config)
    spec['name'] = f'slime-native-{dtype}-{kind}'
    spec['artifacts'].update(code=str(Path(__file__).resolve()),native_helper=str(Path(__file__).resolve().with_name('megatron_parallel.py')),
                             slime_source=str(source),megatron_training=str(Path(__import__('megatron.training', fromlist=['__file__']).__file__).resolve().parent))
    spec['command'][6] = str(Path(__file__).resolve())
    require('worker' in spec['command'], 'Unexpected native command')
    spec['parallel_adapter']['framework'] = 'native-slime-train-one-step-megatron-distributed-adam'
    spec['contract']['precision']['parameters'] = dtype
    spec['required_checks'] += ['native_slime_entry_executed','native_scheduler_executed','canonical_initial_hashes',
                               'native_gradient_clear_completed']
    if optimizer_backend=='cpu-offload':
        spec['contract']['optimizer']['native_backend'] = {'precision_aware':True,'cpu_offload_fraction':1.0,
                                                          'overlap_cpu_optimizer_d2h_h2d':True}
        spec['required_checks'].append('native_cpu_optimizer_offload_executed')
    # Predeclared integration bounds for BF16 storage. No production tolerance
    # is inferred from this fixture; changing bounds requires a new experiment.
    if dtype == 'bfloat16':
        limit = {'relative_l2':0.05,'max_abs':0.003,'near_zero_rms':1e-8,'near_zero_max_abs':1e-8}
        for rows in (spec['acceptance']['parameters'],spec['acceptance']['groups']):
            for row in rows:
                for field in ('gradients','updates'):row[field]=copy.deepcopy(limit)
        spec['acceptance']['observations'] = {k:copy.deepcopy(limit) for k in spec['observables']}
        for row in spec['policy'].values():
            row.update(target_relative_ceiling=0.02,noise_slack=0.02,max_abs_ceiling=0.003)
    save(spec_path,spec)
    return spec_path


def worker(path, stage):
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.transformer.module import Float16Module
    from megatron.training import global_vars
    plan = load_plan(path)
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
    dtype = getattr(torch,model_config['parameter_dtype'])
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
    initial = capture_initial(mapping,writer,dtype)
    options = plan['contract']['optimizer']['options']
    backend = model_config.get('optimizer_backend','distributed-adam')
    require(backend in ('distributed-adam','cpu-offload'), 'Unknown native optimizer backend')
    backend_options = (dict(use_precision_aware_optimizer=True,optimizer_cpu_offload=True,
                           optimizer_offload_fraction=1.0,overlap_cpu_optimizer_d2h_h2d=True)
                       if backend=='cpu-offload' else {})
    optimconfig = OptimizerConfig(optimizer='adam',lr=options['lr'],min_lr=options['lr'],weight_decay=0.,
        adam_beta1=options['betas'][0],adam_beta2=options['betas'][1],adam_eps=options['eps'],
        clip_grad=0.,params_dtype=dtype,use_distributed_optimizer=True,bf16=dtype==torch.bfloat16,fp16=False,
        overlap_param_gather=False,log_num_zeros_in_grad=False,**backend_options)
    optimizer = get_megatron_optimizer(optimconfig,[wrapped])
    inners = getattr(optimizer,'chained_optimizers',[optimizer])
    if backend=='cpu-offload':
        require(all(type(i.optimizer).__name__=='HybridDeviceOptimizer' and i.optimizer.offload_fraction==1.0
                    and i.optimizer.cpu_optimizers and i.optimizer.gpu_optimizer is None for i in inners),
                'Native full CPU optimizer offload was not enabled')
    scheduler = OptimizerParamScheduler(optimizer,init_lr=options['lr'],max_lr=options['lr'],min_lr=options['lr'],
        lr_warmup_steps=0,lr_decay_steps=100,lr_decay_style='constant',start_wd=0.,end_wd=0.,wd_incr_steps=100,wd_incr_style='constant')
    args = SimpleNamespace(custom_megatron_before_train_step_hook_path=None,data_pad_size_multiplier=1,
        allgather_cp=False,qkv_format='bshd',position_embedding_type='rope',uses_unsplit_forward=False,
        loss_type='sft_loss',calculate_per_token_loss=True,enable_mtp_training=False,seq_length=32,micro_batch_size=1,
        decoder_seq_length=None,check_for_nan_in_loss_and_grad=True,ci_test=False,rollout_top_p=1.,
        rollout_temperature=1.,log_probs_chunk_size=32,recompute_loss_function=False)
    global_vars._GLOBAL_ARGS = args
    dp_rank = ps.get_data_parallel_rank()
    samples = list(range(dp_rank,4,layout['dp']))
    current_sample = {'id':None}
    class Iterator(DataIterator):
        def get_next(self, keys):
            current_sample['id'] = samples[self.offset]
            # BSHD requires padding metadata; add it through the public iterator.
            return super().get_next([*keys,'max_seq_lens'])
    rollout = {'tokens':[torch.tensor(data['tokens'][i],device='cuda') for i in samples],
        'total_lengths':[17]*len(samples),'response_lengths':[16]*len(samples),
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
        require(num_tokens is not None and int(num_tokens)==54,'Slime global supervised denominator differs')
        tokens.append(int(num_tokens))
    config.finalize_model_grads_func = finalize
    config.grad_scale_func = optimizer.scale_loss
    if ps.is_pipeline_last_stage():
        def output_hook(module,args,result):
            value = result[0] if isinstance(result,tuple) else result
            require(list(value.shape[:2])==[32,1],'Slime native logits layout differs')
            outputs.append({'sample':current_sample['id'],'tp_rank':ps.get_tensor_model_parallel_rank(),
                            'values':value.detach()[[0,5,10,15],0,:].float().cpu().tolist()})
        handles.append(model.output_layer.register_forward_hook(output_hook))
    with NativeStepObserver(mapping,optimizer,writer,plan['contract']['precision']) as observed:
        losses,grad_norm = train_one_step(args,0,0,iterator,[wrapped],optimizer,scheduler,len(samples),4)
    require(tokens==[54] and len(forward_calls)==len(backward_calls)==len(samples)*(2//layout['pp']),
            'Slime forward/backward/finalization did not fully execute')
    require(math.isfinite(float(grad_norm)) and scheduler.num_steps==4,'Slime gradient norm/scheduler differs')
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
        for sample in range(4):
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
    torch.cuda.synchronize();dist.barrier();ps.destroy_model_parallel();dist.destroy_process_group()
    writer.finish(execution={'backward_completed':True,'gradients_reduced':True,'optimizer_succeeded':observed.updated,
        'optimizer_steps':1,'teardown_completed':not dist.is_initialized()},precision=observed.precision,
        device={'kind':'cuda','name':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability())},
        checks={key:True for key in plan['required_checks']},native_parallelism=groups,
        timings={'native_global_tokens':tokens,'local_samples':samples,'factory_parameters':mapping.factories,
                 'entry_file':str(entry),'entry_sha256':expected,'observer':observed.receipt,'initial':initial,
                 'scheduler_samples':scheduler.num_steps,'router_diagnostics':routing,'optimizer_backend':optimizer_backend})
    print('SLIME_NATIVE_RANK_COMPLETE',stage,os.environ['RANK'],flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);sub=cli.add_subparsers(dest='command',required=True)
    create=sub.add_parser('make');create.add_argument('--output',required=True);create.add_argument('--source',required=True)
    create.add_argument('--kind',choices=('tp','pp','ep','combined'),required=True)
    create.add_argument('--dtype',choices=('float32','bfloat16'),required=True)
    create.add_argument('--diagnose-routing',action='store_true')
    create.add_argument('--optimizer-backend',choices=('distributed-adam','cpu-offload'),default='distributed-adam')
    run=sub.add_parser('worker');run.add_argument('--plan',required=True);run.add_argument('--stage',required=True)
    args=cli.parse_args()
    if args.command=='make':print(make(args.output,args.kind,args.source,args.dtype,args.diagnose_routing,args.optimizer_backend))
    else:worker(args.plan,args.stage)
