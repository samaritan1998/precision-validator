"""Native Megatron GPT, distributed Adam and TP/PP/learned-route EP evidence.

Small random models exercise framework integration, not Qwen 256K acceptance.
All configurations use the same frozen weights, four samples and masked token
denominator. No precision/memory patches from the production Qwen profile run.
"""
import argparse
import copy
from datetime import timedelta
import hashlib
import math
import os
from pathlib import Path
import tempfile
import sys

if os.environ.get("MEGATRON_PATH"):
    sys.path.insert(0, str(Path(os.environ["MEGATRON_PATH"]).resolve()))

import numpy as np
import torch
import torch.distributed as dist

from precision_validator.common import read, save, require, load_plan
from precision_validator.parallel import CONVENTION, parse_layout
from precision_validator.megatron_mapping import (MegatronParameterMap, collective_fragments,
    append_owned, native_groups, elect_owners, fragment_metadata)
from precision_validator.sdk import RankWriter, tensor_digest


def initialize(layout, standalone=False):
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    if standalone:
        temporary = tempfile.NamedTemporaryFile(prefix='precision-mcore-', delete=False)
        temporary.close()
        dist.init_process_group('nccl',init_method='file://' + temporary.name,rank=0,world_size=1,timeout=timedelta(minutes=5))
    else:
        dist.init_process_group('nccl',timeout=timedelta(minutes=5))
    ps.initialize_model_parallel(tensor_model_parallel_size=layout['tp'],pipeline_model_parallel_size=layout['pp'],
        context_parallel_size=layout['cp'],expert_model_parallel_size=layout['ep'],expert_tensor_parallel_size=layout['etp'])
    torch.manual_seed(20261019)
    model_parallel_cuda_manual_seed(20261019)
    return native_groups(layout)


def model_for(kind, layout, *, dtype=torch.float32, position_embedding_type='learned_absolute', sequence_length=16):
    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_config import TransformerConfig
    moe = kind in ('ep','combined')
    config = TransformerConfig(num_layers=2,hidden_size=32,num_attention_heads=4,num_query_groups=2,
        ffn_hidden_size=64,kv_channels=8,tensor_model_parallel_size=layout['tp'],pipeline_model_parallel_size=layout['pp'],
        context_parallel_size=1,expert_model_parallel_size=layout['ep'],expert_tensor_parallel_size=layout['etp'],
        sequence_parallel=moe and layout['tp'] > 1,params_dtype=dtype,pipeline_dtype=dtype,bf16=dtype==torch.bfloat16,
        hidden_dropout=0.0,attention_dropout=0.0,add_bias_linear=False,normalization='RMSNorm',
        gated_linear_unit=True,activation_func=torch.nn.functional.silu,
        gradient_accumulation_fusion=False,bias_activation_fusion=False,bias_dropout_fusion=False,
        masked_softmax_fusion=False,apply_rope_fusion=False,calculate_per_token_loss=True,
        num_moe_experts=4 if moe else None,moe_router_topk=2,moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type='none',moe_token_dispatcher_type='alltoall',
        moe_grouped_gemm=False,moe_permute_fusion=False,moe_router_dtype='fp32')
    spec = get_gpt_layer_local_spec(num_experts=config.num_moe_experts,normalization='RMSNorm',moe_grouped_gemm=False)
    if kind == 'combined':
        # WrappedTorchNorm deliberately rejects sequence parallel. Select the
        # framework's supported TE Norm in every stage of this experiment.
        from megatron.core.extensions.transformer_engine import TENorm
        spec.submodules.input_layernorm = TENorm
        spec.submodules.pre_mlp_layernorm = TENorm
    model = GPTModel(config,spec,vocab_size=64,max_sequence_length=sequence_length,pre_process=ps.is_pipeline_first_stage(),
        post_process=ps.is_pipeline_last_stage(),share_embeddings_and_output_weights=False,
        parallel_output=True,position_embedding_type=position_embedding_type).cuda().train()
    return model, config


def target_layout(kind):
    if kind == 'combined':
        return parse_layout('tp=2,pp=2,cp=1,ep=2,etp=2',8),8
    if kind == 'ep':
        return parse_layout('tp=1,pp=1,cp=1,ep=2,etp=1',2),2
    return parse_layout(f'tp={2 if kind == "tp" else 1},pp={2 if kind == "pp" else 1},cp=1,ep=1,etp={2 if kind == "tp" else 1}',2),2


def make(output, kind, *, model_factory=None, weight_dtype=torch.float32):
    root = Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    reference = parse_layout('tp=1,pp=1,cp=1,ep=1,etp=1',1)
    candidate,world = target_layout(kind)
    initialize(reference,standalone=True)
    model,config = (model_factory or model_for)(kind,reference)
    mapping = MegatronParameterMap(model)
    catalog = mapping.catalog()
    weights = {}
    for name,info in sorted(catalog.items()):
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little')
        rng = np.random.default_rng(seed)
        value = (rng.standard_normal(info['shape']) * 0.02).astype(np.float32)
        if 'layernorm' in name or 'layer_norm' in name:
            value += 1.0
        value = torch.from_numpy(value).to(weight_dtype).float().numpy()
        weights[name] = value
    np.savez(root/'weights.npz',**weights)
    rng = np.random.default_rng(1979)
    ids = rng.integers(0,64,(4,17)).tolist()
    masks = [[0] * prefix + [1] * (16-prefix) for prefix in (1,3,2,4)]
    save(root/'input.json',{'kind':kind,'tokens':ids,'loss_masks':masks,'supervised_units':54,
                            'source':'seeded_integration_fixture_not_production_data'})
    save(root/'model.json',{'kind':kind,'num_layers':2,'hidden_size':32,'vocab_size':64,'sequence_length':16,
                          'experts':config.num_moe_experts,'norm_backend':'native_te' if kind=='combined' else 'native_local',
                          'scope':'native_megatron_small_model'})
    limit = {'relative_l2':0.003,'max_abs':2e-5,'near_zero_rms':1e-8,'near_zero_max_abs':1e-8}
    acceptance = {category:[{'pattern':'*',**{f:copy.deepcopy(limit) for f in ('gradients','updates')}}]
                  for category in ('parameters','groups')}
    acceptance['observations'] = {name:copy.deepcopy(limit) for name in ('loss','sampled_logits')}
    policy = {f:{'repeat_relative_ceiling':1e-6,'target_relative_ceiling':0.003,'noise_multiplier':2,
                 'noise_slack':0.003,'max_abs_ceiling':2e-5} for f in ('gradients','updates')}
    import megatron.core
    spec = {'schema_version':1,'name':f'megatron-native-{kind}','scope':'integration_test',
        'artifacts':{'code':str(Path(__file__).resolve()),'input':str(root/'input.json'),
                     'weights':str(root/'weights.npz'),'model_config':str(root/'model.json'),
                     'framework':str(Path(megatron.core.__file__).parent)},'parameters':catalog,
        'configurations':{'reference':{'world_size':1,'device':'cuda','parallelism':reference},
                          'candidate':{'world_size':world,'device':'cuda','parallelism':candidate}},
        'parallel_adapter':{'convention':CONVENTION,'framework':'native-megatron-core-gpt-distributed-adam',
                            'supported_axes':['tp','pp','ep','etp','dp']},
        'command':['{python}','-m','torch.distributed.run','--standalone','--nproc_per_node','{world_size}',
                   str(Path(__file__).resolve()),'worker','--plan','{plan}','--stage','{stage}'],
        'contract':{'seed':20261019,'gradient_point':'after_reduction_before_clipping',
            'update_point':'optimizer_parameter_after_minus_before','optimizer_steps':1,'optimizer_initial_state':'fresh',
            'optimizer':{'name':'MegatronDistributedAdam','options':{'lr':0.001,'betas':[0.9,0.999],
                'eps':0.0001,'weight_decay':0.0,'clip_grad':0.0},'epsilon_scope':'fixed_integration_fixture_not_production_recommendation'},
            'objective':{'normalization':'native_per_token_loss_sum_divided_by_global_supervised_units',
                         'global_samples':4,'supervised_units':54},
            'precision':{'parameters':'float32','gradients':'float32','optimizer_parameters':'float32','optimizer_state':'float32'}},
        'observables':{'loss':[],'sampled_logits':[4,4,4]},
        'required_checks':['native_checkpoint_mapping_complete','initial_weights_verified','native_global_token_count',
                           'native_distributed_optimizer_step','native_adam_states_fp32','native_forward_backward_executed'],
        'environment':{'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                       'CUDA_DEVICE_MAX_CONNECTIONS':'1'},'policy':policy,'acceptance':acceptance}
    save(root/'spec.json',spec)
    from megatron.core import parallel_state as ps
    ps.destroy_model_parallel(); dist.destroy_process_group()
    return root/'spec.json'


def optimizer_views(optimizer, mapping):
    names = {p:n for n,p in mapping.parameters.items()}
    values,ranges,states = {},{},{}
    for inner in getattr(optimizer,'chained_optimizers',[optimizer]):
        if getattr(inner,'is_stub_optimizer',False):
            continue
        require(hasattr(inner,'model_param_group_index_map'), 'Native optimizer is not sharded')
        for parameter,(gi,pi) in inner.model_param_group_index_map.items():
            name = names[parameter]
            require(name not in values,'Duplicate optimizer parameter')
            owned = inner._get_model_param_range_map(parameter)['param']
            master = inner.optimizer.param_groups[gi]['params'][pi]
            require(master.dtype == torch.float32 and master.numel() == owned.end-owned.start,'Unexpected FP32 optimizer shard')
            values[name] = master
            ranges[name] = (owned.start,owned.end)
            states[name] = inner.optimizer.state[master]
    return values,ranges,states


def worker(path,stage):
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    plan = load_plan(path)
    layout = plan['configurations'][stage]['parallelism']
    require(int(os.environ['WORLD_SIZE']) == plan['configurations'][stage]['world_size'],'Allocation differs from plan')
    groups = initialize(layout)
    data = read(plan['artifacts']['input']['path'])
    model,config = model_for(data['kind'],layout)
    ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True,use_distributed_optimizer=True,
        overlap_grad_reduce=False,overlap_param_gather=False,check_for_nan_in_grad=True)
    wrapped = DDP(config,ddp_config,model)
    # DDP can rebind parameter storage; map only after it has done so.
    mapping = MegatronParameterMap(model)
    catalog = plan['parameters']
    local_catalogs = [None]*dist.get_world_size()
    dist.all_gather_object(local_catalogs,mapping.catalog())
    merged = {}
    for local in local_catalogs:
        for name,info in local.items():
            require(merged.setdefault(name,info)==info,'Native canonical catalog disagreement')
    require(merged==catalog,'Native canonical trainable catalog differs from frozen plan')
    weights = np.load(plan['artifacts']['weights']['path'],allow_pickle=False)
    require(set(weights.files)==set(catalog),'Frozen weights catalog differs')
    with torch.no_grad():
        for name,p in mapping.parameters.items():
            for run in mapping.runs[name]:
                value = weights[run.name].reshape(-1)[run.start:run.end]
                p.reshape(-1)[run.local_start:run.local_end].copy_(torch.from_numpy(value).to(p.device))
    initial_fragments,initial_owners = collective_fragments(mapping,mapping.parameters,catalog)
    for run,value in initial_fragments:
        expected = torch.from_numpy(weights[run.name].reshape(-1)[run.start:run.end])
        require(torch.equal(value.detach().cpu(),expected),'Loaded canonical initial weights differ')
    writer = RankWriter(plan,stage,dist.get_rank())
    if dist.get_rank()==0:
        for name in catalog:
            writer.initial_parameter(name,'torch.float32',hashlib.sha256(weights[name].tobytes()).hexdigest())
    options = plan['contract']['optimizer']['options']
    optimconfig = OptimizerConfig(optimizer='adam',lr=options['lr'],min_lr=options['lr'],weight_decay=0.0,
        adam_beta1=options['betas'][0],adam_beta2=options['betas'][1],adam_eps=options['eps'],
        clip_grad=0.0,params_dtype=torch.float32,use_distributed_optimizer=True,bf16=False,fp16=False,
        overlap_param_gather=False,log_num_zeros_in_grad=False)
    optimizer = get_megatron_optimizer(optimconfig,[wrapped])
    optimizer.zero_grad(); wrapped.zero_grad_buffer()
    masters,ranges,states = optimizer_views(optimizer,mapping)
    before = {name:p.detach().clone() for name,p in masters.items()}
    require(all(not state or ('step' not in state or float(state['step'])==0) for state in states.values()),'Optimizer was not fresh')
    counts, forward_calls, backward_calls, output_rows = [],[],[],[]
    def finalize(chunks,num_tokens=None,**kwargs):
        finalize_model_grads(chunks,num_tokens=num_tokens,**kwargs)
        require(num_tokens is not None and int(num_tokens)==54,'Native global token denominator differs')
        counts.append(int(num_tokens))
    config.finalize_model_grads_func = finalize
    config.grad_scale_func = optimizer.scale_loss
    current_sample = {'id':None}
    if ps.is_pipeline_last_stage():
        def output_hook(module,args,result):
            logits = result[0] if isinstance(result,tuple) else result
            require(logits.ndim==3 and list(logits.shape[:2])==[16,1],'Unexpected native logits layout')
            output_rows.append({'sample':current_sample['id'],'tp_rank':ps.get_tensor_model_parallel_rank(),
                                'logits':logits.detach()[[0,5,10,15],0,:].float().cpu().tolist()})
        model.output_layer.register_forward_hook(output_hook)
    for layer in model.decoder.layers:
        layer.register_forward_hook(lambda *a:forward_calls.append(1))
        layer.register_full_backward_hook(lambda *a:backward_calls.append(1))
    dp_rank = ps.get_data_parallel_rank()
    samples = list(range(dp_rank,4,layout['dp']))
    require(len(samples)*layout['dp']==4,'Global sample partition changed')
    iterator = iter(samples)
    loss_sums = []
    def forward_step(iterator,module):
        sample = next(iterator)
        current_sample['id'] = sample
        tokens = torch.tensor(data['tokens'][sample],device='cuda',dtype=torch.long)
        mask = torch.tensor(data['loss_masks'][sample],device='cuda',dtype=torch.float32)
        attention = torch.ones((1,1,16,16),device='cuda',dtype=torch.bool).triu(1)
        output = module(tokens[:-1].unsqueeze(0),torch.arange(16,device='cuda').unsqueeze(0),attention,
                        labels=tokens[1:].unsqueeze(0))
        def loss_func(per_token):
            require(list(per_token.shape)==[1,16],'Unexpected native token loss shape')
            value = (per_token.float().flatten()*mask).sum()
            loss_sums.append(float(value.detach()))
            return value, mask.sum().to(torch.int64), {'loss_sum':value.detach()}
        return output,loss_func
    get_forward_backward_func()(forward_step_func=forward_step,data_iterator=iterator,model=[wrapped],
        num_microbatches=len(samples),seq_length=16,micro_batch_size=1,forward_only=False)
    require(counts==[54] and len(forward_calls)==len(backward_calls)==len(samples)*(2//layout['pp']),
            'Native forward/backward or gradient finalization incomplete')
    require(not optimizer.prepare_grads(),'Nonfinite native gradients')
    masters,ranges,states = optimizer_views(optimizer,mapping)
    grads = {name:p.grad for name,p in masters.items()}
    require(all(g is not None and g.dtype==torch.float32 for g in grads.values()),'Native FP32 optimizer gradients missing')
    fragments,owners = collective_fragments(mapping,grads,catalog,ranges)
    append_owned(writer,'gradients',fragments,owners)
    success = optimizer.step_with_ready_grads()
    require(success,'Native Adam step failed')
    masters,ranges,states = optimizer_views(optimizer,mapping)
    for state in states.values():
        require(all(key in state and state[key].dtype==torch.float32 and torch.isfinite(state[key]).all()
                    for key in ('exp_avg','exp_avg_sq')),'Native Adam state is missing/non-FP32/nonfinite')
        require('step' not in state or float(state['step'])==1,'Native Adam step differs')
    deltas = {name:p.detach()-before[name] for name,p in masters.items()}
    fragments,owners = collective_fragments(mapping,deltas,catalog,ranges)
    append_owned(writer,'updates',fragments,owners)
    observations = [None]*dist.get_world_size()
    dist.all_gather_object(observations,{'last':ps.is_pipeline_last_stage(),'tp_rank':ps.get_tensor_model_parallel_rank(),
                                      'dp_rank':dp_rank,'samples':samples,'loss_sums':loss_sums,'outputs':output_rows})
    if dist.get_rank()==0:
        sums = [r for r in observations if r['last'] and r['tp_rank']==0]
        require(sorted(i for r in sums for i in r['samples'])==list(range(4)),'Loss observation sample coverage differs')
        writer.observe('loss',sum(sum(r['loss_sums']) for r in sums)/54)
        joined = []
        for sample in range(4):
            parts = [part for row in observations if row['last'] for part in row['outputs'] if part['sample']==sample]
            require(sorted(p['tp_rank'] for p in parts)==list(range(layout['tp'])),'Native sampled logits partition mismatch')
            full = np.concatenate([np.asarray(p['logits']) for p in sorted(parts,key=lambda x:x['tp_rank'])],axis=-1)
            joined.append(full[:,[0,7,31,63]])
        writer.observe('sampled_logits',np.asarray(joined))
    torch.cuda.synchronize(); dist.barrier()
    ps.destroy_model_parallel(); dist.destroy_process_group()
    writer.finish(execution={'backward_completed':True,'gradients_reduced':True,'optimizer_succeeded':bool(success),
        'optimizer_steps':1,'teardown_completed':not dist.is_initialized()},precision=plan['contract']['precision'],
        device={'kind':'cuda','name':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability())},
        checks={key:True for key in plan['required_checks']},native_parallelism=groups,
        timings={'native_global_tokens':counts,'local_samples':samples,'factory_parameters':mapping.factories,
                 'optimizer_local_owned_elements':sum(p.numel() for p in masters.values())})
    print('MEGATRON_NATIVE_RANK_COMPLETE',stage,os.environ['RANK'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    create=sub.add_parser('make');create.add_argument('--output',required=True)
    create.add_argument('--kind',choices=('tp','pp','ep','combined'),required=True)
    run=sub.add_parser('worker');run.add_argument('--plan',required=True);run.add_argument('--stage',required=True)
    args=parser.parse_args()
    if args.command=='make': print(make(args.output,args.kind))
    else: worker(args.plan,args.stage)
