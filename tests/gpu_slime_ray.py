"""Exercise the public Slime train.py -> Ray -> rank bridge with local fake data.

No pretrained model or external dataset is downloaded. The fixture uses a
small random Llama checkpoint and masked multi-turn text examples.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def make(root):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    root.mkdir(parents=True, exist_ok=False)
    words = ['[UNK]', '[BOS]', '[EOS]', '[PAD]', 'user', 'assistant', ':', 'FOR', 'TESTING', 'ONLY']
    words += [f'w{i}' for i in range(118)]
    vocabulary = {word:i for i,word in enumerate(words)}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', bos_token='[BOS]',
        eos_token='[EOS]', pad_token='[PAD]', model_max_length=64)
    tokenizer.chat_template = "{% for message in messages %}{{ '[BOS] ' + message['role'] + ' : ' + message['content'] + ' [EOS] ' }}{% endfor %}{% if add_generation_prompt %}{{ '[BOS] assistant : ' }}{% endif %}"
    checkpoint = root / 'hf'
    tokenizer.save_pretrained(checkpoint)
    torch.manual_seed(1979)
    config = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=32, max_position_embeddings=64,
        attention_dropout=0., rms_norm_eps=1e-6, rope_theta=10000., tie_word_embeddings=False,
        bos_token_id=1, eos_token_id=2, pad_token_id=3)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(checkpoint)
    rows = []
    for sample in range(40):
        rows.append({'messages':[
            {'role':'user','content':' '.join(f'w{(sample+j)%100}' for j in range(3 + sample%5))},
            {'role':'assistant','content':' '.join(f'w{(sample+j+10)%100}' for j in range(4 + sample%7))},
            {'role':'user','content':'w1 w2'},
            {'role':'assistant','content':'w3 w4 w5'}]})
    (root / 'data.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    limit = dict(relative_l2=.05, max_abs=.003, near_zero_rms=1e-8, near_zero_max_abs=1e-8)
    policy = {field:dict(repeat_relative_ceiling=1e-6, target_relative_ceiling=.02, noise_multiplier=2,
        noise_slack=.02, max_abs_ceiling=.003) for field in ('gradients','updates')}
    acceptance = {category:[dict(pattern='*', gradients=limit, updates=limit)] for category in ('parameters','groups')}
    acceptance['observations'] = {'loss':limit}
    (root / 'policy.json').write_text(json.dumps(dict(policy=policy, acceptance=acceptance),indent=2)+'\n')
    return root


def arguments(source, root, output):
    return [str(source / 'train.py'), '--actor-num-nodes','1','--actor-num-gpus-per-node','2',
        '--hf-checkpoint',str(root/'hf'), '--load',str(root/'hf'), '--megatron-to-hf-mode','bridge',
        '--num-layers','2','--hidden-size','128','--ffn-hidden-size','256','--num-attention-heads','4',
        '--group-query-attention','--num-query-groups','4','--kv-channels','32','--seq-length','64',
        '--normalization','RMSNorm','--norm-epsilon','1e-6','--swiglu','--disable-bias-linear',
        '--untie-embeddings-and-output-weights','--position-embedding-type','rope','--rotary-base','10000',
        '--vocab-size','128','--make-vocab-size-divisible-by','64',
        '--rollout-function-path','slime.rollout.sft_rollout.generate_rollout',
        '--prompt-data',str(root/'data.jsonl'),'--input-key','messages','--num-rollout','5',
        '--rollout-batch-size','8','--global-batch-size','8','--micro-batch-size','1',
        '--n-samples-per-prompt','1','--loss-type','sft_loss','--calculate-per-token-loss',
        '--disable-compute-advantages-and-returns','--debug-train-only','--loss-mask-type','qwen',
        '--rollout-max-context-len','64','--tensor-model-parallel-size','2','--expert-tensor-parallel-size','2',
        '--pipeline-model-parallel-size','1','--context-parallel-size','1','--expert-model-parallel-size','1',
        '--attention-backend','flash','--attention-dropout','0','--hidden-dropout','0',
        '--accumulate-allreduce-grads-in-fp32','--no-gradient-accumulation-fusion','--data-pad-size-multiplier','1',
        '--optimizer','adam','--lr','0.001','--min-lr','0.001','--lr-decay-style','constant',
        '--weight-decay','0','--adam-beta1','0.9','--adam-beta2','0.999','--adam-eps','0.0001','--clip-grad','0',
        '--precision-check-steps','3','--precision-check-reference','tp=1,etp=1',
        '--precision-check-dir',str(output),'--precision-check-policy',str(root/'policy.json'),
        '--precision-check-disk-gib','1','--precision-check-timeout','1200']


def launch(source, output):
    import ray
    import torch
    from precision_validator.common import require, read, save, now
    require(torch.cuda.device_count() >= 2, 'Ray driver test needs two GPUs')
    root = make(output.parent / (output.name + '-inputs'))
    source = source.resolve()
    env = dict(os.environ)
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','PET_NODE_RANK','PET_NNODES','PET_NPROC_PER_NODE'):
        env.pop(key,None)
    env.update(PYTHONPATH=str(source)+os.pathsep+os.environ['MEGATRON_PATH'],
        CUDA_DEVICE_MAX_CONNECTIONS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', RAY_USAGE_STATS_ENABLED='0',
        NVTE_ALLOW_NONDETERMINISTIC_ALGO='0', WANDB_MODE='disabled',
        RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES='1',
        RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES='1',
        RAY_EXPERIMENTAL_NOSET_SUPA_VISIBLE_DEVICES='1')
    # This process owns the local Ray cluster; shutdown only this cluster.
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','PET_NODE_RANK','PET_NNODES','PET_NPROC_PER_NODE','RAY_ADDRESS'):
        os.environ.pop(key, None)
    os.environ.update({key:value for key,value in env.items() if key != 'RAY_ADDRESS'})
    context = ray.init(address='local', num_cpus=16, num_gpus=2, include_dashboard=False, log_to_driver=True)
    env['RAY_ADDRESS'] = context.address_info['address']
    command = [sys.executable, *arguments(source,root,output)]
    save(root/'invocation.json',dict(command=command,started_utc=now()))
    try:
        with (root/'driver.log').open('xb') as log:
            result = subprocess.run(command,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT,timeout=3900)
        save(root/'process.json',dict(exit_code=result.returncode,finished_utc=now()))
        require(result.returncode==0, 'Full Slime Ray driver failed; inspect inputs/driver.log')
        report=read(output/'acceptance.json')
        require(report['status']=='passed', 'Full Slime Ray numerical gates failed')
        print('SLIME_RAY_RESULT',report['status'],flush=True)
    finally:
        ray.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('make','launch','args'))
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    cli=parser.parse_args()
    if cli.action=='make': make(cli.output)
    elif cli.action=='args': print(json.dumps(arguments(cli.source,cli.output,cli.output/'results')))
    else: launch(cli.source,cli.output)
