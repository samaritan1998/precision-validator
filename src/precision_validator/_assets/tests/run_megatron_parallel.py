"""Run native Megatron stages, with independent reference and repeat in parallel."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from precision_validator.common import ASSET_ROOT, save, require, now
from precision_validator.protocol import prepare
from precision_validator.protocol_compare import compare_suite


def main(output,kind,source=None,dtype='float32',diagnose=False,optimizer_backend='distributed-adam'):
    require(source or optimizer_backend=='distributed-adam','Optimizer backend option requires Slime integration')
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    folder=root/kind
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    require(len(devices)==(8 if kind=='combined' else 2),'Native integration allocation differs')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[0])
    with (root/'prepare.log').open('x') as stream:
        command=[sys.executable,str(ASSET_ROOT/('examples/slime_parallel.py' if source else 'examples/megatron_parallel.py')),
                 'make','--kind',kind,'--output',str(folder)]
        if source:command+=['--source',source,'--dtype',dtype,'--optimizer-backend',optimizer_backend]
        if diagnose:command+=['--diagnose-routing']
        subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
    prepare(folder/'spec.json',folder/'results',timeout=600,budget_gib=0.2,retain_raw=True)
    plan=folder/'results/plan.json'
    processes=[]
    streams=[]
    try:
        for i,stage in enumerate(('reference','repeat')):
            stream=(root/f'{stage}-launcher.log').open('x');streams.append(stream)
            p=subprocess.Popen([sys.executable,'-m','precision_validator','run','--plan',str(plan),'--stage',stage],
                               env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[i]),stdout=stream,stderr=subprocess.STDOUT)
            processes.append(p)
        for p in processes:
            require(p.wait(timeout=660)==0,'Native reference/repeat stage failed; inspect stage logs')
        subprocess.run([sys.executable,'-m','precision_validator','run','--plan',str(plan),'--stage','candidate'],check=True,timeout=660)
        report=compare_suite(plan,retain_raw=True)
        save(root/'receipt.json',{'status':report['status'],'utc':now(),'plan':str(plan),
            'scope':'small_native_megatron_gpt_and_distributed_adam_not_production_qwen_acceptance',
            'kind':kind,'entry':'slime' if source else 'megatron','parameter_dtype':dtype,'optimizer_backend':optimizer_backend,
            'trainable_elements':report['trainable_elements'],
            'parallel_validation':report['parallel_validation'],'acceptance_coverage':report['acceptance_coverage'],
            'comparisons':{pair:{field:{key:metric[key] for key in ('max_abs','relative_l2','absolute_l2','rmse')}
                                 for field,metric in fields.items()} for pair,fields in report['comparisons'].items()},
            'failed_gates':{name:value for name,value in report['gates'].items() if not value['passed']}})
        require(report['status'] in ('passed','numerical_failure'),'Unexpected numerical report status')
        if report['status']=='numerical_failure':
            print('NATIVE_NUMERICAL_FAILURE_REPORTED',kind,str(root),flush=True)
            return 3
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            if p.poll() is None:p.wait(timeout=30)
        for stream in streams:stream.close()
    print('MEGATRON_INTEGRATION_COMPLETE',kind,str(root),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',required=True)
    cli.add_argument('--kind',choices=('tp','pp','ep','combined'),required=True)
    cli.add_argument('--slime-source');cli.add_argument('--dtype',choices=('float32','bfloat16'),default='float32')
    cli.add_argument('--diagnose-routing',action='store_true')
    cli.add_argument('--optimizer-backend',choices=('distributed-adam','cpu-offload'),default='distributed-adam')
    args=cli.parse_args();sys.exit(main(args.output,args.kind,args.slime_source,args.dtype,args.diagnose_routing,args.optimizer_backend))
