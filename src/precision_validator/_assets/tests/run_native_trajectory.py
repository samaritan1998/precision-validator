"""Exercise actual Slime distributed trajectories; small model integration only."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from precision_validator.common import ASSET_ROOT, require, save, now
from precision_validator.native_trajectory import prepare
from precision_validator.native_trajectory_compare import compare_suite


def main(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    require(len(devices) == (8 if args.kind == 'combined' else 2), 'Native trajectory allocation differs')
    with (root/'prepare.log').open('x') as stream:
        precision_args = ['--reference-dtype', args.reference_dtype] if args.reference_dtype else []
        subprocess.run([sys.executable,str(ASSET_ROOT/'examples/slime_trajectory.py'),'make','--output',str(root/args.kind),
                        '--source',args.slime_source,'--kind',args.kind,'--dtype',args.dtype,'--optimizer-backend',args.optimizer_backend,*precision_args],
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[0]),stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
    prepare(root/args.kind/'spec.json',root/args.kind/'results',timeout=1200,budget_gib=1)
    plan = root/args.kind/'results/plan.json'
    processes, streams = [], []
    try:
        for i, stage in enumerate(('reference','repeat')):
            stream = (root/f'{stage}-launcher.log').open('x');streams.append(stream)
            processes.append(subprocess.Popen([sys.executable,'-m','precision_validator','run','--plan',str(plan),'--stage',stage],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[i]),stdout=stream,stderr=subprocess.STDOUT))
        for process in processes:
            require(process.wait(timeout=1260)==0,'Native trajectory reference/repeat failed')
        for stage in ('candidate','resume'):
            subprocess.run([sys.executable,'-m','precision_validator','run','--plan',str(plan),'--stage',stage],check=True,timeout=1260)
        report = compare_suite(plan)
        save(root/'receipt.json',{'status':report['status'],'utc':now(),'kind':args.kind,'parameter_dtype':args.dtype,
             'reference_dtype':args.reference_dtype or args.dtype,
             'optimizer_backend':args.optimizer_backend,'plan':str(plan),'scope':report['scope'],'coverage':report['coverage'],
             'comparisons':{step:{pair:{field:{k:values[k] for k in ('max_abs','absolute_l2','rmse','relative_l2')}
                 for field,values in fields.items()} for pair,fields in pairs.items()} for step,pairs in report['comparisons'].items()},
             'failed_gates':{k:v for k,v in report['gates'].items() if not v['passed']}})
        return 0 if report['status']=='passed' else 3
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            if process.poll() is None:process.wait(timeout=30)
        for stream in streams:stream.close()


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',required=True);cli.add_argument('--slime-source',required=True)
    cli.add_argument('--kind',choices=('tp','pp','ep','combined','storage'),required=True)
    cli.add_argument('--dtype',choices=('float32','bfloat16'),required=True)
    cli.add_argument('--reference-dtype',choices=('float32',))
    cli.add_argument('--optimizer-backend',choices=('distributed-adam','cpu-offload'),default='distributed-adam')
    sys.exit(main(cli.parse_args()))
