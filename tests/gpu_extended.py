"""Run independent release checks on disjoint GPUs in one eight-GPU allocation."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import zipfile

from precision_validator.common import ASSET_ROOT, require, save, digest, now


def main(args):
    import torch
    import precision_validator
    require(torch.cuda.device_count() == 8, 'Extended suite needs exactly eight visible GPUs')
    package = Path(precision_validator.__file__).parent.parent
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith('precision_validator/'):
                require((package/name).read_bytes() == archive.read(name), 'Installed package differs from wheel')
    out = args.output.resolve() / ('execution-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    save(args.output/'latest-execution.json',dict(path=str(out),utc=now()))
    here=Path(__file__).resolve().parent
    devices=os.environ.get('CUDA_VISIBLE_DEVICES',','.join(map(str,range(8)))).split(',')
    clean=dict(os.environ)
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','PET_NODE_RANK','PET_NNODES','PET_NPROC_PER_NODE','RAY_ADDRESS'):
        clean.pop(key,None)
    clean.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUDA_DEVICE_MAX_CONNECTIONS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    trajectory=[sys.executable,str(here/'gpu_native_trajectory.py'),'--slime-source',str(args.slime_source),'--dtype','bfloat16']
    tasks={
        'ray':([sys.executable,str(here/'gpu_slime_ray.py'),'launch','--source',str(args.ray_source),'--output',str(out/'ray')],devices[:2]),
        'fp32_bf16_storage':([*trajectory,'--kind','storage','--reference-dtype','float32','--output',str(out/'storage')],devices[2:4]),
        'cpu_offload_resume':([*trajectory,'--kind','tp','--optimizer-backend','cpu-offload','--output',str(out/'offload')],devices[6:8]),
    }
    for i,family in enumerate(('llama','gemma'),4):
        tasks['hf_'+family]=([sys.executable,str(ASSET_ROOT/'examples/hf_causal_lm/test_adapter.py'),
            '--family',family,'--device','cuda','--output',str(out/('hf_'+family)),
            '--allowed-device-name',torch.cuda.get_device_name(i),'--capability',*map(str,torch.cuda.get_device_capability(i))],[devices[i]])
    processes=[]
    record=dict(status='running',wheel_sha256=digest(args.wheel),version=precision_validator.__version__,
        device=torch.cuda.get_device_name(),started_utc=now(),cases={})
    save(out/'qualification.json',record)
    try:
        for name,(command,visible) in tasks.items():
            stream=(out/(name+'.log')).open('xb')
            p=subprocess.Popen(command,env=dict(clean,CUDA_VISIBLE_DEVICES=','.join(visible)),
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((name,p,stream))
        deadline=time.monotonic()+4300
        while any(p.poll() is None for _,p,_ in processes):
            save(out/'progress.json',dict(utc=now(),exit_codes={name:p.poll() for name,p,_ in processes}))
            require(time.monotonic()<deadline,'Extended suite timeout')
            time.sleep(10)
        record['cases']={name:dict(exit_code=p.returncode,status='passed' if p.returncode==0 else 'failed')
                         for name,p,_ in processes}
        record['status']='passed' if all(p.returncode==0 for _,p,_ in processes) else 'failed'
        record['finished_utc']=now()
        save(out/'qualification.json',record)
        require(record['status']=='passed','Extended suite has failures; inspect individual logs')
    finally:
        for _,p,_ in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for _,p,stream in processes:
            if p.poll() is None:
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=15)
            stream.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','wheel','slime-source','ray-source'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
