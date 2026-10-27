"""Independent first-step expert GEMM replay in NumPy float64.

Reads actual native parameter names, local expert IDs and optimizer ranges.
It does not import the validator's canonical mapper or report metric code.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text())


def metrics(a, b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    assert a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all()
    d=b-a
    absolute=float(np.linalg.norm(d.reshape(-1)))
    base=float(np.linalg.norm(a.reshape(-1)))
    return dict(relative_l2=absolute/base if base else (0. if not absolute else None),
                absolute_l2=absolute,max_abs=float(np.max(np.abs(d))),rmse=absolute/np.sqrt(a.size))


def load(root, stage):
    rows, grads, seen, sources = {}, {}, {}, []
    for rank in range(2):
        path=root/stage/f'linear-trace.rank{rank}.npz'
        index=read(path.with_suffix('.json'))
        sources.append(dict(rank=rank,stage=stage,npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            index_sha256=hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest()))
        with np.load(path,allow_pickle=False) as arrays:
            for row in index['rows']:
                item=dict(rank=rank,call=row['call'])
                for field in ('input','output','upstream','weight'):
                    value=arrays[row['prefix']+'/'+field]
                    assert np.isfinite(value).all()
                    item[field]=value.reshape(-1,value.shape[-1]).copy()
                assert item['input'].shape[0]==item['output'].shape[0]==item['upstream'].shape[0]
                assert item['weight'].shape==(item['output'].shape[1],item['input'].shape[1])
                rows.setdefault(row['name'],[]).append(item)
            for row in index['gradients']:
                name=row['name'];n=int(np.prod(row['shape']))
                grads.setdefault(name,np.zeros(n,dtype=np.float64))
                seen.setdefault(name,np.zeros(n,dtype=bool))
                start,end=row['start'],row['end'];values=arrays[row['prefix']].reshape(-1)
                assert 0<=start<end<=n and end-start==values.size and not seen[name][start:end].any()
                grads[name][start:end]=values;seen[name][start:end]=True
    assert set(rows)==set(grads) and len(rows)==16 and all(mask.all() for mask in seen.values())
    result={}
    for name,values in rows.items():
        weight=values[0]['weight']
        assert all(np.array_equal(weight,r['weight']) for r in values)
        x=np.concatenate([r['input'] for r in values]);y=np.concatenate([r['output'] for r in values])
        dy=np.concatenate([r['upstream'] for r in values])
        # Sort by complete input/upstream bytes before the FP64 reduction so
        # row permutations alone do not create a replay summation difference.
        order=sorted(range(len(x)),key=lambda i:(x[i].tobytes(),dy[i].tobytes()))
        x,y,dy=x[order],y[order],dy[order]
        result[name]=dict(x=x,y=y,dy=dy,weight=weight,native_gradient=grads[name].reshape(weight.shape),
                          replay_gradient=dy.astype(np.float64).T@x.astype(np.float64),
                          calls=[dict(rank=r['rank'],call=r['call'],rows=len(r['input'])) for r in values])
    return result,sources


def audit(root):
    stages={};sources=[]
    for stage in ('reference','repeat','candidate'):
        stages[stage],files=load(root,stage);sources+=files
    denominator=read(root/'inputs/input.json')['supervised_units']
    out=[]
    for name,ref in stages['reference'].items():
        for stage in ('repeat','candidate'):
            other=stages[stage][name]
            assert np.array_equal(ref['weight'],other['weight'])
            equal=np.array_equal(ref['x'],other['x'])
            # The gradient replay uses the true global supervised-token count;
            # this is a diagnostic reference, never a replacement gradient.
            rr,cr=ref['replay_gradient']/denominator,other['replay_gradient']/denominator
            row=dict(stage=stage,name=name,rows_reference=len(ref['x']),rows_candidate=len(other['x']),
                     input_rows_identical_after_alignment=equal,
                     reference_calls=ref['calls'],candidate_calls=other['calls'],
                     native_gradient_difference=metrics(ref['native_gradient'],other['native_gradient']),
                     replay_gradient_difference=metrics(rr,cr),
                     reference_native_vs_fp64=metrics(rr,ref['native_gradient']),
                     candidate_native_vs_fp64=metrics(cr,other['native_gradient']))
            if equal:
                row['output_difference']=metrics(ref['y'],other['y'])
                row['upstream_difference']=metrics(ref['dy'],other['dy'])
            out.append(row)
    return dict(scope='first_step_full_expert_linears_EP1_vs_EP2',sources=sources,
                supervised_units=denominator,rows=out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    result=audit(args.results)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([r for r in result['rows'] if r['stage']=='candidate'],indent=2,allow_nan=False))
