"""Independent canonical reconstruction of native TP/PP router observations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np



def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(case, stage, samples):
    merged, sources = {}, []
    for rank in range(8):
        path = case/stage/f'router-trace.rank{rank}.json'
        index = json.loads(path.read_text())
        sources.append(dict(rank=rank, index_sha256=sha(path), npz_sha256=sha(path.with_suffix('.npz'))))
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as raw:
            for row in index['rows']:
                key = row['step'], row['router'], row['sample_index']
                n = len(samples[row['sample_index']])
                offset = row['sequence_start']
                prefix = row['prefix']
                input_ = raw[prefix+'/input'].reshape(-1,32)
                count = max(0, min(len(input_), n-offset))
                value = merged.setdefault(key, dict(input=np.empty((n,32),dtype=np.float32),
                    probabilities=np.empty((n,4),dtype=np.float32), routes=np.empty((n,4),dtype=np.bool_),
                    seen=np.zeros(n,dtype=np.bool_), weight=raw[prefix+'/weight'].copy()))
                assert np.array_equal(value['weight'], raw[prefix+'/weight']), 'Router weight replicas differ'
                for field in ('input','probabilities','routes'):
                    source = input_ if field == 'input' else raw[prefix+'/'+field]
                    prior = value[field][offset:offset+count]
                    seen = value['seen'][offset:offset+count]
                    assert np.array_equal(prior[seen], source[:count][seen]), 'Router TP/DP replicas disagree'
                    value[field][offset:offset+count] = source[:count]
                value['seen'][offset:offset+count] = True
    assert len(merged) == 3*2*8, len(merged)
    assert all(x['seen'].all() for x in merged.values()), 'Missing non-padding sequence fragments'
    return merged, sources


def diff(a,b):
    a,b=a.astype(np.float64),b.astype(np.float64)
    delta=b-a;absolute=float(np.linalg.norm(delta));norm=float(np.linalg.norm(a))
    return dict(max_abs=float(np.max(np.abs(delta))),absolute_l2=absolute,
                relative_l2=absolute/norm if norm else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    CASE=args.results
    data=json.loads((CASE/'inputs/input.json').read_text())
    stages,sources={},{}
    for stage in ('reference','repeat','candidate'):
        stages[stage],sources[stage]=assemble(CASE,stage,data['tokens'])
    rows=[]
    for stage in ('repeat','candidate'):
        assert stages[stage].keys()==stages['reference'].keys()
        for key,a in sorted(stages['reference'].items()):
            b=stages[stage][key]
            flips=np.flatnonzero(np.any(a['routes']!=b['routes'],axis=-1))
            changes=[]
            for token in flips:
                score_a=a['weight'].astype(np.float64)@a['input'][token].astype(np.float64)
                score_b=b['weight'].astype(np.float64)@b['input'][token].astype(np.float64)
                changes.append(dict(token=int(token), reference_experts=np.flatnonzero(a['routes'][token]).tolist(),
                    candidate_experts=np.flatnonzero(b['routes'][token]).tolist(),
                    reference_fp64_scores=score_a.tolist(),candidate_fp64_scores=score_b.tolist()))
            rows.append(dict(stage=stage,step=key[0],router=key[1],sample_index=key[2],valid_tokens=len(a['routes']),
                route_flip_tokens=len(flips),flips=changes,
                differences={field:diff(a[field],b[field]) for field in ('input','weight','probabilities')}))
    summary={}
    for stage in ('repeat','candidate'):
        summary[stage]=[]
        for step in range(1,4):
            for layer in range(2):
                selected=[r for r in rows if r['stage']==stage and r['step']==step
                          and r['router']==f'decoder.layers.{layer}.mlp.router']
                summary[stage].append(dict(step=step,layer=layer,valid_tokens=sum(r['valid_tokens'] for r in selected),
                    route_flip_tokens=sum(r['route_flip_tokens'] for r in selected),
                    maximum_absolute={f:max(r['differences'][f]['max_abs'] for r in selected)
                                      for f in ('input','weight','probabilities')}))
    result=dict(scope='read-only router diagnostic; independently aligned TP/SP token fragments and PP layers',
                sources=sources,report_sha256=sha(CASE/'acceptance.json'),summary=summary,rows=rows)
    output=args.output
    if output.exists():
        raise ValueError('Output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,route_flips=[r for r in rows if r['flips']]),indent=2))


if __name__=='__main__':
    main()
