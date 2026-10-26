"""Compare read-only native router traces without importing the validator."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np



def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diff(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    d = b - a
    norm = float(np.linalg.norm(a.reshape(-1)))
    absolute = float(np.linalg.norm(d.reshape(-1)))
    return dict(max_abs=float(np.max(np.abs(d))), absolute_l2=absolute,
                relative_l2=absolute / norm if norm else (0 if not absolute else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='Completed ep_bf16 case directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    CASE = args.results
    data = json.loads((CASE / 'inputs/input.json').read_text())
    rows, sources = [], []
    for rank in range(2):
        paths = {s: CASE / s / f'router-trace.rank{rank}.npz'
                 for s in ('reference', 'repeat', 'candidate')}
        arrays = {s: np.load(p, allow_pickle=False) for s, p in paths.items()}
        indices = {s: json.loads(p.with_suffix('.json').read_text()) for s, p in paths.items()}
        assert all(x['rows'] == indices['reference']['rows'] for x in indices.values())
        sources.extend(dict(stage=s, rank=rank, npz_sha256=sha(p), index_sha256=sha(p.with_suffix('.json')))
                       for s, p in paths.items())
        for row in indices['reference']['rows']:
            length = len(data['tokens'][row['sample_index']])
            prefix = row['prefix']
            for stage in ('repeat', 'candidate'):
                values = {}
                for field in ('input', 'weight', 'probabilities'):
                    a, b = [arrays[s][prefix + '/' + field] for s in ('reference', stage)]
                    if field != 'weight':
                        a, b = a[:length], b[:length]
                    values[field] = diff(a, b)
                a, b = [arrays[s][prefix + '/routes'][:length] for s in ('reference', stage)]
                tokens = np.flatnonzero(np.any(a != b, axis=-1))
                flips = []
                for token in tokens:
                    flips.append(dict(token=int(token), supervised=(token < len(data['loss_masks'][row['sample_index']])
                        and bool(data['loss_masks'][row['sample_index']][token])),
                        reference_experts=np.flatnonzero(a[token]).tolist(),
                        candidate_experts=np.flatnonzero(b[token]).tolist(),
                        reference_probabilities=arrays['reference'][prefix+'/probabilities'][token].tolist(),
                        candidate_probabilities=arrays[stage][prefix+'/probabilities'][token].tolist()))
                rows.append(dict(stage=stage, rank=rank, router=row['router'], call=row['call'],
                    step=row['call']//4+1, sample_index=row['sample_index'], valid_tokens=length,
                    route_flip_tokens=len(tokens), flips=flips, differences=values))
        for x in arrays.values():
            x.close()
    summary = {}
    for stage in ('repeat', 'candidate'):
        summary[stage] = []
        for step in range(1, 4):
            for layer in range(2):
                selected = [x for x in rows if x['stage'] == stage and x['step'] == step
                            and x['router'] == f'decoder.layers.{layer}.mlp.router']
                summary[stage].append(dict(step=step, layer=layer, calls=len(selected),
                    valid_tokens=sum(x['valid_tokens'] for x in selected),
                    route_flip_tokens=sum(x['route_flip_tokens'] for x in selected),
                    maximum_absolute={f:max(x['differences'][f]['max_abs'] for x in selected)
                                      for f in ('input', 'weight', 'probabilities')}))
    result = dict(scope='native_forward_router_outputs_only', report_sha256=sha(CASE/'acceptance.json'),
        sources=sources, summary=summary, route_flips=[r for r in rows if r['route_flip_tokens']], rows=rows)
    path = args.output
    if path.exists():
        raise ValueError('Output already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(summary=summary, route_flips=result['route_flips']), indent=2))


if __name__ == '__main__':
    main()
