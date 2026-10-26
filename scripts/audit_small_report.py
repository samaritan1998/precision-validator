"""Independent NumPy oracle for small Slime reports (no validator imports).

Reassembles every canonical element and recalculates global/per-parameter
metrics directly. This checks reporting arithmetic and fragment coverage; it
cannot independently certify a framework's parameter-to-canonical mapping.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text())


def assemble(folder, world, max_elements):
    data, seen, catalog = {}, {}, {}
    for rank in range(world):
        row = read(folder / f'rank{rank}.json')
        for name, info in row['catalog'].items():
            assert catalog.setdefault(name, info) == info
        assert sum(math.prod(v['shape']) for v in catalog.values()) <= max_elements, 'Small oracle memory limit'
        for name, info in row['storage']['files'].items():
            raw = (folder / name).read_bytes()
            assert len(raw) == info['bytes'] and hashlib.sha256(raw).hexdigest() == info['sha256']
        for piece in row['storage']['pieces']:
            key = piece['field'], piece['name']
            n = math.prod(piece['shape'])
            data.setdefault(key, np.zeros(n, dtype=np.float64))
            seen.setdefault(key, np.zeros(n, dtype=np.bool_))
            start, end = piece['start'], piece['end']
            assert end-start == piece['elements'] and 0 <= start < end <= n
            assert not seen[key][start:end].any(), 'Overlapping ownership'
            values = np.fromfile(folder / piece['file'], dtype='<f4', count=end-start, offset=piece['offset_bytes'])
            assert values.size == end-start and np.isfinite(values).all()
            data[key][start:end] = values
            seen[key][start:end] = True
    expected = {(field, name) for field in ('gradients','updates') for name in catalog}
    assert set(data) == expected and all(v.all() for v in seen.values()), 'Missing canonical elements'
    return data


def metrics(reference, candidate):
    difference = candidate-reference
    absolute = float(np.linalg.norm(difference))
    ref = float(np.linalg.norm(reference))
    return dict(max_abs=float(np.max(np.abs(difference))), absolute_l2=absolute,
                rmse=absolute/math.sqrt(reference.size), relative_l2=absolute/ref if ref else (0. if absolute == 0 else math.inf))


def audit(root, max_elements):
    manifest = read(root / 'manifest.json')
    report = read(root / 'acceptance.json')
    checked, largest = 0, []
    for step in report['steps']:
        stages = {s:assemble(root / s / 'steps' / f'{step["step"]:06d}', manifest['world_size'], max_elements)
                  for s in ('reference','repeat','candidate')}
        for comparison in ('repeat','candidate'):
            assert set(stages[comparison]) == set(stages['reference'])
            for field in ('gradients','updates'):
                names = sorted(name for f,name in stages['reference'] if f == field)
                published = step['comparisons'][comparison][field]
                for name in [*names, None]:
                    ref = stages['reference'][field,name] if name else np.concatenate([stages['reference'][field,n] for n in names])
                    value = stages[comparison][field,name] if name else np.concatenate([stages[comparison][field,n] for n in names])
                    actual = metrics(ref,value)
                    expected = published['parameters'][name] if name else published
                    for metric, number in actual.items():
                        assert math.isclose(number, expected[metric], rel_tol=1e-10, abs_tol=1e-14), (step['step'],comparison,field,name,metric)
                        checked += 1
                    if comparison == 'candidate' and name:
                        largest.append(dict(step=step['step'],field=field,parameter=name,**actual))
    return dict(status='independent_metrics_verified', original_status=report['status'],
                checked_metrics=checked, largest_local_relative_errors=sorted(largest,key=lambda x:x['relative_l2'],reverse=True)[:10])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-elements',type=int,default=2000000)
    args=p.parse_args()
    assert not args.output.exists()
    result=audit(args.results,args.max_elements)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
