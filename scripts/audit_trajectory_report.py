"""Recalculate small native-trajectory reports using NumPy, without SDK imports.

Checks complete fragment coverage, raw-file hashes, rank/step completion seals,
every gradient/update/parameter/Adam-state metric, and forward observations.
This audits recorded evidence, not the framework's canonical parameter mapping.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FIELDS = ('gradients', 'updates', 'parameters', 'master_parameters', 'exp_avg', 'exp_avg_sq')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(folder, world, parameters, fields, max_elements=2_000_000):
    assert sum(math.prod(p['shape']) for p in parameters.values()) <= max_elements
    data, seen, observations, rows = {}, {}, {}, []
    files = set()
    for rank in range(world):
        row = read(folder / f'rank{rank}.json')
        assert row['rank'] == rank
        rows.append(row)
        for name, info in row['storage']['files'].items():
            assert name not in files, 'Duplicate raw-file ownership'
            files.add(name)
            path = folder / name
            assert path.stat().st_size == info['bytes'] and digest(path) == info['sha256']
        for piece in row['storage']['pieces']:
            field, name = piece['field'], piece['name']
            assert field in fields and name in parameters
            assert piece['shape'] == parameters[name]['shape']
            assert piece['file'] in row['storage']['files']
            count = math.prod(piece['shape'])
            key = field, name
            data.setdefault(key, np.zeros(count, dtype=np.float64))
            seen.setdefault(key, np.zeros(count, dtype=bool))
            start, end = piece['start'], piece['end']
            assert 0 <= start < end <= count and end - start == piece['elements']
            assert not seen[key][start:end].any(), 'Overlapping tensor fragments'
            values = np.fromfile(folder / piece['file'], dtype='<f4', count=end-start,
                                 offset=piece['offset_bytes'])
            assert values.size == end-start and np.isfinite(values).all()
            data[key][start:end] = values
            seen[key][start:end] = True
        for name, value in row['observations'].items():
            assert name not in observations, 'Duplicate forward observation'
            array = np.asarray(value['values'], dtype=np.float64)
            assert array.size == math.prod(value['shape']) and np.isfinite(array).all()
            observations[name] = array
    assert set(data) == {(field, name) for field in fields for name in parameters}
    assert all(mask.all() for mask in seen.values()), 'Incomplete tensor coverage'
    return data, observations, rows


def metrics(reference, candidate):
    delta = candidate - reference
    absolute = float(np.linalg.norm(delta))
    reference_norm = float(np.linalg.norm(reference))
    return dict(max_abs=float(np.max(np.abs(delta))), absolute_l2=absolute,
                rmse=absolute / math.sqrt(reference.size),
                relative_l2=absolute / reference_norm if reference_norm else (0. if absolute == 0 else None))


def check_metrics(actual, published):
    for name, value in actual.items():
        other = published[name]
        if value is None:
            assert other is None
        else:
            assert other is not None and math.isclose(value, other, rel_tol=1e-10, abs_tol=1e-14), (name, value, other)
    return len(actual)


def compare(ref, candidate, published, fields):
    left, lo, _ = ref
    right, ro, _ = candidate
    assert set(left) == set(right) and set(lo) == set(ro)
    checked = 0
    for field in fields:
        names = sorted(name for f, name in left if f == field)
        assert set(published[field]['parameters']) == set(names)
        for name in [*names, None]:
            a = left[field, name] if name is not None else np.concatenate([left[field, n] for n in names])
            b = right[field, name] if name is not None else np.concatenate([right[field, n] for n in names])
            expected = published[field]['parameters'][name] if name is not None else published[field]
            checked += check_metrics(metrics(a, b), expected)
    for name in lo:
        checked += check_metrics(metrics(lo[name], ro[name]), published['observation/' + name])
    return checked


def verify_stage(root, plan, stage):
    folder = root / stage
    complete = read(folder / 'complete.json')
    process = read(folder / 'process.json')
    assert complete['status'] == 'stage_complete' and complete['stage'] == stage
    assert complete['plan_sha256'] == plan['plan_sha256']
    assert complete['process_sha256'] == digest(folder / 'process.json')
    assert process['returncode'] == 0
    world = plan['configurations'][stage]['world_size']
    assert set(complete['ranks']) == {f'rank{r}.json' for r in range(world)}
    n_steps = len(plan['trajectory']['steps'])
    for rank in range(world):
        filename = f'rank{rank}.json'
        assert complete['ranks'][filename] == digest(folder / filename)
        row = read(folder / filename)
        assert (row['stage'], row['rank'], row['plan_sha256']) == (stage, rank, plan['plan_sha256'])
        expected = {f'steps/{s:06d}/rank{rank}.json' for s in range(1, n_steps + 1)}
        assert set(row['step_receipts']) == expected
        for path, sha in row['step_receipts'].items():
            assert digest(folder / path) == sha


def audit(root, max_elements=2_000_000):
    plan, report = read(root / 'plan.json'), read(root / 'acceptance.json')
    assert report['plan_sha256'] == plan['plan_sha256']
    assert set(plan['configurations']) == {'reference', 'repeat', 'candidate', 'resume'}
    count = len(plan['trajectory']['steps'])
    assert set(report['comparisons']) == {f'step/{n}' for n in range(1, count + 1)}
    for stage in plan['configurations']:
        verify_stage(root, plan, stage)
    checked, counters, rank_steps = 0, 0, 0
    for step in range(1, count + 1):
        stages = {}
        for stage, configuration in plan['configurations'].items():
            folder = root / stage / 'steps' / f'{step:06d}'
            stages[stage] = assemble(folder, configuration['world_size'], plan['parameters'], FIELDS, max_elements)
            for row in stages[stage][2]:
                assert (row['stage'], row['step'], row['plan_sha256']) == (stage, step, plan['plan_sha256'])
                observer = row['observer']
                assert observer['optimizer_step_calls'] == 1 and observer['prepare_calls'] >= 1
                values = observer['optimizer_state_steps']
                assert not row['storage']['pieces'] or values
                assert all(type(value) in (float, int) and value == step for value in values)
                counters += len(values)
                rank_steps += 1
        for stage in ('repeat', 'candidate', 'resume'):
            reference = 'candidate' if stage == 'resume' else 'reference'
            checked += compare(stages[reference], stages[stage], report['comparisons'][f'step/{step}'][stage], FIELDS)
        assert [r['state'] for r in stages['candidate'][2]] == [r['state'] for r in stages['resume'][2]], 'Resume state differs'
    return dict(status='independent_metrics_verified', original_status=report['status'],
                report_sha256=digest(root / 'acceptance.json'), checked_metrics=checked,
                checked_adam_counters=counters, rank_steps=rank_steps, exact_resume_state=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-elements', type=int, default=2_000_000)
    args = parser.parse_args()
    result = audit(args.results, args.max_elements)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))
