"""Validate complete per-step Slime evidence and reuse the existing metric gates."""
import math
from pathlib import Path
from types import SimpleNamespace

from .common import require, read, save, digest, object_hash, now
from .storage import index_pieces, verify_storage
from .parallel import validate_rank_groups
from .protocol_compare import compare_pair, gates_for
from .acceptance import validate_acceptance
from .slime_check import load_manifest, STAGES


def load_stage(manifest, stage):
    root = Path(manifest['output']) / stage
    complete = read(root / 'complete.json')
    require(complete['manifest_sha256'] == manifest['sha256'] and complete['stage'] == stage, 'Stage completion identity differs')
    require(digest(root / 'process.json') == complete['process_sha256'], 'Stage process receipt changed')
    process = read(root / 'process.json')
    require(process['exit_code'] == 0 and not process['timed_out'], 'Stage did not exit successfully')
    expected = {f'steps/{step:06d}/rank{rank}.json'
                for step in range(1, manifest['steps'] + 1) for rank in range(manifest['world_size'])}
    require(set(complete['receipts']) == expected, 'Stage receipt coverage differs')
    steps, initial, catalog = [], {}, None
    for step in range(1, manifest['steps'] + 1):
        folder = root / 'steps' / f'{step:06d}'
        rows, observations = [], {}
        pieces = {'gradients': [], 'updates': []}
        files = set()
        for rank in range(manifest['world_size']):
            path = folder / f'rank{rank}.json'
            require(digest(path) == complete['receipts'][str(path.relative_to(root))], 'Rank receipt changed')
            row = read(path)
            require(row['manifest_sha256'] == manifest['sha256'] and
                    (row['stage'], row['step'], row['rank']) == (stage, step, rank), 'Step identity differs')
            require(row['optimizer_updated'] is True and row['observer']['optimizer_step_calls'] == 1 and
                    row['observer']['prepare_calls'] >= 1, 'Incomplete native optimizer execution')
            require(all(v is None or v == step for v in row['observer']['optimizer_state_steps']), 'Adam step counters differ')
            require(row['precision']['gradients'] == row['precision']['optimizer_parameters'] ==
                    row['precision']['optimizer_state'] == 'float32', 'Native precision differs')
            if catalog is None:
                catalog = row['catalog']
            require(catalog == row['catalog'], 'Canonical parameter catalog changed')
            for name, value in row['initial'].items():
                require(step == 1 and name in catalog and value['shape'] == catalog[name]['shape'], 'Invalid initial parameter identity')
                require(initial.setdefault(name, value) == value, 'Initial replicas disagree')
            if row['storage']['files']:
                verify_storage(folder, row['storage'])
            else:
                require(not row['storage']['pieces'], 'Unowned rank has tensor pieces')
            require(not files.intersection(row['storage']['files']), 'Duplicate storage ownership')
            files.update(row['storage']['files'])
            for piece in row['storage']['pieces']:
                require(piece['field'] in pieces, 'Unexpected captured field')
                pieces[piece['field']].append(piece)
            for name, value in row['observations'].items():
                require(name == 'loss' and name not in observations and value['shape'] == [] and
                        len(value['values']) == 1 and math.isfinite(value['values'][0]), 'Invalid/duplicate loss observation')
                observations[name] = value
            rows.append(row)
        require(set(initial) == set(catalog) and set(observations) == {'loss'}, 'Initial parameter or loss coverage incomplete')
        validate_rank_groups({'world_size': manifest['world_size'], 'parallelism': manifest['configurations'][stage]}, rows)
        for key in ('objective', 'normalization', 'precision', 'scheduler_steps', 'runtime', 'device'):
            require(all(row[key] == rows[0][key] for row in rows), f'Native rank evidence differs: {key}')
        elements = sum(math.prod(v['shape']) for v in catalog.values())
        indices = {field: index_pieces(value, elements) for field, value in pieces.items()}
        require(all(set(value) == set(catalog) for value in indices.values()), 'Gradient/update parameter coverage differs')
        for index in indices.values():
            require(all(value[0]['shape'] == catalog[name]['shape'] for name, value in index.items()), 'Global tensor shape differs')
        plan = {'parameters': catalog}
        steps.append(SimpleNamespace(folder=folder, indices=indices, initial=initial, auxiliary={},
                    runtime=object_hash(rows[0]['runtime']), plan=plan, observations=observations, rows=rows))
    return steps


def compare(path):
    manifest = load_manifest(path)
    stages = {stage: load_stage(manifest, stage) for stage in STAGES}
    reports = []
    for number in range(manifest['steps']):
        ref = stages['reference'][number]
        for stage in ('repeat', 'candidate'):
            actual = stages[stage][number]
            require(ref.plan == actual.plan, 'Model parameter catalog changed across configurations')
            for key in ('objective', 'normalization', 'precision', 'scheduler_steps', 'device'):
                require(ref.rows[0][key] == actual.rows[0][key], f'Comparisons changed {key}, step {number + 1}')
            # Optimizer groups can be partitioned differently by topology;
            # compare the set of actual group options across all native ranks.
            options = lambda value: {object_hash(o) for r in value.rows for o in r['optimizer_options']}
            require(options(ref) == options(actual), 'Optimizer hyperparameters or LR changed across stages')
        acceptance, policy = manifest['rules']['acceptance'], manifest['rules']['policy']
        if acceptance is not None:
            validate_acceptance({'parameters': ref.plan['parameters'], 'policy': policy,
                                 'acceptance': acceptance, 'observables': {'loss': []}})
        comparisons = {stage: compare_pair(ref, stages[stage][number]) for stage in ('repeat', 'candidate')}
        gates = gates_for(comparisons, policy, acceptance)
        status = 'report_only' if policy is None else ('passed' if all(g['passed'] for g in gates.values()) else 'numerical_failure')
        reports.append({'step': number + 1, 'status': status, 'objective': ref.rows[0]['objective'],
                        'comparisons': comparisons, 'gates': gates})
    status = ('numerical_failure' if any(r['status'] == 'numerical_failure' for r in reports)
              else 'report_only' if manifest['rules']['policy'] is None else 'passed')
    report = {'schema': manifest['schema'], 'manifest_sha256': manifest['sha256'], 'created_utc': now(),
              'status': status, 'scope': manifest['scope'], 'steps': reports,
              'configurations': manifest['configurations'],
              'acceptance_coverage': {'strict': manifest['rules']['acceptance'] is not None,
                                      'fields': ['gradients', 'updates'], 'observations': ['loss'],
                                      'optimizer_state_vectors': False, 'logits': False},
              'limitations': ['First N steps from fresh native Adam; no checkpoint resume',
                              'Synchronous text SFT only; no RL, multimodal, virtual pipeline or stateful model buffers',
                              'Adam state vectors are checked for dtype/finiteness/counters, not compared numerically',
                              'Results certify only the actual declared model/data/topologies/hardware']}
    output = Path(manifest['output'])
    save(output / 'acceptance.json', report)
    lines = ['# Slime 训练精度对照', '', f'状态：**{status}**',
             f'范围：从同一初始权重开始，各独立执行前 {manifest["steps"]} 个真实 optimizer step。', '',
             '比较完整梯度、FP32 主参数的实际更新及 loss；不包含完整 logits、Adam 状态向量或恢复验收。', '',
             '| 步 | 对照 | 观测项 | 最大绝对误差 | 绝对 L2 | RMSE | 相对 L2 |',
             '|---|---|---|---:|---:|---:|---:|']
    for result in reports:
        for stage, fields in result['comparisons'].items():
            for field, metric in fields.items():
                cells = ['未定义' if metric[k] is None else f'{metric[k]:.8g}'
                         for k in ('max_abs', 'absolute_l2', 'rmse', 'relative_l2')]
                lines.append(f'| {result["step"]} | {stage} | {field} | ' + ' | '.join(cells) + ' |')
        for name, gate in result['gates'].items():
            if not gate['passed']:
                lines += ['', f'第 {result["step"]} 步，{name} 未通过：' + '; '.join(gate['reasons'])]
    lines += ['', '未指定运行前冻结的 policy 时，report_only 仅表示报告已生成，不能解释为验收通过。',
              '逐参数、分组误差及每步输入贡献校验见 acceptance.json。', '']
    (output / 'report.md').write_text('\n'.join(lines))
    return report
