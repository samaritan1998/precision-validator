"""Complete distributed state coverage and strict per-step/resume acceptance."""
from fnmatch import fnmatchcase
import math
from pathlib import Path

from .common import read, save, require, digest, load_plan, object_hash, now
from .native_trajectory import STAGES, FIELDS, STATE_FIELDS, identity
from .trajectory_sdk import cursor_for
from .trajectory_compare import exact_gate
from .storage import verify_storage, index_pieces, aligned_chunks
from .metrics import Metrics
from .acceptance import group_for, evaluate_limit
from .parallel import validate_rank_groups
from .protocol_compare import gates_for
from .precision import stage_precision


class Stage:
    def __init__(self, plan, name):
        self.folder, self.plan, self.name = Path(plan['output']) / name, plan, name
        final = read(self.folder / 'complete.json')
        require(final['status'] == 'stage_complete' and final['stage'] == name and final['plan_sha256'] == plan['plan_sha256'], 'Native trajectory completion differs')
        require(digest(self.folder / 'process.json') == final['process_sha256'] and read(self.folder / 'process.json')['returncode'] == 0, 'Native trajectory process failed')
        world, count = plan['configurations'][name]['world_size'], plan['contract']['optimizer_steps']
        require(set(final['ranks']) == {f'rank{i}.json' for i in range(world)}, 'Incomplete native trajectory ranks')
        workers = read(self.folder / 'workers.json')
        phases = ['prefix', 'suffix'] if name == 'resume' else ['continuous']
        require([p['phase'] for p in workers] == phases and all(p['returncode'] == 0 for p in workers), 'Incomplete fresh-process workers')
        self.ranks, self.steps, self.initial, self.raw_files = [], [], {}, {}
        runtime = set()
        precision = stage_precision(plan, name)
        for rank in range(world):
            path = self.folder / f'rank{rank}.json'
            require(digest(path) == final['ranks'][path.name], 'Native trajectory rank checksum differs')
            row = read(path)
            require(all(row.get(k) == v for k, v in identity(plan, name, rank).items()), 'Native trajectory identity differs')
            require(row['precision'] == precision and row['environment'] == plan['environment'], 'Native trajectory precision/environment differs')
            require(row['device']['kind'] == plan['configurations'][name]['device'], 'Native trajectory device differs')
            expected_device = plan['configurations'][name].get('device_name')
            require(not expected_device or expected_device in row['device']['name'], 'Unexpected native device')
            require(row['execution'] == {'optimizer_steps': count, 'teardown_completed': True}, 'Native trajectory execution incomplete')
            require(digest(self.folder / 'workers.json') == row['workers_sha256'], 'Worker receipt changed')
            require(set(row['step_receipts']) == {f'steps/{i:06d}/rank{rank}.json' for i in range(1, count + 1)}, 'Incomplete step rank receipts')
            for phase, worker in zip(phases, workers):
                exit_path = self.folder / f'exit.{phase}.rank{rank}.json'
                require(digest(exit_path) == worker['ranks'][exit_path.name], 'Worker exit checksum differs')
                exit_row = read(exit_path)
                require(exit_row['teardown_completed'] is True and exit_row['plan_sha256'] == plan['plan_sha256'] and
                        exit_row['stage'] == name and exit_row['rank'] == rank and exit_row['phase'] == phase,
                        'Native worker teardown incomplete')
            runtime.add(object_hash(row['runtime']))
            for key, value in row['initial_parameters'].items():
                require(key in plan['parameters'] and value['shape'] == plan['parameters'][key]['shape'] and
                        value['dtype'] == 'torch.' + precision['parameters'] and len(value['sha256']) == 64,
                        'Initial canonical parameter differs')
                if 'precision_by_stage' in plan['contract']:
                    aligned = value.get('storage_alignment', {})
                    require(set(aligned) == {'torch.float32', 'torch.bfloat16'} and
                            all(isinstance(v, str) and len(v) == 64 for v in aligned.values()) and
                            aligned[value['dtype']] == value['sha256'], 'Initial storage alignment evidence differs')
                require(self.initial.setdefault(key, value) == value, 'Initial replicas disagree')
            self.ranks.append(row)
        require(len(runtime) == 1 and self.initial.keys() == plan['parameters'].keys(), 'Runtime/initial coverage differs')
        self.runtime = runtime.pop()
        validate_rank_groups(plan['configurations'][name], self.ranks)
        for number, frozen in enumerate(plan['trajectory']['steps'], 1):
            part = self.folder / 'steps' / f'{number:06d}'
            pieces, observations, rows = {field: [] for field in FIELDS}, {}, []
            for rank, final_row in enumerate(self.ranks):
                path = part / f'rank{rank}.json'
                require(digest(path) == final_row['step_receipts'][str(path.relative_to(self.folder))], 'Native step checksum differs')
                row = read(path)
                require(all(row.get(k) == v for k, v in identity(plan, name, rank).items()) and row['step'] == number, 'Native step identity differs')
                require(row['input_sha256'] == plan['artifacts'][frozen['input_artifact']]['sha256'] and row['objective'] == frozen['objective'], 'Step input/objective differs')
                require(all(row['checks'].get(k) is True for k in plan['required_checks']), 'Missing native step checks')
                require(row['observer']['optimizer_step_calls'] == 1 and row['observer']['prepare_calls'] >= 1 and
                        all(n == number for n in row['observer']['optimizer_state_steps']), 'Native Adam step was not observed')
                state = row['state']
                require(state['data_cursor'] == cursor_for(plan, number), 'Native data cursor differs')
                require(all(r['step'] == number and {'exp_avg', 'exp_avg_sq', 'master'} <= r.keys()
                            for r in state['optimizer_shards'].values()), 'Incomplete native optimizer state')
                expected_samples = sum(s['objective']['global_samples'] for s in plan['trajectory']['steps'][:number])
                require(state['scheduler']['num_steps'] == expected_samples, 'Native scheduler sample count differs')
                previous = final_row['initial_state'] if number == 1 else self.steps[-1]['rows'][rank]['state']
                require(row['lr_used'] == [g['lr'] for g in previous['optimizer_groups']], 'Learning rate progression differs')
                require(len(row['observer']['optimizer_state_steps']) == len(state['optimizer_shards']), 'Native optimizer state ownership count differs')
                storage = row['storage']
                if storage['files']:
                    verify_storage(part, storage)
                else:
                    require(not storage['pieces'], 'Unowned rank has unexpected pieces')
                for piece in storage['pieces']:
                    require(piece['field'] in FIELDS, 'Unexpected native state field')
                    pieces[piece['field']].append(piece)
                for filename, evidence in storage['files'].items():
                    key = f'steps/{number:06d}/{filename}'
                    require(key not in self.raw_files, 'Duplicate raw owner')
                    self.raw_files[key] = evidence
                for key, obs in row['observations'].items():
                    require(key in frozen['observables'] and key not in observations and obs['shape'] == frozen['observables'][key] and
                            len(obs['values']) == math.prod(obs['shape']) and all(math.isfinite(v) for v in obs['values']), 'Observation coverage differs')
                    observations[key] = obs
                rows.append(row)
            indices = {k: index_pieces(v, plan['trainable_elements']) for k, v in pieces.items()}
            require(observations.keys() == frozen['observables'].keys(), 'Missing native forward observations')
            for index in indices.values():
                require(index.keys() == plan['parameters'].keys() and all(v[0]['shape'] == plan['parameters'][k]['shape'] for k, v in index.items()), 'Native state coverage differs')
            self.steps.append({'folder': part, 'rows': rows, 'indices': indices, 'observations': observations})
        for rank, row in enumerate(self.ranks):
            require(row['initial_state']['data_cursor'] == cursor_for(plan, 0) and
                    all(r['step'] == 0 for r in row['initial_state']['optimizer_shards'].values()), 'Native trajectory did not start fresh')
            if name != 'resume':
                require(row['resume'] is None, 'Unexpected resume evidence')
                continue
            record_path = self.folder / f'checkpoint.rank{rank}.json'
            record, evidence = read(record_path), row['resume']
            checkpoint = self.folder / f'checkpoint.rank{rank}.pt'
            require(all(record.get(k) == v for k, v in identity(plan, name, rank).items()), 'Checkpoint identity differs')
            require(digest(record_path) == evidence['checkpoint_record_sha256'] and digest(checkpoint) == record['sha256'] == evidence['checkpoint_sha256'] and
                    checkpoint.stat().st_size == record['bytes'], 'Checkpoint checksum differs')
            boundary = plan['trajectory']['resume_after_step']
            require(record['cursor'] == cursor_for(plan, boundary) and record['state'] == evidence['restored_state'] ==
                    self.steps[boundary - 1]['rows'][rank]['state'], 'Restored native state/cursor differs')
            require(record['producer_pid'] == evidence['producer_pid'] != evidence['consumer_pid'] and
                    record['producer_process_id'] == evidence['producer_process_id'] != evidence['consumer_process_id'], 'Resume worker was reused')
            for phase, key in (('prefix', 'producer_process_id'), ('suffix', 'consumer_process_id')):
                require(read(self.folder / f'exit.{phase}.rank{rank}.json')['process_id'] == evidence[key], 'Resume process provenance differs')


def compare_step(plan, left, right):
    result = {}
    for field in FIELDS:
        total, params, groups = Metrics(), {}, {}
        for name, value, baseline in aligned_chunks(left['folder'], left['indices'][field], right['folder'], right['indices'][field]):
            metric = Metrics().add(value, baseline)
            total.merge(metric)
            params.setdefault(name, Metrics()).merge(metric)
        value = total.result()
        if field in ('gradients', 'updates'):
            require(value['reference2'] > 0 and value['actual2'] > 0, 'All-zero native training field')
        for name, metric in params.items():
            groups.setdefault(group_for(name, plan['parameters'][name]), Metrics()).merge(metric)
        result[field] = {**value, 'parameters': {k: v.result() for k, v in params.items()}, 'groups': {k: v.result() for k, v in groups.items()}}
    for key, obs in left['observations'].items():
        result['observation/' + key] = Metrics().add(right['observations'][key]['values'], obs['values']).result()
    return result


def compare_suite(plan_path, retain_raw=True):
    plan = load_plan(plan_path)
    require(Path(plan_path).resolve().parent == Path(plan['output']), 'Plan moved')
    stages = {n: Stage(plan, n) for n in STAGES}
    reference = stages['reference']
    require(all(s.runtime == reference.runtime for s in stages.values()), 'Native runtime differs')
    if 'precision_by_stage' in plan['contract']:
        canonical = lambda stage: {k: {'shape': v['shape'], 'alignment': v['storage_alignment']}
                                    for k, v in stage.initial.items()}
        require(all(canonical(s) == canonical(reference) for s in stages.values()),
                'Initial FP32 master values or BF16-exact storage alignment differ')
        require(reference.initial == stages['repeat'].initial and
                stages['candidate'].initial == stages['resume'].initial, 'Same-precision initial weights differ')
    else:
        require(all(s.initial == reference.initial for s in stages.values()), 'Initial weights differ')
    require([r['initial_state'] for r in reference.ranks] == [r['initial_state'] for r in stages['repeat'].ranks], 'Reference repeat initial state differs')
    require([r['initial_state'] for r in stages['candidate'].ranks] == [r['initial_state'] for r in stages['resume'].ranks], 'Resume initial state differs')
    comparisons, gates = {}, {}
    for i in range(plan['contract']['optimizer_steps']):
        current = {name: compare_step(plan, reference.steps[i], stages[name].steps[i]) for name in ('repeat', 'candidate')}
        current['resume'] = compare_step(plan, stages['candidate'].steps[i], stages['resume'].steps[i])
        prefix = f'step/{i + 1}'
        comparisons[prefix] = current
        numerical = gates_for(current, plan['policy'], plan['acceptance'])
        for category, rules in plan['trajectory']['state_acceptance'].items():
            for ri, rule in enumerate(rules):
                for field in STATE_FIELDS:
                    for name, value in current['candidate'][field][category].items():
                        if fnmatchcase(name, rule['pattern']):
                            numerical[f'{category}/{field}/{name}/rule{ri}'] = evaluate_limit(current['repeat'][field][category][name], value, rule[field])
        # Same-topology resume requires bitwise-equivalent tensors and all rank-local states.
        for field, value in current['resume'].items():
            numerical['resume/' + field] = exact_gate(value['max_abs'] == 0, 'Resumed values differ')
        numerical['resume/state'] = exact_gate([r['state'] for r in stages['candidate'].steps[i]['rows']] ==
                                              [r['state'] for r in stages['resume'].steps[i]['rows']], 'Resumed checkpointable state/RNG differs')
        for name in ('repeat', 'candidate', 'resume'):
            baseline = stages['candidate'] if name == 'resume' else reference
            left_lrs = {lr for r in baseline.steps[i]['rows'] for lr in r['lr_used']}
            right_lrs = {lr for r in stages[name].steps[i]['rows'] for lr in r['lr_used']}
            numerical[f'{name}/lr_used'] = exact_gate(left_lrs == right_lrs, 'Native learning rates differ between configurations')
            for key in ('scheduler', 'data_cursor'):
                left = {object_hash(r['state'][key]) for r in baseline.steps[i]['rows']}
                right = {object_hash(r['state'][key]) for r in stages[name].steps[i]['rows']}
                numerical[f'{name}/{key}'] = exact_gate(left == right and len(left) == 1, f'{key} differs')
        gates.update({f'{prefix}/{k}': v for k, v in numerical.items()})
    report = {'schema_version': 4, 'adapter': plan['adapter'], 'name': plan['name'], 'plan_sha256': plan['plan_sha256'],
              'status': 'passed' if all(g['passed'] for g in gates.values()) else 'numerical_failure', 'finished_utc': now(),
              'scope': 'native_distributed_small_model_multistep_same_topology_resume', 'comparisons': comparisons, 'gates': gates,
              'coverage': {'steps_per_stage': plan['contract']['optimizer_steps'], 'fields': list(FIELDS), 'stages': list(STAGES),
                           'parameters': len(plan['parameters']), 'all_parameters_and_observations_gated': True,
                           'resume_exact': True, 'resume_fresh_process': True,
                           'distinct_inputs': len({plan['artifacts'][s['input_artifact']]['sha256'] for s in plan['trajectory']['steps']})},
              'configurations': plan['configurations'], 'contract': plan['contract'], 'raw_retained': True,
              'stage_completions': {n: digest(s.folder / 'complete.json') for n, s in stages.items()},
              'raw_artifacts': {n: s.raw_files for n, s in stages.items()}}
    save(Path(plan['output']) / 'acceptance.json', report)
    lines = [f"# 原生分布式多步验证：{plan['name']}", '', f"状态：{report['status']}。范围：小模型集成；不代表生产模型验收。", '',
             '| 步骤 | 对照 | 指标 | 最大绝对误差 | 绝对 L2 | RMSE | 相对 L2 |', '|---|---|---|---:|---:|---:|---:|']
    for step, pairs in comparisons.items():
        for pair, fields in pairs.items():
            for field, value in fields.items():
                lines.append(f'| {step} | {pair} | {field} | ' + ' | '.join('未定义' if value[k] is None else f'{value[k]:.8g}'
                              for k in ('max_abs', 'absolute_l2', 'rmse', 'relative_l2')) + ' |')
    (Path(plan['output']) / 'report.md').write_text('\n'.join(lines) + '\n')
    return report


def cleanup_passed(*args):
    require(False, 'Native trajectories retain checkpoints and raw evidence')


def compare_regression(*args):
    require(False, 'Native trajectory cross-version regression is not yet implemented')
