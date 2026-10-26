"""Frozen native distributed trajectories, including same-topology process restart."""
import copy
from fnmatch import fnmatchcase
import math
from pathlib import Path

from .common import require, read, save, object_hash, now
from .protocol import validate_spec as validate_single, inventory, toolkit_files, input_identity
from .acceptance import group_for, validate_limit
from .precision import stage_precision

STAGES = ('reference', 'repeat', 'candidate', 'resume')
STATE_FIELDS = ('parameters', 'master_parameters', 'exp_avg', 'exp_avg_sq')
FIELDS = ('gradients', 'updates', *STATE_FIELDS)


STORAGE_ALIGNMENT = 'identical_fp32_masters_bf16_exact_initial_values'


def validate_storage_precision(contract):
    stages = contract.get('precision_by_stage')
    if stages is None:
        require('precision_comparison' not in contract, 'Precision comparison requires stage precision')
        values = [contract.get('precision', {})]
    else:
        require('precision' not in contract and isinstance(stages, dict) and
                set(stages) == {'reference', 'candidate'}, 'Declare only reference/candidate storage precision')
        require(contract.get('precision_comparison') == {
                    'allowed_changes': ['parameters'], 'initial_weights_alignment': STORAGE_ALIGNMENT},
                'Freeze parameter storage changes and BF16-exact initial master alignment')
        require(stages['reference'].get('parameters') == 'float32' and
                stages['candidate'].get('parameters') == 'bfloat16',
                'Native storage comparison requires FP32 reference and BF16 candidate')
        values = list(stages.values())
    for precision in values:
        require(precision == {'parameters': precision.get('parameters'), 'gradients': 'float32',
                             'optimizer_parameters': 'float32', 'optimizer_state': 'float32'} and
                precision['parameters'] in ('float32', 'bfloat16'),
                'Native trajectory requires FP32 gradients, master parameters and optimizer state')


def validate_spec(spec):
    require(spec.get('scope') == 'integration_test', 'Native trajectory currently certifies integration tests only')
    contract = spec['contract']
    count = contract.get('optimizer_steps')
    require(type(count) is int and count >= 2 and contract.get('optimizer_initial_state') == 'fresh', 'Trajectory starts fresh with at least two steps')
    require(contract['optimizer']['name'] == 'MegatronDistributedAdam', 'Native trajectory requires distributed Adam')
    validate_storage_precision(contract)
    require('parallel_adapter' in spec, 'Declare the native parallel adapter')
    trajectory = spec['trajectory']
    require(set(trajectory) == {'steps', 'resume_after_step', 'scheduler', 'state_acceptance'}, 'Invalid native trajectory fields')
    boundary = trajectory['resume_after_step']
    require(type(boundary) is int and 0 < boundary < count, 'Resume boundary must be inside the trajectory')
    require(len(trajectory['steps']) == count, 'Input schedule must cover every step')
    require(trajectory['scheduler']['name'] == 'MegatronOptimizerParamScheduler' and
            isinstance(trajectory['scheduler'].get('options'), dict), 'Declare the native scheduler')
    for step in trajectory['steps']:
        require(set(step) == {'input_artifact', 'objective', 'observables'} and step['input_artifact'] in spec['artifacts'] and
                step['input_artifact'] != 'code', 'Declare every frozen step input')
        require(step['observables'].keys() == spec['observables'].keys(), 'Observation names differ between steps')
        base = copy.deepcopy(spec)
        base['contract']['optimizer_steps'] = 1
        base['contract']['objective'] = step['objective']
        base['observables'] = step['observables']
        # This native adapter observes actual FP32 masters. Generic adapters
        # deliberately retain their stricter, single-process precision contract.
        base['contract']['precision'] = stage_precision(spec, 'reference')
        base['contract'].pop('precision_by_stage', None)
        base['contract'].pop('precision_comparison', None)
        validate_single(base)
    rules = trajectory['state_acceptance']
    require(set(rules) == {'parameters', 'groups'}, 'Declare parameter and group state gates')
    for category, names in (('parameters', set(spec['parameters'])),
                            ('groups', {group_for(k, v) for k, v in spec['parameters'].items()})):
        covered = set()
        require(isinstance(rules[category], list) and rules[category], 'Missing native state gates')
        for rule in rules[category]:
            require(set(rule) == {'pattern', *STATE_FIELDS} and isinstance(rule['pattern'], str), 'Invalid state gate')
            matched = {n for n in names if fnmatchcase(n, rule['pattern'])}
            require(matched, 'State gate matches no parameter/group')
            covered |= matched
            for field in STATE_FIELDS:
                validate_limit(rule[field])
        require(covered == names, 'Incomplete native state gate coverage')


def prepare(spec_path, output, timeout=1800, budget_gib=1):
    spec = read(spec_path)
    validate_spec(spec)
    output = Path(output).resolve()
    require(not output.exists(), 'Prepare a new immutable trajectory')
    require(type(timeout) is int and timeout > 0 and math.isfinite(budget_gib) and budget_gib > 0, 'Invalid budget')
    artifacts = {name: inventory(path) for name, path in spec['artifacts'].items()}
    require(all(not output.is_relative_to(Path(a['path'])) for a in artifacts.values()), 'Output is inside an input artifact')
    n = sum(math.prod(p['shape']) for p in spec['parameters'].values())
    estimated = n * 4 * len(FIELDS) * spec['contract']['optimizer_steps'] * len(STAGES)
    # Allow replicated model checkpoints plus native optimizer buffers and snapshots.
    world = max(c['world_size'] for c in spec['configurations'].values())
    peak = estimated + n * 4 * 8 * world + 32 * 2**20
    require(budget_gib * 2**30 > peak * 1.1, 'Insufficient native trajectory checkpoint/raw budget')
    configs = copy.deepcopy(spec['configurations'])
    configs['repeat'], configs['resume'] = copy.deepcopy(configs['reference']), copy.deepcopy(configs['candidate'])
    plan = {k: copy.deepcopy(spec[k]) for k in ('name', 'scope', 'parameters', 'command', 'contract', 'observables',
            'acceptance', 'policy', 'trajectory', 'parallel_adapter')}
    plan.update(schema_version=4, adapter='native-trajectory-v1', created_utc=now(), artifacts=artifacts,
                configurations=configs, trainable_elements=n, required_checks=spec.get('required_checks', []),
                environment=spec.get('environment', {}), toolkit_files=toolkit_files(), output=str(output),
                stage_timeout_seconds=timeout, disk_budget_bytes=int(budget_gib * 2**30),
                estimated_raw_bytes=estimated, estimated_peak_raw_bytes=peak, cleanup_on_pass=False)
    plan['plan_sha256'] = object_hash(plan)
    output.mkdir(parents=True)
    save(output / 'spec.json', spec)
    save(output / 'plan.json', plan)
    return {'status': 'prepared', 'plan': str(output / 'plan.json'), 'stages': list(STAGES), 'estimated_raw_bytes': estimated}


def identity(plan, stage, rank):
    return {'protocol': 'native-trajectory-v1', 'plan_sha256': plan['plan_sha256'], 'stage': stage, 'rank': rank,
            'configuration': plan['configurations'][stage], 'contract': plan['contract'], 'input_identity': input_identity(plan)}
