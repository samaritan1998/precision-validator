"""Opt-in Slime training entry: bounded real training, then offline comparison.

No Ray/GPU imports here. The source bridge imports this module only when the
user opts in. All three children use the ordinary Slime train(args) function.
"""
import argparse
import copy
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import require, read, save, digest, object_hash, now, verify_files
from .parallel import validate_layout

STAGES = ('reference', 'repeat', 'candidate')
AXES = {'tp': 'tensor_model_parallel_size', 'pp': 'pipeline_model_parallel_size',
        'cp': 'context_parallel_size', 'ep': 'expert_model_parallel_size',
        'etp': 'expert_tensor_parallel_size'}


def add_arguments(parser):
    group = parser.add_argument_group('Slime precision check')
    group.add_argument('--precision-check-steps', type=int, default=0,
                       help='Train this many actual optimizer steps per comparison stage, then exit.')
    group.add_argument('--precision-check-reference', default='',
                       help='Reference overrides, e.g. tp=1,cp=1. Omitted axes keep training values.')
    group.add_argument('--precision-check-dir', default=None,
                       help='New shared output directory. Never reuse a previous run directory.')
    group.add_argument('--precision-check-policy', default=None,
                       help='JSON with policy and acceptance. Without it reports are report_only.')
    group.add_argument('--precision-check-disk-gib', type=float, default=1024)
    group.add_argument('--precision-check-timeout', type=int, default=14400)
    group.add_argument('--precision-check-stage', choices=STAGES, default=None,
                       help=argparse.SUPPRESS)
    group.add_argument('--precision-check-manifest', default=None, help=argparse.SUPPRESS)
    return parser


def reference_layout(value, candidate, world):
    result, seen = dict(candidate), set()
    for item in value.split(',') if value else []:
        pair = item.strip().split('=')
        require(len(pair) == 2 and pair[0] in AXES and pair[0] not in seen,
                'Reference must contain unique tp/pp/cp/ep/etp=N overrides')
        key, number = pair
        require(number.isdecimal() and int(number) > 0, 'Parallel degrees must be positive integers')
        seen.add(key)
        result[key] = int(number)
    dense = result['tp'] * result['pp'] * result['cp']
    require(world % dense == 0, 'Reference TP*PP*CP must divide allocated training GPUs')
    result['dp'] = world // dense
    validate_layout(result, world)
    return result


def layout_from_args(args):
    world = args.actor_num_nodes * args.actor_num_gpus_per_node
    layout = {key: getattr(args, field, None) or 1 for key, field in AXES.items()}
    # Megatron defaults ETP to TP when the ETP argument is unset.
    if getattr(args, 'expert_tensor_parallel_size', None) is None:
        layout['etp'] = layout['tp']
    layout['dp'] = world // (layout['tp'] * layout['pp'] * layout['cp'])
    validate_layout(layout, world)
    return layout, world


def preflight(args):
    require(type(args.precision_check_steps) is int and args.precision_check_steps > 0,
            'precision-check-steps must be a positive optimizer-step count')
    require(args.loss_type == 'sft_loss', 'First Slime bridge supports fixed-data SFT only; RL needs frozen rollout replay')
    for field in ('use_critic', 'enable_mtp_training', 'reset_optimizer_states',
                  'use_rollout_routing_replay', 'use_routing_replay', 'megatron_config_path',
                  'virtual_pipeline_model_parallel_size', 'custom_megatron_before_train_step_hook_path',
                  'release_train'):
        require(not getattr(args, field, None), f'Slime precision check does not yet support {field}')
    require(getattr(args, 'hf_checkpoint', None), 'Freeze an existing --hf-checkpoint as the common initial model')
    require(not getattr(args, 'load', None) or Path(args.load).resolve() == Path(args.hf_checkpoint).resolve(),
            'First bridge starts from HF weights and fresh optimizer; checkpoint resume is not supported')
    require(getattr(args, 'use_distributed_optimizer', False), 'Native distributed optimizer is required')
    require(getattr(args, 'optimizer', 'adam') == 'adam', 'First bridge supports native Adam only')
    require(not getattr(args, 'fp16', False), 'Use FP32 or BF16 model parameters')
    require(math.isfinite(args.precision_check_disk_gib) and args.precision_check_disk_gib > 0,
            'Disk budget must be finite and positive')
    require(args.precision_check_timeout > 0, 'Timeout must be positive')
    require((getattr(args, 'start_rollout_id', None) or 0) == 0,
            'This bridge validates the first N optimizer steps, not arbitrary mid-training steps')


def code_inventory(root):
    root = Path(root).resolve()
    names = [p for p in root.rglob('*.py') if '.git' not in p.parts and '__pycache__' not in p.parts]
    return {'root': str(root), 'files': {str(p.relative_to(root)): digest(p) for p in sorted(names)}}


def prepare(args, entry, argv):
    preflight(args)
    candidate, world = layout_from_args(args)
    reference = reference_layout(args.precision_check_reference, candidate, world)
    output = Path(args.precision_check_dir or ('precision-check-' + time.strftime('%Y%m%d-%H%M%S'))).resolve()
    require(not output.exists(), 'Precision output already exists; choose a new directory')
    require(Path(entry).name == 'train.py', 'Initial bridge supports synchronous train.py; train_async.py is not enabled')
    rules = read(args.precision_check_policy) if args.precision_check_policy else {'policy': None, 'acceptance': None}
    require(set(rules) == {'policy', 'acceptance'}, 'Policy file must contain policy and acceptance')
    require((rules['policy'] is None) == (rules['acceptance'] is None),
            'Use complete global/local/forward gates, or explicitly report_only')
    from .protocol import validate_policy
    validate_policy(rules['policy'])
    root = Path(entry).resolve().parent
    require(not output.is_relative_to(root), 'Keep mutable results outside the frozen Slime source')
    frozen = {'schema': 'slime-training-check-v1', 'created_utc': now(), 'output': str(output),
              'steps': args.precision_check_steps, 'world_size': world,
              'configurations': {'reference': reference, 'repeat': reference, 'candidate': candidate},
              'source': code_inventory(root), 'toolkit': code_inventory(Path(__file__).parent),
              'original_argv_sha256': object_hash(argv), 'rules': rules,
              'disk_budget_bytes': int(args.precision_check_disk_gib * 2**30),
              'stage_timeout_seconds': args.precision_check_timeout,
              'scope': 'first_N_fresh_SFT_optimizer_steps',
              'observables': {'loss': []}, 'automatic_model_fixes': False}
    frozen['sha256'] = object_hash(frozen)
    output.mkdir(parents=True)
    save(output / 'manifest.json', frozen)
    return frozen


def load_manifest(path):
    value = read(path)
    expected = value.pop('sha256')
    require(object_hash(value) == expected, 'Slime check manifest was modified')
    value['sha256'] = expected
    for key in ('source', 'toolkit'):
        verify_files(value[key]['root'], value[key]['files'])
    return value


def configure_child(args, manifest):
    require(args.precision_check_steps == manifest['steps'], 'Child step count differs')
    layout = manifest['configurations'][args.precision_check_stage]
    for key, field in AXES.items():
        setattr(args, field, layout[key])
    require(layout_from_args(args) == (layout, manifest['world_size']), 'Child topology differs')
    # Keep the original GBS, loss, optimizer, LR schedule and sample schedule.
    # Disable production checkpoint/evaluation outputs for the bounded run.
    args.save, args.save_interval, args.eval_interval = None, None, None
    args.precision_check_dir = manifest['output']
    args.precision_check_manifest = str(Path(manifest['output']) / 'manifest.json')


def run_child(command, folder, timeout):
    folder.mkdir(parents=True, exist_ok=True)
    started = now()
    with (folder / 'driver.log').open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=15)
        result = {'started_utc': started, 'finished_utc': now(), 'exit_code': code, 'timed_out': timed_out}
        save(folder / 'process.json', result)
        require(code == 0 and not timed_out, f'Slime stage failed; inspect {folder / "driver.log"}')
    return result


def dispatch(args, entry, argv=None):
    """Return True when the parent handled the whole check; child returns False."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if args.precision_check_steps == 0 and not args.precision_check_stage:
        require(not args.precision_check_reference and not args.precision_check_manifest and
                not args.precision_check_policy and not args.precision_check_dir,
                'Precision-check options require a positive step count')
        return False
    if args.precision_check_stage:
        require(args.precision_check_manifest, 'Stage is internal; start the parent with --precision-check-steps')
        manifest = load_manifest(args.precision_check_manifest)
        preflight(args)
        configure_child(args, manifest)
        return False
    require(not args.precision_check_manifest, 'Manifest without stage is invalid')
    manifest = prepare(args, entry, argv)
    root = Path(manifest['output'])
    try:
        for stage in STAGES:
            print(f'[precision-check] {stage}: {manifest["steps"]} optimizer steps; log={root / stage / "driver.log"}', flush=True)
            overrides = [value for axis, field in AXES.items()
                         for value in ('--' + field.replace('_', '-'), str(manifest['configurations'][stage][axis]))]
            command = [sys.executable, str(Path(entry).resolve()), *argv, *overrides,
                       '--precision-check-stage', stage,
                       '--precision-check-manifest', str(root / 'manifest.json')]
            run_child(command, root / stage, manifest['stage_timeout_seconds'])
            seal_stage(manifest, stage)
        from .slime_check_compare import compare
        report = compare(root / 'manifest.json')
        print(f'[precision-check] {report["status"]}: {root / "report.md"}', flush=True)
        if report['status'] == 'numerical_failure':
            raise SystemExit(3)
    except Exception as error:
        save(root / 'failure.json', {'status': 'execution_failure', 'error': str(error), 'utc': now()})
        raise
    return True


def driver_finished(args):
    if not getattr(args, 'precision_check_stage', None):
        return False
    root = Path(args.precision_check_dir) / args.precision_check_stage / 'steps' / f'{args.precision_check_steps:06d}'
    return all((root / f'rank{rank}.json').is_file()
               for rank in range(args.actor_num_nodes * args.actor_num_gpus_per_node))


def seal_stage(manifest, stage):
    root = Path(manifest['output']) / stage
    process = read(root / 'process.json')
    require(process['exit_code'] == 0 and not process['timed_out'], 'Child process did not exit successfully')
    receipts = {}
    for step in range(1, manifest['steps'] + 1):
        for rank in range(manifest['world_size']):
            path = root / 'steps' / f'{step:06d}' / f'rank{rank}.json'
            require(path.is_file(), f'Missing optimizer-step receipt: {path}')
            row = read(path)
            require(row['manifest_sha256'] == manifest['sha256'] and
                    (row['stage'], row['step'], row['rank']) == (stage, step, rank), 'Receipt identity differs')
            require(row['observer']['optimizer_step_calls'] == 1 and row['optimizer_updated'], 'Update did not complete')
            receipts[str(path.relative_to(root))] = digest(path)
    require(not (root / 'complete.json').exists(), 'Stage already sealed')
    save(root / 'complete.json', {'manifest_sha256': manifest['sha256'], 'stage': stage,
                                'process_sha256': digest(root / 'process.json'), 'receipts': receipts,
                                'finished_utc': now()})


def install_bridge(source, destination):
    """Copy an existing Slime tree, then add four opt-in bridges. Never edit source."""
    import shutil
    import ast
    source, destination = Path(source).resolve(), Path(destination).resolve()
    require(not destination.exists() and not destination.is_relative_to(source), 'Use a new isolated destination outside source')
    files = {name: (source / name).read_text() for name in ('train.py', 'slime/backends/megatron_utils/model.py')}
    text = files['train.py']
    legacy = '    args = parse_args()\n    train(args)'
    current = ('    args, restore_plan = parse_args(return_restore_plan=True)\n'
               '    configure_logger()\n    main(args, restore_plan)')
    supported = [entry for entry in (legacy, current) if text.count(entry) == 1]
    require(len(supported) == 1, 'Unsupported Slime driver entry; no files changed')
    old = supported[0]
    if old == legacy:
        parse = 'args = parse_args(add_arguments)'
        invoke = 'train(args)'
    else:
        parse = ('args, restore_plan = parse_args(add_arguments, return_restore_plan=True)\n'
                 '        configure_logger()')
        invoke = 'main(args, restore_plan)'
    new = f'''    import sys
    if any(arg.startswith("--precision-check-") for arg in sys.argv[1:]):
        from precision_validator.slime_check import add_arguments, dispatch
        {parse}
        if not dispatch(args, __file__):
            {invoke}
    else:
''' + '\n'.join('    ' + line for line in old.splitlines())
    text = text.replace(old, new)
    # The current driver wraps checkpoint saving in an additional release_train
    # condition. Find that AST boundary instead of depending on line wrapping.
    saves = [node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.If)
             and any(isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name)
                     and child.value.id == 'args' and child.attr == 'save_interval'
                     for child in ast.walk(node.test))]
    require(len(saves) == 1 and saves[0].col_offset == 8, 'Unsupported Slime driver stop boundary')
    lines = text.splitlines(keepends=True)
    stop = '''        if getattr(args, "precision_check_steps", 0):
            from precision_validator.slime_check import driver_finished
            if driver_finished(args):
                break

'''
    lines.insert(saves[0].lineno - 1, stop)
    text = ''.join(lines)
    files['train.py'] = text
    text = files['slime/backends/megatron_utils/model.py']
    functions = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == 'train']
    require(len(functions) == 1, 'Unsupported native train definition')
    function = functions[0]
    lines = text.splitlines(keepends=True)
    before, after = ''.join(lines[:function.lineno - 1]), ''.join(lines[function.end_lineno:])
    text = ''.join(lines[function.lineno - 1:function.end_lineno])
    old = '    for step_id in range(num_steps_per_rollout):\n'
    require(text.count(old) == 1, 'Unsupported Slime training loop boundary')
    text = text.replace(old, '''    for step_id in range(num_steps_per_rollout):
        run_step = train_one_step
        if getattr(args, "precision_check_steps", 0):
            from precision_validator.slime_check_capture import reached_limit, observed_step
            if reached_limit(args):
                break
            from functools import partial
            run_step = partial(observed_step, train_one_step)
''')
    old = '        loss_dict, grad_norm = train_one_step(\n'
    require(text.count(old) == 1, 'Unsupported Slime step invocation')
    files['slime/backends/megatron_utils/model.py'] = before + text.replace(old, '        loss_dict, grad_norm = run_step(\n') + after
    for name, value in files.items():
        compile(value, name, 'exec')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    for name, value in files.items():
        (destination / name).write_text(value)
    save(destination.parent / (destination.name + '-bridge.json'), {
        'source': str(source), 'destination': str(destination), 'changed_files': list(files),
        'source_sha256': {name: digest(source / name) for name in files},
        'patched_sha256': {name: digest(destination / name) for name in files}, 'utc': now()})
    return str(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    bridge = sub.add_parser('install-bridge')
    bridge.add_argument('--source', required=True)
    bridge.add_argument('--destination', required=True)
    cli = parser.parse_args()
    print(install_bridge(cli.source, cli.destination))
