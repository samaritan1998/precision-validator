import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from precision_validator.common import ValidationError, object_hash, save
from precision_validator.slime_check import (add_arguments, reference_layout, layout_from_args,
    install_bridge, seal_stage, driver_finished, run_child, load_manifest, preflight)
from precision_validator.slime_check_capture import global_inputs
from precision_validator.slime_check_compare import compare
from precision_validator.parallel import CONVENTION
from precision_validator.storage import ShardWriter


def single():
    return dict(tp=1, pp=1, cp=1, ep=1, etp=1, dp=1)


def fixture(root, *, strict=True, bad=None):
    """Synthetic receipt protocol tests, NOT model/GPU acceptance evidence."""
    source = root / 'code'
    source.mkdir()
    manifest = {'schema': 'slime-training-check-v1', 'scope': 'unit_fixture', 'output': str(root),
                'world_size': 1, 'steps': 3, 'configurations': {s: single() for s in ('reference', 'repeat', 'candidate')},
                'source': {'root': str(source), 'files': {}}, 'toolkit': {'root': str(source), 'files': {}}}
    local = dict(relative_l2=0.001, max_abs=0.001, near_zero_rms=1e-8, near_zero_max_abs=1e-8)
    policy = {field: dict(repeat_relative_ceiling=0.001, target_relative_ceiling=0.001, noise_multiplier=3, noise_slack=0.001)
              for field in ('gradients', 'updates')}
    acceptance = {kind: [dict(pattern='*', gradients=local, updates=local)] for kind in ('parameters', 'groups')}
    acceptance['observations'] = {'loss': local}
    manifest['rules'] = {'policy': policy if strict else None, 'acceptance': acceptance if strict else None}
    manifest['sha256'] = object_hash(manifest)
    save(root / 'manifest.json', manifest)
    catalog = {'large': {'shape': [2], 'group': 'dense'}, 'small': {'shape': [1], 'group': 'small'}}
    groups = {'convention': CONVENTION, 'degrees': {**single(), 'edp': 1},
              'groups': {axis: [0] for axis in ('tp', 'pp', 'cp', 'ep', 'etp', 'dp', 'edp')}}
    for stage in ('reference', 'repeat', 'candidate'):
        for step in range(1, 4):
            folder = root / stage / 'steps' / f'{step:06d}'
            folder.mkdir(parents=True)
            writer = ShardWriter(folder, 0, 1024)
            for field in ('gradients', 'updates'):
                for name, values in [('large', [10000., 10000.]), ('small', [0.01])]:
                    if bad == 'local' and stage == 'candidate' and name == 'small' and step == 2:
                        values = [-v for v in values]
                    writer.append(field, dict(name=name, shape=catalog[name]['shape'], start=0, end=len(values)), [values])
            objective = {'global_samples': 2, 'supervised_tokens': 6, 'contributions_sha256': 'f' * 64}
            if bad == 'input' and stage == 'candidate':
                objective['contributions_sha256'] = 'e' * 64
            row = dict(manifest_sha256=manifest['sha256'], stage=stage, step=step, rank=0, catalog=catalog,
                       optimizer_updated=True, observer=dict(optimizer_step_calls=1, prepare_calls=1, optimizer_state_steps=[step]),
                       precision=dict(parameters='float32', gradients='float32', optimizer_parameters='float32', optimizer_state='float32'),
                       initial={name: dict(shape=value['shape'], dtype='torch.float32', sha256='a' * 64)
                                for name, value in catalog.items()} if step == 1 else {},
                       storage=writer.close(), objective=objective, normalization=dict(loss_type='sft_loss'),
                       scheduler_steps=step * 2, runtime={'torch': 'fixture'}, device='fixture-CPU', native_parallelism=groups,
                       optimizer_options=[{'lr': 0.01}],
                       observations={'loss': dict(shape=[], values=[1e6 if bad == 'loss' and stage == 'candidate' else 1.])})
            save(folder / 'rank0.json', row)
        save(root / stage / 'process.json', {'exit_code': 0, 'timed_out': False})
        seal_stage(manifest, stage)
    return manifest


class SlimeCheckTests(unittest.TestCase):
    def test_flags_are_opt_in(self):
        p = add_arguments(argparse.ArgumentParser())
        self.assertEqual(p.parse_args([]).precision_check_steps, 0)
        self.assertEqual(p.parse_args(['--precision-check-steps', '3']).precision_check_steps, 3)

    def test_reference_preserves_other_axes(self):
        value = dict(tp=2, pp=2, cp=2, ep=1, etp=2, dp=1)
        actual = reference_layout('cp=1', value, 8)
        self.assertEqual(actual, dict(tp=2, pp=2, cp=1, ep=1, etp=2, dp=2))

    def test_reference_rejects_duplicate_or_invalid_axes(self):
        for value in ('tp=0', 'tp=2,tp=1', 'cp=3', 'dp=2', 'tp=abc'):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                reference_layout(value, single(), 1)

    def test_etp_defaults_to_tp(self):
        args = SimpleNamespace(actor_num_nodes=1, actor_num_gpus_per_node=2, tensor_model_parallel_size=2)
        self.assertEqual(layout_from_args(args)[0]['etp'], 2)

    def test_fresh_hf_load_is_allowed_but_resume_rejected(self):
        args = SimpleNamespace(precision_check_steps=3, loss_type='sft_loss', hf_checkpoint='/tmp/model', load='/tmp/model',
                               use_distributed_optimizer=True, precision_check_disk_gib=1., precision_check_timeout=5)
        preflight(args)
        args.load = '/tmp/resume'
        with self.assertRaises(ValidationError): preflight(args)

    def test_release_train_rejected_before_replacing_model(self):
        args = SimpleNamespace(precision_check_steps=3, loss_type='sft_loss', hf_checkpoint='/tmp/model',
                               use_distributed_optimizer=True, precision_check_disk_gib=1.,
                               precision_check_timeout=5, release_train=True)
        with self.assertRaisesRegex(ValidationError, 'release_train'):
            preflight(args)

    def test_current_driver_keeps_restore_plan_and_cleanup_entry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'source'
            (source / 'slime/backends/megatron_utils').mkdir(parents=True)
            (source / 'train.py').write_text('''from types import SimpleNamespace
def parse_args(*args, return_restore_plan=False):
    assert return_restore_plan
    return SimpleNamespace(), "restore-sentinel"
def configure_logger(): pass
def main(args, restore_plan):
    assert restore_plan == "restore-sentinel"
    print("ordinary-current-training")
def never_called():
    for rollout_id in []:
        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            pass
if __name__ == "__main__":
    args, restore_plan = parse_args(return_restore_plan=True)
    configure_logger()
    main(args, restore_plan)
''')
            (source / 'slime/backends/megatron_utils/model.py').write_text('''def train():
    for step_id in range(num_steps_per_rollout):
        loss_dict, grad_norm = train_one_step(
            args,
        )
''')
            before = (source / 'train.py').read_bytes()
            output = root / 'copy'
            install_bridge(source, output)
            self.assertEqual((source / 'train.py').read_bytes(), before)
            done = subprocess.run([sys.executable, '-I', str(output / 'train.py')], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(done.stdout.strip(), 'ordinary-current-training')
            patched = (output / 'train.py').read_text()
            self.assertIn('parse_args(add_arguments, return_restore_plan=True)', patched)

    def test_unknown_driver_is_rejected_without_copying(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'source'
            (source / 'slime/backends/megatron_utils').mkdir(parents=True)
            (source / 'train.py').write_text('raise RuntimeError("unsupported")\n')
            (source / 'slime/backends/megatron_utils/model.py').write_text('def train(): pass\n')
            destination = Path(d) / 'copy'
            with self.assertRaisesRegex(ValidationError, 'Unsupported Slime driver'):
                install_bridge(source, destination)
            self.assertFalse(destination.exists())

    def test_global_samples_keep_duplicate_contributions(self):
        item = dict(sha256='same', tokens=8, supervised_tokens=3)
        rows = [dict(dp_rank=dp, inputs=[item]) for dp in (0, 0, 1, 1)]
        value = global_inputs(rows, 2)
        self.assertEqual((value['global_samples'], value['supervised_tokens']), (2, 6))

    def test_replica_input_drift_rejected(self):
        one = dict(sha256='one', tokens=8, supervised_tokens=3)
        two = dict(sha256='two', tokens=8, supervised_tokens=3)
        with self.assertRaises(ValidationError):
            global_inputs([dict(dp_rank=0, inputs=[one]), dict(dp_rank=0, inputs=[two])], 1)

    def test_wrong_global_batch_rejected(self):
        with self.assertRaises(ValidationError): global_inputs([], 2)

    def test_three_step_full_report(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(Path(d))
            report = compare(Path(d) / 'manifest.json')
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(len(report['steps']), 3)
            self.assertTrue(report['acceptance_coverage']['strict'])
            self.assertFalse(report['acceptance_coverage']['optimizer_state_vectors'])

    def test_local_error_cannot_hide_in_global_norm(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(Path(d), bad='local')
            report = compare(Path(d) / 'manifest.json')
            self.assertEqual(report['status'], 'numerical_failure')
            self.assertTrue(report['steps'][1]['gates']['gradients']['passed'])
            self.assertFalse(report['steps'][1]['gates']['parameters/gradients/small/rule0']['passed'])

    def test_loss_failure(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(Path(d), bad='loss')
            report = compare(Path(d) / 'manifest.json')
            self.assertEqual(report['status'], 'numerical_failure')

    def test_no_policy_never_passed(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(Path(d), strict=False)
            self.assertEqual(compare(Path(d) / 'manifest.json')['status'], 'report_only')

    def test_actual_input_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(Path(d), bad='input')
            with self.assertRaisesRegex(ValidationError, 'objective'): compare(Path(d) / 'manifest.json')

    def test_raw_tensor_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root)
            p = root / 'candidate/steps/000003/gradients.rank0.f32'
            p.write_bytes(b'\0' * p.stat().st_size)
            with self.assertRaises(ValidationError): compare(root / 'manifest.json')

    def test_missing_final_step_does_not_seal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = fixture(root)
            (root / 'candidate/complete.json').unlink()
            (root / 'candidate/steps/000003/rank0.json').unlink()
            with self.assertRaises(ValidationError): seal_stage(m, 'candidate')
            args = SimpleNamespace(precision_check_stage='candidate', precision_check_dir=d,
                                   precision_check_steps=3, actor_num_nodes=1, actor_num_gpus_per_node=1)
            self.assertFalse(driver_finished(args))

    def test_child_failure_never_success(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValidationError):
                run_child([sys.executable, '-c', 'raise SystemExit(7)'], Path(d) / 'child', 10)
            self.assertEqual(json.loads((Path(d) / 'child/process.json').read_text())['exit_code'], 7)

    def test_child_timeout_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValidationError):
                run_child([sys.executable, '-c', 'import time;time.sleep(30)'], Path(d) / 'child', 0.05)
            self.assertTrue(json.loads((Path(d) / 'child/process.json').read_text())['timed_out'])

    def test_unmodified_driver_does_not_import_plugin(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'source'; source.mkdir()
            (source / 'slime/backends/megatron_utils').mkdir(parents=True)
            (source / 'train.py').write_text('''from types import SimpleNamespace
def parse_args(): return SimpleNamespace()
def train(args): print("ordinary-training")
def never_called():
    for rollout_id in []:
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            pass
if __name__ == "__main__":
    args = parse_args()
    train(args)
''')
            (source / 'slime/backends/megatron_utils/model.py').write_text('''def train():
    for step_id in range(num_steps_per_rollout):
        loss_dict, grad_norm = train_one_step(
            args,
        )
''')
            before = (source / 'train.py').read_bytes()
            output = root / 'copy'
            install_bridge(source, output)
            self.assertEqual((source / 'train.py').read_bytes(), before)
            done = subprocess.run([sys.executable, '-I', str(output / 'train.py')], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(done.stdout.strip(), 'ordinary-training')
            with self.assertRaises(ValidationError): install_bridge(source, output)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is required for input trace execution')
class InputTraceTests(unittest.TestCase):
    def test_trace_uses_actual_iterator_values_and_restores(self):
        import torch
        from precision_validator.slime_check_capture import InputTrace
        class Iterator:
            def get_next(self, keys):
                return dict(tokens=[torch.tensor([2, 3, 4])], loss_masks=[torch.tensor([1, 0])])
        it = Iterator(); original = it.get_next
        with InputTrace(it) as trace:
            batch = it.get_next(['tokens', 'loss_masks'])
            self.assertEqual(trace.rows[0]['supervised_tokens'], 1)
            self.assertTrue(torch.equal(batch['tokens'][0], torch.tensor([2, 3, 4])))
        self.assertEqual(it.get_next, original)

    def test_trace_restores_after_error(self):
        from precision_validator.slime_check_capture import InputTrace
        class Iterator:
            def get_next(self, keys): return {'tokens': None}
        it = Iterator(); original = it.get_next
        with self.assertRaises(ValidationError), InputTrace(it): it.get_next([])
        self.assertEqual(it.get_next, original)


if __name__ == '__main__':
    unittest.main()
