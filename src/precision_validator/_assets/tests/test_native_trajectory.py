import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import test_megatron_observer as observer_tests
from precision_validator.native_trajectory import validate_spec, STATE_FIELDS, FIELDS
from precision_validator.native_trajectory_sdk import NativeTrajectoryObserver, StepWriter, adam_counter

torch = observer_tests.torch


@unittest.skipIf(torch is None, 'native PyTorch/Megatron unavailable')
class NativeTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        observer_tests.ObserverTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        observer_tests.ObserverTests.tearDownClass()

    def fixture(self):
        helper = observer_tests.ObserverTests()
        self.addCleanup(helper.doCleanups)
        model,mapping,opt,writer,precision = helper.fixture(torch.bfloat16)
        plan = dict(writer.plan,adapter='native-trajectory-v1',contract={'optimizer_steps':3},
                    trajectory={'steps':[{'observables':{'loss':[]}}]*3})
        return model,mapping,opt,plan,precision

    def test_accumulated_moments_and_unrounded_master_updates(self):
        model,mapping,opt,plan,precision = self.fixture()
        for step in (1,2,3):
            writer=StepWriter(plan,'reference',0,step)
            before=opt.master.detach().clone()
            with NativeTrajectoryObserver(mapping,opt,writer,precision,step):opt.step()
            storage=writer.storage.close()
            self.assertEqual(set(p['field'] for p in storage['pieces']),set(FIELDS))
            for field,value in (('updates',opt.master-before),('master_parameters',opt.master),
                                ('parameters',model.weight),('exp_avg',opt.optimizer.state[opt.master]['exp_avg']),
                                ('exp_avg_sq',opt.optimizer.state[opt.master]['exp_avg_sq'])):
                actual=np.fromfile(writer.folder/f'{field}.rank0.f32',dtype='<f4')
                np.testing.assert_array_equal(actual,value.detach().float().numpy())
            self.assertGreater(float(np.abs(np.fromfile(writer.folder/'updates.rank0.f32',dtype='<f4')).sum()),0)
            self.assertEqual(float(opt.optimizer.state[opt.master]['step']),step)

    def test_nonfresh_step_requires_prior_counter_and_full_states(self):
        for defect in ('counter','missing_moment','wrong_precision'):
            with self.subTest(defect=defect):
                _,mapping,opt,plan,precision=self.fixture()
                writer=StepWriter(plan,'reference',0,1)
                with NativeTrajectoryObserver(mapping,opt,writer,precision,1):opt.step()
                writer.storage.close()
                state=opt.optimizer.state[opt.master]
                if defect=='counter':state['step'].zero_()
                elif defect=='missing_moment':del state['exp_avg_sq']
                else:state['exp_avg']=state['exp_avg'].bfloat16()
                with self.assertRaises(RuntimeError):
                    with NativeTrajectoryObserver(mapping,opt,StepWriter(plan,'reference',0,2),precision,2):pass

    def test_native_group_counter_and_invalid_protocol(self):
        _,mapping,opt,plan,precision=self.fixture()
        from precision_validator.megatron_observer import optimizer_shards
        row=next(iter(optimizer_shards(opt,mapping).values()))
        opt.optimizer.param_groups[0]['step']=2
        self.assertEqual(adam_counter(row),2)
        plan['adapter']='tensor-protocol-v1'
        with self.assertRaisesRegex(RuntimeError,'trajectory contract'):
            NativeTrajectoryObserver(mapping,opt,StepWriter(plan,'reference',0,1),precision,1)

    def test_step_observations_have_independent_shapes(self):
        _,_,_,plan,_=self.fixture()
        plan['observables']={'loss':[],'logits':[4]}
        plan['trajectory']['steps']=[{'observables':{'loss':[],'logits':[n]}} for n in (4,2,3)]
        writer=StepWriter(plan,'reference',0,2)
        writer.observe('logits',[1,2])
        with self.assertRaises(RuntimeError):writer.observe('logits',[1,2])
        writer=StepWriter(plan,'reference',0,3)
        with self.assertRaises(RuntimeError):writer.observe('logits',[1,2])

    def test_checkpoint_leaf_restore_is_outside_autograd_and_exact(self):
        import precision_validator.native_trajectory_sdk as sdk
        from precision_validator.trajectory_sdk import rng_state, restore_rng
        def configure(opt):
            type(opt).param_groups = property(lambda value:value.optimizer.param_groups)
            opt.state_dict = lambda:opt.optimizer.state_dict()
            opt.load_state_dict = lambda state:opt.optimizer.load_state_dict(state)
            opt.get_parameter_state_dp_zero = lambda:{'param':opt.master.detach().clone()}
            def load_main(state):
                self.assertFalse(torch.is_grad_enabled())
                opt.master.copy_(state['param'])  # A real leaf parameter; fails with autograd enabled.
            opt.load_parameter_state_from_dp_zero = load_main
            return opt
        model,mapping,opt,plan,precision=self.fixture();configure(opt)
        opt.step()
        plan['plan_sha256']='c'*64
        plan['artifacts']={'input':{'sha256':'d'*64}}
        plan['configurations']['resume']={'world_size':1}
        plan['trajectory'].update(resume_after_step=1,steps=[{'input_artifact':'input'}]*3)
        (Path(plan['output'])/'resume').mkdir()
        scheduler_state={'num_steps':4}
        scheduler=SimpleNamespace(state_dict=lambda:dict(scheduler_state),
                                  load_state_dict=lambda value:scheduler_state.update(value))
        with patch.object(sdk,'native_rng',lambda:{'standard':rng_state(),'megatron':{}}), \
             patch.object(sdk,'restore_native_rng',lambda value:restore_rng(value['standard'])):
            with patch.object(sdk.os,'getpid',return_value=987654321):
                record=sdk.save_checkpoint(model,mapping,opt,scheduler,plan,0,{},'prefix')
            new_model,new_mapping,new_opt,_,_=self.fixture();configure(new_opt)
            result,evidence=sdk.restore_checkpoint(new_model,new_mapping,new_opt,scheduler,plan,0,'suffix')
            self.assertEqual(record['state'],evidence['restored_state'])
            self.assertTrue(torch.equal(opt.master,new_opt.master))
            self.assertTrue(torch.equal(model.weight,new_model.weight))
            self.assertTrue(torch.is_grad_enabled())

    def test_class_metadata_is_safe_and_bound_to_live_optimizer(self):
        import precision_validator.native_trajectory_sdk as sdk
        import io
        classes={'torch.optim.adamw.AdamW':torch.optim.AdamW}
        value={'param_groups':[{'cpu_optimizer_cls':torch.optim.AdamW,'lr':.001}], 'state':{0:torch.ones(2)}}
        packed=sdk.class_metadata(value,classes)
        stream=io.BytesIO();torch.save(packed,stream);stream.seek(0)
        loaded=torch.load(stream,weights_only=True)
        actual=sdk.class_metadata(loaded,classes,restore=True)
        self.assertIs(actual['param_groups'][0]['cpu_optimizer_cls'],torch.optim.AdamW)
        self.assertTrue(torch.equal(actual['state'][0],value['state'][0]))
        with self.assertRaises(RuntimeError):sdk.class_metadata(loaded,{},restore=True)
        with self.assertRaises(RuntimeError):sdk.class_metadata({'unexpected':str},classes)

    def test_stale_cpu_master_metadata_cannot_mask_actual_parameter(self):
        from precision_validator.megatron_observer import master_parameter
        _,_,opt,_,_=self.fixture()
        opt.optimizer.state[opt.master]['master_param']=opt.master.detach().clone()+1
        with self.assertRaisesRegex(RuntimeError,'actual Adam parameter'):
            master_parameter(opt.optimizer,opt.master)
        opt.optimizer.state[opt.master]['master_param']=opt.master.detach().clone()
        self.assertIs(master_parameter(opt.optimizer,opt.master),opt.master)

    def test_redundant_group_counter_does_not_replace_actual_adam_counter(self):
        import precision_validator.native_trajectory_sdk as sdk
        model,mapping,opt,plan,_=self.fixture()
        type(opt).param_groups=property(lambda value:value.optimizer.param_groups)
        opt.step()
        plan['artifacts']={'input':{'sha256':'e'*64}}
        plan['trajectory']['steps']=[{'input_artifact':'input'}]*3
        scheduler=SimpleNamespace(state_dict=lambda:{'num_steps':4})
        with patch.object(sdk,'native_rng',lambda:{}):
            before=sdk.state_signature(model,mapping,opt,scheduler,plan,1)
            opt.optimizer.param_groups[0]['step']=1
            self.assertEqual(before,sdk.state_signature(model,mapping,opt,scheduler,plan,1))
            opt.optimizer.state[opt.master]['step'].add_(1)
            self.assertNotEqual(before,sdk.state_signature(model,mapping,opt,scheduler,plan,1))

    def test_fresh_inspection_does_not_materialize_optimizer_state(self):
        import precision_validator.native_trajectory_sdk as sdk
        model,mapping,opt,plan,precision=self.fixture()
        type(opt).param_groups=property(lambda value:value.optimizer.param_groups)
        plan['artifacts']={'input':{'sha256':'e'*64}}
        plan['trajectory']['steps']=[{'input_artifact':'input'}]*3
        scheduler=SimpleNamespace(state_dict=lambda:{'num_steps':0})
        self.assertFalse(opt.optimizer.state)
        with patch.object(sdk,'native_rng',lambda:{}):
            sdk.state_signature(model,mapping,opt,scheduler,plan,0)
        self.assertFalse(opt.optimizer.state, 'Read-only inspection must not defeat native checkpoint lazy initialization')
        writer=StepWriter(plan,'reference',0,1)
        with NativeTrajectoryObserver(mapping,opt,writer,precision,1):
            self.assertFalse(opt.optimizer.state)
            opt.step()
        writer.storage.close()


if __name__=='__main__':unittest.main()
