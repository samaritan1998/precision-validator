import hashlib
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from collections import defaultdict

import numpy as np

try:
    import torch
    import torch.distributed as dist
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
except ImportError:
    torch = None

from precision_validator.megatron_observer import NativeStepObserver, capture_initial, canonical_digests, master_parameter
from precision_validator.megatron_mapping import MegatronParameterMap
from precision_validator.sdk import RankWriter
from precision_validator.storage import ShardWriter


@unittest.skipIf(torch is None, 'native Megatron/PyTorch runtime is unavailable')
class ObserverTests(unittest.TestCase):
    def test_slime_entry_import_does_not_resolve_megatron_examples_package(self):
        from precision_validator.common import ASSET_ROOT
        result=subprocess.run([sys.executable,str(ASSET_ROOT/'examples/slime_parallel.py'),'make','--help'],
                              capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--dtype',result.stdout)

    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        dist.init_process_group('gloo', init_method='file://' + str(Path(cls.shared.name)/'init'), rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.shared.cleanup()

    def fixture(self, dtype=None):
        dtype = dtype or torch.float32
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.,2.,3.,4.],dtype=dtype))
            def sharded_state_dict(self, metadata):
                return {'weight':ShardedTensor.from_rank_offsets('weight',self.weight)}
        model = Model()
        mapping = MegatronParameterMap(model)
        class Optimizer:
            def __init__(self):
                self.master = torch.nn.Parameter(model.weight.detach().float().clone())
                self.optimizer = torch.optim.Adam([self.master],lr=1e-5,eps=1e-8)
                self.model_param_group_index_map = {model.weight:(0,0)}
            def _get_model_param_range_map(self, parameter):
                return {'param':SimpleNamespace(start=0,end=4)}
            def prepare_grads(self):
                self.master.grad = torch.tensor([2.,-3.,4.,-5.])
                return False
            def step(self):
                self.prepare_grads()
                torch.nn.utils.clip_grad_norm_([self.master],0.1)
                self.optimizer.step()
                with torch.no_grad():model.weight.copy_(self.master)
                return True,7.0,None
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        plan = {'parameters':mapping.catalog(),'configurations':{'reference':{'world_size':1}},
                'output':temporary.name,'disk_budget_bytes':1000000}
        writer = RankWriter(plan,'reference',0)
        precision = {'parameters':str(dtype).removeprefix('torch.'),'gradients':'float32',
                     'optimizer_parameters':'float32','optimizer_state':'float32'}
        optimizer = Optimizer()
        return model,mapping,optimizer,writer,precision

    def test_preclip_gradient_and_unrounded_bf16_update(self):
        model,mapping,optimizer,writer,precision = self.fixture(torch.bfloat16)
        capture_initial(mapping,writer,torch.bfloat16)
        self.assertFalse((writer.folder/'initial-scratch').exists())
        initial = model.weight.detach().clone()
        with NativeStepObserver(mapping,optimizer,writer,precision) as observer:
            optimizer.step()
            optimizer.optimizer.zero_grad(set_to_none=True)  # Slime clears gradients before returning.
        storage = writer.storage.close()
        np.testing.assert_array_equal(np.fromfile(writer.folder/'gradients.rank0.f32',dtype='<f4'),[2,-3,4,-5])
        update = np.fromfile(writer.folder/'updates.rank0.f32',dtype='<f4')
        self.assertTrue(np.all(np.abs(update)>0))
        self.assertTrue(torch.equal(model.weight,initial))  # Rounded model update would falsely look zero.
        self.assertEqual(observer.receipt['optimizer_step_calls'],1)
        self.assertEqual(len(storage['pieces']),2)
        self.assertEqual(writer.initial['weight']['sha256'],hashlib.sha256(initial.view(torch.uint8).numpy().tobytes()).hexdigest())

    def test_hooks_restored_on_error(self):
        _,mapping,opt,writer,precision = self.fixture()
        original_prepare,original_step = opt.prepare_grads,opt.step
        with self.assertRaisesRegex(ValueError,'injected'):
            with NativeStepObserver(mapping,opt,writer,precision):
                raise ValueError('injected')
        self.assertEqual(opt.prepare_grads,original_prepare)
        self.assertEqual(opt.step,original_step)

    def test_fused_adam_group_counter_checked_on_final_step(self):
        # TE fused Adam stores its counter in the parameter group. A final
        # captured step has no subsequent before-step check to catch a freeze.
        for counter in (1, 0, 2):
            with self.subTest(counter=counter):
                _, mapping, opt, writer, precision = self.fixture()
                original_step = opt.optimizer.step
                def fused_step():
                    original_step()
                    opt.optimizer.state[opt.master].pop('step')
                    opt.optimizer.param_groups[0]['step'] = counter
                opt.optimizer.step = fused_step
                observer = NativeStepObserver(mapping, opt, writer, precision)
                try:
                    if counter == 1:
                        with observer:
                            opt.step()
                        self.assertEqual(observer.receipt['optimizer_state_steps'], [1.0])
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'step counter differs'):
                            with observer:
                                opt.step()
                finally:
                    writer.storage.close()

    def test_nonfresh_fused_adam_group_counter_rejected(self):
        _, mapping, opt, writer, precision = self.fixture()
        opt.optimizer.param_groups[0]['step'] = 3
        with self.assertRaisesRegex(RuntimeError, 'did not start fresh'):
            with NativeStepObserver(mapping, opt, writer, precision):
                pass

    def test_reused_optimizer_and_wrong_master_rejected(self):
        for kind in ('counter','moment','master'):
            with self.subTest(kind=kind):
                _,mapping,opt,writer,precision = self.fixture()
                if kind=='counter':opt.optimizer.state[opt.master]['step']=torch.tensor(1.)
                if kind=='moment':opt.optimizer.state[opt.master]['exp_avg']=torch.ones(4)
                if kind=='master':opt.master.data[0]+=1
                with self.assertRaises(RuntimeError):
                    with NativeStepObserver(mapping,opt,writer,precision):pass

    def test_incomplete_and_duplicate_steps_rejected(self):
        for kind in ('missing','duplicate'):
            with self.subTest(kind=kind):
                _,mapping,opt,writer,precision = self.fixture()
                with self.assertRaises(RuntimeError):
                    with NativeStepObserver(mapping,opt,writer,precision):
                        if kind=='duplicate':opt.step();opt.step()
                writer.storage.close()

    def test_nonfinite_gradient_rejected(self):
        _,mapping,opt,writer,precision = self.fixture()
        opt.prepare_grads=lambda:True
        with self.assertRaisesRegex(RuntimeError,'nonfinite'):
            with NativeStepObserver(mapping,opt,writer,precision):opt.step()

    def test_native_remainder_decoding(self):
        # Exercise signed low bits rather than treating the low half as unsigned.
        base=torch.tensor([1.,-2.,3.,-4.],dtype=torch.bfloat16)
        remainder=torch.tensor([-1,1,-32768,32767],dtype=torch.int16)
        opt=SimpleNamespace(state={base:{'master_param':remainder}},store_param_remainders=True)
        expected=((base.view(torch.int16).int()<<16)+remainder.int()).view(torch.float32)
        self.assertTrue(torch.equal(master_parameter(opt,base),expected))
        with self.assertRaisesRegex(RuntimeError,'no FP32 master'):
            master_parameter(SimpleNamespace(state={base:{}}),base)

    def hybrid_fixture(self):
        model,mapping,opt,writer,precision = self.fixture(torch.bfloat16)
        adam = opt.optimizer
        # Native HybridDeviceOptimizer exposes these ownership maps before
        # lazily copying sub-optimizer state into its public state dictionary.
        opt.optimizer = SimpleNamespace(state=defaultdict(dict),
            param_groups=[{'params':[model.weight]}],param_update_in_fp32=True,
            param_to_inner_param={model.weight:opt.master},sub_optimizers=[adam],step=adam.step)
        def prepare():
            model.weight.decoupled_grad = torch.tensor([2.,-3.,4.,-5.])
            opt.master.grad = model.weight.decoupled_grad.clone()
            return False
        opt.prepare_grads = prepare
        return model,mapping,opt,writer,precision,adam

    def test_hybrid_lazy_state_uses_actual_cpu_master(self):
        model,mapping,opt,writer,precision,adam = self.hybrid_fixture()
        self.assertFalse(opt.optimizer.state)
        initial = model.weight.detach().clone()
        with NativeStepObserver(mapping,opt,writer,precision):opt.step()
        writer.storage.close()
        self.assertTrue(torch.equal(model.weight,initial))
        actual = np.fromfile(writer.folder/'updates.rank0.f32',dtype='<f4')
        np.testing.assert_array_equal(actual,(opt.master.detach()-initial.float()).numpy())
        self.assertTrue(np.all(actual!=0))
        self.assertEqual(adam.state[opt.master]['step'],1)

    def test_hybrid_nonfresh_inner_state_rejected(self):
        for key in ('step','exp_avg','exp_avg_sq'):
            with self.subTest(key=key):
                _,mapping,opt,writer,precision,adam = self.hybrid_fixture()
                adam.state[opt.master][key] = torch.tensor(1.) if key=='step' else torch.ones_like(opt.master)
                self.assertFalse(opt.optimizer.state)
                with self.assertRaisesRegex(RuntimeError,'fresh|nonzero'):
                    with NativeStepObserver(mapping,opt,writer,precision):pass

    def test_hybrid_invalid_owner_or_master_rejected(self):
        for kind in ('duplicate','shape','dtype'):
            with self.subTest(kind=kind):
                model,mapping,opt,writer,precision,adam = self.hybrid_fixture()
                if kind=='duplicate':opt.optimizer.sub_optimizers.append(adam)
                else:
                    value=torch.ones(2 if kind=='shape' else 4,dtype=torch.bfloat16 if kind=='dtype' else torch.float32)
                    opt.optimizer.param_to_inner_param[model.weight]=value
                    adam.param_groups[0]['params']=[value]
                with self.assertRaises(RuntimeError):
                    with NativeStepObserver(mapping,opt,writer,precision):pass

    def test_canonical_initial_hash_independent_of_shard_order(self):
        with tempfile.TemporaryDirectory() as root:
            storages=[]
            for rank,(a,b) in enumerate(((2,4),(0,2))):
                writer=ShardWriter(root,rank,100,fields=('parameters',))
                writer.append('parameters',{'name':'x','shape':[4],'start':a,'end':b},[np.arange(a,b,dtype=np.float32)])
                storages.append(writer.close())
            actual=canonical_digests(root,storages,{'x':{'shape':[4]}},torch.float32)
            self.assertEqual(actual['x'],hashlib.sha256(np.arange(4,dtype='<f4').tobytes()).hexdigest())
            with self.assertRaisesRegex(RuntimeError,'coverage|ownership'):
                canonical_digests(root,storages[:1],{'x':{'shape':[4]}},torch.float32)


if __name__=='__main__':unittest.main()
