"""Synthetic protocol corruption tests; not native training acceptance."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import test_protocol as common_fixtures
from test_parallel import strict_policy, synthetic_groups
from precision_validator.common import save, read, digest, load_plan, ValidationError
from precision_validator.native_trajectory import prepare, validate_spec, STAGES, FIELDS, STATE_FIELDS, STORAGE_ALIGNMENT, identity
from precision_validator.precision import stage_precision
from precision_validator.native_trajectory_compare import compare_suite
from precision_validator.native_trajectory_sdk import StepWriter
from precision_validator.parallel import parse_layout, CONVENTION
from precision_validator.trajectory_sdk import cursor_for


def fixture(root, cross_storage=False):
    spec = strict_policy(common_fixtures.specification(root/'inputs'))
    spec['contract'].update(optimizer_steps=3,optimizer={'name':'MegatronDistributedAdam','options':{'lr':.01}},
        precision={k:'float32' for k in ('parameters','gradients','optimizer_parameters','optimizer_state')},
        objective={'normalization':'global_supervised_units','global_samples':2,'supervised_units':4})
    spec['parallel_adapter']={'convention':CONVENTION,'framework':'synthetic_protocol_fixture','supported_axes':['tp','dp']}
    for name,world in (('reference',1),('candidate',2)):
        spec['configurations'][name].update(world_size=world,parallelism=parse_layout('tp=1,pp=1,cp=1,ep=1,etp=1',world))
    limit=spec['acceptance']['parameters'][0]['gradients']
    spec['trajectory']={'resume_after_step':1,'steps':[{'input_artifact':'input','objective':spec['contract']['objective'],
        'observables':spec['observables']} for _ in range(3)],
        'scheduler':{'name':'MegatronOptimizerParamScheduler','options':{}},
        'state_acceptance':{cat:[{'pattern':'*',**{f:copy.deepcopy(limit) for f in STATE_FIELDS}}] for cat in ('parameters','groups')}}
    if cross_storage:
        precision = spec['contract'].pop('precision')
        spec['contract']['precision_by_stage'] = {'reference': precision,
            'candidate': dict(precision, parameters='bfloat16')}
        spec['contract']['precision_comparison'] = {'allowed_changes': ['parameters'],
            'initial_weights_alignment': STORAGE_ALIGNMENT}
    save(root/'spec.json',spec);prepare(root/'spec.json',root/'results',budget_gib=.1)
    path=root/'results/plan.json';plan=load_plan(path)
    weights={name:np.ones(info['shape'] or (),np.float32).reshape(-1) for name,info in spec['parameters'].items()}
    def state(step):
        return {'model':{},'optimizer_shards':{'fake':{'step':step,'master':{},'exp_avg':{},'exp_avg_sq':{}}},
                'scheduler':{'num_steps':step*2},'data_cursor':cursor_for(plan,step),'optimizer_groups':[{'lr':.01}], 'rng':{'value':step}}
    for stage in STAGES:
        world=plan['configurations'][stage]['world_size'];folder=path.parent/stage
        folder.mkdir()
        groups=synthetic_groups(plan['configurations'][stage]['parallelism'],world)
        phases=['prefix','suffix'] if stage=='resume' else ['continuous']
        workers=[]
        for phase in phases:
            exits={}
            for rank in range(world):
                epath=folder/f'exit.{phase}.rank{rank}.json'
                save(epath,{'plan_sha256':plan['plan_sha256'],'stage':stage,'rank':rank,'phase':phase,
                    'teardown_completed':True,'process_id':phase+str(rank)})
                exits[epath.name]=digest(epath)
            workers.append({'phase':phase,'returncode':0,'ranks':exits})
        save(folder/'workers.json',workers)
        finals={}
        for rank in range(world):
            receipts={}
            for number in range(1,4):
                writer=StepWriter(plan,stage,rank,number)
                if rank==0:
                    for name,info in spec['parameters'].items():
                        for field in FIELDS:writer.append(field,name,0,len(weights[name]),[weights[name]*number])
                    writer.observe('loss',1.);writer.observe('custom.activation',[2.,3.])
                writer.seal(state(number),[.01],{'native_backward':True},
                    {'prepare_calls':1,'optimizer_step_calls':1,'optimizer_state_steps':[number]}, {})
                relative=f'steps/{number:06d}/rank{rank}.json';receipts[relative]=digest(folder/relative)
            resumed=None
            if stage=='resume':
                ckpt=folder/f'checkpoint.rank{rank}.pt';ckpt.write_bytes(b'owned-synthetic-checkpoint')
                record={**identity(plan,stage,rank),'cursor':cursor_for(plan,1),'state':state(1),'producer_pid':100+rank,
                    'producer_process_id':'prefix'+str(rank),'sha256':digest(ckpt),'bytes':ckpt.stat().st_size}
                recpath=folder/f'checkpoint.rank{rank}.json';save(recpath,record)
                resumed={'checkpoint_record_sha256':digest(recpath),'checkpoint_sha256':digest(ckpt),'restored_state':state(1),
                    'producer_pid':100+rank,'consumer_pid':200+rank,'producer_process_id':'prefix'+str(rank),'consumer_process_id':'suffix'+str(rank)}
            initial={name:{'shape':spec['parameters'][name]['shape'],'dtype':'torch.float32','sha256':hashlib.sha256(v.tobytes()).hexdigest()}
                     for name,v in weights.items()} if rank==0 else {}
            if cross_storage:
                for name, value in initial.items():
                    value['storage_alignment'] = {'torch.float32': value['sha256'],
                        'torch.bfloat16': hashlib.sha256((weights[name].view('<u4') >> 16).astype('<u2').tobytes()).hexdigest()}
                    value['dtype'] = 'torch.' + stage_precision(plan, stage)['parameters']
                    value['sha256'] = value['storage_alignment'][value['dtype']]
            row={**identity(plan,stage,rank),'execution':{'optimizer_steps':3,'teardown_completed':True},
                'precision':stage_precision(plan, stage),'environment':plan['environment'],'device':{'kind':'cpu'},
                'runtime':{'test':'synthetic'},'initial_parameters':initial,'initial_state':state(0),'resume':resumed,
                'native_parallelism':groups[rank]['native_parallelism'],'workers_sha256':digest(folder/'workers.json'),'step_receipts':receipts}
            save(folder/f'rank{rank}.json',row);finals[f'rank{rank}.json']=digest(folder/f'rank{rank}.json')
        save(folder/'process.json',{'returncode':0})
        save(folder/'complete.json',{'status':'stage_complete','stage':stage,'plan_sha256':plan['plan_sha256'],
            'ranks':finals,'process_sha256':digest(folder/'process.json')})
    return path


class NativeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=fixture(Path(self.temp.name))

    def mutate(self,stage,step,callback):
        folder=self.path.parent/stage;path=folder/'steps'/f'{step:06d}'/'rank0.json'
        row=read(path);callback(row,path.parent);save(path,row)
        final=read(folder/'rank0.json');final['step_receipts'][str(path.relative_to(folder))]=digest(path)
        save(folder/'rank0.json',final)
        complete=read(folder/'complete.json');complete['ranks']['rank0.json']=digest(folder/'rank0.json');save(folder/'complete.json',complete)

    def test_complete_different_rank_layouts_pass(self):
        report=compare_suite(self.path)
        self.assertEqual(report['status'],'passed')
        self.assertEqual(report['coverage']['fields'],list(FIELDS))

    def test_explicit_storage_alignment_accepts_different_byte_encodings(self):
        path = fixture(Path(self.temp.name)/'cross', cross_storage=True)
        report = compare_suite(path)
        self.assertEqual(report['status'], 'passed')
        plan = load_plan(path)
        self.assertEqual(stage_precision(plan, 'repeat')['parameters'], 'float32')
        self.assertEqual(stage_precision(plan, 'resume')['parameters'], 'bfloat16')

    def test_cross_storage_rejects_changed_initial_master_hash(self):
        path = fixture(Path(self.temp.name)/'cross', cross_storage=True)
        folder = path.parent/'candidate'
        row = read(folder/'rank0.json')
        next(iter(row['initial_parameters'].values()))['storage_alignment']['torch.float32'] = 'f'*64
        save(folder/'rank0.json', row)
        final = read(folder/'complete.json');final['ranks']['rank0.json'] = digest(folder/'rank0.json')
        save(folder/'complete.json', final)
        with self.assertRaisesRegex(ValidationError, 'Initial FP32 master values'):
            compare_suite(path)

    def test_cross_storage_requires_explicit_alignment_and_fp32_states(self):
        path = fixture(Path(self.temp.name)/'cross', cross_storage=True)
        spec = read(path.parent/'spec.json')
        for mutate in (
            lambda c: c.pop('precision_comparison'),
            lambda c: c['precision_by_stage']['candidate'].update(optimizer_state='bfloat16'),
            lambda c: c.update(precision={'parameters':'float32'})):
            bad = copy.deepcopy(spec);mutate(bad['contract'])
            with self.assertRaises(ValidationError):validate_spec(bad)

    def test_only_late_momentum_error_is_rejected(self):
        def corrupt(row,folder):
            piece=next(p for p in row['storage']['pieces'] if p['field']=='exp_avg' and p['name']=='scalar_gain')
            p=folder/piece['file'];values=np.fromfile(p,dtype='<f4');values[piece['offset_bytes']//4]*=-1;values.tofile(p)
            row['storage']['files'][p.name]['sha256']=digest(p)
        self.mutate('candidate',3,corrupt)
        report=compare_suite(self.path)
        self.assertFalse(report['gates']['step/3/parameters/exp_avg/scalar_gain/rule0']['passed'])
        self.assertTrue(report['gates']['step/3/gradients']['passed'])
        self.assertEqual(report['status'],'numerical_failure')

    def test_late_forward_mismatch_cannot_hide(self):
        self.mutate('candidate',3,lambda r,p:r['observations']['loss'].update(values=[1e6]))
        report=compare_suite(self.path)
        self.assertFalse(report['gates']['step/3/observation/loss']['passed'])

    def test_resume_rng_difference_is_rejected(self):
        self.mutate('resume',3,lambda r,p:r['state']['rng'].update(value=99))
        self.assertFalse(compare_suite(self.path)['gates']['step/3/resume/state']['passed'])

    def test_missing_rank_state_and_wrong_cursor_are_incomplete(self):
        self.mutate('candidate',2,lambda r,p:r['state']['data_cursor'].update(next_step=77))
        with self.assertRaisesRegex(ValidationError,'cursor'):compare_suite(self.path)

    def test_checkpoint_bytes_are_checked(self):
        (self.path.parent/'resume/checkpoint.rank0.pt').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValidationError,'checksum'):compare_suite(self.path)

    def test_missing_state_policy_and_different_storage_are_rejected(self):
        original=read(self.path.parent/'spec.json')
        for defect in ('master_gate','state_coverage','mixed_storage','single_step'):
            spec=copy.deepcopy(original)
            if defect=='master_gate':del spec['trajectory']['state_acceptance']['parameters'][0]['master_parameters']
            elif defect=='state_coverage':spec['trajectory']['state_acceptance']['parameters'][0]['pattern']='scalar_gain'
            elif defect=='mixed_storage':spec['contract']['precision']['gradients']='bfloat16'
            else:spec['contract']['optimizer_steps']=1
            with self.subTest(defect=defect),self.assertRaises(ValidationError):validate_spec(spec)


if __name__=='__main__':unittest.main()
