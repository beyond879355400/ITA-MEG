"""Synthetic metadata tests only; no true test image, model or native inference."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import contracts as c
import native_runtime as n
import run_external_final as entry


def settings(impl):
    result={'implementation':impl,'input_group':c.IMPLEMENTATIONS[impl],
            'seed_execution':{'mode':'stochastic_each_seed'},
            'runtime':{'repo':{'path':'/synthetic/no_repo','commit':c.COMMITS[impl]},'python_executable':'/synthetic/python',
                       'device':'cpu','dtype':'float32','checkpoint':'/synthetic/checkpoint','base_model':'/synthetic/base','vae_model':'/synthetic/vae',
                       'model_layout_seal':{'path':'/synthetic/magicface_runtime/layout_seal.json','sha256':'d'*64}}}
    if impl=='magicface_native':result['native_parameters']=copy.deepcopy(c.MF_PARAMETERS)
    elif impl=='faceadapter_native':result['native_parameters']=copy.deepcopy(c.FA_PARAMETERS)
    else:result.update(native_parameters={'animation_region':'all','flag_relative_motion':False},seed_execution={'mode':'deterministic_alias','actual_seed':101})
    return result


def fixture():
    rows=[]
    for subject in range(8):
        for k in range(4):
            def frame(number):return {'status':'complete','path':f'/synthetic/person{subject}/frame{number}.png','sha256':'a'*64,
                'width':1024,'height':768,'mode':'RGB','label_frame_1based':number,'video_frame_zero_based':number-1,'camera':'right','original_video_sha256':'b'*64}
            rows.append({'case_id':f'synthetic_s{subject}_c{k}','subject':f'SYN{subject}','split':'test','status':'complete','camera':'right',
                         'generation_input':{'source_image':frame(1),'source_au_vector_0_5':[0]*12,'target_au_vector_0_5':[k]+[0]*11},
                         'target_reference_image':frame(k+1),'target_pixels_passed_to_our_generator':False})
    return {'status':'raw_frames_extracted','protocol_sha256':'1'*64,'sampling_seeds':[101,202,303],'cases':rows,
            'case_count_unchanged':True,'replacement_cases_selected':False,'post_extraction_integrity_errors':[]}


def project(impl,raw=None):return c.project('method_under_test',settings(impl),raw or fixture(),'1'*64,'2'*64)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root=c.HERE/'synthetic_tmp';cls.temp_root.mkdir(exist_ok=True)
    def test_magicface_full96_no_target_pixels(self):
        with patch.object(Path,'open',side_effect=AssertionError('No filesystem read in projection')):
            p=project('magicface_native')
        self.assertEqual((len(p['cases']),len(p['logical_records'])),(96,96))
        self.assertTrue(all(r['split']=='test' and r['test_used'] for r in p['cases']))
        self.assertEqual({r['seed'] for r in p['cases']},{101,202,303})
        self.assertTrue(all(set(r).isdisjoint({'target','target_sha256','target_image_path','driver'}) for r in p['cases']))
        self.assertEqual(p['cases'][3]['au_delta_native'],[1]+[0]*11)

    def test_faceadapter_keeps_driver(self):
        p=project('faceadapter_native');self.assertEqual(len(p['cases']),96)
        self.assertTrue(all(r['target_role']=='real_observed_driver' and r['split']=='test' for r in p['cases']))

    def test_lp_physical32_logical96(self):
        p=project('liveportrait_native');self.assertEqual((len(p['cases']),len(p['logical_records'])),(32,96))
        self.assertEqual({r['seed'] for r in p['cases']},{101})
        self.assertEqual(sum(r['alias_of'] is not None for r in p['logical_records']),64)

    def test_missing_case(self):
        r=fixture();r['cases'].pop()
        with self.assertRaises(ValueError):project('magicface_native',r)

    def test_duplicate_case(self):
        r=fixture();r['cases'][-1]=copy.deepcopy(r['cases'][0])
        with self.assertRaises(ValueError):project('magicface_native',r)

    def test_missing_seed(self):
        r=fixture();r['sampling_seeds']=[101,202]
        with self.assertRaises(ValueError):project('magicface_native',r)

    def test_never_relabel_test_as_dev(self):
        r=fixture();r['cases'][0]['split']='dev'
        with self.assertRaises(ValueError):project('magicface_native',r)

    def test_wrong_protocol_sha(self):
        r=fixture();r['protocol_sha256']='9'*64
        with self.assertRaises(ValueError):project('magicface_native',r)

    def test_wrong_frame_index(self):
        r=fixture();r['cases'][0]['generation_input']['source_image']['video_frame_zero_based']=1
        with self.assertRaises(ValueError):project('magicface_native',r)

    def test_invalid_manual_au(self):
        for invalid in (6,float('nan'),True):
            r=fixture();r['cases'][0]['generation_input']['target_au_vector_0_5'][0]=invalid
            with self.assertRaises(ValueError):project('magicface_native',r)

    def test_wrong_input_group(self):
        s=settings('magicface_native');s['input_group']='source_real_driver'
        with self.assertRaises(ValueError):c.project('m',s,fixture(),'1'*64,'2'*64)

    def test_native_steps_cannot_silently_change(self):
        for impl in ('magicface_native','faceadapter_native'):
            s=settings(impl);s['native_parameters']['steps']=1
            with self.assertRaises(ValueError):c.validate_parameters(s)

    def test_lp_relative_setting_must_be_explicit(self):
        s=settings('liveportrait_native');del s['native_parameters']['flag_relative_motion']
        with self.assertRaises(ValueError):c.validate_parameters(s)

    def test_lp_cannot_fake_stochastic_seeds(self):
        s=settings('liveportrait_native');s['seed_execution']={'mode':'stochastic_each_seed'}
        with self.assertRaises(ValueError):c.validate_parameters(s)

    def test_lp_aliases_same_output_and_execution(self):
        p=project('liveportrait_native');receipts={}
        for i,r in enumerate(p['cases']):receipts[r['execution_id']]={'execution_id':r['execution_id'],'status':'complete','generation_started':True,'output':{'path':f'/synthetic/{i}.png','sha256':f'{i:064x}'}}
        idx=c.update_index(p,receipts,'/synthetic/raw.json',terminal=True)
        self.assertEqual((idx['status'],idx['physical_executions'],idx['unique_output_images'],idx['successful_logical_records']),('complete',32,32,96))
        for i in range(0,96,3):
            a,b,d=idx['records'][i:i+3];self.assertEqual(a['output'],b['output']);self.assertEqual(a['output'],d['output']);self.assertEqual(a['execution_id'],d['execution_id'])

    def test_failed_before_generation_zero_actual_calls(self):
        p=project('liveportrait_native');r=p['cases'][0];receipts={r['execution_id']:{'status':'failed','generation_started':False,'output':None,'error':{'type':'MissingSource'}}}
        idx=c.update_index(p,receipts,'/synthetic/raw.json',terminal=True)
        self.assertEqual((idx['physical_executions'],idx['failed_logical_records'],len(idx['records'])),(0,3,96))
        self.assertEqual(idx['status'],'completed_with_failures')
        self.assertTrue(all(not r['execution_started'] for r in idx['records']))

    def test_final_integrity_error_cannot_be_complete(self):
        p=project('liveportrait_native');receipts={}
        for i,r in enumerate(p['cases']):receipts[r['execution_id']]={'execution_id':r['execution_id'],'status':'complete','generation_started':True,'output':{'path':f'/synthetic/{i}.png','sha256':f'{i:064x}'}}
        idx=c.failed_run_index(p,receipts,'/synthetic/raw.json','model file changed at closing check')
        self.assertEqual(idx['status'],'completed_with_failures');self.assertFalse(idx['usable_for_evaluation'])
        self.assertEqual((idx['physical_executions'],idx['successful_logical_records'],idx['failed_logical_records']),(32,0,96))
        self.assertTrue(all(r['output'] is None for r in idx['records']))
        self.assertTrue(all(r['status']=='complete' for r in receipts.values()))

    def test_real_template_refused_before_common_gate_or_pixels(self):
        path=c.HERE.parent/'protocol.template.json'
        with patch.object(entry,'validate_control',side_effect=AssertionError('Must reject before shared asset gate')):
            with self.assertRaisesRegex(ValueError,'not frozen'):entry.bound_control(path,'x','/never_read.png','x','m')

    def test_source_sha_mismatch_before_backend(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            p=Path(tmp)/'synthetic_source.bin';p.write_bytes(b'not an image; metadata hash fixture')
            job={'split':'test','test_used':True,'input_group':'source_manual_au','source':str(p),'source_sha256':'0'*64}
            with self.assertRaises(ValueError):n.verify_job_inputs(job)

    def test_magicface_input_hash_never_reads_driver(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            p=Path(tmp)/'synthetic_source.bin';p.write_bytes(b'fixture')
            job={'split':'test','test_used':True,'input_group':'source_manual_au','source':str(p),'source_sha256':c.sha(p)}
            self.assertEqual(n.verify_job_inputs(job),['source'])

    def test_venv_interpreter_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            base=Path(tmp)/'base_python';base.write_text('synthetic')
            link=Path(tmp)/'venv_python';link.symlink_to(base)
            self.assertEqual(c.executable_path(str(link)),link.absolute())
            self.assertNotEqual(c.executable_path(str(link)),link.resolve())

    def test_worker_unfrozen_rejects_before_raw_read(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            path=Path(tmp)/'protocol.json';path.write_text(json.dumps({'status':'not_frozen'}))
            with self.assertRaisesRegex(ValueError,'not frozen'):entry.worker_binding({'protocol_path':str(path),'raw_manifest_path':'/never/read/test.png','sampling_path':'/never/read/sampling.json'})

    def test_undeclared_or_wrong_asset_hash_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            root=Path(tmp)/'magicface_runtime';root.mkdir();seal=root/'layout_seal.json'
            names=sorted(n.MF_FILES)
            seal.write_text(json.dumps({'models':[{'target':str(root/name),'sha256':'a'*64} for name in names],'auxiliary_sources':[]}))
            s=settings('magicface_native');s['runtime']['model_layout_seal']={'path':str(seal),'sha256':c.sha(seal)}
            with patch.object(n.subprocess,'check_output',side_effect=[c.COMMITS['magicface_native']+'\n','']):
                with self.assertRaisesRegex(ValueError,'not declared'):n.validate_runtime(s,[{'path':str(root/names[0]),'sha256':'b'*64}])

    def test_observer_proxy_preserves_call_and_attribute(self):
        class Dummy:
            scheduler='old'
            def __call__(self,*a,**k):return (a,k)
        dummy=Dummy();events=[];p=n.ObservedPipeline(dummy,lambda:events.append(1));p.scheduler='new'
        self.assertEqual(dummy.scheduler,'new');self.assertEqual(p(3,x=4),((3,),{'x':4}));self.assertEqual(events,[1])

    def test_mf_computation_ast_unchanged_except_declared_adaptations(self):
        derived=ast.parse(n.magicface_kernel_source());fn=next(x for x in derived.body if isinstance(x,ast.FunctionDef) and x.name=='run_native_body')
        class Undo(ast.NodeTransformer):
            def visit_Dict(self,node):
                self.generic_visit(node)
                for i,k in enumerate(node.keys):
                    if isinstance(k,ast.Constant) and k.value=='test_used':self.assert_true(node.values[i].value);node.values[i]=ast.Constant(False)
                return node
            @staticmethod
            def assert_true(value):assert value is True
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='mark_native_started':return None
                return self.generic_visit(node)
            def visit_If(self,node):
                if ast.unparse(node.test)=="args.phase == 'preprocess'" and len(node.body)==1 and isinstance(node.body[0],ast.Expr) and ast.unparse(node.body[0].value.func)=='torch.set_num_interop_threads':return node.body[0]
                return self.generic_visit(node)
        restored=Undo().visit(ast.Module(body=fn.body,type_ignores=[]))
        original=ast.parse((n.OLD/'run_magicface_smoke.py').read_text());main=next(x for x in original.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        start=next(i for i,x in enumerate(main.body) if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and ast.unparse(x.value.func)=='os.environ.update')
        self.assertEqual(ast.dump(restored,include_attributes=False),ast.dump(ast.Module(body=main.body[start:],type_ignores=[]),include_attributes=False))


if __name__=='__main__':unittest.main(verbosity=2)
