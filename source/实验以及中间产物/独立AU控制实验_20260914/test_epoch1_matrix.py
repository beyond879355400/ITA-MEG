"""Standard-library unit fixtures: no actual trainer subprocess or model."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import run_epoch1_matrix as matrix


class MatrixTests(unittest.TestCase):
    def setUp(self):
        base=matrix.EXPERIMENT/'development_diagnostics/epoch1_matrix_tests';base.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='fixture_',dir=base)
        self.project=Path(self.temporary.name);self.exp=self.project/'experiment';self.exp.mkdir()
        self.trainer=self.exp/'train_centered_adapter.py';self.trainer.write_text('# inert unit-test source, never executed\n')
        self.trainer_sha=matrix.digest(self.trainer)
        tasks=[]
        for variant in matrix.VARIANTS:
            config={'variant':variant,'max_steps':2000,'diagnostic':False,'batch_size':1,'readout':'slot','patch_scale':'unit_mean','device':'mps'}
            path=self.exp/(variant+'.json');self.write(path,config)
            parent_dir=self.exp/('fixture_parent_'+variant);parent_dir.mkdir()
            parent=parent_dir/'checkpoint_00000200.pt';parent.write_bytes(b'not-a-model-unit-test-fixture')
            parent_sha=matrix.digest(parent)
            self.write(parent.with_suffix('.json'),{'kind':matrix.CHECKPOINT_KIND,'model_version':matrix.MODEL_VERSION,'variant':variant,'step':200,'checkpoint_sha256':parent_sha})
            self.write(parent_dir/'configuration.json',{**config,'max_steps':200,'model_version':matrix.MODEL_VERSION})
            self.write(parent_dir/'result.json',{'status':'complete','global_step':200,'variant':variant})
            (parent_dir/'snapshots').mkdir();(parent_dir/'snapshots/train_centered_adapter.py').write_bytes(self.trainer.read_bytes())
            tasks.append({'variant':variant,'config_path':str(path),'config_sha256':matrix.digest(path),'parent_checkpoint':str(parent),'parent_sha256':parent_sha})
        self.document={'kind':matrix.MATRIX_KIND,'parent_step':200,'target_step':2000,'trainer_sha256':self.trainer_sha,'tasks':tasks}
        self.manifest=self.exp/'matrix.json';self.write(self.manifest,self.document)

    def tearDown(self):self.temporary.cleanup()
    def write(self,path,data):Path(path).write_text(json.dumps(data))
    def prepared(self):return matrix.preflight(self.manifest,self.project,self.exp)

    def test_dry_run_no_process_no_outputs(self):
        before={str(p.relative_to(self.project)) for p in self.project.rglob('*')}
        def forbidden(*args,**kwargs):raise AssertionError('dry-run started a process')
        result=matrix.run_manifest(self.manifest,True,self.project,self.exp,forbidden)
        self.assertEqual(result['status'],'dry_run_passed');self.assertFalse(result['training_started'])
        self.assertEqual(result['total_remaining_steps'],7200)
        self.assertEqual(before,{str(p.relative_to(self.project)) for p in self.project.rglob('*')})

    def test_missing_parent_placeholder_and_wrong_variant_rejected(self):
        doc=copy.deepcopy(self.document);doc['tasks'][1]['parent_checkpoint']=None
        self.write(self.manifest,doc)
        with self.assertRaisesRegex(ValueError,'not filled'):self.prepared()
        doc=copy.deepcopy(self.document);doc['tasks'].reverse();self.write(self.manifest,doc)
        with self.assertRaisesRegex(ValueError,'fixed four'):self.prepared()

    def test_budget_and_sha_change_rejected(self):
        task=self.document['tasks'][0];path=Path(task['config_path']);config=matrix.read_json(path);config['max_steps']=1999;self.write(path,config)
        with self.assertRaisesRegex(ValueError,'SHA changed'):self.prepared()
        task['config_sha256']=matrix.digest(path);self.write(self.manifest,self.document)
        with self.assertRaisesRegex(ValueError,'total-2000'):self.prepared()

    def test_integer_float_json_fingerprint_difference_rejected(self):
        task=self.document['tasks'][0];path=Path(task['config_path'])
        config=matrix.read_json(path);config['au_weight']=1;self.write(path,config)
        task['config_sha256']=matrix.digest(path);self.write(self.manifest,self.document)
        parent_path=Path(task['parent_checkpoint']).parent/'configuration.json'
        parent=matrix.read_json(parent_path);parent['au_weight']=1.0;self.write(parent_path,parent)
        with self.assertRaisesRegex(ValueError,'parent setting: au_weight'):self.prepared()
        config['au_weight']=1.0;self.write(path,config);task['config_sha256']=matrix.digest(path)
        result=matrix.validate_task(task,self.trainer_sha,self.project,self.exp)
        self.assertIs(type(result['config']['au_weight']),float)

    def test_first_process_failure_stops_later_tasks(self):
        calls=[]
        def fail(command,**kwargs):calls.append(command);return SimpleNamespace(returncode=7)
        result=matrix.run_manifest(self.manifest,False,self.project,self.exp,fail)
        self.assertEqual(len(calls),1);self.assertEqual(result['status'],'failed')
        self.assertEqual(result['tasks']['bidirectional'],'failed')
        self.assertTrue(all(result['tasks'][key]=='not_started' for key in matrix.VARIANTS[1:]))
        self.assertIn('code 7',matrix.read_json(Path(result['queue_directory'])/'task_status/bidirectional/finished.json')['error'])

    def test_zero_exit_without_completion_is_failure(self):
        calls=[]
        def no_result(command,**kwargs):calls.append(command);return SimpleNamespace(returncode=0)
        result=matrix.run_manifest(self.manifest,False,self.project,self.exp,no_result)
        self.assertEqual(result['status'],'failed');self.assertEqual(len(calls),1)

    def test_successful_fixed_queue_and_project_cwd(self):
        calls=[]
        def complete(command,**kwargs):
            calls.append((command,kwargs));self.assertEqual(kwargs['cwd'],str(self.project))
            config=matrix.read_json(command[command.index('--config')+1])
            output=Path(command[command.index('--output')+1]);self.assertFalse(output.exists());output.mkdir(parents=True)
            self.write(output/'result.json',{'status':'complete','global_step':2000,'variant':config['variant'],'model_version':matrix.MODEL_VERSION})
            return SimpleNamespace(returncode=0)
        result=matrix.run_manifest(self.manifest,False,self.project,self.exp,complete)
        self.assertEqual(result['status'],'complete');self.assertEqual(len(calls),4)
        self.assertEqual(list(result['tasks']),list(matrix.VARIANTS))
        self.assertTrue(all(value=='complete' for value in result['tasks'].values()))

    def test_prelaunch_rechecks_after_initial_preflight(self):
        prepared=self.prepared();Path(self.document['tasks'][0]['config_path']).write_text('{}')
        calls=[]
        result=matrix.execute_queue(prepared,run_process=lambda *a,**k:calls.append(a))
        self.assertEqual(result['status'],'failed');self.assertEqual(calls,[])

    def test_minute_collision_appends_sequence(self):
        first=matrix.minute_directory(self.exp/'runs','same','202609141234')
        marker=first/'keep';marker.write_text('old')
        second=matrix.minute_directory(self.exp/'runs','same','202609141234')
        self.assertEqual(second.name,'same_202609141234-02');self.assertEqual(marker.read_text(),'old')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MatrixTests))
    if args.report:matrix.write_new_json(args.report,{'tests_run':result.testsRun,'failures':result.failures,'errors':result.errors,'passed':result.wasSuccessful(),
        'trainer_subprocesses_started':0,'model_imports':False,'process_outcomes':'inert injected unit-test callbacks only','executor_sha256':matrix.digest(matrix.EXPERIMENT/'run_epoch1_matrix.py')})
    raise SystemExit(not result.wasSuccessful())
