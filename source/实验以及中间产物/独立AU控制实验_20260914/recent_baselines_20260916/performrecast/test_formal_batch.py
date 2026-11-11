"""Synthetic filesystem/control tests only; never imports or invokes face models."""
import importlib.util
import json
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('perform_formal_under_test',HERE/'formal_batch.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class Controls(unittest.TestCase):
    def setUp(self):
        parent=HERE/'tests_tmp';parent.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(prefix='synthetic_',dir=parent)
        self.root=Path(self.tmp.name)

    def tearDown(self):self.tmp.cleanup()

    def terminal(self,case='fixture_1'):
        attempt=self.root/case/'attempt_001';path=attempt/'native_crop_png/frame0000.png';path.parent.mkdir(parents=True)
        path.write_bytes(b'SYNTHETIC_TEST_ONLY_NOT_A_PNG')
        record={'case_id':case,'subject':'fixture_subject','group':'nonzero_target','attempt':str(attempt),'protocol_sha256':'fixture_protocol',
                'status':'complete','output':str(path),'output_sha256':mod.sha(path),'outputs':[{'path':str(path),'sha256':mod.sha(path)}]}
        return record,path

    def test_new_json_readback(self):
        p=self.root/'new.json';mod.new_json(p,{'x':1});self.assertEqual(mod.read(p),{'x':1})

    def test_new_json_refuses_overwrite(self):
        p=self.root/'new.json';mod.new_json(p,{'x':1})
        with self.assertRaises(FileExistsError):mod.new_json(p,{'x':2})
        self.assertEqual(mod.read(p),{'x':1})

    def test_complete_binding(self):
        r,_=self.terminal();mod.validate_complete(r,'fixture_protocol',r)

    def test_changed_png_is_not_regenerated(self):
        r,p=self.terminal();p.write_bytes(b'CHANGED_SYNTHETIC')
        with self.assertRaises(RuntimeError):mod.validate_complete(r,'fixture_protocol',r)

    def test_case_swapping_rejected(self):
        r,_=self.terminal()
        with self.assertRaises(RuntimeError):mod.validate_complete(r,'fixture_protocol',dict(r,case_id='different'))

    def test_selected_frame_cannot_change(self):
        r,_=self.terminal();r['output']=str(Path(r['attempt'])/'native_crop_png/frame0001.png')
        with self.assertRaises(RuntimeError):mod.validate_complete(r,'fixture_protocol',r)

    def test_failed_case_remains_terminal(self):
        mod.validate_complete({'status':'failed','protocol_sha256':'fixture_protocol'},'fixture_protocol')

    def test_interrupted_is_not_terminal(self):
        with self.assertRaises(RuntimeError):mod.validate_complete({'status':'resource_interrupted','protocol_sha256':'fixture_protocol'},'fixture_protocol')

    def test_compute_gate_precedes_loading(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'COMPUTE_GO'):mod.run(self.root,False)

    def test_disk_reserves_upcoming_bytes(self):
        with patch.object(mod.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(100,99,mod.RESERVE+31)):
            with self.assertRaisesRegex(RuntimeError,'RESOURCE_PAUSE'):mod.ensure_free(32)

    def test_disk_exact_bound_passes(self):
        with patch.object(mod.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(100,99,mod.RESERVE+32)):
            self.assertEqual(mod.ensure_free(32),mod.RESERVE+32)

    def test_png_header_reads_without_decoder(self):
        p=self.root/'fixture.png';p.write_bytes(b'\x89PNG\r\n\x1a\n'+struct.pack('>I',13)+b'IHDR'+struct.pack('>II',1024,768))
        self.assertEqual(mod.png_size(p),[1024,768])

    def test_invalid_png_rejected(self):
        p=self.root/'fixture.bin';p.write_bytes(b'not_png')
        with self.assertRaises(RuntimeError):mod.png_size(p)

    def test_attempt_never_overwrites_interruption(self):
        folder=self.root/'case';folder.mkdir();(folder/'attempt_003').mkdir()
        self.assertEqual(mod.new_attempt(folder).name,'attempt_004')
        self.assertTrue((folder/'attempt_003').exists())

    def test_all_completed_resume_loads_no_model_and_exports32(self):
        run_dir=self.root/'run';run_dir.mkdir()
        cases=[]
        for i in range(32):
            case_id='SYNTHETIC_fixture_'+str(i)
            case={'case_id':case_id,'subject':'fixture_subject','group':'nonzero_target',
                  'source_absolute_path':'SYNTHETIC_source','source_sha256':'SYNTHETIC',
                  'target_absolute_path':'SYNTHETIC_target','target_sha256':'SYNTHETIC',
                  'source_au_vector_0_5':[0]*12,'target_au_vector_0_5':[1]*12}
            cases.append(case)
        eval_path=self.root/'SYNTHETIC_EVALUATION.json';eval_path.write_text('{"synthetic_only":true}')
        protocol={'sources':{'fixture':True},'python':{'fixture':True},'cases':cases,'case_audit':{'fixture':True},'weights':{'fixture':True},'evaluation_protocol_sha256':mod.sha(eval_path),'input_hot_layout_sha256':mod.sha(eval_path)}
        mod.new_json(run_dir/'frozen_protocol.json',protocol);digest=mod.sha(run_dir/'frozen_protocol.json')
        for c in cases:
            folder=run_dir/'cases'/c['case_id'];attempt=folder/'attempt_001';png=attempt/'native_crop_png/frame0000.png';png.parent.mkdir(parents=True)
            png.write_bytes(b'SYNTHETIC_TEST_ONLY')
            record={**{k:c[k] for k in ('case_id','subject','group')},'protocol_sha256':digest,'status':'complete','attempt':str(attempt),
                    'output':str(png),'output_sha256':mod.sha(png),'outputs':[{'path':str(png),'sha256':mod.sha(png)}]}
            mod.new_json(folder/'terminal.json',record)
        with patch.dict(os.environ,{'PERFORMRECAST_COMPUTE_GO':'1'}),patch.object(mod,'source_binding',return_value={'fixture':True}),\
             patch.object(mod,'python_binding',return_value={'fixture':True}),patch.object(mod,'cases_binding',return_value=(cases,{'fixture':True})),\
             patch.object(mod,'weights_binding',return_value={'fixture':True}),patch.object(mod,'EVALUATION_PROTOCOL',eval_path),patch.object(mod,'HOT_INPUTS',eval_path),patch.dict('sys.modules',{'torch':None}):
            mod.run(run_dir,True)
        index=mod.read(run_dir/'generation_index.json')
        self.assertEqual(len(index['rows']),32);self.assertEqual(index['seed_aliases'],0)
        self.assertEqual(index['successful_cases'],32)


if __name__=='__main__':unittest.main(verbosity=2)
