"""Synthetic-only final pipeline gates. No production test pixels or models."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'data_prepare'))
import protocol_gate as gate
import prepare_source_cache as source
import generate_ours as inference
from test_extract_final_frames import FinalExtractionTests
import extract_final_test_frames as extraction


class FakeSourceRuntime:
    assets={'fixture':'no neural model loaded'}
    device='cpu'
    dtype='torch.float32'
    environment={'fixture':True,'real_models_loaded':False}
    def __init__(self,fail=False):self.fail=fail;self.source_calls=0
    def source(self,image):
        import torch
        self.source_calls+=1
        if self.fail:raise RuntimeError('synthetic detection failure')
        return image.resize((512,512)),{'image_patches':torch.zeros(49,512),'source_identity':torch.ones(512)/512**.5,
            'base_faceid':torch.ones(512)/512**.5},{'fixture':'synthetic resize, not real alignment'}
    def text(self,prompt):
        import torch
        value=0 if 'a neutral expression' in prompt else 1
        return torch.full((77,768),float(value)),torch.full((768,),float(value))


class FakeGenerationRuntime:
    runtime_receipt={'fixture':True,'models_loaded':False}
    def __init__(self,fail_seed=None):self.fail_seed=fail_seed;self.received=[]
    def prepare(self,method,case_id,source_id):
        self.received.append((method,case_id,source_id))
        return {'condition_hashes':{'fixture':'0'*64}}
    def sample(self,method,prepared,seed):
        from PIL import Image
        if seed==self.fail_seed:raise RuntimeError('synthetic sampling failure')
        return Image.new('RGB',(512,512),(seed%256,0,0))


class FinalSourceInferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture=FinalExtractionTests('test_complete_sha_and_case_seed_gates');self.fixture.setUp()
        self.root=self.fixture.root;self.contract=self.fixture.contract
        self.protocol=copy.deepcopy(self.fixture.protocol)
        sampling=gate.read_json(self.root/'sampling.json')
        sampling['source_features']={'implementation':'training_compatible_source_features_v1','device':'cpu','dtype':'torch.float32',
                                    'anchor_checkpoint':None,'training_cache_manifest':None,'training_cache_features':None}
        for method in sampling['method_settings']:
            sampling['method_settings'][method]={'implementation':'fineface_faceid_baseline','input_group':'source_text_au',
                'seed_execution':{'mode':'stochastic_each_seed'},'runtime':{'device':'cpu','dtype':'torch.float32'},
                'native_parameters':{'height':512,'width':512,'num_inference_steps':20,'guidance_scale':7.5,'au_scale':1.0,
                                     'ip_scale':.9,'scheduler':'DEISMultistepScheduler','negative_prompt':''}}
        self.fixture.write('sampling.json',sampling)
        self.protocol['inputs']['sampling_settings']['sha256']=gate.digest(self.root/'sampling.json')
        self.protocol['execution_sources']={role:{'path':str(path),'sha256':gate.digest(path)} for role,path in
            [('protocol_gate',Path(gate.__file__)),('source_cache',Path(source.__file__)),('our_inference',Path(inference.__file__))]}
        self.fixture.write('protocol.json',self.protocol)
        result=extraction.execute(self.fixture.protocol_path,contract=self.contract)
        self.raw_path=Path(result['output_directory'])/'raw_frame_manifest.json'
        self.raw_sha=gate.digest(self.raw_path)
        self.control=gate.validate_control(self.fixture.protocol_path,self.raw_path,self.raw_sha,contract=self.contract)
        self.methods=inference.validate_methods(self.control,['fixture_A','fixture_B'])

    def tearDown(self):self.fixture.tearDown()

    def cache(self,fail=False):
        import torch
        anchor=torch.zeros(768);runtime=FakeSourceRuntime(fail)
        receipt={'binding':{'text_pooled_sha256':source.tensor_sha(anchor)}}
        path=self.root/('source_cache_fail' if fail else 'source_cache')
        cache=source.build_source_cache(self.control,path,anchor,receipt,runtime)
        return path,cache,runtime

    def test_missing_draft_protocol_rejected_before_image_or_models(self):
        with patch('PIL.Image.open',side_effect=AssertionError('pixel read forbidden')):
            with self.assertRaisesRegex(ValueError,'absent'):gate.validate_control(self.root/'missing.json',self.raw_path,self.raw_sha,contract=self.contract)
            self.protocol['status']='not_frozen';self.fixture.write('protocol.json',self.protocol)
            with self.assertRaisesRegex(ValueError,'not frozen'):gate.validate_control(self.fixture.protocol_path,self.raw_path,self.raw_sha,contract=self.contract)

    def test_gate_never_opens_source_target_pixels(self):
        with patch('PIL.Image.open',side_effect=AssertionError('pixel read forbidden')):
            result=gate.validate_control(self.fixture.protocol_path,self.raw_path,self.raw_sha,contract=self.contract)
        self.assertFalse(result['pixels_opened_by_gate']);self.assertEqual(len(result['case_ids']),4)

    def test_checkpoint_tamper_full_case_seed_and_unknown_method_rejected(self):
        self.fixture.artifact.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):gate.validate_control(self.fixture.protocol_path,self.raw_path,self.raw_sha,contract=self.contract)
        with self.assertRaisesRegex(ValueError,'Unplanned'):inference.validate_methods(self.control,['not_planned'])
        for key,bad in [('case_ids',self.protocol['case_ids'][:-1]),('sampling_seeds',[101])]:
            changed=copy.deepcopy(self.protocol);changed[key]=bad;self.fixture.write('protocol.json',changed)
            with self.assertRaises(ValueError):gate.validate_control(self.fixture.protocol_path,self.raw_path,self.raw_sha,contract=self.contract)

    def test_source_only_can_complete_with_nonexistent_target_files(self):
        source_paths={row['generation_input']['source_image']['path'] for row in self.control['raw_manifest']['cases']}
        for row in self.control['raw_manifest']['cases']:
            path=row['target_reference_image']['path']
            if path not in source_paths:Path(path).unlink()
        from PIL import Image
        original=Image.open;opened=[]
        def permitted(path,*args,**kwargs):
            opened.append(str(path));self.assertIn(str(path),source_paths)
            return original(path,*args,**kwargs)
        with patch('PIL.Image.open',side_effect=permitted):
            path,cache,runtime=self.cache()
        self.assertEqual(cache['status'],'complete');self.assertEqual(runtime.source_calls,1);self.assertEqual(len(opened),1)
        self.assertFalse(cache['target_pixels_read']);self.assertFalse(cache['target_latents_computed']);self.assertFalse(cache['vae_loaded'])
        loaded,tensors=inference.load_source_cache(self.control,path/'source_cache_manifest.json',gate.digest(path/'source_cache_manifest.json'))
        self.assertEqual(len(tensors['conditions']),4);self.assertNotIn('latent_mean',str(tensors.keys()))

    def test_source_failure_retains_all_cases_and_never_starts_sampling(self):
        path,cache,runtime=self.cache(fail=True)
        self.assertEqual(cache['status'],'completed_with_failures');self.assertEqual(len(cache['cases']),4)
        result=inference.run_grid(self.control,self.methods,cache,self.root/'run_source_failure',FakeGenerationRuntime())
        self.assertEqual(len(result['records']),24);self.assertEqual(result['physical_executions'],0)
        self.assertEqual({r['status'] for r in result['records']},{'not_started'})

    def test_grid_failures_and_unique_images_not_counted_as_subjects(self):
        path,cache,_=self.cache();runtime=FakeGenerationRuntime(fail_seed=202)
        result=inference.run_grid(self.control,self.methods,cache,self.root/'run_grid',runtime)
        self.assertEqual(len(result['records']),24);self.assertEqual(result['physical_executions'],24)
        self.assertEqual(sum(r['status']=='failed' for r in result['records']),8)
        self.assertEqual(result['unique_output_images'],2);self.assertEqual(result['unique_output_files'],16)
        self.assertTrue(all(r['alias_of'] is None for r in result['records']))
        self.assertEqual(set(result['case_ids']),set(self.control['case_ids']))
        self.assertEqual(len(runtime.received),8)
        # Runtime preparation receives IDs, never any target path.
        self.assertTrue(all(len(row)==3 for row in runtime.received))
        with self.assertRaises(FileExistsError):inference.run_grid(self.control,self.methods,cache,self.root/'run_grid',runtime)

    def test_labels_anchor_and_forbidden_target_latent_rejected(self):
        import torch
        path,cache,_=self.cache();feature_path=path/'features.pt'
        features=torch.load(feature_path,map_location='cpu',weights_only=True)
        features['latent_mean']=torch.zeros(4,64,64)
        torch.save(features,feature_path)
        cache['features_file']['sha256']=gate.digest(feature_path)
        changed=path/'forbidden_cache.json';source.save_json(changed,cache)
        with self.assertRaisesRegex(ValueError,'target latents forbidden'):inference.load_source_cache(self.control,changed,gate.digest(changed))
        del features['latent_mean'];features['source_anchor']+=1;torch.save(features,feature_path)
        cache['features_file']['sha256']=gate.digest(feature_path);changed=path/'changed_anchor.json';source.save_json(changed,cache)
        with self.assertRaisesRegex(ValueError,'anchor tensor changed'):inference.load_source_cache(self.control,changed,gate.digest(changed))
        cache['cases'][0]['target_au_vector_0_5'][0]=2;changed=path/'changed_label.json';source.save_json(changed,cache)
        with self.assertRaisesRegex(ValueError,'manual labels'):inference.load_source_cache(self.control,changed,gate.digest(changed),load_tensors=False)

    def test_sampling_defaults_drivers_and_unfrozen_source_role_rejected(self):
        for field,value in [('ip_scale',.7),('scheduler','PNDMScheduler'),('au_scale',2),('guidance_scale',1)]:
            control=copy.deepcopy(self.control);control['models']['fixture_A']['settings']['native_parameters'][field]=value
            with self.assertRaises(ValueError):inference.validate_methods(control,['fixture_A'])
        control=copy.deepcopy(self.control);del control['models']['fixture_A']['settings']['native_parameters']['negative_prompt']
        with self.assertRaisesRegex(ValueError,'all be explicit'):inference.validate_methods(control,['fixture_A'])
        control=copy.deepcopy(self.control);control['models']['fixture_A']['settings']['input_group']='source_real_driver'
        with self.assertRaisesRegex(ValueError,'source\+text\+AU'):inference.validate_methods(control,['fixture_A'])
        with self.assertRaisesRegex(ValueError,'not frozen'):gate.verify_runtime_source(self.control,'missing_role',Path(__file__))

    def test_real_v4_prepare_zeroedit_prefix_and_cfg_order_cpu(self):
        import torch
        from modeling.safe_token_residual_adapter import TokenAdapterConfig,RelativeRMSConfig
        from train_centered_adapter import create_centered_adapter
        torch.manual_seed(20260914)
        cfg=TokenAdapterConfig(readout='slot',patch_scale='unit_mean')
        model=create_centered_adapter('bidirectional',cfg,'cpu').eval()
        with torch.no_grad():
            model.au_output.weight.normal_(0,.01);model.image_output.weight.normal_(0,.01)
        class Backbone:
            def baseline_conditions(self,au,identity):
                return torch.cat((au,torch.ones(len(au),756)),dim=-1)[:,None],identity[:,:1,None].expand(-1,16,768).clone()
        rt=inference.OurRuntime.__new__(inference.OurRuntime)
        rt.device='cpu';rt.backbone=Backbone();rt.settings={'x':{'native_parameters':{'au_scale':1.,'ip_scale':.9}}}
        rt.methods={'x':(model,RelativeRMSConfig(enabled=True,rho_au=.1,rho_ip=.025))}
        rt.features={'source_anchor':torch.randn(768),'sources':{'src':{'image_patches':torch.randn(49,512),'base_faceid':torch.ones(512)}},'conditions':{}}
        rt.features['conditions']['noedit']={'text_pooled':rt.features['source_anchor'].clone(),'source_aus':torch.zeros(12),'target_aus':torch.zeros(12),'prompt':'fixture'}
        rt.features['conditions']['edited']={**rt.features['conditions']['noedit'],'text_pooled':torch.randn(768),'target_aus':torch.tensor([1.]+[0.]*11)}
        noedit=rt.prepare('x','noedit','src')['cross'];edited=rt.prepare('x','edited','src')['cross']
        base_au,base_ip=rt.backbone.baseline_conditions(torch.zeros(1,12),torch.ones(1,512))
        self.assertTrue(torch.equal(noedit['au_embedding'][1:],base_au));self.assertTrue(torch.equal(noedit['image_embedding'][1:],base_ip))
        self.assertTrue(torch.equal(edited['au_embedding'][1,0,:12],rt.features['conditions']['edited']['target_aus']))
        self.assertEqual(float(edited['image_embedding'][0].abs().sum()),0.) # null first
        self.assertGreater(float((edited['image_embedding'][1:]-base_ip).abs().sum()),0.)

    def test_runtime_device_dtype_and_ignored_precision_rejected(self):
        self.assertEqual(source.validate_source_feature_settings(self.control,'cpu')['dtype'],'torch.float32')
        with self.assertRaisesRegex(ValueError,'conflicts'):source.validate_source_feature_settings(self.control,'mps')
        control=copy.deepcopy(self.control);control['sampling']['source_features']['dtype']='torch.float16'
        with self.assertRaisesRegex(ValueError,'FP32'):source.validate_source_feature_settings(control)
        for key,value in [('dtype','torch.float16'),('precision','fp16')]:
            control=copy.deepcopy(self.control);control['models']['fixture_A']['settings']['runtime'][key]=value
            with self.assertRaises(ValueError):inference.validate_methods(control,['fixture_A'])
        path,cache,_=self.cache();cache['dtype']='torch.float16';changed=path/'wrong_dtype.json';source.save_json(changed,cache)
        with self.assertRaisesRegex(ValueError,'device/dtype'):inference.load_source_cache(self.control,changed,gate.digest(changed),load_tensors=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FinalSourceInferenceTests))
    if args.report:source.save_json(args.report,{'tests_run':result.testsRun,'passed':result.wasSuccessful(),
        'failures':[(str(t),s) for t,s in result.failures],'errors':[(str(t),s) for t,s in result.errors],
        'production_test_pixels_read':False,'real_models_called':False,'mps_used':False,
        'sources':{p.name:gate.digest(p) for p in (Path(gate.__file__),Path(source.__file__),Path(inference.__file__),Path(__file__))}})
    raise SystemExit(not result.wasSuccessful())
