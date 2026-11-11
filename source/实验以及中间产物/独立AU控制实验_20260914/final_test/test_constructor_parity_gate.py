"""CPU fixtures only. Full constructor/GPU execution remains dormant."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import check_constructor_dev_parity as check
from generate_ours import validate_methods


class ConstructorParityGateTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(20260914)
        ref={'path':str(HERE/'not_loaded_fixture.pt'),'sha256':'a'*64}
        self.control=check.development_control(ref,ref,ref,ref,list(check.TEST_SUBJECTS))
        self.anchor=torch.zeros(768);self.anchor_receipt={'binding':{'fixture':True}}
        source={'image_patches':torch.randn(49,512),'source_identity':torch.randn(512),'base_faceid':torch.randn(512)}
        self.samples={};self.selected=[]
        for i in range(2):
            cid='dev_fixture_'+str(i);aus=torch.tensor([float(i)]+[0.]*11)
            self.samples[cid]={'case_id':cid,'subject':'SYN_DEV','split':'dev',**source,'source_aus':torch.zeros(12),
                'target_aus':aus,'text_pooled':torch.zeros(768),'prompt':'synthetic','latent_mean':torch.zeros(4,64,64)}
            self.selected.append({'case_id':cid,'subject':'SYN_DEV','split':'dev','source_au_vector_0_5':[0.]*12,
                                  'target_au_vector_0_5':aus.tolist()})

    def test_fixture_never_claims_frozen_final_test(self):
        self.assertEqual(self.control['data_role'],'development_only')
        self.assertNotIn('frozen_final_test',json.dumps(self.control));self.assertNotIn('extraction_authorized',json.dumps(self.control))
        self.assertEqual(set(validate_methods(self.control,['baseline','v4'])),{'baseline','v4'})
        with self.assertRaisesRegex(ValueError,'exclusion list'):check.development_control({}, {}, {}, {}, list(check.TEST_SUBJECTS)[:-1])

    def test_source_whitelist_discards_all_target_latents_and_pixels(self):
        cache,features,cases=check.development_inputs(self.samples,self.selected,self.anchor,self.anchor_receipt)
        self.assertEqual(len(cases),2);self.assertEqual(cache['kind'],'development_only_source_fixture')
        self.assertEqual(set(features),{'source_anchor','sources','conditions'})
        self.assertEqual(set(features['conditions']['dev_fixture_0']),{'text_pooled','source_aus','target_aus','prompt'})
        self.assertNotIn('latent_mean',str(features.keys()));self.assertNotIn('target_reference_image',str(features.keys()))

    def test_test_subject_or_role_rejected(self):
        for kind in ('row_role','case_role','subject'):
            samples=copy.deepcopy(self.samples);selected=copy.deepcopy(self.selected)
            if kind=='row_role':samples['dev_fixture_0']['split']='test'
            elif kind=='case_role':selected[0]['split']='test'
            else:samples['dev_fixture_0']['subject']=selected[0]['subject']=check.TEST_SUBJECTS[0]
            with self.assertRaisesRegex(ValueError,'never include test'):check.development_inputs(samples,selected,self.anchor,self.anchor_receipt)

    def test_labels_duplicate_cases_and_shared_source_changes_rejected(self):
        changed=copy.deepcopy(self.selected);changed[0]['target_au_vector_0_5'][0]=2
        with self.assertRaisesRegex(ValueError,'labels changed'):check.development_inputs(self.samples,changed,self.anchor,self.anchor_receipt)
        with self.assertRaisesRegex(ValueError,'two distinct'):check.development_inputs(self.samples,self.selected[:1],self.anchor,self.anchor_receipt)
        changed=copy.deepcopy(self.samples);changed['dev_fixture_1']['base_faceid']=changed['dev_fixture_1']['base_faceid']+1
        with self.assertRaisesRegex(ValueError,'different source'):check.development_inputs(changed,self.selected,self.anchor,self.anchor_receipt)

    def test_default_cli_does_not_execute_constructor_or_gpu(self):
        result=subprocess.run([sys.executable,str(Path(check.__file__))],text=True,capture_output=True,check=True)
        output=json.loads(result.stdout)
        self.assertEqual(output['status'],'not_executed');self.assertFalse(output['test_pixels_read']);self.assertFalse(output['final_protocol_created'])


if __name__=='__main__':
    import argparse
    from prepare_source_cache import save_json,digest
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path);args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ConstructorParityGateTests))
    if args.report:save_json(args.report,{'tests_run':result.testsRun,'passed':result.wasSuccessful(),
        'errors':[(str(t),s) for t,s in result.errors],'failures':[(str(t),s) for t,s in result.failures],
        'models_loaded':False,'mps_used':False,'test_pixels_read':False,'checker_sha256':digest(check.__file__),
        'test_sha256':digest(__file__)})
    raise SystemExit(not result.wasSuccessful())
