"""Synthetic gates only. No future image, inference, detector or checkpoint load."""
import ast
import copy
import json
from pathlib import Path
import unittest
from step2000_gate import (HERE,ROOT,CASEFILE,OLD_PREVIEW,OLD_AUDIT,GateError,LABELS,document_gate,record_gate,validate,input_metadata,check_input_metadata,INPUT_FIELDS)
from step2000_cpu_eval import derived_identity_source,enrich_new_measurement

def fixture():
    settings=json.loads((OLD_PREVIEW/'settings.json').read_text());cases=json.loads(CASEFILE.read_text());original=copy.deepcopy(settings)
    settings['skip_baseline']=True
    settings['methods']=[{'name':f"source_centered_{m['variant']}_step2000_{m['sha256'][:8]}",'checkpoint_sha256':m['sha256'],
        'effective_config':copy.deepcopy(m['effective_config']),'model_config':copy.deepcopy(m['model_config']),'diagnostic_training':False} for m in json.loads(OLD_AUDIT.read_text())['methods']]
    result={'status':'complete','test_data_used':False,'all_fixed_cases_and_seeds_retained':True,'cases':19,'methods':4,'images':76}
    initial=json.loads((HERE/'dev19_items.json').read_text())['items'];sources={r['case_id']:r for r in initial if r['method']=='source_real'}
    prompts={r['case_id']:r['prompt'] for r in [json.loads(l) for l in (OLD_PREVIEW/'generation_records.jsonl').read_text().splitlines()] if r['method']=='baseline'}
    records=[{'case_id':c['case_id'],'subject':c['subject'],'split':'dev','group':c['group'],'method':m['name'],'seed':101,'status':'complete',
              'source_aus_0_5':c['source_au_vector_0_5'],'target_aus_0_5':c['target_au_vector_0_5'],'prompt':prompts[c['case_id']]} for c in cases['cases'] for m in settings['methods']]
    return result,settings,cases,original,sources,prompts,records

class Gates(unittest.TestCase):
    def measured_source(self):
        rows=json.loads((HERE/'comparison_nine/results.json').read_text())['records']
        row=copy.deepcopy(next(r for r in rows if r['method']=='source_real'))
        row.update(vector=[987.0]*12,raw_rows=[{'old_source_sentinel':True}],confidence=.123456,
                   identity_cosine=.123456,facenet_primary={'old_identity_sentinel':True},some_future_metric='must_not_leak')
        return row
    def test_source_input_strips_all_measurements(self):
        old=self.measured_source();new=input_metadata(old)
        self.assertTrue(set(new)<=INPUT_FIELDS)
        for key in ['vector','raw_rows','confidence','identity_cosine','facenet_primary','selected_au17','some_future_metric']:self.assertNotIn(key,new)
        self.assertEqual(new['source_labels_12'],old['source_labels_12']);self.assertEqual(new['sha256'],old['sha256'])
    def test_generated_input_does_not_overwrite_fresh_openface(self):
        item={**input_metadata(self.measured_source()),'method':'bidirectional_step2000','original_method':'synthetic_model','path':'SYNTHETIC_NOT_A_REAL_IMAGE','sha256':'a'*64}
        check_input_metadata(item)
        fresh={'vector':[1.0]*12,'raw_rows':[{'fresh_generation_sentinel':True}],'confidence':.99,'identity_cosine':.88,'parse_status':'primary_face_finite'}
        fresh.update(item)  # Identical metadata merge order to frozen OpenFace core.
        self.assertEqual(fresh['vector'],[1.0]*12);self.assertEqual(fresh['raw_rows'],[{'fresh_generation_sentinel':True}]);self.assertEqual(fresh['confidence'],.99);self.assertEqual(fresh['identity_cosine'],.88)
    def test_dirty_input_is_rejected_not_silently_measured(self):
        clean=input_metadata(self.measured_source());clean['vector']=[987.0]*12
        with self.assertRaises(GateError):check_input_metadata(clean)
    def test_input_label_lists_are_not_aliases(self):
        old=self.measured_source();new=input_metadata(old);new['source_labels_12'][0]=5
        self.assertNotEqual(new['source_labels_12'],old['source_labels_12'])
    def new_measurement_fixture(self):
        keys=[1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]
        source={'vector':[.5]*12,'source_labels_12':[0]*12,'target_labels_12':[2]*12,'selected_au17':{str(a):.5 for a in keys},'identity_cosine':1.0,'raw_rows':[{'old':True}],'confidence':.01}
        target={'selected_au17':{str(a):2.0 for a in keys}}
        fresh={'vector':[2.0]*12,'source_labels_12':[0]*12,'target_labels_12':[2]*12,'sha256':'a'*64,'selected_au17':{str(a):2.0 for a in keys},
               'raw_rows':[{'fresh':True}],'confidence':.99,'all_candidates':[{'fresh_candidate':True}],'selected_row':0,'parse_status':'primary_face_finite'}
        return fresh,source,target,{'sha256':'a'*64,'identity_cosine':.67},{'sha256':'a'*64,'identity_cosine':.76}
    def test_merge_preserves_new_AU_and_identity(self):
        a,s,t,f,b=self.new_measurement_fixture();before=copy.deepcopy(a);merged=enrich_new_measurement(a,s,t,f,b)
        self.assertEqual(a,before);self.assertEqual(merged['vector'],[2.0]*12);self.assertEqual(merged['raw_rows'],[{'fresh':True}]);self.assertEqual(merged['confidence'],.99)
        self.assertEqual(merged['identity_cosine'],.67);self.assertEqual(merged['buffalo_identity_cosine'],.76);self.assertEqual(merged['target_mae_changed'],0);self.assertEqual(merged['delta_mae_changed'],.5)
    def test_missing_new_AU_or_identity_is_not_filled_from_source(self):
        a,s,t,f,b=self.new_measurement_fixture();a.update(vector=None,selected_au17=None,parse_status='no_face');f['identity_cosine']=None
        merged=enrich_new_measurement(a,s,t,f,b);self.assertIsNone(merged['vector']);self.assertIsNone(merged['target_mae_changed']);self.assertIsNone(merged['identity_cosine'])
    def test_existing_completed_group_inputs_do_not_have_this_pollution(self):
        bad={'vector','raw_rows','selected_au17','confidence','all_candidates','selected_row','parse_status','identity','identity_cosine','facenet_primary','buffalo_auxiliary','target_mae_changed'}
        count=0
        for name in ['dev19_items.json','single_directions/items.json','external_driver/items.json','external_magicface/items.json','external_liveportrait_absolute/items.json']:
            for item in json.loads((HERE/name).read_text())['items']:
                self.assertFalse(set(item)&bad,(name,set(item)&bad));count+=1
        self.assertEqual(count,285)
    def test_synthetic_metadata_is_accepted_not_a_real_run(self):
        r,s,c,o,refs,prompts,records=fixture();document_gate(r,s,c,copy.deepcopy(c),o);self.assertEqual(len(record_gate(records,s,c,refs,prompts)),4)
    def test_incomplete_run(self):
        r,s,c,o,*_=fixture();r['status']='running'
        with self.assertRaises(GateError):document_gate(r,s,c,c,o)
    def test_missing_variant(self):
        r,s,c,o,*_=fixture();s['methods'].pop()
        with self.assertRaises(GateError):document_gate(r,s,c,c,o)
    def test_duplicate_variant(self):
        r,s,c,o,*_=fixture();s['methods'][3]=s['methods'][0]
        with self.assertRaises(GateError):document_gate(r,s,c,c,o)
    def test_wrong_step(self):
        r,s,c,o,*_=fixture();s['methods'][0]['name']=s['methods'][0]['name'].replace('_step2000_','_step200_')
        with self.assertRaises(GateError):document_gate(r,s,c,c,o)
    def test_sampling_change(self):
        r,s,c,o,*_=fixture();s['guidance_scale']=5
        with self.assertRaises(GateError):document_gate(r,s,c,c,o)
    def test_case_label_change(self):
        r,s,c,o,*_=fixture();old=copy.deepcopy(c);c['cases'][0]['target_au_vector_0_5'][0]=1
        with self.assertRaises(GateError):document_gate(r,s,c,old,o)
    def test_test_subject(self):
        r,s,c,o,*_=fixture();c['cases'][0]['subject']='SN003'
        with self.assertRaises(GateError):document_gate(r,s,c,c,o)
    def test_parent_changed(self):
        r,s,c,o,*_=fixture();c['source_manifest_sha256']='0'*64
        with self.assertRaises(GateError):document_gate(r,s,c,c,o)
    def test_diagnostic_model(self):
        r,s,c,o,*_=fixture();s['methods'][0]['diagnostic_training']=True
        with self.assertRaises(GateError):document_gate(r,s,c,c,o)
    def test_image_count(self):
        r,s,c,o,*_=fixture();r['images']=75
        with self.assertRaises(GateError):document_gate(r,s,c,c,o)
    def test_missing_record(self):
        r,s,c,o,refs,prompts,records=fixture();records.pop()
        with self.assertRaises(GateError):record_gate(records,s,c,refs,prompts)
    def test_duplicate_record(self):
        r,s,c,o,refs,prompts,records=fixture();records[-1]=records[0]
        with self.assertRaises(GateError):record_gate(records,s,c,refs,prompts)
    def test_failed_record(self):
        r,s,c,o,refs,prompts,records=fixture();records[0]['status']='failed'
        with self.assertRaises(GateError):record_gate(records,s,c,refs,prompts)
    def test_reordered_records(self):
        r,s,c,o,refs,prompts,records=fixture();records[0],records[1]=records[1],records[0]
        with self.assertRaises(GateError):record_gate(records,s,c,refs,prompts)
    def test_wrong_seed(self):
        r,s,c,o,refs,prompts,records=fixture();records[0]['seed']=102
        with self.assertRaises(GateError):record_gate(records,s,c,refs,prompts)
    def test_wrong_prompt(self):
        r,s,c,o,refs,prompts,records=fixture();records[0]['prompt']='changed'
        with self.assertRaises(GateError):record_gate(records,s,c,refs,prompts)
    def test_old200_run_rejected(self):
        with self.assertRaises(GateError):validate(OLD_PREVIEW)
    def test_missing_run_rejected(self):
        with self.assertRaises(GateError):validate(ROOT/'development_previews/this_is_an_explicit_synthetic_missing_fixture')
    def test_identity_derived_source_cpu_only_no_execution(self):
        for kind in ['facenet','buffalo']:
            core,source=derived_identity_source(kind,'step2000_synthetic_fixture','a'*64);tree=ast.parse(source)
            self.assertIn('step2000_synthetic_fixture',source);self.assertNotIn('len(items)==57',source)
            self.assertTrue(any(isinstance(n,ast.Constant) and n.value==95 for n in ast.walk(tree)))
            self.assertNotIn("device='mps'",source);self.assertIn('CPUExecutionProvider' if kind=='buffalo' else "device='cpu'",source)
    def test_unsafe_group_rejected(self):
        with self.assertRaises(GateError):derived_identity_source('facenet','../../escape','a'*64)

if __name__=='__main__':unittest.main()
