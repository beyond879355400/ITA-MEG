"""Pure synthetic controls/metrics only: no real test pixels or evaluator calls."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from evaluation_contract import *
from evaluation_measurements import derive_identity
from evaluation_export import merge_measurements

def fixture():
    methods=['toy_stochastic','toy_deterministic'];ids=[f'synthetic_{s}_{j}' for s in TEST_SUBJECTS for j in range(4)]
    protocol={'kind':'final_test_extraction_protocol','status':'frozen_final_test','extraction_authorized':True,'models':[{'method':m} for m in methods],
              'case_ids':ids,'sampling_seeds':list(SEEDS),'case_seed_policy':'full_cartesian'}
    sampling={'status':'frozen_for_final_test','sampling_seeds':list(SEEDS),'our_generator_uses_target_pixels':False,'method_settings':{
              methods[0]:{'input_group':'source_text_au','seed_execution':{'mode':'stochastic_each_seed'}},methods[1]:{'input_group':'source_real_driver','seed_execution':{'mode':'deterministic_alias','actual_seed':101}}}}
    evaluation={'status':'frozen_for_final_test','methods':methods,'measurement_policy':{'sha256':POLICY_SHA},'reference_image_space':'source_cache_canonical512_and_independent_target_canonical512',
      'metrics':{'toy_C':{'source_metric':'target_mae_changed','scope':'edited','direction':'lower'},'toy_ID0':{'source_metric':'identity_cosine','scope':'no_edit','direction':'higher'}},
      'comparisons':[{'id':'toy_C_comparison','method':methods[0],'reference':methods[1],'metric':'toy_C','family':'toy_predeclared'},
                     {'id':'toy_ID_comparison','method':methods[0],'reference':methods[1],'metric':'toy_ID0','family':'toy_other'}],
      'primary_family':'toy_predeclared','statistics':{'bootstrap_samples':50000,'bootstrap_seed':17,'confidence_level':.95}}
    raw={'cases':[]}
    for n,cid in enumerate(ids):
        target=[0]*12;target[0]=n%4
        raw['cases'].append({'case_id':cid,'subject':TEST_SUBJECTS[n//4],'split':'test','target_pixels_passed_to_our_generator':False,
         'generation_input':{'source_au_vector_0_5':[0]*12,'target_au_vector_0_5':target}})
    cases=case_metadata(protocol,raw);records=[]
    for method in methods:
        for cid in ids:
            for seed in SEEDS:
                actual=seed if method==methods[0] else 101;identity=f'{method}|{cid}|{actual}';digest=hashlib.sha256(identity.encode()).hexdigest()
                records.append({'method':method,'subject':cases[cid]['subject'],'case_id':cid,'seed':seed,'split':'test','status':'complete',
                 'output':{'path':'SYNTHETIC_ONLY_NOT_AN_IMAGE/'+digest+'.png','sha256':digest},'execution_id':identity,'execution_started':True,
                 'alias_of':None if seed==actual else {'method':method,'case_id':cid,'seed':101}})
    index={'status':'complete','protocol_sha256':'a'*64,'raw_manifest':{'sha256':'b'*64},'method_names':methods,'case_ids':ids,'sampling_seeds':list(SEEDS),
           'records':records,'physical_executions':128,'unique_output_images':128}
    return protocol,sampling,evaluation,raw,cases,index

class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.pixel_guard=patch('PIL.Image.open',side_effect=AssertionError('Synthetic suite must not open any image'))
        self.process_guard=patch('subprocess.run',side_effect=AssertionError('Synthetic suite must not launch an evaluator'))
        self.pixel_guard.start();self.process_guard.start()
    def tearDown(self):
        self.process_guard.stop();self.pixel_guard.stop()
    def normalize(self,index=None):
        p,s,e,r,c,i=fixture();return p,s,e,c,normalize_outputs(p,s,c,[i if index is None else index],'a'*64,'b'*64)
    def test_current_unfrozen_template_rejected_before_pixels(self):
        template=HERE/'protocol.template.json'
        with patch('evaluation_contract.ref_path',side_effect=AssertionError('must not read refs')),patch('evaluation_contract.case_metadata',side_effect=AssertionError('must not read case assets')):
            with self.assertRaises(EvaluationGateError):load_control(template,Path('NOT_A_REAL_RAW_MANIFEST'),'0'*64,'final_evaluation_measurements',__file__)
    def test_missing_protocol_rejected(self):
        with self.assertRaises(EvaluationGateError):load_control(HERE/'SYNTHETIC_MISSING_PROTOCOL','x','0'*64,'x',__file__)
    def test_no_metric_or_primary_defaults(self):
        p,s,e,_,_,_=fixture();e.pop('primary_family')
        with self.assertRaises(EvaluationGateError):analysis_contract(p,s,e)
    def test_metric_scope_required(self):
        p,s,e,_,_,_=fixture();e['metrics']['toy_C'].pop('scope')
        with self.assertRaises(EvaluationGateError):analysis_contract(p,s,e)
    def test_wrong_metric_direction(self):
        p,s,e,_,_,_=fixture();e['metrics']['toy_C']['direction']='higher'
        with self.assertRaises(EvaluationGateError):analysis_contract(p,s,e)
    def test_reserved_reference_method_rejected(self):
        p,s,e,_,_,_=fixture();p['models'][0]['method']='source_real'
        with self.assertRaises(EvaluationGateError):frozen_header(p)
    def test_target_generator_boundary(self):
        p,s,e,_,_,_=fixture();s['our_generator_uses_target_pixels']=True
        with self.assertRaises(EvaluationGateError):analysis_contract(p,s,e)
    def test_same_frozen_policy_required(self):
        p,s,e,_,_,_=fixture();e['measurement_policy']['sha256']='0'*64
        with self.assertRaises(EvaluationGateError):analysis_contract(p,s,e)
    def test_actual32_24_8_counts(self):
        p,s,e,raw,cases,index=fixture();analysis_contract(p,s,e)
        self.assertEqual(len(cases),32);self.assertEqual(sum(c['group']=='edited' for c in cases.values()),24);self.assertEqual(sum(c['group']=='no_edit' for c in cases.values()),8)
    def test_full_cartesian_and_deterministic_accounting(self):
        p,s,e,c,(plan,records,account)=self.normalize();self.assertEqual(len(plan),192);self.assertEqual(len(records),192)
        self.assertEqual(account['toy_deterministic']['executed_physical_calls'],32);self.assertEqual(account['toy_deterministic']['alias_rows'],64)
        self.assertEqual(account['toy_stochastic']['executed_physical_calls'],96)
    def test_missing_row_preserves_plan(self):
        p,s,e,r,c,i=fixture();i['records'].pop(4);i.update(status='completed_with_failures',physical_executions=127,unique_output_images=127)
        plan,records,counts=normalize_outputs(p,s,c,[i],'a'*64,'b'*64);self.assertEqual(len(plan),192);self.assertEqual(len(records),191);self.assertEqual(counts['toy_stochastic']['missing_logical_rows'],1)
    def test_failed_before_generation_not_counted_as_execution(self):
        p,s,e,r,c,i=fixture();i['records'][0].update(status='failed',output=None,execution_started=False);i.update(status='completed_with_failures',physical_executions=127,unique_output_images=127)
        _,_,a=normalize_outputs(p,s,c,[i],'a'*64,'b'*64);self.assertEqual(a['toy_stochastic']['executed_physical_calls'],95)
    def test_alias_changed_image_rejected(self):
        p,s,e,r,c,i=fixture();i['records'][97]['output']['sha256']='0'*64
        with self.assertRaises(EvaluationGateError):normalize_outputs(p,s,c,[i],'a'*64,'b'*64)
    def test_alias_cannot_claim_independent_execution(self):
        p,s,e,r,c,i=fixture();i['records'][97]['alias_of']=None
        with self.assertRaises(EvaluationGateError):normalize_outputs(p,s,c,[i],'a'*64,'b'*64)
    def test_physical_counter_must_match(self):
        p,s,e,r,c,i=fixture();i['physical_executions']=192
        with self.assertRaises(EvaluationGateError):normalize_outputs(p,s,c,[i],'a'*64,'b'*64)
    def test_unplanned_seed_rejected(self):
        p,s,e,r,c,i=fixture();i['records'][0]['seed']=999
        with self.assertRaises(EvaluationGateError):normalize_outputs(p,s,c,[i],'a'*64,'b'*64)
    def test_pending_index_rejected(self):
        p,s,e,r,c,i=fixture();i['status']='running'
        with self.assertRaises(EvaluationGateError):normalize_outputs(p,s,c,[i],'a'*64,'b'*64)
    def test_quarantined_complete_index_rejected(self):
        p,s,e,r,c,i=fixture();i['usable_for_evaluation']=False
        with self.assertRaises(EvaluationGateError):normalize_outputs(p,s,c,[i],'a'*64,'b'*64)
    def test_source_cache_binding_required_only_for_our_group(self):
        p,s,e,r,c,i=fixture();reference={'source_cache_manifest':{'path':'SYNTHETIC_CACHE','sha256':'c'*64}};cache={'features_file':{'path':'SYNTHETIC_FEATURES','sha256':'d'*64}}
        i['source_cache_manifest']=reference['source_cache_manifest'];i['source_features_file']=cache['features_file']
        self.assertEqual(source_cache_index_binding(s,reference,cache,[i]),['toy_stochastic'])
        external={'method_names':['toy_deterministic']};self.assertEqual(source_cache_index_binding(s,reference,cache,[external]),[])
    def test_changed_source_cache_rejected(self):
        p,s,e,r,c,i=fixture();reference={'source_cache_manifest':{'path':'SYNTHETIC_CACHE','sha256':'c'*64}};cache={'features_file':{'path':'SYNTHETIC_FEATURES','sha256':'d'*64}}
        i['source_cache_manifest']={'path':'SYNTHETIC_CACHE','sha256':'e'*64};i['source_features_file']=cache['features_file']
        with self.assertRaises(EvaluationGateError):source_cache_index_binding(s,reference,cache,[i])
    def test_input_whitelist_prevents_old_measurement_pollution(self):
        row={'case_id':'toy','subject':'SN003','split':'test','method':'toy','seed':101,'path':'SYNTHETIC','sha256':'a'*64,'vector':[987]*12,'raw_rows':[{'old':True}],'identity_cosine':1}
        clean=clean_item(row);check_item(clean);self.assertNotIn('vector',clean);self.assertNotIn('identity_cosine',clean)
        with self.assertRaises(EvaluationGateError):check_item(row)
    def test_identity_cores_only_derived_not_executed(self):
        for kind in ('facenet','buffalo'):
            core,source=derive_identity(kind,HERE/'SYNTHETIC_NONEXISTENT_JOB','c'*64,256,32);tree=ast.parse(source)
            self.assertIn('SYNTHETIC_NONEXISTENT_JOB',source);self.assertIn("'test'",source);self.assertNotIn("split'] == 'dev'",source)
            self.assertNotIn("device='mps'",source);self.assertIn('CPUExecutionProvider' if kind=='buffalo' else "device='cpu'",source)
            self.assertTrue(any(isinstance(n,ast.Constant) and n.value==256 for n in ast.walk(tree)))
    def test_export_scope_and_alias_seeds_to_subject_statistics(self):
        p,s,e,c,(plan,normalized,account)=self.normalize();metrics=[]
        for row in plan:
            metrics.append({**c[row['case_id']],**row,'target_mae_changed':float(row['seed']!=101) if row['method']=='toy_stochastic' else 0.,'identity_cosine':.8})
        data=statistics_document(p,e,plan,normalized,metrics,account,{'synthetic_only':True})
        spec=importlib.util.spec_from_file_location('synthetic_only_subject_stats',EXP/'statistics/paired_subject_statistics.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        result=module.analyze(data);a=result['comparisons'][0]
        self.assertEqual(a['n_subjects'],8);self.assertEqual(a['n_cases'],24);self.assertEqual(a['n_seed_pairs'],72);self.assertAlmostEqual(a['mean_difference'],2/3)
        self.assertEqual(result['comparisons'][1]['n_cases'],8);self.assertEqual(data['metadata']['physical_execution_accounting']['toy_deterministic']['executed_physical_calls'],32)
    def test_merge_retains_all_planned_missing_outputs(self):
        p,s,e,raw,c,index=fixture();plan=planned_rows(p,c);doc={'planned':plan,'case_metadata':c}
        rows=merge_measurements(doc,{'records':[]},{'records':[]},{'records':[]})
        self.assertEqual(len(rows),192);self.assertTrue(all(r['target_mae_changed'] is None and r['identity_cosine'] is None for r in rows))

if __name__=='__main__':unittest.main()
