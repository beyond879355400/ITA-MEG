"""Synthetic metadata tests only: no image decode, model or real metrics."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol as p
import evaluate_new as e

PH='1'*64
SH='2'*64
TH='3'*64


def case(cid='c1',subject='SN1'):
    return {'case_id':cid,'subject':subject,'split':p.ROLE,'au_order':p.AUS,
            'source_au_vector_0_5':[0]*12,'target_au_vector_0_5':[1]+[0]*11,
            'source_sha256':SH,'target_sha256':TH,'prompt':'fixed expression','sampling_seeds':p.SEEDS}


def indexes(cases):
    result=[]
    for method in p.NEW_METHODS:
        rows=[]
        for c in cases:
            for s in ([None]if method=='performrecast'else p.SEEDS):
                rows.append({'case_id':c['case_id'],'subject':c['subject'],'split':p.ROLE,'seed':s,
                             'seed_not_applicable':s is None,'source_sha256':SH,'target_sha256':TH,
                             'source_au_vector_0_5':c['source_au_vector_0_5'],'target_au_vector_0_5':c['target_au_vector_0_5'],
                             'target_image_used':method=='performrecast','target_prompt':c['prompt'],
                             'primary_output_role':p.METHOD_SETTINGS[method]['primary_output_role'],
                             'status':'complete','execution_id':f'{method}:{c["case_id"]}:{s}','execution_started':True,
                             'output':{'path':'synthetic.png','sha256':'4'*64}})
        result.append({'kind':'recent_baseline_generation_index_v1','method':method,'status':'complete','protocol_sha256':PH,'case_manifest_sha256':p.PINS['cases'][1],
                       'case_ids':[r['case_id']for r in cases],'input_group':p.METHOD_SETTINGS[method]['input_group'],
                       'sampling_seeds':[]if method=='performrecast'else p.SEEDS,'records':rows,
                       'base_source_status':p.METHOD_SETTINGS['swiftedit']['base_source_status']})
    return result


class ContractTests(unittest.TestCase):
    def test_128_actual_plan(self):
        cases=[case(f'c{i}',f'SN{i//4}')for i in range(32)]
        plan,rows=p.normalize(cases,indexes(cases),PH)
        self.assertEqual(len(plan),128);self.assertEqual(sum(r['method']=='performrecast'for r in plan),32)
        self.assertEqual(set(r['seed']for r in plan if r['method']=='performrecast'),{-1})

    def test_no_legacy_mutation(self):
        cases=[case()];docs=indexes(cases);before=copy.deepcopy(docs);p.normalize(cases,docs,PH);self.assertEqual(before,docs)

    def test_missing_output_rejected_not_dropped(self):
        cases=[case()];docs=indexes(cases);docs[0]['records'].pop()
        with self.assertRaisesRegex(ValueError,'Every128'):p.normalize(cases,docs,PH)

    def test_duplicate_rejected(self):
        cases=[case()];docs=indexes(cases);docs[0]['records'].append(copy.deepcopy(docs[0]['records'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):p.normalize(cases,docs,PH)

    def test_perform_fake_seeds_rejected(self):
        cases=[case()];docs=indexes(cases);docs[1]['records'][0]['seed']=101
        with self.assertRaisesRegex(ValueError,'Deterministic'):p.normalize(cases,docs,PH)

    def test_perform_wrong_output_rejected(self):
        cases=[case()];docs=indexes(cases);docs[1]['records'][0]['primary_output_role']='native_full_frame0000'
        with self.assertRaisesRegex(ValueError,'Selected output'):p.normalize(cases,docs,PH)

    def test_driver_swap_rejected(self):
        cases=[case()];docs=indexes(cases);docs[1]['records'][0]['target_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Driver target'):p.normalize(cases,docs,PH)

    def test_swift_target_pixels_rejected(self):
        cases=[case()];docs=indexes(cases);docs[0]['records'][0]['target_image_used']=True
        with self.assertRaisesRegex(ValueError,'boundary'):p.normalize(cases,docs,PH)

    def test_swift_prompt_changed_rejected(self):
        cases=[case()];docs=indexes(cases);docs[0]['records'][0]['target_prompt']='chosen after result'
        with self.assertRaisesRegex(ValueError,'prompt'):p.normalize(cases,docs,PH)

    def test_community_label_required(self):
        cases=[case()];docs=indexes(cases);docs[0].pop('base_source_status')
        with self.assertRaisesRegex(ValueError,'source label'):p.normalize(cases,docs,PH)

    def test_failed_candidate_quarantined(self):
        cases=[case()];docs=indexes(cases);r=docs[0]['records'][0];r.update(status='failed',failure_reason='synthetic failure')
        plan,rows=p.normalize(cases,docs,PH);r=rows[('swiftedit','SN1','c1',101)]
        self.assertIsNone(r['output']);self.assertIn('failed_artifact',r);self.assertEqual(len(rows),4)

    def test_unstarted_contradiction_rejected(self):
        cases=[case()];docs=indexes(cases);docs[0]['records'][0].update(status='not_started',failure_reason='never started')
        with self.assertRaisesRegex(ValueError,'Unstarted'):p.normalize(cases,docs,PH)

    def test_holm_family_fixed_five(self):
        spec=p.analysis_spec();self.assertEqual(len(spec['comparisons']),5)
        self.assertEqual({r['family']for r in spec['comparisons']},{'all_five_external_primary_C'})

    def test_case_mean_not_pseudo_seed_replication(self):
        meta={'c1':{'case_id':'c1','subject':'SN1','group':'edited'}};rows=[];gen={}
        for method,seeds in [(p.CHOSEN,p.SEEDS),('performrecast',[-1])]:
            for i,s in enumerate(seeds):
                r={'method':method,'subject':'SN1','case_id':'c1','seed':s,'target_mae_changed':2 if s==-1 else i+1}
                rows.append(r);gen[p.key(r)]={'status':'complete'}
        doc,coverage=p.case_rows(rows,gen,meta);chosen=next(r for r in doc['records']if r['method']==p.CHOSEN)
        perform=next(r for r in doc['records']if r['method']=='performrecast')
        self.assertEqual(chosen['case_metric_means']['target_mae_changed'],2)
        self.assertEqual(perform['case_metric_means']['target_mae_changed'],2)
        self.assertEqual(chosen['seed_coverage']['target_mae_changed']['planned_outputs'],3)
        self.assertEqual(perform['seed_coverage']['target_mae_changed']['planned_outputs'],1)
        self.assertTrue(all(r['analysis_only_case_mean']for r in doc['records']))

    def test_failure_never_zero_or_included_mean(self):
        meta={'c1':{'case_id':'c1','subject':'SN1','group':'edited'}}
        rows=[{'method':p.CHOSEN,'subject':'SN1','case_id':'c1','seed':s,'target_mae_changed':v}for s,v in zip(p.SEEDS,[100,2,4])]
        gen={p.key(r):{'status':'failed'if r['seed']==101 else 'complete'}for r in rows}
        doc,_=p.case_rows(rows,gen,meta);r=next(x for x in doc['records']if x['method']==p.CHOSEN)
        self.assertEqual(r['case_metric_means']['target_mae_changed'],3)
        self.assertEqual(r['seed_coverage']['target_mae_changed']['valid_outputs'],2)

    def test_out_of_scope_separate(self):
        meta={'c1':{'case_id':'c1','subject':'SN1','group':'no_edit'}};doc,_=p.case_rows([],{},meta)
        self.assertEqual(doc['records'][0]['metric_reasons']['target_mae_changed__edited'],'out_of_predeclared_scope')

    def test_nonfinite_rejected(self):
        meta={'c1':{'case_id':'c1','subject':'SN1','group':'edited'}};r={'method':p.CHOSEN,'subject':'SN1','case_id':'c1','seed':101,'target_mae_changed':float('nan')}
        with self.assertRaisesRegex(ValueError,'Nonfinite'):p.case_rows([r],{p.key(r):{'status':'complete'}},meta)

    def test_top3_case_score_tie_fixed_order(self):
        meta={f'c{i}':{'group':'edited'}for i in range(4)}
        records=[{'method':m,'subject':'S','case_id':f'c{i}','case_metric_means':{'target_mae_changed':v}}
                 for m in p.NEW_METHODS for i,v in enumerate([1,3,3,2])]
        gallery=p.gallery_plan({'records':records},meta,[])
        self.assertEqual([r['case_id']for r in gallery['top3_by_method']['swiftedit']['top3']],['c1','c2','c3'])
        self.assertTrue(all(r['display_seed']==101 for r in gallery['top3_by_method']['swiftedit']['top3']))

    def test_csv_null_seed_is_blank(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_',dir=p.HERE)as tmp:
            file=Path(tmp)/'fixture.csv';e.write_csv(file,[{'seed':None,'seed_not_applicable':True}],['seed','seed_not_applicable'])
            self.assertIn(',True',file.read_text(encoding='utf-8-sig'))

    def test_output_outside_scope_rejected(self):
        with self.assertRaises(ValueError):e.new_directory(p.HERE.parent/'not-authorized')


if __name__=='__main__':unittest.main(verbosity=2)
