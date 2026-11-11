"""Synthetic fixtures only. No real test metrics, models, or evaluators."""
import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import unittest
import paired_subject_statistics as stats


def document():
    return {'kind':'paired_subject_metric_input','metadata':{'data_role':'synthetic','primary_family_frozen':False},
            'analysis_spec':{'metrics':{'m':{'direction':'higher','unit':'synthetic units'}},
                             'comparisons':[{'id':'synthetic_pair','method':'A','reference':'B','metric':'m','family':'synthetic_family'}]},
            'planned':[],'records':[]}


def add(doc,method,subject,case,seed,value,status='complete',record=True):
    row={'method':method,'subject':subject,'case_id':case,'seed':seed}
    doc['planned'].append(row)
    if record:doc['records'].append({**row,'status':status,'metrics':{'m':value}})


class PairedSubjectTests(unittest.TestCase):
    def test_seed_case_subject_equal_weighting(self):
        doc=document()
        for method,offset in [('A',0),('B',-1)]:
            for seed,value in [(1,0),(2,2)]:add(doc,method,'S1','c1',seed,value+offset)
            for case,value in [('c2',9),('c3',11),('c4',13)]:add(doc,method,'S2',case,1,value+offset)
        out=stats.analyze(doc);pair=out['comparisons'][0]
        # S1 mean1; S2 case mean11; subject mean6, not pooled-output7.
        self.assertEqual(out['raw_method_statistics']['A']['m']['mean'],6)
        self.assertEqual(pair['paired_method_mean'],6);self.assertEqual(pair['mean_difference'],1)
        self.assertEqual(pair['n_subjects'],2);self.assertEqual(pair['n_cases'],4);self.assertEqual(pair['n_seed_pairs'],5)

    def test_common_seed_intersection_not_separate_method_averages(self):
        doc=document()
        for seed,value in [(1,1),(2,100),(3,3)]:add(doc,'A','S1','c1',seed,value)
        add(doc,'B','S1','c1',1,0);add(doc,'B','S1','c1',2,None,record=False);add(doc,'B','S1','c1',3,0)
        for seed,value in [(1,4),(2,6),(3,8)]:add(doc,'A','S2','c2',seed,value);add(doc,'B','S2','c2',seed,0)
        pair=stats.analyze(doc)['comparisons'][0]
        self.assertEqual(pair['paired_method_mean'],4);self.assertEqual(pair['mean_difference'],4)
        self.assertEqual(pair['n_seed_pairs'],5);self.assertEqual(pair['coverage']['planned_common_seed_outputs'],6)
        self.assertEqual(pair['coverage']['unavailable_pairs'][0]['reference_reason'],'missing_output')

    def test_three_seeds_are_not_three_subjects(self):
        doc=document()
        for seed in (101,202,303):add(doc,'A','one','c',seed,1);add(doc,'B','one','c',seed,0)
        pair=stats.analyze(doc)['comparisons'][0]
        self.assertEqual(pair['n_subjects'],1);self.assertEqual(pair['n_seed_pairs'],3)
        self.assertIsNone(pair['difference_ci95']);self.assertIsNone(pair['two_sided_sign_flip_p']);self.assertIsNone(pair['holm_adjusted_p'])
        self.assertEqual(pair['bootstrap_samples'],0)

    def test_known_cluster_bootstrap_and_exact_sign_flip(self):
        result=stats.subject_inference([1,2])
        self.assertEqual(result['mean_difference'],1.5)
        self.assertEqual(result['difference_ci95'],[1,2]);self.assertEqual(result['bootstrap_samples'],50000)
        self.assertEqual(result['sign_patterns'],4);self.assertEqual(result['two_sided_sign_flip_p'],.5)
        values=[.3,-.2,1.1];observed=abs(math.fsum(values)/len(values))
        exact=sum(abs(math.fsum(a*b for a,b in zip(signs,values))/len(values))>=observed-1e-12 for signs in itertools.product((-1,1),repeat=3))/8
        self.assertEqual(stats.subject_inference(values)['two_sided_sign_flip_p'],exact)
        self.assertEqual(stats.subject_inference(values),stats.subject_inference(values))

    def test_zero_differences_and_no_subjects(self):
        result=stats.subject_inference([0,0,0]);self.assertEqual(result['difference_ci95'],[0,0]);self.assertEqual(result['two_sided_sign_flip_p'],1)
        result=stats.subject_inference([]);self.assertIsNone(result['mean_difference']);self.assertIsNone(result['difference_ci95'])

    def test_holm_explicit_families_with_NA_retained(self):
        rows=[{'id':'a','family':'one','two_sided_sign_flip_p':.01},{'id':'b','family':'one','two_sided_sign_flip_p':.04},
              {'id':'c','family':'one','two_sided_sign_flip_p':.03},{'id':'na','family':'one','two_sided_sign_flip_p':None},
              {'id':'d','family':'two','two_sided_sign_flip_p':.01}]
        stats.holm_adjust(rows);out={r['id']:r for r in rows}
        self.assertAlmostEqual(out['a']['holm_adjusted_p'],.04);self.assertAlmostEqual(out['b']['holm_adjusted_p'],.09)
        self.assertAlmostEqual(out['c']['holm_adjusted_p'],.09);self.assertIsNone(out['na']['holm_adjusted_p'])
        self.assertEqual(out['a']['holm_family_size'],4);self.assertEqual(out['d']['holm_adjusted_p'],.01)
        with self.assertRaises(ValueError):stats.holm_adjust([{'id':'x','two_sided_sign_flip_p':.1}])

    def test_NA_failure_nonfinite_missing_retained(self):
        doc=document()
        add(doc,'A','S','c',1,999,status='failed');add(doc,'A','S','c',2,None);add(doc,'A','S','c',3,float('nan'));add(doc,'A','S','c',4,None,record=False)
        for seed in range(1,5):add(doc,'B','S','c',seed,0)
        out=stats.analyze(doc);cov=next(r for r in out['coverage_table'] if r['method']=='A')
        self.assertEqual(cov['planned_seed_outputs'],4);self.assertEqual(cov['observed_output_rows'],3);self.assertEqual(cov['valid_metric_outputs'],0)
        self.assertEqual(cov['unavailable_by_reason'],{'row_status:failed':1,'metric_NA':1,'nonfinite_metric':1,'missing_output':1})
        self.assertEqual(out['comparisons'][0]['n_subjects'],0);json.dumps(out,allow_nan=False)

    def test_different_plans_preserve_noncommon_keys(self):
        doc=document();add(doc,'A','S','common',1,1);add(doc,'B','S','common',1,0);add(doc,'A','S','only_A',1,100)
        out=stats.analyze(doc)['comparisons'][0]
        self.assertEqual(out['paired_method_mean'],1);self.assertEqual(out['coverage']['method_only_planned_keys'],[['S','only_A',1]])

    def test_lower_metric_difference_is_not_silently_flipped(self):
        doc=document();doc['analysis_spec']['metrics']['m']['direction']='lower'
        for subject,a,b in [('S1',1,2),('S2',2,4)]:add(doc,'A',subject,'c',1,a);add(doc,'B',subject,'c',1,b)
        pair=stats.analyze(doc)['comparisons'][0]
        self.assertEqual(pair['mean_difference'],-1.5);self.assertEqual(pair['difference_ci95'],[-2,-1]);self.assertEqual(pair['metric_direction'],'lower')

    def test_duplicate_and_unplanned_outputs_rejected(self):
        doc=document();add(doc,'A','S','c',1,1);add(doc,'B','S','c',1,0)
        duplicate=copy.deepcopy(doc);duplicate['records'].append(copy.deepcopy(duplicate['records'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate metric'):stats.analyze(duplicate)
        unplanned=copy.deepcopy(doc);unplanned['records'][0]['seed']=99
        with self.assertRaisesRegex(ValueError,'not in the explicit plan'):stats.analyze(unplanned)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PairedSubjectTests))
    if args.report:
        import hashlib
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open('x') as handle:json.dump({'tests_run':result.testsRun,'failures':result.failures,'errors':result.errors,'passed':result.wasSuccessful(),
            'input_scope':'pure synthetic fixtures only','real_test_metrics_read':False,'models_or_evaluators_called':False,
            'statistics_source_sha256':hashlib.sha256(Path(stats.__file__).read_bytes()).hexdigest()},handle,indent=2)
    raise SystemExit(not result.wasSuccessful())
