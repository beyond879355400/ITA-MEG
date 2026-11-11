"""Same-case subject-equal reporting; never compare unequal valid sample sets."""
import json
from pathlib import Path
from evaluate import mean,digest,save

def main():
    root=Path(__file__).resolve().parent;file=root/'run/results.json';data=json.loads(file.read_text())
    mapping={(r['case_id'],r['method']):r for r in data['records']}
    keys=['target_mae_changed','delta_mae_changed','unchanged_source_relative_drift','direction_correct_fraction','identity_cosine']
    cases=sorted({r['case_id'] for r in data['records'] if r['group']!='no_edit'})
    common=[cid for cid in cases if all(mapping[(cid,m)][k] is not None for m in ('baseline','new_v3') for k in keys)]
    assert len(common)==7
    subjects=sorted({mapping[(cid,'baseline')]['subject'] for cid in common});assert len(subjects)==4
    per_subject={}
    for subject in subjects:
        ids=[cid for cid in common if mapping[(cid,'baseline')]['subject']==subject]
        per_subject[subject]={'paired_case_ids':ids,'metrics':{k:{m:mean([mapping[(cid,m)][k] for cid in ids]) for m in ('source_real','target_real','baseline','new_v3')} for k in keys}}
    summary={k:{m:mean([s['metrics'][k][m] for s in per_subject.values()]) for m in ('source_real','target_real','baseline','new_v3')} for k in keys}
    for k,v in summary.items():v['new_minus_baseline']=v['new_v3']-v['baseline']
    coverage={m:{k:data['by_method'][m]['all'][k] for k in ('planned','au_valid','identity_single_face','identity_pairs_valid')} for m in ('source_real','target_real','baseline','new_v3')}
    report={'results_sha256':digest(file),'primary_scope':'same7 edited cases, then equal mean over4 subjects; all failed/unpaired cases remain in coverage and raw records',
            'paired_case_ids':common,'per_subject':per_subject,'subject_equal_metrics':summary,'coverage_all12':coverage,
            'no_edit_summary':data['paired_difference_summary']['no_edit'],'test_used':False,'sota_conclusion':False}
    save(root/'run/paired_summary.json',report);print(json.dumps({'subject_equal':summary,'coverage':coverage},indent=2))

if __name__=='__main__':main()
