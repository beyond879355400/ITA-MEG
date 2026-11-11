#!/usr/bin/env python3
"""Merge one frozen v2 policy's AU, independent FaceNet and auxiliary buffalo."""
import importlib.util
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent))
from evaluate_patch_variant import compare,e
from openface_v2 import POLICY_SHA,sha,save,AUS12,AUS17
spec=importlib.util.spec_from_file_location('prior_stratification_only',HERE.parent/'v4_slot_2k_mlp200_dev19/evaluate.py')
strata_helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(strata_helper)
METHODS=['baseline','current_model','mlp_model'];EXTRA=[7,10,14,23,45]
PAIRS={'baseline_vs_bidirectional':['baseline','current_model'],'baseline_vs_mlp':['baseline','mlp_model'],'bidirectional_vs_mlp':['current_model','mlp_model']}

def extra_errors(output,target,source):
    result={};errors=[]
    for au in EXTRA:
        p=output.get(str(au)) if output is not None else None;t=target.get(str(au)) if target is not None else None;s=source.get(str(au)) if source is not None else None
        error=abs(p-t) if p is not None and t is not None else None
        result[str(au)]={'output':p,'real_target_measured':t,'real_source_measured':s,'absolute_error_to_target':error,
                          'output_minus_source':p-s if p is not None and s is not None else None,'target_minus_source':t-s if t is not None and s is not None else None}
        if error is not None:errors.append(error)
    return result,sum(errors)/5 if len(errors)==5 else None

def other_compare(rows,methods,keys):
    mapping={(r['case_id'],r['method']):r for r in rows};result={}
    for scope in ('all','edited','no_edit'):
        cases=sorted({r['case_id'] for r in rows if r['method']==methods[0] and (scope=='all' or (r['group']=='no_edit')==(scope=='no_edit'))});part={}
        for key in keys:
            common=[cid for cid in cases if all(mapping[(cid,m)].get(key) is not None for m in methods)];subjects=sorted({mapping[(cid,methods[0])]['subject'] for cid in common})
            per_subject={s:{m:e.mean([mapping[(cid,m)][key] for cid in common if mapping[(cid,m)]['subject']==s]) for m in methods} for s in subjects}
            means={m:e.mean([v[m] for v in per_subject.values()]) for m in methods}
            part[key]={'n':len(common),'case_ids':common,'subjects':subjects,'per_subject':per_subject,'subject_equal_means':means}
        result[scope]=part
    return result

def main():
    assert sha(HERE/'policy.json')==POLICY_SHA
    au=json.loads((HERE/'dev19_openface/results.json').read_text());fn=json.loads((HERE/'facenet_dev19/results.json').read_text());buff=json.loads((HERE/'buffalo_dev19/results.json').read_text())
    assert all(x['policy_sha256']==POLICY_SHA and not x['test_used'] for x in (au,fn,buff))
    assert au['status']==buff['status']=='complete' and len(au['records'])==len(fn['records'])==len(buff['records'])==95
    fm={(r['case_id'],r['method']):r for r in fn['records']};bm={(r['case_id'],r['method']):r for r in buff['records']}
    rows=au['records'];source={r['case_id']:r for r in rows if r['method']=='source_real'};target={r['case_id']:r for r in rows if r['method']=='target_real'}
    for r in rows:
        key=(r['case_id'],r['method']);s=source[r['case_id']];t=target[r['case_id']]
        assert fm[key]['sha256']==bm[key]['sha256']==r['sha256'] and sha(r['path'])==r['sha256']
        assert fm[key]['policy_sha256']==bm[key]['policy_sha256']==POLICY_SHA
        r.update(e.au_metrics(r['vector'],s['vector'],r['source_labels_12'],r['target_labels_12']))
        r['facenet_primary']=fm[key];r['buffalo_auxiliary']=bm[key];r['identity_cosine']=fm[key]['identity_cosine'];r['buffalo_identity_cosine']=bm[key]['identity_cosine']
        r['identity_metric']='FaceNet VGGFace2 primary, not buffalo'
        r['extra5'],r['extra5_proxy_mae_to_real_target']=extra_errors(r['selected_au17'],t['selected_au17'],s['selected_au17'])
        for a in EXTRA:r[f'extra_AU{a}_error_to_real_target']=r['extra5'][str(a)]['absolute_error_to_target']
    cohorts={c:[r for r in rows if c=='all' or r['cohort']==c] for c in ('all','original','additional')}
    pairwise={name:{c:compare(part,methods) for c,part in cohorts.items()} for name,methods in PAIRS.items()}
    threeway={c:compare(part,METHODS) for c,part in cohorts.items()}
    extra_keys=['buffalo_identity_cosine','extra5_proxy_mae_to_real_target']+[f'extra_AU{a}_error_to_real_target' for a in EXTRA]
    extra={name:other_compare(rows,methods,extra_keys) for name,methods in PAIRS.items()};extra['threeway']=other_compare(rows,METHODS,extra_keys)
    coverage={m:{scope:{'planned':len(part:=[r for r in rows if r['method']==m and (scope=='all' or (r['group']=='no_edit')==(scope=='no_edit'))]),
                       'AU_valid':sum(r['vector'] is not None for r in part),'FaceNet_valid':sum(r['identity_cosine'] is not None for r in part),
                       'buffalo_valid':sum(r['buffalo_identity_cosine'] is not None for r in part),
                       'OpenFace_multi_candidates':sum((r['face_count'] or 0)>1 for r in part),'FaceNet_multi_candidates':sum(r['facenet_primary']['face_count']>1 for r in part)} for scope in ('all','edited','no_edit')} for m in ['source_real','target_real']+METHODS}
    oldfile=HERE.parent/'v4_slot_2k_mlp200_dev19/run/results.json';old={(r['case_id'],r['method']):r for r in json.loads(oldfile.read_text())['records']}
    common=[r for r in rows if r['vector'] is not None and old[(r['case_id'],r['method'])]['vector'] is not None]
    all_row_au_equal=[]
    for r in rows:
        previous=old[(r['case_id'],r['method'])]['raw_rows'];now=r['raw_rows']
        equal=len(previous)==len(now) and all(all(p[f'AU{a:02d}_r']==n[f'AU{a:02d}_r'] for a in AUS17) for p,n in zip(previous,now))
        all_row_au_equal.append(equal)
    recovered=[{'case_id':r['case_id'],'method':r['method'],'selected_row':r['selected_row'],'candidates':r['all_candidates'],
                'second_to_first_area_ratio':r['second_to_first_area_ratio']} for r in rows if r['vector'] is not None and old[(r['case_id'],r['method'])]['vector'] is None]
    policy_comparison={'old_results_sha256':sha(oldfile),'old_policy':'singleton-only, buffalo identity main available metric','new_policy_sha256':POLICY_SHA,
          'old_valid_AU_rows':sum(r['vector'] is not None for r in old.values()),'new_valid_AU_rows':sum(r['vector'] is not None for r in rows),
          'common_valid_AU_rows':len(common),'common_AU_vectors_exactly_equal':sum(r['vector']==old[(r['case_id'],r['method'])]['vector'] for r in common),
          'all17_AU_corresponding_rows_unchanged_with_2Dfp':sum(all_row_au_equal),'raw_images_checked':len(all_row_au_equal),'recovered_rows':recovered,
          'identity_comparison_warning':'FaceNet and buffalo values are different spaces; do not interpret encoder role change as model improvement'}
    noedit={}
    for cid in source:
        group=[r for r in rows if r['case_id']==cid and r['method'] in METHODS]
        if group[0]['group']=='no_edit':noedit[cid]={'all_three_PNG_sha_equal':len({r['sha256'] for r in group})==1,
             'FaceNet_values_equal':len({r['identity_cosine'] for r in group})==1,'AU_vectors_equal':len({tuple(r['vector']) if r['vector'] is not None else None for r in group})==1}
    output=HERE/'comparison';output.mkdir(exist_ok=False)
    report={'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'sota_conclusion':False,'case_count':19,'generated_count':57,'records':rows,
            'coverage':coverage,'pairwise':pairwise,'threeway_common_valid':threeway,'extra_diagnostics':extra,
            'au_level_pairwise':{name:strata_helper.stratify(rows,methods) for name,methods in PAIRS.items()},
            'au_level_threeway':strata_helper.stratify(rows,METHODS),'policy_comparison':policy_comparison,'no_edit_invariance':noedit,
            'source_files':{str(p.relative_to(HERE)):sha(p) for p in [HERE/'policy.json',HERE/'dev19_openface/results.json',HERE/'facenet_dev19/results.json',HERE/'buffalo_dev19/results.json',HERE/'INPUT_COORDINATE_NOTE.md']},
            'script_sha256':sha(__file__)}
    save(output/'results.json',report);save(output/'policy_comparison.json',policy_comparison)
    (output/'summarizer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'coverage':coverage,'bidir_vs_MLP':{k:{'n':v['common_cases'],'means':v['subject_equal_means'],'delta':v['new_minus_reference']} for k,v in pairwise['bidirectional_vs_mlp']['all']['edited'].items()},
                      'threeway':{k:{'n':v['common_cases'],'means':v['subject_equal_means']} for k,v in threeway['all']['edited'].items()},
                      'raw17_unchanged_count':sum(all_row_au_equal),'recovered_AU':len(recovered)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
