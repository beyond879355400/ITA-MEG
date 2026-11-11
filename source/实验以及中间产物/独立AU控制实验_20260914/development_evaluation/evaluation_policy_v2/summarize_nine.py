#!/usr/bin/env python3
"""Append external dev outputs, retaining input-information groups explicitly."""
import argparse
import itertools
import json
from pathlib import Path
from summarize_v2 import HERE,POLICY_SHA,sha,save,e,compare,extra_errors,other_compare,strata_helper,EXTRA
INTERNAL=['baseline','current_model','condition_reads_image','image_reads_condition','mlp_model']

def main():
    parser=argparse.ArgumentParser();args=parser.parse_args()
    assert sha(HERE/'policy.json')==POLICY_SHA
    oldfile=HERE/'comparison_eight/results.json';old=json.loads(oldfile.read_text());assert old['status']=='complete' and old['policy_sha256']==POLICY_SHA
    rows=old['records'];source={r['case_id']:r for r in rows if r['method']=='source_real'};target={r['case_id']:r for r in rows if r['method']=='target_real'}
    groups=['external_liveportrait_absolute'];inputs=[oldfile];provenance={}
    for group in groups:
        folder=HERE/group;document=json.loads((folder/'items.json').read_text());assert document['policy_sha256']==POLICY_SHA and not document['test_used'];provenance[group]=document
        files=[folder/k/'results.json' for k in ['openface','facenet','buffalo']];au,fn,buff=[json.loads(p.read_text()) for p in files]
        assert all(p['policy_sha256']==POLICY_SHA and not p['test_used'] for p in [au,fn,buff])
        assert au['status']==buff['status']=='complete';fm={(r['case_id'],r['method']):r for r in fn['records']};bm={(r['case_id'],r['method']):r for r in buff['records']}
        expected=38 if group=='external_driver' else 19;assert len(au['records'])==expected and len(fm)==len(bm)==expected+19
        for r in au['records']:
            key=(r['case_id'],r['method']);s=source[r['case_id']];t=target[r['case_id']];skey=(r['case_id'],'source_real')
            assert fm[key]['sha256']==bm[key]['sha256']==r['sha256']==sha(r['path']) and fm[skey]['sha256']==bm[skey]['sha256']==s['sha256']
            assert r['source_labels_12']==s['source_labels_12'] and r['target_labels_12']==s['target_labels_12'] and r['subject']==s['subject']
            r.update(e.au_metrics(r['vector'],s['vector'],r['source_labels_12'],r['target_labels_12']))
            r.update(facenet_primary=fm[key],buffalo_auxiliary=bm[key],identity_cosine=fm[key]['identity_cosine'],buffalo_identity_cosine=bm[key]['identity_cosine'],identity_metric='FaceNet VGGFace2 primary, not buffalo')
            r['extra5'],r['extra5_proxy_mae_to_real_target']=extra_errors(r['selected_au17'],t['selected_au17'],s['selected_au17'])
            for a in EXTRA:r[f'extra_AU{a}_error_to_real_target']=r['extra5'][str(a)]['absolute_error_to_target']
            rows.append(r)
        inputs+=files+[folder/'items.json',folder/'items_seal.json']
    methods=old['methods']+['liveportrait_absolute'];assert len(rows)==19*(len(methods)+2) and len({(r['case_id'],r['method']) for r in rows})==len(rows)
    for r in rows:assert r['policy_sha256']==POLICY_SHA and r['split']=='dev' and sha(r['path'])==r['sha256']
    pairs={a+'_vs_'+b:[a,b] for a,b in itertools.combinations(methods,2)};cohorts={c:[r for r in rows if c=='all' or r['cohort']==c] for c in ['all','original','additional']}
    coverage={m:{scope:{'planned':len(part:=[r for r in rows if r['method']==m and (scope=='all' or (r['group']=='no_edit')==(scope=='no_edit'))]),
                       'AU_valid':sum(r['vector'] is not None for r in part),'FaceNet_valid':sum(r['identity_cosine'] is not None for r in part),'buffalo_valid':sum(r['buffalo_identity_cosine'] is not None for r in part)} for scope in ['all','edited','no_edit']} for m in ['source_real','target_real']+methods}
    extra_keys=['buffalo_identity_cosine','extra5_proxy_mae_to_real_target']+[f'extra_AU{a}_error_to_real_target' for a in EXTRA]
    result={'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'same_training_protocol_sota_ranking':False,'case_count':19,'generated_count':19*len(methods),'methods':methods,'records':rows,'coverage':coverage,
      'input_groups':{'source_plus_text_manual_AU':INTERNAL,'source_plus_manual_AU_native':['magicface'],'source_plus_real_driver':['liveportrait','faceadapter','liveportrait_absolute']},
      'comparison_scope':'same source identity, real target request, manual12AU labels and evaluators; author native preprocessing and pretrained training settings differ; real-driver group receives the real target image',
      'pairwise':{name:{c:compare(part,ms) for c,part in cohorts.items()} for name,ms in pairs.items()},'all_methods_common_valid':{c:compare(part,methods) for c,part in cohorts.items()},
      'au_level_pairwise':{name:strata_helper.stratify(rows,ms) for name,ms in pairs.items()},'au_level_all_methods':strata_helper.stratify(rows,methods),
      'extra_diagnostics':{'pairwise':{name:other_compare(rows,ms,extra_keys) for name,ms in pairs.items()},'all_methods':other_compare(rows,methods,extra_keys)},
      'internal_no_edit_invariance':old['internal_no_edit_invariance'],'prior190_records_preserved':rows[:190]==json.loads(oldfile.read_text())['records'],
      'setting_count':9,'LP_is_one_method_two_settings':True,'single_factor_LP_relative_to_absolute':compare(rows,['liveportrait','liveportrait_absolute']),
      'external_provenance':provenance,'input_files':{str(p.relative_to(HERE)):sha(p) for p in inputs},'script_sha256':sha(__file__)}
    out=HERE/'comparison_nine';out.mkdir(exist_ok=False);save(out/'results.json',result);(out/'summarizer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'methods':methods,'coverage':coverage,'edited':{k:{'n':v['common_cases'],'means':v['subject_equal_means']} for k,v in result['all_methods_common_valid']['all']['edited'].items()}},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
