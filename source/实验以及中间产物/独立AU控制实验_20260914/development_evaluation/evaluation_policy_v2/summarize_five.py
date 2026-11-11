#!/usr/bin/env python3
"""Merge all five same-dev19 methods, preserving initial policy-v2 records."""
import itertools
import json
from pathlib import Path
from summarize_v2 import HERE,POLICY_SHA,sha,save,e,compare,extra_errors,other_compare,strata_helper,EXTRA
METHODS=['baseline','current_model','condition_reads_image','image_reads_condition','mlp_model']

def main():
    assert sha(HERE/'policy.json')==POLICY_SHA
    oldfile=HERE/'comparison/results.json';old=json.loads(oldfile.read_text());assert old['status']=='complete' and old['policy_sha256']==POLICY_SHA
    folder=HERE/'single_directions';aufile=folder/'openface/results.json';fnfile=folder/'facenet/results.json';bfile=folder/'buffalo/results.json'
    au=json.loads(aufile.read_text());fn=json.loads(fnfile.read_text());buff=json.loads(bfile.read_text())
    assert all(p['policy_sha256']==POLICY_SHA and not p['test_used'] for p in (au,fn,buff))
    assert au['status']==buff['status']=='complete' and len(au['records'])==38 and len(fn['records'])==len(buff['records'])==57
    rows=old['records'];source={r['case_id']:r for r in rows if r['method']=='source_real'};target={r['case_id']:r for r in rows if r['method']=='target_real'}
    fm={(r['case_id'],r['method']):r for r in fn['records']};bm={(r['case_id'],r['method']):r for r in buff['records']}
    for r in au['records']:
        key=(r['case_id'],r['method']);s=source[r['case_id']];t=target[r['case_id']]
        assert fm[key]['sha256']==bm[key]['sha256']==r['sha256']==sha(r['path'])
        assert r['source_labels_12']==s['source_labels_12'] and r['target_labels_12']==s['target_labels_12'] and r['subject']==s['subject']
        skey=(r['case_id'],'source_real');assert fm[skey]['sha256']==bm[skey]['sha256']==s['sha256']
        r.update(e.au_metrics(r['vector'],s['vector'],r['source_labels_12'],r['target_labels_12']))
        r.update(facenet_primary=fm[key],buffalo_auxiliary=bm[key],identity_cosine=fm[key]['identity_cosine'],buffalo_identity_cosine=bm[key]['identity_cosine'],identity_metric='FaceNet VGGFace2 primary, not buffalo')
        r['extra5'],r['extra5_proxy_mae_to_real_target']=extra_errors(r['selected_au17'],t['selected_au17'],s['selected_au17'])
        for a in EXTRA:r[f'extra_AU{a}_error_to_real_target']=r['extra5'][str(a)]['absolute_error_to_target']
        rows.append(r)
    assert len(rows)==133 and len({(r['case_id'],r['method']) for r in rows})==133
    for r in rows:assert r['policy_sha256']==POLICY_SHA and r['split']=='dev' and sha(r['path'])==r['sha256']
    pairs={a+'_vs_'+b:[a,b] for a,b in itertools.combinations(METHODS,2)}
    cohorts={label:[r for r in rows if label=='all' or r['cohort']==label] for label in ('all','original','additional')}
    coverage={m:{scope:{'planned':len(part:=[r for r in rows if r['method']==m and (scope=='all' or (r['group']=='no_edit')==(scope=='no_edit'))]),
                       'AU_valid':sum(r['vector'] is not None for r in part),'FaceNet_valid':sum(r['identity_cosine'] is not None for r in part),'buffalo_valid':sum(r['buffalo_identity_cosine'] is not None for r in part)} for scope in ('all','edited','no_edit')} for m in ['source_real','target_real']+METHODS}
    extra_keys=['buffalo_identity_cosine','extra5_proxy_mae_to_real_target']+[f'extra_AU{a}_error_to_real_target' for a in EXTRA]
    noedit={cid:{'all_five_PNG_sha_equal':len({r['sha256'] for r in rows if r['case_id']==cid and r['method'] in METHODS})==1} for cid,s in source.items() if s['group']=='no_edit'}
    assert len(noedit)==4 and all(r['all_five_PNG_sha_equal'] for r in noedit.values())
    result={'status':'complete','policy_sha256':POLICY_SHA,'test_used':False,'sota_conclusion':False,'case_count':19,'generated_count':95,'methods':METHODS,'records':rows,'coverage':coverage,
            'pairwise':{name:{label:compare(part,methods) for label,part in cohorts.items()} for name,methods in pairs.items()},
            'all_methods_common_valid':{label:compare(part,METHODS) for label,part in cohorts.items()},
            'au_level_pairwise':{name:strata_helper.stratify(rows,methods) for name,methods in pairs.items()},'au_level_all_methods':strata_helper.stratify(rows,METHODS),
            'extra_diagnostics':{'pairwise':{name:other_compare(rows,methods,extra_keys) for name,methods in pairs.items()},'all_methods':other_compare(rows,METHODS,extra_keys)},
            'no_edit_invariance':noedit,'metadata_audit':json.loads((folder/'metadata_audit.json').read_text()),
            'initial_v2_records_preserved':rows[:95]==json.loads(oldfile.read_text())['records'],
            'input_files':{str(p.relative_to(HERE)):sha(p) for p in (oldfile,aufile,fnfile,bfile,folder/'items.json',HERE/'policy.json')},'script_sha256':sha(__file__)}
    out=HERE/'comparison_five';out.mkdir(exist_ok=False);save(out/'results.json',result);(out/'summarizer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'coverage':coverage,'edited':{k:{'n':v['common_cases'],'means':v['subject_equal_means']} for k,v in result['all_methods_common_valid']['all']['edited'].items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
