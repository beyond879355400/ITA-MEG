#!/usr/bin/env python3
"""Independent external comparison verification, no summary helper imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
HERE=Path(__file__).resolve().parent
POLICY='d29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_equal(a,b):
    assert (a is None)==(b is None)
    if a is not None:assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-12,(a,b)
def avg(a):return mean(a) if a else None
def main():
    p=argparse.ArgumentParser();p.add_argument('folder',choices=('comparison_seven','comparison_eight'));args=p.parse_args();folder=HERE/args.folder
    r=json.loads((folder/'results.json').read_text());assert r['status']=='complete' and r['policy_sha256']==POLICY==digest(HERE/'policy.json') and not r['test_used']
    records=r['records'];by={(a['case_id'],a['method']):a for a in records};assert len(by)==len(records)
    metrics=['target_mae_changed','delta_mae_changed','unchanged_source_relative_drift','target_mae_all12','delta_mae_all12','direction_correct_fraction','identity_cosine'];numeric=0
    for a in records:
        assert a['split']=='dev' and digest(a['path'])==a['sha256'] and a['policy_sha256']==POLICY
        s=by[(a['case_id'],'source_real')];t=by[(a['case_id'],'target_real')];sg=a['source_labels_12'];tg=a['target_labels_12'];pred=a['vector'];sp=s['vector']
        c=[j for j in range(12) if sg[j]!=tg[j]];u=[j for j in range(12) if sg[j]==tg[j]]
        q={k:None for k in metrics if k!='identity_cosine'}
        if pred is not None:
            q['target_mae_changed']=avg([abs(pred[j]-tg[j]) for j in c]);q['target_mae_all12']=mean(abs(pred[j]-tg[j]) for j in range(12))
            if sp is not None:
                d=[pred[j]-sp[j] for j in range(12)];w=[tg[j]-sg[j] for j in range(12)]
                q.update(delta_mae_changed=avg([abs(d[j]-w[j]) for j in c]),delta_mae_all12=mean(abs(d[j]-w[j]) for j in range(12)),
                         unchanged_source_relative_drift=avg([abs(d[j]) for j in u]),direction_correct_fraction=avg([float(d[j]*w[j]>0) for j in c]))
        for k,v in q.items():check_equal(v,a[k]);numeric+=1
        check_equal(a['identity_cosine'],a['facenet_primary']['identity_cosine'])
        candidates=[c for c in a['all_candidates'] if c['area68'] is not None];chosen=max(candidates,key=lambda z:(z['area68'],-z['row_index'])) if candidates else None
        assert a['selected_row']==(chosen['row_index'] if chosen else None)
        face=a['facenet_primary'];eligible=[c for c in face['candidates'] if c['geometry_rankable']];selected=max(eligible,key=lambda z:(z['area'],-z['index'])) if eligible else None
        assert face['selected_index']==(selected['index'] if selected else None) and face['sha256']==a['sha256'] and face['policy_sha256']==POLICY
        for au in [7,10,14,23,45]:
            val=a['selected_au17'].get(str(au)) if a['selected_au17'] else None;truth=t['selected_au17'].get(str(au)) if t['selected_au17'] else None
            check_equal(a[f'extra_AU{au}_error_to_real_target'],abs(val-truth) if val is not None and truth is not None else None)
    summary=r.get('all_methods_common_valid',r.get('threeway_common_valid'));aggregate=0
    for cohort,scopes in summary.items():
        subset=[a for a in records if (cohort=='all' or a['cohort']==cohort) and not a['method'].endswith('_real')]
        methods=sorted({a['method'] for a in subset});mapping={(a['case_id'],a['method']):a for a in subset}
        for scope,stats in scopes.items():
            cases=sorted({a['case_id'] for a in subset if scope=='all' or (a['group']=='no_edit')==(scope=='no_edit')})
            for metric,value in stats.items():
                valid=[cid for cid in cases if all(mapping[(cid,m)][metric] is not None for m in methods)];assert value['common_cases']==len(valid) and value['case_ids']==valid
                subjects=sorted({mapping[(cid,methods[0])]['subject'] for cid in valid});assert value['subjects']==subjects
                for m in methods:
                    means=[mean(mapping[(cid,m)][metric] for cid in valid if mapping[(cid,m)]['subject']==s) for s in subjects]
                    check_equal(avg(means),value['subject_equal_means'][m]);aggregate+=1
    result={'status':'pass','policy_sha256':POLICY,'input_sha256':digest(folder/'results.json'),'test_used':False,'record_count':len(records),'independent_metric_checks':numeric,
            'subject_equal_mean_checks':aggregate,'geometry_checks':len(records)*2,'input_images_SHA_verified':len(records),'script_sha256':digest(__file__)}
    with (folder/'independent_verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
