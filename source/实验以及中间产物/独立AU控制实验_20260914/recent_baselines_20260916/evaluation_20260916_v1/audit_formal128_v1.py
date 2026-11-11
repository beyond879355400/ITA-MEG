"""Independent arithmetic/completeness audit; no models or image changes."""
from pathlib import Path
from collections import Counter,defaultdict
import json,math
import extent_io as io
HERE=Path(__file__).resolve().parent
JOB=HERE/'formal128_v1'
STATS=JOB/'statistics'
METRICS=['target_mae_changed','delta_mae_changed','unchanged_source_relative_drift','target_mae_all12','identity_cosine','buffalo_identity_cosine','extra5_proxy_mae_to_real_target']

def close(a,b):return a is None and b is None or a is not None and b is not None and abs(a-b)<1e-12
def mean(a):return sum(a)/len(a)if a else None
def key(r):return r['method'],r['case_id'],r['seed']
def main():
    doc=io.read_json(JOB/'items.json');combined=io.read_json(STATS/'all_case_seed_metrics.json')
    old=io.read_json(doc['old_metrics']['path'],doc['old_metrics']['sha256'])
    assert len(old['records'])==768 and combined['records'][:768]==old['records']
    allrows=combined['records'];assert len(allrows)==896 and len({key(r)for r in allrows})==896
    counts=Counter(r['method']for r in allrows);assert counts['swiftedit']==96 and counts['performrecast']==32
    assert len(doc['items'])==192 and len(doc['planned'])==128 and all(x['status']=='available'for x in doc['availability'])
    cases=io.read_json(STATS/'case_metric_input.json');assert len(cases['records'])==320
    bycase=defaultdict(list)
    for row in allrows:bycase[row['method'],row['case_id']].append(row)
    checked=0
    for row in cases['records']:
        group=bycase[row['method'],row['case_id']]
        for metric in METRICS:
            vals=[r[metric]for r in group if type(r.get(metric))in (float,int)and math.isfinite(r[metric])]
            assert close(mean(vals),row['case_metric_means'][metric]),(row['case_id'],metric)
            checked+=1
    stats=io.read_json(STATS/'paired_subject_results.json')
    comparisons=stats['comparisons'];assert len(comparisons)==5
    assert {r['reference']for r in comparisons}=={'magicface','faceadapter','liveportrait','swiftedit','performrecast'}
    previous=0;holm=[]
    for i,r in enumerate(sorted(comparisons,key=lambda r:r['two_sided_sign_flip_p'])):
        value=min(1,max(previous,(5-i)*r['two_sided_sign_flip_p']));previous=value
        assert close(value,r['holm_adjusted_p'])
        holm.append({'reference':r['reference'],'raw_p':r['two_sided_sign_flip_p'],'holm5_p':value,'n_cases':r['n_cases'],'n_subjects':r['n_subjects']})
    for method,metrics in stats['raw_method_statistics'].items():
        rows=[r for r in cases['records']if r['method']==method and doc['case_metadata'][r['case_id']]['group']=='edited']
        for metric in METRICS:
            grouped=defaultdict(list)
            for r in rows:
                v=r['case_metric_means'][metric]
                if v is not None:grouped[r['subject']].append(v)
            expect=mean([mean(v)for v in grouped.values()]);assert close(expect,metrics[metric+'__edited']['mean'])
    failures=[]
    for r in io.read_json(JOB/'openface/results.json')['records']:
        if r['parse_status']=='primary_face_finite':continue
        raw=JOB/'openface/raw'/('sha_'+r['sha256']);text=io.read_bytes(raw/'run.log').decode()
        assert r['method']=='swiftedit'and r['returncode']==0 and 'Model loaded'in text and 'Starting tracking'in text
        assert not list(raw.glob('*.csv'))and 'timed out'not in text
        failures.append({'case_id':r['case_id'],'seed':r['seed'],'native_returncode':0,'csv_count':0,'input_readable':True,'technical_timeout':False})
    assert len(failures)==6
    export=io.read_json(STATS/'export_receipt.json')
    assert all(io.sha(STATS/name)==digest for name,digest in export['outputs'].items())
    result={'status':'pass','legacy768_deep_equal':True,'all_rows':896,'case_means':320,'independently_recomputed_case_metric_checks':checked,
            'subject_equal_means_match':True,'holm5_match':holm,'technical_input_unavailable':0,'new_AU_no_face_rows':failures,
            'old_measurement_file_sha256':doc['old_metrics']['sha256'],'new_export_receipt':io.ref(STATS/'export_receipt.json'),
            'models_called':False,'old_files_changed':False,'audit_source':io.ref(__file__)}
    with(HERE/'formal128_completion_audit_v1.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':'pass','rows':896,'case_means':320,'independent_checks':checked,'model_calls':False}))

if __name__=='__main__':main()
