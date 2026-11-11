"""Root-requested common-case descriptive means only; no new significance tests."""
from pathlib import Path
from collections import defaultdict
import json
import extent_io as io
HERE=Path(__file__).resolve().parent
CHOSEN='bidirectional_local_lp_s050'

def mean(values):return sum(values)/len(values)
def main():
    job=HERE/'formal128_v1';doc=io.read_json(job/'items.json');file=job/'statistics/case_metric_input.json';data=io.read_json(file)
    lookup={(r['method'],r['case_id']):r for r in data['records']};rows=[]
    for method in ['magicface','faceadapter','liveportrait','swiftedit','performrecast']:
        for metric in ['target_mae_changed','identity_cosine','unchanged_source_relative_drift']:
            groups=defaultdict(list)
            for cid,meta in doc['case_metadata'].items():
                if meta['group']!='edited':continue
                a=lookup[CHOSEN,cid]['case_metric_means'][metric];b=lookup[method,cid]['case_metric_means'][metric]
                if a is not None and b is not None:groups[meta['subject']].append((a,b,cid))
            subject_means=[{'subject':s,'ours':mean([a for a,b,c in v]),'reference':mean([b for a,b,c in v]),'case_ids':[c for a,b,c in v]}for s,v in groups.items()]
            a=mean([r['ours']for r in subject_means]);b=mean([r['reference']for r in subject_means])
            rows.append({'method':CHOSEN,'reference':method,'metric':metric,'ours':a,'reference_mean':b,'difference_ours_minus_reference':a-b,
                         'relative_reduction_if_lower_better':(b-a)/b if metric!='identity_cosine'and b else None,
                         'n_cases':sum(len(v)for v in groups.values()),'n_subjects':len(groups),'subject_means':subject_means})
    output=HERE/'paired_descriptives_v1.json'
    with output.open('x')as f:json.dump({'status':'complete','scope':'edited only, metric-specific common valid cases, subject equal means; descriptive no p/CI',
         'source':io.ref(file),'rows':rows,'models_called':False,'new_significance_tests':False,'script':io.ref(__file__)},f,ensure_ascii=False,indent=2);f.write('\n')
    for r in rows:
        if r['reference']in ['swiftedit','performrecast']:print(json.dumps({k:v for k,v in r.items()if k!='subject_means'},ensure_ascii=False))

if __name__=='__main__':main()
