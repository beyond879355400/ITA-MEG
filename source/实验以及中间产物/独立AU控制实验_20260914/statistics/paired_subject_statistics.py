"""Explicit-plan paired subject statistics. No model/evaluator invocation.

Each metric is aggregated seed -> case -> equally weighted subject. Paired
comparisons use only common valid (subject, case_id, seed) keys. Planned failures
and missing outputs remain in coverage. Families are supplied, never selected.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

BOOTSTRAP_SAMPLES=50000
BOOTSTRAP_SEED=20260914
CONFIDENCE_LEVEL=.95
MAX_EXACT_SUBJECTS=20


def mean(values):
    values=list(values)
    return math.fsum(values)/len(values) if values else None


def row_key(row):
    values=tuple(row.get(key) for key in ('method','subject','case_id','seed'))
    if any(not isinstance(value,str) or not value for value in values[:3]) or type(values[3]) is not int:
        raise ValueError('Rows require method/subject/case_id strings and an integer seed')
    return values


def finite_value(record,metric):
    if record is None:return None,'missing_output'
    if record.get('status','complete') not in ('complete','ok'):return None,'row_status:'+str(record.get('status'))
    value=record.get('metrics',{}).get(metric)
    if value is None:return None,'metric_NA'
    if isinstance(value,bool) or not isinstance(value,(float,int)):return None,'invalid_metric_type'
    if not math.isfinite(value):return None,'nonfinite_metric'
    return float(value),None


def subject_aggregate(key_values):
    """Input keys are (subject, case_id, seed); values are finite floats."""
    groups=collections.defaultdict(list)
    for (subject,case_id,seed),value in sorted(key_values.items()):groups[(subject,case_id)].append(value)
    cases={key:mean(values) for key,values in groups.items()}
    subjects=collections.defaultdict(list)
    for (subject,case_id),value in cases.items():subjects[subject].append(value)
    by_subject={key:mean(values) for key,values in sorted(subjects.items())}
    return {'mean':mean(by_subject.values()),'n_subjects':len(by_subject),'n_cases':len(cases),'n_seed_outputs':len(key_values),
            'subject_means':by_subject,'case_means':[{'subject':s,'case_id':c,'seed_mean':v,'n_seeds':len(groups[(s,c)])} for (s,c),v in sorted(cases.items())]}


def subject_inference(subject_differences,seed=BOOTSTRAP_SEED):
    values=np.asarray(list(subject_differences),dtype=float);n=len(values)
    if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('Subject differences must be finite scalars')
    result={'n_subjects':n,'mean_difference':None if not n else float(values.mean()),'difference_ci95':None,
            'two_sided_sign_flip_p':None,'sign_patterns':None,'bootstrap_samples':0,'bootstrap_seed':seed,
            'inference_status':'no_paired_subjects' if n==0 else 'descriptive_only_one_subject' if n==1 else 'computed'}
    if n<2:return result
    if n>MAX_EXACT_SUBJECTS:raise ValueError('Exact 2^n enumeration supports at most20 subjects; no silent Monte Carlo substitution')
    rng=np.random.default_rng(seed);draws=np.empty(BOOTSTRAP_SAMPLES)
    for start in range(0,BOOTSTRAP_SAMPLES,1000):
        stop=min(start+1000,BOOTSTRAP_SAMPLES)
        indices=rng.integers(0,n,size=(stop-start,n))
        draws[start:stop]=values[indices].mean(axis=1)
    ci=np.quantile(draws,[(1-CONFIDENCE_LEVEL)/2,1-(1-CONFIDENCE_LEVEL)/2],method='linear')
    observed=abs(float(values.mean()));tolerance=1e-12*max(1.,observed)
    count=0;patterns=1<<n
    for start in range(0,patterns,4096):
        masks=np.arange(start,min(start+4096,patterns),dtype=np.uint64)[:,None]
        bits=(masks>>np.arange(n,dtype=np.uint64)[None,:])&1
        signed=(1.-2.*bits.astype(float))*values[None,:]
        count+=int(np.count_nonzero(np.abs(signed.mean(axis=1))>=observed-tolerance))
    result.update(difference_ci95=ci.tolist(),two_sided_sign_flip_p=count/patterns,sign_patterns=patterns,
                  bootstrap_samples=BOOTSTRAP_SAMPLES,sign_flip_extreme_patterns=count)
    return result


def holm_adjust(comparisons):
    """NA hypotheses stay in the declared family, conservatively ordered at1."""
    families=collections.defaultdict(list)
    for row in comparisons:
        family=row.get('family')
        if not isinstance(family,str) or not family:raise ValueError('Every comparison must explicitly declare a nonempty family')
        p=row['two_sided_sign_flip_p']
        if p is not None and not 0<=p<=1:raise ValueError('Invalid p value')
        families[family].append(row)
    for family,rows in families.items():
        ordered=sorted(rows,key=lambda row:(1. if row['two_sided_sign_flip_p'] is None else row['two_sided_sign_flip_p'],row['id']))
        running=0.;size=len(rows)
        for rank,row in enumerate(ordered):
            p=row['two_sided_sign_flip_p'];used=1. if p is None else p
            running=max(running,min(1.,(size-rank)*used))
            row['holm_adjusted_p']=None if p is None else running
            row['holm_family_size']=size
            row['holm_NA_policy']='NA retained in family at p=1 for adjustment of others; its own reported p remains NA'
    return {family:[row['id'] for row in rows] for family,rows in sorted(families.items())}


def analyze(document):
    if document.get('kind')!='paired_subject_metric_input':raise ValueError('Unknown input kind')
    spec=document['analysis_spec'];metrics=spec['metrics'];comparisons=spec['comparisons']
    if not isinstance(metrics,dict) or not metrics:raise ValueError('Explicit metric definitions are required')
    for name,definition in metrics.items():
        if not isinstance(name,str) or not name or definition.get('direction') not in ('higher','lower'):
            raise ValueError('Each metric must explicitly state higher/lower direction')
    if spec.get('bootstrap_samples',BOOTSTRAP_SAMPLES)!=BOOTSTRAP_SAMPLES or spec.get('confidence_level',CONFIDENCE_LEVEL)!=CONFIDENCE_LEVEL:
        raise ValueError('This utility uses fixed50k bootstrap and95% pointwise intervals')
    seed=spec.get('bootstrap_seed',BOOTSTRAP_SEED)
    if type(seed) is not int or seed<0:raise ValueError('Explicit nonnegative bootstrap seed required')
    planned={}
    for row in document['planned']:
        key=row_key(row)
        if key in planned:raise ValueError('Duplicate planned output')
        planned[key]=row
    if not planned:raise ValueError('An explicit nonempty denominator plan is required')
    records={}
    for row in document.get('records',[]):
        key=row_key(row)
        if key in records:raise ValueError('Duplicate metric output')
        if key not in planned:raise ValueError('Metric output was not in the explicit plan')
        records[key]=row
    methods=sorted({key[0] for key in planned});values={};reasons={}
    raw={method:{} for method in methods};coverage=[]
    for method in methods:
        keys=sorted(key for key in planned if key[0]==method)
        for metric in metrics:
            local={};counts=collections.Counter();invalid=[]
            for key in keys:
                value,reason=finite_value(records.get(key),metric);values[(key,metric)]=value;reasons[(key,metric)]=reason
                if reason is None:local[key[1:]]=value
                else:
                    counts[reason]+=1;invalid.append({'subject':key[1],'case_id':key[2],'seed':key[3],'reason':reason})
            raw[method][metric]=subject_aggregate(local)
            coverage.append({'method':method,'metric':metric,'planned_seed_outputs':len(keys),'observed_output_rows':sum(key in records for key in keys),
                'valid_metric_outputs':len(local),'planned_cases':len({key[1:3] for key in keys}),'planned_subjects':len({key[1] for key in keys}),
                'valid_cases':raw[method][metric]['n_cases'],'valid_subjects':raw[method][metric]['n_subjects'],'unavailable_by_reason':dict(counts),'unavailable_outputs':invalid})
    output=[];seen_ids=set()
    for comparison in comparisons:
        cid=comparison['id'];method=comparison['method'];reference=comparison['reference'];metric=comparison['metric'];family=comparison.get('family')
        if cid in seen_ids:raise ValueError('Duplicate comparison ID')
        seen_ids.add(cid)
        if method==reference or method not in methods or reference not in methods or metric not in metrics:raise ValueError('Comparison must name two planned methods and a declared metric')
        method_keys={key[1:] for key in planned if key[0]==method};reference_keys={key[1:] for key in planned if key[0]==reference}
        common=sorted(method_keys&reference_keys);paired_method={};paired_reference={};unavailable=[]
        for key in common:
            ma=(method,)+key;rb=(reference,)+key;a=values[(ma,metric)];b=values[(rb,metric)]
            if a is not None and b is not None:paired_method[key]=a;paired_reference[key]=b
            else:unavailable.append({'subject':key[0],'case_id':key[1],'seed':key[2],'method_reason':reasons[(ma,metric)],'reference_reason':reasons[(rb,metric)]})
        left=subject_aggregate(paired_method);right=subject_aggregate(paired_reference)
        if left['subject_means'].keys()!=right['subject_means'].keys():raise ValueError('Internal paired subject mismatch')
        subject_rows=[{'subject':subject,'method_mean':left['subject_means'][subject],'reference_mean':right['subject_means'][subject],
                       'difference':left['subject_means'][subject]-right['subject_means'][subject]} for subject in left['subject_means']]
        inference=subject_inference([row['difference'] for row in subject_rows],seed)
        output.append({'id':cid,'family':family,'method':method,'reference':reference,'metric':metric,'metric_direction':metrics[metric]['direction'],
            'difference_definition':'method minus reference, no direction flip','paired_method_mean':left['mean'],'paired_reference_mean':right['mean'],
            **inference,'n_cases':left['n_cases'],'n_seed_pairs':left['n_seed_outputs'],'subject_statistics':subject_rows,
            'case_seed_means':{'method':left['case_means'],'reference':right['case_means']},
            'coverage':{'planned_common_seed_outputs':len(common),'valid_common_seed_outputs':len(paired_method),
                        'method_only_planned_keys':[list(key) for key in sorted(method_keys-reference_keys)],
                        'reference_only_planned_keys':[list(key) for key in sorted(reference_keys-method_keys)],
                        'planned_common_cases':len({key[:2] for key in common}),'planned_common_subjects':len({key[0] for key in common}),'unavailable_pairs':unavailable}})
    families=holm_adjust(output)
    return {'kind':'paired_subject_metric_statistics','metadata':document.get('metadata',{}),'analysis_spec':spec,
        'aggregation':'valid seeds averaged within case, valid cases averaged within subject, subjects equally weighted; paired common case+seed intersections only',
        'inference':'50k whole-subject bootstrap percentile95% CI; exact two-sided2^n subject sign flips under independent-subject symmetric-zero null',
        'ci_scope':'pointwise unadjusted difference CI; Holm applies only to explicitly declared p-value families',
        'raw_method_statistics':raw,'coverage_table':coverage,'comparisons':output,'declared_holm_families':families,
        'no_model_selection_performed':True,'no_composite_score_defined':True,'no_generator_or_evaluator_called':True,
        'runtime':{'numpy':np.__version__,'python':sys.version.split()[0]}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    data=args.input.read_bytes();result=analyze(json.loads(data))
    result['input_sha256']=hashlib.sha256(data).hexdigest();result['statistics_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({'output':str(args.output),'comparisons':len(result['comparisons']),'families':result['declared_holm_families']},ensure_ascii=False))


if __name__=='__main__':main()
