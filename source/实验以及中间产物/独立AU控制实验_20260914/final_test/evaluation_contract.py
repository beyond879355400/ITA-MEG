"""Final-test CPU measurement contract. Importing this module reads no pixels."""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
PROJECT=EXP.parent.parent
V2=EXP/'development_evaluation/evaluation_policy_v2'
POLICY_SHA='d29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c'
TEST_SUBJECTS=('SN003','SN005','SN008','SN009','SN021','SN024','SN031','SN032')
SEEDS=(101,202,303)
AU_ORDER=(1,2,4,5,6,9,12,15,17,20,25,26)
SOURCE_METRICS={'target_mae_changed':'lower','delta_mae_changed':'lower','unchanged_source_relative_drift':'lower',
 'target_mae_all12':'lower','delta_mae_all12':'lower','direction_correct_fraction':'higher','identity_cosine':'higher','buffalo_identity_cosine':'higher',
 'extra5_proxy_mae_to_real_target':'lower',**{f'extra_AU{a}_error_to_real_target':'lower' for a in (7,10,14,23,45)}}

class EvaluationGateError(ValueError):pass
def require(condition,message):
    if not condition:raise EvaluationGateError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def resolve(path):
    require(isinstance(path,str) and bool(path),'Missing actual path')
    p=Path(path);return (p if p.is_absolute() else PROJECT/p).resolve()
def ref_path(ref,base=EXP):
    require(isinstance(ref,dict) and re.fullmatch('[0-9a-f]{64}',str(ref.get('sha256',''))) is not None,'Missing actual SHA reference')
    p=resolve(ref.get('path'));require(p.is_relative_to(base.resolve()) and p.is_file() and sha(p)==ref['sha256'],'Reference path/SHA mismatch')
    return p

def frozen_header(protocol):
    require(protocol.get('kind')=='final_test_extraction_protocol' and protocol.get('status')=='frozen_final_test' and protocol.get('extraction_authorized') is True,
            'Final protocol not frozen/authorized; reject before any pixel access')
    require(protocol.get('sampling_seeds')==list(SEEDS) and protocol.get('case_seed_policy')=='full_cartesian','All101/202/303 seeds required')
    ids=protocol.get('case_ids',[]);require(len(ids)==32 and len(set(ids))==32,'Complete explicit32-case plan required')
    methods=[m.get('method') for m in protocol.get('models',[])];require(methods and all(isinstance(m,str) and m for m in methods) and len(set(methods))==len(methods),'Final methods not fixed')
    require(not(set(methods)&{'source_real','target_real'}),'Generated method cannot reuse reserved real-reference roles')
    return methods

def analysis_contract(protocol,sampling,evaluation):
    methods=frozen_header(protocol)
    require(evaluation.get('status')=='frozen_for_final_test' and set(evaluation.get('methods',[]))==set(methods),'Final evaluation method set not frozen')
    require(sampling.get('status')=='frozen_for_final_test' and sampling.get('sampling_seeds')==list(SEEDS) and set(sampling.get('method_settings',{}))==set(methods),'Final sampling not frozen')
    require(sampling.get('our_generator_uses_target_pixels') is False,'Our generator target-pixel boundary missing')
    require(evaluation.get('measurement_policy',{}).get('sha256')==POLICY_SHA,'Must explicitly bind unchangedv2 measurement policy')
    require(evaluation.get('reference_image_space')=='source_cache_canonical512_and_independent_target_canonical512','Real reference space must be explicit')
    metrics=evaluation.get('metrics');require(isinstance(metrics,dict) and bool(metrics),'Explicit metrics required, no default selection')
    for name,definition in metrics.items():
        require(isinstance(name,str) and name and isinstance(definition,dict),'Invalid metric definition')
        source=definition.get('source_metric');require(source in SOURCE_METRICS and definition.get('direction')==SOURCE_METRICS[source],'Unsupported metric or wrong direction')
        require(definition.get('scope') in ('all','edited','no_edit'),'Metric scope must be explicit')
    comparisons=evaluation.get('comparisons');require(isinstance(comparisons,list) and comparisons,'Explicit comparisons/families required')
    seen=set();families=set()
    for c in comparisons:
        require(isinstance(c.get('id'),str) and c['id'] and c['id'] not in seen,'Duplicate/missing comparison ID');seen.add(c['id'])
        require(c.get('method') in methods and c.get('reference') in methods and c['method']!=c['reference'] and c.get('metric') in metrics,'Unplanned comparison')
        require(isinstance(c.get('family'),str) and c['family'],'Comparison family missing');families.add(c['family'])
    primary=evaluation.get('primary_families',[evaluation.get('primary_family')]);require(isinstance(primary,list) and primary and all(isinstance(p,str) and p in families for p in primary),'Primary families must be chosen before test')
    require(evaluation.get('primary_family') in families,'Primary family missing')
    stats=evaluation.get('statistics');require(isinstance(stats,dict) and stats.get('bootstrap_samples')==50000 and stats.get('confidence_level')==.95 and type(stats.get('bootstrap_seed')) is int and stats['bootstrap_seed']>=0,'Explicit fixed-statistics settings required')
    for method in methods:
        require(sampling['method_settings'][method].get('input_group') in ('source_text_au','source_manual_au','source_real_driver'),'Frozen method input-information group required')
        rule=sampling['method_settings'][method].get('seed_execution',{})
        require(rule.get('mode') in ('stochastic_each_seed','deterministic_alias'),'Seed execution policy must be explicit')
        if rule['mode']=='deterministic_alias':require(rule.get('actual_seed')==101,'Deterministic actual seed must be101;202/303 explicit aliases')
    return {'metrics':copy.deepcopy(metrics),'comparisons':copy.deepcopy(comparisons),**copy.deepcopy(stats)}

def source_cache_index_binding(sampling,reference_manifest,source_cache,indexes):
    """Our generation must use the exact source cache whose images are evaluated."""
    ref=reference_manifest['source_cache_manifest'];features=source_cache['features_file'];bound=[]
    for index in indexes:
        ours=[m for m in index['method_names'] if sampling['method_settings'][m]['input_group']=='source_text_au']
        if not ours:continue
        for actual,expected,label in ((index.get('source_cache_manifest'),ref,'source cache'),(index.get('source_features_file'),features,'source features')):
            require(isinstance(actual,dict) and actual.get('sha256')==expected['sha256'] and resolve(actual.get('path'))==resolve(expected['path']),'Generation/evaluation '+label+' binding differs')
        bound.extend(ours)
    return bound

def load_control(protocol_path,raw_path,raw_sha,role,implementation_path):
    # This first check deliberately happens before loading any raw-image metadata,
    # image library, model or evaluator. No not_frozen template may proceed.
    require(Path(protocol_path).is_file(),'Frozen final protocol does not exist')
    frozen_header(read(protocol_path))
    from protocol_gate import validate_control,verify_runtime_source
    control=validate_control(protocol_path,raw_path,raw_sha)
    control['analysis_spec']=analysis_contract(control['protocol'],control['sampling'],control['evaluation'])
    require(ref_path(control['evaluation']['measurement_policy'])==V2/'policy.json' and sha(V2/'policy.json')==POLICY_SHA,'Wrong frozenv2 policy file')
    verify_runtime_source(control,'final_evaluation_contract',__file__)
    verify_runtime_source(control,role,implementation_path)
    control['cases']=case_metadata(control['protocol'],control['raw_manifest'])
    return control

def case_metadata(protocol,raw):
    frozen_header(protocol);cases=raw.get('cases',[]);require([r['case_id'] for r in cases]==protocol['case_ids'],'Raw case list/order differs')
    require({r['subject'] for r in cases}==set(TEST_SUBJECTS),'Unexpected test subjects')
    result={}
    for case in cases:
        require(case.get('split')=='test' and case.get('target_pixels_passed_to_our_generator') is False,'Case role/target boundary changed')
        g=case['generation_input'];a=g['source_au_vector_0_5'];b=g['target_au_vector_0_5']
        require(len(a)==len(b)==12 and all(type(x) in (float,int) and math.isfinite(x) and 0<=x<=5 for x in a+b),'Invalid manual12AU values')
        result[case['case_id']]={'case_id':case['case_id'],'subject':case['subject'],'split':'test','group':'no_edit' if a==b else 'edited',
                               'source_labels_12':copy.deepcopy(a),'target_labels_12':copy.deepcopy(b)}
    return result

def planned_rows(protocol,cases):
    return [{'method':m,'subject':cases[cid]['subject'],'case_id':cid,'seed':seed} for m in frozen_header(protocol) for cid in protocol['case_ids'] for seed in protocol['sampling_seeds']]
def row_key(row):return (row['method'],row['subject'],row['case_id'],row['seed'])

def normalize_outputs(protocol,sampling,cases,index_documents,protocol_sha,raw_sha):
    """Validate logical output records without opening their image files."""
    plan=planned_rows(protocol,cases);expected={row_key(r) for r in plan};records={};declared_methods=set();indexes=[]
    for index in index_documents:
        require(index.get('status') in ('complete','completed_with_failures'),'Generation index is not terminal')
        require(index.get('usable_for_evaluation',True) is True,'Generation index quarantined after run-integrity failure')
        require(index.get('protocol_sha256')==protocol_sha and index.get('raw_manifest',{}).get('sha256')==raw_sha,'Generation protocol/raw binding mismatch')
        names=index.get('method_names');require(isinstance(names,list) and names and not(set(names)&declared_methods) and set(names)<=set(frozen_header(protocol)),'Duplicate/unplanned method index')
        declared_methods.update(names)
        require(index.get('case_ids')==protocol['case_ids'] and index.get('sampling_seeds')==list(SEEDS),'Generation case/seed plan changed')
        for row in index.get('records',[]):
            key=row_key(row);require(key in expected and key not in records and row['method'] in names,'Unplanned/duplicate output row')
            require(row.get('split')=='test' and row.get('status') in ('complete','failed','not_started'),'Invalid output status/role')
            require(isinstance(row.get('execution_id'),str) and row['execution_id'],'Execution ID required even for failed planned call')
            require(type(row.get('execution_started')) is bool,'Physical generation started flag required')
            require(row['status']!='not_started' or not row['execution_started'],'Not-started row cannot count as execution')
            if row['status']=='complete':
                require(row['execution_started'],'A complete fresh generation must have actually run')
                ref=row.get('output');require(isinstance(ref,dict) and isinstance(ref.get('path'),str) and re.fullmatch('[0-9a-f]{64}',str(ref.get('sha256',''))) is not None,'Complete output lacks path/SHA')
            else:require(row.get('output') is None,'Failed/not_started output must not carry a success image')
            records[key]=copy.deepcopy(row)
        indexes.append(index)
    require(declared_methods==set(frozen_header(protocol)),'Every frozen method must have a terminal index; missing rows remain in denominator')
    for key,row in records.items():
        method,subject,cid,seed=key;rule=sampling['method_settings'][method]['seed_execution'];alias=row.get('alias_of')
        if rule['mode']=='stochastic_each_seed':require(alias is None,'Stochastic output cannot masquerade as an alias')
        elif seed==rule['actual_seed']:require(alias is None,'Actual deterministic execution cannot alias itself')
        else:
            expected_alias={'method':method,'case_id':cid,'seed':rule['actual_seed']};require(alias==expected_alias,'Deterministic alias source changed')
            original=records.get((method,subject,cid,rule['actual_seed']));require(original is not None,'Alias without actual base record')
            require(row['execution_id']==original['execution_id'] and row['execution_started']==original['execution_started'] and row['status']==original['status'] and row.get('output')==original.get('output'),'Alias must retain exact same execution/status/path/SHA')
    # A physical run cannot secretly stand for two stochastic draws or two cases.
    physical={}
    for key,row in records.items():
        if row.get('alias_of') is not None:continue
        execution=row['execution_id'];require(execution not in physical,'Repeated physical execution ID across non-alias outputs');physical[execution]=row
    accounting={}
    for method in frozen_header(protocol):
        own=[r for r in records.values() if r['method']==method];physical_rows=[r for r in own if r.get('alias_of') is None]
        rule=sampling['method_settings'][method]['seed_execution']
        accounting[method]={'planned_logical_seed_outputs':len(protocol['case_ids'])*len(SEEDS),'observed_logical_rows':len(own),
            'planned_physical_executions':len(protocol['case_ids'])*(1 if rule['mode']=='deterministic_alias' else len(SEEDS)),
            'executed_physical_calls':sum(r['execution_started'] for r in physical_rows),'successful_physical_calls':sum(r['status']=='complete' for r in physical_rows),
            'unique_successful_image_SHAs':len({r['output']['sha256'] for r in own if r['status']=='complete'}),'alias_rows':sum(r.get('alias_of') is not None for r in own),
            'missing_logical_rows':len(protocol['case_ids'])*len(SEEDS)-len(own)}
    for index in indexes:
        own=index['records'];actual=len({r['execution_id'] for r in own if r.get('alias_of') is None and r['execution_started']})
        unique=len({r['output']['sha256'] for r in own if r['status']=='complete'})
        require(index.get('physical_executions')==actual and index.get('unique_output_images')==unique,'Declared physical/unique counts do not match records')
    return plan,records,accounting

# Metadata whitelist: never pass vectors, confidences or prior face measurements.
ITEM_FIELDS=('sample_id','case_id','subject','split','group','method','seed','path','sha256','source_labels_12','target_labels_12','execution_id','execution_started','alias_of')
def clean_item(row):return {k:copy.deepcopy(row[k]) for k in ITEM_FIELDS if k in row}
def check_item(item):require(set(item)<=set(ITEM_FIELDS) and item.get('split')=='test','Measurement fields or wrong split in evaluation input')

def statistics_document(protocol,evaluation,plan,normalized,metric_rows,accounting,metadata):
    """No metric/family selection; full denominator and explicit scopes only."""
    mapped={row_key(r):r for r in metric_rows};require(len(mapped)==len(metric_rows),'Duplicate measured output')
    expected={row_key(r) for r in plan};require(set(mapped)<=expected,'Unplanned measured output')
    records=[]
    for p in plan:
        key=row_key(p);generation=normalized.get(key)
        if generation is None:continue  # still present in planned; tool reports missing_output
        measured=mapped.get(key);metrics={};reasons={}
        for name,definition in evaluation['metrics'].items():
            group=measured.get('group') if measured else None;scope=definition['scope'];value=None
            if generation['status']!='complete':reason='generation_'+generation['status']
            elif measured is None:reason='measurement_missing'
            elif scope!='all' and group!=scope:reason='out_of_predeclared_scope'
            else:
                value=measured.get(definition['source_metric']);reason=None if value is not None else 'measurement_or_reference_NA'
                if value is not None:require(type(value) in (int,float) and math.isfinite(value),'Nonfinite exported metric')
            metrics[name]=value;reasons[name]=reason
        records.append({**p,'status':'complete' if generation['status']=='complete' else generation['status'],'metrics':metrics,'metric_reasons':reasons,
                        'output':generation.get('output'),'execution_id':generation['execution_id'],'alias_of':generation.get('alias_of')})
    spec={'metrics':copy.deepcopy(evaluation['metrics']),'comparisons':copy.deepcopy(evaluation['comparisons']),**copy.deepcopy(evaluation['statistics'])}
    return {'kind':'paired_subject_metric_input','metadata':{**metadata,'data_role':'final_test','physical_execution_accounting':accounting,
          'alias_seed_rows_are_not_independent_generations':True,'statistical_unit':'subject; seed then case then subject','primary_family':evaluation['primary_family'],
          'primary_families':evaluation.get('primary_families',[evaluation['primary_family']])},'analysis_spec':spec,'planned':copy.deepcopy(plan),'records':records}
