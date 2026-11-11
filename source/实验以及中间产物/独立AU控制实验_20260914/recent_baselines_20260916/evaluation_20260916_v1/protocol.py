"""Append-only external evaluation contract; import is metadata-only."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
NEW=HERE.parent
EXP=NEW.parent
OLD=EXP/'completion_20260915/evaluation'
BUNDLE=NEW/'preservation/bundle_v2_20260916'
POLICY_SHA='d29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c'
ROLE='postfreeze_case_validation'
CHOSEN='bidirectional_local_lp_s050'
OLD_METHODS=['base_local_lp_s050',CHOSEN,'condition_reads_image_local_lp_s050','image_reads_condition_local_lp_s050','mlp_local_lp_s050','magicface','faceadapter','liveportrait']
NEW_METHODS=['swiftedit','performrecast']
METHODS=OLD_METHODS+NEW_METHODS
SEEDS=[101,202,303]
AUS=[1,2,4,5,6,9,12,15,17,20,25,26]
PINS={
 'old_contract':('completion_20260915/evaluation/contract.py','babf8b3075eb905e8d2f748ca39cb4b5075971abcfc5a7b9d093983fee633cd7'),
 'old_evaluate':('completion_20260915/evaluation/evaluate.py','c7691fa1cbd504d12cadb0b133ebc32578a694bae354c55ac89dcebae0fc28b7'),
 'old_metrics':('completion_20260915/evaluation/all_methods_v1/statistics/all_case_seed_metrics.json','8bf2f1245613f2215ae08c3d91cbb4af94fe737e4817336660539643553ffcee'),
 'old_items':('completion_20260915/evaluation/all_methods_v1/items.json','390cd5f571b60b60a12c005717f751720c19c5b35d32d72ba777629ab680571e'),
 'references':('completion_20260915/evaluation/references_v1/references.json','e93c575050f2fa0a3b1e312830fe4b622bc33d0a3fa76efeaf77c7484a168761'),
 'cases':('recent_baselines_20260916/preservation/bundle_v2_20260916/inputs/postfreeze32.json','fa88ca09eb8253ce61ba046cb8d9b4c4affaef6cc421893cedb487c09522413b'),
}
METRICS={'target_mae_changed':'lower','delta_mae_changed':'lower','unchanged_source_relative_drift':'lower',
 'target_mae_all12':'lower','delta_mae_all12':'lower','direction_correct_fraction':'higher','identity_cosine':'higher',
 'buffalo_identity_cosine':'higher','extra5_proxy_mae_to_real_target':'lower',
 **{f'extra_AU{a}_error_to_real_target':'lower'for a in [7,10,14,23,45]}}
METHOD_SETTINGS={
 'performrecast':{'input_group':'source_real_target_driver','seed_policy':'deterministic_one_execution_per_case',
                 'primary_output_role':'native_crop_frame0000','full_output':'display_archive_only','physical_outputs':32},
 'swiftedit':{'input_group':'source_text','seed_policy':'three_real_stochastic_executions',
              'primary_output_role':'target_edit','excluded_output_role':'source_reconstruction','physical_outputs':96,
              'base_source_status':'community_archive_original_origin_not_independently_verified',
              'source_description':'official dedicated checkpoints plus fixed SD2.1 community auxiliary archive',
              'automatic_thesis_promotion':False},
}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def ref(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def verified(value):
    p=Path(value['path']).resolve()
    need(p.is_relative_to(EXP)and p.is_file()and sha(p)==value['sha256'],'Reference path or SHA changed: '+str(p))
    return p


def pin(name):
    relative,expected=PINS[name];p=EXP/relative
    need(sha(p)==expected,'Frozen source changed: '+name)
    return p


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def legacy():
    c=load_module('new_external_frozen_contract',pin('old_contract'))
    prior=sys.modules.get('contract');sys.modules['contract']=c
    try:e=load_module('new_external_frozen_evaluate',pin('old_evaluate'))
    finally:
        if prior is None:sys.modules.pop('contract',None)
        else:sys.modules['contract']=prior
    c.execution_spec()  # Verify all original policy/algorithm/dependency hashes; no model imports.
    return c,e


def context():
    c,e=legacy();cases=read(pin('cases'))['cases'];old=read(pin('old_items'));refs=read(pin('references'))
    need(len(cases)==32 and len({r['subject']for r in cases})==8,'Expected32cases/eight subjects')
    need([r['case_id']for r in cases]==list(old['case_metadata'])==[r['case_id']for r in refs['cases']],'Case order mismatch')
    for row in cases:
        meta=old['case_metadata'][row['case_id']]
        need(row['split']==ROLE and row['au_order']==AUS and row['sampling_seeds']==SEEDS,'Case role/AU/seeds changed')
        need(meta['source_labels_12']==row['source_au_vector_0_5']and meta['target_labels_12']==row['target_au_vector_0_5'],'Labels differ')
        r=next(x for x in refs['cases']if x['case_id']==row['case_id'])
        need(r['source_reference']['raw_image']['sha256']==row['source_sha256']and r['target_reference']['raw_image']['sha256']==row['target_sha256'],'Raw reference differs')
    for subject in {r['subject']for r in cases}:
        rows=[r for r in cases if r['subject']==subject]
        need(len(rows)==4 and sum(r['source_au_vector_0_5']==r['target_au_vector_0_5']for r in rows)==1,'Per-subject case balance changed')
    return c,e,cases,old,refs


def analysis_spec():
    metrics={k+'__'+scope:{'source_metric':k,'scope':scope,'direction':v}for k,v in METRICS.items()for scope in ['all','edited','no_edit']}
    return {'metrics':metrics,'comparisons':[{'id':'chosen_vs_'+m,'method':CHOSEN,'reference':m,
        'metric':'target_mae_changed__edited','family':'all_five_external_primary_C'}for m in ['magicface','faceadapter','liveportrait','swiftedit','performrecast']],
        'bootstrap_samples':50000,'bootstrap_seed':20260915,'confidence_level':.95}


def specification():
    c,e,cases,old,refs=context()
    return {'kind':'recent_external_evaluation_protocol_v1','data_role':ROLE,'previously_exposed_subjects':True,
      'independent_final_test':False,'old_generation_protocol_sha256':old['protocol_sha256'],'measurement_policy_sha256':POLICY_SHA,
      'case_ids':[r['case_id']for r in cases],'subject_count':8,'edited_cases':24,'no_edit_cases':8,
      'old_methods':OLD_METHODS,'new_methods':NEW_METHODS,'method_settings':METHOD_SETTINGS,
      'source_bindings':{k:ref(pin(k))for k in PINS},'old_algorithm_spec':c.execution_spec(),
      'execution_sources':{n:ref(HERE/n)for n in ['protocol.py','evaluate_new.py']},
      'environment_bridge':{'case_selection':'first frozen manifest case, selected method seed101 plus same source/target references',
                            'OpenFace_AU_absolute_tolerance':1e-8,'geometry_absolute_tolerance':1e-4,
                            'FaceNet_and_buffalo_cosine_absolute_tolerance':1e-5,
                            'NA_and_selected_face_status_must_match':True,
                            'pass_required_before_new_environment_results_are_mixed':True},
      'analysis_spec':analysis_spec(),
      'comparison_unit':'per-metric valid seeds averaged within each case; pair methods on common valid cases; subject-equal averaging',
      'deterministic_seed':'producer seed=null; evaluator internal key -1 with seed_not_applicable=true; not an actual random draw',
      'case_analysis_rows':'seed=0 solely an analysis-only case-mean key; never represented as generated images',
      'old_results':'reuse immutable measurements/metrics; new case-level statistics and Holm5 saved independently; old7comparisons untouched',
      'crop_policy':'predeclared native RGB output only; no new crop/resize/padding/enhancement/reencoding before unchanged evaluator',
      'failure_policy':'all planned outputs retained; explicit failure/NA and out-of-scope counts; never impute0 or switch face',
      'gallery':{'full':'all32 in original order; fixed101 for stochastic methods; deterministic only output',
                 'diagnostic':'each new baseline Top3 edited case target_C_MAE valid-seed mean descending; ties original case order',
                 'representative_seed':101,'counterpart':'same case chosen method/source/target',
                 'failures':'separate case/seed-order failure gallery; no image means status placeholder',
                 'not_representative_of_overall':True},'automatic_thesis_promotion':False}


def load_protocol(path,expected):
    need(sha(path)==expected,'Explicit protocolSHA mismatch');doc=read(path)
    need(doc==specification(),'Protocol/code/source binding changed; emit new version instead of overwrite')
    return doc


def gate(approval,protocol_sha,phase):
    doc=read(approval)
    need(doc.get('granted_by')=='root'and doc.get('protocol_sha256')==protocol_sha and phase in doc.get('phases',[])and doc.get('message'),'Root execution authorization missing for '+phase)


def key(row):return row['method'],row['subject'],row['case_id'],row['seed']


def new_plan(cases):
    return [{'method':m,'subject':r['subject'],'case_id':r['case_id'],'seed':s,
             'seed_not_applicable':m=='performrecast'}for m in NEW_METHODS for r in cases for s in ([-1]if m=='performrecast'else SEEDS)]


def normalize(cases,indexes,protocol_sha):
    plan=new_plan(cases);expected={key(r)for r in plan};bycase={r['case_id']:r for r in cases};rows={};seen=set();execution_ids=set()
    for doc in indexes:
        need(doc.get('kind')=='recent_baseline_generation_index_v1','Unknown generation index schema')
        m=doc.get('method');need(m in NEW_METHODS and m not in seen,'Unknown/duplicate new method index');seen.add(m)
        need(doc.get('status')in ['complete','completed_with_failures']and doc.get('protocol_sha256')==protocol_sha,'Index not terminal or wrong protocol')
        need(doc.get('case_ids')==list(bycase)and doc.get('case_manifest_sha256')==PINS['cases'][1],'Frozen case binding changed')
        need(doc.get('input_group')==METHOD_SETTINGS[m]['input_group'],'Method input group mismatch')
        need(doc.get('sampling_seeds')==([]if m=='performrecast'else SEEDS),'Method seed policy changed')
        if m=='swiftedit':need(doc.get('base_source_status')==METHOD_SETTINGS[m]['base_source_status'],'Swift source label missing')
        for original in doc['records']:
            row=copy.deepcopy(original);row['method']=m
            if m=='performrecast':
                need(row.get('seed')is None and row.get('seed_not_applicable')is True,'Deterministic output must not fake seed draws')
                row['seed']=-1
            else:need(row.get('seed')in SEEDS and not row.get('seed_not_applicable',False),'Wrong stochastic seed')
            k=key(row);need(k in expected and k not in rows,'Duplicate/unplanned output')
            c=bycase[row['case_id']]
            need(row.get('source_sha256')==c['source_sha256']and row.get('split')==ROLE,'Source/role mismatch')
            need(row.get('source_au_vector_0_5')==c['source_au_vector_0_5']and row.get('target_au_vector_0_5')==c['target_au_vector_0_5'],'Manual labels changed')
            need(row.get('target_image_used')is (m=='performrecast'),'Target-image input boundary changed')
            if m=='performrecast':need(row.get('target_sha256')==c['target_sha256'],'Driver target differs')
            else:need(row.get('target_prompt')==c['prompt'],'Swift target prompt changed')
            need(row.get('primary_output_role')==METHOD_SETTINGS[m]['primary_output_role'],'Selected output changed')
            need(row.get('status')in ['complete','failed','not_started'],'Unexpected output status')
            eid=row.get('execution_id');need(isinstance(eid,str)and eid and eid not in execution_ids,'Physical execution missing or reused')
            execution_ids.add(eid);need(type(row.get('execution_started'))is bool,'Execution flag required')
            if row['status']=='complete':
                need(row['execution_started']and isinstance(row.get('output'),dict),'Complete output requires execution/image')
                need(re.fullmatch('[0-9a-f]{64}',str(row['output'].get('sha256'))),'Image SHA required')
            else:
                if row.get('output')is not None:row['failed_artifact']=row['output'];row['output']=None
                need(row.get('failure_reason'),'All failed/unstarted outputs need reason')
            need(row.get('status')!='not_started'or row['execution_started']is False,'Unstarted execution contradiction')
            rows[k]=row
    need(seen==set(NEW_METHODS),'Both methods require terminal indexes, even if all failed')
    need(set(rows)==expected,'Every128 planned new output needs terminal row; do not omit failed requests')
    return plan,rows


def finite(value):return type(value)in (int,float)and math.isfinite(value)


def case_rows(metric_rows,generation,case_metadata):
    """No bootstrap/model. Preserve valid/missing seed counts before case pairing."""
    lookup={key(r):r for r in metric_rows};need(len(lookup)==len(metric_rows),'Duplicate metric record')
    planned=[];records=[];counts=[]
    for m in METHODS:
        for cid,meta in case_metadata.items():
            expected=[-1]if m=='performrecast'else SEEDS
            keys=[(m,meta['subject'],cid,s)for s in expected]
            means={};coverage={}
            for metric in METRICS:
                values=[];reasons=[]
                for k in keys:
                    g,r=generation.get(k),lookup.get(k)
                    if g is None or g.get('status')!='complete':reason='generation_'+(g.get('status','missing')if g else 'missing')
                    elif r is None:reason='measurement_missing'
                    elif r.get(metric)is None:reason='measurement_or_reference_NA'
                    else:
                        need(finite(r[metric]),'Nonfinite metric must be NA');values.append(r[metric]);reason=None
                    reasons.append({'seed':None if k[3]==-1 else k[3],'reason':reason})
                means[metric]=math.fsum(values)/len(values)if values else None
                coverage[metric]={'planned_outputs':len(expected),'valid_outputs':len(values),'seed_reasons':reasons}
            p={'method':m,'subject':meta['subject'],'case_id':cid,'seed':0,'analysis_only_case_mean':True};planned.append(p)
            vals={};why={}
            for name,definition in analysis_spec()['metrics'].items():
                if definition['scope']!='all'and meta['group']!=definition['scope']:vals[name]=None;why[name]='out_of_predeclared_scope'
                else:
                    vals[name]=means[definition['source_metric']];why[name]=None if vals[name]is not None else 'no_valid_seed_metric'
            records.append({**p,'status':'complete','metrics':vals,'metric_reasons':why,'case_metric_means':means,'seed_coverage':coverage})
            counts.append({'method':m,'case_id':cid,'subject':meta['subject'],'group':meta['group'],'metrics':coverage})
    return {'kind':'paired_subject_metric_input','planned':planned,'records':records,'analysis_spec':analysis_spec(),
            'metadata':{'data_role':ROLE,'independent_final_test':False,'paired_unit':'case after valid-seed mean then equal subjects',
                        'physical_seed_counts_retained':True,'no_fake_deterministic_repeats':True,'case_analysis_seed0_is_not_generation':True}},counts


def gallery_plan(case_input,case_metadata,per_image_rows):
    rank={cid:i for i,cid in enumerate(case_metadata)};out={}
    for method in NEW_METHODS:
        rows=[r for r in case_input['records']if r['method']==method and case_metadata[r['case_id']]['group']=='edited'and r['case_metric_means']['target_mae_changed']is not None]
        rows.sort(key=lambda r:(-r['case_metric_means']['target_mae_changed'],rank[r['case_id']]))
        out[method]={'top3':[{'case_id':r['case_id'],'subject':r['subject'],'case_target_C_MAE':r['case_metric_means']['target_mae_changed'],
                              'display_seed':None if method=='performrecast'else 101,'chosen_display_seed':101}for r in rows[:3]],
                     'title':'误差较大案例诊断（不是整体代表）'}
    return {'kind':'predeclared_gallery_selection','full_case_order':list(case_metadata),'primary_display_seed':101,
            'top3_by_method':out,'failures':[r for r in per_image_rows if r.get('generation_status')!='complete'or r.get('target_mae_changed')is None and r.get('group')=='edited'],
            'failure_order':'method then original case order then declared seed','no_result_based_output_or_seed_switch':True}
