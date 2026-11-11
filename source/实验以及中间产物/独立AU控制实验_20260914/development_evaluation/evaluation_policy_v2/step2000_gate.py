#!/usr/bin/env python3
"""Read-only gates for a future complete four-structure step2000 dev19 run."""
import hashlib
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
from openface_v2 import HERE,ROOT,POLICY_SHA,sha,save

VARIANTS=('bidirectional','condition_reads_image','image_reads_condition','mlp')
LABELS={v:v+'_step2000' for v in VARIANTS}
REFERENCES=('baseline','magicface','liveportrait','faceadapter','liveportrait_absolute')
REFERENCE=HERE/'comparison_nine/results.json'
REFERENCE_SHA='352836101ebdaa9fb5cf8a95990ddb951c9677313d0b24434a2a4866b2e64e04'
CASEFILE=ROOT/'development_evaluation/dev_coverage_audit/dev19_manifest_2k.json'
CASE_SHA='90a0307b053d369e49286ea63ad3f6e731c8213b3786fbd4591706c1b7f69a28'
PARENT_SHA='d61967fea8461408004bf4c9b229867b78b93a234cd65eb6660109ee2605155c'
SPLIT_SHA='b0837ba1ae1b30eec0dd33c48f2300809625a09437b2c161ac4576dad521fe6d'
CACHE_SHA='7ede6afdfc6948ba4da845309033edc700466d80fc91bb5921410b780e8133e0'
GENERATOR_SHA='fbba652deaf7f1dbdb3972799b00a010e5782eaedddab40f85d4154941e9a83f'
OLD_PREVIEW=ROOT/'development_previews/v4_slot_2k_step200_dev19_202609141941'
OLD_AUDIT=HERE/'single_directions/metadata_audit.json'

class GateError(ValueError):pass
def require(value,message):
    if not value:raise GateError(message)
def read(path):
    require(Path(path).is_file(),'Required complete artifact missing: '+str(path))
    return json.loads(Path(path).read_text())

# An evaluator input is metadata, never an already measured result. In particular,
# openface_v2.measure appends item metadata after parsing, so no measurement key may
# enter an item. Use an allowlist (not an incomplete list of known bad keys).
REQUIRED_INPUT_FIELDS=('case_id','subject','method','original_method','path','sha256','split','group','cohort','source_labels_12','target_labels_12')
OPTIONAL_INPUT_FIELDS=('seed','checkpoint','input_group','extra_driver_available','native_output_size')
INPUT_FIELDS=frozenset(REQUIRED_INPUT_FIELDS+OPTIONAL_INPUT_FIELDS)

def input_metadata(row):
    require(all(k in row for k in REQUIRED_INPUT_FIELDS),'Required input metadata missing')
    clean={k:copy.deepcopy(row[k]) for k in REQUIRED_INPUT_FIELDS+OPTIONAL_INPUT_FIELDS if k in row}
    check_input_metadata(clean)
    return clean

def check_input_metadata(item):
    require(set(item)<=INPUT_FIELDS and set(REQUIRED_INPUT_FIELDS)<=set(item),'Measurement or unknown field in evaluator input')
    require(item['split']=='dev','Only development inputs accepted')
    require(len(item['source_labels_12'])==len(item['target_labels_12'])==12,'Wrong manual AU labels')
    require(all(type(v) in (int,float) and 0<=v<=5 for v in item['source_labels_12']+item['target_labels_12']),'Invalid manual AU intensity')
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def clean_scheduler(value):
    copy=json.loads(json.dumps(value));copy['config']['_use_default_values']=sorted(copy['config'].get('_use_default_values',[]));return copy

def document_gate(result,settings,cases,expected_cases,reference_settings):
    """Pure metadata gate, also used with explicitly synthetic test fixtures."""
    require(result.get('status')=='complete' and result.get('test_data_used') is False,'Generation is incomplete or used test data')
    require(result.get('all_fixed_cases_and_seeds_retained') is True,'Fixed request completeness not recorded')
    require(cases==expected_cases,'Frozen dev19 document changed')
    require(cases.get('test_data_used') is False and cases.get('status')=='frozen_additional_development','Not the approved dev-only superset')
    require(cases.get('source_manifest_sha256')==PARENT_SHA and cases.get('subject_split_sha256')==SPLIT_SHA and cases.get('cache_sha256')==CACHE_SHA,'Parent/split/cache changed')
    require(cases.get('sampling_seeds')==[101] and len(cases.get('cases',[]))==19,'Need all19 cases and seed101')
    require(len({c['case_id'] for c in cases['cases']})==19,'Duplicate case')
    require(all(c['split']=='dev' and c['subject'] in ('SN004','SN012','SN017','SN018') for c in cases['cases']),'Test/non-dev subject is prohibited')
    for key in ('cache_sha256','case_manifest_sha256','preview_group','steps','guidance_scale','sampling_seeds','au_scale','ip_scale','public_assets','model_version','source_anchor','adapter_forward_calls_per_request'):
        require(settings.get(key)==reference_settings.get(key),'Changed generation setting: '+key)
    require(settings.get('test_data_used') is False and settings.get('formal_sota_evidence') is False,'Wrong result scope')
    methods=settings.get('methods',[]);require(len(methods)==4,'Exactly four complete step2000 structures required')
    variants=[m['effective_config']['variant'] for m in methods];require(set(variants)==set(VARIANTS) and len(set(variants))==4,'Missing/duplicate/wrong structure')
    require(type(settings.get('skip_baseline')) is bool,'Missing baseline policy')
    count=4 if settings['skip_baseline'] else 5
    require(result.get('cases')==19 and result.get('methods')==count and result.get('images')==19*count,'Wrong image/case/method count')
    for method in methods:
        v=method['effective_config']['variant'];digest=method['checkpoint_sha256']
        require(re.fullmatch('[0-9a-f]{64}',digest) is not None,'Invalid checkpoint SHA')
        require(method['name']==f'source_centered_{v}_step2000_{digest[:8]}','Wrong checkpoint step/name')
        require(method.get('diagnostic_training') is False and method['effective_config'].get('diagnostic') is False,'Diagnostic checkpoint not accepted')

def record_gate(records,settings,cases,source_rows,prompts):
    method_map={m['name']:LABELS[m['effective_config']['variant']] for m in settings['methods']}
    case_map={c['case_id']:c for c in cases['cases']};wanted=set((cid,m,101) for cid in case_map for m in method_map)
    if not settings['skip_baseline']:wanted|={(cid,'baseline',101) for cid in case_map}
    seen=set()
    for g in records:
        key=(g.get('case_id'),g.get('method'),g.get('seed'));require(key in wanted and key not in seen,'Missing/extra/duplicate generation record');seen.add(key)
        cid=g['case_id'];q=case_map[cid];s=source_rows[cid]
        require(g.get('status')=='complete' and g.get('split')=='dev' and g.get('subject')==q['subject'] and g.get('group')==q['group'],'Failed or wrong-role generation')
        require(g.get('source_aus_0_5')==s['source_labels_12'] and g.get('target_aus_0_5')==s['target_labels_12'],'AU condition mismatch')
        require(g.get('prompt')==prompts[cid],'Prompt changed')
    require(seen==wanted,'Not all19×four structures complete')
    order_names=([] if settings['skip_baseline'] else ['baseline'])+list(method_map)
    require([(r['case_id'],r['method'],r['seed']) for r in records]==[(c['case_id'],m,101) for c in cases['cases'] for m in order_names],'Frozen case/method order changed')
    return method_map

def validate(run):
    run=Path(run).resolve();require(run.is_dir() and run.is_relative_to(ROOT/'development_previews'),'Actual in-project complete development run required')
    require(sha(HERE/'policy.json')==POLICY_SHA and sha(REFERENCE)==REFERENCE_SHA and sha(CASEFILE)==CASE_SHA,'Frozen policy/reference/request changed')
    require(sha(OLD_AUDIT)=='105d4ef94c9c15096295b0c557de4eb320c89ce5016f667011e56809e4928f61','Sealed parent model audit changed')
    result=read(run/'result.json');settings=read(run/'settings.json');cases=read(run/'frozen_cases.json');reference_settings=read(OLD_PREVIEW/'settings.json')
    document_gate(result,settings,cases,read(CASEFILE),reference_settings)
    require(sha(run/'frozen_cases.json')==CASE_SHA and sha(run/'generation_source.py')==GENERATOR_SHA,'Unapproved generation source or case bytes')
    parent=Path(cases['source_manifest_path']).resolve();require(parent.is_relative_to(ROOT) and sha(parent)==PARENT_SHA,'Frozen parent changed')
    parentdoc=read(parent);require(cases['cases'][:12]==parentdoc['cases'] and parentdoc['sampling_seeds']==[101],'Original12 prefix/order changed')
    require(len(settings.get('development_manifest_binding',[]))==4 and all(x==reference_settings['development_manifest_binding'][0] for x in settings['development_manifest_binding']),'Generation parent-binding receipts changed')
    require(clean_scheduler(read(run/'runtime_scheduler.json'))==clean_scheduler(read(OLD_PREVIEW/'runtime_scheduler.json')),'Actual sampler changed')
    old=read(REFERENCE);source={r['case_id']:r for r in old['records'] if r['method']=='source_real'};base={r['case_id']:r for r in old['records'] if r['method']=='baseline'}
    targets={r['case_id']:r for r in old['records'] if r['method']=='target_real'}
    original_records=[json.loads(l) for l in (OLD_PREVIEW/'generation_records.jsonl').read_text().splitlines()];prompts={r['case_id']:r['prompt'] for r in original_records if r['method']=='baseline'}
    records=[json.loads(l) for l in (run/'generation_records.jsonl').read_text().splitlines()];method_map=record_gate(records,settings,cases,source,prompts)
    old_methods={m['variant']:m for m in read(OLD_AUDIT)['methods']};checks=[];schedules=[]
    for method in settings['methods']:
        v=method['effective_config']['variant'];oldmethod=old_methods[v]
        for key,value in method['effective_config'].items():
            if key not in ('checkpoint_every','validate_every'):require(oldmethod['effective_config'].get(key)==value,'Changed training configuration: '+key)
        require(set(method['effective_config'])==set(oldmethod['effective_config']),'Training config keys changed')
        require(method['model_config']==oldmethod['model_config'],'Model/readout/normalization configuration changed')
        paths={g['checkpoint'] for g in records if g['method']==method['name']};require(len(paths)==1,'Inconsistent checkpoint per method')
        checkpoint=Path(paths.pop()).resolve();require(checkpoint.is_relative_to(ROOT/'runs') and sha(checkpoint)==method['checkpoint_sha256'],'Checkpoint bytes/path mismatch')
        final=read(checkpoint.parent/'result.json');seal=read(checkpoint.with_suffix('.json'));bindings=read(checkpoint.parent/'bindings.json')
        require(final['status']=='complete' and final['global_step']==seal['step']==2000 and not final['test_data_used'],'Training not complete2000')
        require(final.get('diagnostic') is False and final.get('old_generated_images_or_metrics_used') is False,'Invalid or reused-old-results training')
        require(final['model_version']==seal['model_version']==settings['model_version'],'Checkpoint algorithm version mismatch')
        require(final['variant']==seal['variant']==v and final['model_config']==method['model_config'],'Training variant/config mismatch')
        require(seal['checkpoint_sha256']==method['checkpoint_sha256'] and seal['bindings_fingerprint']==fingerprint(bindings),'Checkpoint sidecar mismatch')
        expected_parent=Path(oldmethod['checkpoint']);require(final['parent_checkpoint']['sha256']==oldmethod['sha256'] and Path(final['parent_checkpoint']['path']).resolve()==expected_parent.resolve(),'Not resumed from sealed corresponding step200')
        require(sha(expected_parent)==oldmethod['sha256'] and bindings==read(expected_parent.parent/'bindings.json'),'Parent or training data/source/weights binding changed')
        train_config=read(checkpoint.parent/'configuration.json')
        for key,value in method['effective_config'].items():require(train_config.get(key)==value,'Generation seal differs from actual train config: '+key)
        # Only metadata/trace is read; no checkpoint tensor/model loading.
        trace=[json.loads(l) for l in (checkpoint.parent/'training_trace.jsonl').read_text().splitlines()]
        require([r['step'] for r in trace]==list(range(201,2001)),'Expected the full resumed1800-step trace')
        schedules.append([{k:r[k] for k in ('step','case_ids','timesteps','auxiliary_eligible')} for r in trace])
        require(bindings['development_subset']['manifest_sha256']==PARENT_SHA and bindings['split']==SPLIT_SHA and bindings['cache']==CACHE_SHA,'Actual training parent/split/cache mismatch')
        require(set(source)<=set(bindings['all_dev_case_ids']),'Generation contains cases outside frozen dev')
        checks.append({'variant':v,'method':method['name'],'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'parent_step200_sha256':oldmethod['sha256'],
                       'training_bindings_sha256':sha(checkpoint.parent/'bindings.json'),'training_result_sha256':sha(checkpoint.parent/'result.json'),'training_trace_sha256':sha(checkpoint.parent/'training_trace.jsonl')})
    require(all(x==schedules[0] for x in schedules),'Four structures have different resumed sample/timestep/auxiliary schedules')
    items=[]
    from PIL import Image
    for g in records:
        cid=g['case_id'];path=Path(g['output']).resolve();require(path.is_relative_to(run) and sha(path)==g['sha256'],'Generated output path/SHA mismatch')
        with Image.open(path) as im:im.load();require(im.mode=='RGB' and im.size==(512,512),'Expected intact native512 RGB output')
        for role,ref in [('source',source[cid]),('target',targets[cid])]:require(sha(run/cid/(role+'_real.png'))==ref['sha256']==sha(ref['path']),'Real source/target reference changed')
        if g['method']=='baseline':require(g['sha256']==base[cid]['sha256'],'Regenerated baseline differs from sealed reference');continue
        if g['group']=='no_edit':require(g['sha256']==base[cid]['sha256'],'Zero request is not exact sealed baseline output')
        items.append({**input_metadata(source[cid]),'method':method_map[g['method']],'original_method':g['method'],'path':str(path),'sha256':g['sha256'],'seed':101,'checkpoint':g['checkpoint'],
                      'input_group':'source_plus_text_manual_AU','extra_driver_available':False,'native_output_size':[512,512]})
    require(len(items)==76,'Need76 new model outputs')
    clean_items=[input_metadata(s) for s in source.values()]+items
    for item in clean_items:check_input_metadata(item)
    return {'run':str(run),'items':clean_items,'reference_records':[r for r in old['records'] if r['method'] in REFERENCES or r['method'].endswith('_real')],
            'policy_sha256':POLICY_SHA,'reference_results_sha256':REFERENCE_SHA,'case_manifest_sha256':CASE_SHA,'source_parent_sha256':PARENT_SHA,'test_used':False,
            'checkpoints':checks,'resumed_schedule_all_equal':True,'generated_count':76,'real_source_count':19,'logical_count':95,
            'generation_files':{name:sha(run/name) for name in ['result.json','settings.json','frozen_cases.json','generation_source.py','generation_records.jsonl','runtime_scheduler.json']},
            'status':'validated_complete_step2000','script_sha256':sha(__file__)}

def prepare(run,group):
    require(re.fullmatch('step2000_[a-z0-9_]+',group) is not None,'Use a new step2000_<name> group')
    output=HERE/group;require(not output.exists(),'Output group already exists; choose a new name')
    doc=validate(run);output.mkdir();refs=doc.pop('reference_records');save(output/'reference_records.json',{'records':refs,'policy_sha256':POLICY_SHA,'source_sha256':REFERENCE_SHA,'test_used':False})
    save(output/'items.json',doc);save(output/'items_seal.json',{'sha256':sha(output/'items.json'),'reference_records_sha256':sha(output/'reference_records.json'),'policy_sha256':POLICY_SHA})
    (output/'gate_snapshot.py').write_bytes(Path(__file__).read_bytes());return output
