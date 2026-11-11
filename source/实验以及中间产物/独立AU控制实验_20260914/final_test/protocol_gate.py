"""Shared final-test control gate. JSON/model-file hashes only, never pixels.

The original extraction entry remains unchanged. This gate intentionally does
not re-open videos or source/target PNGs; image-consuming stages own their
explicit source-only/evaluation whitelists.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'data_prepare'));sys.dont_write_bytecode=True
from extract_final_test_frames import PRODUCTION,resolve_path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def valid_sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def validate_control(protocol_path,raw_manifest_path,expected_raw_sha,*,contract=PRODUCTION):
    protocol_path=Path(protocol_path).resolve()
    if not protocol_path.is_file():raise ValueError('Final frozen protocol is absent; no final-test stage may start')
    protocol=read_json(protocol_path)
    if protocol.get('kind')!='final_test_extraction_protocol' or protocol.get('status')!='frozen_final_test' or protocol.get('extraction_authorized') is not True:
        raise ValueError('Final-test protocol is not frozen/authorized')
    if contract.root.resolve() not in protocol_path.parents:raise ValueError('Protocol must belong to this experiment')
    checked={}
    def ref(reference,label,expected=None):
        if not isinstance(reference,dict) or not valid_sha(reference.get('sha256')):raise ValueError(label+' path/SHA is not frozen')
        path=resolve_path(reference.get('path'))
        if expected is not None and reference['sha256']!=expected:raise ValueError(label+' differs from accepted contract')
        if str(path) not in checked:checked[str(path)]=digest(path)
        if checked[str(path)]!=reference['sha256']:raise ValueError(label+' SHA mismatch')
        return path
    refs=protocol['inputs']
    plan_path=ref(refs['label_plan'],'label plan',contract.plan_sha256)
    split_path=ref(refs['subject_split'],'subject split',contract.split_sha256)
    inventory_path=ref(refs['label_inventory'],'label inventory',contract.inventory_sha256)
    acceptance_path=ref(refs['label_acceptance'],'label acceptance')
    sampling_path=ref(refs['sampling_settings'],'sampling settings')
    evaluation_path=ref(refs['evaluation_policy'],'evaluation policy')
    plan=read_json(plan_path);split=read_json(split_path);accepted=read_json(acceptance_path)
    case_ids=[row['case_id'] for row in plan['cases']]
    if tuple(plan['test_subjects'])!=contract.subjects or tuple(split['groups']['test'])!=contract.subjects or split.get('test_locked') is not True:
        raise ValueError('Test subjects changed')
    if len(case_ids)!=contract.case_count or len(set(case_ids))!=len(case_ids) or protocol.get('case_ids')!=case_ids:
        raise ValueError('Complete accepted case32 order is required')
    if accepted.get('status')!='label_only_plan_selected' or accepted.get('plan_sha256')!=contract.plan_sha256 or accepted.get('case_ids')!=case_ids:
        raise ValueError('Label acceptance mismatch')
    seeds=protocol.get('sampling_seeds')
    if seeds!=plan['sampling_seeds_proposed'] or not seeds or len(set(seeds))!=len(seeds) or any(type(seed) is not int for seed in seeds) or protocol.get('case_seed_policy')!='full_cartesian':
        raise ValueError('Complete accepted case/seed grid must be frozen')
    models={}
    for model in protocol.get('models',[]):
        method=model.get('method')
        if not isinstance(method,str) or not method or method in models or not model.get('artifacts'):raise ValueError('Final models/artifacts are not fixed')
        artifacts=[{'path':str(ref(a,'model '+method)),'sha256':a['sha256']} for a in model['artifacts']]
        models[method]={'method':method,'artifacts':artifacts}
    if not models:raise ValueError('Final model list is empty')
    sampling=read_json(sampling_path);evaluation=read_json(evaluation_path)
    if sampling.get('status')!='frozen_for_final_test' or sampling.get('sampling_seeds')!=seeds or set(sampling.get('method_settings',{}))!=set(models) or sampling.get('our_generator_uses_target_pixels') is not False:
        raise ValueError('Final sampling settings are incomplete')
    for method,settings in sampling['method_settings'].items():
        execution=settings.get('seed_execution',{})
        if execution.get('mode') not in ('stochastic_each_seed','deterministic_alias'):raise ValueError('Each method must explicitly declare seed execution')
        if execution['mode']=='deterministic_alias' and execution.get('actual_seed') not in seeds:raise ValueError('Deterministic actual seed is outside the frozen seed grid')
        models[method]['settings']=settings
    if evaluation.get('status')!='frozen_for_final_test' or set(evaluation.get('methods',[]))!=set(models) or not evaluation.get('metrics') or not evaluation.get('primary_family'):
        raise ValueError('Final evaluation policy/primary family is not frozen')
    if not evaluation.get('comparisons') or evaluation['primary_family'] not in {row.get('family') for row in evaluation['comparisons']}:
        raise ValueError('Primary comparison family is absent')
    for role,reference in protocol.get('execution_sources',{}).items():ref(reference,'execution source '+role)
    if not valid_sha(expected_raw_sha):raise ValueError('Explicit raw manifest SHA is required')
    raw_path=Path(raw_manifest_path).resolve()
    if contract.root.resolve() not in raw_path.parents:raise ValueError('Raw manifest is outside this experiment')
    if digest(raw_path)!=expected_raw_sha:raise ValueError('Raw manifest SHA mismatch')
    raw=read_json(raw_path);protocol_sha=digest(protocol_path)
    if raw.get('status')!='raw_frames_extracted' or raw.get('protocol_sha256')!=protocol_sha or raw.get('label_plan_sha256')!=contract.plan_sha256:
        raise ValueError('Raw extraction is incomplete or bound to another protocol/plan')
    if raw.get('extractor_sha256')!=protocol.get('extractor_sha256') or raw.get('legacy_helper_sha256')!=protocol.get('legacy_helper_sha256'):
        raise ValueError('Raw extractor source differs from the frozen protocol')
    raw_cases=raw.get('cases',[])
    if [row['case_id'] for row in raw_cases]!=case_ids or len(raw_cases)!=len(plan['cases']):raise ValueError('Raw case list/order changed')
    video_refs={row['subject']:row for row in protocol['videos']}
    for row,planned in zip(raw_cases,plan['cases']):
        if row.get('status')!='complete' or row.get('split')!='test' or row.get('subject')!=planned['subject'] or row.get('camera')!=planned['camera']:
            raise ValueError('Raw case identity/status mismatch')
        inputs=row['generation_input']
        if inputs.get('source_au_vector_0_5')!=planned['source_au_vector_0_5'] or inputs.get('target_au_vector_0_5')!=planned['target_au_vector_0_5'] or row.get('target_pixels_passed_to_our_generator') is not False:
            raise ValueError('Raw condition labels or target separation changed')
        for side,image in (('source',inputs['source_image']),('target',row['target_reference_image'])):
            # Metadata validation only: NEVER stat/open image.path here.
            if image.get('status')!='complete' or not valid_sha(image.get('sha256')) or image.get('label_frame_1based')!=planned[side+'_frame_1based'] or image.get('video_frame_zero_based')!=planned[side+'_frame_zero_based'] or image.get('camera')!=planned['camera']:
                raise ValueError('Raw image metadata/frame mapping changed')
            if image.get('original_video_sha256')!=video_refs[planned['subject']]['sha256']:raise ValueError('Raw image video binding changed')
    return {'protocol':protocol,'protocol_path':str(protocol_path),'protocol_sha256':protocol_sha,
        'sampling':sampling,'sampling_path':str(sampling_path),'sampling_sha256':checked[str(sampling_path)],
        'evaluation':evaluation,'evaluation_path':str(evaluation_path),'evaluation_sha256':checked[str(evaluation_path)],
        'raw_manifest':raw,'raw_manifest_path':str(raw_path),'raw_manifest_sha256':expected_raw_sha,
        'label_plan':plan,'label_plan_path':str(plan_path),'label_plan_sha256':contract.plan_sha256,
        'label_inventory_path':str(inventory_path),'label_inventory_sha256':contract.inventory_sha256,
        'models':models,'case_ids':case_ids,'sampling_seeds':seeds,'case_map':{row['case_id']:row for row in raw_cases},
        'verified_file_shas':checked,'pixels_opened_by_gate':False}


def verify_runtime_source(control,role,path):
    ref=control['protocol'].get('execution_sources',{}).get(role)
    path=Path(path).resolve()
    if not ref or resolve_path(ref['path'])!=path or ref['sha256']!=digest(path):raise ValueError('Runtime source not frozen for role: '+role)


def require_declared_artifact(control,reference,method=None):
    path=resolve_path(reference.get('path'));sha=reference.get('sha256')
    methods=[control['models'][method]] if method else control['models'].values()
    if not any(str(path)==item['path'] and sha==item['sha256'] for model in methods for item in model['artifacts']):
        raise ValueError('Checkpoint is not a declared frozen model artifact')
    return path
