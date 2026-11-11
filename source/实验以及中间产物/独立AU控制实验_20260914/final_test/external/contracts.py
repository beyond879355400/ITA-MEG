"""Final-test metadata contracts. Importing this module never opens images/models."""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
EXP=HERE.parent.parent
THESIS=EXP.parent.parent
AUS=[1,2,4,5,6,9,12,15,17,20,25,26]
SEEDS=[101,202,303]
IMPLEMENTATIONS={'magicface_native':'source_manual_au','faceadapter_native':'source_real_driver','liveportrait_native':'source_real_driver'}
COMMITS={'magicface_native':'bcaca61620450681233529d472c452091c25c92b',
         'faceadapter_native':'6e313885188e05740a7c40e12c1b9015c748bf08',
         'liveportrait_native':'9b294b3d0536135442ea73cb01e6cb3ca7029dd3'}
MF_PARAMETERS={'steps':50,'scheduler':'UniPCMultistepScheduler','guidance_scale':7.5,'image_guidance_scale':1.5,
               'prompt':'A close up of a person.','image_size':[512,512]}
FA_PARAMETERS={'branch':'reenactment','scheduler':'EulerDiscreteScheduler','steps':25,'guidance_scale':5.0,
               'crop_ratio':.81,'image_size':512,'condition_seed':999,'preload_final_components':True,
               'base_model_kind':'stable-diffusion-v1-5','detection_policy':'author_default'}


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def resolve(path):
    require(isinstance(path,str) and path,'Explicit nonempty path required')
    p=Path(path)
    return (p if p.is_absolute() else THESIS/p).resolve()


def executable_path(path):
    """Preserve a venv's interpreter symlink; resolving it can lose the venv."""
    require(isinstance(path,str) and path,'Explicit interpreter path required')
    p=Path(path)
    return (p if p.is_absolute() else THESIS/p).absolute()


def frame_metadata(frame):
    required=('status','path','sha256','width','height','mode','label_frame_1based','video_frame_zero_based','camera','original_video_sha256')
    require(all(k in frame for k in required),'Incomplete raw-frame metadata')
    require(frame['status']=='complete' and frame['mode']=='RGB','Raw frame not complete RGB')
    require(type(frame['width']) is int and type(frame['height']) is int and min(frame['width'],frame['height'])>0,'Invalid raw frame dimensions')
    require(type(frame['label_frame_1based']) is int and frame['label_frame_1based']>0 and frame['video_frame_zero_based']==frame['label_frame_1based']-1,'Frame index changed')
    for key in ('sha256','original_video_sha256'):require(re.fullmatch('[0-9a-f]{64}',frame[key]) is not None,'Invalid raw SHA')
    require(frame['camera'] in ('left','right'),'Unknown raw camera')
    result={k:copy.deepcopy(frame[k]) for k in required};result['path']=str(resolve(result['path']))
    return result


def validate_parameters(settings):
    impl=settings.get('implementation');require(impl in IMPLEMENTATIONS,'Unknown external implementation')
    require(settings.get('input_group')==IMPLEMENTATIONS[impl],'Incorrect native input group')
    native=settings.get('native_parameters');require(isinstance(native,dict) and native,'Native parameters not frozen')
    runtime=settings.get('runtime');require(isinstance(runtime,dict),'Explicit runtime required')
    require(runtime.get('repo',{}).get('commit')==COMMITS[impl],'Wrong official repo commit')
    resolve(runtime['repo']['path']);resolve(runtime['python_executable'])
    require(runtime.get('device') in ('cpu','mps') and runtime.get('dtype')=='float32','Only verified FP32 CPU/MPS runtime supported')
    if impl=='magicface_native':
        require(native==MF_PARAMETERS,'MagicFace settings differ from the verified native50 path')
        require(settings.get('seed_execution')=={'mode':'stochastic_each_seed'},'MagicFace must run every seed')
        ref=runtime.get('model_layout_seal',{});resolve(ref.get('path'))
        require(re.fullmatch('[0-9a-f]{64}',ref.get('sha256','')) is not None,'Missing MagicFace layout SHA')
    elif impl=='faceadapter_native':
        require(native==FA_PARAMETERS,'Face-Adapter settings differ from verified native25 path')
        require(settings.get('seed_execution')=={'mode':'stochastic_each_seed'},'Face-Adapter must run every seed')
        for key in ('checkpoint','base_model','vae_model'):resolve(runtime[key])
    else:
        require(runtime['device']=='cpu','LivePortrait final path is the verified CPU route')
        require(settings.get('seed_execution')=={'mode':'deterministic_alias','actual_seed':101},'LP must disclose one actual run and two aliases')
        require(native.get('animation_region')=='all' and type(native.get('flag_relative_motion')) is bool,'LP all mode and final relative setting must be explicit')
        resolve(runtime['checkpoint'])
    return impl


def project(method,settings,raw,protocol_sha,raw_sha):
    impl=validate_parameters(settings)
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',method) is not None,'Unsafe method key')
    require(raw.get('status')=='raw_frames_extracted' and raw.get('protocol_sha256')==protocol_sha,'Raw extraction not bound to frozen protocol')
    require(raw.get('sampling_seeds')==SEEDS and len(raw.get('cases',[]))==32,'Need32 raw cases and all3seeds')
    require(raw.get('case_count_unchanged') is True and raw.get('replacement_cases_selected') is False,'Raw case substitution forbidden')
    require(not raw.get('post_extraction_integrity_errors'),'Extraction integrity errors')
    physical=[];logical=[];ids=[]
    for raw_case in raw['cases']:
        cid=raw_case['case_id'];ids.append(cid)
        require(raw_case.get('split')=='test' and raw_case.get('status')=='complete','Only actual complete test records accepted')
        require(raw_case.get('target_pixels_passed_to_our_generator') is False,'Raw input role violation')
        inp=raw_case['generation_input'];s=frame_metadata(inp['source_image']);t=frame_metadata(raw_case['target_reference_image'])
        require(s['camera']==t['camera']==raw_case['camera'],'Source/driver camera mismatch')
        source_au=inp['source_au_vector_0_5'];target_au=inp['target_au_vector_0_5']
        require(len(source_au)==len(target_au)==12,'Full12 manual AU labels required')
        require(all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=5 for v in source_au+target_au),'Invalid manual AU units')
        actual_seeds=[101] if impl=='liveportrait_native' else SEEDS
        for seed in actual_seeds:
            job_id=cid+'_seed'+str(seed)
            require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',job_id) is not None,'Unsafe execution case id')
            row={'case_id':job_id,'request_case_id':cid,'subject':raw_case['subject'],'split':'test','seed':seed,
                 'source':s['path'],'source_sha256':s['sha256'],'source_width':s['width'],'source_height':s['height'],
                 'source_frame_1based':s['label_frame_1based'],'camera':s['camera'],'source_video_sha256':s['original_video_sha256'],
                 'source_au_vector_0_5':copy.deepcopy(source_au),'target_au_vector_0_5':copy.deepcopy(target_au),'au_order':AUS,
                 'input_group':settings['input_group'],'test_used':True,'method':method,
                 'execution_id':method+'/'+job_id,'protocol_sha256':protocol_sha,'raw_manifest_sha256':raw_sha}
            if impl=='magicface_native':
                row.update(au_delta_native=[b-a for a,b in zip(source_au,target_au)],native_inference_steps=native_step(settings),prompt=settings['native_parameters']['prompt'])
                require(not any(k.startswith('target') and k not in ('target_au_vector_0_5',) for k in row),'MF projection contains target pixels')
            else:
                row.update(target=t['path'],target_sha256=t['sha256'],target_width=t['width'],target_height=t['height'],
                           target_frame_1based=t['label_frame_1based'],target_role='real_observed_driver')
            physical.append(row)
        for seed in SEEDS:
            actual_seed=101 if impl=='liveportrait_native' else seed
            logical.append({'method':method,'subject':raw_case['subject'],'case_id':cid,'seed':seed,'split':'test',
                            'status':'not_started','output':None,'execution_started':False,'execution_id':method+'/'+cid+'_seed'+str(actual_seed),
                            'alias_of':None if seed==actual_seed else {'method':method,'case_id':cid,'seed':actual_seed}})
    require(len(set(ids))==32,'Duplicate final case')
    return {'kind':'final_test_external_projection','status':'frozen_projection','method':method,'implementation':impl,
            'protocol_sha256':protocol_sha,'raw_manifest_sha256':raw_sha,'settings_sha256':signature(settings),
            'test_used':True,'split':'test','case_ids':ids,'sampling_seeds':SEEDS,'cases':physical,'logical_records':logical,
            'planned_physical_executions':len(physical),'planned_logical_records':len(logical)}


def native_step(settings):return settings['native_parameters']['steps']


def update_index(projection,receipts,raw_path,terminal=False):
    rows=copy.deepcopy(projection['logical_records'])
    for row in rows:
        receipt=receipts.get(row['execution_id'])
        if receipt:
            row.update(status=receipt['status'],output=copy.deepcopy(receipt.get('output')),execution_started=bool(receipt.get('generation_started')))
            if receipt.get('error'):row['error']=copy.deepcopy(receipt['error'])
    successful=[r for r in rows if r['status']=='complete']
    finished=all(r['status'] in ('complete','failed') for r in rows)
    status='complete' if len(successful)==len(rows) else 'completed_with_failures' if finished or terminal else 'in_progress'
    return {'kind':'final_test_generation_index','status':status,
            'protocol_sha256':projection['protocol_sha256'],'raw_manifest':{'path':str(raw_path),'sha256':projection['raw_manifest_sha256']},
            'method_names':[projection['method']],'case_ids':projection['case_ids'],'sampling_seeds':SEEDS,'records':rows,
            'physical_executions':sum(bool(r.get('generation_started')) for r in receipts.values()),
            'physical_attempts':len(receipts),'planned_physical_executions':projection['planned_physical_executions'],
            'execution_receipts':list(receipts.values()),
            'unique_output_images':len({r['output']['sha256'] for r in successful}),
            'successful_logical_records':len(successful),'failed_logical_records':sum(r['status']=='failed' for r in rows),
            'all_fixed_cases_and_seeds_retained':True,'test_data_used':True,
            'deterministic_aliases_are_not_independent_samples':projection['implementation']=='liveportrait_native'}


def failed_run_index(projection,receipts,raw_path,error):
    """A run-level integrity failure cannot leave a usable complete index."""
    quarantined=copy.deepcopy(receipts)
    for receipt in quarantined.values():
        if receipt.get('status')=='complete':
            receipt['saved_output_unverified']=receipt.get('output')
            receipt.update(status='failed',output=None,error={'error_type':'RunLevelValidationFailure','error':str(error)})
    result=update_index(projection,quarantined,raw_path,terminal=True)
    result.update(status='completed_with_failures',run_integrity_failure=str(error),usable_for_evaluation=False)
    return result
