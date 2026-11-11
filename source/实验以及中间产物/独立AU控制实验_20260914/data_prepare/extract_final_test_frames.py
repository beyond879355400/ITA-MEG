"""Future final-test RAW frame extraction behind a complete frozen protocol.

Default production protocol does not yet exist. Missing/draft protocols stop
before any label/video inspection. No features, models, evaluation or generation.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PROJECT=ROOT.parent.parent
sys.dont_write_bytecode=True
from prepare_disfa_pilot import labels_for_subject,extract_frame
from plan_final_test_labels import avi_header_only,source_interval

AUS=(1,2,4,5,6,9,12,15,17,20,25,26)


@dataclass(frozen=True)
class Contract:
    root: Path
    label_root: Path
    video_root: Path
    subjects: tuple
    case_count: int
    plan_sha256: str
    inventory_sha256: str
    split_sha256: str


PRODUCTION=Contract(ROOT,Path.home()/'Desktop/Cursor/geme-main/DISFA/ActionUnit_Labels',Path.home()/'Desktop/Cursor/geme-main/DISFA',
    ('SN003','SN005','SN008','SN009','SN021','SN024','SN031','SN032'),32,
    'b2052f261a5caf81b444e5a9aaa2620c917b399904f7f404c7cf6fe2a04dc684',
    '2475a2e971922c3983321f74d7de18edffef8157cb4b0ea15b16685f53a3c8ab',
    'b0837ba1ae1b30eec0dd33c48f2300809625a09437b2c161ac4576dad521fe6d')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve_path(value):
    if not isinstance(value,str) or not value:raise ValueError('A frozen input path is missing')
    path=Path(value)
    return (path if path.is_absolute() else PROJECT/path).resolve()


def verify_ref(ref,label,expected=None,root=None):
    if not isinstance(ref,dict) or not isinstance(ref.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',ref['sha256']):
        raise ValueError(label+' SHA is missing; protocol is not complete')
    path=resolve_path(ref.get('path'))
    if root is not None and Path(root).resolve() not in path.parents:raise ValueError(label+' is outside the experiment')
    if expected is not None and ref['sha256']!=expected:raise ValueError(label+' differs from the accepted contract')
    if not path.is_file() or digest(path)!=ref['sha256']:raise ValueError(label+' input SHA mismatch')
    return path


def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def preflight(protocol_path,contract=PRODUCTION):
    protocol_path=Path(protocol_path).resolve()
    # This gate deliberately precedes all label/video reads.
    if not protocol_path.is_file():raise ValueError('Final frozen protocol does not exist; test extraction is blocked')
    protocol=read_json(protocol_path)
    if protocol.get('kind')!='final_test_extraction_protocol' or protocol.get('status')!='frozen_final_test' or protocol.get('extraction_authorized') is not True:
        raise ValueError('Final protocol is not frozen/authorized; test extraction is blocked')
    if contract.root.resolve() not in protocol_path.parents:raise ValueError('Protocol must be inside this experiment')
    if protocol.get('extractor_sha256')!=digest(__file__) or protocol.get('legacy_helper_sha256')!=digest(HERE/'prepare_disfa_pilot.py'):
        raise ValueError('Extraction implementation SHA is not frozen')
    refs=protocol['inputs']
    plan_path=verify_ref(refs.get('label_plan'),'Accepted label plan',contract.plan_sha256,contract.root)
    inventory_path=verify_ref(refs.get('label_inventory'),'Label inventory',contract.inventory_sha256,contract.root)
    split_path=verify_ref(refs.get('subject_split'),'Subject split',contract.split_sha256,contract.root)
    acceptance_path=verify_ref(refs.get('label_acceptance'),'Label acceptance',root=contract.root)
    sampling_path=verify_ref(refs.get('sampling_settings'),'Sampling settings',root=contract.root)
    evaluation_path=verify_ref(refs.get('evaluation_policy'),'Evaluation policy',root=contract.root)
    plan=read_json(plan_path);inventory=read_json(inventory_path);split=read_json(split_path);accepted=read_json(acceptance_path)
    cases=plan['cases'];ids=[case['case_id'] for case in cases]
    if tuple(plan['test_subjects'])!=contract.subjects or tuple(split['groups']['test'])!=contract.subjects or split.get('test_locked') is not True:
        raise ValueError('Frozen test subjects changed')
    if len(ids)!=contract.case_count or len(set(ids))!=len(ids) or protocol.get('case_ids')!=ids:
        raise ValueError('Complete accepted case list/order is not frozen')
    if accepted.get('status')!='label_only_plan_selected' or accepted.get('plan_sha256')!=contract.plan_sha256 or accepted.get('case_ids')!=ids:
        raise ValueError('Label acceptance does not cover this complete plan')
    seeds=protocol.get('sampling_seeds')
    if seeds!=plan['sampling_seeds_proposed'] or not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds))!=len(seeds) or protocol.get('case_seed_policy')!='full_cartesian':
        raise ValueError('Complete case/seed Cartesian protocol is not frozen')
    models=protocol.get('models',[])
    methods=[row.get('method') for row in models]
    if not methods or any(not isinstance(name,str) or not name for name in methods) or len(set(methods))!=len(methods):
        raise ValueError('Final model methods are not fixed')
    model_refs=[]
    for model in models:
        if not model.get('artifacts'):raise ValueError('Final model checkpoint artifacts are missing')
        for artifact in model['artifacts']:
            path=verify_ref(artifact,'Model checkpoint '+model['method'])
            model_refs.append({'method':model['method'],'path':str(path),'sha256':artifact['sha256']})
    sampling=read_json(sampling_path);evaluation=read_json(evaluation_path)
    if sampling.get('status')!='frozen_for_final_test' or sampling.get('sampling_seeds')!=seeds or set(sampling.get('method_settings',{}))!=set(methods) or sampling.get('our_generator_uses_target_pixels') is not False:
        raise ValueError('Final sampling settings or target-pixel boundary are incomplete')
    if any(not isinstance(value,dict) or not value for value in sampling['method_settings'].values()):raise ValueError('Empty method sampling settings')
    if evaluation.get('status')!='frozen_for_final_test' or set(evaluation.get('methods',[]))!=set(methods) or not evaluation.get('metrics') or not evaluation.get('primary_family'):
        raise ValueError('Final evaluation/primary family is not frozen')
    if not evaluation.get('comparisons') or evaluation['primary_family'] not in {row.get('family') for row in evaluation['comparisons']}:
        raise ValueError('Primary comparison family is missing')
    # Only now inspect original labels and original video content checksums.
    declared_labels={(row['subject'],row['au']):row for row in inventory['label_records']}
    labels={};label_receipts=[]
    for subject in contract.subjects:
        values,_,records=labels_for_subject(contract.label_root,subject);labels[subject]=values
        for record in records:
            declared=declared_labels.get((subject,record['au']))
            if declared is None or Path(declared['path']).resolve()!=Path(record['path']).resolve() or declared['sha256']!=record['sha256']:
                raise ValueError('Original manual labels changed')
            label_receipts.append(record)
        if source_interval(values,subject)!=plan['source_intervals'][subject]:raise ValueError('Accepted longest neutral source interval changed')
    video_specs=protocol.get('videos',[])
    if len(video_specs)!=len(contract.subjects) or {row.get('subject') for row in video_specs}!=set(contract.subjects):raise ValueError('Complete original-video checksums are missing')
    headers={(row['subject'],row['camera']):row for row in inventory['video_headers']}
    videos={}
    for spec in video_specs:
        subject=spec['subject'];subject_cases=[case for case in cases if case['subject']==subject]
        if not subject_cases:raise ValueError('Video has no accepted cases')
        cameras={case['camera'] for case in subject_cases};paths={case['video_path'] for case in subject_cases}
        if cameras!={spec.get('camera')} or len(paths)!=1 or resolve_path(spec.get('path'))!=resolve_path(next(iter(paths))):raise ValueError('Source and targets must retain the same accepted camera/video')
        video=verify_ref(spec,'Original video '+subject)
        if contract.video_root.resolve() not in video.parents:raise ValueError('Original video outside the declared raw dataset')
        old=headers[(subject,spec['camera'])]
        if spec.get('video_metadata_sha256')!=old['video_metadata_sha256'] or any(case['video_metadata_sha256']!=old['video_metadata_sha256'] for case in subject_cases):raise ValueError('Accepted video metadata binding changed')
        fresh=avi_header_only(video)
        if fresh['header']!=old['header'] or fresh['video_stream']!=old['video_stream'] or fresh['frames']!=len(labels[subject]):raise ValueError('Video header does not match labels; no frame shift may be guessed')
        videos[subject]={'path':str(video),'sha256':spec['sha256'],'camera':spec['camera'],'width':fresh['header']['width'],'height':fresh['header']['height'],
                         'frames':fresh['frames'],'planned_video_metadata_sha256':old['video_metadata_sha256'],'current_video_metadata_sha256':fresh['video_metadata_sha256']}
    for case in cases:
        subject=case['subject']
        if subject not in labels or case['split']!='test':raise ValueError('Unexpected case subject/role')
        for side in ('source','target'):
            frame=case[side+'_frame_1based']
            if type(frame) is not int or frame not in labels[subject] or case[side+'_frame_zero_based']!=frame-1:raise ValueError('Invalid exact label/video frame mapping')
            vector=[labels[subject][frame]['au'+str(au)] for au in AUS]
            if vector!=case[side+'_au_vector_0_5']:raise ValueError('Source/target labels changed')
    return {'protocol':protocol,'protocol_path':str(protocol_path),'protocol_sha256':digest(protocol_path),'plan':plan,'videos':videos,'label_receipts':label_receipts,
            'model_artifacts':model_refs,'sampling_settings_sha256':digest(sampling_path),'evaluation_policy_sha256':digest(evaluation_path),
            'frame_extraction_started':False,'models_loaded':False,'generation_requests_planned':len(ids)*len(seeds)*len(methods)}


def new_directory(parent,minute=None):
    parent=Path(parent);stamp=minute or datetime.now().strftime('%Y%m%d%H%M');index=1
    while True:
        path=parent/('raw_test_frames_'+stamp+('' if index==1 else '-'+str(index).zfill(2)))
        try:path.mkdir(parents=True,exist_ok=False);return path
        except FileExistsError:index+=1


def image_record(path,video,frame):
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        if image.mode!='RGB' or image.size!=(video['width'],video['height']):raise ValueError('Extracted frame is not full-size original RGB')
    return {'status':'complete','path':str(path),'sha256':digest(path),'width':video['width'],'height':video['height'],'mode':'RGB',
            'label_frame_1based':frame,'video_frame_zero_based':frame-1,'camera':video['camera'],'original_video_sha256':video['sha256']}


def execute(protocol_path,output_parent=None,contract=PRODUCTION,decoder=extract_frame):
    prepared=preflight(protocol_path,contract)
    parent=Path(output_parent) if output_parent is not None else contract.root/'data_prepare/final_test_extractions'
    if contract.root.resolve() not in parent.resolve().parents:raise ValueError('Output must remain in this experiment')
    output=new_directory(parent);start=time.monotonic()
    save_new(output/'protocol_snapshot.json',prepared['protocol']);save_new(output/'label_plan_snapshot.json',prepared['plan'])
    save_new(output/'preflight.json',{k:v for k,v in prepared.items() if k not in ('protocol','plan')})
    requests=sorted({(case['subject'],case[side+'_frame_1based']) for case in prepared['plan']['cases'] for side in ('source','target')})
    records={};failed=False
    with (output/'extraction_events.jsonl').open('x',buffering=1) as events:
        for subject,frame in requests:
            video=prepared['videos'][subject];path=output/'raw'/subject/video['camera']/f'frame_{frame:05d}.png'
            if failed:record={'status':'not_started','subject':subject,'label_frame_1based':frame,'reason':'earlier extraction failure; no replacement case'}
            else:
                try:
                    if path.exists():raise ValueError('Never overwrite an existing extracted frame')
                    decoder(Path(video['path']),frame,path)
                    record={'subject':subject,**image_record(path,video,frame)}
                except Exception as error:
                    failed=True;record={'status':'failed','subject':subject,'label_frame_1based':frame,'error_type':type(error).__name__,'error':str(error)}
            records[(subject,frame)]=record;events.write(json.dumps(record)+'\n')
    integrity_errors=[]
    checks=[('frozen protocol',protocol_path,prepared['protocol_sha256'])]
    checks.extend(('original video '+subject,video['path'],video['sha256']) for subject,video in prepared['videos'].items())
    checks.extend(('manual label '+str(label['path']),label['path'],label['sha256']) for label in prepared['label_receipts'])
    for name,path,expected in checks:
        try:
            if digest(path)!=expected:integrity_errors.append(name+' changed during extraction')
        except OSError as error:
            integrity_errors.append(name+' became unavailable: '+type(error).__name__)
    failed=failed or bool(integrity_errors)
    cases=[]
    for case in prepared['plan']['cases']:
        source=records[(case['subject'],case['source_frame_1based'])];target=records[(case['subject'],case['target_frame_1based'])]
        cases.append({'case_id':case['case_id'],'subject':case['subject'],'split':'test','camera':case['camera'],
            'status':'complete' if source['status']==target['status']=='complete' and not integrity_errors else 'incomplete',
            'generation_input':{'source_image':source,'source_au_vector_0_5':case['source_au_vector_0_5'],'target_au_vector_0_5':case['target_au_vector_0_5']},
            'target_reference_image':target,'target_pixels_passed_to_our_generator':False})
    result={'status':'failed' if failed else 'raw_frames_extracted','protocol_sha256':prepared['protocol_sha256'],'label_plan_sha256':contract.plan_sha256,
        'output_directory':str(output),'planned_cases':len(cases),'planned_unique_frames':len(requests),'successful_unique_frames':sum(r['status']=='complete' for r in records.values()),
        'case_count_unchanged':True,'replacement_cases_selected':False,'cases':cases,'sampling_seeds':prepared['protocol']['sampling_seeds'],
        'post_extraction_integrity_errors':integrity_errors,
        'features_computed':False,'models_called':False,'evaluation_run':False,'generation_run':False,'elapsed_seconds':time.monotonic()-start,
        'extractor_sha256':digest(__file__),'legacy_helper_sha256':digest(HERE/'prepare_disfa_pilot.py'),
        'ffmpeg_version':subprocess.check_output([shutil.which('ffmpeg'),'-version'],text=True).splitlines()[0]}
    save_new(output/'raw_frame_manifest.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,default=ROOT/'final_test/frozen_protocol.json');parser.add_argument('--dry-run',action='store_true');parser.add_argument('--output-parent',type=Path);args=parser.parse_args()
    try:
        if args.dry_run:
            result=preflight(args.protocol);print(json.dumps({'status':'frozen_protocol_verified','frame_extraction_started':False,'protocol_sha256':result['protocol_sha256']}));return 0
        result=execute(args.protocol,args.output_parent);print(json.dumps({k:result[k] for k in ('status','output_directory','planned_cases','successful_unique_frames')},ensure_ascii=False));return 0 if result['status']=='raw_frames_extracted' else 1
    except Exception as error:
        print(json.dumps({'status':'blocked','error_type':type(error).__name__,'error':str(error),'no_model_or_generation_called':True},ensure_ascii=False),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
