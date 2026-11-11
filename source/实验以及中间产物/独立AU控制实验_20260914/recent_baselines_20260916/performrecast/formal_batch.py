#!/usr/bin/env python3
"""Frozen32 PerformRecast cases. Prepare only by default; explicit COMPUTE_GO to run."""
import argparse
import collections
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from runtime_io_v1 import hash_exact
IO_HELPER=ROOT.parent/'runtime_io_v1.py'
HOT_INPUTS=ROOT/'hot_inputs_v1/layout_receipt.json'
RUNTIME=ROOT/'runtime_v1'
BUNDLE=ROOT.parent/'preservation/bundle_v2_20260916'
MANIFEST=BUNDLE/'inputs/postfreeze32.json'
MANIFEST_SHA='fa88ca09eb8253ce61ba046cb8d9b4c4affaef6cc421893cedb487c09522413b'
EVALUATION_PROTOCOL=ROOT.parent/'evaluation_20260916_v1/protocol_v1.json'
EVALUATION_PROTOCOL_SHA='281fb67d7bddeb3635d0bca2801307b4c546153efc56cab0cf20fc1ff201494f'
ORIGINAL=Path.home()/'Projects/github/youku-aigc/PerformRecast'
REVISION='0206abf3ba9e5b1eaa3cdc584879d5fa20d6a0b4'
AU_ORDER=[1,2,4,5,6,9,12,15,17,20,25,26]
RESERVE=8*1024**3
CASE_WORKSPACE=128*1024**2
PACKAGES=['torch','torchvision','numpy','onnxruntime','opencv-python','Pillow','timm','pytorch-lightning','torchmetrics','lightning-utilities','pykalman','imageio','imageio-ffmpeg','PyYAML','tyro']
EXPECTED_CORE={'torch':'2.1.1','torchvision':'0.16.1','numpy':'1.26.4','onnxruntime':'1.23.2',
               'timm':'1.0.19','pytorch-lightning':'2.1.4','torchmetrics':'1.3.2','lightning-utilities':'0.10.1'}


def sha(path):
    path=Path(path)
    return hash_exact(path,path.stat().st_size)['sha256']


def now():return datetime.now(timezone.utc).isoformat()


def new_json(path,data):
    """Publish immutable JSON atomically without replacing existing files."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.publishing-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(data,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
        os.link(name,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        os.unlink(name)


def read(path):
    path=Path(path);size=path.stat().st_size
    with path.open('rb') as stream:data=stream.read(size)
    if len(data)!=size:raise RuntimeError('Short JSON read: '+str(path))
    return json.loads(data)


def hot_input(path,digest):
    original=str(Path(path).resolve());layout=read(HOT_INPUTS)
    if layout['status']!='complete':raise RuntimeError('Hot input layout incomplete')
    matching=[x for x in layout['records'] if x['original_path']==original]
    if len(matching)!=1 or matching[0]['sha256']!=digest:raise RuntimeError('Hot input mapping differs')
    entry=matching[0];candidate=Path(entry['path'])
    if candidate.is_symlink() or candidate.parent!=ROOT/'hot_inputs_v1':raise RuntimeError('Hot input must be private byte copy')
    hash_exact(candidate,entry['bytes'],digest)
    return candidate


def new_attempt(folder):
    numbers=[]
    for p in folder.glob('attempt_*'):
        tail=p.name.removeprefix('attempt_')
        if tail.isdigit():numbers.append(int(tail))
    attempt=folder/('attempt_'+str(max(numbers,default=0)+1).zfill(3))
    attempt.mkdir(exist_ok=False)
    return attempt


def ensure_free(required=CASE_WORKSPACE):
    free=shutil.disk_usage(ROOT).free
    if free<RESERVE+required:raise RuntimeError('RESOURCE_PAUSE:8GiB reserve plus upcoming case workspace unavailable')
    return free


def png_size(path):
    with Path(path).open('rb') as f:header=f.read(24)
    if header[:8]!=b'\x89PNG\r\n\x1a\n' or header[12:16]!=b'IHDR':raise RuntimeError('Not native PNG input')
    return list(struct.unpack('>II',header[16:24]))


def source_binding():
    revision=subprocess.check_output(['git','-C',str(ORIGINAL),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(ORIGINAL),'status','--porcelain'],text=True).strip()
    if revision!=REVISION or dirty:raise RuntimeError('Original author checkout changed')
    sources={str(p.relative_to(RUNTIME)):sha(p) for p in sorted(RUNTIME.rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.png','.pkl') and 'pretrained_weights' not in p.parts}
    smoke=read(ROOT/'runs/smoke_cpu_v1/receipt.json')
    for relative,digest in smoke['source_files'].items():
        if sources.get(relative)!=digest:raise RuntimeError('Runtime differs from accepted smoke source: '+relative)
    return {'original_commit':revision,'adapted_sources':sources,'driver_sha256':sha(Path(__file__)),'io_helper_sha256':sha(IO_HELPER),
            'import_io_compat_sha256':sha(ROOT/'import_io_compat.py'),'hot_python_layout_sha256':sha(ROOT/'hot_python_v1/layout_receipt_v2.json')}


def cases_binding():
    if sha(MANIFEST)!=MANIFEST_SHA:raise RuntimeError('Frozen32 input manifest changed')
    d=read(MANIFEST);old=read(BUNDLE/'assets/completion_20260915/protocol/proposal_v1/cases_label_only.json')
    if d['case_count']!=32 or d['au_order']!=AU_ORDER or len(d['cases'])!=32:raise RuntimeError('Expected original32 cases')
    if [x['case_id'] for x in d['cases']]!=[x['case_id'] for x in old['cases']]:raise RuntimeError('Case order changed')
    if collections.Counter(x['group'] for x in d['cases'])!={'source_noedit':8,'nonzero_target':24}:raise RuntimeError('Edit/noedit counts differ')
    subjects=collections.Counter(x['subject'] for x in d['cases'])
    if len(subjects)!=8 or set(subjects.values())!={4}:raise RuntimeError('Subject balance changed')
    cases=[];unique={}
    for current,original in zip(d['cases'],old['cases']):
        for key in ('case_id','subject','camera','source_au_vector_0_5','target_au_vector_0_5','au_order'):
            if current[key]!=original[key]:raise RuntimeError('Original case field differs: '+key)
        case=dict(current)
        if case['sampling_seeds']!=[101,202,303]:raise RuntimeError('Original seed metadata changed')
        for role in ('source','target'):
            path=(BUNDLE/case[role+'_path']).resolve()
            if BUNDLE.resolve() not in path.parents:raise RuntimeError('Input escapes bundle')
            hot=hot_input(path,case[role+'_sha256'])
            case[role+'_absolute_path']=str(path);case[role+'_size']=png_size(hot)
            unique[str(path)]=hot.stat().st_size
        cases.append(case)
    return cases,{'unique_pngs':len(unique),'unique_png_bytes':sum(unique.values()),'subjects':dict(subjects)}


def weights_binding():
    complete=[]
    for path in (ROOT/'download_receipts').glob('*.json'):
        data=read(path)
        if data.get('status')=='complete' and 'files' in data.get('extraction',{}):complete.append((path,data))
    if not complete:raise RuntimeError('Checked author weight receipt missing')
    path,data=sorted(complete,key=lambda x:x[0].name)[-1]
    original_entries=data['files']+data['extraction']['files']
    layout_path=ROOT/'hot_assets_v2/layout_receipt.json'
    if not layout_path.exists():raise RuntimeError('Verified materialized hot layout required')
    layout=read(layout_path)
    if layout['status']!='complete' or layout['old_receipt_sha256']!=sha(path):raise RuntimeError('Hot layout not bound to original valid-run receipt')
    by_original={e['path']:e for e in original_entries}
    entries=layout['records']
    if len(entries)!=len(original_entries) or set(e['original_path'] for e in entries)!=set(by_original):raise RuntimeError('Hot layout asset list differs')
    for entry in entries:
        original=by_original[entry['original_path']]
        if entry['sha256']!=original['sha256'] or entry['bytes']!=original['bytes']:raise RuntimeError('Restoration must keep original hash and size')
        p=Path(entry['path'])
        if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise RuntimeError('Weight hash/size mismatch: '+str(p))
    return {'receipt_path':str(path),'receipt_sha256':sha(path),'files':entries,
            'hot_layout_sha256':sha(layout_path),'asset_root':str(ROOT/'hot_assets_v2')}


def python_binding(python):
    probe='import importlib.metadata as m,json,sys; print(json.dumps({"executable":sys.executable,"version":sys.version,"packages":{p:m.version(p) for p in '+repr(PACKAGES)+'}}))'
    result=json.loads(subprocess.check_output([str(python),'-B','-c',probe],text=True))
    for key,expected in EXPECTED_CORE.items():
        if result['packages'][key].split('+')[0]!=expected:raise RuntimeError('Accepted smoke runtime version differs: '+key)
    return result


def prepare(run_dir,python):
    if run_dir.exists():raise RuntimeError('New formal directory required; never overwrite')
    if sha(EVALUATION_PROTOCOL)!=EVALUATION_PROTOCOL_SHA:raise RuntimeError('Unified evaluation protocol differs')
    cases,audit=cases_binding();sources=source_binding();weights=weights_binding();runtime=python_binding(python)
    protocol={'kind':'performrecast_fixed32_v1','prepared_utc':now(),'manifest_path':str(MANIFEST),
              'manifest_sha256':MANIFEST_SHA,'data_role':'previously_exposed_subject_expanded_cases',
              'evaluation_protocol_path':str(EVALUATION_PROTOCOL),'evaluation_protocol_sha256':EVALUATION_PROTOCOL_SHA,
              'input_hot_layout_sha256':sha(HOT_INPUTS),'input_storage_only_change':True,
              'case_count':32,'edited':24,'noedit':8,'case_order':[c['case_id'] for c in cases],
              'case_audit':audit,'cases':cases,'sources':sources,'weights':weights,'python':runtime,
              'method':{'mode':1,'smoothing':False,'device':'cpu','dtype':'float32','torch_threads':4,'torch_interop_threads':1,
                        'seed':101,'actual_runs_per_case':1,'logical_seed_aliases':False,'video_frames':4,'video_fps':25,
                        'input_codec':'PNG lossless MOV','driver':'real target image repeated4 times',
                        'input_group':'real_image_driving_not_numeric_AU','main_output':'native_crop_png/frame0000.png',
                        'full_output':'display_archive_only','crop_scale':1.5,'crop_vy_ratio':-0.125,
                        'output_selection':'fixed frame0, no score based selection','all_original_frames_retained':True,
                        'adaptation':'static4-frame packaging and smoothing disabled; not original continuous-video benchmark'},
              'failure_policy':'retain failed case in32 denominators, no replacement, default resume skips terminal success/failure',
              'disk_reserve_bytes':RESERVE,'workspace_per_case_guard_bytes':CASE_WORKSPACE,'models_loaded':False}
    run_dir.mkdir(parents=True,exist_ok=False)
    new_json(run_dir/'frozen_protocol.json',protocol)
    print(json.dumps({'status':'prepared_not_run','protocol':str(run_dir/'frozen_protocol.json'),'sha256':sha(run_dir/'frozen_protocol.json'),'audit':audit}),flush=True)


def validate_complete(record,protocol_sha,case=None):
    if record['protocol_sha256']!=protocol_sha:raise RuntimeError('Case belongs to different protocol')
    if case is not None and any(record[k]!=case[k] for k in ('case_id','subject','group')):raise RuntimeError('Terminal case identity differs')
    if record['status']=='complete':
        expected=Path(record['attempt'])/'native_crop_png/frame0000.png'
        if record['output']!=str(expected) or sha(expected)!=record['output_sha256']:raise RuntimeError('Selected native frame0 binding changed')
        for output in record['outputs']:
            path=Path(output['path'])
            if Path(record['attempt']) not in path.parents:raise RuntimeError('Output path escapes its attempt')
            if not path.is_file() or sha(path)!=output['sha256']:raise RuntimeError('Completed output changed; do not silently regenerate')
    elif record['status']!='failed':raise RuntimeError('Unrecognized terminal case status')


def prepare_case(case,attempt):
    from PIL import Image
    inputs=attempt/'inputs';inputs.mkdir()
    receipt={}
    for role in ('source','target'):
        original=hot_input(case[role+'_absolute_path'],case[role+'_sha256'])
        ensure_free(original.stat().st_size+CASE_WORKSPACE)
        copied=inputs/(role+'.png');shutil.copy2(original,copied)
        image=Image.open(copied).convert('RGB');clip=inputs/(role+'.mov')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-loop','1','-framerate','25','-i',str(copied),'-frames:v','4','-c:v','png','-pix_fmt','rgb24','-an',str(clip)],check=True)
        decoded=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(clip),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
        if decoded!=image.tobytes()*4:raise RuntimeError('Static clip is not lossless4-frame RGB parity')
        receipt[role]={'png':str(copied),'original_path':case[role+'_absolute_path'],'hot_path':str(original),'original_sha256':sha(copied),'clip':str(clip),'clip_sha256':sha(clip),'size':list(image.size),'decoded4_frames_exact':True}
    new_json(inputs/'manifest.json',receipt)
    return receipt


def output_records(attempt,source_size):
    import numpy as np
    from PIL import Image
    records=[]
    sizes={'native_crop_png':(512,512),'native_full_png':tuple(source_size),'native_concat_png':(2048,512)}
    for folder,size in sizes.items():
        files=sorted((attempt/folder).glob('frame*.png'))
        if [p.name for p in files]!=['frame0000.png','frame0001.png','frame0002.png','frame0003.png']:raise RuntimeError('Native frame list differs')
        for path in files:
            with Image.open(path) as image:
                pixels=np.asarray(image)
                if image.format!='PNG' or image.mode!='RGB' or image.size!=size or not np.isfinite(pixels).all():raise RuntimeError('Output format/size/finite gate failed')
                records.append({'path':str(path),'sha256':sha(path),'size':list(size),'mode':'RGB','format':'PNG','min':int(pixels.min()),'max':int(pixels.max()),'std':float(pixels.std())})
    return records


def run(run_dir,resume):
    if os.environ.get('PERFORMRECAST_COMPUTE_GO')!='1':raise RuntimeError('COMPUTE_GO absent; no models loaded')
    protocol_path=run_dir/'frozen_protocol.json';protocol=read(protocol_path);protocol_sha=sha(protocol_path)
    if sha(EVALUATION_PROTOCOL)!=protocol['evaluation_protocol_sha256']:raise RuntimeError('Unified evaluation protocol changed')
    if sha(HOT_INPUTS)!=protocol['input_hot_layout_sha256']:raise RuntimeError('Hot input byte mapping changed')
    if source_binding()!=protocol['sources']:raise RuntimeError('Frozen code differs')
    runtime=python_binding(Path(sys.executable))
    if runtime!=protocol['python']:raise RuntimeError('Frozen runtime differs; resume blocked')
    cases,audit=cases_binding()
    if cases!=protocol['cases'] or audit!=protocol['case_audit']:raise RuntimeError('Frozen cases differ')
    if weights_binding()!=protocol['weights']:raise RuntimeError('Frozen weights differ')
    with (run_dir/'.compute.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run_locked(run_dir,resume,protocol,protocol_sha,cases)


def run_locked(run_dir,resume,protocol,protocol_sha,cases):
    terminal={}
    for case in cases:
        path=run_dir/'cases'/case['case_id']/'terminal.json'
        if path.exists():
            if not resume:raise RuntimeError('Existing case outputs require --resume')
            record=read(path);validate_complete(record,protocol_sha,case);terminal[case['case_id']]=record
    pending=[c for c in cases if c['case_id'] not in terminal]
    session={'kind':'performrecast32_session','started_utc':now(),'protocol_sha256':protocol_sha,'resume':resume,'already_terminal':len(terminal),'cases':[]}
    session_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session_path=run_dir/'sessions'/(session_id+'.json')
    if pending:
        ensure_free()
        from import_io_compat import install
        session['import_io_compat']=install()
        os.environ.update(PERFORMRECAST_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',IMAGEIO_FFMPEG_EXE='/opt/homebrew/bin/ffmpeg')
        sys.dont_write_bytecode=True;sys.path.insert(0,str(RUNTIME))
        import numpy as np
        import torch
        random.seed(101);np.random.seed(101);torch.manual_seed(101)
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        from src.pipeline import PerformRecastPipeline
        from src.config.argument_config import ArgumentConfig
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        tick=time.monotonic()
        assets=Path(protocol['weights']['asset_root'])
        os.environ['PERFORMRECAST_DINO_CKPT']=str(assets/'vit_base_patch16_224.dino/pytorch_model.bin')
        pipeline=PerformRecastPipeline(
            inference_cfg=InferenceConfig(flag_use_half_precision=False,
                checkpoint_F=str(assets/'performrecast/appearance_feature_extractor.pth'),
                checkpoint_M=str(assets/'performrecast/motion_extractor.pth'),
                checkpoint_W=str(assets/'performrecast/warping_module.pth'),
                checkpoint_G=str(assets/'performrecast/spade_generator.pth')),
            crop_cfg=CropConfig(flag_force_cpu=True,insightface_root=str(assets/'insightface'),landmark_ckpt_path=str(assets/'landmark.onnx')))
        session['model_load_seconds']=round(time.monotonic()-tick,3)
        parse=pipeline.wrapper.parse_output;finite=[]
        def checked_parse(out):
            if not torch.isfinite(out).all().item():raise FloatingPointError('Native generated tensor is nonfinite')
            finite.append({'shape':list(out.shape),'min':float(out.min()),'max':float(out.max())})
            return parse(out)
        pipeline.wrapper.parse_output=checked_parse
        try:
            for case in pending:
                ensure_free();folder=run_dir/'cases'/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
                attempt=new_attempt(folder)
                started=time.monotonic();before=len(finite)
                record={'case_id':case['case_id'],'subject':case['subject'],'group':case['group'],'protocol_sha256':protocol_sha,'attempt':str(attempt),'started_utc':now(),'status':'running','stage':'input_preparation'}
                new_json(attempt/'started.json',record)
                stop=threading.Event()
                def heartbeat():
                    while not stop.wait(30):print(json.dumps({'event':'heartbeat','case_id':case['case_id'],'seconds':round(time.monotonic()-started,1)}),flush=True)
                thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
                try:
                    inputs=prepare_case(case,attempt);record['inputs']=inputs
                    args=ArgumentConfig(source=inputs['source']['clip'],driving=inputs['target']['clip'],output_dir=str(attempt),flag_use_half_precision=False,flag_force_cpu=True,flag_smooth=False,inference_mode=1)
                    args.src_audio=args.source;args.drv_audio=args.driving;args.output_path=str(attempt/'source__target.mp4')
                    for key,name in [('frames_dir','native_full_png'),('crop_output_dir','native_crop_png'),('concat_output_dir','native_concat_png')]:
                        setattr(args,key,str(attempt/name));(attempt/name).mkdir()
                    record['stage']='native_inference'
                    pipeline.execute(args)
                    record['stage']='output_validation'
                    record['outputs']=output_records(attempt,case['source_size'])
                    record['finite_generated_tensors']=finite[before:]
                    if len(record['finite_generated_tensors'])!=4:raise RuntimeError('Expected4 generated tensors')
                    selected=attempt/'native_crop_png/frame0000.png'
                    record.update(status='complete',output=str(selected),output_sha256=sha(selected),actual_seed=101,seed_aliases=False)
                except Exception as e:
                    message=type(e).__name__+': '+str(e)
                    resource=isinstance(e,MemoryError) or any(s in message.lower() for s in ('out of memory','resource_pause','no space left'))
                    record.update(status='resource_interrupted' if resource else 'failed',error=message,traceback=traceback.format_exc(),output=None,output_sha256=None)
                    print(record['traceback'],flush=True)
                finally:
                    stop.set();thread.join(timeout=1)
                    record.update(finished_utc=now(),seconds=round(time.monotonic()-started,3))
                new_json(attempt/'result.json',record)
                if record['status']=='resource_interrupted':
                    session['cases'].append({'case_id':case['case_id'],'status':record['status'],'seconds':record['seconds']})
                    raise RuntimeError('Resource interruption preserved; same case remains pending for --resume')
                new_json(folder/'terminal.json',record)
                terminal[case['case_id']]=record;session['cases'].append({'case_id':case['case_id'],'status':record['status'],'seconds':record['seconds']})
                print(json.dumps({'event':'case_finished',**session['cases'][-1],'terminal_count':len(terminal),'planned':32},ensure_ascii=False),flush=True)
        finally:
            session['finished_utc']=now();new_json(session_path,session)
    rows=[]
    for case in cases:
        record=terminal.get(case['case_id'])
        if record is None:raise RuntimeError('Incomplete run; resume required')
        validate_complete(record,protocol_sha,case)
        rows.append({'case_id':case['case_id'],'subject':case['subject'],'group':case['group'],'source':case['source_absolute_path'],'source_sha256':case['source_sha256'],
                     'target_reference':case['target_absolute_path'],'target_sha256':case['target_sha256'],'source_aus_0_5':case['source_au_vector_0_5'],
                     'target_aus_0_5':case['target_au_vector_0_5'],'status':record['status'],'seed':101,'seed_alias':False,
                     'output':record['output'],'sha256':record['output_sha256'],'failure_stage':record.get('stage') if record['status']=='failed' else None,
                     'error':record.get('error'),'receipt_path':str(run_dir/'cases'/case['case_id']/'terminal.json')})
    index={'kind':'performrecast32_selected_frame_index','status':'complete' if all(r['status']=='complete' for r in rows) else 'completed_with_failures',
           'protocol_sha256':protocol_sha,'evaluation_protocol_sha256':protocol['evaluation_protocol_sha256'],
           'planned_cases':32,'successful_cases':sum(r['status']=='complete' for r in rows),'failed_cases':sum(r['status']=='failed' for r in rows),
           'case_level_deterministic_outputs':True,'seed_aliases':0,'selected_frame':0,'native_frames_per_case':4,'rows':rows,'created_utc':now()}
    if (run_dir/'generation_index.json').exists():
        old=read(run_dir/'generation_index.json')
        if old['rows']!=rows or old['protocol_sha256']!=protocol_sha:raise RuntimeError('Existing index differs')
    else:new_json(run_dir/'generation_index.json',index)
    print(json.dumps({'status':index['status'],'planned':32,'success':index['successful_cases'],'failure':index['failed_cases'],'index':str(run_dir/'generation_index.json')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('audit','prepare','run'))
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--python',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.action=='audit':
        cases,audit=cases_binding();print(json.dumps({'status':'input_audit_pass','manifest_sha256':MANIFEST_SHA,'audit':audit,'cases':[x['case_id'] for x in cases],'models_loaded':False},ensure_ascii=False,indent=2))
    elif args.action=='prepare':
        if args.run_dir is None or args.python is None:parser.error('prepare requires --run-dir and --python')
        # Preserve the venv executable path; resolving its symlink would bypass that venv.
        prepare(args.run_dir.resolve(),args.python.absolute())
    elif not args.execute:print(json.dumps({'status':'plan_only_no_models','action':'run','cases':32,'selected_frame':0,'seed_aliases':False}))
    else:
        if args.run_dir is None:parser.error('run requires --run-dir')
        run(args.run_dir.resolve(),args.resume)
