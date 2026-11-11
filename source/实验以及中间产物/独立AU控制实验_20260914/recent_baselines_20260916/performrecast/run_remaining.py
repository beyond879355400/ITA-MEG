#!/usr/bin/env python3
"""Fixed remaining3 smoke_dev4 cases; one CPU FP32 model load, no scoring/tuning."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime_v1'
EXPECTED_CASES=['fresh_v2_dev_SN012_s02549_t00086','fresh_v2_dev_SN017_s02034_t04507','fresh_v2_dev_SN018_s00885_t04693']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main(args):
    if os.environ.get('PERFORMRECAST_GPU_RUN_GO')!='1':
        raise RuntimeError('Parent compute gate absent')
    if args.output.exists():raise RuntimeError('Append-only output path required')
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Less than8GiB free')
    source_policy=ROOT/'smoke_policy.json'
    p=json.loads(source_policy.read_text())
    manifest=(ROOT/p['input_manifest']).resolve()
    if sha(manifest)!=p['input_manifest_sha256']:raise RuntimeError('Input manifest changed')
    cases=json.loads(manifest.read_text())['cases'][1:]
    if [x['case_id'] for x in cases]!=EXPECTED_CASES:raise RuntimeError('Remaining case order changed')
    repo=Path.home()/'Projects/github/youku-aigc/PerformRecast'
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=p['official_repo_commit']:
        raise RuntimeError('Official revision changed')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Official checkout not clean')
    receipts=[]
    for path in (ROOT/'download_receipts').glob('*.json'):
        d=json.loads(path.read_text())
        if d.get('status')=='complete' and 'files' in d.get('extraction',{}):receipts.append((path,d))
    path,weights=sorted(receipts,key=lambda x:x[0].name)[-1]
    for w in weights['files']+weights['extraction']['files']:
        if Path(w['path']).stat().st_size!=w['bytes'] or sha(w['path'])!=w['sha256']:
            raise RuntimeError('Weight binding changed')
    args.output.mkdir(parents=True,exist_ok=False)
    receipt={'kind':'performrecast_remaining3_smoke_v1','status':'preparing','case_order':EXPECTED_CASES,
             'input_manifest_sha256':sha(manifest),'single_case_policy_sha256':sha(source_policy),
             'driver_sha256':sha(Path(__file__)),'original_commit':p['official_repo_commit'],
             'weights_receipt_sha256':sha(path),'device':'cpu','dtype':'float32','onnx_provider':'CPUExecutionProvider',
             'mode':1,'smoothing':False,'seed':101,'frames_per_case':4,'fps':25,'selected_frame':0,
             'static_adaptation':'fixed original source/target images repeated4 times losslessly; not native continuous video benchmark',
             'effect_tuning':False,'model_loading_count':0,'cases':[],'started_utc':datetime.now(timezone.utc).isoformat(),
             'source_files':{str(x.relative_to(RUNTIME)):sha(x) for x in sorted(RUNTIME.rglob('*.py'))}}
    report=args.output/'receipt.json'
    started=time.monotonic()
    try:
        from PIL import Image
        bundle=manifest.parent.parent
        for case in cases:
            out=args.output/case['case_id'];out.mkdir()
            inp=out/'inputs';inp.mkdir()
            row={'case_id':case['case_id'],'status':'inputs_prepared','inputs':{},'output_dir':str(out)}
            for role in ('source','target'):
                original=bundle/case[role+'_path']
                if sha(original)!=case[role+'_sha256']:raise RuntimeError('Input image changed')
                image=Image.open(original).convert('RGB')
                copied=inp/(role+'.png');shutil.copy2(original,copied)
                video=inp/(role+'.mov')
                cmd=['ffmpeg','-nostdin','-v','error','-loop','1','-framerate','25','-i',str(original),'-frames:v','4','-c:v','png','-pix_fmt','rgb24','-an',str(video)]
                subprocess.run(cmd,check=True)
                decoded=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(video),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
                if decoded!=image.tobytes()*4:raise RuntimeError('Lossless pixel parity failed')
                row['inputs'][role]={'png':str(copied),'original_sha256':sha(original),'video':str(video),'video_sha256':sha(video),
                                     'width':image.width,'height':image.height,'all4_decoded_frames_exact':True}
            receipt['cases'].append(row)
        sys.dont_write_bytecode=True
        os.environ.update(PERFORMRECAST_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                          PYTORCH_ENABLE_MPS_FALLBACK='0',IMAGEIO_FFMPEG_EXE='/opt/homebrew/bin/ffmpeg')
        sys.path.insert(0,str(RUNTIME))
        import numpy as np
        import torch
        random.seed(101);np.random.seed(101);torch.manual_seed(101)
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        from src.pipeline import PerformRecastPipeline
        from src.config.argument_config import ArgumentConfig
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        tick=time.monotonic()
        pipeline=PerformRecastPipeline(inference_cfg=InferenceConfig(flag_use_half_precision=False),crop_cfg=CropConfig(flag_force_cpu=True))
        receipt.update(model_loading_count=1,model_loading_seconds=round(time.monotonic()-tick,3),status='running')
        native_parse=pipeline.wrapper.parse_output
        finite_calls=[]
        def checked_parse(out):
            if not torch.isfinite(out).all().item():raise FloatingPointError('Native generated tensor contains nonfinite values')
            finite_calls.append({'shape':list(out.shape),'min':float(out.min()),'max':float(out.max())})
            return native_parse(out)
        pipeline.wrapper.parse_output=checked_parse
        for row in receipt['cases']:
            if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Less than8GiB free')
            tick=time.monotonic();before=len(finite_calls);out=Path(row['output_dir'])
            native=ArgumentConfig(source=row['inputs']['source']['video'],driving=row['inputs']['target']['video'],output_dir=str(out),
                                  flag_use_half_precision=False,flag_force_cpu=True,flag_smooth=False,inference_mode=1)
            native.src_audio=native.source;native.drv_audio=native.driving
            native.output_path=str(out/'source__target.mp4')
            for key,folder in [('frames_dir','native_full_png'),('crop_output_dir','native_crop_png'),('concat_output_dir','native_concat_png')]:
                setattr(native,key,str(out/folder));(out/folder).mkdir()
            row['status']='running';report.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
            pipeline.execute(native)
            row['outputs']=[]
            for folder in ('native_crop_png','native_full_png','native_concat_png'):
                files=sorted((out/folder).glob('frame*.png'))
                if len(files)!=4:raise RuntimeError('Not4 native output PNGs')
                for image_path in files:
                    with Image.open(image_path) as image:
                        array=np.asarray(image)
                        expected=(512,512) if folder=='native_crop_png' else ((2048,512) if folder=='native_concat_png' else (row['inputs']['source']['width'],row['inputs']['source']['height']))
                        if image.format!='PNG' or image.size!=expected or image.mode!='RGB' or not np.isfinite(array).all():
                            raise RuntimeError('PNG format/size/finite gate failed')
                        row['outputs'].append({'path':str(image_path),'sha256':sha(image_path),'size':list(image.size),'mode':image.mode,'format':image.format,
                                               'pixel_min':int(array.min()),'pixel_max':int(array.max()),'pixel_std':float(array.std())})
            row.update(status='complete_pending_visual_review',seconds=round(time.monotonic()-tick,3),
                       finite_output_tensors=finite_calls[before:],selected_crop_png=str(out/'native_crop_png/frame0000.png'),
                       selected_full_png=str(out/'native_full_png/frame0000.png'))
            print(json.dumps({'case_id':row['case_id'],'status':row['status'],'seconds':row['seconds'],'selected_crop_png':row['selected_crop_png']}),flush=True)
            report.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        receipt['status']='complete_pending_visual_review'
        receipt['versions']={name:importlib.metadata.version(name) for name in ('torch','torchvision','timm','onnxruntime','numpy','pytorch-lightning')}
    except Exception as e:
        receipt.update(status='failed_outputs_preserved',error=type(e).__name__+': '+str(e))
        raise
    finally:
        receipt['seconds_total']=round(time.monotonic()-started,3)
        receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
        report.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    return {'status':receipt['status'],'receipt':str(report),'cases':[{'case_id':r['case_id'],'seconds':r['seconds']} for r in receipt['cases']]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();a.output=a.output.resolve()
    if not a.execute:print(json.dumps({'status':'plan_only','case_order':EXPECTED_CASES,'device':'cpu','frames':4,'selected_frame':0}))
    else:print(json.dumps(main(a),ensure_ascii=False,indent=2))
