#!/usr/bin/env python3
"""One fixed PerformRecast case only; explicit parent GPU gate before model use."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime_v1'
ORIGINAL = Path.home()/'Projects/github/youku-aigc/PerformRecast'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def run(args):
    if os.environ.get('PERFORMRECAST_GPU_RUN_GO') != '1':
        raise RuntimeError('Parent GPU_RUN_GO gate absent; no models loaded')
    if args.output.exists():
        raise RuntimeError('Output already exists; append-only run required')
    policy = json.loads((ROOT/'smoke_policy.json').read_text())
    revision = subprocess.check_output(['git','-C',str(ORIGINAL),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(ORIGINAL),'status','--porcelain'],text=True).strip()
    if revision != policy['official_repo_commit'] or dirty:
        raise RuntimeError('Original author checkout changed')
    inputs = json.loads((ROOT/'inputs_smoke_v1/manifest.json').read_text())
    if inputs['policy_sha256'] != sha(ROOT/'smoke_policy.json') or inputs['case_id'] != policy['case_id']:
        raise RuntimeError('Input policy binding changed')
    for item in inputs['inputs'].values():
        if sha(Path(item['clip'])) != item['clip_sha256']:
            raise RuntimeError('Lossless clip changed')
    complete = []
    for p in (ROOT/'download_receipts').glob('*.json'):
        d = json.loads(p.read_text())
        if d.get('status') == 'complete':
            complete.append((p,d))
    if not complete:
        raise RuntimeError('Complete checked official-weight receipt required')
    receipt_path, downloaded = sorted(complete, key=lambda p:p[0].name)[-1]
    for entry in downloaded['files'] + downloaded['extraction']['files']:
        p = Path(entry['path'])
        if p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise RuntimeError('Official weight binding changed: '+str(p))
    args.output.mkdir(parents=True,exist_ok=False)
    sys.dont_write_bytecode = True
    os.environ.update(PERFORMRECAST_DEVICE=args.device, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTORCH_ENABLE_MPS_FALLBACK='0', IMAGEIO_FFMPEG_EXE='/opt/homebrew/bin/ffmpeg')
    sys.path.insert(0,str(RUNTIME))
    record = {'kind':'performrecast_fixed_first_case_smoke_v1','case_id':policy['case_id'],
              'started_utc':datetime.now(timezone.utc).isoformat(),'status':'starting',
              'original_commit':revision,'policy_sha256':sha(ROOT/'smoke_policy.json'),
              'input_receipt_sha256':sha(ROOT/'inputs_smoke_v1/manifest.json'),
              'download_receipt_sha256':sha(receipt_path),'device':args.device,'dtype':'float32',
              'onnx_provider':'CPUExecutionProvider','native_fp16_changed_for_mac':True,
              'native_frame_selection':0,'source_frame_count':4,'driving_frame_count':4,
              'inference_mode':1,'smoothing':False,'source_geometry_and_target_driver_used':True,
              'numeric_au_input':False,'effect_tuning':False,
              'source_files':{str(p.relative_to(RUNTIME)):sha(p) for p in sorted(RUNTIME.rglob('*.py'))}}
    started=time.monotonic()
    try:
        import numpy as np
        import torch
        if args.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; explicit CPU run is required')
        random.seed(101); np.random.seed(101); torch.manual_seed(101)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        from src.config.argument_config import ArgumentConfig
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.pipeline import PerformRecastPipeline
        native = ArgumentConfig(source=inputs['inputs']['source']['clip'],driving=inputs['inputs']['target']['clip'],
                                output_dir=str(args.output),flag_use_half_precision=False,flag_force_cpu=True,
                                flag_smooth=False,inference_mode=1)
        native.src_audio=native.source; native.drv_audio=native.driving
        native.output_path=str(args.output/'source__target.mp4')
        native.frames_dir=str(args.output/'native_full_png')
        native.crop_output_dir=str(args.output/'native_crop_png')
        native.concat_output_dir=str(args.output/'native_concat_png')
        for name in ('frames_dir','crop_output_dir','concat_output_dir'):
            Path(getattr(native,name)).mkdir()
        config=InferenceConfig(flag_use_half_precision=False)
        crop=CropConfig(flag_force_cpu=True)
        pipeline=PerformRecastPipeline(inference_cfg=config,crop_cfg=crop)
        record['status']='running'
        (args.output/'receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        pipeline.execute(native)
        from PIL import Image
        record['outputs']=[]
        for folder in (Path(native.crop_output_dir),Path(native.frames_dir)):
            files=sorted(folder.glob('frame*.png'))
            if len(files)!=4:
                raise RuntimeError('Expected4 native output frames in '+str(folder))
            for p in files:
                with Image.open(p) as im:
                    record['outputs'].append({'path':str(p),'sha256':sha(p),'width':im.width,'height':im.height})
        record['selected_crop_png']=str(Path(native.crop_output_dir)/'frame0000.png')
        record['selected_full_png']=str(Path(native.frames_dir)/'frame0000.png')
        record['versions']={p:importlib.metadata.version(p) for p in ('torch','torchvision','timm','onnxruntime','numpy','pytorch-lightning')}
        record['status']='complete_pending_visual_review'
    except Exception as e:
        record.update(status='failed_outputs_preserved',error=type(e).__name__+': '+str(e))
        raise
    finally:
        record['seconds']=round(time.monotonic()-started,3)
        record['finished_utc']=datetime.now(timezone.utc).isoformat()
        (args.output/'receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('mps','cpu'),default='mps')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();a.output=a.output.resolve()
    if not a.execute:
        print(json.dumps({'status':'plan_only','device':a.device,'output':str(a.output),'case':'first frozen smoke_dev4','models_loaded':False}))
    else:
        print(json.dumps(run(a),ensure_ascii=False,indent=2))
