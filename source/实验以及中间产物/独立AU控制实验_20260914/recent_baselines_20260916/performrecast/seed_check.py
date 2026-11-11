#!/usr/bin/env python3
"""Bounded dev1 seed/runtime parity check only; explicit root COMPUTE_GO required."""
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
    from formal_batch import sha as exact_sha
    return exact_sha(path)


def run(args):
    if os.environ.get('PERFORMRECAST_COMPUTE_GO') != '1':
        raise RuntimeError('Parent COMPUTE_GO gate absent; no models loaded')
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
    from formal_batch import weights_binding,hot_input,HOT_INPUTS
    for item in inputs['inputs'].values():
        item['original_clip_path']=item['clip']
        item['clip']=str(hot_input(item['clip'],item['clip_sha256']))
    bound_weights=weights_binding()
    receipt_path=Path(bound_weights['receipt_path'])
    assets=Path(bound_weights['asset_root'])
    baseline_receipt=Path(args.baseline).resolve().parents[1]/'receipt.json'
    old=json.loads(baseline_receipt.read_text())
    if old['case_id']!=policy['case_id'] or old['selected_crop_png']!=str(Path(args.baseline).resolve()):
        raise RuntimeError('Parity baseline is not the same predeclared dev1 case')
    old_output=next(x for x in old['outputs'] if x['path']==old['selected_crop_png'])
    baseline_hot=hot_input(args.baseline,old_output['sha256'])
    args.output.mkdir(parents=True,exist_ok=False)
    sys.dont_write_bytecode = True
    os.environ.update(PERFORMRECAST_DEVICE=args.device, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTORCH_ENABLE_MPS_FALLBACK='0', IMAGEIO_FFMPEG_EXE='/opt/homebrew/bin/ffmpeg')
    os.environ['PERFORMRECAST_DINO_CKPT']=str(assets/'vit_base_patch16_224.dino/pytorch_model.bin')
    sys.path.insert(0,str(RUNTIME))
    record = {'kind':'performrecast_dev1_seed_runtime_parity_v1','case_id':policy['case_id'],
              'started_utc':datetime.now(timezone.utc).isoformat(),'status':'starting',
              'original_commit':revision,'policy_sha256':sha(ROOT/'smoke_policy.json'),
              'input_receipt_sha256':sha(ROOT/'inputs_smoke_v1/manifest.json'),'input_hot_layout_sha256':sha(HOT_INPUTS),
              'download_receipt_sha256':sha(receipt_path),'hot_layout_sha256':bound_weights['hot_layout_sha256'],'device':args.device,'dtype':'float32',
              'onnx_provider':'CPUExecutionProvider','native_fp16_changed_for_mac':True,
              'native_frame_selection':0,'source_frame_count':4,'driving_frame_count':4,
              'inference_mode':1,'smoothing':False,'actual_seed':args.seed,'source_geometry_and_target_driver_used':True,
              'numeric_au_input':False,'effect_tuning':False,
              'source_files':{str(p.relative_to(RUNTIME)):sha(p) for p in sorted(RUNTIME.rglob('*.py'))}}
    started=time.monotonic()
    try:
        from import_io_compat import install
        record['import_io_compat']=install()
        import numpy as np
        import torch
        if args.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; explicit CPU run is required')
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
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
        config=InferenceConfig(flag_use_half_precision=False,
            checkpoint_F=str(assets/'performrecast/appearance_feature_extractor.pth'),
            checkpoint_M=str(assets/'performrecast/motion_extractor.pth'),
            checkpoint_W=str(assets/'performrecast/warping_module.pth'),
            checkpoint_G=str(assets/'performrecast/spade_generator.pth'))
        crop=CropConfig(flag_force_cpu=True,insightface_root=str(assets/'insightface'),landmark_ckpt_path=str(assets/'landmark.onnx'))
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
        record['versions']={p:importlib.metadata.version(p) for p in ('torch','torchvision','timm','onnxruntime','numpy','pytorch-lightning','Pillow','opencv-python','scipy')}
        previous=Path(args.baseline).resolve()
        with Image.open(baseline_hot) as a,Image.open(record['selected_crop_png']) as b:
            aa=np.asarray(a.convert('RGB')).astype(np.int16);bb=np.asarray(b.convert('RGB')).astype(np.int16)
            if aa.shape!=bb.shape:raise RuntimeError('Baseline shape differs')
            diff=np.abs(aa-bb)
        record['parity']={'baseline_path':str(previous),'baseline_hot_path':str(baseline_hot),'baseline_sha256':sha(baseline_hot),
                          'current_sha256':sha(Path(record['selected_crop_png'])),'pixels_equal':bool(np.array_equal(aa,bb)),
                          'max_absolute_pixel_difference':int(diff.max()),'mean_absolute_pixel_difference':float(diff.mean())}
        record['status']='complete_seed_parity_checked'
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
    p.add_argument('--device',choices=('cpu',),default='cpu')
    p.add_argument('--seed',type=int,choices=(101,202,303),default=202)
    p.add_argument('--baseline',type=Path,default=ROOT/'runs/smoke_cpu_v1/native_crop_png/frame0000.png')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();a.output=a.output.resolve()
    if not a.execute:
        print(json.dumps({'status':'plan_only','device':a.device,'seed':a.seed,'output':str(a.output),'case':'first frozen smoke_dev4','models_loaded':False}))
    else:
        print(json.dumps({'event':'seed_check_started','pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),'seed':a.seed,'models_loaded':False}),flush=True)
        print(json.dumps(run(a),ensure_ascii=False,indent=2))
