#!/usr/bin/env python3
"""Immutable batched ImageNet feature cache; no AU fitting or test frames."""
import argparse
import dataclasses
import importlib.metadata
import json
from pathlib import Path
import os
import sys
import time
from align_real_frames import HERE, EXPERIMENT, SPLIT_SHA, environment, sha, write_json, copy_exact


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=HERE/'features_v1')
    ap.add_argument('--device',choices=['mps','cpu'],default='mps')
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    environment()
    sys.path.insert(0,str(EXPERIMENT))
    import numpy as np
    from PIL import Image
    import torch
    from modeling.au_regressor import AUIntensityRegressor, AURegressorConfig, preprocess_for_au, IMAGENET_RESNET18_SHA256
    out=args.output.resolve()
    if HERE not in out.parents or args.batch_size<1: raise ValueError('Unsafe output/batch')
    manifest_path=HERE/'alignment_v1/aligned_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['subject_split_sha256']!=SPLIT_SHA or manifest['count']!=840 or manifest['test_count']!=0: raise ValueError('Wrong aligned manifest')
    rows=manifest['samples']
    if any(r['status']!='aligned' or r['split'] not in ('train','dev') for r in rows): raise ValueError('Unresolved alignment or test row')
    subjects=json.loads((EXPERIMENT/'data_prepare/au_supervision_v2/subjects_frozen.json').read_text())['groups']
    for r in rows:
        if r['subject'] not in subjects[r['split']] or r['subject'] in subjects['test']: raise ValueError('Subject leakage')
    weights=HERE/'models/resnet18-f37072fd.pth'
    copy_exact(Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth',weights,IMAGENET_RESNET18_SHA256)
    cfg=AURegressorConfig()
    binding={'split_sha256':SPLIT_SHA,'alignment_manifest_sha256':sha(manifest_path), 'imagenet_sha256':IMAGENET_RESNET18_SHA256,
             'model_module_sha256':sha(EXPERIMENT/'modeling/au_regressor.py'),'cache_script_sha256':sha(__file__),
             'architecture':dataclasses.asdict(cfg),'device':args.device,'dtype':'float32','batch_size':args.batch_size,
             'preprocess':'RGB tensor /255; preprocess_for_au(zero_one), bilinear224 align_corners=False antialias=False; ImageNet normalization',
             'backbone':'ImageNet ResNet18 fc=Identity; eval; all params frozen; no AU-trained weights', 'test_used':False,
             'runtime':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','Pillow']}}
    if out.exists():
        if not args.resume or json.loads((out/'binding.json').read_text())!=binding: raise ValueError('Cache exact-resume mismatch')
        if (out/'features_manifest.json').exists():
            finished=json.loads((out/'features_manifest.json').read_text())
            for chunk in finished['chunks']:
                if sha(chunk['path'])!=chunk['sha256']: raise ValueError('Completed chunk changed')
            print('Completed immutable feature cache verified; nothing rewritten.')
            return 0
    else:
        out.mkdir(parents=True)
        write_json(out/'binding.json',binding)
        copy_exact(__file__,out/'cache_script_snapshot.py',binding['cache_script_sha256'])
        copy_exact(EXPERIMENT/'modeling/au_regressor.py',out/'au_regressor_snapshot.py',binding['model_module_sha256'])
    device=torch.device(args.device)
    if args.device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    model=AUIntensityRegressor(cfg,imagenet_weights=weights).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    chunks=[]; started=time.monotonic()
    for start in range(0,len(rows),args.batch_size):
        batch=rows[start:start+args.batch_size]
        file=out/'chunks'/f'{start:04d}.pt'; file.parent.mkdir(exist_ok=True)
        ids=[r['sample_id'] for r in batch]
        for r in batch:
            if sha(r['aligned_path'])!=r['aligned_sha256']: raise ValueError('Aligned image changed: '+r['sample_id'])
        if file.exists():
            saved=torch.load(file,map_location='cpu',weights_only=True)
            if saved['sample_ids']!=ids or saved['binding_sha256']!=sha(out/'binding.json'): raise ValueError('Chunk mismatch')
        else:
            images=torch.stack([torch.from_numpy(np.asarray(Image.open(r['aligned_path']).convert('RGB')).copy()).permute(2,0,1).float()/255 for r in batch]).to(device)
            with torch.inference_mode():
                features=model.backbone(preprocess_for_au(images,input_range='zero_one',image_size=cfg.image_size)).cpu()
            if features.shape!=(len(batch),512) or not torch.isfinite(features).all(): raise RuntimeError('Bad features')
            saved={'features':features,'labels_raw':torch.tensor([r['input']['au_vector_0_5'] for r in batch],dtype=torch.float32),
                   'sample_ids':ids,'subjects':[r['subject'] for r in batch],'splits':[r['split'] for r in batch],
                   'binding_sha256':sha(out/'binding.json')}
            tmp=file.with_suffix('.pt.tmp'); torch.save(saved,tmp); os.replace(tmp,file)
            del images,features
        chunks.append({'path':str(file),'sha256':sha(file),'count':len(batch),'sample_ids':ids})
        write_json(out/'progress.json',{'processed':min(start+args.batch_size,len(rows)),'count':len(rows),'elapsed_seconds':time.monotonic()-started},replace=True)
        print(f'features {min(start+args.batch_size,len(rows))}/{len(rows)}',flush=True)
    if args.device=='mps': torch.mps.synchronize(); torch.mps.empty_cache()
    receipt={'status':'complete','count':840,'train_count':600,'dev_count':240,'test_count':0,'binding_sha256':sha(out/'binding.json'),
             'chunks':chunks,'elapsed_seconds':time.monotonic()-started,'imagenet_weights_path':str(weights)}
    write_json(out/'features_manifest.json',receipt,replace=args.resume)
    print(json.dumps({k:v for k,v in receipt.items() if k!='chunks'},indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
