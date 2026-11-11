#!/usr/bin/env python3
"""Bounded last-block adaptation after the frozen-head qualification failure.

Only layer4 and the AU head are fitted; all BN running statistics stay fixed.
The initial head is the preregistered diagnostic-best head from stage1.
"""
import argparse
import dataclasses
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
from align_real_frames import HERE,EXPERIMENT,SPLIT_SHA,environment,sha,write_json,copy_exact
from train_head_only import torch_save_new,rng_state,restore_rng


def freeze_except_layer4(model):
    for p in model.parameters():p.requires_grad_(False)
    for p in model.backbone.layer4.parameters():p.requires_grad_(True)
    for p in model.head.parameters():p.requires_grad_(True)


def training_mode_frozen_bn(model,torch):
    model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=HERE/'layer4_v1')
    ap.add_argument('--device',choices=['mps','cpu'],default='mps')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--stop-after-epoch',type=int,default=30,help='Scheduling pause only; total protocol remains 30 epochs')
    args=ap.parse_args();environment();sys.path.insert(0,str(EXPERIMENT))
    import numpy as np
    from PIL import Image
    import torch
    from modeling.au_regressor import AUIntensityRegressor,AURegressorConfig,IMAGENET_RESNET18_SHA256,au_intensity_mse
    from supervisor_metrics import metrics,constant_baselines,supported_aus,qualify,AUS
    out=args.output.resolve()
    if HERE not in out.parents or not 1<=args.stop_after_epoch<=30:raise ValueError('Unsafe output/scheduling stop')
    splitfile=EXPERIMENT/'data_prepare/au_supervision_v2/subjects_frozen.json'
    if sha(splitfile)!=SPLIT_SHA:raise ValueError('Split changed')
    groups=json.loads(splitfile.read_text())['groups']
    manifestfile=HERE/'alignment_v1/aligned_manifest.json';manifest=json.loads(manifestfile.read_text());rows=manifest['samples']
    if manifest['subject_split_sha256']!=SPLIT_SHA or len(rows)!=840 or manifest['test_count']!=0:raise ValueError('Wrong alignment')
    for r in rows:
        if r['status']!='aligned' or r['split'] not in ('train','dev') or r['subject'] not in groups[r['split']] or r['subject'] in groups['test']:raise ValueError('Leakage or unresolved alignment')
        if sha(r['aligned_path'])!=r['aligned_sha256']:raise ValueError('Aligned input changed')
    parentfile=HERE/'head_only_v1/qualification.json';parent=json.loads(parentfile.read_text())
    if not parent['training_complete'] or parent['qualified'] or parent['subject_split_sha256']!=SPLIT_SHA or parent['test_used']:raise ValueError('Unexpected head stage')
    initial=parent['diagnostic_best']
    if initial['epoch']!=29 or sha(initial['checkpoint'])!=initial['sha256']:raise ValueError('Approved initial head changed')
    parent_checkpoint=torch.load(initial['checkpoint'],map_location='cpu',weights_only=True)
    if parent_checkpoint['binding_sha256']!=parent['binding_sha256']:raise ValueError('Parent checkpoint binding changed')
    train_rows=[r for r in rows if r['split']=='train'];dev_rows=[r for r in rows if r['split']=='dev']
    train_y=np.array([r['input']['au_vector_0_5'] for r in train_rows],dtype=np.float32)
    dev_y=np.array([r['input']['au_vector_0_5'] for r in dev_rows],dtype=np.float32)
    dev_ids=[r['sample_id'] for r in dev_rows];dev_subjects=[r['subject'] for r in dev_rows]
    support=supported_aus(dev_y);baselines=constant_baselines(train_y,dev_y,dev_subjects)
    if [AUS[i] for i in support]!=parent['predeclared_gate']['supported_au_numbers']:raise ValueError('Gate support changed')
    cfg=AURegressorConfig();weights=HERE/'models/resnet18-f37072fd.pth'
    if sha(weights)!=IMAGENET_RESNET18_SHA256:raise ValueError('ImageNet checkpoint changed')
    sources={'trainer':Path(__file__),'resume_helper':HERE/'train_head_only.py','metrics':HERE/'supervisor_metrics.py',
             'runtime_helper':HERE/'align_real_frames.py','model':EXPERIMENT/'modeling/au_regressor.py'}
    binding={'stage':'layer4_and_head','subject_split_sha256':SPLIT_SHA,'train_subjects':groups['train'],'dev_subjects':groups['dev'],'test_used':False,
             'alignment_manifest_sha256':sha(manifestfile),'parent_qualification_sha256':sha(parentfile),'initial_head_checkpoint_sha256':initial['sha256'],
             'initial_head_epoch':29,'architecture':dataclasses.asdict(cfg),'imagenet_weights_sha256':IMAGENET_RESNET18_SHA256,
             'sources':{k:{'path':str(v),'sha256':sha(v)} for k,v in sources.items()},'epochs':30,'batch_size':16,'seed':20260914,
             'device':args.device,'dtype':'float32','optimizer':{'name':'AdamW','layer4_lr':.0001,'head_lr':.0005,'weight_decay':.0001},
             'loss':'unchanged full12AU normalized MSE = rawMSE/25','batchnorm':'all running means/variances fixed eval; layer4 affine trainable',
             'image_preprocess':'identical modeling.au_regressor.preprocess_for_au(zero_one) inside model, on training device; full512 aligned RGB',
             'augmentation':'none','qualification_gate':parent['predeclared_gate'],
             'hypothesis':'generic frozen ImageNet features miss fine AU changes; adapt final spatial block while preserving earlier filters and BN statistics',
             'runtime':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','scipy','Pillow']}}
    if out.exists():
        if not args.resume or json.loads((out/'binding.json').read_text())!=binding:raise ValueError('Layer4 exact-resume mismatch')
    else:
        out.mkdir(parents=True);write_json(out/'binding.json',binding);write_json(out/'constant_baselines.json',baselines)
        write_json(out/'predeclared_gate.json',parent['predeclared_gate'])
        for key,p in sources.items():copy_exact(p,out/'snapshots'/p.name,binding['sources'][key]['sha256'])
        copy_exact(parentfile,out/'snapshots/parent_qualification.json',sha(parentfile))
    binding_sha=sha(out/'binding.json')
    # One decode per input; uint8 CPU buffers avoid repeated PNG I/O and retain original pixels.
    train_pixels=torch.from_numpy(np.stack([np.asarray(Image.open(r['aligned_path']).convert('RGB')) for r in train_rows]))
    dev_pixels=torch.from_numpy(np.stack([np.asarray(Image.open(r['aligned_path']).convert('RGB')) for r in dev_rows]))
    def batch_images(pixels,index):return pixels[index].permute(0,3,1,2).to(args.device,dtype=torch.float32)/255
    random.seed(20260914);np.random.seed(20260914);torch.manual_seed(20260914)
    if args.device=='mps':torch.mps.manual_seed(20260914)
    shuffle=torch.Generator(device='cpu').manual_seed(20260914)
    model=AUIntensityRegressor(cfg,imagenet_weights=weights).to(args.device)
    model.head.load_state_dict(parent_checkpoint['head_state_dict'],strict=True);freeze_except_layer4(model)
    optimizer=torch.optim.AdamW([{'params':list(model.backbone.layer4.parameters()),'lr':.0001},{'params':list(model.head.parameters()),'lr':.0005}],weight_decay=.0001)
    labels=torch.tensor(train_y,device=args.device)
    history=[];paths=sorted((out/'checkpoints').glob('epoch_*.pt')) if (out/'checkpoints').exists() else []
    for path in paths:
        c=torch.load(path,map_location='cpu',weights_only=True)
        if c['binding_sha256']!=binding_sha or c['epoch']!=len(history)+1:raise ValueError('Bad checkpoint sequence/binding')
        history.append({'epoch':c['epoch'],'metrics':c['dev_metrics'],'qualification':c['qualification'],'checkpoint':str(path),'sha256':sha(path),
                        'train_loss':c['train_loss'],'epoch_seconds':c['epoch_seconds']})
        report=out/'epochs'/f'epoch_{c["epoch"]:03d}.json'
        if not report.exists():write_json(report,{'epoch':c['epoch'],'train_loss':c['train_loss'],'epoch_seconds':c['epoch_seconds'],'dev_metrics':c['dev_metrics'],
                'qualification':c['qualification'],'predictions':[{'sample_id':i,'subject':s,'true_raw':y.tolist(),'predicted_raw':p.tolist()}
                           for i,s,y,p in zip(dev_ids,dev_subjects,c['dev_labels_raw'],c['dev_predictions_raw'])]})
    start=1
    if paths:
        model.load_state_dict(c['model_state_dict'],strict=True);optimizer.load_state_dict(c['optimizer_state_dict']);restore_rng(torch,np,shuffle,c['rng_state'],args.device)
        start=c['epoch']+1
    bn_before={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if any(s in n for s in ('running_mean','running_var','num_batches_tracked'))}
    for epoch in range(start,args.stop_after_epoch+1):
        tic=time.monotonic();training_mode_frozen_bn(model,torch);order=torch.randperm(len(train_rows),generator=shuffle);loss_sum=0
        for index in order.split(16):
            optimizer.zero_grad(set_to_none=True);images=batch_images(train_pixels,index)
            pred=model(images,input_range='zero_one');loss=au_intensity_mse(pred,labels[index.to(args.device)])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();optimizer.step();loss_sum+=float(loss.detach().cpu())*len(index)
            del images,pred,loss
        model.eval();predictions=[]
        with torch.inference_mode():
            for index in torch.arange(len(dev_rows)).split(16):predictions.append(model(batch_images(dev_pixels,index),input_range='zero_one').cpu())
        pred=torch.cat(predictions).numpy();result=metrics(dev_y,pred,dev_subjects);qualified=qualify(result,baselines,support)
        for n,v in bn_before.items():
            if not torch.equal(v,model.state_dict()[n].detach().cpu()):raise RuntimeError('BN running statistics unexpectedly updated')
        seconds=time.monotonic()-tic
        record={'epoch':epoch,'epoch_seconds':seconds,'binding_sha256':binding_sha,'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'optimizer_state_dict':optimizer.state_dict(),'rng_state':rng_state(torch,np,shuffle,args.device),'train_loss':loss_sum/len(train_rows),
                'dev_metrics':result,'qualification':qualified,'dev_predictions_raw':torch.tensor(pred),'dev_labels_raw':torch.tensor(dev_y),
                'dev_ids':dev_ids,'dev_subjects':dev_subjects}
        path=out/'checkpoints'/f'epoch_{epoch:03d}.pt';torch_save_new(torch,record,path)
        history.append({'epoch':epoch,'metrics':result,'qualification':qualified,'checkpoint':str(path),'sha256':sha(path),'train_loss':record['train_loss'],'epoch_seconds':seconds})
        write_json(out/'epochs'/f'epoch_{epoch:03d}.json',{'epoch':epoch,'train_loss':record['train_loss'],'epoch_seconds':seconds,'dev_metrics':result,'qualification':qualified,
                'predictions':[{'sample_id':i,'subject':s,'true_raw':y.tolist(),'predicted_raw':p.tolist()} for i,s,y,p in zip(dev_ids,dev_subjects,dev_y,pred)]})
        write_json(out/'progress.json',{'epoch':epoch,'total_epochs':30,'epoch_seconds':seconds,'train_loss':record['train_loss'],
                   'macro_mae':result['macro_mae'],'active_mae':result['active_mae'],'qualified':qualified['passed']},replace=True)
        print(f'layer4 epoch {epoch:02d} seconds={seconds:.2f} loss={record["train_loss"]:.6f} macro={result["macro_mae"]:.4f} active={result["active_mae"]:.4f} qualified={qualified["passed"]}',flush=True)
    complete=len(history)==30;eligible=[h for h in history if h['qualification']['passed']]
    key=lambda h:(h['metrics']['macro_mae'],h['metrics']['active_mae'],h['epoch'])
    selected=min(eligible,key=key) if complete and eligible else None;diagnostic=min(history,key=key);exported=None
    if selected:
        saved=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
        exported=out/'qualified_supervisor.pt'
        payload={'model_state_dict':saved['model_state_dict'],'config':dataclasses.asdict(cfg),'provenance':{'binding_sha256':binding_sha,'subject_split_sha256':SPLIT_SHA,
                 'trained_epoch':selected['epoch'],'test_used':False,'qualified':True,'selected_checkpoint_sha256':selected['sha256']}}
        if not exported.exists():torch_save_new(torch,payload,exported)
        else:
            old=torch.load(exported,map_location='cpu',weights_only=True)
            if old['provenance']!=payload['provenance']:raise ValueError('Export provenance changed')
            for k,v in payload['model_state_dict'].items():
                if not torch.equal(v,old['model_state_dict'][k]):raise ValueError('Export tensor changed')
    final={'qualified':bool(selected),'training_complete':complete,'stage':'layer4_and_head','checkpoint_path':str(exported) if exported else None,
           'checkpoint_sha256':sha(exported) if exported else None,'subject_split_sha256':SPLIT_SHA,'train_subjects':groups['train'],'dev_subjects':groups['dev'],
           'test_used':False,'architecture':dataclasses.asdict(cfg),'preprocess':parent['preprocess'],'imagenet_weights':parent['imagenet_weights'],
           'binding_sha256':binding_sha,'gate':(selected or diagnostic)['qualification'],'predeclared_gate':parent['predeclared_gate'],
           'selected_epoch':selected['epoch'] if selected else None,'selected_metrics':selected['metrics'] if selected else None,
           'diagnostic_best':diagnostic,'epochs_completed':len(history),'baseline_file':str(out/'constant_baselines.json')}
    write_json(out/'qualification.json',final,replace=args.resume);write_json(out/'history.json',history,replace=args.resume)
    if args.device=='mps':torch.mps.synchronize();torch.mps.empty_cache()
    print(json.dumps({'training_complete':complete,'qualified':bool(selected),'selected_epoch':final['selected_epoch'],
                      'diagnostic_epoch':diagnostic['epoch'],'epochs_completed':len(history)},indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
