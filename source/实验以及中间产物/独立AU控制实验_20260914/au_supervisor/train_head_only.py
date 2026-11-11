#!/usr/bin/env python3
"""Train the randomly initialized AU head on new train subjects only.

Epoch checkpoints are committed atomically before reporting. Resume restores the
last complete epoch, optimizer and all RNGs; an interrupted epoch is replayed.
The eight final-test subjects are never loaded. No layer4 training is implicit.
"""
import argparse
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
from align_real_frames import HERE, EXPERIMENT, SPLIT_SHA, environment, sha, write_json, copy_exact


def torch_save_new(torch, obj, path):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)


def rng_state(torch,np,shuffle,device):
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch':torch.get_rng_state(),'shuffle':shuffle.get_state(),
            'mps':torch.mps.get_rng_state() if device=='mps' else None}


def restore_rng(torch,np,shuffle,state,device):
    random.setstate(state['python'])
    n=state['numpy']; np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch']); shuffle.set_state(state['shuffle'])
    if device=='mps': torch.mps.set_rng_state(state['mps'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=HERE/'head_only_v1')
    ap.add_argument('--features',type=Path,default=HERE/'features_v1')
    ap.add_argument('--device',choices=['mps','cpu'],default='mps')
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--seed',type=int,default=20260914)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args(); environment(); sys.path.insert(0,str(EXPERIMENT))
    import numpy as np
    import torch
    from modeling.au_regressor import AUIntensityRegressor, AURegressorConfig, IMAGENET_RESNET18_SHA256, au_intensity_mse
    from supervisor_metrics import metrics,constant_baselines,supported_aus,qualify,AUS
    out=args.output.resolve(); cache=args.features.resolve()
    if HERE not in out.parents or HERE not in cache.parents or min(args.epochs,args.batch_size)<1: raise ValueError('Unsafe path/arguments')
    split_path=EXPERIMENT/'data_prepare/au_supervision_v2/subjects_frozen.json'
    if sha(split_path)!=SPLIT_SHA: raise ValueError('Split changed')
    groups=json.loads(split_path.read_text())['groups']
    feature_manifest=json.loads((cache/'features_manifest.json').read_text())
    feature_binding=json.loads((cache/'binding.json').read_text())
    if feature_binding['split_sha256']!=SPLIT_SHA or feature_manifest['test_count']!=0: raise ValueError('Wrong feature split')
    if sha(cache/'binding.json')!=feature_manifest['binding_sha256']: raise ValueError('Feature binding changed')
    if feature_binding['model_module_sha256']!=sha(EXPERIMENT/'modeling/au_regressor.py'): raise ValueError('Preprocessing/model changed')
    chunks=[]
    for chunk in feature_manifest['chunks']:
        if sha(chunk['path'])!=chunk['sha256']: raise ValueError('Feature chunk changed')
        c=torch.load(chunk['path'],map_location='cpu',weights_only=True)
        if c['binding_sha256']!=feature_manifest['binding_sha256']: raise ValueError('Chunk binding changed')
        chunks.append(c)
    features=torch.cat([c['features'] for c in chunks]); labels=torch.cat([c['labels_raw'] for c in chunks])
    ids=sum([c['sample_ids'] for c in chunks],[]); subjects=sum([c['subjects'] for c in chunks],[]); splits=sum([c['splits'] for c in chunks],[])
    if len(set(ids))!=840 or len(ids)!=840: raise ValueError('Duplicate/missing frames')
    for sub,split in zip(subjects,splits):
        if split not in ('train','dev') or sub not in groups[split] or sub in groups['test']: raise ValueError('Subject leakage')
    tm=torch.tensor([s=='train' for s in splits]); dm=~tm
    if int(tm.sum())!=600 or int(dm.sum())!=240: raise ValueError('Wrong split counts')
    train_y=labels[tm].numpy(); dev_y=labels[dm].numpy()
    dev_ids=[i for i,s in zip(ids,splits) if s=='dev']; dev_subjects=[i for i,s in zip(subjects,splits) if s=='dev']
    support=supported_aus(dev_y)  # GT-only, before any head prediction or update
    baselines=constant_baselines(train_y,dev_y,dev_subjects)
    gate={'supported_au_numbers':[AUS[i] for i in support], 'support_rule':'dev >=2 distinct raw label levels and >=10 active frames; chosen using labels only before predictions',
          'relative_improvement_over_best_constant_macro':.05,'relative_improvement_over_best_constant_active':.05,
          'minimum_positive_pearson_fraction':2/3,'minimum_median_pearson':.2,
          'constant_baselines':['train_mean','train_median','zero'], 'undefined_pearson_for_gate':0,
          'selection':'among qualified epochs, smallest dev macro_mae, then active_mae, then earliest epoch',
          'unqualified_policy':'save diagnostic best but prohibit generator AU-loss use',
          'interpretation':'engineering pilot qualification, not statistical significance or final test validation'}
    cfg=AURegressorConfig(); weights=HERE/'models/resnet18-f37072fd.pth'
    if sha(weights)!=IMAGENET_RESNET18_SHA256: raise ValueError('ImageNet changed')
    source_paths={'trainer':Path(__file__), 'metrics':HERE/'supervisor_metrics.py','runtime_helper':HERE/'align_real_frames.py','model':EXPERIMENT/'modeling/au_regressor.py'}
    binding={'subject_split_sha256':SPLIT_SHA,'train_subjects':groups['train'],'dev_subjects':groups['dev'],'test_used':False,
             'architecture':dataclasses.asdict(cfg),'imagenet_weights_sha256':IMAGENET_RESNET18_SHA256,
             'feature_manifest_sha256':sha(cache/'features_manifest.json'),'feature_binding_sha256':sha(cache/'binding.json'),
             'sources':{k:{'path':str(p),'sha256':sha(p)} for k,p in source_paths.items()},'stage':'head_only',
             'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.0001,'betas':[.9,.999],'eps':1e-8},
             'loss':'unweighted normalized AU MSE = raw0-5 MSE /25; complete true12AU labels',
             'epochs':args.epochs,'batch_size':args.batch_size,'seed':args.seed,'device':args.device,'dtype':'float32',
             'sampler':'CPU torch.randperm without replacement per epoch; independent saved generator',
             'frozen_backbone':'ImageNet ResNet18 eval, no BN updates, no parameter changes', 'qualification_gate':gate,
             'runtime':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','scipy']}}
    if out.exists():
        if not args.resume or json.loads((out/'binding.json').read_text())!=binding: raise ValueError('Training exact-resume mismatch')
    else:
        out.mkdir(parents=True)
        write_json(out/'binding.json',binding)
        write_json(out/'predeclared_gate.json',gate)
        write_json(out/'constant_baselines.json',baselines)
        for key,p in source_paths.items(): copy_exact(p,out/'snapshots'/p.name,binding['sources'][key]['sha256'])
        copy_exact(split_path,out/'snapshots/subjects_frozen.json',SPLIT_SHA)
    binding_sha=sha(out/'binding.json')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=='mps': torch.mps.manual_seed(args.seed)
    shuffle=torch.Generator(device='cpu').manual_seed(args.seed)
    model=AUIntensityRegressor(cfg,imagenet_weights=weights).to(args.device).eval()
    for p in model.backbone.parameters(): p.requires_grad_(False)
    model.head.train(); optimizer=torch.optim.AdamW(model.head.parameters(),lr=.001,weight_decay=.0001)
    train_x=features[tm].to(args.device); train_t=labels[tm].to(args.device);dev_x=features[dm].to(args.device)
    checkpoints=sorted((out/'checkpoints').glob('epoch_*.pt')) if (out/'checkpoints').exists() else []
    history=[]; start_epoch=1
    for path in checkpoints:
        c=torch.load(path,map_location='cpu',weights_only=True)
        if c['binding_sha256']!=binding_sha or c['epoch']!=len(history)+1: raise ValueError('Checkpoint sequence/binding mismatch')
        history.append({'epoch':c['epoch'],'metrics':c['dev_metrics'],'qualification':c['qualification'],'checkpoint':str(path),'sha256':sha(path),'train_loss':c['train_loss']})
        epoch_json=out/'epochs'/f'epoch_{c["epoch"]:03d}.json'
        if not epoch_json.exists():
            write_json(epoch_json,{'epoch':c['epoch'],'train_loss':c['train_loss'],'dev_metrics':c['dev_metrics'],'qualification':c['qualification'],
                       'predictions':[{'sample_id':i,'subject':s,'true_raw':y.tolist(),'predicted_raw':p.tolist()}
                                      for i,s,y,p in zip(c['dev_ids'],c['dev_subjects'],c['dev_labels_raw'],c['dev_predictions_raw'])]})
    if checkpoints:
        model.head.load_state_dict(c['head_state_dict'],strict=True);optimizer.load_state_dict(c['optimizer_state_dict'])
        restore_rng(torch,np,shuffle,c['rng_state'],args.device);start_epoch=c['epoch']+1
    started=time.monotonic()
    for epoch in range(start_epoch,args.epochs+1):
        model.head.train(); order=torch.randperm(len(train_x),generator=shuffle); loss_sum=0.0
        for index in order.split(args.batch_size):
            index=index.to(args.device);optimizer.zero_grad(set_to_none=True)
            p=model.head(train_x[index])*cfg.intensity_scale
            loss=au_intensity_mse(p,train_t[index],intensity_scale=cfg.intensity_scale)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite AU loss')
            loss.backward();optimizer.step();loss_sum+=float(loss.detach().cpu())*len(index)
        model.head.eval()
        with torch.inference_mode(): pred=(model.head(dev_x)*cfg.intensity_scale).cpu().numpy()
        dev_metrics=metrics(dev_y,pred,dev_subjects); qualification=qualify(dev_metrics,baselines,support)
        record={'epoch':epoch,'binding_sha256':binding_sha,'head_state_dict':{k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},
                'optimizer_state_dict':optimizer.state_dict(),'rng_state':rng_state(torch,np,shuffle,args.device),
                'train_loss':loss_sum/len(train_x),'dev_metrics':dev_metrics,'qualification':qualification,
                'dev_predictions_raw':torch.tensor(pred),'dev_labels_raw':labels[dm].clone(),'dev_ids':dev_ids,'dev_subjects':dev_subjects}
        path=out/'checkpoints'/f'epoch_{epoch:03d}.pt';torch_save_new(torch,record,path)
        history.append({'epoch':epoch,'metrics':dev_metrics,'qualification':qualification,'checkpoint':str(path),'sha256':sha(path),'train_loss':record['train_loss']})
        write_json(out/'epochs'/f'epoch_{epoch:03d}.json',{'epoch':epoch,'train_loss':record['train_loss'],'dev_metrics':dev_metrics,'qualification':qualification,
                   'predictions':[{'sample_id':i,'subject':s,'true_raw':y.tolist(),'predicted_raw':p.tolist()} for i,s,y,p in zip(dev_ids,dev_subjects,dev_y,pred)]})
        write_json(out/'progress.json',{'epoch':epoch,'total_epochs':args.epochs,'elapsed_seconds':time.monotonic()-started,'macro_mae':dev_metrics['macro_mae'],
                   'active_mae':dev_metrics['active_mae'],'qualified':qualification['passed']},replace=True)
        print(f'epoch {epoch:03d} train_loss={record["train_loss"]:.6f} dev_mae={dev_metrics["macro_mae"]:.4f} active={dev_metrics["active_mae"]:.4f} qualified={qualification["passed"]}',flush=True)
    if args.device=='mps': torch.mps.synchronize()
    qualified=[h for h in history if h['qualification']['passed']]
    key=lambda h:(h['metrics']['macro_mae'],h['metrics']['active_mae'],h['epoch'])
    selected=min(qualified,key=key) if qualified else None
    diagnostic=min(history,key=key)
    exported=None
    if selected:
        c=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
        model.head.load_state_dict(c['head_state_dict'],strict=True)
        exported=out/'qualified_supervisor.pt'
        if not exported.exists():
            torch_save_new(torch,{'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':dataclasses.asdict(cfg),
                          'provenance':{'binding_sha256':binding_sha,'subject_split_sha256':SPLIT_SHA,'trained_epoch':selected['epoch'],
                                        'test_used':False,'qualified':True,'selected_checkpoint_sha256':selected['sha256']}},exported)
        else:
            old=torch.load(exported,map_location='cpu',weights_only=True)
            if old['provenance']['binding_sha256']!=binding_sha or old['provenance']['selected_checkpoint_sha256']!=selected['sha256']:
                raise ValueError('Existing qualified export does not match selection')
            for k,v in model.state_dict().items():
                if not torch.equal(v.detach().cpu(),old['model_state_dict'][k]): raise ValueError('Existing export tensor changed')
    final={'qualified':bool(selected),'training_complete':True,'stage':'head_only','checkpoint_path':str(exported) if exported else None,
           'checkpoint_sha256':sha(exported) if exported else None,'subject_split_sha256':SPLIT_SHA,
           'train_subjects':groups['train'],'dev_subjects':groups['dev'],'test_used':False,'architecture':dataclasses.asdict(cfg),
           'preprocess':{'path':str(EXPERIMENT/'modeling/au_regressor.py'),'sha256':sha(EXPERIMENT/'modeling/au_regressor.py'),
                         'function':'preprocess_for_au','input_range':'zero_one for true images; minus_one_one for generated tensors',
                         'resize':'bilinear224 align_corners=False antialias=False','normalization':'ImageNet'},
           'imagenet_weights':{'path':str(weights),'sha256':IMAGENET_RESNET18_SHA256},'binding_sha256':binding_sha,
           'gate':(selected or diagnostic)['qualification'],'predeclared_gate':gate,'selected_epoch':selected['epoch'] if selected else None,
           'selected_metrics':selected['metrics'] if selected else None,'diagnostic_best':diagnostic,
           'epochs_completed':len(history),'baseline_file':str(out/'constant_baselines.json')}
    write_json(out/'qualification.json',final,replace=args.resume)
    write_json(out/'history.json',history,replace=args.resume)
    if args.device=='mps': torch.mps.empty_cache()
    print(json.dumps({'qualified':bool(selected),'selected_epoch':final['selected_epoch'],'diagnostic_epoch':diagnostic['epoch'],
                     'diagnostic_macro_mae':diagnostic['metrics']['macro_mae'],'diagnostic_active_mae':diagnostic['metrics']['active_mae'],'epochs':len(history)},indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
