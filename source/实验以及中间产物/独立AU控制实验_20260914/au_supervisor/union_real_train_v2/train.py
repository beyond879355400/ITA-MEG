#!/usr/bin/env python3
"""Warmstart AU layer4+head on the expanded real train union; fixed dev240.

The previous optimizer is intentionally NOT loaded. --resume is only for a
checkpoint produced by this same new data/config/source binding.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import random
import sys
import time
HERE=Path(__file__).resolve().parent;SUP=HERE.parent;ROOT=SUP.parent
sys.path.insert(0,str(SUP));sys.path.insert(0,str(ROOT))
from align_real_frames import environment,sha,write_json,copy_exact,SPLIT_SHA
from train_head_only import rng_state,restore_rng,torch_save_new
from train_layer4 import freeze_except_layer4,training_mode_frozen_bn

def inputs(config_path):
    cfg=json.loads(config_path.read_text());datafile=Path(cfg['data_manifest']);parentfile=Path(cfg['parent_qualification'])
    assert sha(datafile)==cfg['data_manifest_sha256'] and sha(parentfile)==cfg['parent_qualification_sha256']
    assert sha(cfg['parent_checkpoint'])==cfg['parent_checkpoint_sha256']
    assert sha(ROOT/'modeling/au_regressor.py')==cfg['model_source_sha256']
    data=json.loads(datafile.read_text());parent=json.loads(parentfile.read_text())
    assert data['subject_split_sha256']==cfg['subject_split_sha256']==SPLIT_SHA and not data['test_used'] and not parent['test_used']
    assert parent['qualified'] and parent['training_complete'] and parent['checkpoint_sha256']==cfg['parent_checkpoint_sha256']
    assert len(data['train'])==data['train_count']==2502 and len(data['dev'])==data['dev_count']==240
    assert not set(data['train_subjects'])&set(data['dev_subjects']) and sha(data['split_path'])==SPLIT_SHA
    assert cfg['epochs']==30 and cfg['batch_size']==16 and cfg['layer4_lr']==.0001 and cfg['head_lr']==.0005 and cfg['weight_decay']==.0001
    assert cfg['qualification_gate']==parent['predeclared_gate'] and cfg['model_config']==parent['architecture']
    return cfg,data,parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=HERE/'train_config.json');ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--stop-after-epoch',type=int,default=30)
    args=ap.parse_args();cfg,data,parent=inputs(args.config.resolve())
    if args.preflight:
        print(json.dumps({'status':'pass','train':2502,'dev':240,'test_used':False,'parent_checkpoint_sha256':cfg['parent_checkpoint_sha256'],
                          'optimizer':'new AdamW on first entry; exact resume only within this binding','epochs':30,'updates_per_epoch':157,
                          'mps_started':False,'output':cfg['output_dir']},indent=2));return 0
    environment()
    import numpy as np
    from PIL import Image
    import torch
    from modeling.au_regressor import AUIntensityRegressor,AURegressorConfig,au_intensity_mse
    from supervisor_metrics import metrics,constant_baselines,supported_aus,qualify,AUS
    assert 1<=args.stop_after_epoch<=30 and cfg['device'] in ('mps','cpu')
    out=Path(cfg['output_dir']).resolve();assert HERE in out.parents
    rows=data['train'];devrows=data['dev']
    for row in rows+devrows:
        assert row['status']=='aligned' and sha(row['aligned_path'])==row['aligned_sha256']
    train_y=np.asarray([r['input']['au_vector_0_5'] for r in rows],dtype=np.float32);dev_y=np.asarray([r['input']['au_vector_0_5'] for r in devrows],dtype=np.float32)
    dev_ids=[r['sample_id'] for r in devrows];dev_subjects=[r['subject'] for r in devrows]
    support=supported_aus(dev_y);assert [AUS[i] for i in support]==cfg['qualification_gate']['supported_au_numbers']
    baselines=constant_baselines(train_y,dev_y,dev_subjects)
    sources=[Path(__file__),SUP/'train_head_only.py',SUP/'train_layer4.py',SUP/'supervisor_metrics.py',ROOT/'modeling/au_regressor.py']
    binding={'config_sha256':sha(args.config),'config':cfg,'sources':{str(p):sha(p) for p in sources},'data_manifest_sha256':cfg['data_manifest_sha256'],
             'subject_split_sha256':SPLIT_SHA,'parent_checkpoint_sha256':cfg['parent_checkpoint_sha256'],'dev_rows_canonical_sha256':data['dev_rows_canonical_sha256']}
    if out.exists():assert args.resume and json.loads((out/'binding.json').read_text())==binding
    else:
        out.mkdir(parents=True);write_json(out/'binding.json',binding);write_json(out/'constant_baselines.json',baselines)
        for path in sources:copy_exact(path,out/'snapshots'/path.name,sha(path))
    binding_sha=sha(out/'binding.json')
    train_pixels=torch.from_numpy(np.stack([np.asarray(Image.open(r['aligned_path']).convert('RGB')) for r in rows]))
    dev_pixels=torch.from_numpy(np.stack([np.asarray(Image.open(r['aligned_path']).convert('RGB')) for r in devrows]))
    device=cfg['device']
    def image_batch(buffer,index):return buffer[index].permute(0,3,1,2).to(device,dtype=torch.float32)/255
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if device=='mps':torch.mps.manual_seed(cfg['seed'])
    shuffle=torch.Generator(device='cpu').manual_seed(cfg['seed'])
    model=AUIntensityRegressor(AURegressorConfig(**cfg['model_config'])).to(device)
    initial=torch.load(cfg['parent_checkpoint'],map_location='cpu',weights_only=True);assert initial['config']==cfg['model_config']
    model.load_state_dict(initial['model_state_dict'],strict=True);freeze_except_layer4(model)
    optimizer=torch.optim.AdamW([{'params':list(model.backbone.layer4.parameters()),'lr':cfg['layer4_lr']},{'params':list(model.head.parameters()),'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])
    assert not optimizer.state  # warmstart is weights-only, never old optimizer moments
    labels=torch.tensor(train_y,device=device)
    def evaluate():
        model.eval();pred=[]
        with torch.inference_mode():
            for index in torch.arange(len(devrows)).split(cfg['batch_size']):pred.append(model(image_batch(dev_pixels,index),input_range='zero_one').cpu())
        return torch.cat(pred).numpy()
    parent_report_path=out/'parent_entry.json'
    if not parent_report_path.exists():
        prediction=evaluate();m=metrics(dev_y,prediction,dev_subjects)
        write_json(parent_report_path,{'parent_checkpoint_sha256':cfg['parent_checkpoint_sha256'],'metrics':m,'gate_on_new_train_baselines':qualify(m,baselines,support),
                   'difference_from_parent_recorded_macro':m['macro_mae']-parent['selected_metrics']['macro_mae'],
                   'predictions':[{'sample_id':i,'subject':s,'true_raw':y.tolist(),'predicted_raw':p.tolist()} for i,s,y,p in zip(dev_ids,dev_subjects,dev_y,prediction)]})
        if abs(m['macro_mae']-parent['selected_metrics']['macro_mae'])>1e-3:raise RuntimeError('Parent restore/preprocess sanity differs from recorded evaluation')
    history=[];paths=sorted((out/'checkpoints').glob('epoch_*.pt')) if (out/'checkpoints').exists() else []
    for path in paths:
        c=torch.load(path,map_location='cpu',weights_only=True);assert c['binding_sha256']==binding_sha and c['epoch']==len(history)+1
        history.append({'epoch':c['epoch'],'metrics':c['dev_metrics'],'qualification':c['qualification'],'checkpoint':str(path),'sha256':sha(path),'train_loss':c['train_loss'],'epoch_seconds':c['epoch_seconds']})
        report=out/'epochs'/f'epoch_{c["epoch"]:03d}.json'
        if not report.exists():write_json(report,{'epoch':c['epoch'],'train_loss':c['train_loss'],'dev_metrics':c['dev_metrics'],'qualification':c['qualification'],
                'predictions':[{'sample_id':i,'subject':s,'true_raw':y.tolist(),'predicted_raw':p.tolist()} for i,s,y,p in zip(dev_ids,dev_subjects,c['dev_labels_raw'],c['dev_predictions_raw'])]})
    start=1
    if paths:
        model.load_state_dict(c['model_state_dict'],strict=True);optimizer.load_state_dict(c['optimizer_state_dict']);restore_rng(torch,np,shuffle,c['rng_state'],device);start=c['epoch']+1
    bn={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if any(s in k for s in ('running_mean','running_var','num_batches_tracked'))}
    for epoch in range(start,args.stop_after_epoch+1):
        tick=time.monotonic();training_mode_frozen_bn(model,torch);order=torch.randperm(len(rows),generator=shuffle);loss_sum=0
        for index in order.split(cfg['batch_size']):
            optimizer.zero_grad(set_to_none=True);pixels=image_batch(train_pixels,index);pred=model(pixels,input_range='zero_one')
            loss=au_intensity_mse(pred,labels[index.to(device)]);assert torch.isfinite(loss)
            loss.backward();optimizer.step();loss_sum+=float(loss.detach().cpu())*len(index);del pixels,pred,loss
        predictions=evaluate();m=metrics(dev_y,predictions,dev_subjects);gate=qualify(m,baselines,support)
        for k,v in bn.items():assert torch.equal(v,model.state_dict()[k].detach().cpu())
        seconds=time.monotonic()-tick
        c={'epoch':epoch,'epoch_seconds':seconds,'binding_sha256':binding_sha,'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
           'optimizer_state_dict':optimizer.state_dict(),'rng_state':rng_state(torch,np,shuffle,device),'train_loss':loss_sum/len(rows),'dev_metrics':m,'qualification':gate,
           'dev_predictions_raw':torch.tensor(predictions),'dev_labels_raw':torch.tensor(dev_y),'dev_ids':dev_ids,'dev_subjects':dev_subjects}
        path=out/'checkpoints'/f'epoch_{epoch:03d}.pt';torch_save_new(torch,c,path)
        history.append({'epoch':epoch,'metrics':m,'qualification':gate,'checkpoint':str(path),'sha256':sha(path),'train_loss':c['train_loss'],'epoch_seconds':seconds})
        write_json(out/'epochs'/f'epoch_{epoch:03d}.json',{'epoch':epoch,'train_loss':c['train_loss'],'dev_metrics':m,'qualification':gate,
                'predictions':[{'sample_id':i,'subject':s,'true_raw':y.tolist(),'predicted_raw':p.tolist()} for i,s,y,p in zip(dev_ids,dev_subjects,dev_y,predictions)]})
        write_json(out/'progress.json',{'epoch':epoch,'epochs':30,'seconds':seconds,'loss':c['train_loss'],'macro_mae':m['macro_mae'],'active_mae':m['active_mae'],'qualified':gate['passed']},replace=True)
        print(json.dumps({'epoch':epoch,'seconds':seconds,'loss':c['train_loss'],'macro':m['macro_mae'],'active':m['active_mae'],'qualified':gate['passed']}),flush=True)
    complete=len(history)==30;eligible=[h for h in history if h['qualification']['passed']];key=lambda h:(h['metrics']['macro_mae'],h['metrics']['active_mae'],h['epoch'])
    selected=min(eligible,key=key) if complete and eligible else None;diagnostic=min(history,key=key);export=None
    if selected:
        saved=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True);export=out/'qualified_supervisor.pt'
        payload={'model_state_dict':saved['model_state_dict'],'config':cfg['model_config'],'provenance':{'binding_sha256':binding_sha,'subject_split_sha256':SPLIT_SHA,
                 'parent_checkpoint_sha256':cfg['parent_checkpoint_sha256'],'training_data_manifest_sha256':cfg['data_manifest_sha256'],'trained_epoch':selected['epoch'],'test_used':False,'qualified':True}}
        if not export.exists():torch_save_new(torch,payload,export)
        else:
            stored=torch.load(export,map_location='cpu',weights_only=True);assert stored['provenance']==payload['provenance']
            for k,v in payload['model_state_dict'].items():assert torch.equal(v,stored['model_state_dict'][k])
    entry=json.loads(parent_report_path.read_text())['metrics'];picked=selected or diagnostic
    upgrade=bool(selected and selected['metrics']['macro_mae']<=entry['macro_mae'] and selected['metrics']['active_mae']<=entry['active_mae']
                 and (selected['metrics']['macro_mae']<entry['macro_mae'] or selected['metrics']['active_mae']<entry['active_mae']))
    report={'qualified':bool(selected),'training_complete':complete,'stage':'expanded_real_train_union_layer4_head','checkpoint_path':str(export) if export else None,
            'checkpoint_sha256':sha(export) if export else None,'subject_split_sha256':SPLIT_SHA,'train_subjects':data['train_subjects'],'dev_subjects':data['dev_subjects'],'test_used':False,
            'architecture':cfg['model_config'],'preprocess':parent['preprocess'],'imagenet_weights':parent['imagenet_weights'],'binding_sha256':binding_sha,
            'gate':picked['qualification'],'predeclared_gate':cfg['qualification_gate'],'selected_epoch':selected['epoch'] if selected else None,
            'selected_metrics':selected['metrics'] if selected else None,'diagnostic_best':diagnostic,'epochs_completed':len(history),
            'parent_checkpoint_sha256':cfg['parent_checkpoint_sha256'],'training_data_manifest_sha256':cfg['data_manifest_sha256'],
            'parent_entry_metrics':entry,'selected_or_diagnostic_minus_parent':{k:picked['metrics'][k]-entry[k] for k in ('macro_mae','active_mae','nonactive_mae')},
            'optimizer_reinitialized_for_changed_data':True,'recommended_upgrade':upgrade,
            'upgrade_note':'original qualification unchanged; upgrade recommendation additionally requires macro and active MAE both no worse than parent and at least one better; parent is never overwritten or automatically switched'}
    write_json(out/'qualification.json',report,replace=args.resume);write_json(out/'history.json',history,replace=args.resume)
    if device=='mps':torch.mps.synchronize();torch.mps.empty_cache()
    print(json.dumps({'complete':complete,'qualified':bool(selected),'selected_epoch':report['selected_epoch'],'parent_comparison':report['selected_or_diagnostic_minus_parent']},indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
