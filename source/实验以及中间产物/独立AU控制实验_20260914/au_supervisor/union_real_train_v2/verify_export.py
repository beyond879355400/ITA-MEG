"""CPU readback of union-trained candidate, metrics and four real dev predictions."""
import importlib.metadata
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;SUP=HERE.parent;ROOT=SUP.parent
sys.path.insert(0,str(SUP));sys.path.insert(0,str(ROOT))
from align_real_frames import environment,sha,write_json,SPLIT_SHA

def main():
    environment()
    import numpy as np
    from PIL import Image
    import torch
    from modeling.au_regressor import AUIntensityRegressor,AURegressorConfig
    from supervisor_metrics import metrics,constant_baselines,supported_aus,qualify
    torch.set_num_threads(2)
    run=HERE/'training';q=json.loads((run/'qualification.json').read_text());data=json.loads((HERE/'data_union.json').read_text());config=json.loads((HERE/'train_config.json').read_text())
    assert q['training_complete'] and q['epochs_completed']==30 and q['subject_split_sha256']==SPLIT_SHA and not q['test_used']
    if not q['qualified']:
        write_json(run/'export_verification.json',{'status':'no_qualified_candidate','test_used':False,'recommended_upgrade':False});return
    assert sha(q['checkpoint_path'])==q['checkpoint_sha256']
    export=torch.load(q['checkpoint_path'],map_location='cpu',weights_only=True)
    selected=torch.load(run/'checkpoints'/f'epoch_{q["selected_epoch"]:03d}.pt',map_location='cpu',weights_only=True)
    parent=torch.load(config['parent_checkpoint'],map_location='cpu',weights_only=True)
    for key,value in export['model_state_dict'].items():assert torch.equal(value,selected['model_state_dict'][key]) and torch.isfinite(value).all()
    early=[];bn=[]
    for key,value in export['model_state_dict'].items():
        if key.startswith('backbone.') and not key.startswith('backbone.layer4.'):
            assert torch.equal(value,parent['model_state_dict'][key]);early.append(key)
        if any(name in key for name in ('running_mean','running_var','num_batches_tracked')):
            assert torch.equal(value,parent['model_state_dict'][key]);bn.append(key)
    train_y=np.asarray([r['input']['au_vector_0_5'] for r in data['train']],dtype=float);dev_y=selected['dev_labels_raw'].numpy();pred=selected['dev_predictions_raw'].numpy()
    baselines=constant_baselines(train_y,dev_y,selected['dev_subjects']);fresh=metrics(dev_y,pred,selected['dev_subjects'])
    assert fresh==q['selected_metrics'] and qualify(fresh,baselines,supported_aus(dev_y))==q['gate']
    model=AUIntensityRegressor(AURegressorConfig(**export['config']));model.load_state_dict(export['model_state_dict'],strict=True);model.freeze_for_generator_loss()
    seen=set();probes=[]
    for index,row in enumerate(data['dev']):
        if row['subject'] in seen:continue
        seen.add(row['subject']);assert sha(row['aligned_path'])==row['aligned_sha256']
        rgb=np.asarray(Image.open(row['aligned_path']).convert('RGB')).copy();x=torch.from_numpy(rgb).permute(2,0,1)[None].float()/255
        with torch.no_grad():actual=model(x,input_range='zero_one')[0]
        target=selected['dev_predictions_raw'][index]
        assert torch.allclose(actual,target,atol=1e-3,rtol=1e-3)
        probes.append({'subject':row['subject'],'sample_id':row['sample_id'],'max_abs_cpu_vs_saved_mps':float((actual-target).abs().max())})
    assert len(seen)==4
    row=data['train'][0];rgb=np.asarray(Image.open(row['aligned_path']).convert('RGB')).copy()
    x=(torch.from_numpy(rgb).permute(2,0,1)[None].float()/127.5-1).requires_grad_(True);model(x,input_range='minus_one_one').sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 and all(p.grad is None for p in model.parameters())
    report={'status':'pass','checkpoint_sha256':q['checkpoint_sha256'],'data_manifest_sha256':sha(HERE/'data_union.json'),'parent_checkpoint_sha256':config['parent_checkpoint_sha256'],
            'strict_state_and_selected_tensor_equality':True,'unchanged_early_tensors':len(early),'unchanged_bn_buffers':len(bn),'metrics_and_gate_recomputed':True,
            'one_fixed_dev_frame_per_subject_cpu_forward_checks':probes,'frozen_parameters_have_no_grad':True,'finite_nonzero_training_image_gradient':True,
            'recommended_upgrade':q['recommended_upgrade'],'selected_minus_parent':q['selected_or_diagnostic_minus_parent'],'test_used':False,'mps_used':False,
            'runtime_versions':{name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','scipy','Pillow')},'verification_script_sha256':sha(__file__)}
    write_json(run/'export_verification.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
