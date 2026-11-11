#!/usr/bin/env python3
"""Independent CPU readback of qualification, weights and input-gradient path."""
import dataclasses
import json
from pathlib import Path
import sys
from align_real_frames import HERE,EXPERIMENT,SPLIT_SHA,environment,sha,write_json


def main():
    environment();sys.path.insert(0,str(EXPERIMENT))
    import numpy as np
    from PIL import Image
    import torch
    from modeling.au_regressor import AUIntensityRegressor,AURegressorConfig
    from supervisor_metrics import metrics,qualify,supported_aus
    torch.set_num_threads(2)
    run=HERE/'layer4_v1';q=json.loads((run/'qualification.json').read_text());binding=json.loads((run/'binding.json').read_text())
    assert q['qualified'] and q['training_complete'] and q['epochs_completed']==30 and not q['test_used']
    assert q['subject_split_sha256']==SPLIT_SHA and sha(run/'binding.json')==q['binding_sha256']
    assert all(q['gate']['checks'].values()) and q['gate']['passed']
    for source in binding['sources'].values():assert sha(source['path'])==source['sha256']
    assert sha(q['checkpoint_path'])==q['checkpoint_sha256']
    export=torch.load(q['checkpoint_path'],map_location='cpu',weights_only=True)
    selected=torch.load(run/'checkpoints'/f'epoch_{q["selected_epoch"]:03d}.pt',map_location='cpu',weights_only=True)
    assert export['config']==dataclasses.asdict(AURegressorConfig())==q['architecture']
    assert export['provenance']['subject_split_sha256']==SPLIT_SHA and not export['provenance']['test_used']
    for k,v in export['model_state_dict'].items():assert torch.equal(v,selected['model_state_dict'][k]) and torch.isfinite(v).all()
    # Legacy ImageNet file omits num_batches_tracked; the supported constructor
    # restores these zero-initialized buffers through PyTorch's loading path.
    reference=AUIntensityRegressor(AURegressorConfig(),imagenet_weights=q['imagenet_weights']['path']).state_dict()
    unchanged_early=[];unchanged_bn=[];changed_layer4=[]
    for key,value in export['model_state_dict'].items():
        if key.startswith('backbone.'):
            original=reference[key]
            if not key.startswith('backbone.layer4.'):
                assert torch.equal(value,original);unchanged_early.append(key)
            if any(x in key for x in ['running_mean','running_var','num_batches_tracked']):
                assert torch.equal(value,original);unchanged_bn.append(key)
            elif key.startswith('backbone.layer4.') and not torch.equal(value,original):changed_layer4.append(key)
    assert changed_layer4
    baseline=json.loads((run/'constant_baselines.json').read_text())
    y=selected['dev_labels_raw'].numpy();p=selected['dev_predictions_raw'].numpy()
    recalculated=metrics(y,p,selected['dev_subjects']);gate=qualify(recalculated,baseline,supported_aus(y))
    assert recalculated==q['selected_metrics'] and gate==q['gate']
    groups=json.loads((EXPERIMENT/'data_prepare/au_supervision_v2/subjects_frozen.json').read_text())['groups']
    assert set(selected['dev_subjects'])==set(groups['dev']) and not set(selected['dev_subjects'])&set(groups['test'])
    model=AUIntensityRegressor(AURegressorConfig())
    model.load_state_dict(export['model_state_dict'],strict=True);model.freeze_for_generator_loss()
    assert not model.training and all(not par.requires_grad for par in model.parameters())
    manifest=json.loads((HERE/'alignment_v1/aligned_manifest.json').read_text())
    row=next(r for r in manifest['samples'] if r['split']=='train')
    assert sha(row['aligned_path'])==row['aligned_sha256']
    arr=np.asarray(Image.open(row['aligned_path']).convert('RGB')).copy()
    pixels=(torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()/127.5-1).requires_grad_(True)
    result=model(pixels,input_range='minus_one_one');result.sum().backward()
    assert result.shape==(1,12) and torch.isfinite(result).all() and float(result.min())>=0 and float(result.max())<=5
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and float(pixels.grad.abs().sum())>0
    assert all(par.grad is None for par in model.parameters())
    report={'status':'passed','qualification_sha256':sha(run/'qualification.json'),'checkpoint_sha256':q['checkpoint_sha256'],
            'strict_state_dict_load':True,'selected_checkpoint_tensor_equality':True,'source_hashes_match':True,
            'early_backbone_tensors_unchanged':len(unchanged_early),'bn_buffers_unchanged':len(unchanged_bn),
            'changed_layer4_parameter_tensors':len(changed_layer4),'metrics_recomputed_exactly':True,'gate_recomputed_exactly':True,
            'dev_subjects_checked':sorted(set(selected['dev_subjects'])),'test_used':False,
            'frozen_parameters_have_no_grad':True,'finite_nonzero_input_gradient':True,'input_gradient_l1':float(pixels.grad.abs().sum()),
            'gradient_probe_split':'train','gradient_probe_sample_id':row['sample_id'],'gradient_probe_image_sha256':row['aligned_sha256'],
            'verification_device':'cpu','verification_script_sha256':sha(__file__)}
    write_json(run/'export_verification.json',report);print(json.dumps(report,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
