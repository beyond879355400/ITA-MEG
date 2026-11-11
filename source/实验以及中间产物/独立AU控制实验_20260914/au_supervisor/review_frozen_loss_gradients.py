"""CPU64x64 infrastructure gradient probe, not a generated experimental sample."""
import json
import sys
from align_real_frames import HERE,EXPERIMENT,environment,sha,write_json

def main():
    environment();sys.path.insert(0,str(EXPERIMENT))
    import torch
    from modeling.frozen_backbone import FrozenImageTextEncoders
    from modeling.au_regressor import AUIntensityRegressor,AURegressorConfig
    torch.set_num_threads(2);torch.manual_seed(20260914)
    enc=FrozenImageTextEncoders('cpu')
    q=json.loads((HERE/'layer4_v1/qualification.json').read_text());assert q['qualified']
    au=AUIntensityRegressor(AURegressorConfig());payload=torch.load(q['checkpoint_path'],map_location='cpu',weights_only=True)
    au.load_state_dict(payload['model_state_dict'],strict=True);au.freeze_for_generator_loss()
    cached=torch.load(EXPERIMENT/'training_cache/expanded_v2_202609141711/features.pt',map_location='cpu',weights_only=True)
    row=next(r for r in cached['samples'] if r['split']=='train')
    epsilon=torch.nn.Parameter(torch.zeros(1,4,8,8));noisy=torch.randn(1,4,8,8)*.1
    alpha=torch.tensor(.8).view(1,1,1,1);x0=(noisy-(1-alpha).sqrt()*epsilon)/alpha.sqrt()
    decoded=enc.decode_latents(x0).clamp(-1,1)
    identity=enc.identity_embedding(decoded);pred_au=au(decoded,input_range='minus_one_one')
    id_loss=(1-torch.nn.functional.cosine_similarity(identity,row['source_identity'][None])).mean()
    au_loss=(pred_au/5-row['target_aus'][None]/5).square().mean()
    gi=torch.autograd.grad(id_loss,epsilon,retain_graph=True)[0]
    ga=torch.autograd.grad(au_loss,epsilon)[0]
    assert torch.isfinite(gi).all() and torch.isfinite(ga).all() and gi.abs().sum()>0 and ga.abs().sum()>0
    assert all(not p.requires_grad and p.grad is None for module in (enc,au) for p in module.parameters())
    report={'status':'passed','scope':'reduced64x64 CPU gradient connectivity only; no image saved, no quality score or test sample',
            'review_script_sha256':sha(__file__),'frozen_backbone_source_sha256':sha(EXPERIMENT/'modeling/frozen_backbone.py'),
            'au_checkpoint_sha256':q['checkpoint_sha256'],'identity_target_sample_id':row['case_id'],'target_split':'train','test_used':False,'mps_used':False,
            'frozen_public_parameters_have_no_grad':True,'identity_loss_gradient_to_epsilon_l1':float(gi.abs().sum()),
            'au_loss_gradient_to_epsilon_l1':float(ga.abs().sum()),'au_loss_scale':'raw0-5 MSE/25','x0_formula':'(x_t-sqrt(1-alpha)*epsilon)/sqrt(alpha)',
            'alpha':.8,'snr':4.0,'identity_loss':float(id_loss.detach()),'au_loss':float(au_loss.detach())}
    write_json(HERE/'adapter_contract_review/frozen_loss_gradient_report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
