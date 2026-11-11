"""Explicit DEV-only low-level sampling parity, dormant without --execute.

This never fabricates a final-test protocol and never calls the final CLI with
development data. It compares the already-validated dev recipe with the new
OurRuntime.prepare/sample functions on identical existing dev tensors.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from prepare_source_cache import digest,save_json,new_directory,tensor_sha
from check_source_runtime_dev_parity import CACHE,CACHE_SHA,CHECKPOINT,CHECKPOINT_SHA

DEV=ROOT/'development_evaluation/dev_coverage_audit/dev19_manifest_2k.json'
DEV_SHA='90a0307b053d369e49286ea63ad3f6e731c8213b3786fbd4591706c1b7f69a28'
OLD_GENERATOR_SHA='fbba652deaf7f1dbdb3972799b00a010e5782eaedddab40f85d4154941e9a83f'
PARAMETERS={'height':512,'width':512,'num_inference_steps':20,'guidance_scale':7.5,
            'au_scale':1.0,'ip_scale':.9,'scheduler':'DEISMultistepScheduler','negative_prompt':''}


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:
        print(json.dumps({'status':'not_executed','requires_explicit_gpu_authorization':True,'data_role':'dev_only','test_pixels_read':False}));return
    import torch
    from diffusers import StableDiffusionPipeline
    from modeling.frozen_backbone import FrozenConditionedDenoiser,FrozenImageTextEncoders,RV
    from modeling.safe_token_residual_adapter import RelativeRMSConfig
    from train_centered_adapter import load_checkpoint,resolve_source_anchor,verify_source_anchor,adapter_from_checkpoint,centered_batch_conditions,source_and_asset_bindings
    from generate_ours import OurRuntime
    for path,expected in ((CACHE/'features.pt',CACHE_SHA),(CHECKPOINT,CHECKPOINT_SHA),(DEV,DEV_SHA),
                          (ROOT/'generate_centered_development.py',OLD_GENERATOR_SHA)):
        if digest(path)!=expected:raise ValueError('Pinned development fixture or frozen old recipe changed')
    document=json.loads(DEV.read_text())
    if document['test_data_used'] is not False or document['cache_sha256']!=CACHE_SHA:raise ValueError('Not frozen development')
    rows=torch.load(CACHE/'features.pt',map_location='cpu',weights_only=True)['samples']
    if any(r['split'] not in ('train','dev') for r in rows):raise ValueError('Unexpected test cache row')
    samples={r['case_id']:r for r in rows}
    selected=[next(c for c in document['cases'] if not any(c['target_au_vector_0_5'])),
              next(c for c in document['cases'] if any(c['target_au_vector_0_5']))]
    for case in selected:
        if case['split']!='dev' or samples[case['case_id']]['split']!='dev':raise ValueError('No test accepted')
    anchor,binding=resolve_source_anchor([r for r in rows if r['split']=='train'],[r for r in rows if r['split']=='dev'])
    state=load_checkpoint(CHECKPOINT);verify_source_anchor(anchor,binding,state['bindings']['source_anchor'])
    current,_=source_and_asset_bindings({})
    if current['public_assets']!=state['bindings']['public_assets']:raise ValueError('Public base changed')
    output=new_directory(HERE/'runtime_checks','mps_dev_sampling')
    receipt={'status':'started','selection':'first frozen noedit and first edited DEV case in original order',
        'case_ids':[c['case_id'] for c in selected],'seed':101,'native_parameters':PARAMETERS,
        'checkpoint_sha256':CHECKPOINT_SHA,'cache_sha256':CACHE_SHA,'dev_manifest_sha256':DEV_SHA,
        'original_generator_sha256':OLD_GENERATOR_SHA,'new_generator_sha256':digest(HERE/'generate_ours.py'),
        'checker_sha256':digest(__file__),'test_used':False,'claim':'engineering parity only, not final model selection'}
    save_json(output/'started.json',receipt);shutil.copyfile(__file__,output/'checker_snapshot.py')
    backbone=FrozenConditionedDenoiser('mps',checkpointing=False);encoders=FrozenImageTextEncoders('mps')
    pipe=StableDiffusionPipeline.from_pretrained(str(RV),unet=backbone.unet,vae=encoders.vae,text_encoder=encoders.text_encoder,
        tokenizer=encoders.tokenizer,safety_checker=None,feature_extractor=None,requires_safety_checker=False,local_files_only=True).to('mps')
    if type(pipe.scheduler).__name__!='DEISMultistepScheduler':raise ValueError('Unexpected actual sampler')
    pipe.set_progress_bar_config(disable=True)
    model=adapter_from_checkpoint(state,'mps').eval().requires_grad_(False);e=state['effective_config']
    trust=RelativeRMSConfig(enabled=e['trust_region'],rho_au=e['rho_au'],rho_ip=e['rho_ip'],reference_rms_floor=e['reference_rms_floor'])
    runtime=OurRuntime.__new__(OurRuntime);runtime.device='mps';runtime.backbone=backbone;runtime.pipe=pipe
    runtime.settings={m:{'native_parameters':PARAMETERS} for m in ('baseline','v4')};runtime.methods={'baseline':(None,None),'v4':(model,trust)}
    runtime.features={'source_anchor':anchor,'sources':{},'conditions':{}};checks=[]
    for case in selected:
        cid=case['case_id'];sample=samples[cid];sid=sample['subject']
        runtime.features['sources'][sid]={k:sample[k] for k in ('image_patches','base_faceid','source_identity')}
        runtime.features['conditions'][cid]={k:sample[k] for k in ('text_pooled','target_aus','source_aus','prompt')}
        with torch.no_grad():
            patches,text,aus,identity=[sample[k][None].to('mps') for k in ('image_patches','text_pooled','target_aus','base_faceid')]
            base_au,base_ip=backbone.baseline_conditions(aus,identity)
            null_au,null_ip=backbone.baseline_conditions(torch.zeros_like(aus),torch.zeros_like(identity))
            for method in ('baseline','v4'):
                if method=='baseline':values={'au_embedding':base_au,'image_embedding':base_ip}
                else:
                    batch={'image_patches':patches,'text_pooled':text,'target_aus':aus,'source_aus':sample['source_aus'][None].to('mps')}
                    _,_,values=centered_batch_conditions(model,batch,anchor.to('mps'),base_au,base_ip,trust)
                legacy_cross={'au_embedding':torch.cat((null_au,values['au_embedding'])),'image_embedding':torch.cat((null_ip,values['image_embedding'])),
                              'au_scale':1.0,'ip_scale':.9}
                prepared=runtime.prepare(method,cid,sid)
                condition_equal=all(torch.equal(legacy_cross[k],prepared['cross'][k]) for k in ('au_embedding','image_embedding'))
                if not condition_equal:raise RuntimeError('Legacy/new conditions differ; no sampling')
                legacy=pipe(sample['prompt'],negative_prompt='',height=512,width=512,num_inference_steps=20,guidance_scale=7.5,
                            generator=torch.Generator(device='cpu').manual_seed(101),cross_attention_kwargs=legacy_cross).images[0]
                new=runtime.sample(method,prepared,101)
                oldpath=output/(cid+'_'+method+'_legacy.png');newpath=output/(cid+'_'+method+'_new.png')
                legacy.save(oldpath);new.save(newpath)
                check={'case_id':cid,'method':method,'conditions_equal':condition_equal,'legacy_sha256':digest(oldpath),
                       'new_sha256':digest(newpath),'png_exact':digest(oldpath)==digest(newpath)}
                checks.append(check);print(json.dumps(check),flush=True)
    save_json(output/'result.json',{**receipt,'status':'complete','passed':all(c['png_exact'] for c in checks),
                                  'checks':checks,'actual_scheduler':type(pipe.scheduler).__name__})


if __name__=='__main__':main()
