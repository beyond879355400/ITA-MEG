"""Dev-only region projection during denoising, never post-hoc image retouching."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
V5 = HERE.parent / 'quality_diagnosis_v5_20260915'
V5RUN = V5 / 'runs/source_init_202609151525'
spec = importlib.util.spec_from_file_location('fixed_v5_generation', V5 / 'run_diagnostic.py')
v5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v5)
need, sha, save, ref = v5.need, v5.sha, v5.save, v5.ref
CONFIGS = [('masked_s055_a100', .55, 1.), ('masked_s075_a100', .75, 1.), ('masked_s075_a150', .75, 1.5)]

def inputs():
    need(sha(V5 / 'run_diagnostic.py') == '6296388a3ffeafd6cc413589f0f498966e113c8c0e183e36e30017fce96edd89', 'Original v5 code changed')
    need(sha(V5RUN / 'cases.json') == '7daad59d5540d7b49243cb03da38f7aaed5d2c4294c8cb46ac7ce58b46d43145', 'Original v5 cases changed')
    need(sha(V5RUN / 'result.json') == '4e92651f47abadfefdf0be2a91600544c1142cbbd99ebab411b2bea0b22177ce', 'Original v5 output changed')
    original = v5.preflight()
    cases = v5.read(V5RUN / 'cases.json')['cases']
    for c,e in zip(cases,original):
        need(all(c[k] == e[k] for k in e), 'Case differs from original dev request')
    records = {(r['case_id'],r['method']):r for r in v5.read(V5RUN/'result.json')['records']}
    for c in cases:
        for method in ('baseline_img2img_s055','bidirectional_img2img_s055','baseline_img2img_s075','bidirectional_img2img_s075'):
            r=records[c['case_id'],method]
            need(r['status']=='complete' and sha(r['output'])==r['sha256'], 'Original v5 generation changed')
            c[method]={'path':r['output'],'sha256':r['sha256']}
    return cases

def project_latents(predicted, source, noise, mask, scheduler, next_t, torch):
    """Project outside the source-derived editing mask to the NEXT noise level."""
    reference = source if next_t is None else scheduler.add_noise(source, noise, next_t.reshape(1))
    result = mask * predicted + (1 - mask) * reference
    need(torch.isfinite(result).all().item(), 'Nonfinite region projection')
    return result

def load_prepared_masks(path,cases):
    import numpy as np
    from region_mask import build_mask
    doc=v5.read(path)
    need(doc['status']=='complete' and doc['source_only'] and doc['source_unchanged'] and not doc['test_used'] and not doc['target_geometry_used'],'Invalid source mask declaration')
    need(doc['inputs']['code_sha256']==sha(HERE/'region_mask.py'),'Mask code changed')
    for r in (doc['inputs']['cases'],doc['inputs']['source_measurements']): need(sha(r['path'])==r['sha256'],'Mask input changed')
    need(Path(doc['inputs']['cases']['path'])==V5RUN/'cases.json','Masks must use exact v5 cases')
    source=v5.read(doc['inputs']['source_measurements']['path'])
    mapping={r['case_id']:r for r in source['records'] if r['method']=='source_real'}
    entries={r['case_id']:r for r in doc['cases']}
    need(set(entries)=={c['case_id'] for c in cases},'Mask cases changed')
    refs={}
    for c in cases:
        entry=entries[c['case_id']]; r=entry['mask64']
        need(sha(r['path'])==r['sha256'] and entry['source_sha256']==c['source_real']['sha256'],'Mask source/file SHA mismatch')
        array=np.load(r['path'],allow_pickle=False); expected,receipt=build_mask(c,mapping[c['case_id']])
        need(np.array_equal(array,expected),'Prepared mask differs from fixed source-only rule')
        refs[c['case_id']]={**r,'source_sha256':entry['source_sha256'],'area':entry['mask_area'],'receipt':{'path':entry['receipt_path'],'sha256':entry['receipt_sha256']}}
    return refs

def execute(out, mask_manifest):
    cases = inputs()
    mask_refs = load_prepared_masks(mask_manifest, cases)
    need(out.parent == HERE/'runs' and not out.exists(), 'New v6 run required')
    out.mkdir(parents=True)
    for c in cases:
        c['region_mask'] = mask_refs[c['case_id']]
    save(out/'cases.json', {'kind':'v6_development_region_preserving_cases','test_data_used':False,'case_count':8,'cases':cases})
    configuration={'kind':'v6_region_preserving_diagnostic','test_data_used':False,'training_performed':False,
        'script':ref(__file__),'region_code':ref(HERE/'region_mask.py'),'mask_manifest':ref(mask_manifest),
        'v5_cases':ref(V5RUN/'cases.json'),'configs':[{'name':n,'strength':s,'au_scale':a} for n,s,a in CONFIGS],
        'seed':101,'nominal_steps':20,'guidance_scale':7.5,'ip_scale':.9,
        'projection':'mask * next_latent + (1-mask) * source_with_same_seed_noise_at_next_t; final source is clean',
        'no_edit_semantics':'empty mask means source VAE reconstruction by construction, not a learned identity claim',
        'post_generation_pixel_blend':False}
    save(out/'configuration.json',configuration)
    torch=v5.environment()
    import numpy as np
    from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL, DEISMultistepScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from modeling.frozen_backbone import FrozenConditionedDenoiser, RV
    from modeling.safe_token_residual_adapter import RelativeRMSConfig
    from train_centered_adapter import (load_checkpoint, adapter_from_checkpoint, centered_batch_conditions,
        resolve_source_anchor, verify_source_anchor, source_and_asset_bindings)
    payload=load_checkpoint(v5.CKPT)
    need(payload['global_step']==200 and payload['variant']=='bidirectional','Early200 module required')
    for p,h in payload['bindings']['sources'].items(): need(sha(EXP/p)==h,'Trained source changed')
    current,_=source_and_asset_bindings({}); need(current['public_assets']==payload['bindings']['public_assets'],'Public weights changed')
    data=torch.load(v5.CACHE/'features.pt',map_location='cpu',weights_only=True)['samples']
    need(all(r['split'] in ('train','dev') for r in data),'No test data')
    samples={r['case_id']:r for r in data}
    anchor,binding=resolve_source_anchor([r for r in data if r['split']=='train'],[r for r in data if r['split']=='dev'])
    verify_source_anchor(anchor,binding,payload['bindings']['source_anchor'])
    adapter=adapter_from_checkpoint(payload,'mps').eval().requires_grad_(False)
    e=payload['effective_config']; trust=RelativeRMSConfig(enabled=e['trust_region'],rho_au=e['rho_au'],rho_ip=e['rho_ip'],reference_rms_floor=e['reference_rms_floor'])
    backbone=FrozenConditionedDenoiser('mps',checkpointing=False)
    vae=AutoencoderKL.from_pretrained(str(RV/'vae'),local_files_only=True,torch_dtype=torch.float32).eval().requires_grad_(False)
    text=CLIPTextModel.from_pretrained(str(RV/'text_encoder'),local_files_only=True,torch_dtype=torch.float32).eval().requires_grad_(False)
    pipe=StableDiffusionImg2ImgPipeline(vae=vae,text_encoder=text,tokenizer=CLIPTokenizer.from_pretrained(str(RV/'tokenizer'),local_files_only=True),
        unet=backbone.unet,scheduler=DEISMultistepScheduler.from_pretrained(str(RV/'scheduler'),local_files_only=True),
        safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('mps')
    pipe.set_progress_bar_config(disable=True)
    def finite_decode(module,args,result): need(torch.isfinite(result).all().item(),'Nonfinite VAE decode')
    decode_guard=vae.decoder.register_forward_hook(finite_decode)
    info=v5.read(V5RUN/'vae_check.json'); need(sha(info['latents_file']['path'])==info['latents_file']['sha256'],'V5 source latent changed')
    latents=torch.load(info['latents_file']['path'],map_location='cpu',weights_only=True)
    save(out/'runtime.json',{'scheduler':type(pipe.scheduler).__name__,'scheduler_config':dict(pipe.scheduler.config),
        'checkpoint':ref(v5.CKPT),'public_assets':current['public_assets'],'source_latents':info['latents_file'],
        'v5_code':ref(V5/'run_diagnostic.py'),'source_anchor':binding,'torch':torch.__version__,'target_pixels_supplied':False})
    results=[]
    with torch.no_grad(), (out/'generation_records.jsonl').open('x',buffering=1) as log:
        for c in cases:
            s=samples[c['case_id']]; need(s['split']=='dev' and s['subject']==c['subject'],'Dev case mismatch')
            src=latents[c['source_real']['sha256']].to('mps')
            mask_ref=c['region_mask']; mask=torch.from_numpy(np.load(mask_ref['path'],allow_pickle=False))[None,None].to('mps')
            need(mask.shape==(1,1,64,64) and mask.min()>=0 and mask.max()<=1,'Invalid editing mask')
            need((mask.max().item()==0)==(c['group']=='no_edit'),'Empty mask and noedit mismatch')
            aus,identity=s['target_aus'][None].to('mps'),s['base_faceid'][None].to('mps')
            need(s['target_aus'].tolist()==c['target_aus_0_5'] and s['source_aus'].tolist()==c['source_aus_0_5'],'AU labels changed')
            base_au,base_ip=backbone.baseline_conditions(aus,identity)
            null_au,null_ip=backbone.baseline_conditions(torch.zeros_like(aus),torch.zeros_like(identity))
            batch={k:s[k][None].to('mps') for k in ('image_patches','text_pooled','target_aus','source_aus')}
            _,_,adapted=centered_batch_conditions(adapter,batch,anchor.to('mps'),base_au,base_ip,trust)
            for method,values in [('baseline',{'au_embedding':base_au,'image_embedding':base_ip}),('bidirectional',adapted)]:
                cross={'au_embedding':torch.cat((null_au,values['au_embedding'])),'image_embedding':torch.cat((null_ip,values['image_embedding'])),'ip_scale':.9}
                condition_hashes={k:v5.tensor_sha(v) for k,v in cross.items() if isinstance(v,torch.Tensor)}
                need(condition_hashes==c[method+'_text2img_condition_hashes'],'Conditions changed unexpectedly')
                for config,strength,au_scale in CONFIGS:
                    name=method+'_'+config; cross['au_scale']=au_scale
                    # prepare_latents on a4-channel source consumes precisely this
                    # CPU standard-normal sample; no posterior sampling is used.
                    noise=torch.randn(src.shape,generator=torch.Generator(device='cpu').manual_seed(101),dtype=torch.float32).to('mps')
                    history=[]
                    def step_end(pipeline,i,t,kwargs):
                        need(torch.isfinite(kwargs['latents']).all().item(),'Nonfinite denoising latent')
                        ts=pipeline.scheduler.timesteps[20-int(20*strength):]
                        need(int(t)==int(ts[i]),'Callback index mismatch')
                        nt=ts[i+1] if i+1<len(ts) else None
                        projected=project_latents(kwargs['latents'],src,noise,mask,pipeline.scheduler,nt,torch)
                        history.append({'t':int(t),'next_t':None if nt is None else int(nt),'latent_sha256':v5.tensor_sha(projected)})
                        return {'latents':projected}
                    row={k:c[k] for k in ('case_id','subject','split','group','source_aus_0_5','target_aus_0_5','prompt')}
                    path=out/'images'/c['case_id']/(name+'.png'); path.parent.mkdir(parents=True,exist_ok=True)
                    row.update(method=name,config=config,seed=101,strength=strength,au_scale=au_scale,ip_scale=.9,nominal_steps=20,
                        output=str(path),condition_hashes=condition_hashes,region_mask=mask_ref,source_latent_sha256=v5.tensor_sha(src),noise_sha256=v5.tensor_sha(noise))
                    started=time.monotonic()
                    try:
                        image=pipe(prompt=c['prompt'],image=src,strength=strength,num_inference_steps=20,guidance_scale=7.5,negative_prompt='',
                            generator=torch.Generator(device='cpu').manual_seed(101),cross_attention_kwargs=cross,
                            callback_on_step_end=step_end,callback_on_step_end_tensor_inputs=['latents']).images[0]
                        need(len(history)==int(20*strength),'Wrong actual step count')
                        image.save(path); row.update(status='complete',sha256=sha(path))
                        if c['group']=='no_edit': need(row['sha256']==c['vae_reconstruction']['sha256'],'Empty mask must reconstruct source')
                    except Exception as error:
                        row.update(status='failed',sha256=None,error=str(error),traceback=traceback.format_exc())
                    row.update(seconds=time.monotonic()-started,actual_steps=len(history),actual_timesteps=[r['t'] for r in history],projection_trace=history)
                    results.append(row); log.write(json.dumps(row,ensure_ascii=False)+'\n')
                    print(json.dumps({k:row[k] for k in ('case_id','method','status','seconds')},ensure_ascii=False),flush=True)
            if len(results)==6: need(all(r['status']=='complete' for r in results),'First-case engineering gate failed')
    decode_guard.remove()
    save(out/'result.json',{'kind':'v6_region_preserving_diagnostic','status':'complete' if all(r['status']=='complete' for r in results) else 'completed_with_failures',
        'test_data_used':False,'planned':48,'completed':sum(r['status']=='complete' for r in results),'records':results,
        'cases':ref(out/'cases.json'),'configuration':ref(out/'configuration.json')})
    print(json.dumps({'status':'v6_complete','run':str(out)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True);p.add_argument('--masks',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute: print(json.dumps({'status':'not_executed','cases':len(inputs())}));sys.exit(0)
    execute(a.run.resolve(),a.masks.resolve())
