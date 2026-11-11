"""Fresh v3 development sampling on a pre-frozen, manually labelled case list."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  NO_ALBUMENTATIONS_UPDATE='1',HF_HOME=str(ROOT/'cache/huggingface'),
                  TORCH_HOME=str(ROOT/'cache/torch'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'))
sys.dont_write_bytecode=True
sys.pycache_prefix=str(ROOT/'cache/pycache')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--guidance-scale',type=float,default=7.5)
    p.add_argument('--preview-group',choices=('all','no_edit','lowest_available','second_edited'),default='all')
    p.add_argument('--skip-baseline',action='store_true',help='Generate only new checkpoints; compare against an already sealed baseline from this same new protocol')
    args=p.parse_args()
    if not 10<=args.steps<=50 or not 1<args.guidance_scale<=15:
        raise ValueError('Bounded development sampling requires CFG guidance >1 because conditions are packed as null+conditional')
    args.output=args.output.resolve()
    if ROOT not in args.output.parents or args.output.exists():
        raise ValueError('New in-experiment directory required')
    import torch
    from diffusers import StableDiffusionPipeline
    from modeling.frozen_backbone import FrozenConditionedDenoiser,FrozenImageTextEncoders,RV,sha256
    from modeling.safe_token_residual_adapter import TokenAdapterConfig,SafeTokenCrossModalResidualAdapter,SafeTokenPooledMLPResidualAdapter,RelativeRMSConfig,augment_condition_embeddings
    from train_adapter import load_checkpoint,MODEL_VERSION,source_and_asset_bindings
    meta=json.loads((args.cache/'manifest.json').read_text())
    case_doc=json.loads(args.cases.read_text())
    actual_sha=sha256(args.cache/'features.pt')
    if meta['test_data_used'] or case_doc['test_data_used'] or actual_sha!=meta['cache_sha256'] or actual_sha!=case_doc['cache_sha256'] or meta['subject_split_sha256']!=case_doc['subject_split_sha256']:
        raise ValueError('Cache/case/split bindings mismatch')
    cached=torch.load(args.cache/'features.pt',map_location='cpu',weights_only=True)['samples']
    samples={r['case_id']:r for r in cached}
    pairs={r['case_id']:r for r in meta['pairs']}
    selected=[c for c in case_doc['cases'] if args.preview_group=='all' or c['group']==args.preview_group]
    if not selected or len({c['case_id'] for c in selected})!=len(selected):
        raise ValueError('Empty or duplicate developer list')
    for case in selected:
        s=samples[case['case_id']]
        if s['split']!='dev' or case['split']!='dev' or s['subject']!=case['subject'] or s['target_aus'].tolist()!=case['target_au_vector_0_5']:
            raise ValueError('Not the fixed manual dev case')
    if args.skip_baseline and not args.checkpoint:
        raise ValueError('Cannot skip baseline with no checkpoint')
    methods=[] if args.skip_baseline else [('baseline',None,None,None)]
    seals=[]
    current_public,_=source_and_asset_bindings({})
    for checkpoint in args.checkpoint:
        if ROOT not in checkpoint.resolve().parents:
            raise ValueError('Only this experiment checkpoints accepted')
        state=load_checkpoint(checkpoint)
        if state['model_version']!=MODEL_VERSION or state['bindings']['cache']!=actual_sha or state['bindings']['split']!=meta['subject_split_sha256']:
            raise ValueError('Checkpoint is not bound to this exact data/version')
        if state['bindings']['public_assets']!=current_public['public_assets']:
            raise ValueError('Public model weights/configuration differ from training')
        if state['bindings']['development_subset']['manifest_sha256']!=sha256(args.cases):
            raise ValueError('Development case manifest differs from checkpoint-bound selection')
        for source in ('modeling/cross_modal_adapter.py','modeling/token_residual_adapter.py','modeling/safe_token_residual_adapter.py','modeling/frozen_backbone.py'):
            if state['bindings']['sources'][source]!=sha256(ROOT/source):
                raise ValueError('Model source differs from the trained source')
        cfg=TokenAdapterConfig(**state['model_config'])
        variant=state['variant']
        model=SafeTokenPooledMLPResidualAdapter(cfg,1032) if variant=='mlp' else SafeTokenCrossModalResidualAdapter(cfg)
        model.load_state_dict(state['adapter'],strict=True)
        model.eval().requires_grad_(False).to('mps')
        e=state['effective_config']
        trust=RelativeRMSConfig(enabled=e['trust_region'],rho_au=e['rho_au'],rho_ip=e['rho_ip'],reference_rms_floor=e['reference_rms_floor'])
        name=variant+'_step'+str(state['global_step'])+'_'+sha256(checkpoint)[:8]
        methods.append((name,model,trust,str(checkpoint.resolve())))
        seals.append({'name':name,'checkpoint_sha256':sha256(checkpoint),'effective_config':e,
                      'model_config':state['model_config'],'diagnostic_training':state['diagnostic']})
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'generation_source.py')
    shutil.copyfile(args.cases,args.output/'frozen_cases.json')
    (args.output/'settings.json').write_text(json.dumps({'cache_sha256':actual_sha,'case_manifest_sha256':sha256(args.cases),
        'preview_group':args.preview_group,'steps':args.steps,'guidance_scale':args.guidance_scale,'skip_baseline':args.skip_baseline,
        'sampling_seeds':case_doc['sampling_seeds'],'au_scale':1.0,'ip_scale':0.9,'methods':seals,
        'public_assets':current_public['public_assets'],
        'test_data_used':False,'formal_sota_evidence':False},indent=2)+'\n')
    backbone=FrozenConditionedDenoiser('mps',checkpointing=False)
    encoders=FrozenImageTextEncoders('mps')
    pipe=StableDiffusionPipeline.from_pretrained(str(RV),unet=backbone.unet,vae=encoders.vae,
        text_encoder=encoders.text_encoder,tokenizer=encoders.tokenizer,safety_checker=None,
        feature_extractor=None,requires_safety_checker=False,local_files_only=True).to('mps')
    pipe.set_progress_bar_config(disable=True)
    (args.output/'runtime_scheduler.json').write_text(json.dumps({'class':type(pipe.scheduler).__name__,
        'config':dict(pipe.scheduler.config),'steps':args.steps},indent=2)+'\n')
    records=[]
    with (args.output/'generation_records.jsonl').open('x',buffering=1) as log, torch.no_grad():
        for case in selected:
            s=samples[case['case_id']]
            pair=pairs[case['case_id']]
            out=args.output/case['case_id']
            out.mkdir()
            for role in ('source','target'):
                row=pair['aligned_'+role]
                if sha256(row['aligned_path'])!=row['aligned_sha256']:
                    raise ValueError('Real reference image changed')
                shutil.copyfile(row['aligned_path'],out/(role+'_real.png'))
            patches,text,aus,identity=[s[k][None].to('mps') for k in ('image_patches','text_pooled','target_aus','base_faceid')]
            base_au,base_ip=backbone.baseline_conditions(aus,identity)
            null_au,null_ip=backbone.baseline_conditions(torch.zeros_like(aus),torch.zeros_like(identity))
            for name,adapter,trust,checkpoint in methods:
                values={'au_embedding':base_au,'image_embedding':base_ip} if adapter is None else augment_condition_embeddings(base_au,base_ip,adapter(patches,text,aus),trust_region=trust)
                if not torch.equal(values['au_embedding'][...,:12],base_au[...,:12]):
                    raise RuntimeError('Explicit AU request changed')
                cross={'au_embedding':torch.cat((null_au,values['au_embedding'])),'image_embedding':torch.cat((null_ip,values['image_embedding'])),'au_scale':1.0,'ip_scale':0.9}
                hashes={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in cross.items() if isinstance(v,torch.Tensor)}
                for seed in case_doc['sampling_seeds']:
                    path=out/(name+'_seed'+str(seed)+'.png')
                    record={'case_id':s['case_id'],'subject':s['subject'],'split':'dev','group':case['group'],
                            'method':name,'checkpoint':checkpoint,'seed':seed,'prompt':s['prompt'],
                            'target_aus_0_5':s['target_aus'].tolist(),'source_aus_0_5':s['source_aus'].tolist(),
                            'condition_hashes':hashes,'output':str(path)}
                    started=time.monotonic()
                    try:
                        result=pipe(s['prompt'],negative_prompt='',height=512,width=512,num_inference_steps=args.steps,
                            guidance_scale=args.guidance_scale,generator=torch.Generator(device='cpu').manual_seed(seed),cross_attention_kwargs=cross)
                        result.images[0].save(path)
                        record.update(status='complete',sha256=sha256(path),seconds=time.monotonic()-started)
                    except Exception as error:
                        record.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
                        records.append(record);log.write(json.dumps(record)+'\n')
                        raise
                    records.append(record);log.write(json.dumps(record)+'\n')
                    print(json.dumps({k:record[k] for k in ('case_id','method','status','seconds')}),flush=True)
    (args.output/'result.json').write_text(json.dumps({'status':'complete','images':len(records),'cases':len(selected),
        'methods':len(methods),'test_data_used':False,'formal_sota_evidence':False,
        'all_fixed_cases_and_seeds_retained':True},indent=2)+'\n')


if __name__=='__main__':
    main()
