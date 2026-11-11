"""Fresh v4 source-centered sampling on a pre-frozen development case list."""
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


def validate_development_manifest(case_path, case_doc, samples, cache_sha, split_sha, checkpoint_bindings=None):
    """Exact frozen cases, or an explicit dev-only append to checkpoint cases."""
    case_path=Path(case_path).resolve()
    if ROOT not in case_path.parents or json.loads(case_path.read_text())!=case_doc:
        raise ValueError('Development manifest path/content changed')
    actual_sha=hashlib.sha256(case_path.read_bytes()).hexdigest()
    def check_document(document):
        if document.get('test_data_used') is not False or document.get('cache_sha256')!=cache_sha or document.get('subject_split_sha256')!=split_sha:
            raise ValueError('Development document cache/split/test binding failed')
        seeds=document.get('sampling_seeds')
        if not isinstance(seeds,list) or not seeds or any(type(seed) is not int or seed<0 for seed in seeds) or len(set(seeds))!=len(seeds):
            raise ValueError('Invalid or duplicate frozen sampling seeds')
    check_document(case_doc)
    cases=case_doc.get('cases',[])
    ids=[case['case_id'] for case in cases]
    if not ids or len(set(ids))!=len(ids):
        raise ValueError('Empty or duplicate development cases')
    allowed={key for key,row in samples.items() if row['split']=='dev'}
    expected_sha=None
    if checkpoint_bindings is not None:
        if checkpoint_bindings.get('cache')!=cache_sha or checkpoint_bindings.get('split')!=split_sha:
            raise ValueError('Protocol checkpoint belongs to another cache/split')
        expected_sha=checkpoint_bindings['development_subset']['manifest_sha256']
        allowed &= set(checkpoint_bindings['all_dev_case_ids'])
    for case in cases:
        case_id=case['case_id']
        if case_id not in allowed:
            raise ValueError('Case is outside checkpoint/cache development set')
        row=samples[case_id]
        if case.get('split')!='dev' or case.get('subject')!=row['subject']:
            raise ValueError('Development subject/role changed')
        for cached_key,case_key in (('source_aus','source_au_vector_0_5'),('target_aus','target_au_vector_0_5')):
            value=row[cached_key]
            value=value.tolist() if hasattr(value,'tolist') else value
            if case.get(case_key)!=value:
                raise ValueError('Development source/target AU labels changed')
    parent_sha=None;parent_path=None;parent_count=0
    if case_doc.get('status')=='frozen_for_development':
        if expected_sha is not None and actual_sha!=expected_sha:
            raise ValueError('Exact development manifest differs from checkpoint')
        if checkpoint_bindings is not None and ids!=checkpoint_bindings['dev_case_ids']:
            raise ValueError('Exact development case order differs from checkpoint')
        policy='exact frozen development manifest'
    elif case_doc.get('status')=='frozen_additional_development':
        if checkpoint_bindings is None:
            raise ValueError('Additional development requires a protocol checkpoint; supply --checkpoint (baseline plus model)')
        parent_path=Path(case_doc['source_manifest_path']).resolve()
        if ROOT not in parent_path.parents:
            raise ValueError('Parent manifest must be inside this experiment')
        parent_sha=hashlib.sha256(parent_path.read_bytes()).hexdigest()
        if parent_sha!=case_doc.get('source_manifest_sha256') or parent_sha!=expected_sha:
            raise ValueError('Parent manifest SHA differs from the protocol checkpoint')
        parent=json.loads(parent_path.read_text())
        check_document(parent)
        if parent.get('status')!='frozen_for_development':
            raise ValueError('Only a direct append to an exact frozen parent is supported')
        original=parent['cases'];parent_count=len(original)
        if [case['case_id'] for case in original]!=checkpoint_bindings['dev_case_ids']:
            raise ValueError('Parent cases differ from the protocol checkpoint')
        if len(cases)<=parent_count or cases[:parent_count]!=original:
            raise ValueError('The complete original case prefix/order must remain unchanged')
        if case_doc['sampling_seeds']!=parent['sampling_seeds']:
            raise ValueError('Additional development must preserve all original sampling seeds')
        policy='frozen dev-only superset; original full prefix and seeds unchanged; no test selection'
    else:
        raise ValueError('Unsupported development manifest status')
    return {'policy':policy,'manifest_sha256':actual_sha,'parent_manifest_sha256':parent_sha,
            'parent_manifest_path':None if parent_path is None else str(parent_path),
            'checkpoint_original_manifest_sha256':expected_sha,'original_case_count':parent_count if parent_sha else len(cases),
            'case_count':len(cases),'additional_case_ids':ids[parent_count:] if parent_sha else [],
            'sampling_seeds':case_doc['sampling_seeds'],'selection':case_doc.get('selection'),
            'all_cases_are_frozen_development':True,'test_data_used':False}


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
    from train_centered_adapter import load_checkpoint,MODEL_VERSION,source_and_asset_bindings,resolve_source_anchor,verify_source_anchor,adapter_from_checkpoint,centered_batch_conditions
    meta=json.loads((args.cache/'manifest.json').read_text())
    case_doc=json.loads(args.cases.read_text())
    actual_sha=sha256(args.cache/'features.pt')
    if meta['test_data_used'] or case_doc['test_data_used'] or actual_sha!=meta['cache_sha256'] or actual_sha!=case_doc['cache_sha256'] or meta['subject_split_sha256']!=case_doc['subject_split_sha256']:
        raise ValueError('Cache/case/split bindings mismatch')
    cached=torch.load(args.cache/'features.pt',map_location='cpu',weights_only=True)['samples']
    if any(r['split'] not in ('train','dev') for r in cached):
        raise ValueError('Source anchor/cache must not include test records')
    source_anchor,source_anchor_binding=resolve_source_anchor([r for r in cached if r['split']=='train'],[r for r in cached if r['split']=='dev'])
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
    manifest_receipts=[]
    if not args.checkpoint:
        manifest_receipts.append(validate_development_manifest(args.cases,case_doc,samples,actual_sha,meta['subject_split_sha256']))
    current_public,_=source_and_asset_bindings({})
    for checkpoint in args.checkpoint:
        if ROOT not in checkpoint.resolve().parents:
            raise ValueError('Only this experiment checkpoints accepted')
        state=load_checkpoint(checkpoint)
        if state['model_version']!=MODEL_VERSION or state['bindings']['cache']!=actual_sha or state['bindings']['split']!=meta['subject_split_sha256']:
            raise ValueError('Checkpoint is not bound to this exact data/version')
        if state['bindings']['public_assets']!=current_public['public_assets']:
            raise ValueError('Public model weights/configuration differ from training')
        manifest_receipts.append(validate_development_manifest(args.cases,case_doc,samples,actual_sha,meta['subject_split_sha256'],state['bindings']))
        verify_source_anchor(source_anchor,source_anchor_binding,state['bindings']['source_anchor'])
        for source in ('modeling/cross_modal_adapter.py','modeling/token_residual_adapter.py','modeling/safe_token_residual_adapter.py','modeling/source_centered_adapter.py','modeling/frozen_backbone.py','train_centered_adapter.py'):
            if state['bindings']['sources'][source]!=sha256(ROOT/source):
                raise ValueError('Model source differs from the trained source')
        variant=state['variant']
        model=adapter_from_checkpoint(state)
        model.eval().requires_grad_(False).to('mps')
        e=state['effective_config']
        trust=RelativeRMSConfig(enabled=e['trust_region'],rho_au=e['rho_au'],rho_ip=e['rho_ip'],reference_rms_floor=e['reference_rms_floor'])
        name='source_centered_'+variant+'_step'+str(state['global_step'])+'_'+sha256(checkpoint)[:8]
        methods.append((name,model,trust,str(checkpoint.resolve())))
        seals.append({'name':name,'checkpoint_sha256':sha256(checkpoint),'effective_config':e,
                      'model_config':state['model_config'],'diagnostic_training':state['diagnostic']})
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'generation_source.py')
    shutil.copyfile(args.cases,args.output/'frozen_cases.json')
    (args.output/'settings.json').write_text(json.dumps({'cache_sha256':actual_sha,'case_manifest_sha256':sha256(args.cases),
        'preview_group':args.preview_group,'steps':args.steps,'guidance_scale':args.guidance_scale,'skip_baseline':args.skip_baseline,
        'sampling_seeds':case_doc['sampling_seeds'],'au_scale':1.0,'ip_scale':0.9,'methods':seals,
        'public_assets':current_public['public_assets'],'model_version':MODEL_VERSION,'source_anchor':source_anchor_binding,'adapter_forward_calls_per_request':2,
        'development_manifest_binding':manifest_receipts,
        'test_data_used':False,'formal_sota_evidence':False},indent=2)+'\n')
    backbone=FrozenConditionedDenoiser('mps',checkpointing=False)
    encoders=FrozenImageTextEncoders('mps')
    pipe=StableDiffusionPipeline.from_pretrained(str(RV),unet=backbone.unet,vae=encoders.vae,
        text_encoder=encoders.text_encoder,tokenizer=encoders.tokenizer,safety_checker=None,
        feature_extractor=None,requires_safety_checker=False,local_files_only=True).to('mps')
    pipe.set_progress_bar_config(disable=True)
    source_anchor=source_anchor.to('mps')
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
                if adapter is None:
                    values={'au_embedding':base_au,'image_embedding':base_ip}
                else:
                    centered_batch={'image_patches':patches,'text_pooled':text,'target_aus':aus,'source_aus':s['source_aus'][None].to('mps')}
                    _,_,values=centered_batch_conditions(adapter,centered_batch,source_anchor,base_au,base_ip,trust)
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
