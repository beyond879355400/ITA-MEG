"""Build a NEW train2k feature cache: exact dev/source reuse, batch-4 VAE.

No model is instantiated on import or in --preflight-only mode. This is a
cache-engineering change, not a training or generation experiment.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
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
import torch
from train_adapter import sha256, read_json, json_new, write_new
from prepare_training_cache import make_prompt

AU_ORDER=[1,2,4,5,6,9,12,15,17,20,25,26]
BATCH_SIZE=4
BATCH_MAX_ABS=2e-4
SHAPES={'image_patches':(49,512),'source_identity':(512,),'base_faceid':(512,),
        'text_tokens':(77,768),'text_pooled':(768,),'target_aus':(12,),'source_aus':(12,),
        'latent_mean':(4,64,64),'latent_std':(4,64,64)}
SOURCE_FIELDS=('image_patches','source_identity','base_faceid')


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def image_inventory(pairs):
    rows=[pair[side] for pair in pairs for side in ('source','target')]
    return {'unique_images':len({row['raw_sha256'] for row in rows}),
            'unique_image_records':len({row['sample_id'] for row in rows}),
            'unique_aligned_images':len({row['aligned_sha256'] for row in rows})}


def compare_samples(left,right):
    if set(left)!=set(right):
        raise ValueError('Cached sample field set changed')
    for key,value in left.items():
        other=right[key]
        if isinstance(value,torch.Tensor):
            if not isinstance(other,torch.Tensor) or value.dtype!=other.dtype or not torch.equal(value,other):
                raise ValueError('Reused tensor changed: '+key)
        elif value!=other:
            raise ValueError('Reused metadata changed: '+key)


def validate_pair(pair,groups):
    role,subject=pair['split'],pair['subject']
    if role not in ('train','dev') or subject not in groups[role] or subject in groups['test'] or pair['au_order']!=AU_ORDER:
        raise ValueError('Pair violates frozen train/dev subject/AU role')
    for side in ('source','target'):
        row=pair[side]
        if row['split']!=role or row['subject']!=subject or row['status']!='aligned':
            raise ValueError('Image metadata role/status mismatch')
        values=pair[side+'_au_vector_0_5']
        labels=[row['input']['labels']['au'+str(au)] for au in AU_ORDER]
        if len(values)!=12 or values!=labels or values!=row['input']['au_vector_0_5'] or row['input']['au_order']!=AU_ORDER:
            raise ValueError('Manual AU values differ from the frozen pair')
        if any(not math.isfinite(x) or not 0<=x<=5 for x in values):
            raise ValueError('Invalid manual AU intensity')


def validate_cached_sample(sample,pair):
    expected_keys=set(SHAPES)|{'case_id','split','subject','prompt'}
    if set(sample)!=expected_keys:
        raise ValueError('Unexpected cache sample schema')
    if any(sample[k]!=pair[k] for k in ('case_id','split','subject')) or sample['prompt']!=make_prompt(pair['target_au_vector_0_5']):
        raise ValueError('Cached identity/prompt differs from its original plan')
    for key,shape in SHAPES.items():
        tensor=sample[key]
        if tensor.shape!=shape or tensor.dtype!=torch.float32 or tensor.device.type!='cpu' or not torch.isfinite(tensor).all():
            raise ValueError('Invalid frozen cache tensor: '+key)
    if (sample['latent_std']<0).any():
        raise ValueError('Negative posterior standard deviation')
    for side in ('source','target'):
        if not torch.equal(sample[side+'_aus'],torch.tensor(pair[side+'_au_vector_0_5'],dtype=torch.float32)):
            raise ValueError('Cached AU differs from original labels')


def make_reuse_index(new_pairs,old_pairs,old_metadata,old_samples):
    old_plan={row['case_id']:row for row in old_pairs}
    old_meta={row['case_id']:row for row in old_metadata}
    samples={row['case_id']:row for row in old_samples}
    if len(old_plan)!=len(old_pairs) or len(samples)!=len(old_samples) or set(old_plan)!=set(samples) or set(old_meta)!=set(samples):
        raise ValueError('Original cache cases are incomplete or duplicated')
    sources={};source_records={};source_case_ids={};text={}
    for case_id,pair in old_plan.items():
        row=samples[case_id]
        validate_cached_sample(row,pair)
        expected_meta={**{k:v for k,v in pair.items() if k not in ('source','target')},'prompt':row['prompt'],'aligned_source':pair['source'],'aligned_target':pair['target']}
        if old_meta[case_id]!=expected_meta:
            raise ValueError('Original metadata differs from original plan')
        source_id=pair['source']['sample_id']
        values={k:row[k] for k in SOURCE_FIELDS}
        if source_id in sources:
            if pair['source']!=source_records[source_id] or any(not torch.equal(values[k],sources[source_id][k]) for k in SOURCE_FIELDS):
                raise ValueError('One source has inconsistent metadata or feature tensors')
        else:
            sources[source_id]=values;source_records[source_id]=pair['source'];source_case_ids[source_id]=case_id
        # New training prompts may reuse TRAIN text encodings only. Dev rows
        # are reused whole and never determine new train prompt features.
        if row['split']=='train':
            values=(row['text_tokens'],row['text_pooled'])
            if row['prompt'] in text and any(not torch.equal(a,b) for a,b in zip(values,text[row['prompt']])):
                raise ValueError('Identical original TRAIN prompts have different encodings')
            text[row['prompt']]=values
    dev={};dev_meta={};requested_sources=set()
    for pair in new_pairs:
        source_id=pair['source']['sample_id'];requested_sources.add(source_id)
        if source_id not in sources or pair['source']!=source_records[source_id]:
            raise ValueError('Original source anchor changed; do not reselect or re-extract it')
        if pair['split']=='dev':
            case_id=pair['case_id']
            if case_id not in old_plan or pair!=old_plan[case_id]:
                raise ValueError('Development pair changed')
            dev[case_id]=samples[case_id];dev_meta[case_id]=old_meta[case_id]
    old_dev={row['case_id'] for row in old_pairs if row['split']=='dev'}
    if set(dev)!=old_dev:
        raise ValueError('Development case coverage changed')
    return {'sources':{k:sources[k] for k in requested_sources},'source_records':{k:source_records[k] for k in requested_sources},
            'source_case_ids':{k:source_case_ids[k] for k in requested_sources},'dev':dev,'dev_metadata':dev_meta,'train_text':text}


def check_batch_outputs(batch_result,single_results,threshold=BATCH_MAX_ABS):
    if threshold!=BATCH_MAX_ABS:
        raise ValueError('The predeclared batch consistency threshold cannot be relaxed')
    batch_mean,batch_std=batch_result
    if len(single_results)!=len(batch_mean) or batch_mean.shape!=batch_std.shape:
        raise ValueError('Batch/single result shape mismatch')
    records=[]
    for index,(mean,std) in enumerate(single_results):
        if mean.shape!=batch_mean[index:index+1].shape or std.shape!=mean.shape:
            raise ValueError('Single-image posterior shape mismatch')
        if not all(torch.isfinite(value).all() for value in (batch_mean[index],batch_std[index],mean,std)):
            raise ValueError('Nonfinite VAE posterior')
        records.append({'index':index,'mean_max_abs':float((batch_mean[index]-mean[0]).abs().max().cpu()),'std_max_abs':float((batch_std[index]-std[0]).abs().max().cpu())})
    maximum=max(max(r['mean_max_abs'],r['std_max_abs']) for r in records)
    return {'passed':maximum<=threshold,'threshold_max_abs':threshold,'maximum_absolute_difference':maximum,'per_image':records,
            'comparison':'same first batch images, batch4 versus individual batch1; scaled VAE posterior mean and std'}


def model_assets():
    from modeling.frozen_backbone import RV,ARC_WEIGHTS
    faceid=ROOT.parent/'baselines/ganimation_emotionnet_20260718/evaluation_models/insightface/models/buffalo_l/w600k_r50.onnx'
    metadata={'rv_snapshot':str(RV),'arc_weights_sha256':sha256(ARC_WEIGHTS),'faceid_condition_weights_sha256':sha256(faceid),
              'vae_sha256':sha256(RV/'vae/diffusion_pytorch_model.safetensors'),'text_encoder_sha256':sha256(RV/'text_encoder/model.safetensors')}
    extra={name:sha256(RV/name) for name in ('vae/config.json','text_encoder/config.json','tokenizer/vocab.json','tokenizer/merges.txt','tokenizer/tokenizer_config.json','tokenizer/special_tokens_map.json')}
    return metadata,extra


def preflight(plan_path,reuse_cache):
    plan_path=plan_path.resolve();reuse_cache=reuse_cache.resolve()
    if any(ROOT not in p.parents for p in (plan_path,reuse_cache)):
        raise ValueError('Only the current independent experiment may supply cache inputs')
    plan=read_json(plan_path);old=read_json(reuse_cache/'manifest.json');old_plan=read_json(reuse_cache/'pair_plan.json')
    if any(doc.get('test_data_used') is not False or doc.get('old_generated_images_or_metrics_used') is not False for doc in (plan,old)):
        raise ValueError('Plan/cache must explicitly exclude test and old generated data')
    if old.get('status')!='complete' or sha256(reuse_cache/'features.pt')!=old['cache_sha256'] or sha256(reuse_cache/'pair_plan.json')!=old['source_manifest_sha256']:
        raise ValueError('Original cache integrity failed')
    for path_key,hash_key in (('alignment_manifest','alignment_sha256'),('train_pair_plan_path','train_pair_plan_sha256'),('previous_pair_plan_path','previous_pair_plan_sha256'),('previous_development_manifest_path','previous_development_manifest_sha256'),('subject_split_path','subject_split_sha256')):
        path=Path(plan[path_key]).resolve()
        if ROOT not in path.parents or sha256(path)!=plan[hash_key]:
            raise ValueError('Frozen upstream manifest changed: '+path_key)
    split=read_json(plan['subject_split_path'])
    if old['subject_split_sha256']!=plan['subject_split_sha256'] or split.get('test_locked') is not True:
        raise ValueError('Subject split binding failed')
    groups={key:set(value) for key,value in split['groups'].items()}
    if any(groups[a]&groups[b] for a,b in (('train','dev'),('train','test'),('dev','test'))):
        raise ValueError('Subject split overlap')
    pairs=plan['pairs']
    counts={role:sum(p['split']==role for p in pairs) for role in ('train','dev')}
    if counts!={'train':2000,'dev':129} or len({p['case_id'] for p in pairs})!=len(pairs):
        raise ValueError('Expected the fixed 2000 train / 129 dev plan')
    checked={};images={}
    for pair in pairs:
        validate_pair(pair,groups)
        for side in ('source','target'):
            row=pair[side]
            if row['sample_id'] in images and images[row['sample_id']]!=row:
                raise ValueError('One sample ID has inconsistent image records')
            images[row['sample_id']]=row
            for path_key,hash_key in (('raw_path','raw_sha256'),('aligned_path','aligned_sha256')):
                path=Path(row[path_key]).resolve()
                if ROOT not in path.parents:
                    raise ValueError('Image outside independent experiment')
                if str(path) not in checked:checked[str(path)]=sha256(path)
                if checked[str(path)]!=row[hash_key]:raise ValueError('Raw/aligned image hash changed: '+str(path))
    previous=torch.load(reuse_cache/'features.pt',map_location='cpu',weights_only=True)
    if previous.get('schema_version')!=2:raise ValueError('Unexpected original feature schema')
    index=make_reuse_index(pairs,old_plan['pairs'],old['pairs'],previous['samples'])
    if len(index['sources'])!=19:raise ValueError('Expected all 19 unchanged source anchors')
    assets,configs=model_assets()
    if any(old['model_assets'][key]!=value for key,value in assets.items()):
        raise ValueError('Feature encoder weights changed since the reused cache')
    receipt={'status':'preflight_passed','plan_sha256':sha256(plan_path),'reuse_cache_sha256':old['cache_sha256'],'reuse_manifest_sha256':sha256(reuse_cache/'manifest.json'),
             'subject_split_sha256':plan['subject_split_sha256'],'counts':counts,'sources_reused':len(index['sources']),'dev_samples_reused':len(index['dev']),
             'all_raw_aligned_sha_verified':True,'verified_image_files':len(checked),'model_assets':assets,'encoder_config_sha256':configs,
             'source_tensor_sha256':{sid:{key:tensor_sha(value) for key,value in data.items()} for sid,data in sorted(index['sources'].items())},
             'reuse_source_case_ids':index['source_case_ids'],'dev_metadata_exact':True,'training_started':False,'generation_started':False}
    return plan,index,receipt


class CacheEncoders:
    def __init__(self,device,need_text):
        from diffusers import AutoencoderKL
        from modeling.frozen_backbone import RV
        self.device=device
        self.vae=AutoencoderKL.from_pretrained(str(RV/'vae'),local_files_only=True,torch_dtype=torch.float32).requires_grad_(False).eval().to(device)
        self.text_encoder=None
        if need_text:
            from transformers import CLIPTextModel,CLIPTokenizer
            self.text_encoder=CLIPTextModel.from_pretrained(str(RV/'text_encoder'),local_files_only=True,torch_dtype=torch.float32).requires_grad_(False).eval().to(device)
            self.tokenizer=CLIPTokenizer.from_pretrained(str(RV/'tokenizer'),local_files_only=True)

    def encode_latents(self,tensor):
        posterior=self.vae.encode(tensor).latent_dist
        return posterior.mean*self.vae.config.scaling_factor,posterior.std*self.vae.config.scaling_factor

    def encode_text(self,prompt):
        tokens=self.tokenizer([prompt],padding='max_length',max_length=77,truncation=True,return_tensors='pt')
        result=self.text_encoder(tokens.input_ids.to(self.device))
        return result.last_hidden_state[0].cpu(),result.pooler_output[0].cpu()


def image_tensor(row):
    import numpy as np
    from PIL import Image
    if sha256(row['aligned_path'])!=row['aligned_sha256']:
        raise ValueError('Aligned image changed after preflight')
    with Image.open(row['aligned_path']) as image:
        if image.size!=(512,512):raise ValueError('Aligned image must be exactly 512x512')
        rgb=np.asarray(image.convert('RGB'),dtype=np.float32)/127.5-1
    return torch.from_numpy(rgb.transpose(2,0,1).copy())


def encode_new_training_targets(plan,index,output,device):
    targets={p['target']['sample_id']:p['target'] for p in plan['pairs'] if p['split']=='train'}
    ordered=sorted(targets)
    if len(ordered)!=2000:raise ValueError('The frozen plan must retain 2000 distinct train targets')
    prompts=sorted({make_prompt(p['target_au_vector_0_5']) for p in plan['pairs'] if p['split']=='train'})
    missing_prompts=[p for p in prompts if p not in index['train_text']]
    encoder=CacheEncoders(device,bool(missing_prompts))
    latents={}
    sync=torch.mps.synchronize if device=='mps' else lambda:None
    with torch.no_grad(),(output/'encoding_batches.jsonl').open('x',buffering=1) as log:
        for begin in range(0,len(ordered),BATCH_SIZE):
            ids=ordered[begin:begin+BATCH_SIZE]
            tensor=torch.stack([image_tensor(targets[sid]) for sid in ids]).to(device)
            sync();start=time.monotonic()
            mean,std=encoder.encode_latents(tensor)
            if begin==0:
                singles=[encoder.encode_latents(tensor[i:i+1]) for i in range(len(ids))]
                check=check_batch_outputs((mean,std),singles)
                json_new(output/'vae_batch_equivalence.json',{**check,'sample_ids':ids,'batch_size':BATCH_SIZE,'device':device})
                if not check['passed']:raise RuntimeError('Batch VAE difference exceeded fixed 2e-4 threshold; stop without cache publication')
            if mean.shape!=(len(ids),4,64,64) or std.shape!=mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<0).any():
                raise ValueError('Invalid batched latent posterior')
            for i,sid in enumerate(ids):latents[sid]=(mean[i].cpu().clone(),std[i].cpu().clone())
            sync();record={'batch':begin//BATCH_SIZE,'sample_ids':ids,'completed_images':begin+len(ids),'seconds':time.monotonic()-start}
            log.write(json.dumps(record)+'\n')
            if begin%(BATCH_SIZE*10)==0:print(json.dumps(record),flush=True)
        for prompt in missing_prompts:index['train_text'][prompt]=encoder.encode_text(prompt)
    return latents,{'train_target_images_encoded':len(ordered),'vae_batch_size':BATCH_SIZE,'vae_batches':(len(ordered)+BATCH_SIZE-1)//BATCH_SIZE,
                    'train_prompt_encodings_reused':len(prompts)-len(missing_prompts),'new_prompt_encodings':len(missing_prompts),
                    'source_patch_id_model_loaded':False,'vae_posterior_sampled':False}


def assemble_samples(plan,index,latents):
    samples=[];metadata=[]
    for pair in plan['pairs']:
        if pair['split']=='dev':
            # Absolutely no field additions: keep the original object payload.
            row=index['dev'][pair['case_id']];meta=index['dev_metadata'][pair['case_id']]
        else:
            source=index['sources'][pair['source']['sample_id']]
            mean,std=latents[pair['target']['sample_id']]
            prompt=make_prompt(pair['target_au_vector_0_5']);tokens,pooled=index['train_text'][prompt]
            row={'case_id':pair['case_id'],'split':pair['split'],'subject':pair['subject'],'prompt':prompt,**source,
                 'text_tokens':tokens,'text_pooled':pooled,'target_aus':torch.tensor(pair['target_au_vector_0_5'],dtype=torch.float32),
                 'source_aus':torch.tensor(pair['source_au_vector_0_5'],dtype=torch.float32),'latent_mean':mean,'latent_std':std}
            meta={**{k:v for k,v in pair.items() if k not in ('source','target')},'prompt':prompt,'aligned_source':pair['source'],'aligned_target':pair['target']}
        validate_cached_sample(row,pair)
        samples.append(row);metadata.append(meta)
    return samples,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'training_cache/train2k_fixeddev_202609141836/pair_plan.json')
    parser.add_argument('--reuse-cache',type=Path,default=ROOT/'training_cache/expanded_v2_202609141711')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','mps'),default='mps')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    if ROOT not in args.output.parents or args.output.exists():raise ValueError('Choose a NEW cache directory, separate from the frozen plan and original cache')
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'prepare_source.py')
    status={'status':'running','device':'cpu' if args.preflight_only else args.device,'requested_encoding_device':args.device,
            'encoding_started':False,'batch_size':BATCH_SIZE,'script_sha256':sha256(__file__),'old_cache_modified':False,'plan_modified':False}
    start=time.monotonic()
    try:
        torch.set_num_threads(1);torch.manual_seed(20260914)
        plan,index,receipt=preflight(args.plan,args.reuse_cache)
        json_new(args.output/'preflight.json',receipt)
        if args.preflight_only:
            status.update(status='preflight_passed',encoding_started=False)
        else:
            if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
            # Copy bytes, not a reserialized/reselected plan.
            with args.plan.open('rb') as source:
                write_new(args.output/'pair_plan.json',lambda destination:shutil.copyfileobj(source,destination))
            status['encoding_started']=True
            latents,encoding=encode_new_training_targets(plan,index,args.output,args.device)
            samples,metadata=assemble_samples(plan,index,latents)
            feature_path=args.output/'features.pt'
            write_new(feature_path,lambda handle:torch.save({'schema_version':2,'samples':samples},handle))
            # Read back the published tensor file and prove every dev field and
            # every reused source feature survived serialization identically.
            saved=torch.load(feature_path,map_location='cpu',weights_only=True)['samples']
            if len(saved)!=len(samples):raise ValueError('Serialized sample count changed')
            for row,pair in zip(saved,plan['pairs']):
                validate_cached_sample(row,pair)
                if row['split']=='dev':compare_samples(index['dev'][row['case_id']],row)
                for key in SOURCE_FIELDS:
                    if not torch.equal(row[key],index['sources'][pair['source']['sample_id']][key]):raise ValueError('Reused source tensor changed on serialization')
            reused_dev_metadata=[index['dev_metadata'][p['case_id']] for p in plan['pairs'] if p['split']=='dev']
            if [p for p in metadata if p['split']=='dev']!=reused_dev_metadata:raise ValueError('Development metadata changed')
            manifest={'status':'complete','schema_version':2,'test_data_used':False,'old_generated_images_or_metrics_used':False,
                'source_manifest_sha256':sha256(args.output/'pair_plan.json'),'source_manifest_path':str(args.output/'pair_plan.json'),
                'subject_split_sha256':plan['subject_split_sha256'],'subject_split_path':plan['subject_split_path'],
                'subject_splits':{r:sorted({p['subject'] for p in plan['pairs'] if p['split']==r}) for r in ('train','dev')},
                'samples':len(saved),**image_inventory(plan['pairs']),
                'patch_shape':[49,512],'resolution':512,'device':args.device,'cache_path':str(feature_path),'cache_sha256':sha256(feature_path),
                'model_assets':receipt['model_assets'],'encoder_config_sha256':receipt['encoder_config_sha256'],
                'base_faceid_encoder':'buffalo_l/w600k_r50','patch_and_id_loss_encoder':'IResNet100/glint360k',
                'elapsed_seconds':time.monotonic()-start,'pairs':metadata,'encoding':encoding,
                'reuse':{'source_cache_path':str(args.reuse_cache.resolve()),'source_cache_sha256':receipt['reuse_cache_sha256'],'source_manifest_sha256':receipt['reuse_manifest_sha256'],
                        'source_features':19,'dev_samples':129,'dev_sample_all_fields_and_tensors_exact':True,'dev_metadata_all_fields_exact':True,
                        'source_features_all_fields_exact':True,'source_tensor_sha256':receipt['source_tensor_sha256'],'source_case_ids':receipt['reuse_source_case_ids'],
                        'scope':'features from this independent experiment real-image encoder cache only; no generated images or prior experiment metrics'},
                'batch_equivalence':read_json(args.output/'vae_batch_equivalence.json'),
                'provenance':{'script_sha256':sha256(__file__),'original_frozen_plan_path':str(args.plan.resolve()),'original_frozen_plan_sha256':receipt['plan_sha256'],
                    'reference_builder_sha256':sha256(ROOT/'prepare_expanded_training_cache.py'),'frozen_backbone_sha256':sha256(ROOT/'modeling/frozen_backbone.py'),
                    'prompt_source_sha256':sha256(ROOT/'prepare_training_cache.py'),'utility_source_sha256':sha256(ROOT/'train_adapter.py'),
                    'environment':{name:importlib.metadata.version(name) for name in ('torch','diffusers','transformers','numpy','Pillow')}}}
            if manifest['source_manifest_sha256']!=receipt['plan_sha256']:raise ValueError('Copied plan bytes changed')
            json_new(args.output/'manifest.json',manifest)
            status.update(status='complete',samples=len(saved),cache_sha256=manifest['cache_sha256'],encoding=encoding)
    except Exception as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    status['elapsed_seconds']=time.monotonic()-start
    json_new(args.output/'result.json',status);print(json.dumps(status,indent=2),flush=True)
    return 0 if status['status'] in ('complete','preflight_passed') else 1


if __name__=='__main__':raise SystemExit(main())
