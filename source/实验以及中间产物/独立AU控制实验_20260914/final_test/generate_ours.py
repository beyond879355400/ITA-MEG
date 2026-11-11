"""Frozen final-test source+text+AU generation, independent of dev-only entry.

The pipeline never receives a target path or target pixels. Labels and frozen
cached text/features are its only per-request inputs. No final protocol is
created or selected by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from protocol_gate import validate_control,verify_runtime_source,require_declared_artifact,digest,read_json,resolve_path
from prepare_source_cache import (new_directory,save_json,load_train_anchor,source_only_plan,
                                  check_tensor,tensor_sha,verify_ref,runtime_environment,validate_source_feature_settings)

IMPLEMENTATIONS=('source_centered_v4','fineface_faceid_baseline')


def validate_methods(control,methods):
    if not methods or len(set(methods))!=len(methods):raise ValueError('Explicit unique method list required')
    result={}
    for method in methods:
        if method not in control['models'] or not re.fullmatch(r'[A-Za-z0-9_.-]+',method):raise ValueError('Unplanned/unsafe method name')
        settings=control['models'][method]['settings']
        if settings.get('implementation') not in IMPLEMENTATIONS or settings.get('input_group')!='source_text_au':
            raise ValueError('This entry accepts source+text+AU methods only; driver-image methods are separate')
        if settings.get('seed_execution')!={'mode':'stochastic_each_seed'}:raise ValueError('Our diffusion methods require every frozen sampling seed')
        p=settings.get('native_parameters',{})
        required={'height','width','num_inference_steps','guidance_scale','au_scale','ip_scale','scheduler','negative_prompt'}
        if set(p)!=required:raise ValueError('Sampling parameters must all be explicit, no hidden defaults')
        if p['height']!=512 or p['width']!=512 or type(p['num_inference_steps']) is not int or not 10<=p['num_inference_steps']<=50:
            raise ValueError('Only validated512 sampling at10..50steps is supported')
        if not 1<float(p['guidance_scale'])<=15 or p['au_scale']!=1.0 or p['ip_scale'] not in (.9,1.0):
            raise ValueError('CFG/AU/IP settings are outside validated explicit bounds')
        if p['scheduler']!='DEISMultistepScheduler' or p['negative_prompt']!='':raise ValueError('Scheduler/negative prompt changed')
        runtime=settings.get('runtime',{})
        runtime_fields={'device','dtype'}|({'adapter_checkpoint'} if settings['implementation']=='source_centered_v4' else set())
        if set(runtime)!=runtime_fields:raise ValueError('Runtime fields must be explicit; unsupported precision/ignored fields rejected')
        if runtime.get('device') not in ('cpu','mps') or runtime.get('dtype')!='torch.float32':raise ValueError('Explicit CPU/MPS torch.float32 runtime required')
        if settings['implementation']=='source_centered_v4':require_declared_artifact(control,runtime['adapter_checkpoint'],method)
        result[method]=settings
    if len({s['runtime']['device'] for s in result.values()})!=1:raise ValueError('One invocation requires one explicit device')
    return result


def load_source_cache(control,path,expected_sha,*,load_tensors=True):
    """Read source-only tensors and source image hashes, never target images."""
    path=Path(path).resolve()
    if digest(path)!=expected_sha:raise ValueError('Explicit source cache manifest SHA mismatch')
    cache=read_json(path)
    if cache.get('kind')!='final_test_source_feature_cache' or cache.get('status') not in ('complete','completed_with_failures'):
        raise ValueError('Source cache is not terminal')
    if cache.get('protocol_sha256')!=control['protocol_sha256'] or cache.get('raw_manifest')!={'path':control['raw_manifest_path'],'sha256':control['raw_manifest_sha256']}:
        raise ValueError('Source cache protocol/raw binding mismatch')
    if cache.get('case_ids')!=control['case_ids'] or cache.get('sampling_seeds')!=control['sampling_seeds']:
        raise ValueError('Source cache full case/seed list changed')
    if cache.get('target_pixels_read') is not False or cache.get('target_latents_computed') is not False or cache.get('vae_loaded') is not False:
        raise ValueError('Source cache crossed target/VAE boundary')
    if cache.get('source_cache_code_sha256')!=control['protocol']['execution_sources']['source_cache']['sha256'] or cache.get('protocol_gate_sha256')!=control['protocol']['execution_sources']['protocol_gate']['sha256']:
        raise ValueError('Cache implementation was not frozen')
    feature_settings=validate_source_feature_settings(control)
    if cache.get('device')!=feature_settings['device'] or cache.get('dtype')!=feature_settings['dtype']:raise ValueError('Source-feature device/dtype differs from frozen policy')
    expected_sources,expected_cases=source_only_plan(control)
    if len(cache.get('sources',[]))!=len(expected_sources) or len(cache.get('cases',[]))!=len(expected_cases):raise ValueError('Source/cache row count changed')
    source_map={}
    for actual,expected in zip(cache['sources'],expected_sources):
        if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError('Source metadata differs from raw plan')
        if actual.get('status') not in ('complete','failed'):raise ValueError('Incomplete source row')
        if actual['status']=='complete':
            ref=actual['canonical_source'];verify_ref(ref,'canonical SOURCE')
            if (ref.get('width'),ref.get('height'),ref.get('mode'))!=(512,512,'RGB'):raise ValueError('Canonical source space changed')
        elif actual.get('canonical_source') is not None:raise ValueError('Failed source cannot carry a success image')
        source_map[actual['source_id']]=actual
    for actual,expected in zip(cache['cases'],expected_cases):
        if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError('Cached case/manual labels or target-reference metadata changed')
        if actual.get('condition_status') not in ('complete','failed'):raise ValueError('Incomplete text-condition row')
    feature_path=verify_ref(cache['features_file'],'source-only features')
    if not load_tensors:return cache,None
    import torch
    features=torch.load(feature_path,map_location='cpu',weights_only=True)
    if set(features)!={'source_anchor','sources','conditions'}:raise ValueError('Unexpected source cache fields; target latents forbidden')
    check_tensor(features['source_anchor'],(768,),'source_anchor')
    if tensor_sha(features['source_anchor'])!=cache['source_anchor']['binding']['text_pooled_sha256']:raise ValueError('Source anchor tensor changed')
    expected_source_keys={s['source_id'] for s in cache['sources'] if s['status']=='complete'}
    if set(features['sources'])!=expected_source_keys:raise ValueError('Source tensor success set changed')
    for tensors in features['sources'].values():
        shapes={'image_patches':(49,512),'source_identity':(512,),'base_faceid':(512,)}
        if set(tensors)!=set(shapes):raise ValueError('Unexpected source tensor fields')
        for key,shape in shapes.items():check_tensor(tensors[key],shape,key)
    condition_ids={c['case_id'] for c in cache['cases'] if c['condition_status']=='complete'}
    if set(features['conditions'])!=condition_ids:raise ValueError('Condition tensor success set changed')
    for case in cache['cases']:
        if case['condition_status']!='complete':continue
        values=features['conditions'][case['case_id']]
        shapes={'text_tokens':(77,768),'text_pooled':(768,),'source_aus':(12,),'target_aus':(12,)}
        if set(values)!=set(shapes)|{'prompt'}:raise ValueError('Unexpected condition/target latent fields')
        for key,shape in shapes.items():check_tensor(values[key],shape,key)
        for key,label in (('source_aus','source_au_vector_0_5'),('target_aus','target_au_vector_0_5')):
            if not torch.equal(values[key],torch.tensor(case[label],dtype=torch.float32)):raise ValueError('Cached AU values changed')
        if values['prompt']!=case['prompt']:raise ValueError('Frozen label-derived prompt changed')
        if not any(case['target_au_vector_0_5']) and not torch.equal(values['text_pooled'],features['source_anchor']):raise ValueError('No-edit target text differs from TRAIN source anchor')
    return cache,features


class OurRuntime:
    def __init__(self,control,methods,cache,features):
        import torch
        from diffusers import StableDiffusionPipeline,AutoencoderKL
        from transformers import CLIPTextModel,CLIPTokenizer
        from modeling.frozen_backbone import FrozenConditionedDenoiser,RV
        from modeling.safe_token_residual_adapter import RelativeRMSConfig
        from train_centered_adapter import load_checkpoint,MODEL_VERSION,source_and_asset_bindings,adapter_from_checkpoint,verify_source_anchor
        self.settings=methods;self.device=next(iter(methods.values()))['runtime']['device'];self.features=features
        anchor,receipt,anchor_payload=load_train_anchor(control)
        if receipt!=cache['source_anchor'] or not torch.equal(anchor,features['source_anchor']):raise ValueError('Cached source anchor differs from checkpoint/TRAIN cache')
        current,_=source_and_asset_bindings({})
        if current['public_assets']!=anchor_payload['bindings']['public_assets']:raise ValueError('Public base weights differ from anchor checkpoint')
        self.methods={};self.seals={}
        for method,settings in methods.items():
            if settings['implementation']=='fineface_faceid_baseline':self.methods[method]=(None,None);continue
            ref=settings['runtime']['adapter_checkpoint'];checkpoint=require_declared_artifact(control,ref,method)
            state=load_checkpoint(checkpoint)
            if state.get('model_version')!=MODEL_VERSION or state.get('diagnostic') is not False or state.get('eligible_for_final_selection') is not True:
                raise ValueError('Final generation requires explicitly eligible non-diagnosticv4 checkpoint')
            if state['bindings']['public_assets']!=current['public_assets'] or state['bindings']['cache']!=anchor_payload['bindings']['cache'] or state['bindings']['split']!=anchor_payload['bindings']['split']:
                raise ValueError('Method checkpoint public base/cache/split differs')
            verify_source_anchor(anchor,receipt['binding'],state['bindings']['source_anchor'])
            for source,expected in state['bindings']['sources'].items():
                if digest(ROOT/source)!=expected:raise ValueError('Frozen trained source changed: '+source)
            adapter=adapter_from_checkpoint(state,self.device).eval().requires_grad_(False)
            e=state['effective_config'];trust=RelativeRMSConfig(enabled=e['trust_region'],rho_au=e['rho_au'],rho_ip=e['rho_ip'],reference_rms_floor=e['reference_rms_floor'])
            self.methods[method]=(adapter,trust)
            self.seals[method]={'checkpoint':ref,'variant':state['variant'],'global_step':state['global_step'],'model_config':state['model_config'],'effective_config':e}
        self.backbone=FrozenConditionedDenoiser(self.device,checkpointing=False)
        # VAE is used only to DECODE generated latents. No test target is available.
        vae=AutoencoderKL.from_pretrained(str(RV/'vae'),local_files_only=True,torch_dtype=torch.float32)
        text=CLIPTextModel.from_pretrained(str(RV/'text_encoder'),local_files_only=True,torch_dtype=torch.float32)
        tokenizer=CLIPTokenizer.from_pretrained(str(RV/'tokenizer'),local_files_only=True)
        self.pipe=StableDiffusionPipeline.from_pretrained(str(RV),unet=self.backbone.unet,vae=vae,text_encoder=text,tokenizer=tokenizer,
            safety_checker=None,feature_extractor=None,requires_safety_checker=False,local_files_only=True).to(self.device)
        for module in (vae,text):module.eval().requires_grad_(False)
        self.pipe.set_progress_bar_config(disable=True)
        if type(self.pipe.scheduler).__name__!='DEISMultistepScheduler':raise ValueError('Actual scheduler is not the frozen DEIS sampler')
        self.runtime_receipt={'scheduler_class':type(self.pipe.scheduler).__name__,'scheduler_config':dict(self.pipe.scheduler.config),
            'device':self.device,'dtype':'torch.float32','cpu_generator_per_sampling_seed':True,'public_assets':current['public_assets'],
            'methods':self.seals,'target_image_encoder_calls':0,'environment':runtime_environment()}

    def prepare(self,method,case_id,source_id):
        import torch
        from train_centered_adapter import centered_batch_conditions
        # This whitelist excludes target_reference metadata/pixels entirely.
        source=self.features['sources'][source_id];condition=self.features['conditions'][case_id]
        batch={k:condition[k][None].to(self.device) for k in ('text_pooled','target_aus','source_aus')}
        batch['image_patches']=source['image_patches'][None].to(self.device)
        identity=source['base_faceid'][None].to(self.device);aus=batch['target_aus']
        with torch.no_grad():
            base_au,base_ip=self.backbone.baseline_conditions(aus,identity)
            null_au,null_ip=self.backbone.baseline_conditions(torch.zeros_like(aus),torch.zeros_like(identity))
            adapter,trust=self.methods[method]
            if adapter is None:values={'au_embedding':base_au,'image_embedding':base_ip}
            else:_,_,values=centered_batch_conditions(adapter,batch,self.features['source_anchor'].to(self.device),base_au,base_ip,trust)
            if not torch.equal(values['au_embedding'][...,:12],base_au[...,:12]):raise RuntimeError('Explicit12AU prefix changed')
        p=self.settings[method]['native_parameters']
        cross={'au_embedding':torch.cat((null_au,values['au_embedding'])),'image_embedding':torch.cat((null_ip,values['image_embedding'])),
               'au_scale':p['au_scale'],'ip_scale':p['ip_scale']}
        return {'prompt':condition['prompt'],'cross':cross,'condition_hashes':{key:tensor_sha(value) for key,value in cross.items() if isinstance(value,torch.Tensor)}}

    def sample(self,method,prepared,seed):
        import torch
        p=self.settings[method]['native_parameters']
        with torch.no_grad():
            result=self.pipe(prepared['prompt'],negative_prompt=p['negative_prompt'],height=p['height'],width=p['width'],
                num_inference_steps=p['num_inference_steps'],guidance_scale=p['guidance_scale'],
                generator=torch.Generator(device='cpu').manual_seed(seed),cross_attention_kwargs=prepared['cross'])
        return result.images[0]


def run_grid(control,methods,cache,output,runtime,source_cache_reference=None):
    """Complete method×case×seed accounting; injected runtime is synthetic-testable."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[];sources={s['source_id']:s for s in cache['sources']};cases={c['case_id']:c for c in cache['cases']}
    save_json(output/'settings.json',{'protocol_sha256':control['protocol_sha256'],'method_settings':methods,
        'source_cache_manifest':source_cache_reference,'source_features_file':cache['features_file'],
        'case_ids':control['case_ids'],'sampling_seeds':control['sampling_seeds'],'target_pixels_used':False,
        'runtime':runtime.runtime_receipt,'generator_source_sha256':digest(__file__),'gate_source_sha256':digest(HERE/'protocol_gate.py')})
    shutil.copyfile(__file__,output/'generator_snapshot.py')
    with (output/'generation_records.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for method in methods:
            for case_id in control['case_ids']:
                case=cases[case_id];source=sources[case['source_id']];prepared=None;pre_error=None
                if source['status']!='complete' or case['condition_status']!='complete':pre_error='source_or_text_condition_failed'
                else:
                    try:prepared=runtime.prepare(method,case_id,case['source_id'])
                    except Exception as error:pre_error=type(error).__name__+': '+str(error)
                for seed in control['sampling_seeds']:
                    record={'method':method,'subject':case['subject'],'case_id':case_id,'seed':seed,'split':'test',
                            'execution_id':method+'::'+case_id+'::'+str(seed),'execution_started':False,'alias_of':None,'output':None}
                    if pre_error:record.update(status='not_started',error=pre_error)
                    else:
                        started=time.monotonic()
                        try:
                            record['execution_started']=True
                            image=runtime.sample(method,prepared,seed)
                            if image.mode!='RGB' or image.size!=(512,512):raise ValueError('Expected native512 RGB output')
                            folder=output/'images'/method/case_id;folder.mkdir(parents=True,exist_ok=True)
                            path=folder/(str(seed)+'.png')
                            if path.exists():raise ValueError('Refusing to overwrite generated output')
                            image.save(path)
                            record.update(status='complete',output={'path':str(path.resolve()),'sha256':digest(path),'width':512,'height':512},
                                          condition_hashes=prepared.get('condition_hashes',{}))
                        except Exception as error:record.update(status='failed',error=type(error).__name__+': '+str(error))
                        record['seconds']=time.monotonic()-started
                    records.append(record);log.write(json.dumps(record,ensure_ascii=False)+'\n')
    index={'kind':'final_test_generation_index','status':'complete' if all(r['status']=='complete' for r in records) else 'completed_with_failures',
        'protocol_sha256':control['protocol_sha256'],'raw_manifest':{'path':control['raw_manifest_path'],'sha256':control['raw_manifest_sha256']},
        'source_cache_manifest':source_cache_reference,'source_features_file':cache['features_file'],
        'method_names':list(methods),'case_ids':control['case_ids'],'sampling_seeds':control['sampling_seeds'],'records':records,
        'physical_executions':len({r['execution_id'] for r in records if r['execution_started']}),
        'unique_output_images':len({r['output']['sha256'] for r in records if r['status']=='complete'}),
        'unique_output_files':len({r['output']['path'] for r in records if r['status']=='complete'}),
        'target_pixels_used':False,'target_latents_computed':False,'all_planned_cases_and_seeds_retained':True}
    save_json(output/'generation_index.json',index)
    return index


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True)
    p.add_argument('--raw-manifest',type=Path,required=True);p.add_argument('--raw-manifest-sha',required=True)
    p.add_argument('--source-cache',type=Path,required=True);p.add_argument('--source-cache-sha',required=True)
    p.add_argument('--method',action='append',required=True);p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    if not args.protocol.is_file() or digest(args.protocol)!=args.protocol_sha:raise ValueError('Explicit frozen protocol SHA required')
    control=validate_control(args.protocol,args.raw_manifest,args.raw_manifest_sha)
    for role,path in (('protocol_gate',HERE/'protocol_gate.py'),('source_cache',HERE/'prepare_source_cache.py'),('our_inference',Path(__file__))):verify_runtime_source(control,role,path)
    methods=validate_methods(control,args.method)
    cache,features=load_source_cache(control,args.source_cache,args.source_cache_sha,load_tensors=not args.preflight_only)
    if args.preflight_only:
        print(json.dumps({'status':'metadata_preflight_passed','methods':list(methods),'planned_outputs':len(methods)*len(control['case_ids'])*len(control['sampling_seeds']),
                          'models_loaded':False,'source_pixels_opened':False,'target_pixels_opened':False}));return
    runtime=OurRuntime(control,methods,cache,features)
    reserved=new_directory(HERE/'generation_runs','ours');reserved.rmdir()
    index=run_grid(control,methods,cache,reserved,runtime,{'path':str(args.source_cache.resolve()),'sha256':args.source_cache_sha})
    print(json.dumps({'status':index['status'],'index':str(reserved/'generation_index.json'),'sha256':digest(reserved/'generation_index.json')}))


if __name__=='__main__':main()
