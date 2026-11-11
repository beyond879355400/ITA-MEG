"""Full constructor parity on two fixed DEV tensors; --execute is explicit.

No final-test protocol is constructed, persisted, or accepted here. This is a
development-only in-memory fixture for the actual OurRuntime constructor.
Legacy four outputs are saved and all legacy models released before new load.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from prepare_source_cache import digest,save_json,new_directory,tensor_sha,load_train_anchor
from check_inference_dev_parity import (CACHE,CACHE_SHA,CHECKPOINT,CHECKPOINT_SHA,DEV,DEV_SHA,
                                        OLD_GENERATOR_SHA,PARAMETERS)
TEST_SUBJECTS=('SN003','SN005','SN008','SN009','SN021','SN024','SN031','SN032')
SPLIT=ROOT/'data_prepare/au_supervision_v2/subjects_frozen.json'
SPLIT_SHA='b0837ba1ae1b30eec0dd33c48f2300809625a09437b2c161ac4576dad521fe6d'


def development_control(meta_ref,feature_ref,checkpoint_ref,split_ref,test_subjects):
    """Explicit DEV fixture, never pretends to be a frozen final protocol."""
    if tuple(test_subjects)!=TEST_SUBJECTS:raise ValueError('Locked test subject exclusion list changed')
    settings={
        'baseline':{'implementation':'fineface_faceid_baseline','input_group':'source_text_au',
                    'seed_execution':{'mode':'stochastic_each_seed'},'native_parameters':copy.deepcopy(PARAMETERS),
                    'runtime':{'device':'mps','dtype':'torch.float32'}},
        'v4':{'implementation':'source_centered_v4','input_group':'source_text_au',
              'seed_execution':{'mode':'stochastic_each_seed'},'native_parameters':copy.deepcopy(PARAMETERS),
              'runtime':{'device':'mps','dtype':'torch.float32','adapter_checkpoint':copy.deepcopy(checkpoint_ref)}}}
    return {'kind':'development_only_constructor_parity_fixture','data_role':'development_only',
        'protocol':{'kind':'development_only_binding_subset','inputs':{'subject_split':copy.deepcopy(split_ref)}},
        'label_plan':{'test_subjects':list(TEST_SUBJECTS),'role':'exclusion_list_only'},
        'sampling':{'source_features':{'implementation':'training_compatible_source_features_v1','device':'mps','dtype':'torch.float32',
                                     'anchor_checkpoint':copy.deepcopy(checkpoint_ref),'training_cache_manifest':copy.deepcopy(meta_ref),
                                     'training_cache_features':copy.deepcopy(feature_ref)}},
        'models':{m:{'method':m,'settings':s,'artifacts':[copy.deepcopy(checkpoint_ref)] if m=='v4' else []} for m,s in settings.items()}}


def development_inputs(samples,selected,anchor,anchor_receipt):
    """Whitelist features from DEV rows; drop all train-cache target latents."""
    import torch
    if len(selected)!=2 or len({c['case_id'] for c in selected})!=2:raise ValueError('Exactly two distinct fixed development cases required')
    features={'source_anchor':anchor.detach().cpu().clone(),'sources':{},'conditions':{}}
    cases=[]
    for case in selected:
        cid=case['case_id'];row=samples[cid]
        if case.get('split')!='dev' or row.get('split')!='dev' or row['subject']!=case['subject'] or row['subject'] in TEST_SUBJECTS:
            raise ValueError('This parity fixture must never include test subjects/rows')
        for field,key in (('source_aus','source_au_vector_0_5'),('target_aus','target_au_vector_0_5')):
            if not torch.equal(row[field],torch.tensor(case[key],dtype=torch.float32)):raise ValueError('Fixed development labels changed')
        sid=row['subject']
        source={k:row[k].detach().cpu().clone() for k in ('image_patches','base_faceid','source_identity')}
        if sid in features['sources'] and any(not torch.equal(source[k],features['sources'][sid][k]) for k in source):
            raise ValueError('Same dev subject unexpectedly has different source features')
        features['sources'][sid]=source
        features['conditions'][cid]={**{k:row[k].detach().cpu().clone() for k in ('text_pooled','target_aus','source_aus')},'prompt':row['prompt']}
        cases.append({'case_id':cid,'subject':sid,'source_id':sid,'split':'dev'})
    return {'kind':'development_only_source_fixture','source_anchor':copy.deepcopy(anchor_receipt)},features,cases


def actual_runtime_metadata(backbone,pipe,public_assets):
    """Read actual instantiated modules, not only requested config strings."""
    from prepare_source_cache import runtime_environment
    modules={'unet':backbone.unet,'vae':pipe.vae,'text_encoder':pipe.text_encoder,
             'au_encoder':backbone.au_encoder,'ip_projection':backbone.ip_proj}
    states={}
    for name,module in modules.items():
        parameters=list(module.parameters())
        states[name]={'devices':sorted({str(p.device) for p in parameters}),
                      'dtypes':sorted({str(p.dtype) for p in parameters}),
                      'trainable_parameters':sum(p.numel() for p in parameters if p.requires_grad),
                      'parameter_count':sum(p.numel() for p in parameters)}
        if states[name]['devices']!=['mps:0'] or states[name]['dtypes']!=['torch.float32'] or states[name]['trainable_parameters']:
            raise ValueError('Unexpected actual device/dtype/freezing for '+name+': '+str(states[name]))
    return {'actual_modules':states,'scheduler_class':type(pipe.scheduler).__name__,'scheduler_config':dict(pipe.scheduler.config),
            'public_assets':public_assets,'environment':runtime_environment()}


def legacy_phase(output,samples,cases,anchor,state,current):
    """Build exactly the frozen dev recipe; return CPU-only comparison tensors."""
    import torch
    from diffusers import StableDiffusionPipeline
    from modeling.frozen_backbone import FrozenConditionedDenoiser,FrozenImageTextEncoders,RV
    from modeling.safe_token_residual_adapter import RelativeRMSConfig
    from train_centered_adapter import adapter_from_checkpoint,centered_batch_conditions
    backbone=FrozenConditionedDenoiser('mps',checkpointing=False);encoders=FrozenImageTextEncoders('mps')
    pipe=StableDiffusionPipeline.from_pretrained(str(RV),unet=backbone.unet,vae=encoders.vae,text_encoder=encoders.text_encoder,
        tokenizer=encoders.tokenizer,safety_checker=None,feature_extractor=None,requires_safety_checker=False,local_files_only=True).to('mps')
    if type(pipe.scheduler).__name__!='DEISMultistepScheduler':raise ValueError('Unexpected actual sampler')
    pipe.set_progress_bar_config(disable=True)
    model=adapter_from_checkpoint(state,'mps').eval().requires_grad_(False);e=state['effective_config']
    trust=RelativeRMSConfig(enabled=e['trust_region'],rho_au=e['rho_au'],rho_ip=e['rho_ip'],reference_rms_floor=e['reference_rms_floor'])
    meta=actual_runtime_metadata(backbone,pipe,current['public_assets']);records=[];conditions={}
    save_json(output/'legacy_runtime.json',meta)
    with torch.no_grad():
        for case in cases:
            cid=case['case_id'];sample=samples[cid]
            patches,text,aus,identity=[sample[k][None].to('mps') for k in ('image_patches','text_pooled','target_aus','base_faceid')]
            base_au,base_ip=backbone.baseline_conditions(aus,identity)
            null_au,null_ip=backbone.baseline_conditions(torch.zeros_like(aus),torch.zeros_like(identity))
            for method in ('baseline','v4'):
                if method=='baseline':values={'au_embedding':base_au,'image_embedding':base_ip}
                else:
                    batch={'image_patches':patches,'text_pooled':text,'target_aus':aus,'source_aus':sample['source_aus'][None].to('mps')}
                    _,_,values=centered_batch_conditions(model,batch,anchor.to('mps'),base_au,base_ip,trust)
                cross={'au_embedding':torch.cat((null_au,values['au_embedding'])),'image_embedding':torch.cat((null_ip,values['image_embedding'])),
                       'au_scale':1.,'ip_scale':.9}
                conditions[(cid,method)]={k:v.detach().cpu().clone() for k,v in cross.items() if isinstance(v,torch.Tensor)}
                image=pipe(sample['prompt'],negative_prompt='',height=512,width=512,num_inference_steps=20,guidance_scale=7.5,
                    generator=torch.Generator(device='cpu').manual_seed(101),cross_attention_kwargs=cross).images[0]
                path=output/(cid+'_'+method+'_legacy.png');image.save(path)
                record={'case_id':cid,'method':method,'legacy_path':str(path),'legacy_sha256':digest(path),
                        'condition_hashes':{k:tensor_sha(v) for k,v in conditions[(cid,method)].items()}}
                records.append(record);print(json.dumps({'phase':'legacy',**record}),flush=True)
    save_json(output/'legacy_completed.json',{'status':'complete','images':records})
    # All GPU objects are local to this function. No pipe/module is returned.
    return conditions,records,meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:
        print(json.dumps({'status':'not_executed','constructor_parity_prepared':True,'requires_explicit_gpu_authorization':True,
                          'data_role':'development_only','final_protocol_created':False,'test_pixels_read':False}));return
    import torch
    from train_centered_adapter import load_checkpoint,source_and_asset_bindings
    from generate_ours import OurRuntime,validate_methods
    for path,expected in ((CACHE/'features.pt',CACHE_SHA),(CHECKPOINT,CHECKPOINT_SHA),(DEV,DEV_SHA),(SPLIT,SPLIT_SHA),
                          (ROOT/'generate_centered_development.py',OLD_GENERATOR_SHA)):
        if digest(path)!=expected:raise ValueError('Fixed development fixture changed')
    split=json.loads(SPLIT.read_text());document=json.loads(DEV.read_text())
    if document['test_data_used'] is not False or document['cache_sha256']!=CACHE_SHA or tuple(split['groups']['test'])!=TEST_SUBJECTS:
        raise ValueError('Development/test exclusion binding changed')
    ref=lambda path:{'path':str(path),'sha256':digest(path)}
    control=development_control(ref(CACHE/'manifest.json'),ref(CACHE/'features.pt'),ref(CHECKPOINT),ref(SPLIT),split['groups']['test'])
    methods=validate_methods(control,['baseline','v4'])
    anchor,anchor_receipt,state=load_train_anchor(control)
    rows=torch.load(CACHE/'features.pt',map_location='cpu',weights_only=True)['samples']
    if any(r['split'] not in ('train','dev') for r in rows):raise ValueError('Unexpected test cache row')
    samples={r['case_id']:r for r in rows}
    selected=[next(c for c in document['cases'] if not any(c['target_au_vector_0_5'])),
              next(c for c in document['cases'] if any(c['target_au_vector_0_5']))]
    cache,features,cases=development_inputs(samples,selected,anchor,anchor_receipt)
    current,_=source_and_asset_bindings({})
    if current['public_assets']!=state['bindings']['public_assets']:raise ValueError('Public base changed')
    output=new_directory(HERE/'runtime_checks','mps_dev_full_constructor')
    receipt={'status':'started','data_role':'development_only','selection':'first noedit and first edited DEV case in frozen order',
        'case_ids':[c['case_id'] for c in cases],'seed':101,'native_parameters':PARAMETERS,'checkpoint_sha256':CHECKPOINT_SHA,
        'cache_sha256':CACHE_SHA,'dev_manifest_sha256':DEV_SHA,'split_sha256':SPLIT_SHA,'test_subjects_excluded':list(TEST_SUBJECTS),
        'original_generator_sha256':OLD_GENERATOR_SHA,'new_generator_sha256':digest(HERE/'generate_ours.py'),'checker_sha256':digest(__file__),
        'test_used':False,'final_protocol_created':False,'production_cli_called':False,'legacy_instances_reused_by_new_constructor':False}
    save_json(output/'started.json',receipt);shutil.copyfile(__file__,output/'checker_snapshot.py')
    allocated_before=int(torch.mps.current_allocated_memory())
    conditions,legacy,old_meta=legacy_phase(output,samples,cases,anchor,state,current)
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    released={'torch_mps_allocated_before_legacy':allocated_before,'torch_mps_allocated_after_legacy_release':int(torch.mps.current_allocated_memory()),
              'python_legacy_phase_returned':True,'models_returned':False,'only_CPU_condition_tensors_retained':all(t.device.type=='cpu' for values in conditions.values() for t in values.values())}
    save_json(output/'legacy_release.json',released)
    if released['torch_mps_allocated_after_legacy_release']>allocated_before:raise RuntimeError('Legacy GPU tensors remain; refusing overlapping constructor load')
    # REAL constructor: no __new__, no borrowed pipe, no monkeypatching.
    runtime=OurRuntime(control,methods,cache,features)
    new_meta=actual_runtime_metadata(runtime.backbone,runtime.pipe,runtime.runtime_receipt['public_assets'])
    save_json(output/'new_runtime.json',{'actual':new_meta,'constructor_receipt':runtime.runtime_receipt})
    metadata_equal=old_meta==new_meta;checks=[];legacy_map={(r['case_id'],r['method']):r for r in legacy}
    if not metadata_equal:raise RuntimeError('Actual loaded weights/device/dtype/scheduler/environment differ')
    for case in cases:
        cid=case['case_id']
        for method in ('baseline','v4'):
            prepared=runtime.prepare(method,cid,case['source_id'])
            differences={key:float((prepared['cross'][key].detach().cpu()-value).abs().max()) for key,value in conditions[(cid,method)].items()}
            condition_equal=all(value==0 for value in differences.values())
            if not condition_equal:raise RuntimeError('New real-constructor conditions differ; refusing sampling')
            image=runtime.sample(method,prepared,101);path=output/(cid+'_'+method+'_new.png');image.save(path)
            check={**legacy_map[(cid,method)],'condition_max_abs':differences,'conditions_equal':condition_equal,
                   'new_path':str(path),'new_sha256':digest(path),'png_exact':digest(path)==legacy_map[(cid,method)]['legacy_sha256']}
            checks.append(check);print(json.dumps({'phase':'new_constructor',**check}),flush=True)
    result={**receipt,'status':'complete','passed':metadata_equal and all(c['png_exact'] for c in checks),
            'actual_constructor_used':True,'runtime_metadata_exact':metadata_equal,'legacy_release':released,'checks':checks}
    save_json(output/'result.json',result)
    print(json.dumps({'report':str(output/'result.json'),'passed':result['passed']}),flush=True)
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
