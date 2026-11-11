"""Pinned native kernels; all image/model operations require an executed job."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from contracts import EXP,HERE,COMMITS,MF_PARAMETERS,FA_PARAMETERS,require,resolve,sha,signature

OLD=EXP/'external_baselines'
CORE_HASHES={
 'run_faceadapter.py':'c43e1b73cd431688590e14df07810514ae21fbcd76fe77a5b39bc7a798461e11',
 'faceadapter_detection_policy.py':'a7d4507f76d4d355ce942c50cc75a939ba354593bc0f403f448cb4763041dec3',
 'run_liveportrait.py':'7d227e268f0e00c079d9b4e84609c72896a6867d8b57922a7660378fa0a6a558',
 'liveportrait_detection_policy.py':'82e2f76504f1df42eb55c88de439b14f0759103de16d0dd94560d6e1515aea03',
 'second_baseline_review/liveportrait_required_assets.json':'6d62a14a42220397f9c9fb0c3f5b0379a8213e0b32371ca4ca461b64e7aecf87',
}
MF_CORE_SHA='e3a1a0e05d5213d7e7f402a4fb593793c1569894738eb0d0c73b598bf1f5ea67'
MF_CONTRACT_SHA='14f8bcb3e3ae0b37991c0e09bb19118a6e1b5e323c97c4a6bbb407bef9c5c59f'
MF_FILES={
 'models/generator/ID_enc/config.json','models/generator/ID_enc/diffusion_pytorch_model.safetensors',
 'models/generator/denoising_unet/config.json','models/generator/denoising_unet/diffusion_pytorch_model.safetensors',
 'models/generator/text_encoder/config.json','models/generator/text_encoder/model.safetensors',
 'models/preprocess/79999_iter.pth','models/preprocess/d3dfr_res50_nofc.pth','models/preprocess/BFM_model_front.mat',
 'models/preprocess/scrfd_10g_bnkps.onnx','models/base/vae/config.json','models/base/vae/diffusion_pytorch_model.safetensors',
 'models/base/scheduler/scheduler_config.json','models/base/tokenizer/merges.txt','models/base/tokenizer/special_tokens_map.json',
 'models/base/tokenizer/tokenizer_config.json','models/base/tokenizer/vocab.json',
 'auxiliary/third_party/model_resnet_d3dfr.py','auxiliary/third_party/d3dfr/bfm.py'}


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def load_module(path,name):
    path=Path(path);sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def verify_core_files():
    for name,digest in CORE_HASHES.items():require(sha(OLD/'runtime'/name)==digest,'Frozen native wrapper/helper changed: '+name)
    require(sha(OLD/'run_magicface_smoke.py')==MF_CORE_SHA,'MagicFace kernel changed')
    require(sha(OLD/'magicface_contract.py')==MF_CONTRACT_SHA,'MagicFace native call changed')


def validate_runtime(settings,declared_artifacts):
    """No images or model calls. Required loader assets must be declared in protocol."""
    verify_core_files();impl=settings['implementation'];runtime=settings['runtime'];repo=resolve(runtime['repo']['path'])
    require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==COMMITS[impl],'Official repo commit changed')
    require(not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Official repo is dirty')
    declared={str(resolve(a['path'])):a['sha256'] for a in declared_artifacts}
    required=[]
    if impl=='magicface_native':
        seal_ref=runtime['model_layout_seal'];seal_path=resolve(seal_ref['path'])
        require(sha(seal_path)==seal_ref['sha256'],'MagicFace layout seal changed')
        seal=json.loads(seal_path.read_text());required=[(resolve(r['target']),r['sha256']) for r in seal['models']+seal['auxiliary_sources']]
        require(len(required)==19 and all(p.is_relative_to(seal_path.parent) for p,_ in required),'Incomplete or escaping MF layout')
        require({str(p.relative_to(seal_path.parent)) for p,_ in required}==MF_FILES,'MF layout does not cover actual native loader files')
    elif impl=='faceadapter_native':
        w=load_module(OLD/'runtime/run_faceadapter.py','final_fa_static_inventory')
        inventory=w.resource_inventory(repo,resolve(runtime['checkpoint']),resolve(runtime['base_model']),resolve(runtime['vae_model']),False,True,'stable-diffusion-v1-5')
        require(not inventory['missing'],'Missing Face-Adapter assets: '+str(inventory['missing']))
        required=[(Path(r['path']).resolve(),declared.get(str(Path(r['path']).resolve()))) for r in inventory['files']]
    else:
        w=load_module(OLD/'runtime/run_liveportrait.py','final_lp_static_inventory')
        inventory,missing=w.model_inventory(resolve(runtime['checkpoint']));require(not missing,'Missing LivePortrait models')
        required=[(Path(r['path']).resolve(),r['sha256']) for r in inventory]
        defaults=w.author_parameters(repo,'cpu');candidate=dict(defaults,flag_relative_motion=settings['native_parameters']['flag_relative_motion'])
        require(settings['native_parameters']==candidate,'LP settings outside verified all-mode single-factor routes')
    lease=[]
    for path,expected in required:
        require(expected is not None and declared.get(str(path))==expected,'Actual native loader asset not declared: '+str(path))
        st=path.stat();require(not(getattr(st,'st_flags',0)&0x40000000),'Dataless asset: '+str(path))
        lease.append({'path':str(path),'sha256':expected,'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino})
    return {'implementation':impl,'repo':str(repo),'repo_commit':COMMITS[impl],'assets':lease,'pixels_read':False,'models_called':False}


def verify_lease(lease):
    for row in lease['assets']:
        st=Path(row['path']).stat()
        require((st.st_size,st.st_mtime_ns,st.st_ino)==(row['size'],row['mtime_ns'],row['inode']),'Asset changed after full protocol SHA verification')


def verify_job_inputs(job):
    require(job['split']=='test' and job['test_used'] is True,'Explicit final-test job required')
    names=['source']+(['target'] if job['input_group']=='source_real_driver' else [])
    for name in names:require(sha(job[name])==job[name+'_sha256'],'Raw '+name+' image SHA changed')
    return names


def mark_started(out,execution_id):
    marker=Path(out)/'generation_started.json'
    if not marker.exists():save(marker,{'execution_id':execution_id,'pid':os.getpid(),'started_at_unix':time.time(),
                                      'meaning':'native diffusion pipeline or LivePortrait warp_decode call reached'})


def stage_wrappers(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for name,expected in CORE_HASHES.items():
        src=OLD/'runtime'/name;dst=root/name;require(sha(src)==expected,'Native source changed')
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copyfile(src,dst)
        require(sha(dst)==expected,'Native staged source differs')
    return root


class ObservedPipeline:
    def __init__(self,pipeline,callback):object.__setattr__(self,'_native',pipeline);object.__setattr__(self,'_callback',callback)
    def __getattr__(self,name):return getattr(self._native,name)
    def __setattr__(self,name,value):setattr(self._native,name,value)
    def __call__(self,*args,**kwargs):self._callback();return self._native(*args,**kwargs)


def driver_job(context,out):
    settings=context['settings'];runtime=settings['runtime'];job=context['job'];impl=settings['implementation']
    staged=stage_wrappers(out/'native_runtime');manifest=out/'physical_case.json'
    save(manifest,{'kind':'final_test_native_case','test_used':True,'cases':[job]})
    name='run_faceadapter.py' if impl=='faceadapter_native' else 'run_liveportrait.py'
    wrapper=load_module(staged/name,'final_native_'+impl)
    native_out=staged/'runs/actual';marker=lambda:mark_started(out,job['execution_id'])
    if impl=='faceadapter_native':
        original=wrapper.load_pipeline_components
        wrapper.load_pipeline_components=lambda *a,**k:ObservedPipeline(original(*a,**k),marker)
        args=['--repo',runtime['repo']['path'],'--checkpoint',runtime['checkpoint'],'--base-model',runtime['base_model'],
              '--base-model-kind','stable-diffusion-v1-5','--vae-model',runtime['vae_model'],'--manifest',str(manifest),
              '--output',str(native_out),'--device',runtime['device'],'--dtype','float32','--preload-final-components',
              '--detection-policy','author_default','--execute']
    else:
        native_parameters=dict(settings['native_parameters']);original=wrapper.initialize
        wrapper.author_parameters=lambda repo,device:dict(native_parameters)
        def initialize(*a,**k):
            pipeline,detector,base_args,torch,metadata=original(*a,**k)
            require(pipeline.live_portrait_wrapper.device=='cpu','Unexpected LP device')
            for key,value in native_parameters.items():
                if hasattr(pipeline.live_portrait_wrapper.inference_cfg,key):require(getattr(pipeline.live_portrait_wrapper.inference_cfg,key)==value,'Effective LP parameter mismatch')
            fn=pipeline.live_portrait_wrapper.warp_decode
            def warp(*a,**k):marker();return fn(*a,**k)
            pipeline.live_portrait_wrapper.warp_decode=warp
            return pipeline,detector,base_args,torch,metadata
        wrapper.initialize=initialize
        args=['--repo',runtime['repo']['path'],'--checkpoint',runtime['checkpoint'],'--manifest',str(manifest),'--output',str(native_out),
              '--device','cpu','--detection-policy','author_default','--execute']
    returncode=wrapper.main(args)
    summary=json.loads((native_out/'summary.json').read_text());require(summary['case_count']==1,'Unexpected native case count')
    result=summary['results'][0];require(result['input']==job,'Native input changed')
    config=json.loads((native_out/'configuration.json').read_text())
    if impl=='faceadapter_native':
        for key in ('branch','scheduler','steps','guidance_scale','crop_ratio','image_size','condition_seed'):require(config['policy'][key]==settings['native_parameters'][key],'FA effective parameter changed')
    else:require(config['policy']['effective_author_cli_parameters']==settings['native_parameters'],'LP effective parameters changed')
    return result,returncode


def magicface_kernel_source():
    """Reuse only verified computational body; replace dev gate by external final gate."""
    source=OLD/'run_magicface_smoke.py';require(sha(source)==MF_CORE_SHA,'MF kernel SHA changed')
    tree=ast.parse(source.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
               and ast.unparse(n.value.func)=='os.environ.update')
    body=main.body[start:]
    class FinalLabels(ast.NodeTransformer):
        def visit_Expr(self,node):
            self.generic_visit(node)
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='torch.set_num_interop_threads':
                return ast.If(test=ast.Compare(ast.Attribute(ast.Name('args',ast.Load()),'phase',ast.Load()),[ast.Eq()],[ast.Constant('preprocess')]),body=[node],orelse=[])
            return node
        def visit_Dict(self,node):
            self.generic_visit(node)
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='test_used':node.values[i]=ast.Constant(True)
            return node
        def visit_Assign(self,node):
            self.generic_visit(node)
            if any(isinstance(t,ast.Name) and t.id=='sampled' for t in node.targets):
                return [ast.Expr(ast.Call(ast.Name('mark_native_started',ast.Load()),[],[])),node]
            return node
    fn=ast.FunctionDef(name='run_native_body',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    prefix=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.ClassDef) or isinstance(n,ast.FunctionDef) and n.name!='main']
    final=FinalLabels().visit(ast.Module(body=prefix+[fn],type_ignores=[]));ast.fix_missing_locations(final)
    return ast.unparse(final)+'\n'


def magicface_job(context,out):
    settings=context['settings'];runtime=settings['runtime'];job=context['job']
    require(job['input_group']=='source_manual_au' and not any(k in job for k in ('target','target_path','target_image_path','driver','target_sha256')),'Target pixels in MF job')
    contract=load_module(OLD/'magicface_contract.py','final_mf_native_call')
    def validate(case):
        require(case['split']=='test' and case['test_used'] is True and case['seed'] in (101,202,303),'Not a frozen test request')
        require(case['au_order']==[1,2,4,5,6,9,12,15,17,20,25,26],'Wrong AU order')
        delta=[b-a for a,b in zip(case['source_au_vector_0_5'],case['target_au_vector_0_5'])]
        require(delta==case['au_delta_native'],'Wrong native AU delta');return delta
    contract.validate_case=validate
    derived=magicface_kernel_source();(out/'native_kernel_snapshot.py').write_text(derived)
    sys.path.insert(0,str(OLD));scope={'__name__':'final_magicface_native_kernel','__file__':str(OLD/'run_magicface_smoke.py')}
    exec(compile(derived,str(OLD/'run_magicface_smoke.py'),'exec'),scope)
    layout=resolve(runtime['model_layout_seal']['path']);seal=json.loads(layout.read_text())
    scope.update(out=out,case=job,repo=resolve(runtime['repo']['path']),runtime=layout.parent,seal=seal,HERE=out/'cache_root',
                 native_inference_kwargs=contract.native_inference_kwargs,mark_native_started=lambda:mark_started(out,job['execution_id']))
    save(out/'case.json',{'case':job,'protocol_sha256':context['protocol_sha256'],'raw_manifest_sha256':context['raw_manifest_sha256'],
                         'target_image_read':False,'test_used':True,'kernel_original_sha256':MF_CORE_SHA})
    for phase,device in [('preprocess','cpu'),('generate',runtime['device'])]:
        scope['args']=SimpleNamespace(phase=phase,device=device)
        scope['run_native_body']()
    result=json.loads((out/'generation.json').read_text())
    require(result['test_used'] is True and result['target_image_read'] is False,'MF test/input labels wrong')
    require(result['inference_steps']==50 and result['scheduler']=='UniPCMultistepScheduler' and result['au_variation_sent']==job['au_delta_native'],'Native MF settings changed')
    return {'status':'success','input':job,'output':result['output'],'output_sha256':result['output_sha256']},0


def execute_job(context,out):
    """Only called by an explicit frozen-context worker. Never used by metadata tests."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();job=context['job'];result=None
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_ENABLE_MPS_FALLBACK='1')
    sys.dont_write_bytecode=True
    try:
        verify_lease(context['asset_lease']);verify_job_inputs(job)
        if context['settings']['implementation']=='magicface_native':result,code=magicface_job(context,out)
        else:result,code=driver_job(context,out)
        verify_job_inputs(job);verify_lease(context['asset_lease'])
        success=result['status']=='success'
        output={'path':result['output'],'sha256':result['output_sha256']} if success else None
        if success:require(sha(output['path'])==output['sha256'],'Native output SHA mismatch')
        receipt={'status':'complete' if success else 'failed','output':output,'returncode':code}
        if not success:receipt['error']={k:result.get(k) for k in ('error_type','error','traceback')}
    except Exception as error:
        receipt={'status':'failed','output':None,'error':{'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}}
    receipt.update(execution_id=job['execution_id'],generation_started=(out/'generation_started.json').exists(),
                   seconds=time.monotonic()-started,pid=os.getpid(),test_data_used=True,target_pixels_supplied=job['input_group']=='source_real_driver')
    save(out/'worker_result.json',receipt);return receipt
