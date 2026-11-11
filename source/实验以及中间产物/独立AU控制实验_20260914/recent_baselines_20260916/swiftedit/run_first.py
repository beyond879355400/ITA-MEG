"""Offline local-feasibility wrapper, fixed development inputs, no training.

Default is model-free preflight. Model construction requires --execute and a
fresh root GPU_RUN_GO in approvals.json. The community base is not a verified
official migration and outputs are not thesis main-table results.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback

from stage_weights import HERE, RESERVE, gate, need, read, save_new, sha
from stage_components import STATUS

VERSION = HERE/'versions/device_v2_full_ip'
BUNDLE = HERE.parent/'preservation/bundle_v2_20260916'
MANIFEST = BUNDLE/'inputs/smoke_dev4.json'
SOURCE_PROMPT = 'A tightly cropped photograph of a face with a neutral expression.'
DEFAULTS = dict(scale_ta=1, scale_edit=.2, scale_non_edit=1, clamp_rate=3., mask_threshold=.5)
SEED = 101
FTFY = HERE/'downloads/ftfy-6.3.1-py3-none-any.whl'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def relative_inside(root, value):
    path=(root/value).resolve()
    need(root.resolve() in path.parents, 'Path escaped controlled root')
    return path


def fixed_cases(start=0, count=1):
    data=read(MANIFEST)
    need(data['kind']=='recent_baseline_case_inputs_v1' and data['selection']=='smoke_dev4', 'Wrong manifest')
    need(len(data['cases'])==4 and 0<=start<4 and 1<=count<=4-start, 'Unexpected case selection')
    cases=data['cases'][start:start+count]
    for row in cases:
        need(row['split']=='dev' and row['sampling_seeds']==[SEED], 'Only frozen dev/seed101 is authorized')
        need(not any(row['source_au_vector_0_5']) and row['prompt'], 'Expected frozen neutral-source input')
        source=relative_inside(BUNDLE,row['source_path'])
        need(source.is_file() and sha(source)==row['source_sha256'], 'Source pixel SHA mismatch')
        # target_path is reference metadata only. Never open target pixels here.
    return cases


def verify_assets():
    spec=read(HERE/'model_paths.json')
    need(spec['base_source_status']==STATUS, 'Source status must be explicit')
    extraction=relative_inside(HERE,spec['official_extraction'])
    need(spec['construction']=='config_then_strict_full_ip_state','Unexpected construction mode')
    proof_path=relative_inside(HERE,spec['meta_equivalence']['path'])
    need(sha(proof_path)==spec['meta_equivalence']['sha256'],'Meta coverage proof changed')
    proof=read(proof_path)
    need(proof['status']=='pass' and proof['strict_load_passed'] and proof['shape_dtype_coverage_passed'], 'Meta full-state audit absent')
    need(proof['audit_code_sha256']==sha(HERE/'audit_full_ip_state.py'),'Meta audit source differs')
    need(sha(extraction/'extraction_failure.json')==spec['extraction_failure_sha256'],'Partial extraction evidence changed')
    er=read(extraction/'extraction_failure.json')
    need(er['status']=='failed','Preserved partial extraction provenance differs')
    required={'swiftedit_weights/inverse_ckpt-120k/unet_ema/config.json',
              'swiftedit_weights/inverse_ckpt-120k/unet_ema/diffusion_pytorch_model.safetensors',
              'swiftedit_weights/sbv2_0.5/config.json','swiftedit_weights/ip_adapter_ckpt-90k/ip_adapter.bin'}
    need(required.issubset({row['path'] for row in er['completed_files']}),'Required model file omitted from evidence')
    checked=[]
    for row in er['completed_files']:
        path=relative_inside(extraction,row['path'])
        need(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], 'Official extracted content mismatch')
        checked.append({'path':str(path),'sha256':row['sha256'],'bytes':row['bytes']})
    auxiliary=HERE/'weights/auxiliary'
    ar=read(auxiliary/'download_receipt.json')
    need(ar['base_source_status']==STATUS and ar['manifest_sha256']==sha(HERE/'aux_components.json'), 'Auxiliary manifest differs')
    for row in ar['files']:
        path=Path(row['path']).resolve()
        need(auxiliary.resolve() in path.parents and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], 'Auxiliary content mismatch')
        checked.append({'path':str(path),'sha256':row['sha256'],'bytes':row['bytes']})
    meta=read(HERE/'aux_components.json')
    need(len(ar['files'])==len(meta['components']), 'Incomplete auxiliary receipt')
    expected={str(relative_inside(auxiliary,row['local_path'])) for row in meta['components']}
    need(expected=={row['path'] for row in ar['files']}, 'Auxiliary file set differs')
    official=relative_inside(extraction,spec['official_subdir'])
    paths={'inverse':official/'inverse_ckpt-120k','sbv2':official/'sbv2_0.5',
           'ip_checkpoint':official/'ip_adapter_ckpt-90k/ip_adapter.bin',
           'turbo_aux':auxiliary/'sd_turbo','sd21_aux':auxiliary/'sd21_community','ip_encoder':auxiliary/'ip_adapter'}
    need(all(path.exists() for path in paths.values()), 'Model path missing')
    need(sha(paths['ip_checkpoint'])==proof['ip_checkpoint_sha256'] and sha(paths['sbv2']/'config.json')==proof['sbv2_config_sha256'], 'Required checkpoint/config differs from proof')
    need(read(paths['ip_encoder']/'models/image_encoder/config.json')['projection_dim']==proof['clip_projection_dim'], 'Pinned CLIP projection dimension differs')
    return paths,checked


def preflight(start=0,count=1,full_assets=False):
    cases=fixed_cases(start,count)
    lock=read(HERE/'source_lock.json')
    repo=Path.home()/'Projects/github/Qualcomm-AI-research/SwiftEdit'
    for path,digest in lock['files'].items():
        need(sha(repo/path)==digest,'Official source changed')
    code={str(path.relative_to(HERE)):sha(path) for path in [Path(__file__).resolve(),HERE/'stage_weights.py',HERE/'stage_components.py',HERE/'consume_extract.py',HERE/'audit_full_ip_state.py',*sorted(VERSION.rglob('*.py'))]}
    need(FTFY.is_file() and sha(FTFY)==read(HERE/'ftfy_wheel.json')['sha256'],'ftfy private wheel absent or changed')
    ready=(HERE/'model_paths.json').exists() and (HERE/'weights/auxiliary/download_receipt.json').exists()
    if full_assets:
        need(ready,'Weights not fully staged')
        verify_assets()
    return {'kind':'swiftedit_local_feasibility_preflight','at':timestamp(),
            'base_source_status':STATUS,'eligible_for_thesis_main_table':False,
            'case_ids':[row['case_id'] for row in cases],'seed':SEED,'source_prompt':SOURCE_PROMPT,
            'target_prompts':[row['prompt'] for row in cases],'defaults':DEFAULTS,
            'manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST)},'code':code,
            'asset_declarations':{name:sha(HERE/name) for name in ['model_paths.json','aux_components.json','release_assets.json','logs/full_ip_meta_coverage.json']},
            'free_bytes':shutil.disk_usage(HERE).free,'minimum_free_bytes':RESERVE,
            'assets_ready':ready,'asset_hashes_verified':full_assets,
            'model_construction_performed':False,'gpu_run_go':bool(read(HERE/'approvals.json').get('gpu_run_go'))}


@contextmanager
def offline_and_safe_load(torch):
    original_connect,original_load=socket.socket.connect,torch.load
    def blocked(*args,**kwargs):
        raise RuntimeError('Network is forbidden during controlled inference')
    def safe_load(*args,**kwargs):
        kwargs['map_location']='cpu'
        kwargs['weights_only']=True
        return original_load(*args,**kwargs)
    socket.socket.connect=blocked
    torch.load=safe_load
    try:
        yield
    finally:
        socket.socket.connect=original_connect
        torch.load=original_load


def execute(args):
    gate('cache_ready'); gate('gpu_run_go')
    need(shutil.disk_usage(HERE).free>=RESERVE,'Less than8GiB free; no model load')
    evidence=preflight(args.start,args.count)
    paths,assets=verify_assets()
    cases=fixed_cases(args.start,args.count)
    permission=read(HERE/'approvals.json')['gpu_run_go']
    need(args.device==permission['device'] and SEED==permission['seed'] and
         all(row['case_id'] in permission['case_ids'] for row in cases),'Requested cases/device outside current root run scope')
    runs=HERE/'runs'; runs.mkdir(exist_ok=True)
    out=runs/args.run_name; out.mkdir(exist_ok=False)
    evidence.pop('model_construction_performed')
    save_new(out/'configuration.json',{**evidence,'asset_hashes_verified':True,'assets':assets,'device':args.device,'dtype':'float32',
          'model_construction_requested':True,
          'generator_construction':'config_then_strict_full_ip_state','archive_extraction_complete':False,
          'meta_equivalence_sha256':sha(HERE/'logs/full_ip_meta_coverage.json'),
          'input_resize':[512,512],'target_pixels_used':False,'weights_only':True,'network_during_inference':False,
          'source_vae_posterior':'sample','official_unused_noise_decode_preserved':True,
          'execution_scope':'local feasibility only; not an exact verified original SD2.1 reproduction'})
    events=[]
    def event(stage,**extra):
        row={'at':timestamp(),'stage':stage,**extra};events.append(row)
        with (out/'events.jsonl').open('a',buffering=1) as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    result={'kind':'swiftedit_local_feasibility_result','base_source_status':STATUS,
            'eligible_for_thesis_main_table':False,'configuration_sha256':sha(out/'configuration.json'),
            'status':'failed','records':[],'model_construction_attempted':False,'all_models_loaded':False}
    try:
        for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']:
            os.environ[key]='1'
        os.environ['TOKENIZERS_PARALLELISM']='false'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
        watermark=os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
        need(watermark is None or float(watermark)>0,'Unlimited MPS high watermark is forbidden')
        sys.path[:0]=[str(FTFY),str(VERSION)]
        import numpy as np
        import torch
        import ftfy
        from torchvision.utils import save_image
        from models import InverseModel,AuxiliaryModel,IPSBV2Model
        from infer import edit_image
        need(args.device!='mps' or torch.backends.mps.is_available(),'MPS unavailable')
        event('environment',python=sys.version,torch=torch.__version__,ftfy=ftfy.__version__,
              mps_fallback='0',mps_high_watermark_ratio=os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO','torch_default'),
              versions={k:importlib.metadata.version(k) for k in ['diffusers','transformers','safetensors','numpy']})
        def sync():
            if args.device=='mps':torch.mps.synchronize()
        def memory():
            readings={'free_disk_bytes':shutil.disk_usage(HERE).free}
            for name,cmd in [('memory_pressure',['/usr/bin/memory_pressure','-Q']),('swap',['/usr/sbin/sysctl','vm.swapusage'])]:
                probe=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
                readings[name]={'returncode':probe.returncode,'stdout':probe.stdout.strip(),'stderr':probe.stderr.strip()}
            if args.device=='mps':
                readings.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory())
            return readings
        with offline_and_safe_load(torch):
            start=time.monotonic();event('load_inverse_start',**memory())
            result['model_construction_attempted']=True
            inverse=InverseModel(str(paths['inverse']),model_name=str(paths['turbo_aux']),device=args.device)
            sync();event('load_inverse_complete',seconds=time.monotonic()-start,**memory())
            need(shutil.disk_usage(HERE).free>=RESERVE,'Disk reserve reached after inverse load')
            start=time.monotonic();event('load_aux_start')
            aux=AuxiliaryModel(model_name=str(paths['sd21_aux']),image_encoder_path=str(paths['ip_encoder']),device=args.device)
            sync();event('load_aux_complete',seconds=time.monotonic()-start,**memory())
            need(shutil.disk_usage(HERE).free>=RESERVE,'Disk reserve reached after auxiliary load')
            start=time.monotonic();event('load_generator_start')
            model=IPSBV2Model(str(paths['sbv2']),str(paths['ip_checkpoint']),aux,device=args.device,with_ip_mask_controller=True)
            sync();event('load_generator_complete',seconds=time.monotonic()-start,**memory())
            need(shutil.disk_usage(HERE).free>=RESERVE,'Disk reserve reached after generator load')
            result['all_models_loaded']=True
            # Instrument only top-level UNet outputs, without changing returns.
            trace=[]
            def hook(name):
                def observe(module,inputs,output):
                    tensor=output if isinstance(output,torch.Tensor) else output.sample
                    finite=bool(torch.isfinite(tensor).all().item())
                    row={'module':name,'shape':list(tensor.shape),'finite':finite,
                         'abs_max':float(tensor.abs().max().item()) if finite else None}
                    trace.append(row);event('forward',**row)
                    need(finite,'Nonfinite '+name)
                return observe
            inverse.unet_inverse.register_forward_hook(hook('inverse_unet'))
            model.unet.register_forward_hook(hook('generation_unet'))
            inverse.vae.encoder.register_forward_hook(hook('inverse_vae_encoder'))
            aux.vae.decoder.register_forward_hook(hook('generation_vae_decoder'))
            for row in cases:
                gate('gpu_run_go');need(shutil.disk_usage(HERE).free>=RESERVE,'Disk reserve reached')
                random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
                if args.device=='mps':torch.mps.manual_seed(SEED)
                trace.clear();start=time.monotonic();event('case_start',case_id=row['case_id'],seed=SEED)
                images=edit_image(str(relative_inside(BUNDLE,row['source_path'])),SOURCE_PROMPT,row['prompt'],inverse,aux,model,device=args.device,**DEFAULTS)
                sync();seconds=time.monotonic()-start
                need(tuple(images.shape)==(2,3,512,512) and bool(torch.isfinite(images).all().item()),'Invalid/nonfinite image tensor')
                need([x['module'] for x in trace]==['inverse_vae_encoder','inverse_unet','generation_unet','generation_vae_decoder','generation_vae_decoder'],'Unexpected model call budget')
                # This remains the author clamp-to-PNG behavior, not RGB repair.
                files=[]
                for index,role in enumerate(['source_reconstruction','target_edit']):
                    path=out/(row['case_id']+'_'+role+'.png')
                    need(not path.exists(),'Output collision');save_image(images[index],str(path))
                    files.append({'role':role,'path':str(path),'sha256':sha(path)})
                # Capture the exact mask assigned by the author controller.
                controllers=[p.controller for p in model.unet.attn_processors.values() if getattr(p,'controller',None) is not None]
                need(controllers and all(p is controllers[0] for p in controllers),'Unexpected controller wiring')
                mask=controllers[0].mask_s.detach().cpu().numpy()
                need(mask.shape==(64,64) and np.isfinite(mask).all() and np.isin(mask,[0,1]).all(),'Invalid author editing mask')
                mask_path=out/(row['case_id']+'_author_mask.npy')
                with mask_path.open('xb') as stream:np.save(stream,mask,allow_pickle=False)
                record={'case_id':row['case_id'],'seed':SEED,'status':'complete','seconds':seconds,
                        'source_path':row['source_path'],'source_sha256':row['source_sha256'],
                        'source_prompt':SOURCE_PROMPT,'target_prompt':row['prompt'],'defaults':DEFAULTS,
                        'base_source_status':STATUS,'finite':True,'forward_trace':list(trace),'files':files,
                        'image_unclamped_min':float(images.min().item()),'image_unclamped_max':float(images.max().item()),
                        'controller_count':len(controllers),'author_mask':{'path':str(mask_path),'sha256':sha(mask_path),'fraction':float(mask.mean())},**memory()}
                result['records'].append(record)
                save_new(out/(row['case_id']+'_receipt.json'),record)
                event('case_complete',case_id=row['case_id'],seconds=seconds,**memory())
            result['status']='complete'
    except BaseException as error:
        result['error']=str(error);result['traceback']=traceback.format_exc()
        event('failed',error=str(error))
        raise
    finally:
        result['finished_at']=timestamp();save_new(out/'result.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--verify-assets',action='store_true')
    p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=1)
    p.add_argument('--device',choices=['mps','cpu'],default='mps')
    p.add_argument('--run-name',default='first_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=p.parse_args()
    need(Path(args.run_name).name==args.run_name and args.run_name not in ['.','..'],'Unsafe run name')
    if args.execute:execute(args)
    else:print(json.dumps(preflight(args.start,args.count,args.verify_assets),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
