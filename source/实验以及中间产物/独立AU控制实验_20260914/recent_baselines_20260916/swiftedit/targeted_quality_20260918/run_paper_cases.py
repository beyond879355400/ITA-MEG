"""Remaining five paper cases using a single source-aligned preprocessing policy."""
from pathlib import Path
import argparse,gc,hashlib,io,json,os,random,shutil,sys,time,traceback

T=Path(__file__).resolve().parent; S=T.parent; NEW=S.parent; E=NEW.parent
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                  PYTORCH_ENABLE_MPS_FALLBACK='0',TMPDIR=str(T/'tmp'))
(T/'tmp').mkdir(exist_ok=True)
sys.path[:0]=[str(S),str(S/'recovery_v3_20260916')]
import run_first as core
sys.path[:0]=[str(core.FTFY),str(core.VERSION)]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--device',choices=['mps','cpu'],default='mps')
p.add_argument('--case',action='append');a=p.parse_args()
assert Path(a.run).name==a.run
allowed=['fresh_v2_dev_SN004_s00745_t04266','fresh_v2_dev_SN012_s02549_t00086',
         'fresh_v2_dev_SN017_s02034_t04507','fresh_v2_dev_SN018_s00885_t04693',
         'postfreeze_SN024_s02323_t00214']
ids=a.case or allowed
assert ids and len(ids)<=5 and all(c in allowed for c in ids)
out=T/'runs'/a.run;out.mkdir(parents=True,exist_ok=False)
def dump(name,d):
    with (out/name).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def event(stage,**kw):
    d=dict(stage=stage,at=time.time(),**kw)
    with (out/'events.jsonl').open('a') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
    print(json.dumps(d,ensure_ascii=False),flush=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'status':'running','records':[],'new_training':False,'quantitative_tables_changed':False}
try:
    event('imports_start')
    import numpy as np,torch
    from PIL import Image
    from torchvision.utils import save_image
    from models import InverseModel,AuxiliaryModel,IPSBV2Model
    from infer import edit_image
    from hot_vae_route import install
    torch.set_num_threads(4)
    install(core,S)
    event('assets_verify_start',torch=torch.__version__)
    assert shutil.disk_usage(T).free>8*1024**3
    paths,assets=core.verify_assets()
    raw=(S/'recovery_v3_20260916/verified_inputs/postfreeze32.json').read_bytes()
    manifest=json.loads(raw)
    dev_file=core.BUNDLE/'inputs/smoke_dev4.json';dev=json.loads(dev_file.read_text())
    cases=[next(c for c in manifest['cases']+dev['cases'] if c['case_id']==i) for i in ids]
    old_index=json.loads((S/'formal96_20260916_1603_cachebound/generation_index.json').read_text())
    # The case-specific receipt gives original PNG and hashes without metric selection.
    previous={}
    for row in cases:
        matches=list((S/'runs/formal96_20260916_1603_cachebound_seed101').glob(row['case_id']+'_receipt.json'))
        if not matches:matches=list((S/'runs/formal96_20260916_1543_hotvae_seed101').glob(row['case_id']+'_receipt.json'))
        if not matches:
            for name in ['recovery_mps_fp32_sn004_20260916_1517','recovery_mps_fp32_remaining3_20260916_1523']:
                matches+=list((S/'runs'/name).glob(row['case_id']+'_receipt.json'))
        assert len(matches)==1
        previous[row['case_id']]=json.loads(matches[0].read_text())
    canonical_file=E/'improvements/quality_refinement_20260917/same_config_replacement/all24_seed101_recorded_paths.json'
    canonical={c['case_id']:c['source'] for c in json.loads(canonical_file.read_text())['cases']}
    root=E.parents[1]
    dev_audit=json.loads((root/'实验以及中间产物/全图核查与SwiftEdit诊断_20260918/dev_figures/audit.json').read_text())
    for c in dev_audit['cases']:
        src=next(x for x in c['existing_six'] if x['method']=='source_real')
        canonical[c['case_id']]={'path':src['path'],'recorded_sha256':src['sha256']}
    config=dict(kind='targeted_swiftedit_paper_aligned_cases',user_authorized_scope='Only necessary paper images, no96batch',
                case_ids=ids,seed=101,device=a.device,defaults=core.DEFAULTS,source_prompt=core.SOURCE_PROMPT,
                manifest_sha256=hashlib.sha256(raw).hexdigest(),assets=assets,
                script_sha256=sha(__file__),core_sha256=sha(S/'run_first.py'),source_paths=[],
                versions={k:core.importlib.metadata.version(k) for k in ['torch','torchvision','diffusers','transformers','Pillow','numpy']},
                base_source_status=core.STATUS,target_pixels_used=False,
                input_preprocessing='existing canonical source-only 512 face crop, instead of author resize of raw1024x768',
                native_table_configuration=False,canonical_manifest_sha256=sha(canonical_file))
    inputs={}
    for row in cases:
        src=core.relative_inside(core.BUNDLE,row['source_path']);data=src.read_bytes()
        assert hashlib.sha256(data).hexdigest()==row['source_sha256']
        entry=canonical[row['case_id']];aligned=Path(entry['path']);data=aligned.read_bytes()
        assert hashlib.sha256(data).hexdigest()==entry['recorded_sha256']
        with Image.open(io.BytesIO(data)) as im:assert im.size==(512,512)
        path=out/(row['case_id']+'_source.png');path.write_bytes(data);inputs[row['case_id']]=path
        config['source_paths'].append({'case_id':row['case_id'],'original':str(src),'original_raw_sha256':row['source_sha256'],
            'canonical_source':str(aligned),'sha256':entry['recorded_sha256']})
    dump('configuration.json',config)
    event('assets_verified',count=len(assets))
    sync=lambda:torch.mps.synchronize() if a.device=='mps' else None
    with core.offline_and_safe_load(torch):
        event('inverse_load');inverse=InverseModel(str(paths['inverse']),model_name=str(paths['turbo_aux']),device=a.device)
        event('aux_load');aux=AuxiliaryModel(model_name=str(paths['sd21_aux']),image_encoder_path=str(paths['ip_encoder']),device=a.device)
        event('generator_load');model=IPSBV2Model(str(paths['sbv2']),str(paths['ip_checkpoint']),aux,device=a.device,with_ip_mask_controller=True)
        sync();event('models_loaded')
        for row in cases:
            case=row['case_id'];folder=out/case;folder.mkdir()
            if a.device=='mps':sync();gc.collect();torch.mps.empty_cache()
            random.seed(101);np.random.seed(101);torch.manual_seed(101)
            if a.device=='mps':torch.mps.manual_seed(101)
            tensors={};handles=[];callcounts={}
            def take(name,t):
                v=t.detach().cpu().clone();assert torch.isfinite(v).all().item(),name
                tensors[name]=v;torch.save(v,folder/(name+'.pt'))
            def pre(name):
                def hook(module,args):
                    i=callcounts.get(name,0);callcounts[name]=i+1
                    for j,v in enumerate(args):
                        if isinstance(v,torch.Tensor):take(f'{name}_{i}_arg{j}',v)
                return hook
            def post(name):
                def hook(module,args,output):
                    v=output if isinstance(output,torch.Tensor) else output.sample
                    take(name,v)
                return hook
            handles += [inverse.unet_inverse.register_forward_pre_hook(pre('inverse_unet')),
                        inverse.unet_inverse.register_forward_hook(post('inverse_prediction')),
                        model.unet.register_forward_pre_hook(pre('generation_unet')),
                        model.unet.register_forward_hook(post('generation_prediction')),
                        aux.vae.decoder.register_forward_pre_hook(pre('decoder'))]
            event('case_start',case_id=case);start=time.monotonic()
            try:
                images=edit_image(str(inputs[case]),core.SOURCE_PROMPT,row['prompt'],inverse,aux,model,
                                  device=a.device,**core.DEFAULTS)
                sync()
            finally:
                for h in handles:h.remove()
            assert images.shape==(2,3,512,512) and torch.isfinite(images).all().item()
            torch.save(images.detach().cpu(),folder/'output_float.pt')
            parity=[]
            for i,role in enumerate(['source_reconstruction','target_edit']):
                path=folder/(role+'.png');save_image(images[i],str(path))
                old=next(f for f in previous[case]['files'] if f['role']==role)
                x=np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
                y=np.asarray(Image.open(old['path']).convert('RGB')).astype(np.float32)
                parity.append(dict(role=role,path=str(path),sha256=sha(path),old=old,
                    byte_equal=sha(path)==old['sha256'],pixel_mae=float(np.abs(x-y).mean()),pixel_max=float(np.abs(x-y).max())))
            z=tensors['inverse_unet_0_arg0'][:1].to(a.device)/inverse.vae.config.scaling_factor
            with torch.no_grad():
                turbo=(inverse.vae.decode(z).sample+1)/2
                vae21=(aux.vae.decode(z).sample+1)/2
            save_image(turbo[0],str(folder/'turbo_vae_roundtrip.png'))
            save_image(vae21[0],str(folder/'sd21_vae_roundtrip.png'))
            with Image.open(inputs[case]) as im: im.resize((512,512)).save(folder/'resized_input.png')
            controllers=[p.controller for p in model.unet.attn_processors.values() if getattr(p,'controller',None) is not None]
            mask=controllers[0].mask_s.detach().cpu();torch.save(mask,folder/'mask.pt');save_image(mask,str(folder/'mask.png'))
            rec=dict(case_id=case,status='complete',seconds=time.monotonic()-start,source_sha256=sha(inputs[case]),
                original_raw_sha256=row['source_sha256'],native_table_configuration=False,
                difference_to_old_interpretation='input preprocessing changed; not a replay parity test',
                prompt=row['prompt'],parity=parity,mask_fraction=float(mask.float().mean()),
                tensor_shapes={k:list(v.shape) for k,v in tensors.items()},stage_files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
            dump(case+'_receipt.json',rec);result['records'].append(rec);event('case_complete',case_id=case,parity=parity,seconds=rec['seconds'])
            del images,turbo,vae21,z,tensors
        result['status']='complete'
except BaseException as e:
    result.update(status='failed',error=str(e),traceback=traceback.format_exc());event('failed',error=str(e));raise
finally:
    dump('result.json',result)
