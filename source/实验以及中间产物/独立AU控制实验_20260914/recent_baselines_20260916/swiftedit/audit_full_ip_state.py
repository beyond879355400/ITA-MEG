"""Restricted metadata-only checkpoint coverage audit; no storage tensor load.

Torch2.1's map_location=meta still reads CPU storages first. Instead this
parses only data.pkl through an allowlisted unpickler with inert rebuilders;
never calls torch.load on the real archive. All constructed tensors are meta.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from stage_weights import HERE, need, read, sha, save_new, now

EXTRACT=HERE/'weights/official_v1_20260916_132327'
IP=EXTRACT/'swiftedit_weights/ip_adapter_ckpt-90k/ip_adapter.bin'
CONFIG=EXTRACT/'swiftedit_weights/sbv2_0.5/config.json'


@dataclass(frozen=True)
class Storage:
    key:str
    dtype:str
    numel:int
    original_device:str


@dataclass(frozen=True)
class Tensor:
    storage:Storage
    offset:int
    shape:tuple
    stride:tuple
    requires_grad:bool


def rebuild(storage,offset,shape,stride,requires_grad=False,hooks=None):
    need(isinstance(storage,Storage) and isinstance(offset,int) and offset>=0,'Invalid tensor metadata')
    need(isinstance(shape,tuple) and isinstance(stride,tuple) and len(shape)==len(stride),'Invalid shape/stride')
    need(all(isinstance(n,int) and n>=0 for n in shape+stride),'Invalid dimensions')
    need(hooks is None or isinstance(hooks,OrderedDict) and not hooks,'Unexpected serialized tensor hooks')
    return Tensor(storage,offset,shape,stride,bool(requires_grad))


class MetadataOnly(pickle.Unpickler):
    DTYPES={'FloatStorage':'float32','HalfStorage':'float16','DoubleStorage':'float64',
            'BFloat16Storage':'bfloat16','LongStorage':'int64','IntStorage':'int32',
            'ShortStorage':'int16','CharStorage':'int8','ByteStorage':'uint8','BoolStorage':'bool'}
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):return OrderedDict
        if module=='torch._utils' and name in ['_rebuild_tensor_v2','_rebuild_tensor']:return rebuild
        if module=='torch' and name in self.DTYPES:return self.DTYPES[name]
        raise ValueError('Unapproved pickle global: '+module+'.'+name)

    def persistent_load(self,value):
        need(isinstance(value,tuple) and len(value)==5 and value[0]=='storage','Unknown persistent object')
        _,dtype,key,device,numel=value
        need(dtype in self.DTYPES.values() and isinstance(key,str) and key.isdigit(),'Invalid storage key/dtype')
        need(isinstance(device,str) and isinstance(numel,int) and numel>=0,'Invalid storage location/size')
        return Storage(key,dtype,numel,device)


def metadata(path):
    widths={'float32':4,'float16':2,'float64':8,'bfloat16':2,'int64':8,'int32':4,'int16':2,'int8':1,'uint8':1,'bool':1}
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();need(len(names)==len(set(names)),'Duplicate ZIP names')
        pkls=[name for name in names if name.endswith('/data.pkl')]
        need(len(pkls)==1,'Expected one state dictionary pickle')
        pkl=archive.read(pkls[0]);need(len(pkl)<16*1024*1024,'Metadata unexpectedly large')
        state=MetadataOnly(io.BytesIO(pkl)).load()
        need(isinstance(state,OrderedDict) and all(isinstance(k,str) and isinstance(v,Tensor) for k,v in state.items()),'Not a plain tensor state dictionary')
        root=pkls[0][:-len('data.pkl')]
        storages={v.storage for v in state.values()}
        for storage in storages:
            need(archive.getinfo(root+'data/'+storage.key).file_size==storage.numel*widths[storage.dtype], 'Storage record length differs')
        return state,{'pickle_bytes':len(pkl),'pickle_sha256':hashlib.sha256(pkl).hexdigest(),
                      'storage_records':len(storages),'storage_data_bytes_read':0,
                      'serialized_storage_bytes':sum(v.numel*widths[v.dtype] for v in storages)}


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    failed=read(EXTRACT/'extraction_failure.json')
    for path in [IP,CONFIG]:
        row=next(row for row in failed['completed_files'] if row['path']==str(path.relative_to(EXTRACT)))
        need(sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'Previously extracted file changed')
    state,provenance=metadata(IP)
    for x in state.values():need(x.storage.dtype=='float32','Non-FP32 checkpoint requires separate casting-equivalence review')
    clip_dim=state['image_proj_model.proj.weight'].shape[1]
    need(clip_dim==1024,'Unexpected IP projection input')
    sys.path.insert(0,str(HERE/'versions/device_v1'))
    import torch
    from models import IPSBV2Model,UNet2DConditionModel
    constructed=[];captured=[]
    def construct(path,*args,**kwargs):
        need(Path(path).resolve()==CONFIG.parent,'Unexpected pretrained construction call')
        model=UNet2DConditionModel.from_config(read(CONFIG))
        need(all(x.is_meta for x in model.parameters()),'A real model parameter was allocated')
        constructed.append('from_config_meta');return model
    def provide_meta(path,*args,**kwargs):
        need(Path(path).resolve()==IP,'Unexpected checkpoint load request')
        captured.append({'requested_map_location':kwargs.get('map_location'),'weights_only':kwargs.get('weights_only')})
        # Shared serialized storage/offset/shape/stride means identical values.
        tensors={};result=OrderedDict()
        for key,value in state.items():
            alias=(value.storage.key,value.offset,value.shape,value.stride)
            if alias not in tensors:
                tensors[alias]=torch.empty_strided(value.shape,value.stride,device='meta',dtype=torch.float32)
            result[key]=tensors[alias]
        return result
    def no_network(*args,**kwargs):raise RuntimeError('No network allowed in meta audit')
    with patch.object(socket.socket,'connect',no_network),patch.object(UNet2DConditionModel,'from_pretrained',side_effect=construct),patch.object(torch,'load',side_effect=provide_meta),torch.device('meta'):
        aux=SimpleNamespace(noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000),alphas_cumprod=torch.empty(1000,device='meta')),
                            image_encoder=SimpleNamespace(config=SimpleNamespace(projection_dim=clip_dim)))
        model=IPSBV2Model(str(CONFIG.parent),str(IP),aux,device='meta',with_ip_mask_controller=True)
    need(constructed==['from_config_meta'] and len(captured)==1,'Unexpected construction/checkpoint count')
    expected=model.state_dict(keep_vars=True)
    need(set(state)==set(expected),'Full strict-state coverage differs')
    aliases={}
    for key,value in expected.items():
        need(value.is_meta and tuple(value.shape)==state[key].shape and str(value.dtype)=='torch.'+state[key].storage.dtype,'Shape/dtype/device mismatch: '+key)
        aliases.setdefault(id(value),[]).append(key)
    shared=[keys for keys in aliases.values() if len(keys)>1]
    for keys in shared:
        signatures={(state[key].storage.key,state[key].offset,state[key].shape,state[key].stride) for key in keys}
        need(len(signatures)==1,'Shared module aliases use different serialized values: '+str(keys))
    buffers=list(model.named_buffers(remove_duplicate=False))
    need(all(name in state for name,_ in buffers),'Nonpersistent buffer needs separate equivalence proof')
    need(all(value.is_meta for value in model.parameters()) and all(value.is_meta for _,value in buffers),'Real tensor allocated')
    report={'kind':'swiftedit_full_ip_checkpoint_meta_coverage','status':'pass','at':now(),
            'audit_code_sha256':sha(Path(__file__)),'ip_checkpoint_sha256':sha(IP),'sbv2_config_sha256':sha(CONFIG),
            'extraction_status':'partial_failed_retained_not_complete_archive','real_parameter_data_loaded':False,'gpu_used':False,
            'parser':provenance,'state_key_count':len(state),'unique_parameter_count':len(list(model.parameters())),
            'parameter_numel':sum(p.numel() for p in model.parameters()),'persistent_buffer_count':len(buffers),
            'dtype':'float32','strict_load_passed':True,'shape_dtype_coverage_passed':True,
            'shared_alias_groups':shared,'shared_alias_value_equivalence':'same serialized storage key/offset/shape/stride',
            'full_unet_keys':sum(k.startswith('unet.') for k in state),'adapter_module_keys':sum(k.startswith('adapter_modules.') for k in state),
            'image_projection_keys':sum(k.startswith('image_proj_model.') for k in state),
            'clip_projection_dim':clip_dim,'clip_projection_dim_requires_locked_aux_config_check':True,
            'unregistered_alpha_sigma_timestep':'derived from unchanged auxiliary scheduler in both constructions; excluded from checkpoint coverage',
            'rng_policy':'runner seeds random/numpy/torch/MPS with101 AFTER all model construction before EVERY case',
            'conditional_equivalence':'Given identical config and fixed auxiliary objects, from_config+strict full IP load yields identical final registered parameters/buffers and shared aliases; author SB initialization values are entirely overwritten. No pixel run or same-output empirical claim.'}
    save_new(HERE/'logs/full_ip_meta_coverage.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='shared_alias_groups'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
