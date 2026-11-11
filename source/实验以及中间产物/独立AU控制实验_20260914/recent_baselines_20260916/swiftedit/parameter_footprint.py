"""Read safetensors JSON headers only; estimate FP32 resident parameter bytes."""
import json
import struct
from stage_weights import HERE,read,need

paths=[HERE/'weights/official_v1_20260916_132327/swiftedit_weights/inverse_ckpt-120k/unet_ema/diffusion_pytorch_model.safetensors',
       *[HERE/'weights/auxiliary'/r['local_path'] for r in read(HERE/'aux_components.json')['components'] if r['path'].endswith('.safetensors')]]
rows=[]
for path in paths:
    with path.open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        need(size<16*1024*1024,'Unexpectedly large header')
        meta=json.loads(stream.read(size))
    values=[v for k,v in meta.items() if k!='__metadata__']
    need(all(v['dtype'] in ['F32','I64','I32','BOOL'] for v in values),'Unexpected dtype needs explicit accounting')
    dtypes={dtype:sum(v['data_offsets'][1]-v['data_offsets'][0] for v in values if v['dtype']==dtype) for dtype in sorted({v['dtype'] for v in values})}
    rows.append({'path':str(path),'tensor_bytes':sum(v['data_offsets'][1]-v['data_offsets'][0] for v in values),'tensor_count':len(values),'dtype_bytes':dtypes})
proof=read(HERE/'logs/full_ip_meta_coverage.json')
rows.append({'path':'full IP generator registered state (not initial SB weights)',
             'tensor_bytes':proof['parameter_numel']*4,'tensor_count':proof['unique_parameter_count']})
total=sum(r['tensor_bytes'] for r in rows)
print(json.dumps({'rows':rows,'resident_parameter_bytes':total,'resident_parameter_GiB':total/1024**3,
                  'excludes':'activations,attention matrices,allocator caches,temporary CPU checkpoint loading,framework/system memory',
                  'not_peak_memory_measurement':True,'model_tensor_loading':False},ensure_ascii=False,indent=2))
