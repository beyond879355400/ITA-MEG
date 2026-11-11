"""Exact pinned asset preverification only; no model construction."""
import json
from pathlib import Path
import shutil
import types
import extent_io as extent

HERE=Path(__file__).resolve().parent;EXP=HERE.parent.parent
helper=HERE.parent/'runtime_io_v1.py';HELPER_SHA='0562670cf396abf680f4eeebe0b33b8db27e5bf1d389583f8e75d47ce0ecf2e8'
source=extent.read_bytes(helper,HELPER_SHA);module=types.ModuleType('pinned_exact_runtime_io');module.__file__=str(helper)
exec(compile(source,str(helper),'exec'),module.__dict__)
if shutil.disk_usage(HERE).free<8*1024**3:raise RuntimeError('EightGiB reserve unavailable')
policy=extent.read_json(EXP/'development_evaluation/evaluation_policy_v2/policy.json','d29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c')
rows=[]
for name,policy_key in [('openface','openface'),('facenet','identity_primary')]:
    root=EXP/'tools'/name;seal=extent.read_json(root/'tool_seal.json',policy[policy_key]['tool_seal_sha256'])
    assets=[seal['binary'],seal['launcher'],*seal['model_files']]if name=='openface'else [*seal['source_files'],*seal['weights']]
    for asset in assets:
        path=root/asset['path'];rows.append(module.hash_exact(path,path.stat().st_size,asset['sha256']))
    if name=='facenet':
        for asset in seal['weights']:
            path=root/'runtime/facenet_pytorch/data'/Path(asset['path']).name
            if path.is_file():rows.append(module.hash_exact(path,path.stat().st_size,asset['sha256']))
root=EXP.parent/'baselines/ganimation_emotionnet_20260718/evaluation_models/insightface/models/buffalo_l'
for name,pin in policy['identity_auxiliary']['model_sha256'].items():
    path=root/name;rows.append(module.hash_exact(path,path.stat().st_size,pin))
out=HERE/'bridge_assets_preverified_v1.json'
with out.open('x')as f:json.dump({'status':'pass','exact_io_source':str(helper),'exact_io_sha256':HELPER_SHA,
    'model_calls':False,'cuda_mps_calls':False,'assets':rows,'asset_count':len(rows),'total_verified_bytes':sum(r['bytes']for r in rows)},f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'asset_preverify_pass','assets':len(rows),'bytes':sum(r['bytes']for r in rows),'models_called':False}))
