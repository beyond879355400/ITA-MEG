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
def verify_asset(path,pin):
    before=module.signature(path.stat());flags=path.stat().st_flags
    try:
        result=module.hash_exact(path,before['size'],pin)
        with (HERE/'bridge_preverify_progress_v2.jsonl').open('a')as f:json.dump(result,f,ensure_ascii=False);f.write('\n')
        return result
    except BaseException as error:
        failure={'status':'failed','path':str(path),'expected_sha256':pin,'before':before,'after':module.signature(path.stat()),'flags_before':flags,'flags_after':path.stat().st_flags,'error':str(error),'already_verified_count':len(rows),'model_calls':False}
        with (HERE/'bridge_preverify_failure_v2.json').open('x')as f:json.dump(failure,f,ensure_ascii=False,indent=2)
        print(json.dumps(failure,ensure_ascii=False),flush=True)
        raise

for name,policy_key in [('openface','openface'),('facenet','identity_primary')]:
    root=EXP/'tools'/name;seal=extent.read_json(root/'tool_seal.json',policy[policy_key]['tool_seal_sha256'])
    assets=[seal['binary'],seal['launcher'],*seal['model_files']]if name=='openface'else [*seal['source_files'],*seal['weights']]
    for asset in assets:
        path=root/asset['path'];rows.append(verify_asset(path,asset['sha256']))
    if name=='facenet':
        for asset in seal['weights']:
            path=root/'runtime/facenet_pytorch/data'/Path(asset['path']).name
            if path.is_file():rows.append(verify_asset(path,asset['sha256']))
root=EXP.parent/'baselines/ganimation_emotionnet_20260718/evaluation_models/insightface/models/buffalo_l'
for name,pin in policy['identity_auxiliary']['model_sha256'].items():
    path=root/name;rows.append(verify_asset(path,pin))
out=HERE/'bridge_assets_preverified_v2.json'
with out.open('x')as f:json.dump({'status':'pass','exact_io_source':str(helper),'exact_io_sha256':HELPER_SHA,
    'model_calls':False,'cuda_mps_calls':False,'assets':rows,'asset_count':len(rows),'total_verified_bytes':sum(r['bytes']for r in rows)},f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'asset_preverify_pass','assets':len(rows),'bytes':sum(r['bytes']for r in rows),'models_called':False}))
