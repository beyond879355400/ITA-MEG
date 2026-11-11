"""Pinned fixed bridge images only; diagnose hydration metadata changes."""
import json
from pathlib import Path
import types
import extent_io as extent

HERE=Path(__file__).resolve().parent;EXP=HERE.parent.parent
source=HERE.parent/'runtime_io_v1.py';code=extent.read_bytes(source,'0562670cf396abf680f4eeebe0b33b8db27e5bf1d389583f8e75d47ce0ecf2e8')
m=types.ModuleType('pinned_io');exec(compile(code,str(source),'exec'),m.__dict__)
cases=extent.read_json(HERE.parent/'preservation/bundle_v2_20260916/inputs/postfreeze32.json','fa88ca09eb8253ce61ba046cb8d9b4c4affaef6cc421893cedb487c09522413b')['cases']
ids=[cases[0]['case_id'],next(c['case_id']for c in cases if c['source_au_vector_0_5']!=c['target_au_vector_0_5'])]
items=extent.read_json(EXP/'completion_20260915/evaluation/all_methods_v1/items.json','390cd5f571b60b60a12c005717f751720c19c5b35d32d72ba777629ab680571e')['items']
picked=[r for r in items if r['case_id']in ids and (r['method']in ['source_real','target_real']or r['method']=='bidirectional_local_lp_s050'and r['seed']==101)]
assert len(picked)==6
cache={};records=[]
for row in picked:
    p=Path(row['path']);key=(str(p),row['sha256'])
    if key not in cache:
        before=m.signature(p.stat());flags=p.stat().st_flags
        try:result=m.hash_exact(p,before['size'],row['sha256'])
        except ValueError as error:
            after=m.signature(p.stat());after_flags=p.stat().st_flags
            if str(error)!='File changed during read'or before['device']!=after['device']or before['inode']!=after['inode']or before['size']!=after['size']or not flags&0x40000000 or after_flags&0x40000000:raise
            result=m.hash_exact(p,after['size'],row['sha256'])
            result['materialization_retry']={'before':before,'after':after,'flags_before':flags,'flags_after':after_flags}
        cache[key]=result
    records.append({'case_id':row['case_id'],'method':row['method'],'seed':row['seed'],**cache[key]})
with (HERE/'bridge_inputs_preverified_v1.json').open('x')as f:json.dump({'status':'pass','cases':ids,'records':records,'unique_images':len(cache),'models_called':False},f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'fixed_inputs_pass','logical_images':len(records),'unique_images':len(cache),'models_called':False}))
