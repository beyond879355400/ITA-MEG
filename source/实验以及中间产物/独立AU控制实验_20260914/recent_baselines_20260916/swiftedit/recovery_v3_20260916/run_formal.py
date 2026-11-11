"""Frozen32x3 SwiftEdit: same model/inference core, one load per seed."""
import argparse,hashlib,io,json,os,sys,tempfile,time
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent;SWIFT=HERE.parent;NEW=SWIFT.parent
TMP=NEW/'tmp_swiftedit_formal_v1';TMP.mkdir(exist_ok=True)
for key in ['TMPDIR','TEMP','TMP']:os.environ[key]=str(TMP)
tempfile.tempdir=str(TMP)
sys.path.insert(0,str(SWIFT))
import run_first as core

MANIFEST_SHA='fa88ca09eb8253ce61ba046cb8d9b4c4affaef6cc421893cedb487c09522413b'
MANIFEST_BYTES=50750
SEEDS=[101,202,303]
SOURCE_MANIFEST=core.BUNDLE/'inputs/postfreeze32.json'
FROZEN_MANIFEST=HERE/'verified_inputs/postfreeze32.json'
HOT_SOURCES={}


def validate_manifest(raw):
    core.need(len(raw)==MANIFEST_BYTES and hashlib.sha256(raw).hexdigest()==MANIFEST_SHA,'Frozen32 manifest size/SHA mismatch')
    doc=json.loads(raw)
    core.need(doc['kind']=='recent_baseline_case_inputs_v1' and doc['selection']=='postfreeze32' and doc['case_count']==32 and len(doc['cases'])==32,'Wrong32manifest')
    core.need(len({r['case_id'] for r in doc['cases']})==32,'Duplicate cases')
    for row in doc['cases']:
        core.need(row['sampling_seeds']==SEEDS and row['split']=='postfreeze_case_validation','Original protocol mismatch')
        core.need(not any(row['source_au_vector_0_5']) and row['prompt'],'Nonneutral/unprompted case unexpected')
    return doc


def stage_manifest():
    if FROZEN_MANIFEST.exists():return validate_manifest(FROZEN_MANIFEST.read_bytes())
    attempts=[]
    for attempt in range(3):
        raw=SOURCE_MANIFEST.read_bytes()
        attempts.append({'attempt':attempt+1,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        if len(raw)==MANIFEST_BYTES and hashlib.sha256(raw).hexdigest()==MANIFEST_SHA:
            doc=validate_manifest(raw)
            with FROZEN_MANIFEST.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            core.save_new(HERE/'verified_inputs/formal_manifest_receipt.json',{'source':str(SOURCE_MANIFEST),'copy':str(FROZEN_MANIFEST),'sha256':MANIFEST_SHA,'attempts':attempts,'original_modified':False})
            return doc
        time.sleep(attempt+1)
    raise ValueError('Original32manifest unavailable; refusing empty/mismatched read: '+json.dumps(attempts))


def configure(seed):
    core.need(seed in SEEDS,'Only frozen three seeds')
    from hot_vae_route import install
    install(core,SWIFT)
    core.SEED=seed;core.MANIFEST=FROZEN_MANIFEST
    original_read=core.read
    def permission_read(path):
        if Path(path).resolve()==(SWIFT/'approvals.json').resolve():
            doc=original_read(HERE/'approvals.json');grant=doc.get('formal_batch_go')
            if not grant:return {'cache_ready':doc['cache_ready'],'gpu_run_go':None}
            return {'cache_ready':doc['cache_ready'],'gpu_run_go':{**grant,'case_ids':[r['case_id'] for r in validate_manifest(FROZEN_MANIFEST.read_bytes())['cases']], 'seed':seed,'device':'mps'}}
        return original_read(path)
    def gate(key):
        entry=permission_read(SWIFT/'approvals.json').get(key)
        core.need(isinstance(entry,dict) and entry.get('granted_by')=='root' and entry.get('message'),'Formal root gate absent')
    def cases(start=0,count=32):
        core.need(start==0 and count==32,'No formal case subsampling')
        rows=validate_manifest(FROZEN_MANIFEST.read_bytes())['cases']
        for row in rows:
            path=core.relative_inside(core.BUNDLE,row['source_path'])
            if str(path) not in HOT_SOURCES:
                for attempt in range(3):
                    raw=path.read_bytes()
                    if raw and hashlib.sha256(raw).hexdigest()==row['source_sha256']:
                        HOT_SOURCES[str(path)]=raw;break
                    time.sleep(attempt+1)
                core.need(str(path) in HOT_SOURCES,'Frozen source bytes unavailable/mismatched after3reads')
            core.need(hashlib.sha256(HOT_SOURCES[str(path)]).hexdigest()==row['source_sha256'],'Cached source SHA mismatch')
        return rows
    core.read=permission_read;core.gate=gate;core.fixed_cases=cases
    old_preflight=core.preflight
    def preflight(*args,**kwargs):
        result=old_preflight(*args,**kwargs)
        core.need(Path(sys.prefix).resolve()==(NEW/'runtime_local_v2').resolve(),'Fresh runtime required')
        result.update(formal_wrapper_sha256=core.sha(Path(__file__)),runtime_lock_sha256=core.sha(HERE/'requirements.lock'),
                      case_manifest_sha256=MANIFEST_SHA,temporary_root=str(TMP),sampling_seeds=SEEDS,
                      source_input_transport='PIL Image.open(BytesIO(exact SHA-verified original PNG bytes)); unchanged author resize',
                      hot_source_count=len(HOT_SOURCES),hot_source_bytes=sum(map(len,HOT_SOURCES.values())),
                      hot_vae_migration_sha256=core.sha(SWIFT/'hot_assets_v1/migration_receipt.json'),
                      hot_vae_route_sha256=core.sha(HERE/'hot_vae_route.py'),
                      declared_comparison_scope='fixed source_text community-archive comparison; no automatic thesis change')
        return result
    core.preflight=preflight
    from PIL import Image
    original_open=Image.open
    def from_hot_bytes(fp,*args,**kwargs):
        key=str(Path(fp).resolve()) if isinstance(fp,(str,Path)) else None
        return original_open(io.BytesIO(HOT_SOURCES[key]),*args,**kwargs) if key in HOT_SOURCES else original_open(fp,*args,**kwargs)
    Image.open=from_hot_bytes


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--verify-assets',action='store_true')
    p.add_argument('--run-name',required=True)
    args=p.parse_args();core.need(Path(args.run_name).name==args.run_name,'Unsafe run name')
    stage_manifest();configure(args.seed)
    args.start=0;args.count=32;args.device='mps'
    if args.execute:core.execute(args)
    else:print(json.dumps(core.preflight(0,32,args.verify_assets),ensure_ascii=False,indent=2))
