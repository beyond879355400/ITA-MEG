"""Root-authorized byte-identical private dylib copies; no original mutation."""
from pathlib import Path
import hashlib, json, os, stat, shutil
import extent_io as io

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'tools/openface/runtime/lib'
OUT=HERE/'openface_hot_runtime_v1'

def main():
    if shutil.disk_usage(HERE).free<8*1024**3:raise RuntimeError('8GiB reserve')
    OUT.mkdir(exist_ok=False)
    # Only dynamic library entries required by this sealed standalone runtime.
    entries=sorted(p for p in OLD.iterdir()if p.name.endswith('.dylib'))
    resolved={p.resolve()for p in entries}
    if any(not p.is_relative_to(OLD.resolve())for p in resolved):raise RuntimeError('External dylib symlink')
    total=sum(p.stat().st_size for p in resolved)
    if total>512*1024**2:raise RuntimeError('Unexpected >512MiB dylib copy')
    result=[]
    for original in sorted(resolved):
        before=original.stat()
        try:data=io.read_bytes(original)
        except ValueError:
            after=original.stat()
            if not(before.st_flags&0x40000000 and not after.st_flags&0x40000000 and
                   (before.st_dev,before.st_ino,before.st_size)==(after.st_dev,after.st_ino,after.st_size)):raise
            data=io.read_bytes(original)
        digest=hashlib.sha256(data).hexdigest();target=OUT/original.name
        with target.open('xb',buffering=0)as f:
            f.write(data);os.fsync(f.fileno())
        target.chmod(stat.S_IMODE(original.stat().st_mode))
        if io.sha(target)!=digest:raise RuntimeError('Copy SHA mismatch')
        result.append({'original':str(original),'target':str(target),'bytes':len(data),'sha256':digest,
                       'target_flags':target.stat().st_flags,'nlink':target.stat().st_nlink})
        print(json.dumps({'copied':len(result),'total':len(resolved),'name':original.name,'bytes':len(data)}),flush=True)
    links=[]
    for p in entries:
        if p.name==p.resolve().name:continue
        target=OUT/p.name;target.symlink_to(p.resolve().name)
        links.append({'alias':p.name,'target':p.resolve().name})
    with (OUT/'migration_receipt.json').open('x')as f:
        json.dump({'status':'pass','scope':'byte-identical library relocation only','source_root':str(OLD),
                   'files':result,'aliases':links,'bytes':total,'originals_changed':False},f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'pass','files':len(result),'aliases':len(links),'bytes':total}),flush=True)

if __name__=='__main__':main()
