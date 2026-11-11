"""Root-approved <1GiB evaluation overlay; never install core or run models."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
NEW=HERE.parent
CORE=NEW/'runtime_local_v2'
OLD=Path('/Users/mayfair/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/lib/python3.10/site-packages')
SITE=HERE/'runtime_overlay_v1'
TMP=HERE/'overlay_tmp_v1'
RESERVE=8*1024**3
CAP=1024**3


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256();n=0;size=Path(path).stat().st_size
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b);n+=len(b)
    need(n==size,'Dataless/truncated read: '+str(path));return h.hexdigest()


def save(path,data):
    with Path(path).open('x')as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')


def total(root):return sum(p.stat().st_size for p in root.rglob('*')if p.is_file()and not p.is_symlink())


def guard(extra=0):
    need(shutil.disk_usage(HERE).free>=RESERVE+extra,'EightGiB reserve reached')
    if SITE.exists():need(total(SITE)+extra<CAP,'OneGiB overlay cap reached')


def copy_one(source,dest):
    source=Path(source);dest=Path(dest)
    need(source.is_relative_to(OLD)and dest.is_relative_to(SITE),'Copy boundary')
    need(not source.is_symlink()and source.is_file()and not dest.exists(),'Only new regular-file copy')
    before=source.stat();data=source.read_bytes()
    need(len(data)==before.st_size,'Dataless empty/short read rejected')
    guard(len(data));dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb')as f:f.write(data)
    h=hashlib.sha256(data).hexdigest();need(sha(dest)==h,'Copy SHA mismatch')
    return {'source':str(source),'path':str(dest),'bytes':len(data),'sha256':h,'source_flags_before':before.st_flags,
            'source_mtime_unchanged':source.stat().st_mtime_ns==before.st_mtime_ns,'new_inode':source.stat().st_ino!=dest.stat().st_ino}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--copy-one',nargs=2)
    a=parser.parse_args()
    if a.copy_one:print(json.dumps(copy_one(*a.copy_one)));return
    if not a.execute:print(json.dumps({'action':'private exact-version PyPI wheel overlay and byte-verified old InsightFace copy','core_unchanged':True,'cap_bytes':CAP,'reserve':RESERVE}));return
    guard();need(not SITE.exists(),'Append-only: overlay already exists');TMP.mkdir(exist_ok=False)
    req=HERE/'overlay_requirements.txt';pins=req.read_text().splitlines()
    need(not any(r.lower().startswith(('torch','numpy','diffusers','transformers'))for r in pins),'Core install forbidden')
    cmd=[str(Path.home()/'.local/bin/uv'),'pip','install','--python',str(CORE/'bin/python'),
         '--target',str(SITE),'--no-deps','--only-binary',':all:','--default-index','https://pypi.org/simple',
         '--no-config','--no-cache','--link-mode','copy','-r',str(req)]
    env={**os.environ,'TMPDIR':str(TMP),'PYTHONDONTWRITEBYTECODE':'1','PIP_DISABLE_PIP_VERSION_CHECK':'1','NO_ALBUMENTATIONS_UPDATE':'1'}
    save(HERE/'overlay_install_plan.json',{'command':cmd,'requirements_sha256':sha(req),'core':str(CORE),
          'started':datetime.now(timezone.utc).isoformat(),'root_authorized_overlay_only':True,'source':'public PyPI exact versions, binary wheels only'})
    with (HERE/'overlay_install.log').open('x')as log:
        p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    need(p.returncode==0,'Wheel install failed; log/partial overlay retained');guard()
    records=[]
    try:
        for package in ['insightface','insightface-0.7.3.dist-info']:
            for source in sorted((OLD/package).rglob('*')):
                if not source.is_file()or '__pycache__'in source.parts:continue
                dest=SITE/source.relative_to(OLD)
                result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--copy-one',str(source),str(dest)],
                                      capture_output=True,text=True,timeout=20,env=env)
                need(result.returncode==0,'Guarded source copy failed: '+result.stderr[-2000:])
                records.append(json.loads(result.stdout))
        guard();save(HERE/'insightface_copy_receipt.json',{'status':'complete','records':records,'no_compilation':True,'old_source_modified':False})
    except BaseException as error:
        save(HERE/'insightface_copy_failure.json',{'status':'failed','error':str(error),'completed':records,'old_source_modified':False});raise
    files=[]
    for p in sorted(SITE.rglob('*')):
        if not p.is_file():continue
        need(not p.is_symlink(),'Overlay must not link old env/cache');st=p.stat()
        need(not st.st_flags&0x40000000,'Overlay remained SF_DATALESS')
        files.append({'path':str(p.relative_to(SITE)),'bytes':st.st_size,'sha256':sha(p),'nlink':st.st_nlink,'flags':st.st_flags})
    save(HERE/'overlay_files.json',{'status':'complete','total_bytes':total(SITE),'files':files,'core_unchanged':True,'no_models_run':True})
    print(json.dumps({'status':'overlay_prepared_import_pending','bytes':total(SITE),'files':len(files),'free_bytes':shutil.disk_usage(HERE).free}))


if __name__=='__main__':main()
