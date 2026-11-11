"""Root-authorized sequential two-case CPU bridge only; stop on first failure."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import extent_io as extent

HERE=Path(__file__).resolve().parent
IO=HERE/'io_compat_launch_v2.py'
IO_SHA='3bcef86da482d3814d66d3e13eeb47c596f355ce92f79015ee2fb19f92263f5b'
ROOT_AUTH=HERE/'bridge_approval_root.json'


def now():return datetime.now(timezone.utc).isoformat()
def save(p,value):
    with Path(p).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    assets=extent.read_json(HERE/'bridge_assets_preverified_v3.json')
    if assets['status']!='pass' or assets['model_calls'] is not False:raise RuntimeError('Asset preverify not passed')
    input_check=extent.read_json(HERE/'bridge_inputs_preverified_v1.json')
    if input_check['status']!='pass':raise RuntimeError('Fixed bridge inputs not SHA verified')
    plan=extent.read_json(HERE/'bridge_command_plan.json');auth=extent.read_json(ROOT_AUTH)
    if auth['granted_by']!='root' or auth['protocol_sha256']!=plan['protocol_sha256']:raise RuntimeError('Root authorization missing')
    if extent.sha(IO)!=IO_SHA:raise RuntimeError('IO compatibility source changed')
    runtime=HERE/'runtime_candidate_root.json';runtime_sha=extent.sha(runtime)
    out=HERE/'bridge_identity_execution_v4';out.mkdir(exist_ok=False)
    commands=[]
    boot=('from pathlib import Path; import hashlib,os; p=Path('+repr(str(IO))+'); s=p.stat(); f=p.open("rb",buffering=0); b=f.read(s.st_size); a=os.fstat(f.fileno()); f.close(); '
          'assert b and len(b)==s.st_size and (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns) and hashlib.sha256(b).hexdigest()=='+repr(IO_SHA)+'; '
          'exec(compile(b,str(p),"exec"),{"__name__":"__main__","__file__":str(p)})')
    for step in plan['steps']:
        if step['id'] not in ['neutral_prepare','edited_prepare','neutral_facenet','neutral_buffalo','edited_facenet','edited_buffalo']:continue
        argv=[str(x).replace('<ROOT_RUNTIME_RECEIPT_SHA256>',runtime_sha).replace('bridge_neutral_v1','bridge_neutral_v2').replace('bridge_edited_v1','bridge_edited_v2')for x in step['argv']]
        argv[3]=boot;commands.append({**step,'argv':argv})
    save(out/'configuration.json',{'root_authorization':extent.ref(ROOT_AUTH),'protocol_sha256':plan['protocol_sha256'],
           'runtime_candidate':extent.ref(runtime),'asset_preverification':extent.ref(HERE/'bridge_assets_preverified_v3.json'),
           'io_adapter':extent.ref(IO),'io_scope':'exact-size source/JSON/fileSHA reads only; algorithm sourcebytes unchanged',
           'steps':commands,'cpu_only':True,'MPS_or_CUDA':False,'formal128_measurement':False})
    done=[]
    for i,step in enumerate(commands):
        if shutil.disk_usage(HERE).free<8*1024**3:raise RuntimeError('EightGiB reserve reached before bridge step')
        print(json.dumps({'event':'bridge_step_start','step':step['id'],'at':now()},ensure_ascii=False),flush=True)
        logfile=out/(f'{i+1:02d}_'+step['id']+'.log');start=now()
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','TMPDIR':str(HERE/'overlay_tmp_v1')}
        with logfile.open('x')as log:
            try:r=subprocess.run(step['argv'],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            except subprocess.TimeoutExpired:
                row={'step':step['id'],'started':start,'finished':now(),'status':'timeout','log':str(logfile),'timeout_seconds':180}
                save(out/'failed_step.json',row);print(json.dumps(row),flush=True);return 1
        row={'step':step['id'],'started':start,'finished':now(),'returncode':r.returncode,'log':str(logfile)};done.append(row)
        save(out/(f'{i+1:02d}_'+step['id']+'_receipt.json'),row)
        print(json.dumps({'event':'bridge_step_end',**row},ensure_ascii=False),flush=True)
        if r.returncode:
            save(out/'failed_step.json',row);return r.returncode
    save(out/'completion.json',{'status':'identity_phases_complete_pending_openface_and_comparison','steps':done,'formal128_run':False})
    print('Identity phases complete; no numerical bridge PASS claimed',flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
