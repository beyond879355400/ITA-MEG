"""Root-authorized complete formal128 measurement; versioned outputs only."""
from pathlib import Path
from datetime import datetime,timezone
import json, os, shutil, subprocess, sys
import extent_io as io

HERE=Path(__file__).resolve().parent
IO=HERE/'io_compat_launch_v3.py'
IO_SHA='7edd6001d4d3a8c8605f20acc046ae3cd1465ca0343df4b88f5e226e212faa38'
PROTO_SHA='281fb67d7bddeb3635d0bca2801307b4c546153efc56cab0cf20fc1ff201494f'

def save(path,x):
    with path.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()

def main():
    assert io.sha(IO)==IO_SHA
    auth=HERE/'formal_approval_root_v1.json';runtime=HERE/'runtime_bridged_root_v1.json'
    assert io.read_json(auth)['granted_by']=='root'
    assert io.read_json(runtime)['runtime_status']=='bridged_environment_passed'
    job=HERE/'formal128_v1';out=HERE/'formal128_execution_v1';out.mkdir(exist_ok=False)
    boot=('from pathlib import Path;import hashlib;p=Path('+repr(str(IO))+');f=p.open("rb",buffering=0);b=f.read(p.stat().st_size);f.close();assert hashlib.sha256(b).hexdigest()=='+repr(IO_SHA)+';exec(compile(b,str(p),"exec"),{"__name__":"__main__","__file__":str(p)})')
    base=[str(HERE.parent/'runtime_local_v2/bin/python'),'-B','-c',boot]
    params=['--protocol',str(HERE/'protocol_v1.json'),'--protocol-sha',PROTO_SHA]
    for p,s in [(HERE.parent/'swiftedit/formal96_20260916_1603_cachebound/generation_index.json','8aeb28626ac79b67405357804e004ce77540a48307690cd8db9118b216ab7eaf'),(HERE/'performrecast_normalized_index_v1.json','fac2c6ff998b974650cb0f248d6cdb30447b8fc969e66c3c5d14877278c075b0')]:
        assert io.sha(p)==s;params+=['--generation-index',str(p),'--generation-sha',s]
    steps=[('preflight',base+['preflight',*params]),('prepare',base+['prepare',*params,'--output',str(job),'--approval',str(auth),'--execute'])]
    for kind in ['openface','facenet','buffalo']:
        steps.append((kind,base+[kind,'--job',str(job),'--runtime-receipt',str(runtime),'--runtime-sha',io.sha(runtime),'--approval',str(auth),'--execute']))
    steps.append(('export',base+['export','--job',str(job),'--approval',str(auth),'--execute']))
    save(out/'configuration.json',{'approval':io.ref(auth),'runtime':io.ref(runtime),'protocol_sha256':PROTO_SHA,'steps':[{'id':s,'argv':a}for s,a in steps],'cpu_only':True})
    for i,(name,args)in enumerate(steps):
        if shutil.disk_usage(HERE).free<8*1024**3:raise RuntimeError('8GiB disk reserve')
        print(json.dumps({'event':'start','phase':name,'utc':now()}),flush=True)
        start=now();logpath=out/(f'{i+1:02d}_{name}.log')
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','TMPDIR':str(HERE/'overlay_tmp_v1')}
        with logpath.open('x')as log:
            try:r=subprocess.run(args,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            except subprocess.TimeoutExpired:
                save(out/'failed_step.json',{'phase':name,'status':'timeout','log':str(logpath)});return 1
        receipt={'phase':name,'start':start,'finished':now(),'returncode':r.returncode,'log':str(logpath)}
        save(out/(f'{i+1:02d}_{name}_receipt.json'),receipt);print(json.dumps(receipt),flush=True)
        if r.returncode:return r.returncode
        if name=='prepare':
            d=io.read_json(job/'items.json');bad=[a for a in d['availability']if a['status']!='available']
            save(out/'input_check.json',{'items':len(d['items']),'planned':len(d['planned']),'unavailable':bad})
            if len(d['items'])!=192 or len(d['planned'])!=128 or bad:raise RuntimeError('Input IO unavailable: stop before measurement, preserve all entries')
    print(json.dumps({'status':'measurement_and_export_complete','job':str(job)}),flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
