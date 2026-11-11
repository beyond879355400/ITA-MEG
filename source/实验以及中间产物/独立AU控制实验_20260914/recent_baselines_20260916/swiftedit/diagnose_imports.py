"""Final bounded import-only comparison; never constructs model parameters."""
import argparse
import faulthandler
import importlib._bootstrap_external as loaders
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def child(log_path):
    stream=Path(log_path).open('x',buffering=1)
    faulthandler.dump_traceback_later(55,exit=True,file=stream)
    original=loaders.FileLoader.get_data
    def watched(self,path):
        before=time.monotonic()
        stream.write(json.dumps({'event':'read_begin','path':str(path),'flags':getattr(os.stat(path),'st_flags',None) if os.path.exists(path) else None})+'\n')
        result=original(self,path)
        stream.write(json.dumps({'event':'read_done','path':str(path),'seconds':time.monotonic()-before,'bytes':len(result)})+'\n')
        return result
    loaders.FileLoader.get_data=watched
    sys.path[:0]=[str(HERE/'downloads/ftfy-6.3.1-py3-none-any.whl'),str(HERE/'versions/device_v2_full_ip')]
    before=time.monotonic()
    try:
        import torch,diffusers,transformers,numpy
        from models import InverseModel,AuxiliaryModel,IPSBV2Model
        import infer
        result={'status':'import_only_pass','seconds':time.monotonic()-before,'torch':torch.__version__,
                'diffusers':diffusers.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,
                'model_construction':False,'interpreter':sys.executable,'pycache_prefix':sys.pycache_prefix}
        stream.write(json.dumps(result)+'\n');print(json.dumps(result),flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later();stream.close()


def main():
    interpreters=[('magicface','/Users/mayfair/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python'),
                  ('perform_overlay',str(HERE.parent/'performrecast/env_overlay_v1/bin/python'))]
    rows=[]
    for label,python in interpreters:
        log=HERE/'logs'/('final_import_probe_'+label+'.txt')
        command=[python,'-B',str(Path(__file__)),'--child',str(log)]
        env=dict(os.environ);env.pop('PYTHONPYCACHEPREFIX',None)
        start=time.monotonic()
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=60,env=env)
            row={'label':label,'command':command,'seconds':time.monotonic()-start,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'trace':str(log)}
        except subprocess.TimeoutExpired as error:
            row={'label':label,'command':command,'seconds':time.monotonic()-start,'returncode':'timeout60','stdout':str(error.stdout),'stderr':str(error.stderr),'trace':str(log)}
        print(json.dumps(row),flush=True);rows.append(row)
    with (HERE/'logs/final_import_probes.json').open('x') as stream:json.dump(rows,stream,ensure_ascii=False,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--child');args=parser.parse_args()
    child(args.child) if args.child else main()
