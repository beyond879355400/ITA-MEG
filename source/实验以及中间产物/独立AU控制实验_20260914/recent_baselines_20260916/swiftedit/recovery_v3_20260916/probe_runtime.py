"""Fresh-runtime import gate; no pretrained model construction."""
import faulthandler,hashlib,importlib.metadata,json,os,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
NEW=HERE.parent.parent
assert Path(sys.prefix).resolve()==(NEW/'runtime_local_v2').resolve()
assert not any('geme-main' in value or '.venv_magicface' in value for value in sys.path)
start=time.monotonic()
with (HERE/'import_trace.log').open('x') as trace:
    faulthandler.dump_traceback_later(75,exit=True,file=trace)
    import numpy,torch,torchvision,diffusers,transformers,accelerate,safetensors,ftfy,einops
    sys.path.insert(0,str(HERE.parent/'versions/device_v2_full_ip'))
    from models import InverseModel,AuxiliaryModel,IPSBV2Model
    from infer import edit_image
    faulthandler.cancel_dump_traceback_later()
modules=[numpy,torch,torchvision,diffusers,transformers,accelerate,safetensors,ftfy,einops]
rows=[]
for module in modules:
    path=Path(module.__file__);info=path.stat()
    assert not info.st_flags&0x40000000
    assert info.st_nlink==1
    rows.append({'name':module.__name__,'version':importlib.metadata.version(module.__name__),
                 'path':str(path),'flags':info.st_flags,'nlink':info.st_nlink})
result={'status':'pass','seconds':time.monotonic()-start,'interpreter':sys.executable,'sys_path':sys.path,
        'mps_available':torch.backends.mps.is_available(),'model_constructed':False,'old_environment_imported':False,
        'modules':rows,'free_bytes':__import__('shutil').disk_usage(NEW).free,
        'requirements_lock_sha256':hashlib.sha256((HERE/'requirements.lock').read_bytes()).hexdigest()}
with (HERE/'runtime_import_pass.json').open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
