"""Exact-source report bootstrap; no model or detector calls."""
from pathlib import Path
import sys,types,importlib.machinery
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime_overlay_v1')]
import extent_io as io
original_read=io.read_bytes
def stable_read(path,*args,**kwargs):
    before=Path(path).stat()
    try:return original_read(path,*args,**kwargs)
    except ValueError as error:
        after=Path(path).stat()
        if(str(error).startswith(('File changed','Path changed'))and before.st_flags&0x40000000 and not after.st_flags&0x40000000 and (before.st_dev,before.st_ino,before.st_size)==(after.st_dev,after.st_ino,after.st_size)):
            return original_read(path,*args,**kwargs)
        raise
io.read_bytes=stable_read
importlib.machinery.SourceFileLoader.get_data=lambda self,path:io.read_bytes(path,nonempty=False)
sys.dont_write_bytecode=True
p=types.ModuleType('protocol');p.__file__=str(HERE/'protocol.py');sys.modules['protocol']=p
exec(compile(io.read_bytes(p.__file__,'8345ce3765fc9a436053f28626b6c0146e0f187b11d67b23243b8d2e1e7fe041'),p.__file__,'exec'),p.__dict__)
p.sha=io.sha;p.read=io.read_json
path=HERE/'render_report.py'
exec(compile(io.read_bytes(path,'c242e830064b242b0124e6b3be0d88cc04bec3d32f162c401ca02ee9463572fc'),str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
