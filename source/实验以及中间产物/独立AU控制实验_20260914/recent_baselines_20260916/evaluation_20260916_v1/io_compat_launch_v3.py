"""Equivalent exact-extent IO adapter; frozen evaluator/model bytes unchanged.

No selector, preprocessing, metric, tensor, RNG, weight or tolerance changes.
Only source/JSON reads and pure file-SHA helpers avoid a redundant OS EOF read.
"""
import hashlib
import importlib.machinery
import io
import json
import os
from pathlib import Path
import stat
import sys
import types

HERE=Path(__file__).resolve().parent
PROTOCOL_SHA='281fb67d7bddeb3635d0bca2801307b4c546153efc56cab0cf20fc1ff201494f'


def sig(s):return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns


def raw_bytes(path):
    path=Path(path);before=path.stat()
    if not stat.S_ISREG(before.st_mode):raise ValueError('Expected regular read source')
    chunks=[];left=before.st_size
    with io.open(path,'rb',buffering=0)as f:
        if sig(os.fstat(f.fileno()))!=sig(before):raise ValueError('File identity changed before read')
        while left:
            b=f.read(min(left,4*1024*1024))
            if not b:raise ValueError('Premature empty read: '+str(path))
            chunks.append(b);left-=len(b)
        after_fd=os.fstat(f.fileno())
    if sig(before)!=sig(after_fd)or sig(before)!=sig(path.stat()):raise ValueError('File changed during exact read')
    return b''.join(chunks)


def file_sha(path):
    # Stream and discard blocks: do not materialize large model bytes inRAM.
    path=Path(path);before=path.stat();left=before.st_size;h=hashlib.sha256()
    with io.open(path,'rb',buffering=0)as f:
        if sig(os.fstat(f.fileno()))!=sig(before):raise ValueError('File changed before hash')
        while left:
            b=f.read(min(left,4*1024*1024))
            if not b:raise ValueError('Premature empty hash read')
            h.update(b);left-=len(b)
        after_fd=os.fstat(f.fileno())
    if sig(before)!=sig(after_fd)or sig(before)!=sig(path.stat()):raise ValueError('File changed during hash')
    return h.hexdigest()


def checked(path,expected):
    data=raw_bytes(path)
    if not data or hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Source/pin mismatch')
    return data


def read_text(path,encoding=None,errors=None):
    # TextIOWrapper over memory preserves normal universal-newline/encoding
    # semantics; only the filesystem read is exact-length.
    return io.TextIOWrapper(io.BytesIO(raw_bytes(path)),encoding=encoding,errors=errors).read()



# Only retry the specifically diagnosed SF_DATALESS-to-resident metadata
# transition. The fresh second read must satisfy the original strict signature
# and every caller still checks the same frozen SHA. Never accept a changed read.
def materialized_retry(reader,path):
    before=Path(path).stat()
    try:return reader(path)
    except ValueError as error:
        after=Path(path).stat()
        if (str(error).startswith('File changed') and before.st_flags&0x40000000 and not after.st_flags&0x40000000
                and (before.st_dev,before.st_ino,before.st_size)==(after.st_dev,after.st_ino,after.st_size)):
            event={'path':str(path),'before':sig(before),'after':sig(after),'flags_before':before.st_flags,
                   'flags_after':after.st_flags,'action':'fresh exact reread required; original pin unchanged'}
            with io.open(HERE/'io_materialization_retries_v2.jsonl','a',encoding='utf-8')as log:
                log.write(json.dumps(event,ensure_ascii=False)+'\n');log.flush()
            return reader(path)
        raise
_raw_bytes_once=raw_bytes
_file_sha_once=file_sha
raw_bytes=lambda path:materialized_retry(_raw_bytes_once,path)
file_sha=lambda path:materialized_retry(_file_sha_once,path)

spec=json.loads(checked(HERE/'protocol_v1.json',PROTOCOL_SHA))
allowed={str(Path(r['path']).resolve())for r in spec['source_bindings'].values()if r['path'].endswith('.py')}
allowed.update(str(Path(r['path']).resolve())for r in spec['old_algorithm_spec']['algorithm_sources'].values())
allowed.update(str(Path(r['path']).resolve())for r in spec['old_algorithm_spec']['algorithm_dependencies'].values())
original_exec=importlib.machinery.SourceFileLoader.exec_module


def patch_io(module):
    path=getattr(module,'__file__','')
    if path and str(Path(path).resolve())in allowed:
        if callable(getattr(module,'sha',None)):module.sha=file_sha
        if callable(getattr(module,'read',None)):module.read=lambda p:json.loads(raw_bytes(p))


def source_get_data(self,path):return raw_bytes(path)
def source_exec(self,module):
    result=original_exec(self,module);patch_io(module);return result
importlib.machinery.SourceFileLoader.get_data=source_get_data
importlib.machinery.SourceFileLoader.exec_module=source_exec
Path.read_bytes=lambda self:raw_bytes(self)
Path.read_text=read_text
sys.dont_write_bytecode=True
sys.path[:0]=[str(HERE),str(HERE/'insightface_built_v1'),str(HERE/'runtime_overlay_v1')]
os.environ.update(PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  TMPDIR=str(HERE/'overlay_tmp_v1'),PYTHONNOUSERSITE='1')
p=types.ModuleType('protocol');p.__file__=str(HERE/'protocol.py');sys.modules['protocol']=p
exec(compile(checked(p.__file__,spec['execution_sources']['protocol.py']['sha256']),p.__file__,'exec'),p.__dict__)
p.sha=file_sha;p.read=lambda q:json.loads(raw_bytes(q))
e=types.ModuleType('evaluate_new');e.__file__=str(HERE/'evaluate_new.py');sys.modules['evaluate_new']=e
exec(compile(checked(e.__file__,spec['execution_sources']['evaluate_new.py']['sha256']),e.__file__,'exec'),e.__dict__)
# Root-approved same-byte dylib relocation for only the sealed OpenFace child.
import subprocess
_hot=HERE/'openface_hot_runtime_v2'
_migration=json.loads(checked(_hot/'migration_receipt.json','f316af33138de7f2fa69059ab4c8063ed5bb10c497e00db15babc1c410434f00'))
for _asset in _migration['files']:
    if file_sha(_asset['target'])!=_asset['sha256']:raise ValueError('Hot dylib SHA mismatch')
_tool=HERE.parents[1]/'tools/openface'
if file_sha(_tool/'build/bin/FaceLandmarkImg')!='665aa1e681a9efac2adf037559f7c38e2c2fffca0413c5c4dfca53342c655328':raise ValueError('Original OpenFace binary mismatch')
_popen_original=subprocess.Popen
class _SameByteOpenFacePopen(_popen_original):
    def __init__(self,args,*positional,**kwargs):
        if isinstance(args,(list,tuple))and args and str(args[0])==str(_tool/'FaceLandmarkImg'):
            original_args=list(args)
            args=[str(_tool/'build/bin/FaceLandmarkImg'),*args[1:]]
            env=dict(kwargs.get('env')or os.environ)
            env.update(DYLD_LIBRARY_PATH=str(_hot),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',XDG_CACHE_HOME=str(_tool/'cache'))
            kwargs['env']=env
            event={'original_argv':original_args,'actual_argv':args,'hot_library_receipt_sha256':'f316af33138de7f2fa69059ab4c8063ed5bb10c497e00db15babc1c410434f00','scope':'library IO relocation only; identical binary/weights/arguments'}
            with io.open(HERE/'openface_hot_invocations_v1.jsonl','a',encoding='utf-8')as log:log.write(json.dumps(event,ensure_ascii=False)+'\n');log.flush()
        super().__init__(args,*positional,**kwargs)
subprocess.Popen=_SameByteOpenFacePopen

args=sys.argv[1:]
if args and args[0]=='--extra-bridge':
    addon=json.loads(raw_bytes(HERE/'additional_bridge_source.json'));path=HERE/'extra_bridge.py';sys.argv=[str(path),*args[1:]]
    exec(compile(checked(path,addon['sha256']),str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
else:
    sys.argv=[e.__file__,*args];e.main()
