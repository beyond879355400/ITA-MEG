"""Process-local exact-size Python import reads on the experiment volume.

Returns unchanged source/bytecode bytes; never changes imported package files.
"""
import importlib._bootstrap_external as loaders
import os
import sys
import json
import hashlib
from pathlib import Path

_ORIGINAL=loaders.FileLoader.get_data
_SCOPE=str(Path(__file__).resolve().parent.parent)+os.sep

def _signature(value):return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns

def _exact_data(path):
    before=os.stat(path);remaining=before.st_size;parts=[]
    with open(path,'rb',buffering=0) as stream:
        if _signature(os.fstat(stream.fileno()))!=_signature(before):raise OSError('Import source changed before read')
        while remaining:
            block=stream.read(min(1024*1024,remaining))
            if not block:raise OSError('Premature EOF loading '+str(path))
            parts.append(block);remaining-=len(block)
        after_open=os.fstat(stream.fileno())
    if _signature(before)!=_signature(after_open) or _signature(before)!=_signature(os.stat(path)):
        raise OSError('Import source changed during read')
    return b''.join(parts)

def _get_data(self,path):
    if not os.path.abspath(path).startswith(_SCOPE):return _ORIGINAL(self,path)
    for attempt in range(2):
        try:return _exact_data(path)
        except OSError:
            # A first access may hydrate the file and change metadata. Read it
            # afresh once; never accept bytes spanning a changed file state.
            if attempt:raise

def install():
    hot=Path(__file__).resolve().parent/'hot_python_v1'
    receipt_path=hot/'layout_receipt_v2.json'
    data=_exact_data(str(receipt_path));receipt=json.loads(data)
    for record in receipt['records']:
        if hashlib.sha256(_exact_data(record['path'])).hexdigest()!=record['sha256']:raise OSError('Hot Python bytes differ')
    sys.path.insert(0,str(hot))
    loaders.FileLoader.get_data=_get_data
    return {'scope':_SCOPE,'kind':'process_local_exact_size_source_reads','package_files_modified':False,'hot_pure_python_layout_sha256':hashlib.sha256(data).hexdigest()}
