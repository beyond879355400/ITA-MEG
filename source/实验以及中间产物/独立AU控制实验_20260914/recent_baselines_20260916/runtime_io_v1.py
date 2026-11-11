"""Exact-size local hashing: no read beyond declared EOF, no file mutation."""
import hashlib,os,stat
from pathlib import Path


def signature(value):
    return {'device':value.st_dev,'inode':value.st_ino,'size':value.st_size,'mtime_ns':value.st_mtime_ns}


def hash_exact(path,expected_size,expected_sha256=None):
    """Use a frozen expected_size/pin for assets; None pin only computes hash.

    Stops exactly at expected_size. Premature EOF, changed inode/size/mtime or
    a mismatched pin fail closed. This bounds bytes, not filesystem latency.
    """
    path=Path(path);before=path.stat()
    if not isinstance(expected_size,int) or expected_size<0 or before.st_size!=expected_size or not stat.S_ISREG(before.st_mode):
        raise ValueError('Unexpected file type/size: '+str(path))
    digest=hashlib.sha256();count=0
    with path.open('rb',buffering=0) as stream:
        if signature(os.fstat(stream.fileno()))!=signature(before):raise ValueError('File changed before read')
        while count<expected_size:
            block=stream.read(min(4*1024*1024,expected_size-count))
            if not block:raise ValueError('Premature EOF at%d/%d: %s'%(count,expected_size,path))
            count+=len(block);digest.update(block)
        opened_after=os.fstat(stream.fileno())
    after=path.stat()
    if signature(before)!=signature(after) or signature(before)!=signature(opened_after):raise ValueError('File changed during read')
    value=digest.hexdigest()
    if expected_sha256 is not None and value!=expected_sha256:raise ValueError('Pinned SHA mismatch: '+str(path))
    return {'path':str(path),'bytes':count,'sha256':value,'stat':signature(after),'flags_after':getattr(after,'st_flags',None),
            'read_past_expected_size':False,'pin_verified':expected_sha256 is not None}
