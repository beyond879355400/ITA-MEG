"""Read exactly the pinned/stat file extent; no extra potentially stalled EOF read."""
import hashlib
import json
import os
from pathlib import Path
import stat


def identity(st):return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns


def stream_exact(path,expected_size=None):
    path=Path(path);before=path.stat()
    if not stat.S_ISREG(before.st_mode):raise ValueError('Expected regular file')
    if expected_size is not None and before.st_size!=expected_size:raise ValueError('Pinned size mismatch')
    with path.open('rb',buffering=0)as f:
        if identity(os.fstat(f.fileno()))!=identity(before):raise ValueError('File changed before open')
        remaining=before.st_size
        while remaining:
            b=f.read(min(1048576,remaining))
            if not b or len(b)>remaining:raise ValueError('Empty/short/unexpected extent read')
            remaining-=len(b);yield b
        if identity(os.fstat(f.fileno()))!=identity(before):raise ValueError('File changed during read')
    if identity(path.stat())!=identity(before):raise ValueError('Path changed after read')


def read_bytes(path,expected_sha=None,expected_size=None,nonempty=True):
    data=b''.join(stream_exact(path,expected_size))
    if nonempty and not data:raise ValueError('Unexpected empty file')
    if expected_sha is not None and hashlib.sha256(data).hexdigest()!=expected_sha:raise ValueError('Pinned SHA mismatch')
    return data


def sha(path,expected_size=None):
    h=hashlib.sha256()
    for b in stream_exact(path,expected_size):h.update(b)
    return h.hexdigest()


def read_json(path,expected_sha=None):return json.loads(read_bytes(path,expected_sha))


def ref(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
