"""Exact-pinned auxiliary downloads and a tiny private ftfy overlay; no models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

from stage_weights import HERE, RESERVE, gate, log, need, read, save_new, sha

STATUS = 'community_archive_original_origin_not_independently_verified'


def content_hash(path, kind):
    if kind == 'sha256':
        return sha(path)
    need(kind == 'git_blob_sha1', 'Unknown content hash')
    h = hashlib.sha1(('blob %d\0' % path.stat().st_size).encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def get_one(row, root):
    gate('cache_ready')
    path = (root / row['local_path']).resolve()
    need(root.resolve() in path.parents, 'Escaped component root')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.stat().st_size == row['size'] and content_hash(path, row['hash_kind']) == row['hash'],
             'Existing component mismatch; never overwrite')
        log({'event': 'component_reuse', 'path': str(path), 'bytes': row['size']})
        return {'path': str(path), 'bytes': row['size'], 'sha256': sha(path), 'download_bytes': 0}
    if row.get('local_reuse_source'):
        source=Path(row['local_reuse_source'])
        need(source.is_file() and source.stat().st_size==row['size'] and content_hash(source,row['hash_kind'])==row['hash'],
             'Declared local reuse differs; do not silently download a substitute')
        # APFS copy-on-write clone gives an independent inode. No write/chmod
        # touches the preserved source; clone failure is explicit, not fallback.
        copied=subprocess.run(['/bin/cp','-c',str(source),str(path)],capture_output=True,text=True,timeout=30)
        need(copied.returncode==0,'APFS clone failed: '+copied.stderr)
        need(content_hash(path,row['hash_kind'])==row['hash'],'Cloned asset changed')
        result={'path':str(path),'bytes':row['size'],'sha256':sha(path),'download_bytes':0,
                'reuse_source_path':str(source),'reuse_kind':'apfs_clone_exact_verified_bytes'}
        log({'event':'component_local_reuse',**result})
        return result
    partial = path.with_name(path.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    need(0 <= offset <= row['size'], 'Oversized partial')
    received, started = 0, time.monotonic()
    try:
        if offset < row['size']:
            need(shutil.disk_usage(HERE).free >= row['size']-offset+RESERVE, 'Component would violate8GiB reserve')
            req = urllib.request.Request(row['url'], headers={'Range': 'bytes=%d-' % offset} if offset else {})
            with urllib.request.urlopen(req, timeout=60) as response:
                need(response.status == (206 if offset else 200), 'Unexpected response status')
                if offset:
                    need(response.headers.get('Content-Range','').startswith('bytes %d-' % offset), 'Range not honored')
                log({'event':'component_start','path':str(path),'url':row['url'],'offset':offset,'size':row['size']})
                with partial.open('ab' if partial.exists() else 'xb') as out:
                    last = 0
                    while True:
                        need(shutil.disk_usage(HERE).free >= RESERVE, 'Disk reserve reached; partial retained')
                        chunk = response.read(4*1024*1024)
                        if not chunk:
                            break
                        received += len(chunk)
                        need(offset+received <= row['size'], 'Response exceeds approved size')
                        need(shutil.disk_usage(HERE).free >= RESERVE+len(chunk), 'Upcoming component chunk would violate8GiB reserve')
                        out.write(chunk)
                        if received-last >= 64*1024*1024:
                            out.flush(); last=received
                            log({'event':'component_progress','path':str(path),'partial_bytes':offset+received,'network_bytes_this_attempt':received})
        need(partial.stat().st_size == row['size'], 'Truncated component')
        need(content_hash(partial,row['hash_kind']) == row['hash'], 'Pinned component hash mismatch')
        need(not path.exists(), 'Target appeared concurrently')
        partial.rename(path)
        result = {'path':str(path),'bytes':row['size'],'sha256':sha(path),'download_bytes':received}
        log({'event':'component_verified',**result,'seconds':time.monotonic()-started})
        return result
    except BaseException as error:
        log({'event':'component_failed','path':str(path),'network_bytes_this_attempt':received,'error':str(error)})
        raise


def auxiliary():
    gate('cache_ready')
    meta = read(HERE/'aux_components.json')
    need(meta['base_source_status'] == STATUS and meta['unet_downloads'] is False, 'Unexpected base source contract')
    rows = meta['components']
    need(not any(row['path'].startswith('unet/') or '/unet/' in row['path'] for row in rows), 'No auxiliary UNet permitted')
    root = HERE/'weights/auxiliary'; root.mkdir(parents=True, exist_ok=True)
    need(not list((HERE/'downloads').glob('swiftedit*.partial')), 'Finish official multipart transfer first')
    remaining = sum(row['size'] for row in rows if not (root/row['local_path']).exists())
    need(shutil.disk_usage(HERE).free >= remaining+RESERVE, 'Auxiliary total would violate8GiB reserve')
    ordered=sorted(rows,key=lambda row:0 if row.get('local_reuse_source') else 1)
    receipt = {'base_source_status':STATUS,'manifest_sha256':sha(HERE/'aux_components.json'),
               'files':[get_one(row,root) for row in ordered]}
    save_new(root/'download_receipt.json',receipt)
    print(json.dumps({'status':'complete','files':len(rows),'bytes':sum(row['size'] for row in rows)}))


def ftfy():
    gate('cache_ready')
    row=read(HERE/'ftfy_wheel.json')
    need(row['version']=='6.3.1' and row['size']==44821 and row['url'].startswith('https://files.pythonhosted.org/'), 'Unexpected ftfy pin')
    root=HERE/'downloads'; root.mkdir(exist_ok=True)
    wheel=get_one({'local_path':row['filename'],'size':row['size'],'hash_kind':'sha256','hash':row['sha256'],'url':row['url']},root)
    # This wheel is pure Python. zipimport is an isolated runtime overlay, not
    # an installation into the existing environment; no extracted copy needed.
    with zipfile.ZipFile(wheel['path']) as archive:
        need('Root-Is-Purelib: true' in archive.read('ftfy-6.3.1.dist-info/WHEEL').decode(), 'Not a pure-Python wheel')
        need(not any('.data/' in name for name in archive.namelist()), 'Wheel requires installer relocation')
    cmd=[sys.executable,'-B','-c','import sys;sys.path.insert(0,sys.argv[1]);import ftfy;print(ftfy.__version__);assert ftfy.fix_text("test")=="test"',wheel['path']]
    result=subprocess.run(cmd, text=True, capture_output=True, timeout=30)
    save_new(HERE/'logs/ftfy_zipimport.json',{'command':cmd,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'wheel':wheel,'old_environment_modified':False})
    need(result.returncode==0,'ftfy private wheel import failed; see log')
    print(result.stdout)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['ftfy','auxiliary'])
    args=parser.parse_args()
    ftfy() if args.action=='ftfy' else auxiliary()
