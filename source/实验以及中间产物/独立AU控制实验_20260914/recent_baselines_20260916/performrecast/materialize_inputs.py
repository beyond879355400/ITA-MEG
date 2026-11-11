#!/usr/bin/env python3
"""Copy predeclared small input assets byte-for-byte; no model or image rewrite."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT.parent/'preservation/bundle_v2_20260916'
HOT=ROOT/'hot_inputs_v1'

def exact(path):
    path=Path(path);before=path.stat()
    with path.open('rb') as f:
        data=f.read(before.st_size)
    after=path.stat()
    if len(data)!=before.st_size or (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise RuntimeError('Incomplete or changing source: '+str(path)+' read='+str(len(data))+' expected='+str(before.st_size)+' flags='+str(after.st_flags))
    return data

def bounded(path):
    for attempt in range(2):
        try:return subprocess.check_output([sys.executable,'-B',str(Path(__file__)),'--read-exact',str(path)],timeout=10)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
            print(json.dumps({'event':'bounded_read_failed','path':str(path),'attempt':attempt+1}),file=sys.stderr,flush=True)
            if attempt:raise
            time.sleep(.5)

def read_json(path):return json.loads(bounded(path))

def main():
    if (HOT/'layout_receipt.json').exists():raise RuntimeError('Completed hot layout is immutable')
    manifests=[BUNDLE/'inputs/postfreeze32.json']
    expected=['fa88ca09eb8253ce61ba046cb8d9b4c4affaef6cc421893cedb487c09522413b']
    assets={};bindings=[]
    for path,digest in zip(manifests,expected):
        data=bounded(path)
        if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('Frozen manifest mismatch')
        manifest=json.loads(data);bindings.append({'path':str(path),'sha256':digest})
        for case in manifest['cases']:
            for role in ('source','target'):
                original=str((BUNDLE/case[role+'_path']).resolve());sha=case[role+'_sha256']
                if original in assets and assets[original]!=sha:raise RuntimeError('Conflicting asset SHA')
                assets[original]=sha
    smoke=read_json(ROOT/'inputs_smoke_v1/manifest.json')
    for item in smoke['inputs'].values():assets[item['clip']]=item['clip_sha256']
    baseline=read_json(ROOT/'runs/smoke_cpu_v1/receipt.json')
    selected=baseline['selected_crop_png']
    assets[selected]=next(x['sha256'] for x in baseline['outputs'] if x['path']==selected)
    HOT.mkdir(exist_ok=True);records=[]
    for original,digest in sorted(assets.items()):
        data=bounded(original)
        if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('Original bytes differ: '+original)
        target=HOT/(digest+Path(original).suffix)
        if not target.exists():
            with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        reread=bounded(target)
        if hashlib.sha256(reread).hexdigest()!=digest:raise RuntimeError('Hot copy mismatch')
        records.append({'original_path':original,'path':str(target),'sha256':digest,'bytes':len(data),'nlink':target.stat().st_nlink,'flags':target.stat().st_flags})
    receipt={'kind':'performrecast_identical_byte_input_storage_v1','status':'complete','manifests':bindings,'records':records,'original_unchanged':True,'images_reencoded':False}
    with (HOT/'layout_receipt.json').open('x') as f:json.dump(receipt,f,indent=2,ensure_ascii=False)
    print(json.dumps({'status':'complete','records':len(records),'total_bytes':sum(x['bytes'] for x in records),'receipt':str(HOT/'layout_receipt.json'),'sha256':hashlib.sha256(exact(HOT/'layout_receipt.json')).hexdigest()}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--read-exact',type=Path);args=parser.parse_args()
    if args.read_exact:sys.stdout.buffer.write(exact(args.read_exact))
    else:main()
