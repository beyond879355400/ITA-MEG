#!/usr/bin/env python3
"""Private identical-byte copies of cold pure Python import packages."""
import base64,concurrent.futures,csv,hashlib,io,json,os,subprocess,sys,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SITE=ROOT/'env_overlay_v2/lib/python3.10/site-packages'
DEST=ROOT/'hot_python_v1'

def read_exact(path):
    before=path.stat()
    with path.open('rb',buffering=0) as f:data=f.read(before.st_size)
    after=path.stat()
    if len(data)!=before.st_size:raise OSError('Short Python package read')
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise OSError('Package hydrated during read; retry exact bytes')
    return data

def worker(relative,pin):
    source=SITE/relative;target=DEST/relative
    for i in range(2):
        try:data=read_exact(source);break
        except OSError:
            if i:raise
    if hashlib.sha256(data).hexdigest()!=pin:raise OSError('Wheel RECORD hash mismatch: '+relative)
    target.parent.mkdir(exist_ok=True,parents=True)
    if target.exists():
        if hashlib.sha256(read_exact(target)).hexdigest()!=pin:raise OSError('Existing hot copy differs')
    else:
        with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    print(json.dumps({'relative':relative,'source':str(source),'path':str(target),'sha256':pin,'bytes':len(data)}))

def main():
    if (DEST/'layout_receipt.json').exists():raise RuntimeError('Immutable complete layout')
    records=[];wheels=[];wheel_dir=ROOT/'recovery_wheels';wheel_dir.mkdir(exist_ok=True)
    for pkg,version in [('rich','13.7.1'),('tyro','0.8.5')]:
        meta=json.load(urllib.request.urlopen('https://pypi.org/pypi/'+pkg+'/'+version+'/json',timeout=20))
        item=next(x for x in meta['urls'] if x['filename'].endswith('py3-none-any.whl'))
        wheel=urllib.request.urlopen(item['url'],timeout=20).read()
        if hashlib.sha256(wheel).hexdigest()!=item['digests']['sha256'] or len(wheel)!=item['size']:raise RuntimeError('PyPI wheel hash mismatch')
        saved=wheel_dir/item['filename']
        if not saved.exists():saved.write_bytes(wheel)
        wheels.append({'version':version,'url':item['url'],'sha256':item['digests']['sha256'],'bytes':len(wheel)})
        with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
            record_name=next(n for n in archive.namelist() if n.endswith('.dist-info/RECORD'))
            for row in csv.reader(io.StringIO(archive.read(record_name).decode())):
                if not row[0].startswith(pkg+'/') or not row[1].startswith('sha256='):continue
                encoded=row[1].split('=',1)[1];pin=base64.urlsafe_b64decode(encoded+'='*((-len(encoded))%4)).hex()
                data=archive.read(row[0]);target=DEST/row[0]
                if hashlib.sha256(data).hexdigest()!=pin:raise RuntimeError('Wheel RECORD mismatch')
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if hashlib.sha256(read_exact(target)).hexdigest()!=pin:raise RuntimeError('Existing copy differs')
                else:
                    with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
                records.append({'relative':row[0],'path':str(target),'sha256':pin,'bytes':len(data)})
    receipt={'kind':'same_version_verified_wheel_pure_python_hot_copy','packages':['rich13.7.1','tyro0.8.5'],'wheels':wheels,'records':records,'original_package_bytes_unchanged':True}
    with (DEST/'layout_receipt.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'complete','files':len(records),'bytes':sum(x['bytes'] for x in records)}),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:worker(sys.argv[2],sys.argv[3])
    else:main()
