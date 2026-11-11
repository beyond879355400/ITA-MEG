"""Copy only installed-distribution RECORD members, never cloud duplicate junk."""
import base64
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import prepare_overlay as p

source=p.OLD/'insightface-0.7.3.dist-info/RECORD'
rows=list(csv.reader(source.open()))
records=[]
try:
    for relative,digest,size in rows:
        parts=Path(relative).parts
        if not parts or parts[0]not in ['insightface','insightface-0.7.3.dist-info']or '__pycache__'in parts:continue
        p.need('..'not in parts,'Unexpected source traversal')
        src=p.OLD/relative;dest=p.SITE/relative
        expected=base64.urlsafe_b64decode(digest.split('=',1)[1]+'='*((-len(digest.split('=',1)[1]))%4)).hex()if digest else None
        p.need(not size or src.stat().st_size==int(size),'Source size differs from installed RECORD')
        if dest.exists():
            p.need(expected and p.sha(dest)==expected,'Existing destination not bound to installed record')
            record={'source':str(src),'path':str(dest),'bytes':dest.stat().st_size,'sha256':expected,'reuse_verified':True}
        else:
            r=subprocess.run([sys.executable,'-B',str(p.HERE/'prepare_overlay.py'),'--copy-one',str(src),str(dest)],
                             capture_output=True,text=True,timeout=20)
            p.need(r.returncode==0,'Source materialization failed: '+r.stderr[-2000:]);record=json.loads(r.stdout)
            p.need(expected is None or record['sha256']==expected,'Copied source differs from installed RECORD')
        record['installed_record_sha256']=expected;records.append(record)
    p.save(p.HERE/'insightface_recorded_copy_receipt.json',{'status':'complete','installed_record':str(source),
          'installed_record_sha256':p.sha(source),'files':records,'only_recorded_runtime_members':True,
          'excluded':'untracked duplicate files, pycache, unrelated bin/include/data paths; old source preserved'})
    print(json.dumps({'status':'complete','copied_recorded_files':len(records),'overlay_bytes':p.total(p.SITE)}))
except BaseException as error:
    p.save(p.HERE/'insightface_recorded_copy_failure.json',{'status':'failed','error':str(error),'completed':records})
    raise
