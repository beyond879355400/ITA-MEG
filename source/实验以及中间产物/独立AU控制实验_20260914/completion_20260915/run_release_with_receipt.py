"""Run project make release and bind the appended PDF to unchanged source inputs."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory():
    pending=['main.tex','chapters/body.tex','assets/frontmatter.pdf','Makefile','scripts/archive_pdf.py','scripts/verify.sh']
    found={}
    while pending:
        rel=pending.pop()
        if rel in found:continue
        p=ROOT/rel
        if not p.is_file():raise ValueError('Missing manuscript input: '+rel)
        found[rel]=sha(p)
        if p.suffix=='.tex':pending.extend(re.findall(r'figures/[^}\s]+',p.read_text()))
    return found
def pdf_stats():
    return {str(p.relative_to(ROOT)):(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)
            for p in (ROOT/'build').glob('王超悦-硕士学位论文-*.pdf')}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve()
    if HERE not in out.parents or out.exists():raise ValueError('New completion output directory required')
    out.mkdir(parents=True)
    before=inventory();old_pdfs=pdf_stats();started=time.time()
    env=dict(os.environ,PATH='/Library/TeX/texbin:'+os.environ.get('PATH',''))
    with (out/'make_release.log').open('x') as log:
        process=subprocess.run(['make','release'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    text=(out/'make_release.log').read_text();after=inventory();current=pdf_stats()
    archives_unchanged=all(current.get(k)==v for k,v in old_pdfs.items())
    names=re.findall(r'^FINAL_PDF=(.+)$',text,re.M)
    result={'status':'failed','started_utc':datetime.fromtimestamp(started,timezone.utc).isoformat(),
        'finished_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.time()-started,
        'returncode':process.returncode,'command':['make','release'],'inputs_before':before,'inputs_after':after,
        'source_inputs_unchanged':before==after,'prior_pdf_count':len(old_pdfs),'prior_pdf_stats_unchanged':archives_unchanged,
        'log_sha256':sha(out/'make_release.log'),'script_sha256':sha(__file__)}
    if process.returncode==0 and 'VERIFY_OK' in text and before==after and archives_unchanged and len(names)==1:
        pdf=ROOT/names[0]
        if pdf.is_file() and names[0] not in old_pdfs:
            result.update(status='compiled_and_auto_verified_pending_visual',pdf={'path':str(pdf),'relative_path':names[0],'sha256':sha(pdf)})
    with (out/'receipt.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':result['status'],'pdf':result.get('pdf'),'receipt':str(out/'receipt.json'),'seconds':result['seconds']},ensure_ascii=False),flush=True)
    if result['status']=='failed':print(text[-6000:]);raise SystemExit(1)
if __name__=='__main__':main()
