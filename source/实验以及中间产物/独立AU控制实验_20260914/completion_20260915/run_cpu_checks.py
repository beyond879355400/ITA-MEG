"""Append-only CPU unit regression; no model generation or experiment metrics."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PY='/Users/mayfair/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python'
GROUPS=[
 ('generation',['test_generation.py','test_dev_completion.py']),
 ('external',['test_external.py']),
 ('evaluation',['test_evaluation.py']),
 ('evaluation/dev19_tools',['test_dev19_evaluate.py']),
 ('protocol',['test_extract_frames.py','test_select_postfreeze_cases.py']),
 ('review_tools',['test_dev19_verification.py','test_verify_completed.py']),
 ('visual_review_tools',['test_render_contacts.py']),
 ('manuscript_drafts/publication_visual',['test_prepare_candidate.py']),
 ('delivery',['test_review_driver.py','test_reproduce_driver.py']),
]
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=HERE/'unit_check_reports'/stamp
out.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
rows=[]
for name, tests in GROUPS:
    started=time.monotonic()
    p=subprocess.run([PY,'-B','-m','unittest','-v',*tests],cwd=HERE/name,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    label=name.replace('/','_')
    with (out/(label+'.log')).open('x') as f:f.write(p.stdout)
    counts=re.findall(r'Ran (\d+) tests?',p.stdout)
    row={'group':name,'returncode':p.returncode,'tests':int(counts[-1]) if counts else None,'seconds':time.monotonic()-started,'log':label+'.log'}
    rows.append(row);print(json.dumps(row),flush=True)
    if p.returncode: break
result={'status':'PASS' if len(rows)==len(GROUPS) and all(r['returncode']==0 for r in rows) else 'FAIL','groups':rows,'total_tests':sum(r['tests'] or 0 for r in rows),'scope':'CPU/synthetic unit tests only; not new experiment results','output':str(out)}
with (out/'summary.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['status']=='PASS' else 1)
