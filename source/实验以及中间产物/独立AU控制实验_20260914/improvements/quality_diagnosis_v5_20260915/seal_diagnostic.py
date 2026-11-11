"""Seal completed diagnostic outputs, preserving all previous experiment files."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    verification=HERE/'evaluation/source_init_202609151525/comparison/independent_verification.json'
    assert json.loads(verification.read_text())['status']=='PASS'
    result=HERE/'runs/source_init_202609151525/result.json'
    assert json.loads(result.read_text())['completed']==48
    output=HERE/'DELIVERY_SEAL.json'
    assert not output.exists()
    roots=[HERE/'runs/source_init_202609151525', HERE/'evaluation/source_init_202609151525',
           HERE/'rendered/source_init_202609151525_20260915T074714893787Z',HERE/'rendered/source_init_202609151525_compact']
    files={p for root in roots for p in root.rglob('*') if p.is_file() and not p.is_symlink()
           and not any('cache' in part.lower() for part in p.relative_to(root).parts[:-1])}
    files.update(p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.md','.py','.sh'))
    records=[{'path':str(p.relative_to(HERE)), 'sha256':sha(p), 'bytes':p.stat().st_size} for p in sorted(files)]
    with output.open('x') as f:
        json.dump({'status':'completed_development_diagnostic', 'utc':datetime.now(timezone.utc).isoformat(),
                   'records':records,'result_sha256':sha(result),'verification_sha256':sha(verification),
                   'test_data_used':False,'training_performed':False,'all48_retained':True,
                   'quality_threshold_set':False,'thesis_modified':False}, f, ensure_ascii=False, indent=2)
    print(json.dumps({'files':len(records),'path':str(output),'sha256':sha(output)}))
