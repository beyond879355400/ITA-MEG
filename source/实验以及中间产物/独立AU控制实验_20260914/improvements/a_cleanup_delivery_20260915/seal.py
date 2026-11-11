"""Append-only integrity snapshot of completed A-cleanup development rounds."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERSIONS = [
    ('a_texture_schedule_v9_20260915', 'late_au_202609151925', 48,
     'verification/late_au_202609151925_independent.json'),
    ('a_terminal_cleanup_v10_20260915', 'terminal_202609151940', 48,
     'verification_tools/reports/terminal_202609151940_independent.json'),
    ('a_local_cleanup_v11_20260915', 'local_202609152016', 32,
     'verification_tools/reports/local_202609152016_independent.json'),
]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()

def main():
    files, rounds = set(), []
    for name, run, count, verification in VERSIONS:
        base = ROOT / name
        result = json.loads((base / 'runs' / run / 'result.json').read_text())
        audit = json.loads((base / verification).read_text())
        assert result['status'] == 'complete' and result['planned'] == result['completed'] == count
        assert audit['status'] == 'verified_without_model_rerun' and audit['test_used'] is False
        rounds.append({'version': name, 'run': run, 'new_images': count,
                       'verification_sha256': sha(base / verification)})
        for p in base.rglob('*'):
            if p.is_file() and not p.is_symlink() and not any(
                'synthetic' in x.lower() or 'cache' in x.lower() or x.startswith('.')
                for x in p.relative_to(base).parts):
                files.add(p)
    files.update(p for p in HERE.iterdir() if p.is_file() and not p.name.startswith('SEAL'))
    records = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(files)]
    now = datetime.now(timezone.utc)
    target = HERE / ('SEAL_' + now.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    doc = {'status': 'completed_development_rounds_pending_user_visual_feedback',
           'utc': now.isoformat(), 'rounds': rounds, 'new_images': sum(r['new_images'] for r in rounds),
           'records': records, 'formal_test_passed': False, 'user_visual_approval_received': False,
           'thesis_modified': False}
    with target.open('x') as stream:
        json.dump(doc, stream, ensure_ascii=False, indent=2, allow_nan=False)
    assert all(sha(ROOT / r['path']) == r['sha256'] for r in records)
    print(json.dumps({'status': doc['status'], 'verified_files': len(records), 'seal': str(target),
                      'sha256': sha(target)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
