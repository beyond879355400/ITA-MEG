"""One-time manifest sealing of this completed run. Never modifies its outputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
EXP = FINAL.parent
RUN = FINAL / 'execution_202609150734'
TEXT = {'.json', '.md', '.csv', '.tsv', '.txt', '.log', '.py', '.sh'}

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def eligible(p, base):
    return p.is_file() and not p.is_symlink() and not any(
        x.startswith('.') or 'cache' in x.lower() or x.startswith('fonts_')
        for x in p.relative_to(base).parts[:-1])

if __name__ == '__main__':
    assert not (HERE / 'SHA256SUMS').exists() and not (HERE / 'SEAL.json').exists()
    complete = read(RUN / 'complete.json')
    assert complete['status'] == 'computed_pending_independent_review'
    assert read(RUN / 'measurements/independent_measurement_verification.json')['status'] == 'PASS'
    assert read(FINAL / 'reproduction_runs/statistics_202609151446/reproduction_complete.json')['byte_identical_to_original']
    for seed in (101, 202, 303):
        assert (RUN / f'visual_review/review_seed{seed}.md').is_file()
    indexes = [Path(r['path']) for r in complete['generation_indexes']]
    for ref in complete['generation_indexes'] + [complete['references'], complete['statistics']]:
        assert sha(ref['path']) == ref['sha256']
    roots = [RUN, *[p.parent for p in indexes],
             FINAL / 'source_caches/source_202609150731',
             EXP / 'data_prepare/final_test_extractions/raw_test_frames_202609150730',
             FINAL / 'reproduction_runs/statistics_202609151446']
    files = {p for base in roots for p in base.rglob('*') if eligible(p, base)}
    for base in (FINAL, HERE, FINAL / 'external', FINAL / 'review_tools'):
        files.update(p for p in base.iterdir() if p.is_file() and not p.is_symlink()
                     and p.suffix in TEXT and p.name not in {'SEAL.json', 'SHA256SUMS'})
    protocol = read(FINAL / 'frozen_protocol.json')
    for model in protocol['models']:
        for ref in model['artifacts']:
            p = Path(ref['path'])
            if not p.is_absolute():
                p = EXP.parent.parent / p
            if p.is_relative_to(EXP / 'runs'):
                assert sha(p) == ref['sha256']
                files.add(p)
    records = [{'path': str(p.relative_to(EXP)), 'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in sorted(files)]
    metadata = [r for r in records if Path(r['path']).suffix in TEXT]
    with (HERE / 'SHA256SUMS').open('x') as f:
        f.writelines(r['sha256'] + '  ' + r['path'] + '\n' for r in metadata)
    seal = {'status': 'sealed_completed_test_for_diagnosis', 'sealed_utc': datetime.now(timezone.utc).isoformat(),
            'protocol_sha256': sha(FINAL / 'frozen_protocol.json'),
            'statistics_sha256': complete['statistics']['sha256'],
            'metadata_manifest_sha256': sha(HERE / 'SHA256SUMS'),
            'records': records, 'files': len(records), 'metadata_files': len(metadata),
            'image_files': sum(Path(r['path']).suffix == '.png' for r in records),
            'quality_acceptance_passed': False, 'thesis_modified': False,
            'scope': 'Current run outputs, references, raw frames, selected checkpoints and local entry/review/delivery metadata. Public prerequisite assets remain in frozen protocol; cache directories excluded.'}
    with (HERE / 'SEAL.json').open('x') as f:
        json.dump(seal, f, ensure_ascii=False, indent=2)
    print(json.dumps({'status': seal['status'], 'files': len(records), 'metadata_files': len(metadata),
                      'seal_sha256': sha(HERE / 'SEAL.json'), 'manifest_sha256': seal['metadata_manifest_sha256']}))
