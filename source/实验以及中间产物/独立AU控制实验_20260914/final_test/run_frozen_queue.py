"""Execute the already frozen local test, without model or protocol changes."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PY = '/Users/mayfair/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python'
PROTOCOL = HERE / 'frozen_protocol.json'
PSHA = '41d0f77e1307fd7da22e212a2485101fa6ab9662f11b02ad63fdb94d0bceda55'
RAW = EXP / 'data_prepare/final_test_extractions/raw_test_frames_202609150730/raw_frame_manifest.json'
RSHA = '4dbe2efefe37ca4101ecd83b6fb37d8d4cdc984de7e0416eccb04774681f01a4'
CACHE = HERE / 'source_caches/source_202609150731/source_cache_manifest.json'
CSHA = '2ea270ffe98a4ad07acc1fe8b6d7f48bb5dd905f1d4e1d70bce11035844fb0e5'
OWN = ['public_baseline', 'bidirectional', 'condition_reads_image', 'image_reads_condition', 'mlp']
LOCK = threading.Lock()

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def save(p, v):
    with Path(p).open('x') as f:
        json.dump(v, f, ensure_ascii=False, indent=2)

def event(kind, **kw):
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'event': kind, **kw}
    with LOCK:
        with (RUN / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

def command(name, entry, args, prime=False, allowed=(0,)):
    cmd = [PY, '-B']
    if prime:
        cmd += [str(HERE / 'run_with_fontcache.py'), str(RUN / ('fonts_' + name))]
    cmd += [str(entry), *map(str, args)]
    event('start', stage=name, command=cmd)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               NO_ALBUMENTATIONS_UPDATE='1', PYTORCH_ENABLE_MPS_FALLBACK='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (RUN / (name + '.log')).open('x') as f:
        result = subprocess.run(cmd, cwd=EXP, env=env, stdout=f, stderr=subprocess.STDOUT)
    event('exit', stage=name, returncode=result.returncode)
    if result.returncode not in allowed:
        raise RuntimeError(f'{name} failed with {result.returncode}; see preserved log')

def check_index(path, methods):
    doc = read(path)
    assert doc['status'] in ('complete', 'completed_with_failures')
    assert doc.get('usable_for_evaluation', True)
    assert doc['protocol_sha256'] == PSHA
    assert set(doc['method_names']) == set(methods)
    assert len(doc['records']) == len(methods) * 96
    event('generation_terminal', methods=methods, index=str(path), sha256=sha(path),
          status=doc['status'], physical_executions=doc['physical_executions'])
    return path

def external(method):
    dest = HERE / 'external_runs' / (method + '_' + RUN.name)
    command(method, HERE / 'external/run_external_final.py', [
        'execute', '--protocol', PROTOCOL, '--protocol-sha256', PSHA,
        '--raw-manifest', RAW, '--raw-manifest-sha256', RSHA,
        '--method', method, '--output', dest], allowed=(0, 1))
    return check_index(dest / 'generation_index.json', [method])

def references():
    dest = RUN / 'references'
    command('references', HERE / 'evaluation_references.py', [
        '--protocol', PROTOCOL, '--raw-manifest', RAW, '--raw-sha', RSHA,
        '--source-cache', CACHE, '--source-cache-sha', CSHA, '--output', dest], prime=True)
    ref = dest / 'references.json'
    event('references_terminal', path=str(ref), sha256=sha(ref),
          source_valid_cases=read(ref)['source_valid_cases'], target_valid_cases=read(ref)['target_valid_cases'])
    return ref

def main():
    for path, expected in [(PROTOCOL, PSHA), (RAW, RSHA), (CACHE, CSHA)]:
        assert sha(path) == expected
    assert read(CACHE)['status'] == 'complete'
    RUN.mkdir(parents=True, exist_ok=False)
    save(RUN / 'queue.json', {'protocol': str(PROTOCOL), 'protocol_sha256': PSHA,
         'raw_manifest': str(RAW), 'raw_sha256': RSHA, 'source_cache': str(CACHE),
         'source_cache_sha256': CSHA, 'queue_sha256': sha(__file__),
         'font_wrapper_sha256': sha(HERE / 'run_with_fontcache.py'), 'test_data_used': True,
         'no_training': True, 'no_tuning': True, 'no_thesis_changes': True})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        lp = pool.submit(external, 'liveportrait')
        refs = pool.submit(references)
        # GPU consumers run serially. CPU LP/reference preparation can overlap.
        before = set((HERE / 'generation_runs').glob('ours_*/generation_index.json'))
        args = ['--protocol', PROTOCOL, '--protocol-sha', PSHA, '--raw-manifest', RAW,
                '--raw-manifest-sha', RSHA, '--source-cache', CACHE, '--source-cache-sha', CSHA]
        for m in OWN:
            args += ['--method', m]
        command('ours', HERE / 'generate_ours.py', args)
        added = set((HERE / 'generation_runs').glob('ours_*/generation_index.json')) - before
        assert len(added) == 1
        indexes = [check_index(added.pop(), OWN)]
        indexes.append(external('magicface'))
        indexes.append(external('faceadapter'))
        indexes.append(lp.result())
        ref = refs.result()
    job = RUN / 'measurements'
    args = ['prepare', '--protocol', PROTOCOL, '--raw-manifest', RAW, '--raw-sha', RSHA,
            '--references', ref, '--references-sha', sha(ref), '--output', job]
    for index in indexes:
        args += ['--generation-index', index, '--generation-sha', sha(index)]
    command('prepare_measurements', HERE / 'evaluation_measurements.py', args)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(command, m, HERE / 'evaluation_measurements.py',
                               [m, '--job', job], m != 'openface')
                   for m in ('openface', 'facenet', 'buffalo')]
        for f in futures:
            f.result()
    command('export', HERE / 'evaluation_export.py', ['--job', job])
    statistics = job / 'statistics/paired_subject_results.json'
    command('statistics', EXP / 'statistics/paired_subject_statistics.py', [
        '--input', job / 'statistics/paired_subject_input.json', '--output', statistics])
    save(RUN / 'complete.json', {'status': 'computed_pending_independent_review',
        'protocol_sha256': PSHA, 'generation_indexes': [{'path': str(p), 'sha256': sha(p)} for p in indexes],
        'references': {'path': str(ref), 'sha256': sha(ref)},
        'statistics': {'path': str(statistics), 'sha256': sha(statistics)}})
    event('queue_complete', report=str(RUN / 'complete.json'))

if __name__ == '__main__':
    if sys.argv[1:] != ['--execute']:
        raise SystemExit('Explicit --execute required; no action taken')
    RUN = HERE / 'execution_202609150734'
    try:
        main()
    except BaseException as error:
        if RUN.exists():
            event('queue_failed', error_type=type(error).__name__, error=str(error))
        raise
