"""Run the already frozen completion protocol; never changes parameters or cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
THESIS = HERE.parents[2]
PY = '/Users/mayfair/Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python'
FREEZE = HERE / 'frozen_protocol.json'
FSHA = '498298977f3589248084e37d053ec7cd8f37f71b35e914849465081a3df9af33'
RAW = HERE / 'data/raw_202609152329/raw_frame_manifest.json'
RSHA = 'f3139d03d7b7808aea47cd46aed2f0107e77bf443f69f476ed845f8a1c1875ec'
SOURCE_RAW = HERE / 'data/raw_202609152329/raw_source_only_manifest.json'
SSHA = '60c711e03d8f4d4b3ef9cd9ee50a83eb486fc761379678731572952baf34289d'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8388608), b''): h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    with path.open('x') as stream: json.dump(value, stream, ensure_ascii=False, indent=2)

def require(ok, message):
    if not ok: raise ValueError(message)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for path, expected in ((FREEZE, FSHA), (RAW, RSHA), (SOURCE_RAW, SSHA)):
        require(sha(path) == expected, 'Frozen input changed: ' + str(path))
    require(read(FREEZE)['status'] == 'frozen' and read(RAW)['status'] == 'raw_frames_extracted', 'Complete frozen data required')
    dev = HERE / 'dev19_runs/dev19_202609152322/result.json'
    require(dev.is_file() and read(dev)['status'] in ('complete', 'completed_with_failures'), 'Prior MPS dev19 grid must finish before this queue')
    if not a.execute:
        print(json.dumps({'status': 'ready_no_models', 'new_ours': 480, 'magicface': 96, 'faceadapter': 96,
                          'liveportrait': 'existing concurrent frozen32 run', 'training': False})); return
    out = a.output.resolve()
    require(HERE in out.parents and not out.exists(), 'New completion orchestration directory required')
    out.mkdir(parents=True)
    write(out / 'configuration.json', {'freeze_sha256': FSHA, 'raw_sha256': RSHA, 'source_raw_sha256': SSHA,
        'script_sha256': sha(__file__), 'start_utc': datetime.now(timezone.utc).isoformat(), 'training_performed': False})
    events = []
    def run(label, script, args, allow_failure=False):
        command = [PY, '-B', str(HERE / script), *map(str, args)]
        started = time.time()
        print(json.dumps({'phase': label, 'status': 'started', 'utc': datetime.now(timezone.utc).isoformat()}), flush=True)
        with (out / (label + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=THESIS, stdout=log, stderr=subprocess.STDOUT, check=False)
        event = {'phase': label, 'returncode': result.returncode, 'seconds': time.time() - started, 'command': command}
        events.append(event); write(out / (label + '.receipt.json'), event)
        print(json.dumps(event), flush=True)
        require(allow_failure or result.returncode == 0, 'Phase failed: ' + label)
    fixed = ['--freeze', FREEZE, '--freeze-sha', FSHA]
    cache_dir = HERE / 'source_caches/selected_v11_v1'
    run('source_cache', 'generation/prepare_cache.py', [*fixed, '--raw-manifest', SOURCE_RAW, '--raw-manifest-sha', SSHA,
        '--output', cache_dir, '--execute'])
    cache = cache_dir / 'manifest.json'; csha = sha(cache)
    require(read(cache)['status'] in ('complete', 'completed_with_failures'), 'Source cache not terminal')
    source_dir = HERE / 'evaluation/source_v1'
    run('source_openface', 'evaluation/evaluate.py', ['source-openface', *fixed, '--source-cache', cache, '--source-cache-sha', csha,
        '--output', source_dir, '--execute'])
    source = source_dir / 'results.json'; source_sha = sha(source)
    reference_dir = HERE / 'evaluation/references_v1'
    ours_dir = HERE / 'generation_runs/selected_v11_32_v1'
    with ThreadPoolExecutor(max_workers=2) as pool:
        reference = pool.submit(run, 'references', 'evaluation/evaluate.py', ['references', *fixed, '--source-cache', cache,
            '--source-cache-sha', csha, '--raw-manifest', RAW, '--raw-sha', RSHA, '--output', reference_dir, '--execute'])
        ours = pool.submit(run, 'ours', 'generation/run_grid.py', [*fixed, '--source-cache', cache, '--source-cache-sha', csha,
            '--source-measurements', source, '--source-measurements-sha', source_sha, '--output', ours_dir, '--execute'])
        ours.result(); reference.result()
    ours_index = ours_dir / 'result.json'
    require(read(ours_index)['status'] in ('complete', 'completed_with_failures') and len(read(ours_index)['records']) == 480, 'Our complete grid absent')
    external_indexes = []
    for method in ('magicface', 'faceadapter'):
        target = HERE / 'external_runs' / (method + '_postfreeze_v1')
        run(method, 'external/run_external.py', [*fixed, '--raw-manifest', RAW, '--raw-sha', RSHA, '--method', method,
            '--output', target, '--execute'], allow_failure=True)
        index = target / 'generation_index.json'
        require(index.is_file() and read(index)['status'] in ('complete', 'completed_with_failures')
                and read(index).get('usable_for_evaluation') is True, method + ' has non-evaluable run-level failure')
        external_indexes.append(index)
    lp = HERE / 'external_runs/liveportrait_postfreeze_v1/generation_index.json'
    require(lp.is_file() and read(lp)['status'] in ('complete', 'completed_with_failures')
            and read(lp).get('usable_for_evaluation') is True, 'Existing frozen LP32 run must be terminal/evaluable')
    external_indexes.append(lp)
    references = reference_dir / 'references.json'
    job = HERE / 'evaluation/all_methods_v1'
    args = ['prepare', *fixed, '--references', references, '--references-sha', sha(references), '--output', job, '--execute']
    for index in [ours_index, *external_indexes]: args += ['--generation-index', index, '--generation-sha', sha(index)]
    run('evaluation_prepare', 'evaluation/evaluate.py', args)
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(run, kind, 'evaluation/evaluate.py', [kind, '--job', job, '--execute']) for kind in ('openface', 'facenet', 'buffalo')]
        for task in tasks: task.result()
    run('statistics', 'evaluation/evaluate.py', ['export', '--job', job, '--execute'])
    stats = job / 'statistics/paired_subject_results.json'
    write(out / 'result.json', {'status': 'computation_complete_pending_independent_and_visual_review',
        'events': events, 'statistics': {'path': str(stats), 'sha256': sha(stats)}, 'thesis_modified_by_queue': False,
        'freeze_sha256': FSHA, 'end_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'status': 'computation_complete_pending_review', 'statistics': str(stats)}), flush=True)

if __name__ == '__main__':
    main()
