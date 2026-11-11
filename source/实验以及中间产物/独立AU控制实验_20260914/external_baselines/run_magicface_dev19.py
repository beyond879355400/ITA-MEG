#!/usr/bin/env python3
"""Full fixed dev19 via isolated native MagicFace processes, never reusing old smoke outputs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from prepare_dev_contracts import HERE, PYTHON, sha


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--batch-dir', type=Path)
    p.add_argument('--phase', choices=('prepare', 'generate'), required=True)
    args = p.parse_args()
    manifest = args.manifest.resolve()
    document = json.loads(manifest.read_text())
    assert document['schema_version'] == 3 and len(document['cases']) == 19 and not document['test_used']
    assert document['old_12_rows_exactly_preserved'] and document['sampling_seeds'] == [101]
    runner = HERE / 'run_magicface_smoke.py'
    if args.batch_dir:
        out = args.batch_dir.resolve()
    else:
        assert args.phase == 'prepare'
        base = HERE / ('runs/magicface_dev19_raw_native_' + datetime.now().strftime('%Y%m%d%H%M'))
        out = base
        i = 2
        while out.exists():
            out = base.with_name(base.name + f'-{i:02d}')
            i += 1
    assert HERE in out.parents
    configuration = {'manifest': str(manifest), 'manifest_sha256': sha(manifest), 'cases': document['cases'],
                     'case_count': 19, 'seeds': [101], 'steps': 50, 'scheduler': 'UniPCMultistepScheduler',
                     'guidance_scale': 7.5, 'image_guidance_scale': 1.5, 'source_reference_reset': 'one fresh Python process per case',
                     'runner_sha256': sha(runner), 'layout_seal_sha256': sha(HERE / 'magicface_runtime/layout_seal.json'),
                     'old_single_case_outputs_reused': False, 'test_used': False}
    if out.exists():
        assert json.loads((out / 'configuration.json').read_text()) == configuration
    else:
        out.mkdir(parents=True)
        write_json(out / 'configuration.json', configuration)
        (out / 'runner_snapshot.py').write_bytes(runner.read_bytes())
        (out / 'orchestrator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out / 'cases').mkdir(exist_ok=True)
    records = []
    for index, case in enumerate(document['cases'], 1):
        case_dir = out / 'cases' / case['case_id']
        phase = 'preprocess' if args.phase == 'prepare' else 'generate'
        marker = case_dir / ('preprocess.json' if args.phase == 'prepare' else 'generation.json')
        if marker.exists():
            existing = json.loads(marker.read_text())
            assert existing['status'] == 'complete'
            identity = json.loads((case_dir / 'case.json').read_text())
            assert identity['case'] == case and identity['manifest_sha256'] == sha(manifest)
            if args.phase == 'generate':
                assert existing['seed'] == 101 and existing['inference_steps'] == 50
                assert sha(case_dir / 'generated.png') == existing['output_sha256']
            records.append({'index': index, 'case_id': case['case_id'], 'status': 'complete', 'same_batch_resume': True})
            continue
        if args.phase == 'generate' and not (case_dir / 'preprocess.json').exists():
            records.append({'index': index, 'case_id': case['case_id'], 'status': 'failed', 'reason': 'source_preprocessing_failed'})
            continue
        logdir = out / 'attempt_logs' / case['case_id']
        logdir.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (logdir / f'{phase}_{attempt:03d}.log').exists():
            attempt += 1
        command = [str(PYTHON), '-B', str(runner), '--manifest', str(manifest), '--case-id', case['case_id'],
                   '--output', str(case_dir), '--phase', phase, '--device', 'cpu' if args.phase == 'prepare' else 'mps']
        started = time.monotonic()
        with (logdir / f'{phase}_{attempt:03d}.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            pid = process.pid
            code = process.wait()
        record = {'index': index, 'case_id': case['case_id'], 'group': case['group'], 'phase': phase, 'attempt': attempt,
                  'status': 'complete' if code == 0 and marker.exists() else 'failed', 'returncode': code,
                  'process_id': pid, 'seconds': time.monotonic() - started, 'command': command}
        if record['status'] == 'complete' and args.phase == 'generate':
            gen = json.loads(marker.read_text())
            assert gen['case_id'] == case['case_id'] and gen['seed'] == 101 and gen['inference_steps'] == 50
            assert gen['source_raw_sha256'] == case['source_sha256'] and not gen['target_image_read']
            assert gen['au_variation_sent'] == case['au_delta_native'] and sha(case_dir / 'generated.png') == gen['output_sha256']
            record.update(output=str(case_dir / 'generated.png'), output_sha256=gen['output_sha256'])
        records.append(record)
        write_json(logdir / f'{phase}_{attempt:03d}.json', record)
        summary = {'phase': args.phase, 'case_count': 19, 'processed_count': len(records),
                   'success_count': sum(r['status'] == 'complete' for r in records), 'failed_count': sum(r['status'] == 'failed' for r in records),
                   'records': records, 'test_used': False}
        write_json(out / (args.phase + '_summary.json'), summary)
        print(json.dumps({'batch': str(out), **{k: record[k] for k in ('index', 'case_id', 'phase', 'status', 'seconds')}}, ensure_ascii=False), flush=True)
        if code != 0:
            # Do not silently alter parameters or hop over an engineering failure.
            print('Stopped at the fixed case; inspect its preserved log and resume the same batch after an authorized technical fix.', flush=True)
            raise SystemExit(code)
    write_json(out / (args.phase + '_summary.json'), {'phase': args.phase, 'case_count': 19, 'processed_count': len(records),
               'success_count': sum(r['status'] == 'complete' for r in records), 'failed_count': sum(r['status'] == 'failed' for r in records),
               'records': records, 'test_used': False})
    print(json.dumps({'batch': str(out), 'phase': args.phase, 'status': 'complete', 'case_count': 19}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
