"""Frozen complete native comparison; workers preserve all cases and failures."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from bridge import *

def index(path, document):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)

def validate_context(context):
    doc = frozen(context['protocol_path'], context['protocol_sha256'])
    method = context['method']; spec = doc['external']['methods'][method]
    need(spec['settings'] == context['settings'], 'Worker settings changed')
    planned = projection(doc, context['protocol_sha256'], context['raw_manifest_path'], context['raw_manifest_sha256'], method)
    matches = [r for r in planned['cases'] if r['execution_id'] == context['job']['execution_id']]
    need(matches == [context['job']], 'Worker request not exactly in freeze')
    need(not context['job'].get('preflight_error'), 'Raw-failed job must not enter native worker')
    NATIVE.verify_core_files(); NATIVE.verify_lease(context['asset_lease'])
    return doc

def worker(args):
    need(sha(args.context) == args.context_sha, 'Worker context changed')
    context = read(args.context); validate_context(context)
    out = args.output.resolve()
    need(COMPLETION / 'external_runs' in out.parents and not out.exists(), 'New external worker directory required')
    need(Path(sys.executable).absolute() == CONTRACTS.executable_path(context['settings']['runtime']['python_executable']), 'Wrong frozen Python environment')
    result = NATIVE.execute_job(context, out)
    return 0 if result['status'] == 'complete' else 1

def run_index(planned, receipts, raw_path, terminal=False, error=None):
    result = (CONTRACTS.failed_run_index(planned, receipts, raw_path, error) if error is not None
              else CONTRACTS.update_index(planned, receipts, raw_path, terminal=terminal))
    result.update(data_role=ROLE, independent_subject_test=False,
        preflight_failed_physical_jobs=sum(bool(r.get('preflight_failure')) for r in receipts.values()),
        physical_attempts=sum(bool(r.get('worker_dispatched')) for r in receipts.values()),
        actual_native_started=sum(bool(r.get('generation_started')) for r in receipts.values()))
    if error is None:
        # Intermediate all-complete receipts are not yet a terminal integrity acceptance.
        result['usable_for_evaluation'] = bool(terminal)
        if not terminal and result['status'] == 'complete':
            result['status'] = 'awaiting_terminal_integrity'
    return result

def run(args):
    doc = frozen(args.freeze, args.freeze_sha)
    spec = doc['external']['methods'][args.method]
    planned = projection(doc, args.freeze_sha, args.raw_manifest, args.raw_sha, args.method)
    lease = NATIVE.validate_runtime(spec['settings'], spec['artifacts'])
    report = {'status': 'validated_no_models', 'method': args.method, 'physical_count': len(planned['cases']),
              'logical_count': len(planned['logical_records']), 'data_role': ROLE, 'independent_subject_test': False}
    if not args.execute:
        print(json.dumps(report)); return 0
    need(args.output is not None, 'New output required')
    out = args.output.resolve()
    need(COMPLETION / 'external_runs' in out.parents and not out.exists(), 'New completion external run required')
    # Full bytes once, then stat/identity lease before and after every native worker.
    for asset in lease['assets']:
        need(sha(asset['path']) == asset['sha256'], 'Model asset SHA changed')
    out.mkdir(parents=True)
    base_context = {'protocol_path': str(args.freeze.resolve()), 'protocol_sha256': args.freeze_sha,
                    'raw_manifest_path': str(args.raw_manifest.resolve()), 'raw_manifest_sha256': args.raw_sha,
                    'method': args.method, 'settings': spec['settings'], 'asset_lease': lease,
                    'data_role': ROLE, 'independent_subject_test': False}
    save(out / 'configuration.json', base_context); save(out / 'projection.json', planned)
    receipts = copy.deepcopy(planned.get('preflight_receipts', {})); path = out / 'generation_index.json'
    raw_path = str(args.raw_manifest.resolve())
    index(path, run_index(planned, receipts, raw_path))
    try:
        for number, job in enumerate(planned['cases'], 1):
            if job.get('preflight_error'):
                receipt = receipts[job['execution_id']]
                save(out / 'receipts' / (job['case_id'] + '.json'), receipt)
                continue
            NATIVE.verify_lease(lease)
            context = {**copy.deepcopy(base_context), 'job': job}
            cp = out / 'contexts' / (job['case_id'] + '.json'); save(cp, context)
            jobout = out / 'executions' / job['case_id']; log = out / 'logs' / (job['case_id'] + '.log'); log.parent.mkdir(exist_ok=True)
            command = [str(CONTRACTS.executable_path(spec['settings']['runtime']['python_executable'])), '-B', str(Path(__file__).resolve()),
                       '--worker', '--context', str(cp), '--context-sha', sha(cp), '--output', str(jobout)]
            with log.open('x') as stream:
                process = subprocess.run(command, cwd=CONTRACTS.THESIS, stdout=stream, stderr=subprocess.STDOUT, check=False)
            receipt_path = jobout / 'worker_result.json'
            receipt = read(receipt_path) if receipt_path.is_file() else {'execution_id': job['execution_id'], 'status': 'failed', 'output': None,
                'generation_started': (jobout / 'generation_started.json').exists(), 'error': {'error_type': 'WorkerNoReceipt', 'returncode': process.returncode}}
            need(receipt['execution_id'] == job['execution_id'], 'Worker receipt has wrong request')
            need(receipt['status'] in ('complete', 'failed'), 'Worker receipt has nonterminal state')
            need(not (receipt['status'] == 'complete' and process.returncode != 0), 'Successful receipt contradicts failed worker process')
            receipt.update(worker_context_sha256=sha(cp), process_returncode=process.returncode, worker_dispatched=True)
            receipts[job['execution_id']] = receipt
            save(out / 'receipts' / (job['case_id'] + '.json'), receipt)
            index(path, run_index(planned, receipts, raw_path))
            print(json.dumps({'method': args.method, 'done': number, 'planned': len(planned['cases']), 'case_id': job['request_case_id'],
                              'seed': job['seed'], 'status': receipt['status']}, ensure_ascii=False), flush=True)
        # Revalidate control files, native source, assets and all successful output bytes.
        fresh = frozen(args.freeze, args.freeze_sha)
        need(projection(fresh, args.freeze_sha, args.raw_manifest, args.raw_sha, args.method) == planned, 'Projection changed during run')
        NATIVE.verify_core_files(); NATIVE.verify_lease(lease)
        for asset in lease['assets']:
            need(sha(asset['path']) == asset['sha256'], 'Model asset changed during native run')
        for receipt in receipts.values():
            if receipt['status'] == 'complete':
                need(sha(receipt['output']['path']) == receipt['output']['sha256'], 'Output changed during native run')
        result = run_index(planned, receipts, raw_path, terminal=True)
        result['settings'] = spec['settings']
        index(path, result)
        print(json.dumps({'method': args.method, 'status': result['status'], 'generation_index': ref(path)}), flush=True)
        return 0 if result['status'] == 'complete' else 1
    except Exception as error:
        result = run_index(planned, receipts, raw_path, terminal=True, error=error)
        index(path, result)
        save(out / 'run_failure.json', {'status': 'run_integrity_failure', 'error_type': type(error).__name__,
            'error': str(error), 'usable_for_evaluation': False, 'saved_outputs_preserved': True})
        print(json.dumps({'method': args.method, 'status': result['status'], 'usable_for_evaluation': False, 'error': str(error)}), flush=True)
        return 1

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--emit-spec', type=Path); p.add_argument('--freeze', type=Path); p.add_argument('--freeze-sha')
    p.add_argument('--raw-manifest', type=Path); p.add_argument('--raw-sha'); p.add_argument('--method', choices=METHODS)
    p.add_argument('--output', type=Path); p.add_argument('--execute', action='store_true')
    p.add_argument('--worker', action='store_true'); p.add_argument('--context', type=Path); p.add_argument('--context-sha')
    a = p.parse_args()
    if a.emit_spec:
        save(a.emit_spec, execution_spec()); print(json.dumps({'status': 'spec_only', 'file': ref(a.emit_spec)})); return 0
    if a.worker:
        need(a.context and a.context_sha and a.output, 'Worker context/output required'); return worker(a)
    need(all((a.freeze, a.freeze_sha, a.raw_manifest, a.raw_sha, a.method)), 'Explicit frozen controls required')
    return run(a)

if __name__ == '__main__':
    raise SystemExit(main())
