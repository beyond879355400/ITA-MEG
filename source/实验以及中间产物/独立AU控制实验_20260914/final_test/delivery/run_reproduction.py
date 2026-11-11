"""Same-machine reproduction only. No mode trains, downloads, or edits protocol."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PSHA = '41d0f77e1307fd7da22e212a2485101fa6ab9662f11b02ad63fdb94d0bceda55'
QSHA = '9f7c48d47af4af9f9913067534d46c0c9999933a92585cfb29beb2f24148b3f5'
FSHA = '6533d603eac9ba6bc69a69ca8bc42676ea0261ceb2c24db897e64763def8c4a0'
SSHA = 'f0f1baaf08464ac2e46bfe607ae7ea34ae1c49f200288b57b45e32df2a44d4ed'
VSHA = '1f36a1999f0296927c720d5dae50eae20c2d9e8d6fda80cbd79b4eadeab3836a'


@dataclass(frozen=True)
class Context:
    root: Path = ROOT
    python: str = str(Path.home() / 'Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python')
    protocol_sha: str = PSHA
    queue_sha: str = QSHA
    font_sha: str = FSHA
    statistics_sha: str = SSHA
    verify_sha: str = VSHA


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(path, expected):
    path = Path(path)
    require(path.is_file() and sha(path) == expected, 'Missing or changed SHA: ' + str(path))
    return {'path': str(path), 'sha256': expected}


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def preparation(context):
    final = context.root / 'final_test'
    original = final / 'execution_202609150734'
    complete_path = original / 'complete.json'
    # Refuse before any directory creation, subprocess launch, or model hashing.
    require(complete_path.is_file(), 'Original queue is not complete; reproduction is blocked')
    complete = read(complete_path)
    require(complete.get('status') == 'computed_pending_independent_review' and
            complete.get('protocol_sha256') == context.protocol_sha, 'Original completion binding changed')
    pinned = [check(final / 'frozen_protocol.json', context.protocol_sha),
              check(final / 'run_frozen_queue.py', context.queue_sha),
              check(final / 'run_with_fontcache.py', context.font_sha),
              check(context.root / 'statistics/paired_subject_statistics.py', context.statistics_sha)]
    result_path = original / 'measurements/statistics/paired_subject_results.json'
    require(complete['statistics']['path'] == str(result_path), 'Original statistics path changed')
    pinned.append(check(result_path, complete['statistics']['sha256']))
    result = read(result_path)
    require(result.get('statistics_source_sha256') == context.statistics_sha and
            result.get('metadata', {}).get('protocol_sha256') == context.protocol_sha, 'Original result provenance changed')
    input_path = result_path.with_name('paired_subject_input.json')
    pinned.append(check(input_path, result['input_sha256']))
    require(read(input_path).get('metadata', {}).get('protocol_sha256') == context.protocol_sha, 'Original statistics input protocol changed')
    pinned.append(check(complete_path, sha(complete_path)))
    return final, original, input_path, result_path, pinned


def active_commands(context):
    """Inspect process arguments only; never signal, kill, or modify processes."""
    listing = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
    names = ('run_frozen_queue.py', 'generate_ours.py', 'run_external_final.py',
             'train_centered_adapter.py', 'run_reproduction.py')
    busy = []
    for line in listing.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) in (os.getpid(), os.getppid()):
            continue
        command = parts[1]
        if any(name in command for name in names) and ('--execute' in command or '--full' in command or
                '--statistics' in command or '--method' in command or '--config' in command):
            busy.append({'pid': int(parts[0]), 'command': command})
    return busy


def new_run(final, mode):
    parent = final / 'reproduction_runs'
    require(not parent.is_symlink() and parent.resolve().is_relative_to(final.resolve()), 'Reproduction output escapes final_test')
    parent.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d%H%M')
    for number in range(1, 10000):
        path = parent / (mode + '_' + stamp + ('' if number == 1 else '-' + str(number).zfill(2)))
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise ValueError('Too many existing reproduction directories')


def execute(mode, context=Context(), *, busy_check=active_commands):
    require(mode in ('statistics', 'full'), 'Only explicit statistics or full execution is supported')
    require(sys.flags.optimize == 0, 'Optimized Python disables original queue gates; do not use -O')
    final, original, input_path, old_result, pinned = preparation(context)
    require(not busy_check(context), 'Original queue, training, generation, or another reproduction is still running')
    lock = final / '.reproduction.lock'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump({'pid': os.getpid(), 'mode': mode, 'started_utc': datetime.now(timezone.utc).isoformat()}, handle)
    output = None
    try:
        require(not busy_check(context), 'Concurrent generation started before lock acquisition')
        # Recheck accepted metadata under the common reproduction lock.
        preparation(context)
        output = new_run(final, mode)
        save(output / 'reproduction_request.json', {'mode': mode, 'protocol_sha256': context.protocol_sha,
             'original_run': str(original), 'verified_originals': pinned, 'python': context.python,
             'wrapper_sha256': sha(__file__), 'no_training': True, 'no_downloads': True,
             'no_protocol_changes': True, 'same_machine_environment_only': True})
        if mode == 'statistics':
            target = output / 'paired_subject_results.json'
            command = [context.python, '-B', str(context.root / 'statistics/paired_subject_statistics.py'),
                       '--input', str(input_path), '--output', str(target)]
            with (output / 'statistics.log').open('x') as logfile:
                subprocess.run(command, cwd=context.root, stdout=logfile, stderr=subprocess.STDOUT, check=True)
            fresh = read(target)
            require(fresh.get('input_sha256') == sha(input_path) and
                    fresh.get('statistics_source_sha256') == context.statistics_sha, 'Recomputed statistics binding changed')
            receipt = {'status': 'statistics_recomputed', 'output': check(target, sha(target)),
                       'original_result': check(old_result, read(original / 'complete.json')['statistics']['sha256']),
                       'byte_identical_to_original': sha(target) == sha(old_result), 'models_run': False}
        else:
            queue = final / 'run_frozen_queue.py'
            check(queue, context.queue_sha)
            spec = importlib.util.spec_from_file_location('frozen_reproduction_queue', queue)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # ONLY RUN changes. The original main creates it exclusively and
            # uses this unique basename for all external native output folders.
            module.RUN = output / ('run_' + output.name)
            require(module.PY == context.python and module.PSHA == context.protocol_sha, 'Original queue environment/protocol differs')
            module.main()
            receipt = {'status': 'full_computed_pending_independent_review',
                       'queue_run': str(module.RUN), 'queue_sha256': context.queue_sha,
                       'complete': check(module.RUN / 'complete.json', sha(module.RUN / 'complete.json')),
                       'models_run': True, 'original_protocol_unchanged': True}
        for ref in pinned:
            check(ref['path'], ref['sha256'])
        save(output / 'reproduction_complete.json', receipt)
        lock.unlink()
        return {'directory': str(output), **receipt}
    except BaseException as error:
        if output is not None:
            save(output / 'reproduction_failed.json', {'status': 'failed', 'error_type': type(error).__name__,
                 'error': str(error), 'lock_retained': str(lock), 'automatic_retry': False})
        # Fail closed: never release a lease while a child could still use MPS.
        # After checking all processes/logs, the operator may remove this lock.
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--verify', action='store_true', help='Default: read-only delivery hash checks')
    group.add_argument('--statistics', action='store_true', help='New output directory; original completion required')
    group.add_argument('--full', action='store_true', help='All eight frozen methods plus evaluation; original completion required')
    args = parser.parse_args(argv)
    if not args.statistics and not args.full:
        script = HERE / 'verify_delivery.sh'
        check(script, VSHA)
        return subprocess.run(['bash', str(script)], cwd=ROOT, check=False).returncode
    try:
        result = execute('statistics' if args.statistics else 'full')
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps({'status': 'blocked_or_failed', 'error_type': type(error).__name__, 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
