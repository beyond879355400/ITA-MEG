#!/usr/bin/env python3
"""Restore only the authorized fixed-revision ID encoder and four exact small files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def main():
    plan = json.loads((HERE / 'magicface_missing_resources.json').read_text())
    base = ROOT / 'models/magicface_base'
    copied = []
    for row in plan['small_files']:
        source = Path(row['local_resident_exact_copy'])
        stat = source.stat()
        if getattr(stat, 'st_flags', 0) & 0x40000000 or stat.st_blocks == 0:
            raise RuntimeError('Exact small-file source is no longer resident')
        data = source.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != row['official_git_blob_sha1']:
            raise RuntimeError('Small file differs from official Git blob')
        if hashlib.sha256(data).hexdigest() != row['local_verified_sha256']:
            raise RuntimeError('Local small-file SHA256 drift')
        target = base / row['filename']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != row['local_verified_sha256']:
                raise RuntimeError('Do not overwrite an inconsistent destination')
        else:
            shutil.copyfile(source, target)
        copied.append({'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha(target),
                       'official_git_blob_sha1': row['official_git_blob_sha1'], 'source': str(source)})
    model = plan['missing_large_weight']
    destination = ROOT / 'models/magicface' / model['filename']
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    logdir = HERE / ('resource_restore/' + datetime.now().strftime('%Y%m%d%H%M'))
    stem = logdir
    i = 2
    while logdir.exists():
        logdir = stem.with_name(stem.name + f'-{i:02d}')
        i += 1
    logdir.mkdir(parents=True)
    def record(name, data):
        with (logdir / name).open('x') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
    record('small_files.json', copied)
    if not destination.exists() and partial.exists() and partial.stat().st_size == model['bytes']:
        if sha(partial) != model['official_sha256']:
            raise RuntimeError('Full-size partial has an incorrect SHA; do not redownload over it')
        partial.rename(destination)
        record('complete_partial_recovered.json', {'status': 'verified_without_network', 'sha256': model['official_sha256']})
    if not destination.exists():
        initial = partial.stat().st_size if partial.exists() else 0
        if initial > model['bytes']:
            raise RuntimeError('Partial download exceeds official file size')
        if shutil.disk_usage(ROOT).free < model['bytes'] - initial + 2147483648:
            raise RuntimeError('Insufficient disk for weight plus 2GiB safety margin')
        command = ['curl', '--fail', '--location', '--continue-at', '-', '--retry', '0', '--connect-timeout', '30',
                   '--max-time', '900', '--output', str(partial), model['download_url']]
        record('request.json', {'command': command, 'initial_partial_bytes': initial,
                               'official_bytes': model['bytes'], 'official_sha256': model['official_sha256'],
                               'restorer_sha256': sha(Path(__file__)),
                               'started_utc': datetime.now(timezone.utc).isoformat(), 'no_automatic_http_retry': True})
        with (logdir / 'curl.log').open('x') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if run.returncode:
            record('incomplete.json', {'returncode': run.returncode, 'partial_bytes': partial.stat().st_size if partial.exists() else 0,
                                       'action': 'preserve .part; inspect HTTP/rate-limit before resuming; no substitute model'})
            print(json.dumps({'status': 'incomplete', 'run': str(logdir), 'returncode': run.returncode}), flush=True)
            raise SystemExit(run.returncode)
        if partial.stat().st_size != model['bytes'] or sha(partial) != model['official_sha256']:
            raise RuntimeError('Downloaded ID encoder size/SHA mismatch; leave partial unpromoted')
        partial.rename(destination)
    if destination.stat().st_size != model['bytes'] or sha(destination) != model['official_sha256']:
        raise RuntimeError('Existing final weight is not the authorized checkpoint')
    record('completed.json', {'status': 'verified', 'weight': str(destination), 'bytes': destination.stat().st_size,
                              'sha256': sha(destination), 'official_source': model['download_url'],
                              'small_files': copied, 'environment_modified': False})
    print(json.dumps({'status': 'verified', 'weight': str(destination), 'run': str(logdir), 'bytes': destination.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
