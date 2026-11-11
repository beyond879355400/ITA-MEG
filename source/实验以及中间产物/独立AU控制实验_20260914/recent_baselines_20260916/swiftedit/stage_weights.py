"""Audited sequential multipart staging; no merged tar, no implicit downloads.

All mutations require CACHE_READY plus an explicit action. No model imports.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3
RESERVE = 8 * GIB
MAX_EXPANDED = 20 * GIB


def need(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, document):
    with Path(path).open('x') as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)


def gate(key):
    entry = read(HERE / 'approvals.json').get(key)
    need(isinstance(entry, dict) and entry.get('message') and entry.get('granted_by') == 'root',
         'Waiting for root permission: ' + key)


def assets():
    meta = read(HERE / 'release_assets.json')
    entries = meta['assets']
    need(len(entries) == 5 and sum(x['size'] for x in entries) == 9748940800, 'Unexpected official release size')
    need([x['name'][-2:] for x in entries] == ['aa','ab','ac','ad','ae'], 'Unexpected multipart order')
    for entry in entries:
        need(entry['url'].startswith('https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/')
             and entry['digest'].startswith('sha256:') and len(entry['digest']) == 71,
             'Unexpected official URL/hash')
    return entries


def log(event):
    folder = HERE / 'logs'; folder.mkdir(exist_ok=True)
    with (folder / 'staging_events.jsonl').open('a', buffering=1) as stream:
        stream.write(json.dumps({'at': now(), **event}, ensure_ascii=False, allow_nan=False) + '\n')


def download():
    gate('cache_ready')
    folder = HERE / 'downloads'; folder.mkdir(exist_ok=True)
    entries = assets()
    remaining = sum(max(0, x['size'] - ((folder / x['name']).stat().st_size if (folder / x['name']).exists()
                    else (folder / (x['name'] + '.partial')).stat().st_size if (folder / (x['name'] + '.partial')).exists() else 0)) for x in entries)
    # Reserved future extraction space is a planning bound, then exact tar sizes
    # are checked before extraction. This deliberately refuses tight disks.
    need(shutil.disk_usage(HERE).free >= remaining + 12 * GIB + RESERVE,
         'Insufficient space for remaining parts + provisional12GiB extraction +8GiB reserve')
    for entry in entries:
        gate('cache_ready')
        complete, partial = folder / entry['name'], folder / (entry['name'] + '.partial')
        if complete.exists():
            need(complete.stat().st_size == entry['size'] and sha(complete) == entry['digest'][7:], 'Existing complete part differs; never overwrite')
            log({'event': 'reuse_verified_part', 'name': entry['name'], 'bytes': entry['size']}); continue
        offset = partial.stat().st_size if partial.exists() else 0
        need(0 <= offset <= entry['size'], 'Partial file exceeds official size')
        transferred = 0
        started = time.monotonic()
        try:
            if offset < entry['size']:
                request = urllib.request.Request(entry['url'], headers={'Range': 'bytes=%d-' % offset} if offset else {})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if offset:
                        need(response.status == 206 and response.headers.get('Content-Range', '').startswith('bytes %d-' % offset),
                             'Resume server did not honor range; preserving partial for review')
                    else:
                        need(response.status == 200, 'Unexpected download response')
                    log({'event': 'download_start', 'name': entry['name'], 'offset': offset, 'http_status': response.status,
                         'expected_bytes': entry['size'], 'url': entry['url']})
                    with partial.open('ab' if partial.exists() else 'xb') as stream:
                        last_log = 0
                        while True:
                            need(shutil.disk_usage(HERE).free >= RESERVE, 'Free-space reserve reached; pause retaining partial')
                            chunk = response.read(4 * 1024 * 1024)
                            if not chunk:
                                break
                            transferred += len(chunk)
                            need(offset + transferred <= entry['size'], 'Response exceeds approved byte limit')
                            need(shutil.disk_usage(HERE).free >= RESERVE + len(chunk), 'Upcoming chunk would violate8GiB reserve')
                            stream.write(chunk)
                            if transferred - last_log >= 64 * 1024 * 1024:
                                stream.flush(); last_log = transferred
                                log({'event': 'download_progress', 'name': entry['name'], 'network_bytes_this_attempt': transferred,
                                     'partial_bytes': offset + transferred})
            need(partial.stat().st_size == entry['size'], 'Truncated multipart asset')
            digest = sha(partial)
            need(digest == entry['digest'][7:], 'Official SHA mismatch; bad partial retained')
            need(not complete.exists(), 'Complete target appeared concurrently')
            partial.rename(complete)
            log({'event': 'download_verified', 'name': entry['name'], 'network_bytes_this_attempt': transferred,
                 'bytes': entry['size'], 'sha256': digest, 'seconds': time.monotonic()-started})
        except BaseException as error:
            log({'event': 'download_failed', 'name': entry['name'], 'network_bytes_this_attempt': transferred,
                 'seconds': time.monotonic()-started, 'error': str(error)}); raise


class MultipartReader(io.RawIOBase):
    """Concatenate files in memory-bounded reads, never write a merged archive."""
    def __init__(self, paths):
        self.paths, self.index, self.stream = list(paths), 0, None

    def readable(self):
        return True

    def readinto(self, target):
        while self.index < len(self.paths):
            if self.stream is None:
                self.stream = Path(self.paths[self.index]).open('rb')
            data = self.stream.read(len(target))
            if data:
                target[:len(data)] = data
                return len(data)
            self.stream.close(); self.stream = None; self.index += 1
        return 0

    def close(self):
        if self.stream is not None:
            self.stream.close(); self.stream = None
        super().close()


@contextmanager
def archive(paths):
    with io.BufferedReader(MultipartReader(paths), buffer_size=1024*1024) as stream:
        with tarfile.open(fileobj=stream, mode='r|*') as tar:
            yield tar


def safe_member(member):
    path = PurePosixPath(member.name)
    need(not path.is_absolute() and '..' not in path.parts and len(member.name) < 1024, 'Unsafe tar member path')
    need(member.isfile() or member.isdir(), 'Links/devices/nonregular members are forbidden')
    need(0 <= member.size <= 8 * GIB, 'Individual member exceeds8GiB guard')
    return path


def verified_parts():
    paths = []
    for entry in assets():
        path = HERE / 'downloads' / entry['name']
        need(path.is_file() and path.stat().st_size == entry['size'] and sha(path) == entry['digest'][7:], 'Missing/unverified multipart asset')
        paths.append(path)
    return paths


def inspect(paths):
    rows, total = [], 0
    with archive(paths) as tar:
        for member in tar:
            safe_member(member)
            total += member.size
            need(total <= MAX_EXPANDED and len(rows) < 10000, 'Expanded archive exceeds approved size/count guard')
            rows.append({'name': member.name, 'bytes': member.size, 'directory': member.isdir()})
    return {'members': rows, 'expanded_bytes': total}


def extract():
    gate('cache_ready')
    paths = verified_parts()
    index = inspect(paths)
    need(shutil.disk_usage(HERE).free >= index['expanded_bytes'] + RESERVE, 'Exact expansion would violate8GiB reserve')
    parent = HERE / 'weights'; parent.mkdir(exist_ok=True)
    out = parent / ('official_v1_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    save_new(out / 'archive_index.json', index)
    rows = []
    try:
        with archive(paths) as tar:
            for member in tar:
                relative = safe_member(member)
                target = out.joinpath(*relative.parts)
                need(target.resolve() == out or out in target.resolve().parents, 'Tar escaped output root')
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True); continue
                need(shutil.disk_usage(HERE).free >= member.size + RESERVE, 'Disk reserve reached during extraction')
                target.parent.mkdir(parents=True, exist_ok=True)
                h, written = hashlib.sha256(), 0
                with tar.extractfile(member) as source, target.open('xb') as dest:
                    while True:
                        chunk = source.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        need(shutil.disk_usage(HERE).free >= RESERVE + len(chunk), 'Upcoming output chunk would violate8GiB reserve')
                        dest.write(chunk); h.update(chunk); written += len(chunk)
                need(written == member.size, 'Tar member truncated')
                rows.append({'path': str(target.relative_to(out)), 'bytes': written, 'sha256': h.hexdigest()})
        receipt = {'status': 'complete', 'at': now(), 'files': rows, 'expanded_bytes': index['expanded_bytes'],
                   'merged_archive_written': False, 'source_parts_deleted': False}
        save_new(out / 'extraction_receipt.json', receipt)
        log({'event': 'extraction_complete', 'output': str(out), 'expanded_bytes': index['expanded_bytes'], 'files': len(rows)})
        print(json.dumps({'status': 'complete', 'output': str(out), 'expanded_bytes': index['expanded_bytes']}))
    except BaseException as error:
        save_new(out / 'extraction_failure.json', {'status': 'failed', 'error': str(error), 'completed_files': rows})
        log({'event': 'extraction_failed', 'output': str(out), 'error': str(error)}); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan','download','inspect','extract'], default='plan', nargs='?')
    a = p.parse_args()
    if a.action == 'plan':
        print(json.dumps({'download_bytes': sum(x['size'] for x in assets()), 'free_bytes': shutil.disk_usage(HERE).free,
             'minimum_reserve_bytes': RESERVE, 'merged_tar': False, 'cache_ready': bool(read(HERE/'approvals.json')['cache_ready'])}))
    elif a.action == 'download':
        download()
    elif a.action == 'inspect':
        gate('cache_ready'); print(json.dumps(inspect(verified_parts()), indent=2))
    else:
        extract()


if __name__ == '__main__':
    main()
