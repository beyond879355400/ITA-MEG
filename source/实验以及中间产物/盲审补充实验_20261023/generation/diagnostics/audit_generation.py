#!/usr/bin/env python3
"""Read-only generation audit; incomplete JSONL tail/unfinished slots are pending."""
import argparse
import array
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ARMS = ('full', 'no_projection', 'no_smoothing', 'both_off')
INPUT_HASH_FIELDS = ('condition_hashes', 'source_latent_sha256', 'noise_sha256',
                     'edit_mask_sha256', 'cleanup_mask_sha256')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_ref(item):
    path = Path(item['path']).resolve()
    need(path.is_file(), 'Referenced file missing: ' + str(path))
    need(sha(path) == item['sha256'], 'Referenced file SHA changed: ' + str(path))
    return path


def read_float32_mask(path):
    """Read the frozen small NumPy mask without importing NumPy/torch/models."""
    with Path(path).open('rb') as stream:
        need(stream.read(6) == b'\x93NUMPY', 'Invalid NPY magic')
        version = tuple(stream.read(2))
        need(version in ((1, 0), (2, 0)), 'Unsupported NPY version')
        size_bytes = 2 if version == (1, 0) else 4
        header_size = int.from_bytes(stream.read(size_bytes), 'little')
        need(0 < header_size < 10000, 'Unexpected mask header size')
        header = ast.literal_eval(stream.read(header_size).decode('latin1'))
        need(header == {'descr': '<f4', 'fortran_order': False, 'shape': (64, 64)}, 'Mask dtype/shape/order changed')
        payload = stream.read()
    need(len(payload) == 64 * 64 * 4, 'Mask payload size changed')
    values = array.array('f')
    values.frombytes(payload)
    if sys.byteorder != 'little':
        values.byteswap()
    need(all(0 <= v <= 1 for v in values), 'Mask values outside finite [0,1]')
    return values, hashlib.sha256(payload).hexdigest()


def load_complete_lines(path):
    """Snapshot bytes once. A concurrent writer's unterminated tail is not a failure."""
    if not Path(path).exists():
        return [], {'complete_lines': 0, 'partial_tail_bytes': 0, 'journal_missing_pending': True}, []
    raw = Path(path).read_bytes()
    parts = raw.split(b'\n')
    tail = parts.pop()
    records, errors = [], []
    for index, line in enumerate(parts, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            need(isinstance(row, dict), 'JSONL row is not an object')
            records.append(row)
        except (ValueError, UnicodeDecodeError) as error:
            errors.append({'check': 'journal_complete_line', 'line': index, 'error': str(error)})
    return records, {'complete_lines': len(parts), 'partial_tail_bytes': len(tail),
                      'journal_missing_pending': False, 'snapshot_bytes': len(raw),
                      'snapshot_sha256': hashlib.sha256(raw).hexdigest()}, errors


def append_report(path, document):
    path = Path(path).resolve()
    need(HERE in path.parents, 'Report must remain within the new diagnostics directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def key_for(row):
    return row['case_id'], row['seed'], row['arm']


def check_success(row, case, masks, reference, run_root):
    key = key_for(row)
    need(row['subject'] == case['subject'], 'Subject metadata differs from frozen case')
    need(row['source_au12'] == case['source_au_vector_0_5'] and
         row['target_au12'] == case['target_au_vector_0_5'], 'AU conditions differ from frozen case')
    need(all(field in row for field in INPUT_HASH_FIELDS), 'Missing input hash field')
    for field in ('condition_hashes', 'source_latent_sha256', 'noise_sha256'):
        need(row[field] == reference[field], 'Historical common input hash differs: ' + field)
    need(row['edit_mask_sha256'] == masks['edit_tensor_sha256'], 'Edit mask tensor changed')
    need(row['cleanup_mask_sha256'] == masks['cleanup_tensor_sha256'], 'Cleanup mask tensor changed')
    expected_p = row['arm'] in ('full', 'no_smoothing')
    expected_s = row['arm'] in ('full', 'no_projection')
    need(row['switches']['projection'] is expected_p and row['switches']['smoothing'] is expected_s, 'Arm switch declaration inconsistent')
    trace = row['projection_trace']
    need(len(trace) == 15 and all(t['projection_applied'] is expected_p for t in trace), 'Projection trace count/state mismatch')
    need(row['projection_count'] == (15 if expected_p else 0), 'Projection count mismatch')
    terminal = row['terminal_trace']
    need(terminal['projection_enabled'] is expected_p and terminal['smoothing_enabled'] is expected_s, 'Terminal switch receipt inconsistent')
    need(terminal['base_mode'] == 'terminal_last_eps', 'Base terminal correction changed')
    need(row['terminal_unet_calls'] == 0 and row['deis_unet_calls'] == row['total_unet_calls'] == 15, 'UNet call count changed')
    if expected_s:
        need(isinstance(terminal['filter'], dict), 'Enabled smoothing has no filter receipt')
        need(terminal['filter'].get('sigma_latent') == .5, 'Smoothing sigma changed')
    else:
        need(terminal['filter'] is None, 'Disabled smoothing still has a filter receipt')
    if not expected_p and expected_s:
        need(terminal['filter'].get('final_projection') == 'disabled', 'Disabled projection incorrectly recorded as enabled')
    image = Path(row['output_path']).resolve()
    need(run_root in image.parents, 'Output image is outside the audited run')
    need(re.fullmatch(r'[0-9a-f]{64}', row['output_sha256']) is not None, 'Invalid output SHA')
    need(image.is_file() and sha(image) == row['output_sha256'], 'Output image missing or SHA differs')
    if row['arm'] == 'full':
        parity = row['parity']
        need(parity.get('passed') is True and parity.get('pixels_equal') is True and parity.get('sha256_equal') is True,
             'Full-arm parity not passed')
        need(row['output_sha256'] == reference['sha256'] == parity['reference']['sha256'], 'Full-arm output SHA differs from historical reference')
    return key


def audit(freeze_path, run_root):
    freeze_path, run_root = Path(freeze_path).resolve(), Path(run_root).resolve()
    plan, config = read(freeze_path), read(run_root / 'configuration.json')
    need(plan['kind'] == 'frozen_preservation_factorial_v1' and plan['status'] == 'frozen', 'Invalid frozen ablation plan')
    need(config['freeze']['sha256'] == sha(freeze_path) and Path(config['freeze']['path']).resolve() == freeze_path, 'Run uses another freeze')
    need(config['smoke'] is False and config['planned'] == 288, 'Audit requires the formal 288-slot run, not smoke')
    cases = {c['case_id']: c for c in plan['cases']}
    need(len(cases) == 24 and plan['seeds'] == [101, 202, 303], 'Frozen population or seeds changed')
    expected = {(cid, seed, arm) for cid in cases for seed in plan['seeds'] for arm in ARMS}
    slots = config['slot_plan']
    need(len(slots) == len(expected) == 288 and {key_for(row) for row in slots} == expected, 'Slot plan differs from the full grid')
    cleanup = {}
    for cid in cases:
        masks = plan['masks'][cid]
        values, digest = read_float32_mask(verify_ref(masks['cleanup']))
        need(digest == masks['cleanup_tensor_sha256'], 'Frozen cleanup tensor SHA mismatch')
        cleanup[cid] = {'nonzero_pixels': sum(v != 0 for v in values), 'subject': cases[cid]['subject']}
    records, journal, errors = load_complete_lines(run_root / 'generation_records.jsonl')
    zero_cleanup_count = sum(v['nonzero_pixels'] == 0 for v in cleanup.values())
    if zero_cleanup_count != 12:
        errors.append({'check': 'zero_cleanup_population', 'expected': 12, 'actual': zero_cleanup_count})
    selected, attempts = {}, Counter()
    for row in records:
        try:
            key = key_for(row)
            need(key in expected, 'Unexpected case/seed/arm in journal')
            need(row['status'] in ('success', 'failed'), 'Unknown execution status')
            attempts[key] += 1
            if row['status'] == 'success':
                check_success(row, cases[key[0]], plan['masks'][key[0]], plan['parity_references'][f'{key[0]}::{key[1]}'], run_root)
                selected[key] = row
            elif key not in selected or selected[key]['status'] != 'success':
                selected[key] = row
        except (KeyError, ValueError, TypeError, OSError) as error:
            errors.append({'check': 'completed_record', 'execution_id': row.get('execution_id'), 'error': str(error)})
    groups, equality_controls = [], []
    for cid in cases:
        for seed in plan['seeds']:
            completed = [selected[(cid, seed, arm)] for arm in ARMS
                         if (cid, seed, arm) in selected and selected[(cid, seed, arm)]['status'] == 'success']
            mismatch = [field for field in INPUT_HASH_FIELDS if len(completed) > 1 and
                        any(r[field] != completed[0][field] for r in completed[1:])]
            if mismatch:
                errors.append({'check': 'paired_inputs', 'case_id': cid, 'seed': seed, 'fields': mismatch})
            groups.append({'case_id': cid, 'seed': seed, 'successful_arms': [r['arm'] for r in completed],
                           'status': 'failed' if mismatch else ('pass' if len(completed) == 4 else 'pending'),
                           'input_hashes_consistent_so_far': not mismatch})
            if cleanup[cid]['nonzero_pixels'] == 0:
                for left, right in (('full', 'no_smoothing'), ('no_projection', 'both_off')):
                    a, b = selected.get((cid, seed, left)), selected.get((cid, seed, right))
                    available = a is not None and b is not None and a['status'] == b['status'] == 'success'
                    equal = available and a['output_sha256'] == b['output_sha256']
                    equality_controls.append({'case_id': cid, 'seed': seed, 'left': left, 'right': right,
                                              'status': ('pass' if equal else 'failed') if available else 'pending',
                                              'sha256_equal': bool(equal) if available else None})
                    if available and not equal:
                        errors.append({'check': 'zero_cleanup_identity', 'case_id': cid, 'seed': seed, 'arms': [left, right]})
    successes = sum(row['status'] == 'success' for row in selected.values())
    failed = [key for key, row in selected.items() if row['status'] == 'failed']
    pending = sorted(expected - set(selected))
    controls = Counter(row['status'] for row in equality_controls)
    if len(equality_controls) != 72:
        errors.append({'check': 'zero_cleanup_pair_count', 'expected': 72, 'actual': len(equality_controls)})
    if errors or failed:
        status = 'audit_failed'
    elif pending or journal['partial_tail_bytes']:
        status = 'valid_so_far_pending'
    else:
        status = 'complete_pass'
    return {'kind': 'read_only_generation_factorial_audit', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': status, 'freeze': {'path': str(freeze_path), 'sha256': sha(freeze_path)},
            'run_root': str(run_root), 'models_loaded': False, 'new_inference': False,
            'planned': 288, 'successes': successes, 'failed_slots': failed, 'pending_count': len(pending),
            'pending_journal_tail': journal['partial_tail_bytes'] > 0,
            'pending_slots': pending, 'journal_snapshot': journal,
            'attempt_count': len(records), 'slots_with_multiple_attempts': sum(n > 1 for n in attempts.values()),
            'cleanup_cases': cleanup, 'zero_cleanup_case_count': zero_cleanup_count,
            'nonzero_cleanup_case_count': sum(v['nonzero_pixels'] > 0 for v in cleanup.values()),
            'full_parity_passed': sum(r['status'] == 'success' and r['arm'] == 'full' for r in selected.values()),
            'paired_input_groups': groups, 'zero_cleanup_controls': equality_controls,
            'zero_cleanup_control_counts': dict(controls), 'errors': errors,
            'interpretation': 'Zero-cleanup equality is expected: disabling an already inactive local smoother must have no effect. Pending slots are not experimental failures.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = audit(args.freeze, args.run_root)
    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report = args.report or HERE / 'reports' / f'audit_{now}.json'
    append_report(report, result)
    print(json.dumps({'status': result['status'], 'successes': result['successes'], 'pending': result['pending_count'],
                      'errors': len(result['errors']), 'failed_slots': len(result['failed_slots']),
                      'zero_cleanup_controls': result['zero_cleanup_control_counts'], 'report': str(report.resolve())}, ensure_ascii=False))
    raise SystemExit(2 if result['status'] == 'audit_failed' or
                     (args.require_complete and result['status'] != 'complete_pass') else 0)


if __name__ == '__main__':
    main()
