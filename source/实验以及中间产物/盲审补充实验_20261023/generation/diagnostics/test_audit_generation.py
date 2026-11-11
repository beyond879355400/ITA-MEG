"""Synthetic stdlib fixtures only: no model imports or real experimental writes."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import audit_generation as audit


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def mask_fixture(path, active):
    header = repr({'descr': '<f4', 'fortran_order': False, 'shape': (64, 64)}).encode('ascii') + b'\n'
    payload = struct.pack('<4096f', *([.5 if active else 0] + [0] * 4095))
    path.write_bytes(b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header + payload)
    return {'path': str(path), 'sha256': audit.sha(path)}, digest(payload)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=audit.HERE)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'run'
        (self.run / 'images').mkdir(parents=True)
        self.plan = {'kind': 'frozen_preservation_factorial_v1', 'status': 'frozen', 'seeds': [101, 202, 303],
                     'cases': [], 'masks': {}, 'parity_references': {}}
        self.rows, self.slots = [], []
        for index in range(24):
            cid, subject = f'fixture_{index:02d}', f'SN{index // 3:03d}'
            case = {'case_id': cid, 'subject': subject, 'source_au_vector_0_5': [0] * 12,
                    'target_au_vector_0_5': [1] + [0] * 11}
            self.plan['cases'].append(case)
            mask, tensor_hash = mask_fixture(self.root / f'mask_{index}.npy', index >= 12)
            self.plan['masks'][cid] = {'cleanup': mask, 'cleanup_tensor_sha256': tensor_hash, 'edit_tensor_sha256': 'e' * 64}
            for seed in [101, 202, 303]:
                reference = {'condition_hashes': {'au_embedding': 'a' * 64, 'image_embedding': 'b' * 64},
                             'noise_sha256': 'c' * 64, 'source_latent_sha256': 'd' * 64}
                full_bytes = f'{cid}::{seed}::full'.encode()
                reference['sha256'] = digest(full_bytes)
                self.plan['parity_references'][f'{cid}::{seed}'] = reference
                for arm in audit.ARMS:
                    projection, smoothing = arm in ('full', 'no_smoothing'), arm in ('full', 'no_projection')
                    slot = {'case_id': cid, 'subject': subject, 'seed': seed, 'arm': arm}
                    self.slots.append(slot)
                    group = ('full' if projection else 'unprojected') if index < 12 else arm
                    raw = f'{cid}::{seed}::{group}'.encode()
                    path = self.run / 'images' / f'{cid}_{seed}_{arm}.bin'
                    path.write_bytes(raw)
                    row = {**slot, **reference, 'status': 'success', 'execution_id': f'{cid}::{seed}::{arm}',
                           'source_au12': case['source_au_vector_0_5'], 'target_au12': case['target_au_vector_0_5'],
                           'edit_mask_sha256': 'e' * 64, 'cleanup_mask_sha256': tensor_hash,
                           'output_path': str(path), 'output_sha256': digest(raw),
                           'switches': {'projection': projection, 'smoothing': smoothing},
                           'projection_count': 15 if projection else 0,
                           'projection_trace': [{'projection_applied': projection} for _ in range(15)],
                           'terminal_trace': {'projection_enabled': projection, 'smoothing_enabled': smoothing,
                                              'base_mode': 'terminal_last_eps',
                                              'filter': {'sigma_latent': .5, 'final_projection': 'original_editmask_once' if projection else 'disabled'} if smoothing else None},
                           'terminal_unet_calls': 0, 'deis_unet_calls': 15, 'total_unet_calls': 15}
                    if arm == 'full':
                        row['parity'] = {'passed': True, 'pixels_equal': True, 'sha256_equal': True, 'reference': reference}
                    self.rows.append(row)
        self.freeze = self.root / 'freeze.json'
        write_json(self.freeze, self.plan)
        write_json(self.run / 'configuration.json', {'freeze': {'path': str(self.freeze), 'sha256': audit.sha(self.freeze)},
                                                    'smoke': False, 'planned': 288, 'slot_plan': self.slots})

    def tearDown(self):
        self.temporary.cleanup()

    def write_rows(self, rows=None, tail=b''):
        rows = self.rows if rows is None else rows
        payload = b''.join(json.dumps(row).encode() + b'\n' for row in rows) + tail
        (self.run / 'generation_records.jsonl').write_bytes(payload)

    def inspect(self):
        return audit.audit(self.freeze, self.run)

    def test_complete_288_passes_and_checks_72_zero_cleanup_pairs(self):
        self.write_rows()
        result = self.inspect()
        self.assertEqual(result['status'], 'complete_pass')
        self.assertEqual(result['successes'], 288)
        self.assertEqual(result['full_parity_passed'], 72)
        self.assertEqual(result['zero_cleanup_case_count'], 12)
        self.assertEqual(result['zero_cleanup_control_counts'], {'pass': 72})
        self.assertTrue(all(g['status'] == 'pass' for g in result['paired_input_groups']))

    def test_missing_journal_is_pending_not_failed(self):
        result = self.inspect()
        self.assertEqual(result['status'], 'valid_so_far_pending')
        self.assertEqual(result['pending_count'], 288)
        self.assertEqual(result['errors'], [])

    def test_partial_final_line_is_pending_not_failed(self):
        self.write_rows(self.rows[:1], tail=b'{"case_id":"unfinished')
        result = self.inspect()
        self.assertEqual(result['status'], 'valid_so_far_pending')
        self.assertEqual(result['successes'], 1)
        self.assertEqual(result['pending_count'], 287)
        self.assertGreater(result['journal_snapshot']['partial_tail_bytes'], 0)
        self.assertEqual(result['errors'], [])

    def test_all_slots_done_but_partial_tail_is_still_pending(self):
        self.write_rows(tail=b'{"unfinished": true')
        result = self.inspect()
        self.assertEqual(result['status'], 'valid_so_far_pending')
        self.assertEqual(result['successes'], 288)
        self.assertEqual(result['pending_count'], 0)
        self.assertTrue(result['pending_journal_tail'])
        self.assertEqual(result['errors'], [])

    def test_wrong_zero_cleanup_population_is_error(self):
        cid = self.plan['cases'][12]['case_id']
        reference, tensor = mask_fixture(self.root / 'changed_zero.npy', False)
        self.plan['masks'][cid].update(cleanup=reference, cleanup_tensor_sha256=tensor)
        write_json(self.freeze, self.plan)
        config = audit.read(self.run / 'configuration.json')
        config['freeze']['sha256'] = audit.sha(self.freeze)
        write_json(self.run / 'configuration.json', config)
        result = self.inspect()
        self.assertEqual(result['status'], 'audit_failed')
        self.assertTrue(any(e['check'] == 'zero_cleanup_population' for e in result['errors']))
        self.assertTrue(any(e['check'] == 'zero_cleanup_pair_count' for e in result['errors']))

    def test_malformed_complete_line_is_error(self):
        self.write_rows([], tail=b'{broken}\n')
        result = self.inspect()
        self.assertEqual(result['status'], 'audit_failed')
        self.assertEqual(result['errors'][0]['check'], 'journal_complete_line')

    def test_projection_count_changed_is_error(self):
        self.rows[1]['projection_count'] = 15
        self.write_rows(self.rows[:2])
        result = self.inspect()
        self.assertEqual(result['status'], 'audit_failed')
        self.assertTrue(any('Projection count mismatch' in e['error'] for e in result['errors']))

    def test_disabled_smoothing_filter_is_error(self):
        self.rows[2]['terminal_trace']['filter'] = {'sigma_latent': .5}
        self.write_rows(self.rows[:3])
        self.assertTrue(any('Disabled smoothing' in e['error'] for e in self.inspect()['errors']))

    def test_old_input_hash_mismatch_is_error(self):
        self.rows[1]['noise_sha256'] = '0' * 64
        self.write_rows(self.rows[:2])
        self.assertTrue(any('common input hash' in e['error'] for e in self.inspect()['errors']))

    def test_full_parity_false_is_error(self):
        self.rows[0]['parity']['passed'] = False
        self.write_rows(self.rows[:1])
        self.assertTrue(any('parity not passed' in e['error'] for e in self.inspect()['errors']))

    def test_image_tamper_is_error(self):
        Path(self.rows[0]['output_path']).write_bytes(b'tamper')
        self.write_rows(self.rows[:1])
        self.assertTrue(any('Output image missing or SHA differs' in e['error'] for e in self.inspect()['errors']))

    def test_zero_cleanup_mismatch_is_error(self):
        row = self.rows[2]
        Path(row['output_path']).write_bytes(b'mismatch')
        row['output_sha256'] = digest(b'mismatch')
        self.write_rows(self.rows[:4])
        self.assertTrue(any(e['check'] == 'zero_cleanup_identity' for e in self.inspect()['errors']))

    def test_failure_then_success_retains_history_and_passes_slot(self):
        failure = {**self.rows[0], 'status': 'failed', 'error': 'synthetic earlier attempt'}
        self.write_rows([failure, self.rows[0]])
        result = self.inspect()
        self.assertEqual(result['status'], 'valid_so_far_pending')
        self.assertEqual(result['successes'], 1)
        self.assertEqual(result['failed_slots'], [])
        self.assertEqual(result['slots_with_multiple_attempts'], 1)

    def test_failed_slot_is_not_mislabeled_pending(self):
        failure = {**self.rows[0], 'status': 'failed', 'error': 'synthetic failure'}
        self.write_rows([failure])
        result = self.inspect()
        self.assertEqual(result['status'], 'audit_failed')
        self.assertEqual(len(result['failed_slots']), 1)
        self.assertEqual(result['pending_count'], 287)

    def test_report_does_not_overwrite(self):
        path = self.root / 'report.json'
        audit.append_report(path, {'fixture': True})
        with self.assertRaises(FileExistsError):
            audit.append_report(path, {'fixture': False})
        self.assertEqual(audit.read(path), {'fixture': True})


if __name__ == '__main__':
    unittest.main(verbosity=2)
