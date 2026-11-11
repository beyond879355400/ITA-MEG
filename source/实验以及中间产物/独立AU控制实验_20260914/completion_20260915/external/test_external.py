"""Metadata-only external bridge tests; no real images or models are loaded."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import bridge
import run_external as runner


class ExternalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.label_path = bridge.COMPLETION / 'protocol/proposal_v1/cases_label_only.json'
        cls.labels = bridge.read(cls.label_path)['cases']
        cls.spec = bridge.execution_spec()
        video_ref = bridge.ref(bridge.COMPLETION / 'protocol/video_input_refs.json')
        cls.videos = {v['subject']: v for v in bridge.read(video_ref['path'])['videos']}
        cls.freeze = {'kind': 'completion_v11_frozen_protocol', 'status': 'frozen', 'extraction_authorized': True,
            'case_ids': [c['case_id'] for c in cls.labels], 'label_plan': bridge.ref(cls.label_path), 'external': cls.spec,
            'extraction': {'inputrefs': video_ref}}

    def raw(self):
        rows = []
        for label in self.labels:
            frames = {}
            for side in ('source', 'target'):
                frame = label[side + '_frame_1based']
                frames[side] = {'subject': label['subject'], 'split': bridge.ROLE, 'camera': label['camera'],
                    'status': 'complete', 'path': f'/synthetic/{label["subject"]}/frame_{frame:05d}.png',
                    'sha256': 'd' * 64, 'width': 1024, 'height': 768, 'mode': 'RGB',
                    'label_frame_1based': frame, 'video_frame_zero_based': frame - 1,
                    'original_video_sha256': self.videos[label['subject']]['sha256'], 'original_video_path': label['video_path']}
            rows.append({'case_id': label['case_id'], 'subject': label['subject'], 'split': bridge.ROLE,
                'camera': label['camera'], 'status': 'complete', 'kind': label['kind'],
                'generation_input': {'source_image': frames['source'], 'source_au_vector_0_5': label['source_au_vector_0_5'],
                    'target_au_vector_0_5': label['target_au_vector_0_5']},
                'target_reference_image': frames['target'], 'target_pixels_passed_to_our_generator': False})
        return {'status': 'raw_frames_extracted', 'protocol_sha256': 'a' * 64,
            'label_plan_sha256': self.freeze['label_plan']['sha256'], 'cases': rows, 'sampling_seeds': [101, 202, 303],
            'case_count_unchanged': True, 'replacement_cases_selected': False, 'post_extraction_integrity_errors': []}

    def projected(self, method, raw=None):
        with tempfile.TemporaryDirectory(dir=bridge.HERE, prefix='metadata_test_') as directory:
            path = Path(directory) / 'raw.json'
            bridge.save(path, self.raw() if raw is None else raw)
            return bridge.projection(self.freeze, 'a' * 64, path, bridge.sha(path), method)

    def test_native_operators_still_match_original_pins(self):
        bridge.NATIVE.verify_core_files()
        self.assertFalse(self.spec['algorithm_changes'])
        self.assertEqual(self.spec['methods']['magicface']['settings']['native_parameters']['steps'], 50)
        self.assertEqual(self.spec['methods']['faceadapter']['settings']['native_parameters']['steps'], 25)
        self.assertFalse(self.spec['methods']['liveportrait']['settings']['native_parameters']['flag_relative_motion'])

    def test_all_methods_keep_role_seed_grid_and_driver_boundary(self):
        for method, calls in [('magicface', 96), ('faceadapter', 96), ('liveportrait', 32)]:
            projected = self.projected(method)
            self.assertEqual(len(projected['cases']), calls)
            self.assertEqual(len(projected['logical_records']), 96)
            self.assertEqual(projected['case_ids'], self.freeze['case_ids'])
            self.assertTrue(all(r['split'] == bridge.ROLE for r in projected['cases']))
            label_map = {c['case_id']: c for c in self.labels}
            for row in projected['logical_records']:
                self.assertEqual(row['source_aus_0_5'], label_map[row['case_id']]['source_au_vector_0_5'])
                self.assertEqual(row['target_aus_0_5'], label_map[row['case_id']]['target_au_vector_0_5'])
            if method == 'magicface':
                self.assertTrue(all('target' not in r and 'target_sha256' not in r for r in projected['cases']))
            else:
                self.assertTrue(all(r['target_role'] == 'real_observed_driver' for r in projected['cases']))
            aliases = sum(r['alias_of'] is not None for r in projected['logical_records'])
            self.assertEqual(aliases, 64 if method == 'liveportrait' else 0)

    def test_changed_raw_au_under_same_case_id_is_rejected(self):
        raw = self.raw()
        raw['cases'][1]['generation_input']['target_au_vector_0_5'] = [3] * 12
        with self.assertRaises(ValueError):
            self.projected('magicface', raw)

    def test_changed_raw_subject_under_same_case_id_is_rejected(self):
        raw = self.raw()
        raw['cases'][1]['subject'] = 'SN999'
        with self.assertRaises(ValueError):
            self.projected('faceadapter', raw)

    def test_changed_driver_frame_or_video_hash_is_rejected(self):
        for field, value in [('label_frame_1based', 123), ('original_video_sha256', 'f' * 64), ('camera', 'wrong_camera')]:
            raw = self.raw()
            raw['cases'][1]['target_reference_image'][field] = value
            with self.assertRaises(ValueError):
                self.projected('faceadapter', raw)

    def test_raw_integrity_failure_is_rejected_for_all_methods(self):
        raw = self.raw()
        raw['status'] = 'integrity_failed'
        raw['post_extraction_integrity_errors'] = ['synthetic corruption']
        for method in bridge.METHODS:
            with self.assertRaises(ValueError):
                self.projected(method, raw)

    def test_raw_source_failure_retains_full_logical_grid(self):
        raw = self.raw()
        raw['status'] = 'completed_with_failures'
        subject = raw['cases'][0]['subject']
        for row in raw['cases']:
            if row['subject'] == subject:
                row['status'] = 'incomplete'
                row['generation_input']['source_image'].update(status='failed', sha256=None, error='synthetic source failure')
        projected = self.projected('liveportrait', raw)
        self.assertEqual(len(projected['logical_records']), 96)
        self.assertEqual(set(projected['case_ids']), set(self.freeze['case_ids']))
        failed = [r for r in projected['logical_records'] if r.get('status') == 'failed']
        self.assertEqual(len(failed), 12)

    def test_target_failure_does_not_block_magicface_source_au_request(self):
        raw = self.raw()
        raw['status'] = 'completed_with_failures'
        raw['cases'][1]['status'] = 'incomplete'
        raw['cases'][1]['target_reference_image'].update(status='failed', sha256=None, error='synthetic driver failure')
        projected = self.projected('magicface', raw)
        self.assertEqual(len(projected['logical_records']), 96)
        self.assertFalse(any(r.get('status') == 'failed' for r in projected['logical_records']))

    def test_prefailed_source_jobs_never_dispatch_workers(self):
        raw = self.raw(); raw['status'] = 'completed_with_failures'
        subject = raw['cases'][0]['subject']
        for row in raw['cases']:
            if row['subject'] == subject:
                row['status'] = 'incomplete'
                row['generation_input']['source_image'].update(status='failed', sha256=None)
        planned = self.projected('liveportrait', raw)
        with tempfile.TemporaryDirectory(dir=bridge.HERE, prefix='metadata_test_') as directory:
            root = Path(directory)
            args = SimpleNamespace(freeze=root / 'freeze.json', freeze_sha='a' * 64,
                raw_manifest=root / 'raw.json', raw_sha='b' * 64, method='liveportrait', execute=True,
                output=root / 'external_runs/run')
            with mock.patch.object(runner, 'COMPLETION', root), \
                    mock.patch.object(runner, 'frozen', return_value=self.freeze), \
                    mock.patch.object(runner, 'projection', return_value=planned), \
                    mock.patch.object(runner.NATIVE, 'validate_runtime', return_value={'assets': []}), \
                    mock.patch.object(runner.NATIVE, 'verify_lease'), \
                    mock.patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as worker, \
                    mock.patch('builtins.print'):
                runner.run(args)
            index = bridge.read(args.output / 'generation_index.json')
            self.assertEqual(worker.call_count, 28)
            self.assertEqual(index['planned_physical_executions'], 32)
            self.assertEqual(index['preflight_failed_physical_jobs'], 4)
            self.assertEqual(index['physical_attempts'], 28)
            self.assertEqual(index['actual_native_started'], 0)
            self.assertEqual(len(index['records']), 96)
            label_map = {c['case_id']: c for c in self.labels}
            for row in index['records']:
                self.assertEqual(row['source_aus_0_5'], label_map[row['case_id']]['source_au_vector_0_5'])
                self.assertEqual(row['target_aus_0_5'], label_map[row['case_id']]['target_au_vector_0_5'])

    def test_terminal_integrity_failure_quarantines_previously_complete_index(self):
        planned = self.projected('liveportrait')
        with tempfile.TemporaryDirectory(dir=bridge.HERE, prefix='metadata_test_') as directory:
            root = Path(directory)
            asset = root / 'asset.bin'
            asset.write_bytes(b'synthetic metadata test, not a model')
            original_sha = bridge.sha(asset)
            args = SimpleNamespace(freeze=root / 'freeze.json', freeze_sha='a' * 64,
                raw_manifest=root / 'raw.json', raw_sha='b' * 64, method='liveportrait', execute=True,
                output=root / 'external_runs/run')
            lease = {'assets': [{'path': str(asset), 'sha256': original_sha}]}
            calls = []
            def fake_worker(command, **kwargs):
                context = bridge.read(command[command.index('--context') + 1])
                output = Path(command[command.index('--output') + 1])
                output.mkdir(parents=True)
                fake_image = output / 'synthetic_output.bin'
                fake_image.write_bytes(b'not real image data')
                bridge.save(output / 'worker_result.json', {'execution_id': context['job']['execution_id'],
                    'status': 'complete', 'output': bridge.ref(fake_image), 'generation_started': True})
                calls.append(context['job']['execution_id'])
                return SimpleNamespace(returncode=0)
            def digest(path):
                if Path(path) == asset and len(calls) == len(planned['cases']):
                    return 'f' * 64
                return bridge.sha(path)
            with mock.patch.object(runner, 'COMPLETION', root), \
                    mock.patch.object(runner, 'frozen', return_value=self.freeze), \
                    mock.patch.object(runner, 'projection', return_value=planned), \
                    mock.patch.object(runner.NATIVE, 'validate_runtime', return_value=lease), \
                    mock.patch.object(runner.NATIVE, 'verify_lease'), \
                    mock.patch.object(runner.subprocess, 'run', side_effect=fake_worker), \
                    mock.patch.object(runner, 'sha', side_effect=digest), \
                    mock.patch('builtins.print'):
                try:
                    runner.run(args)
                except ValueError:
                    pass
            index = bridge.read(args.output / 'generation_index.json')
            self.assertNotEqual(index['status'], 'complete', 'Terminal asset mismatch left a usable complete index')
            self.assertFalse(index.get('usable_for_evaluation', True))
            self.assertFalse(any(r['status'] == 'complete' for r in index['records']))


if __name__ == '__main__':
    unittest.main()
