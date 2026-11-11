"""Pure fixture tests: no real test videos/images/model bytes are read."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluation_contract
import freeze_protocol as f


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else f.encode(value)
    path.write_bytes(data)
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


class FreezeTests(unittest.TestCase):
    def setUp(self):
        parent = f.HERE / 'freeze_preparation/fixtures'
        parent.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.tmp.name)
        self.final = self.root / 'final_test'
        self.draft = self.final / 'draft_configuration_20260915'
        self.draft.mkdir(parents=True)
        # The only production inputs used to manufacture synthetic fixtures are
        # small, accepted JSON metadata. No model/video/image is read here.
        original = [f.read(f.HERE / 'draft_configuration_20260915' / name) for name in
                    ('protocol.DRAFT.json', 'sampling.DRAFT.json', 'evaluation.DRAFT.json')]
        p, s, e = copy.deepcopy(original)
        meta = {key: f.read(p['inputs'][key]['path']) for key in
                ('label_plan', 'label_inventory', 'subject_split', 'label_acceptance')}
        self.videos = []
        for video in p['videos']:
            old = video['path']
            new = self.root / 'raw_videos' / (video['subject'] + '.avi')
            put(new, b'NOT A VIDEO - synthetic bytes only: ' + video['subject'].encode())
            self.videos.append(new)
            video['path'] = str(new)
            for case in meta['label_plan']['cases']:
                if case['video_path'] == old:
                    case['video_path'] = str(new)
        checkpoints = {}
        mapping = {}
        for i, method in enumerate(f.METHODS[:4]):
            original_ref = s['method_settings'][method]['runtime']['adapter_checkpoint']
            checkpoint = self.root / 'runs' / method / 'checkpoint_00000200.pt'
            ckpt_ref = put(checkpoint, b'FAKE MODEL BYTES ' + method.encode())
            checkpoints[method] = ckpt_ref['sha256']
            source = put(self.root / 'fixture_sources' / (method + '.py'), b'# dummy source\n')
            bindings = {'sources': {str(Path(source['path']).relative_to(self.root)): source['sha256']}}
            put(checkpoint.parent / 'bindings.json', bindings)
            side = put(checkpoint.with_suffix('.json'), {'step': 200, 'variant': method,
                       'checkpoint_sha256': ckpt_ref['sha256'], 'bindings_fingerprint': f.fingerprint(bindings)})
            mapping[original_ref['path']] = ckpt_ref
            mapping[str(Path(original_ref['path']).with_suffix('.json'))] = side
        # All other references become inert fixture bytes, then known JSON seals
        # and source constants are populated below.
        for obj in (p['models'], p['execution_sources'], s, e):
            for ref in f.refs(obj):
                old = ref['path']
                if old not in mapping:
                    mapping[old] = put(self.root / 'assets' / (str(len(mapping)) + '.bin'), b'fixture asset ' + old.encode())
                ref.update(mapping[old])
        policy = {'openface': {}, 'identity_primary': {}}
        for tool, key in (('openface', 'openface'), ('facenet', 'identity_primary')):
            folder = self.root / 'tools' / tool
            asset = put(folder / 'fixture.bin', b'fake evaluator ' + tool.encode())
            seal = put(folder / 'tool_seal.json', {'source_files': [{'path': 'fixture.bin', 'sha256': asset['sha256']}]})
            policy[key]['tool_seal_sha256'] = seal['sha256']
        e['measurement_policy'] = put(self.root / 'measurement_policy.json', policy)
        self.policy_sha = e['measurement_policy']['sha256']
        source_ref = p['execution_sources']['final_evaluation_measurements']
        core = put(self.root / 'development_evaluation/evaluation_policy_v2/core.py', b'# fake core\n')
        source_ref.update(put(Path(source_ref['path']), ('CORES=' + repr({'test': ('core.py', core['sha256'])})).encode()))
        source_ref = p['execution_sources']['external_native']
        mf = put(self.root / 'external_baselines/run_magicface_smoke.py', b'# fake magicface\n')
        mfc = put(self.root / 'external_baselines/magicface_contract.py', b'# fake contract\n')
        source_ref.update(put(Path(source_ref['path']),
                          ('CORE_HASHES={}\nMF_CORE_SHA=' + repr(mf['sha256']) + '\nMF_CONTRACT_SHA=' + repr(mfc['sha256'])).encode()))
        seal = s['method_settings']['magicface']['runtime']['model_layout_seal']
        seal.update(put(Path(seal['path']), {'models': [], 'auxiliary_sources': []}))
        for obj in (p['models'], p['execution_sources'], s, e):
            for ref in f.refs(obj):
                ref['sha256'] = hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()
        for name, key in (('extract_final_test_frames.py', 'extractor_sha256'), ('prepare_disfa_pilot.py', 'legacy_helper_sha256')):
            p[key] = put(self.root / 'data_prepare' / name, b'# fixture extraction only\n')['sha256']
        inputs = {}
        for key in ('label_plan', 'label_inventory', 'subject_split'):
            p['inputs'][key] = put(self.root / 'metadata' / (key + '.json'), meta[key])
            inputs[key] = p['inputs'][key]['sha256']
        meta['label_acceptance']['plan_sha256'] = inputs['label_plan']
        p['inputs']['label_acceptance'] = put(self.root / 'metadata/acceptance.json', meta['label_acceptance'])
        self.p, self.s, self.e = p, s, e
        self.contract = f.Contract(self.root, self.root, self.root / 'raw_videos', {}, inputs, checkpoints)
        self.reseal()
        evidence = put(self.root / 'development_evaluation/comparison/results.json', {'fixture': True})
        self.decision = self.final / 'selection_decision.json'
        put(self.decision, {'kind': 'final_test_selection_decision', 'status': 'selected_on_development_only',
            'selected_num_inference_steps': 20, 'selected_method': 'bidirectional',
            'selected_checkpoint_step': 200, 'selected_ip_scale': .9, 'test_pixels_read': False,
            'test_video_full_content_read': False, 'reason': 'Synthetic test only', 'evidence': [evidence]})

    def tearDown(self):
        self.tmp.cleanup()

    def reseal(self):
        p, s, e = self.p, self.s, self.e
        p['inputs']['sampling_settings'] = put(self.draft / 'sampling.DRAFT.json', s)
        p['inputs']['evaluation_policy'] = put(self.draft / 'evaluation.DRAFT.json', e)
        put(self.draft / 'protocol.DRAFT.json', p)
        self.contract = replace(self.contract, draft_shas={name: hashlib.sha256((self.draft / name).read_bytes()).hexdigest()
            for name in ('protocol.DRAFT.json', 'sampling.DRAFT.json', 'evaluation.DRAFT.json')})

    def run_fixture(self, *, execute=False, steps=20):
        with patch.object(evaluation_contract, 'POLICY_SHA', self.policy_sha):
            return f.run(steps, self.decision, execute=execute, contract=self.contract)

    def assert_not_frozen(self):
        self.assertFalse((self.final / 'frozen_protocol.json').exists())

    def test_preview_does_not_open_any_video_or_write_outputs(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            self.assertNotIn(path.resolve(), self.videos, 'Preview opened raw video')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            report = self.run_fixture()
        self.assertEqual(report['status'], 'ready_not_frozen')
        self.assert_not_frozen()
        self.assertFalse((self.final / 'frozen_sampling.json').exists())

    def test_execute20_retains_data_and_exclusive_second_run(self):
        before = {n: (self.draft / n).read_bytes() for n in self.contract.draft_shas}
        report = self.run_fixture(execute=True)
        self.assertEqual(report['status'], 'frozen_final_test')
        protocol = f.read(self.final / 'frozen_protocol.json')
        sampling = f.read(self.final / 'frozen_sampling.json')
        evaluation = f.read(self.final / 'frozen_evaluation.json')
        self.assertEqual(protocol['freeze_receipt']['video_full_hash_count'], 8)
        self.assertEqual(protocol['models'], self.p['models'])
        self.assertEqual(protocol['case_ids'], self.p['case_ids'])
        for method in f.METHODS:
            expected = copy.deepcopy(self.s['method_settings'][method])
            if method in f.METHODS[:5]:
                expected['native_parameters']['num_inference_steps'] = 20
            self.assertEqual(sampling['method_settings'][method], expected)
        old_e = copy.deepcopy(self.e)
        old_e.update(kind='final_test_evaluation_policy', status='frozen_for_final_test')
        self.assertEqual(evaluation, old_e)
        for key in ('sampling_settings', 'evaluation_policy'):
            ref = protocol['inputs'][key]
            self.assertEqual(hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest(), ref['sha256'])
        for name, data in before.items():
            self.assertEqual((self.draft / name).read_bytes(), data)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_fixture(execute=True)

    def test_execute50_only_after_matching_decision(self):
        with self.assertRaises(ValueError):
            self.run_fixture(steps=50)
        decision = f.read(self.decision)
        decision['selected_num_inference_steps'] = 50
        put(self.decision, decision)
        self.run_fixture(execute=True, steps=50)
        sampling = f.read(self.final / 'frozen_sampling.json')
        self.assertEqual([sampling['method_settings'][m]['native_parameters']['num_inference_steps'] for m in f.METHODS[:5]], [50] * 5)

    def test_invalid_steps_rejected(self):
        for steps in (None, True, 0, 10, 21, '20'):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.run_fixture(steps=steps)
        self.assert_not_frozen()

    def test_missing_decision(self):
        self.decision.unlink()
        with self.assertRaisesRegex(ValueError, 'selection_decision'):
            self.run_fixture()

    def test_draft_bytes_changed_rejected(self):
        (self.draft / 'sampling.DRAFT.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            self.run_fixture()
        self.assert_not_frozen()

    def test_exact_invariants_even_if_fixture_draft_resigned(self):
        original = copy.deepcopy((self.p, self.s, self.e))
        changes = [lambda: self.p['case_ids'].reverse(), lambda: self.p['sampling_seeds'].reverse(),
            lambda: self.p['models'].reverse(),
            lambda: self.s['method_settings']['mlp']['native_parameters'].update(ip_scale=1),
            lambda: self.s['method_settings']['public_baseline']['native_parameters'].update(num_inference_steps=20),
            lambda: self.s['method_settings']['magicface']['native_parameters'].update(steps=20),
            lambda: self.s['method_settings']['faceadapter']['native_parameters'].update(steps=50),
            lambda: self.s['method_settings']['liveportrait']['seed_execution'].update(actual_seed=202),
            lambda: self.e['comparisons'][0].update(analysis_role='descriptive'),
            lambda: self.e['comparisons'][7].update(metric='delta_mae_changed__all'),
            lambda: self.p['draft_pending'].update(extra='unknown')]
        for index, change in enumerate(changes):
            self.p, self.s, self.e = copy.deepcopy(original)
            change()
            self.reseal()
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.run_fixture()
            self.assert_not_frozen()

    def test_model_and_code_hash_tamper(self):
        for obj in (self.p['models'][0]['artifacts'][0], self.p['execution_sources']['source_cache']):
            path = Path(obj['path'])
            original = path.read_bytes()
            path.write_bytes(b'TAMPERED')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                self.run_fixture()
            path.write_bytes(original)
        self.assert_not_frozen()

    def test_no_raw_video_disguised_as_model_reference(self):
        ref = self.p['models'][0]['artifacts'][-1]
        ref['path'] = str(self.videos[0])
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'Raw dataset'):
            self.run_fixture()
        self.assert_not_frozen()

    def test_partial_publication_and_lock_do_not_publish_protocol(self):
        original = f.exclusive_write
        def fail_second(path, data):
            if path.name == 'frozen_evaluation.json':
                raise OSError('simulated disk failure')
            original(path, data)
        with patch.object(f, 'exclusive_write', fail_second), self.assertRaises(OSError):
            self.run_fixture(execute=True)
        self.assert_not_frozen()
        self.assertTrue((self.final / 'frozen_sampling.json').exists())
        self.assertFalse((self.final / '.freeze_protocol.lock').exists())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_fixture(execute=True)

    def test_existing_lock_rejected_before_video_read(self):
        put(self.final / '.freeze_protocol.lock', b'other process')
        with self.assertRaises(FileExistsError):
            self.run_fixture(execute=True)
        self.assertEqual((self.final / '.freeze_protocol.lock').read_bytes(), b'other process')
        self.assert_not_frozen()

    def test_publication_order_and_shared_gate_compatibility(self):
        order = []
        original = f.exclusive_write
        def record(path, data):
            order.append(path.name)
            original(path, data)
        with patch.object(f, 'exclusive_write', record):
            self.run_fixture(execute=True)
        self.assertEqual(order, ['.freeze_protocol.lock', 'frozen_sampling.json', 'frozen_evaluation.json', 'frozen_protocol.json'])
        from protocol_gate import validate_control
        from extract_final_test_frames import Contract as ExtractionContract
        p = f.read(self.final / 'frozen_protocol.json')
        plan = f.read(p['inputs']['label_plan']['path'])
        videos = {v['subject']: v for v in p['videos']}
        cases = []
        for row in plan['cases']:
            def image(side):
                return {'path': str(self.root / 'images_never_created' / (row['case_id'] + side + '.png')),
                    'status': 'complete', 'sha256': 'a' * 64, 'camera': row['camera'],
                    'label_frame_1based': row[side + '_frame_1based'],
                    'video_frame_zero_based': row[side + '_frame_zero_based'],
                    'original_video_sha256': videos[row['subject']]['sha256']}
            cases.append({'case_id': row['case_id'], 'subject': row['subject'], 'camera': row['camera'],
                'split': 'test', 'status': 'complete', 'target_pixels_passed_to_our_generator': False,
                'generation_input': {'source_image': image('source'),
                    'source_au_vector_0_5': row['source_au_vector_0_5'], 'target_au_vector_0_5': row['target_au_vector_0_5']},
                'target_reference_image': image('target')})
        raw = put(self.final / 'synthetic_raw.json', {'status': 'raw_frames_extracted',
            'protocol_sha256': hashlib.sha256((self.final / 'frozen_protocol.json').read_bytes()).hexdigest(),
            'label_plan_sha256': self.contract.input_shas['label_plan'], 'extractor_sha256': p['extractor_sha256'],
            'legacy_helper_sha256': p['legacy_helper_sha256'], 'cases': cases})
        contract = ExtractionContract(self.root, self.root / 'unused_labels', self.contract.video_root,
            f.SUBJECTS, 32, self.contract.input_shas['label_plan'], self.contract.input_shas['label_inventory'], self.contract.input_shas['subject_split'])
        result = validate_control(self.final / 'frozen_protocol.json', raw['path'], raw['sha256'], contract=contract)
        self.assertFalse(result['pixels_opened_by_gate'])
        self.assertEqual(result['case_ids'], p['case_ids'])


if __name__ == '__main__':
    unittest.main()
