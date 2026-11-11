"""CPU-only synthetic tensors/geometry; no model assets or real pixels loaded."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
import common
import geometry
import run_grid as runner
import prepare_cache
import check_dev_parity


def load_fixture(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


V11 = common.module('v11')
FIXTURES = load_fixture('completion_terminal_fixtures', V11.V10 / 'test_terminal.py')
# Loading the old test file imports only synthetic test definitions.
import sys
sys.modules['run_local'] = V11
GEOMETRY_FIXTURES = load_fixture('completion_geometry_fixtures', V11.HERE / 'test_local.py')


def source_fixture(split='postfreeze_case_validation', changed=True):
    path = str(common.HERE / 'synthetic_tests' / 'source.png')
    points = GEOMETRY_FIXTURES.source_points()
    row = {f'{axis}_{i}': float(point[j]) for i, point in enumerate(points) for j, axis in enumerate(('x', 'y'))}
    target = [0] * 12
    if changed:
        target[0], target[8] = 1, 2
    case = {'case_id': 'synthetic_case', 'subject': 'synthetic', 'split': split, 'au_order': common.AUS,
            'source_au_vector_0_5': [0] * 12, 'target_au_vector_0_5': target}
    source = {'canonical_source': {'path': path, 'sha256': 'a' * 64, 'width': 512, 'height': 512, 'mode': 'RGB'}}
    measurement = {'case_id': case['case_id'], 'method': 'source_real', 'split': split, 'path': path,
        'sha256': 'a' * 64, 'image_size': [512, 512], 'returncode': 0, 'parse_status': 'primary_face_finite',
        'confidence': .95, 'selected_row': 0, 'raw_rows': [row]}
    return case, source, measurement


def trajectory(use_old, mask_value=.4, cleanup_value=.7):
    source = torch.linspace(-.3, .6, 4 * 8 * 8).reshape(1, 4, 8, 8)
    edit = torch.full((1, 1, 8, 8), mask_value)
    cleanup = torch.zeros_like(edit)
    if mask_value:
        cleanup[:, :, 3:5, 3:5] = cleanup_value
    cross = {'au_scale': 1.5, 'ip_scale': .9}
    text = torch.arange(12.).reshape(2, 2, 3)
    unet = FIXTURES.TinyUNet()
    scheduler = FIXTURES.TinyScheduler()
    scheduler.timesteps = torch.tensor([999, 949, 899, 849, 799] + common.TIMESTEPS)
    noise_calls = []
    def add_noise(src, noise, timestep):
        noise_calls.append(int(timestep[0]))
        return src + noise * timestep[0] / 1000
    scheduler.add_noise = add_noise
    pipe = SimpleNamespace(unet=unet, scheduler=scheduler, do_classifier_free_guidance=True,
        guidance_scale=7.5, cross_attention_kwargs=cross)
    noise = torch.ones_like(source)
    if use_old:
        controller = V11.LocalController(unet, 'local_lp_s050', source, edit, cross, torch, cleanup, {}, [])
        tree = ast.parse(V11.source)
        callback_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'step_end')
        history = []
        namespace = dict(torch=torch, need=common.need, strength=.75, controller=controller,
            src=source, mask=edit, noise=noise, history=history,
            project_latents=V11.v10.core['project_latents'], v5=SimpleNamespace(tensor_sha=common.tensor_sha))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(callback_node)], type_ignores=[])), '<old-v11-callback>', 'exec'), namespace)
        callback = namespace['step_end']
    else:
        controller = runner.fixed_controller(V11, unet, source, edit, cross, torch, cleanup)
        callback, history = runner.sampling_callback(controller, source, noise, edit, torch)
    latent = source.clone()
    try:
        for i, timestep in enumerate(common.TIMESTEPS):
            raw = unet(torch.cat([latent, latent]), timestep, encoder_hidden_states=text,
                timestep_cond=None, cross_attention_kwargs=cross, added_cond_kwargs=None)[0]
            next_latent = latent - V11.v10.cfg_epsilon(raw, latent, torch) * .001
            latent = callback(pipe, i, timestep, {'latents': next_latent, 'prompt_embeds': text})['latents']
        controller.assert_complete()
        receipt = controller.record()
    finally:
        controller.close()
    return latent, history, receipt, noise_calls, source, len(unet._forward_hooks)


class GeometryTests(unittest.TestCase):
    def test_source_parity_accepts_identical_pixels_with_different_png_encodings(self):
        parent = common.HERE / 'synthetic_tests'
        parent.mkdir(exist_ok=True)
        pixels = np.indices((512, 512)).sum(axis=0).astype(np.uint8)
        pixels = np.stack((pixels, np.roll(pixels, 1, axis=0), np.roll(pixels, 2, axis=1)), axis=-1)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            first, second = Path(directory) / 'uncompressed.png', Path(directory) / 'compressed.png'
            Image.fromarray(pixels).save(first, compress_level=0)
            Image.fromarray(pixels).save(second, compress_level=9)
            self.assertNotEqual(common.sha(first), common.sha(second))
            receipt = check_dev_parity.source_pixel_parity(second, common.ref(first))
            self.assertTrue(receipt['decoded_pixels_equal'])
            self.assertFalse(receipt['png_bytes_equal'])
            # The same gate must still reject a one-pixel change, not use a tolerance.
            changed = pixels.copy()
            changed[0, 0, 0] ^= 1
            third = Path(directory) / 'changed.png'
            Image.fromarray(changed).save(third, compress_level=9)
            with self.assertRaisesRegex(ValueError, 'source pixels'):
                check_dev_parity.source_pixel_parity(third, common.ref(first))

    def test_new_geometry_matches_old_v7_and_v11_exactly(self):
        case, source, measurement = source_fixture('dev')
        edit, cleanup, receipt = geometry.build_masks(case, source, measurement)
        old_case = {**case, 'source_real': source['canonical_source'],
            'source_aus_0_5': case['source_au_vector_0_5'], 'target_aus_0_5': case['target_au_vector_0_5']}
        old_edit, _ = geometry.V7.build_mask(old_case, measurement)
        old_cleanup, _, _ = geometry.V11.build_mask(old_case, measurement, old_edit)
        np.testing.assert_array_equal(edit, old_edit)
        np.testing.assert_array_equal(cleanup, old_cleanup)
        self.assertFalse(receipt['target_geometry_used'])

    def test_validation_split_is_not_disguised_dev(self):
        case, source, measurement = source_fixture()
        _, _, receipt = geometry.build_masks(case, source, measurement)
        self.assertEqual(receipt['split'], 'postfreeze_case_validation')

    def test_zeroedit_empty_both_masks(self):
        case, source, measurement = source_fixture(changed=False)
        edit, cleanup, _ = geometry.build_masks(case, source, measurement)
        self.assertFalse(edit.any())
        self.assertFalse(cleanup.any())

    def test_bad_or_target_geometry_rejected(self):
        for field, value in (('method', 'target_real'), ('sha256', 'b' * 64), ('confidence', .1), ('selected_row', -1)):
            case, source, measurement = source_fixture()
            measurement[field] = value
            with self.assertRaises(ValueError):
                geometry.build_masks(case, source, measurement)


class ControllerTests(unittest.TestCase):
    def test_full_trajectory_matches_old_v11_bytewise(self):
        old, old_history, old_receipt, *_ = trajectory(True)
        new, history, receipt, calls, _, hooks = trajectory(False)
        self.assertTrue(torch.equal(old, new))
        self.assertEqual(old_history, history)
        for key in ('input_latent_sha256', 'epsilon_cfg_sha256', 'base_terminal_clean_latent_sha256', 'post_filter_clean_latent_sha256', 'projected_latent_sha256'):
            self.assertEqual(old_receipt['terminal_trace'][key], receipt['terminal_trace'][key])
        self.assertEqual(calls, common.TIMESTEPS[1:])
        self.assertEqual(hooks, 0)

    def test_actual_unet_hooks_report_15_zero_15(self):
        _, history, record, _, _, hooks = trajectory(False)
        self.assertEqual(len(history), 15)
        self.assertEqual((record['deis_unet_calls'], record['terminal_unet_calls'], record['total_unet_calls']), (15, 0, 15))
        self.assertEqual([r['t'] for r in record['unet_call_trace']], common.TIMESTEPS)
        self.assertEqual(hooks, 0)

    def test_zeroedit_returns_exact_source(self):
        latent, history, record, _, source, _ = trajectory(False, 0)
        self.assertTrue(torch.equal(latent, source))
        self.assertEqual(history[-1]['latent_sha256'], common.tensor_sha(source))

    def test_callback_rejects_wrong_actual_schedule(self):
        controller = SimpleNamespace()
        src = torch.zeros(1, 4, 8, 8)
        callback, _ = runner.sampling_callback(controller, src, src, src[:, :1], torch)
        pipe = SimpleNamespace(scheduler=SimpleNamespace(timesteps=torch.arange(20)))
        with self.assertRaises(ValueError):
            callback(pipe, 0, 749, {'latents': src})

    def test_base_conditions_equal_direct_backbone(self):
        class Backbone:
            def baseline_conditions(self, aus, identity):
                return torch.cat((aus, aus * 2), -1), identity[:, None].repeat(1, 16, 1)
        backbone = Backbone()
        source = {'base_faceid': torch.arange(24.).float()}
        condition = {'target_aus': torch.arange(12.).float() / 3}
        result = runner.conditions_for_variant(backbone, None, None, source, condition, None, 'cpu', torch)
        positive = backbone.baseline_conditions(condition['target_aus'][None], source['base_faceid'][None])
        negative = backbone.baseline_conditions(torch.zeros(1, 12), torch.zeros(1, 24))
        self.assertTrue(torch.equal(result['au_embedding'], torch.cat((negative[0], positive[0]))))
        self.assertTrue(torch.equal(result['image_embedding'], torch.cat((negative[1], positive[1]))))
        self.assertEqual((result['au_scale'], result['ip_scale']), (1.5, .9))


class AccountingTests(unittest.TestCase):
    def test_saved_output_followed_by_zeroedit_gate_failure_is_not_evaluable(self):
        case, source, _ = source_fixture(changed=False)
        case.update(source_id='s', condition_status='complete')
        source.update(source_id='s', status='complete')
        class Runtime:
            def prepare(self, *args):
                return {'reconstruction': {'sha256': 'incorrect reconstruction SHA'}}
            def sample(self, *args):
                return Image.new('RGB', (512, 512)), {'total_unet_calls': 15}
        parent = common.HERE / 'synthetic_tests'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory, patch('builtins.print'):
            rows = runner.execute_grid({'cases': [case], 'sources': [source]}, {}, Path(directory), Runtime(), {case['case_id']: ()})
            self.assertEqual(len(rows), 15)
            for row in rows:
                self.assertEqual(row['status'], 'failed')
                self.assertIsNone(row['output'])
                self.assertIsNone(row['sha256'])
                artifact = row['saved_output_unverified']
                self.assertFalse(artifact['eligible_for_evaluation'])
                self.assertEqual(common.sha(artifact['path']), artifact['sha256'])

    def test_raw_sources_bind_subject_camera_frame_video_and_integrity(self):
        labels, rows = [], []
        for i in range(8):
            label = {'case_id': 'case' + str(i), 'subject': 'subject' + str(i), 'split': 'postfreeze_case_validation',
                'camera': 'right', 'video_path': '/synthetic/video' + str(i), 'source_frame_1based': 12,
                'source_au_vector_0_5': [0] * 12, 'target_au_vector_0_5': [1] + [0] * 11}
            source = {'subject': label['subject'], 'split': label['split'], 'camera': 'right', 'label_frame_1based': 12,
                'video_frame_zero_based': 11, 'original_video_path': label['video_path'], 'original_video_sha256': 'a' * 64}
            row = {k: label[k] for k in ('case_id', 'subject', 'split', 'camera')}
            row['generation_input'] = {'source_image': source, 'source_au_vector_0_5': label['source_au_vector_0_5'], 'target_au_vector_0_5': label['target_au_vector_0_5']}
            labels.append(label)
            rows.append(row)
        raw = {'protocol_sha256': 'f' * 64, 'status': 'raw_frames_extracted', 'cases': rows}
        sources, cases = prepare_cache.raw_source_plan(labels, raw, 'f' * 64)
        self.assertEqual(len(sources), 8)
        for change in ('integrity', 'subject', 'camera', 'frame', 'video'):
            bad = copy.deepcopy(raw)
            if change == 'integrity':
                bad['status'] = 'integrity_failed'
            else:
                key = {'subject': 'subject', 'camera': 'camera', 'frame': 'video_frame_zero_based', 'video': 'original_video_path'}[change]
                bad['cases'][0]['generation_input']['source_image'][key] = 'wrong'
            with self.assertRaises(ValueError):
                prepare_cache.raw_source_plan(labels, bad, 'f' * 64)

    def test_all_failure_rows_retained_for_all_variants_and_seeds(self):
        case, source, _ = source_fixture()
        case.update(source_id='s', condition_status='failed')
        source.update(source_id='s', status='failed')
        cache = {'cases': [case], 'sources': [source]}
        parent = common.HERE / 'synthetic_tests'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            with patch('builtins.print'):
                rows = runner.execute_grid(cache, {}, Path(directory), None, {})
            self.assertEqual(len(rows), 15)
            self.assertEqual(len({r['execution_id'] for r in rows}), 15)
            self.assertTrue(all(r['status'] == 'not_started' for r in rows))
            self.assertEqual({r['variant'] for r in rows}, set(common.VARIANTS))
            self.assertEqual({r['seed'] for r in rows}, {101, 202, 303})

    def test_failed_generation_keeps_trace_and_failed_image_not_success(self):
        case, source, _ = source_fixture(changed=False)
        case.update(source_id='s', condition_status='complete')
        source.update(source_id='s', status='complete')
        class Broken:
            def prepare(self, *args):
                return {}
            def sample(self, *args):
                exc = ValueError('synthetic failure')
                exc.generation_receipt = {'total_unet_calls': 3}
                raise exc
        parent = common.HERE / 'synthetic_tests'
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            with patch('builtins.print'):
                rows = runner.execute_grid({'cases': [case], 'sources': [source]}, {}, Path(directory), Broken(), {case['case_id']: ()})
            self.assertEqual(len(rows), 15)
            self.assertTrue(all(r['status'] == 'failed' and r['total_unet_calls'] == 3 for r in rows))

    def test_fixed_policy_has_all_variants_early200_mps_and_three_seeds(self):
        self.assertEqual(common.POLICY['checkpoint_step'], 200)
        self.assertEqual(common.POLICY['source_device'], 'mps')
        self.assertEqual(common.SEEDS, [101, 202, 303])
        self.assertEqual(set(common.CKPTS), set(common.VARIANTS) - {'base'})
        self.assertEqual(common.POLICY['cleanup_sigma'], .5)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
