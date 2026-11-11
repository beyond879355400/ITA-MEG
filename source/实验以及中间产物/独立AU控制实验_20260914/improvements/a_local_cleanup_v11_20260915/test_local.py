"""CPU synthetic tests only: generic source geometry, filter algebra, actual hooks."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import numpy as np
import torch
import run_local as runner

geometry = runner.geometry
spec = importlib.util.spec_from_file_location('v11_v10_cpu_fixtures', runner.V10 / 'test_terminal.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def source_points():
    p = np.zeros((68, 2), dtype=np.float64)
    theta = np.linspace(np.pi, 0, 17)
    p[:17] = np.stack((256 + 156 * np.cos(theta), 180 + 290 * np.sin(theta)), axis=1)
    p[17:27] = [[150,185],[173,174],[196,174],[219,181],[235,191],
                 [275,191],[291,181],[315,174],[339,174],[360,185]]
    p[27:36] = [[256,221],[256,245],[256,270],[256,291],[232,299],[244,305],[256,308],[268,305],[280,299]]
    p[36:48] = [[170,220],[185,210],[202,210],[216,220],[202,230],[185,230],
                 [296,220],[310,210],[327,210],[342,220],[327,230],[310,230]]
    p[48:60] = [[209,350],[229,341],[244,340],[256,342],[268,340],[283,341],
                 [303,350],[285,365],[267,371],[256,373],[242,371],[225,365]]
    p[60:68] = [[223,351],[242,347],[256,349],[270,347],[289,351],[272,359],[256,362],[240,359]]
    return p


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.points = source_points()
        self.scale = np.linalg.norm(self.points[16] - self.points[0])
        self.edit = np.ones((64, 64), dtype=np.float32)

    def test_only_explicit_au_triggers_and_noedit_empty(self):
        for changed, regions in (([], []), ([12,25,26], []), ([1], ['glabella']),
                                  ([2,4], ['glabella']), ([17], ['chin']), ([1,17], ['glabella','chin'])):
            mask, receipt, _ = geometry.geometry_mask(self.points, self.scale, changed, self.edit)
            self.assertEqual(receipt['active_regions'], regions)
            self.assertEqual(bool(mask.any()), bool(regions))
            self.assertEqual(mask.dtype, np.float32)
            self.assertEqual(mask.shape, (64, 64))

    def test_geometry_is_translation_equivariant_on_latent_grid(self):
        a, _, _ = geometry.geometry_mask(self.points, self.scale, [1,17], self.edit)
        b, _, _ = geometry.geometry_mask(self.points + [8,0], self.scale, [1,17], self.edit)
        np.testing.assert_array_equal(b, np.roll(a, 1, axis=1))

    def test_edit_support_intersection_and_protection(self):
        edit = self.edit.copy(); edit[:, :32] = 0
        mask, receipt, protected = geometry.geometry_mask(self.points, self.scale, [1,17], edit)
        self.assertEqual(np.count_nonzero(mask[:, :32]), 0)
        self.assertEqual(receipt['raster_protected_pixels_overlap'], 0)
        for index in (19, 24, 37, 44, 51, 57):
            x, y = self.points[index].astype(int)
            self.assertTrue(protected[y, x])

    def test_fixed_bounds_and_cropping_receipt(self):
        points = self.points.copy(); points[8, 1] = 550
        mask, receipt, _ = geometry.geometry_mask(points, self.scale, [17], self.edit)
        self.assertIn(8, receipt['source_landmarks_outside_frame'])
        self.assertTrue(receipt['regions']['chin']['source_support_clipped_by_frame'])
        self.assertTrue(mask.any())
        bounds = receipt['regions']['chin']['local_box_unclipped']
        origin_y = receipt['frame_origin_512'][1]
        self.assertAlmostEqual(bounds[3], 550 - origin_y - .035 * self.scale)

    def test_rejects_invalid_mask_or_geometry(self):
        for mask in (self.edit.astype(np.float64), self.edit * np.nan, self.edit * -1):
            with self.assertRaises(ValueError):
                geometry.geometry_mask(self.points, self.scale, [1], mask)
        with self.assertRaises(ValueError):
            geometry.geometry_mask(self.points * np.nan, self.scale, [1], self.edit)


class LocalFilterTests(unittest.TestCase):
    def setUp(self):
        self.src = torch.linspace(-.7, .6, 4 * 12 * 12).reshape(1, 4, 12, 12)
        self.edit = torch.linspace(.1, .9, 144).reshape(1, 1, 12, 12)
        self.local = torch.zeros_like(self.edit); self.local[:, :, 4:8, 4:8] = .7

    def test_constant_delta_normalization_and_outside_exact(self):
        for sigma in (.5, .75):
            clean = self.src + 3
            result = runner.local_filtered_clean(clean, self.src, self.edit, self.local, sigma, torch)
            torch.testing.assert_close(result, clean, atol=5e-7, rtol=1e-6)
            outside = (self.local == 0).expand_as(clean)
            self.assertTrue(torch.equal(result[outside], clean[outside]))

    def test_cleanup_zero_keeps_clean_not_source(self):
        clean = self.src + 3
        result = runner.local_filtered_clean(clean, self.src, self.edit, self.local * 0, .5, torch)
        self.assertTrue(torch.equal(result, clean))

    def test_only_local_interpolation_and_final_editmask_once(self):
        clean = self.src.clone(); clean[:, :, 6, 6] += 2
        full = runner.local_filtered_clean(clean, self.src, self.edit, torch.ones_like(self.local), .75, torch)
        actual = runner.local_filtered_clean(clean, self.src, self.edit, self.local, .75, torch)
        expected = torch.where(self.local > 0, clean + self.local * (full - clean), clean)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
        projected = runner.core['project_latents'](actual, self.src, torch.ones_like(self.src), self.edit, None, None, torch)
        torch.testing.assert_close(projected, self.edit * actual + (1 - self.edit) * self.src, rtol=0, atol=0)

    def test_denominator_underflow_falls_back_to_original_delta(self):
        clean = self.src + torch.randn_like(self.src)
        actual = runner.local_filtered_clean(clean, self.src, self.edit * 1e-12, self.local, .5, torch)
        self.assertTrue(torch.equal(actual, clean))

    def test_values_outside_editmask_are_not_filtered_in(self):
        edit = torch.zeros_like(self.edit); edit[:, :, 3:9, 3:9] = .5
        clean = torch.where(edit > 0, self.src + 2, self.src + 10000)
        actual = runner.local_filtered_clean(clean, self.src, edit, self.local, .5, torch)
        local = (self.local > 0).expand_as(clean)
        torch.testing.assert_close(actual[local], (self.src + 2)[local], atol=5e-7, rtol=1e-6)

    def test_nonfinite_support_outside_edit_and_fixed_sigma_rejected(self):
        with self.assertRaises(ValueError):
            runner.local_filtered_clean(self.src + float('nan'), self.src, self.edit, self.local, .5, torch)
        with self.assertRaises(ValueError):
            runner.local_filtered_clean(self.src, self.src, self.edit * 0, self.local, .5, torch)
        with self.assertRaises(ValueError):
            runner.local_filtered_clean(self.src, self.src, self.edit, self.local, 1., torch)

    def test_real_controller_inherits15call_last_eps_without_extra_forward(self):
        for mode in runner.MODES:
            src, mask, text, pipe, old = fixtures.fixture('terminal_last_eps', .5)
            old.close()
            local = torch.zeros_like(mask); local[:, :, 2:6, 2:6] = .7
            ref = {'path': 'synthetic', 'sha256': 'synthetic'}
            control = runner.LocalController(pipe.unet, mode, src, mask, pipe.cross_attention_kwargs, torch, local, ref, [])
            try:
                for t in runner.v10.TIMESTEPS:
                    pipe.unet(torch.cat((src, src)), t, encoder_hidden_states=text, timestep_cond=None,
                              cross_attention_kwargs=pipe.cross_attention_kwargs, added_cond_kwargs=None)
                last = control.last_raw.clone()
                corrected = control.correct(src + .1, pipe, text)
                control.finalize_projection(corrected); control.assert_complete()
                record = control.record()
                self.assertEqual((record['deis_unet_calls'], record['terminal_unet_calls'], record['total_unet_calls']), (15,0,15))
                self.assertEqual(record['terminal_trace']['mode'], mode)
                self.assertEqual(record['terminal_trace']['epsilon_source'], 'last_t50')
                self.assertTrue(torch.equal(last, control.last_raw))
                self.assertEqual(pipe.scheduler.step_index, 20)
            finally:
                control.close()
            self.assertEqual(len(pipe.unet._forward_hooks), 0)

    def test_actual_derived_kernel_scope_and_gate(self):
        tree = ast.parse(runner.source)
        original_tree = ast.parse(runner.v10.source)
        get = lambda tr: next(node for node in tr.body if isinstance(node, ast.FunctionDef) and node.name == 'project_latents')
        self.assertEqual(ast.dump(get(tree)), ast.dump(get(original_tree)))
        self.assertEqual(runner.core['CONFIGS'], [(mode,.75,1.5) for mode in runner.MODES])
        self.assertIn('if len(results)==4:', runner.source)
        self.assertIn("'planned':32", runner.source)
        self.assertIn('controller.a_prefix_verified=True', runner.source)
        self.assertIn('finally:\n                        controller.close()', runner.source)
        self.assertNotIn('scheduler.step(', Path(runner.__file__).read_text())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
