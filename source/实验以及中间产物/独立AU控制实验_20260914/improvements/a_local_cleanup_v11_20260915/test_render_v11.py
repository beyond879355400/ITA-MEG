"""Synthetic CPU layouts plus frozen metadata; no experiment image pixels opened."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v11_renderer_tested', HERE / 'render_v11.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
R = renderer.R


class RenderTests(unittest.TestCase):
    def test_complete_uniform_columns_and_32new_budget(self):
        overview = renderer.columns(True)
        self.assertEqual([c['key'] for c in overview], ['source_real', 'target_real', 'a_bidirectional',
            'v10_global_lp_bidirectional', 'bidirectional_local_lp_s050', 'bidirectional_local_lp_s075'])
        cols = renderer.columns()
        self.assertEqual(len(cols), 10)
        self.assertEqual(len({c['key'] for c in cols}), 10)
        self.assertEqual(sum(c['kind'] == 'v11' for c in cols) * 8, 32)
        self.assertEqual(sum(c['kind'] == 'v10_lp' for c in cols) * 8, 16)
        self.assertEqual([c['sigma'] for c in cols if c['kind'] == 'v11'], [.5, .5, .75, .75])
        self.assertIs(R.load_inputs, renderer.load_inputs)

    def test_exact_v9_source_roi_policy_and_padding_retained(self):
        self.assertEqual(R.ROI_POLICY['SN012']['indices'], list(range(5, 12)) + list(range(48, 68)))
        self.assertEqual(R.ROI_POLICY['SN018']['indices'], list(range(17, 28)))
        self.assertEqual(R.ROI_PADDING, 20)
        points = [[100.2, 120.1] for _ in range(68)]
        points[5], points[67] = [5.1, 40.2], [500.6, 502.3]
        self.assertEqual(R.roi_bounds(points, R.ROI_POLICY['SN012']['indices']),
                         ([-15, 20, 521, 523], [0, 20, 512, 512]))

    def test_local_filter_and_extra_call_mislabel_rejected(self):
        for col in [c for c in renderer.columns() if c['kind'] == 'v11']:
            alpha = 1 / math.sqrt(1 + .029 ** 2)
            row = {'status': 'complete', 'config': col['config'], 'seed': 101, 'strength': .75, 'au_scale': 1.5,
                'ip_scale': .9, 'nominal_steps': 20, 'cleanup_sigma': col['sigma'], 'actual_steps': 15,
                'denoising_steps': 15, 'deis_unet_calls': 15, 'terminal_unet_calls': 0, 'total_unet_calls': 15,
                'actual_timesteps': renderer.TIMESTEPS, 'projection_trace': [{'latent_sha256': 'final'}],
                'unet_call_trace': [{'phase': 'deis', 't': t, 'cfg_batch': 2, 'au_scale': 1.5} for t in renderer.TIMESTEPS],
                'terminal_trace': {'mode': col['config'], 'base_mode': 'terminal_last_eps', 'last_deis_timestep': 50,
                    'extra_cfg': None, 'terminal_raw_epsilon_sha256': None, 'epsilon_source': 'last_t50',
                    'filter': renderer.filter_spec(col['sigma'], col['radius']), 'outside_cleanup_clean_exact': True,
                    'alpha0': alpha, 'sigma0': alpha * .029, 'rho0': .029,
                    'scheduler_step_index_before': 20, 'scheduler_step_index_after': 20, 'projected_latent_sha256': 'final'}}
            renderer.validate_terminal_row(row, col)
            bad = copy.deepcopy(row); bad['total_unet_calls'] = 16
            with self.assertRaisesRegex(ValueError, 'budget'):
                renderer.validate_terminal_row(bad, col)
            bad = copy.deepcopy(row); bad['terminal_trace']['filter']['radius'] += 1
            with self.assertRaisesRegex(ValueError, 'normalized rule'):
                renderer.validate_terminal_row(bad, col)

    def test_frozen_global_lp_comparators_exist_for_all_eight_cases(self):
        cases, rows = renderer.frozen_v10()
        self.assertEqual(len(cases), 8)
        for cid in cases:
            for family in ('baseline', 'bidirectional'):
                row = rows[cid, family + '_terminal_t0_lp']
                self.assertEqual((row['status'], row['actual_steps'], row['terminal_unet_calls'], row['total_unet_calls']),
                                 ('complete', 15, 1, 16))

    def test_six_synthetic_pages_keep_full_coverage_and_readonly_sources(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_v11_', dir=HERE / 'render_review') as tmp:
            output = Path(tmp)
            path = output / 'source.png'
            Image.new('RGB', (512, 512), '#667686').save(path)
            ref = {'path': str(path), 'sha256': renderer.sha(path)}
            cases, rows = [], {}
            reference_names = [c['key'] for c in renderer.columns() if c['kind'] != 'v11']
            for subject in R.SUBJECTS:
                for group in R.GROUPS:
                    case = {'case_id': 'synthetic_' + subject + '_' + group, 'subject': subject, 'group': group,
                        'source_aus_0_5': [0.] * 12, 'target_aus_0_5': [0.] * 12,
                        **{name: ref for name in reference_names}}
                    cases.append(case)
                    for col in renderer.columns():
                        if col['kind'] == 'v11':
                            rows[case['case_id'], col['key']] = {'output': str(path), 'sha256': ref['sha256'], 'status': 'complete',
                                'actual_steps': 15, 'denoising_steps': 15, 'terminal_unet_calls': 0, 'total_unet_calls': 15}
            fonts, _ = R.layout.make_fonts()
            edited = [c for c in cases if c['group'] == 'lowest_available']
            pages = [R.render_page('overview', edited, renderer.columns(True), rows, output, fonts, True)]
            pages += [R.render_page(s, R.base.ordered_cases(cases, [s]), renderer.columns(), rows, output, fonts)
                      for s in R.SUBJECTS]
            def synthetic_roi(case):
                return {'case_id': case['case_id'], 'subject': case['subject'], 'label': 'synthetic fixed source ROI',
                        'bbox_xyxy': [100, 120, 240, 280], 'for_display_only': True, 'sent_to_evaluation': False}
            with patch.object(R, 'source_roi', synthetic_roi):
                pages.append(R.render_debug(cases, rows, output, fonts))
            self.assertEqual(len(pages), 6)
            detail = [c for p in pages if p['kind'] == 'subject' for c in p['cells']]
            self.assertEqual(len(detail), 80)
            self.assertEqual(sum('_local_lp_' in c['column_key'] for c in detail), 32)
            self.assertEqual(sum(c['column_key'].startswith('v10_global_lp') for c in detail), 16)
            self.assertTrue(all(c['render_status'] == 'shown' for p in pages for c in p['cells']))
            self.assertTrue(all('UNet15' in c['display_step_caption'] for c in detail if '_local_lp_' in c['column_key']))
            self.assertTrue(all('UNet16' in c['display_step_caption'] for c in detail if c['column_key'].startswith('v10_global_lp')))
            self.assertFalse(pages[-1]['sent_to_evaluation'])
            self.assertEqual(renderer.sha(path), ref['sha256'])
            with self.assertRaises(FileExistsError):
                R.render_page('overview', edited, renderer.columns(True), rows, output, fonts, True)
            draw = ImageDraw.Draw(Image.new('RGB', (1814, 1968)))
            for text in R.render_page.__code__.co_consts:
                if isinstance(text, str) and (text.startswith('V11局部') or text.startswith('仅整图')):
                    self.assertLessEqual(draw.textlength(text, font=fonts[22]), 1814 - 2 * R.MARGIN)


if __name__ == '__main__':
    unittest.main(verbosity=2)
