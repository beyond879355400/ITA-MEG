"""Synthetic CPU-only V10 layout checks, no real/generated experiment images."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v10_renderer_tested', HERE / 'render_v10.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
R = renderer.R


class RenderTests(unittest.TestCase):
    def test_exact_overview_all_columns_and_visible_budget(self):
        overview = renderer.columns(True)
        self.assertEqual([c['key'] for c in overview], ['source_real', 'target_real', 'a_bidirectional',
            'bidirectional_terminal_last_eps', 'bidirectional_terminal_t0', 'bidirectional_terminal_t0_lp'])
        cols = renderer.columns()
        self.assertEqual(len(cols), 10)
        self.assertEqual(len({c['key'] for c in cols}), 10)
        self.assertEqual([c['terminal_unet_calls'] for c in cols if c['kind'] == 'v10'], [0, 0, 1, 1, 1, 1])
        self.assertEqual(sum(c['kind'] == 'v10' for c in cols) * 8, 48)
        self.assertIs(R.columns, renderer.columns)
        self.assertIs(R.load_inputs, renderer.load_inputs)

    def test_source_roi_policy_is_verbatim_v9(self):
        self.assertEqual(R.ROI_POLICY['SN012']['indices'], list(range(5, 12)) + list(range(48, 68)))
        self.assertEqual(R.ROI_POLICY['SN018']['indices'], list(range(17, 28)))
        self.assertEqual(R.ROI_PADDING, 20)
        points = [[100.2, 120.1] for _ in range(68)]
        points[5], points[67] = [5.1, 40.2], [500.6, 502.3]
        self.assertEqual(R.roi_bounds(points, R.ROI_POLICY['SN012']['indices']), ([-15, 20, 521, 523], [0, 20, 512, 512]))

    def test_terminal_budget_mismatch_is_rejected(self):
        for col in renderer.columns()[4:]:
            extra = col['terminal_unet_calls']
            alpha = 1 / math.sqrt(1 + .029 ** 2)
            row = {'status': 'complete', 'config': col['config'], 'seed': 101, 'strength': .75, 'au_scale': 1.5,
                'ip_scale': .9, 'nominal_steps': 20, 'actual_steps': 15, 'denoising_steps': 15, 'deis_unet_calls': 15,
                'terminal_unet_calls': extra, 'total_unet_calls': 15 + extra, 'actual_timesteps': renderer.frozen.TIMESTEPS,
                'projection_trace': [{'latent_sha256': 'final'}],
                'unet_call_trace': [{'phase': phase, 't': t, 'cfg_batch': 2, 'au_scale': 1.5}
                    for phase, t in [('deis', t) for t in renderer.frozen.TIMESTEPS] + ([('terminal', 0)] if extra else [])],
                'terminal_trace': {'mode': col['config'], 'last_deis_timestep': 50, 'extra_cfg': 7.5 if extra else None,
                    'filter': copy.deepcopy(renderer.frozen.FILTER) if col['config'] == 'terminal_t0_lp' else None,
                    'alpha0': alpha, 'sigma0': alpha * .029, 'rho0': .029,
                    'scheduler_step_index_before': 20, 'scheduler_step_index_after': 20, 'projected_latent_sha256': 'final'}}
            renderer.validate_terminal_row(row, col)
            row['total_unet_calls'] += 1
            with self.assertRaisesRegex(ValueError, 'call budget'):
                renderer.validate_terminal_row(row, col)

    def test_subject_sheet_keeps_all_twenty_cells_and_source_hash(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_v10_', dir=HERE / 'render_review') as tmp:
            output = Path(tmp)
            path = output / 'source.png'
            Image.new('RGB', (512, 512), '#667686').save(path)
            ref = {'path': str(path), 'sha256': renderer.sha(path)}
            cases, rows = [], {}
            for group in R.GROUPS:
                case = {'case_id': 'synthetic_' + group, 'subject': 'SN012', 'group': group,
                    'source_aus_0_5': [0.] * 12, 'target_aus_0_5': [0.] * 12,
                    **{name: ref for name in ('source_real', 'target_real', 'a_baseline', 'a_bidirectional')}}
                cases.append(case)
                for col in renderer.columns():
                    if col['kind'] == 'v10':
                        rows[case['case_id'], col['key']] = {'output': str(path), 'sha256': ref['sha256'], 'status': 'complete',
                            'actual_steps': 15, 'denoising_steps': 15, 'terminal_unet_calls': col['terminal_unet_calls'],
                            'total_unet_calls': 15 + col['terminal_unet_calls']}
            fonts, _ = R.layout.make_fonts()
            page = R.render_page('synthetic_all', cases, renderer.columns(), rows, output, fonts)
            self.assertEqual(len(page['cells']), 20)
            self.assertTrue(all(c['render_status'] == 'shown' and c['crop'] is False for c in page['cells']))
            self.assertTrue(all('UNet16' in c['display_step_caption'] for c in page['cells']
                                if c.get('terminal_unet_calls') == 1))
            self.assertEqual(renderer.sha(path), ref['sha256'])
            with self.assertRaises(FileExistsError):
                R.render_page('synthetic_all', cases, renderer.columns(), rows, output, fonts)


if __name__ == '__main__':
    unittest.main(verbosity=2)
