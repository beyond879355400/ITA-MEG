"""Synthetic-only renderer tests; never opens real/generated experiment images."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v9_renderer_tested', HERE / 'render_v9.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class RenderTests(unittest.TestCase):
    def test_overview_and_all_methods_have_exact_columns(self):
        overview, all_cols = renderer.columns(True), renderer.columns()
        self.assertEqual([c['key'] for c in overview], ['source_real', 'target_real', 'a_bidirectional',
            'bidirectional_late_au100', 'bidirectional_late_au050', 'bidirectional_late_au000'])
        self.assertEqual(len(all_cols), 10)
        self.assertEqual(len({c['key'] for c in all_cols}), 10)
        self.assertEqual([c['late'] for c in all_cols if c['kind'] == 'v9'], [1, 1, .5, .5, 0, 0])
        self.assertEqual(sum(c['kind'] == 'v9' for c in all_cols) * 8, 48)

    def test_source_roi_is_fixed_index_box_plus_twenty(self):
        points = [[100.2, 120.1] for _ in range(68)]
        points[5], points[67] = [5.1, 40.2], [500.6, 502.3]
        raw, clipped = renderer.roi_bounds(points, renderer.ROI_POLICY['SN012']['indices'])
        self.assertEqual(raw, [-15, 20, 521, 523])
        self.assertEqual(clipped, [0, 20, 512, 512])
        self.assertEqual(renderer.ROI_POLICY['SN018']['indices'], list(range(17, 28)))
        with self.assertRaises(ValueError):
            renderer.roi_bounds([[float('nan'), 0]] * 68, [1])

    def test_failed_record_is_not_replaced(self):
        case = {'case_id': 'synthetic_case'}
        col = renderer.columns()[-1]
        source = renderer.cell_source(case, col, {})
        self.assertEqual(source['generation_status'], 'missing_record')
        self.assertIsNone(source['path'])

    def test_crop_does_not_modify_source_and_keeps_one_roi(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_crop_', dir=HERE / 'render_review') as tmp:
            path = Path(tmp) / 'source.png'
            Image.new('RGB', (512, 512), '#ac7920').save(path)
            before = renderer.sha(path)
            fonts, _ = renderer.layout.make_fonts()
            page = Image.new('RGB', (1600, 700), 'white')
            source = {'path': str(path), 'expected_sha256': before, 'generation_status': 'complete', 'actual_steps': 15}
            roi = {'bbox_xyxy': [100, 120, 250, 270]}
            first = renderer.paste_crop(page, ImageDraw.Draw(page), 0, 0, source, roi, fonts)
            second = renderer.paste_crop(page, ImageDraw.Draw(page), 550, 0, source, roi, fonts)
            self.assertEqual(renderer.sha(path), before)
            self.assertEqual(first['crop_box_xyxy'], second['crop_box_xyxy'])
            self.assertEqual(first['crop_size'], [150, 150])
            self.assertEqual(first['render_status'], 'shown')
            self.assertFalse(first['sent_to_evaluation'])
            corrupt = {**source, 'expected_sha256': 'wrong'}
            blocked = renderer.paste_crop(page, ImageDraw.Draw(page), 1050, 0, corrupt, roi, fonts)
            self.assertEqual(blocked['render_status'], 'placeholder')

    def test_full_subject_sheet_keeps_twenty_cells_and_exclusive_output(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_sheet_', dir=HERE / 'render_review') as tmp:
            output = Path(tmp)
            path = output / 'source.png'
            Image.new('RGB', (512, 512), '#606b78').save(path)
            source_ref = {'path': str(path), 'sha256': renderer.sha(path)}
            cases, rows = [], {}
            for group in renderer.GROUPS:
                case = {'case_id': 'synthetic_' + group, 'subject': 'SN012', 'group': group,
                        'source_aus_0_5': [0.] * 12, 'target_aus_0_5': [0.] * 12,
                        **{name: source_ref for name in ('source_real', 'target_real', 'a_baseline', 'a_bidirectional')}}
                cases.append(case)
                for col in renderer.columns():
                    if col['kind'] == 'v9':
                        rows[case['case_id'], col['key']] = {'output': str(path), 'sha256': source_ref['sha256'],
                                                            'status': 'complete', 'actual_steps': 15}
            fonts, _ = renderer.layout.make_fonts()
            sheet = renderer.render_page('synthetic_all', cases, renderer.columns(), rows, output, fonts)
            self.assertEqual(len(sheet['cells']), 20)
            self.assertTrue(all(c['render_status'] == 'shown' and c['crop'] is False for c in sheet['cells']))
            self.assertEqual(renderer.sha(path), source_ref['sha256'])
            with self.assertRaises(FileExistsError):
                renderer.render_page('synthetic_all', cases, renderer.columns(), rows, output, fonts)


if __name__ == '__main__':
    unittest.main(verbosity=2)
