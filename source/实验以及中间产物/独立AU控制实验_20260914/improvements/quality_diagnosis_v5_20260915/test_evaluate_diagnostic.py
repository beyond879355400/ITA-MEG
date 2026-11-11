#!/usr/bin/env python3
"""Synthetic, standard-library-only checks. No real models or images loaded."""
import ast
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_diagnostic as e


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = e.HERE / 'synthetic_tests'
        parent.mkdir(exist_ok=True)
        cls.output = Path(tempfile.mkdtemp(prefix='unit_evaluator_', dir=parent))

    def fixtures(self):
        run = e.HERE / 'synthetic_tests/not_a_real_generation'
        cases, records = [], []
        for subject in sorted(e.SUBJECTS):
            for group in sorted(e.GROUPS):
                cid = subject + '_' + group
                source = [0] * 12
                target = [1] + [0] * 11 if group != 'no_edit' else source[:]
                case = {'case_id': cid, 'subject': subject, 'split': 'dev', 'group': group,
                        'source_aus_0_5': source, 'target_aus_0_5': target}
                for method in e.REFERENCES:
                    case[method] = {'path': str(run / (cid + '_' + method + '.png')), 'sha256': 'a' * 64}
                cases.append(case)
                for method in e.METHODS:
                    records.append({**{key: case[key] for key in ('case_id', 'subject', 'split', 'group',
                                                                'source_aus_0_5', 'target_aus_0_5')},
                                    'method': method, 'seed': 101, 'status': 'complete',
                                    'output': str(run / (cid + '_' + method + '.png')), 'sha256': 'b' * 64})
        return ({'kind': 'v5_development_diagnostic_cases', 'test_data_used': False, 'case_count': 8, 'cases': cases},
                {'kind': 'v5_source_initialization_diagnostic', 'test_data_used': False, 'status': 'complete', 'records': records}, run)

    def build(self, cases, result, run):
        with patch.object(e, 'checked_image', side_effect=lambda reference: dict(reference)):
            return e.build_items(cases, result, run)

    def test_complete_closed_manifest(self):
        rows = self.build(*self.fixtures())
        self.assertEqual(len(rows), 88)
        self.assertEqual(sum(row['role'] == 'new_generation' for row in rows), 48)
        self.assertEqual(len({row['sample_id'] for row in rows}), 88)

    def test_failed_generation_is_retained_without_image(self):
        cases, result, run = self.fixtures()
        result['status'] = 'completed_with_failures'
        result['records'][0].update(status='error', output=None, sha256=None, error='synthetic')
        rows = self.build(cases, result, run)
        failures = [row for row in rows if not row['image_available']]
        self.assertEqual(len(rows), 88)
        self.assertEqual(len(failures), 1)
        self.assertIsNone(failures[0]['sha256'])
        self.assertEqual(failures[0]['generation_record']['error'], 'synthetic')

    def test_nonterminal_and_nondev_are_rejected_before_image_read(self):
        cases, result, run = self.fixtures()
        result['status'] = 'running'
        with patch.object(e, 'checked_image') as probe, self.assertRaises(ValueError):
            e.build_items(cases, result, run)
        probe.assert_not_called()
        result['status'] = 'complete'
        cases['cases'][0]['subject'] = 'SN999'
        with patch.object(e, 'checked_image') as probe, self.assertRaises(ValueError):
            e.build_items(cases, result, run)
        probe.assert_not_called()

    def test_missing_method_or_wrong_seed_is_rejected(self):
        cases, result, run = self.fixtures()
        result['records'][-1] = copy.deepcopy(result['records'][0])
        with self.assertRaises(ValueError):
            self.build(cases, result, run)
        cases, result, run = self.fixtures()
        result['records'][0]['seed'] = 102
        with self.assertRaises(ValueError):
            self.build(cases, result, run)

    def test_noedit_changed_metrics_na(self):
        metrics = e.au_metrics([.2] * 12, [.2] * 12, [0] * 12, [0] * 12)
        for key in ('target_mae_changed', 'delta_mae_changed', 'direction_correct_fraction'):
            self.assertIsNone(metrics[key])
        self.assertEqual(metrics['unchanged_source_relative_drift'], 0)
        self.assertAlmostEqual(metrics['target_mae_all12'], .2)

    def test_source_copy_baseline_is_not_perfect_edit(self):
        metrics = e.au_metrics([.2] * 12, [.2] * 12, [0] * 12, [1] + [0] * 11)
        self.assertAlmostEqual(metrics['target_mae_changed'], .8)
        self.assertEqual(metrics['delta_mae_changed'], 1)
        self.assertEqual(metrics['direction_correct_fraction'], 0)
        self.assertEqual(metrics['unchanged_source_relative_drift'], 0)

    def test_missing_au_is_na_not_zero(self):
        metrics = e.au_metrics(None, [.2] * 12, [0] * 12, [1] + [0] * 11)
        self.assertTrue(all(metrics[key] is None for key in e.METRICS[:6]))
        metrics = e.au_metrics([.2] * 12, None, [0] * 12, [1] + [0] * 11)
        self.assertIsNotNone(metrics['target_mae_changed'])
        self.assertIsNone(metrics['delta_mae_changed'])

    def test_subject_equal_not_image_weighted_and_common_valid(self):
        rows = []
        for cid, subject, value in [('a', 'A', 0), ('b', 'A', 4), ('c', 'B', 10)]:
            for method in ('left', 'right'):
                rows.append({'case_id': cid, 'subject': subject, 'group': 'lowest_available', 'method': method,
                             **{metric: value for metric in e.METRICS}})
        stats = e.aggregate(rows, ['left', 'right'])['edited']['target_mae_changed']
        self.assertEqual(stats['subject_equal_means'], {'left': 6, 'right': 6})
        rows[-1]['target_mae_changed'] = None
        stats = e.aggregate(rows, ['left', 'right'])['edited']['target_mae_changed']
        self.assertEqual(stats['common_cases'], 2)
        self.assertEqual(stats['subject_count'], 1)
        self.assertEqual(stats['subject_equal_means'], {'left': 2, 'right': 2})

    def test_extra5_requires_all_five_real_references(self):
        pred = {str(au): 1 for au in e.EXTRA}
        target = {str(au): .5 for au in e.EXTRA}
        detail, average = e.extras(pred, target, None)
        self.assertEqual(average, .5)
        self.assertIsNone(detail['45']['real_source_measured'])
        target['45'] = None
        detail, average = e.extras(pred, target, None)
        self.assertIsNone(average)
        self.assertIsNone(detail['45']['absolute_error_to_target'])

    def test_facenet_adapter_preserves_model_and_measurement_loop(self):
        out = self.output / 'facenet_binding'
        out.mkdir()
        e.save(out / 'items.json', {'items': [], 'test_used': False})
        core = (e.POLICY_DIR / 'run_facenet_single_directions.py').read_text()
        derived = e.derived_facenet_source(out, 87)
        model_start, model_end = '        detector = MTCNN(', "    save(out / 'protocol.json'"
        self.assertEqual(core[core.index(model_start):core.index(model_end)],
                         derived[derived.index(model_start):derived.index(model_end)])
        start, end = '    measurements, embeddings = {}, {}', '    records = [{'
        self.assertEqual(core[core.index(start):core.index(end)], derived[derived.index(start):derived.index(end)])
        start, end = '    for row in records:\n        source_sha', '    coverage = {'
        self.assertEqual(core[core.index(start):core.index(end)], derived[derived.index(start):derived.index(end)])
        self.assertNotIn("'logical_records': 57", derived)
        self.assertIn("'logical_records': 87", derived)
        self.assertIn('len(sources) == 8', derived)
        self.assertNotIn('shared_cache', derived)
        self.assertIn("len({(row['case_id'], row['method']) for row in items}) == 87", derived)
        self.assertFalse(any(isinstance(node, ast.Constant) and type(node.value) is int and node.value == 57
                             for node in ast.walk(ast.parse(derived))))
        # Terminal failures can legitimately leave exactly57 available records.
        self.assertIn('len(items) == 57', e.derived_facenet_source(out, 57))
        self.assertIn('len(items) == 40', e.derived_facenet_source(out, 40))
        self.assertIn('len(items) == 88', e.derived_facenet_source(out, 88))

    def test_openface_largest_invalid_candidate_not_replaced(self):
        helper = e.load_openface()
        rows = []
        for size, au1 in [(10, 0), (20, 'nan')]:
            row = {'face': len(rows), 'confidence': .1}
            row.update({f'x_{index}': size if index else 0 for index in range(68)})
            row.update({f'y_{index}': size if index else 0 for index in range(68)})
            row.update({f'AU{au:02d}_r': 0 for au in helper.AUS17})
            row['AU01_r'] = au1
            rows.append(row)
        path = self.output / 'synthetic_faces.csv'
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        parsed = helper.parse(path)
        self.assertEqual(parsed['selected_row'], 1)
        self.assertEqual(len(parsed['all_candidates']), 2)
        self.assertEqual(len(parsed['raw_rows']), 2)
        self.assertIsNone(parsed['vector'])

    def test_all_original14_columns_retained(self):
        self.assertEqual(len(e.METRICS), 14)
        self.assertIn('buffalo_identity_cosine', e.METRICS)

    def test_reference_lineage_rejects_changed_hash(self):
        original = e.read(e.POLICY_DIR / 'dev19_items.json')['items']
        case_id = original[0]['case_id']
        prior = {row['method']: row for row in original if row['case_id'] == case_id}
        case = {'case_id': case_id, 'subject': prior['source_real']['subject'], 'split': 'dev',
                'source_aus_0_5': prior['source_real']['source_labels_12'],
                'target_aus_0_5': prior['source_real']['target_labels_12']}
        for role, method in [('source_real', 'source_real'), ('target_real', 'target_real'),
                             ('baseline_text2img', 'baseline'), ('bidirectional_text2img', 'current_model')]:
            case[role] = {'sha256': prior[method]['sha256']}
        result = e.reference_lineage([case])
        self.assertEqual(len(result['bindings']), 4)
        case['target_real']['sha256'] = 'c' * 64
        with self.assertRaises(ValueError):
            e.reference_lineage([case])

    def test_merge_all88_with_generation_failure(self):
        cases, generation, run = self.fixtures()
        generation['status'] = 'completed_with_failures'
        generation['records'][0].update(status='failed', output=None, sha256=None)
        items = self.build(cases, generation, run)
        available = [item for item in items if item['image_available']]
        output = self.output / 'synthetic_merge'
        output.mkdir()
        (output / 'openface').mkdir()
        (output / 'facenet').mkdir()
        for name in ('manifest.json', 'items.json', 'input_seal.json'):
            e.save(output / name, {'synthetic': True})
        au = [{**item, 'vector': [.2] * 12, 'selected_au17': {str(au): .2 for au in e.AUS + e.EXTRA},
               'all_candidates': [], 'raw_rows': [], 'face_count': 1} for item in available]
        fn = [{**item, 'identity_cosine': .9, 'candidates': [], 'face_count': 1} for item in available]
        e.save(output / 'openface/results.json', {'status': 'complete', 'policy_sha256': e.POLICY_SHA,
                                                'test_used': False, 'records': au})
        e.save(output / 'facenet/results.json', {'summary': {'status': 'complete'}, 'policy_sha256': e.POLICY_SHA,
                                               'test_used': False, 'records': fn})
        document = {'items': items, 'generation_status': generation['status']}
        with patch.object(e, 'checked', return_value=(output, document)):
            e.merge(output)
        report = e.read(output / 'comparison/results.json')
        self.assertEqual(len(report['records']), 88)
        self.assertEqual(len(report['source_copy_references']), 8)
        failed = next(row for row in report['records'] if row['generation_status'] == 'failed')
        self.assertTrue(all(failed[metric] is None for metric in e.METRICS))
        self.assertEqual(failed['facenet_primary']['status'], 'generation_failed_not_evaluated')
        self.assertEqual(report['coverage'][failed['method']]['edited']['AU_valid'], 3)
        self.assertIsNone(report['coverage'][failed['method']]['edited']['buffalo_valid_rate'])
        self.assertTrue(all(row['buffalo_auxiliary']['status'] == 'not_run' for row in report['records']))

    def test_test_path_rejected_without_reading(self):
        with self.assertRaises(ValueError):
            e.scoped_path(e.EXP / 'final_test/not_opened.png')


if __name__ == '__main__':
    unittest.main(verbosity=2)
