#!/usr/bin/env python3
"""Synthetic/metadata checks only; never start OpenFace or FaceNet models."""
import ast
import copy
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluate_v6 as adapter

e = adapter.CORE


class V6EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = e.HERE / 'synthetic_tests'
        parent.mkdir(exist_ok=True)
        cls.output = Path(tempfile.mkdtemp(prefix='v6_evaluator_', dir=parent))

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
                    # Historical references intentionally live outside the v6 run.
                    case[method] = {'path': str(adapter.V5_RUN / (cid + '_' + method + '.png')),
                                    'sha256': 'a' * 64}
                cases.append(case)
                for method in e.METHODS:
                    config = method.split('_', 1)[1]
                    params = adapter.CONFIGS[config]
                    records.append({**{key: case[key] for key in ('case_id', 'subject', 'split', 'group',
                                       'source_aus_0_5', 'target_aus_0_5')},
                                    'method': method, 'seed': 101, 'status': 'complete',
                                    'config': config, **params, 'nominal_steps': 20,
                                    'actual_steps': 11 if params['strength'] == .55 else 15,
                                    'output': str(run / (cid + '_' + method + '.png')), 'sha256': 'b' * 64})
        return ({'kind': adapter.CASE_KIND, 'test_data_used': False, 'case_count': 8, 'cases': cases},
                {'kind': adapter.RESULT_KIND, 'test_data_used': False, 'status': 'complete', 'records': records}, run)

    def build(self, cases, result, run):
        with patch.object(e, 'checked_image', side_effect=lambda reference: dict(reference)):
            return e.build_items(cases, result, run)

    def test_complete120_and_external_v5_vae(self):
        rows = self.build(*self.fixtures())
        self.assertEqual(len(rows), 120)
        self.assertEqual(sum(row['role'] == 'new_generation' for row in rows), 48)
        self.assertEqual(len({row['sample_id'] for row in rows}), 120)
        self.assertEqual(len(e.REFERENCES), 9)
        self.assertEqual(len(e.METRICS), 14)

    def test_failed_image_retained_not_opened(self):
        cases, result, run = self.fixtures()
        result['status'] = 'completed_with_failures'
        result['records'][0].update(status='error', output=None, sha256=None, error='synthetic')
        rows = self.build(cases, result, run)
        self.assertEqual(len(rows), 120)
        failed = [row for row in rows if not row['image_available']]
        self.assertEqual(len(failed), 1)
        self.assertIsNone(failed[0]['path'])

    def test_nonterminal_and_nondev_rejected_before_images(self):
        for mutation in ('running', 'subject'):
            cases, result, run = self.fixtures()
            if mutation == 'running':
                result['status'] = 'running'
            else:
                cases['cases'][0]['subject'] = 'SN999'
            with patch.object(e, 'checked_image') as probe, self.assertRaises(ValueError):
                e.build_items(cases, result, run)
            probe.assert_not_called()

    def test_duplicate_method_wrong_seed_and_new_output_escape(self):
        for mutation in ('duplicate', 'seed', 'escape'):
            cases, result, run = self.fixtures()
            if mutation == 'duplicate':
                result['records'][-1] = copy.deepcopy(result['records'][0])
            elif mutation == 'seed':
                result['records'][0]['seed'] = 102
            else:
                result['records'][0]['output'] = str(adapter.V5_RUN / 'not_a_v6_output.png')
            with self.assertRaises(ValueError):
                self.build(cases, result, run)

    def test_frozen_lineage_all72_refs_and_paths(self):
        cases = adapter.frozen_cases()
        lineage = e.reference_lineage(cases)
        self.assertEqual(len(lineage['v5_reference_bindings']), 72)
        self.assertFalse(lineage['same_sha_different_path_allowed'])
        cases[0]['baseline_img2img_s055']['path'] += '.different_path'
        with self.assertRaises(ValueError):
            e.reference_lineage(cases)

    def test_frozen_case_labels_and_hash_cannot_change(self):
        for field in ('sha', 'labels'):
            cases = adapter.frozen_cases()
            if field == 'sha':
                cases[0]['vae_reconstruction']['sha256'] = 'c' * 64
            else:
                cases[0]['source_aus_0_5'][0] = 4
            with self.assertRaises(ValueError):
                e.reference_lineage(cases)

    def test_facenet_all_count_rebindings_and_model_loops_unchanged(self):
        out = self.output / 'facenet_binding'
        out.mkdir()
        e.save(out / 'items.json', {'items': [], 'test_used': False})
        core = (e.POLICY_DIR / 'run_facenet_single_directions.py').read_text()
        for count in (72, 119, 120):
            derived = e.derived_facenet_source(out, count)
            for start, end in [('        detector = MTCNN(', "    save(out / 'protocol.json'"),
                               ('    measurements, embeddings = {}, {}', '    records = [{'),
                               ('    for row in records:\n        source_sha', '    coverage = {')]:
                self.assertEqual(core[core.index(start):core.index(end)],
                                 derived[derived.index(start):derived.index(end)])
            self.assertIn(f'len(items) == {count}', derived)
            self.assertIn("len({(row['case_id'], row['method']) for row in items}) == " + str(count), derived)
            self.assertIn("'logical_records': " + str(count), derived)
            self.assertIn("'input_logical_count': " + str(count), derived)
            self.assertIn('len(sources) == 8', derived)
            self.assertNotIn('shared_cache', derived)
            self.assertFalse(any(isinstance(node, ast.Constant) and type(node.value) is int and node.value == 57
                                 for node in ast.walk(ast.parse(derived))))
        for count in (57, 71, 121):
            with self.assertRaises(ValueError):
                e.derived_facenet_source(out, count)

    def test_core_metric_functions_byte_preserved(self):
        old = adapter.V5_CORE.read_text()
        def functions(source):
            tree = ast.parse(source)
            return {node.name: ast.get_source_segment(source, node) for node in tree.body
                    if isinstance(node, ast.FunctionDef)}
        a, b = functions(old), functions(adapter.DERIVED_SOURCE)
        for name in ('au_metrics', 'extras', 'aggregate', 'load_openface', 'openface', 'facenet'):
            self.assertEqual(a[name], b[name], name)

    def test_noedit_changed_na_and_missing_au_na(self):
        metrics = e.au_metrics([.2] * 12, [.2] * 12, [0] * 12, [0] * 12)
        for key in ('target_mae_changed', 'delta_mae_changed', 'direction_correct_fraction'):
            self.assertIsNone(metrics[key])
        self.assertEqual(metrics['unchanged_source_relative_drift'], 0)
        missing = e.au_metrics(None, [.2] * 12, [0] * 12, [1] + [0] * 11)
        self.assertTrue(all(missing[key] is None for key in e.METRICS[:6]))

    def test_subject_equal_and_common_valid(self):
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
        self.assertEqual(stats['subject_equal_means'], {'left': 2, 'right': 2})

    def test_openface_largest_invalid_not_replaced(self):
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
        self.assertIsNone(parsed['vector'])

    def test_merge120_all_refs_failure_na_and_buffalo_not_run(self):
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
        with patch.object(e, 'checked', return_value=(output, {'items': items, 'generation_status': generation['status']})):
            e.merge(output)
        report = e.read(output / 'comparison/results.json')
        self.assertEqual(len(report['records']), 120)
        self.assertEqual(report['logical_records'], 120)
        self.assertEqual(len(report['coverage']), 15)
        self.assertEqual(len(report['pairwise']), 66)  # 12 generation methods, including old 6 refs.
        self.assertEqual(len(report['source_copy_references']), 8)
        failed = next(row for row in report['records'] if row['generation_status'] == 'failed')
        self.assertTrue(all(failed[key] is None for key in e.METRICS))
        self.assertTrue(all(row['buffalo_auxiliary']['status'] == 'not_run' for row in report['records']))
        self.assertIsNone(report['coverage'][failed['method']]['edited']['buffalo_valid_rate'])

    def test_v6_config_binding(self):
        items = self.build(*self.fixtures())
        document = {'items': items}
        with patch.object(adapter, '_validate', return_value=document):
            self.assertEqual(adapter.validate(e.HERE)['configs'], adapter.CONFIGS)
            next(item for item in items if item['role'] == 'new_generation')['generation_record']['au_scale'] = 9
            with self.assertRaises(ValueError):
                adapter.validate(e.HERE)

    def test_prepare_and_seal_with_all_failed_synthetic_generations(self):
        run = self.output / 'synthetic_all_failed_run'
        run.mkdir()
        cases = adapter.frozen_cases()
        records = []
        for case in cases:
            for method in e.METHODS:
                config = method.split('_', 1)[1]
                records.append({**{key: case[key] for key in ('case_id', 'subject', 'split', 'group',
                                'source_aus_0_5', 'target_aus_0_5')},
                                'method': method, 'seed': 101, 'config': config,
                                **adapter.CONFIGS[config], 'nominal_steps': 20,
                                'status': 'failed', 'output': None, 'sha256': None,
                                'error': 'synthetic_no_generation_performed'})
        e.save(run / 'cases.json', {'kind': adapter.CASE_KIND, 'test_data_used': False,
                                   'case_count': 8, 'cases': cases})
        e.save(run / 'result.json', {'kind': adapter.RESULT_KIND, 'test_data_used': False,
                                    'status': 'completed_with_failures', 'records': records})
        output = self.output / 'synthetic_prepared'
        e.prepare(run, output)
        _, document = e.checked(output)
        self.assertEqual(document['logical_count'], 120)
        self.assertEqual(document['measurement_count'], 72)
        self.assertEqual(len(document['items']), 120)
        self.assertEqual(document['adapter_provenance']['v5_core_sha256'], adapter.V5_CORE_SHA)
        self.assertFalse((output / 'openface').exists())
        self.assertFalse((output / 'facenet').exists())
        with self.assertRaises(FileExistsError):
            e.prepare(run, output)

    def test_no_model_modules_imported_and_final_test_path_forbidden(self):
        for name in ('torch', 'facenet_pytorch', 'insightface'):
            self.assertNotIn(name, sys.modules)
        with self.assertRaises(ValueError):
            e.scoped_path(e.EXP / 'final_test/not_opened.png')


if __name__ == '__main__':
    unittest.main(verbosity=2)
