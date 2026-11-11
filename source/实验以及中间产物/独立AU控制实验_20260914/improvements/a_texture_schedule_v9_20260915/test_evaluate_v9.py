#!/usr/bin/env python3
"""Synthetic and frozen-metadata tests; never run ML model inference."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluate_v9 as adapter

e = adapter.CORE


class V9EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = e.HERE / 'synthetic_tests'
        parent.mkdir(exist_ok=True)
        cls.output = Path(tempfile.mkdtemp(prefix='v9_evaluator_', dir=parent))

    def fixtures(self):
        run = self.output / 'not_a_real_generation'
        cases = adapter.frozen_cases()
        _, historical = adapter.frozen_v7()
        records = []
        for case in cases:
            for method in e.METHODS:
                family, config = method.split('_', 1)
                row = copy.deepcopy(historical[(case['case_id'], family + '_smooth_s075_a150')])
                row.update(method=method, config=config, **adapter.CONFIGS[config],
                           output=str(run / (case['case_id'] + '_' + method + '.png')), sha256='b' * 64,
                           au_scale_trace=[{'t': t, 'au_scale': adapter.CONFIGS[config]['au_scale_late']
                                            if t <= 250 else 1.5} for t in adapter.TIMESTEPS])
                for entry, observed in zip(row['projection_trace'], row['au_scale_trace']):
                    entry['au_scale_used'] = observed['au_scale']
                records.append(row)
        return ({'kind': adapter.CASE_KIND, 'test_data_used': False, 'case_count': 8, 'cases': cases},
                {'kind': adapter.RESULT_KIND, 'test_data_used': False, 'status': 'complete', 'records': records}, run)

    def build(self, cases, result, run):
        with patch.object(e, 'checked_image', side_effect=lambda ref: dict(ref)):
            return e.build_items(cases, result, run)

    def test_metadata136_all88_historical_refs_and_v7_fields(self):
        report = adapter.metadata()
        self.assertEqual((report['case_count'], report['reference_count'], report['logical_records']), (8, 88, 136))
        self.assertEqual(len(report['original_metric_columns']), 14)
        self.assertEqual(len(report['reference_lineage']['v7_reference_bindings']), 88)
        self.assertEqual(len(report['reference_lineage']['v5_reference_bindings']), 72)
        cases, _ = adapter.frozen_v7()
        current = adapter.frozen_cases()
        self.assertTrue(all(all(row[key] == old[key] for key in old) for row, old in zip(current, cases)))
        for key in ('region_mask', 'prompt', 'source_aus_0_5', 'a_bidirectional'):
            wrong = copy.deepcopy(current)
            wrong[0][key] = None
            with self.assertRaises(ValueError):
                adapter.reference_lineage(wrong)

    def test_complete136_and_failure_record_retained(self):
        cases, result, run = self.fixtures()
        rows = self.build(cases, result, run)
        self.assertEqual(len(rows), 136)
        self.assertEqual(sum(row['role'] == 'new_generation' for row in rows), 48)
        result['status'] = 'completed_with_failures'
        result['records'][0].update(status='failed', output=None, sha256=None)
        rows = self.build(cases, result, run)
        self.assertEqual(len(rows), 136)
        self.assertEqual(sum(not row['image_available'] for row in rows), 1)

    def test_nonterminal_duplicate_and_test_scope_rejected(self):
        for failure in ('running', 'duplicate', 'seed', 'subject', 'escape'):
            cases, result, run = self.fixtures()
            if failure == 'running':
                result['status'] = 'running'
            elif failure == 'duplicate':
                result['records'][-1] = copy.deepcopy(result['records'][0])
            elif failure == 'seed':
                result['records'][0]['seed'] = 102
            elif failure == 'subject':
                cases['cases'][0]['subject'] = 'SN999'
            else:
                result['records'][0]['output'] = str(adapter.V7_RUN / 'not_new.png')
            with self.assertRaises(ValueError):
                self.build(cases, result, run)
        with self.assertRaises(ValueError):
            e.scoped_path(e.EXP / 'final_test/forbidden.png')

    def test_actual_schedule_boundary_and_A_bindings(self):
        items = self.build(*self.fixtures())
        document = {'items': items}
        with patch.object(adapter.ADAPTER, '_validate', return_value=document), \
                patch.object(adapter, 'generation_configuration', return_value={}):
            report = adapter.validate(e.HERE)
        self.assertEqual(len(report['schedule_audit']), 48)
        for mutation in ('trace', 'boundary', 'timestep', 'count', 'condition', 'noise', 'projection', 'config'):
            invalid = copy.deepcopy(document)
            row = next(item['generation_record'] for item in invalid['items'] if item['role'] == 'new_generation')
            if mutation == 'trace':
                row['au_scale_trace'][-1]['au_scale'] = 1.5
            elif mutation == 'boundary':
                row['au_scale_trace'][10]['au_scale'] = 1.5
            elif mutation == 'timestep':
                row['actual_timesteps'][0] = 750
            elif mutation == 'count':
                row['actual_steps'] = 14
            elif mutation == 'condition':
                row['condition_hashes']['au_embedding'] = 'c' * 64
            elif mutation == 'noise':
                row['noise_sha256'] = 'c' * 64
            elif mutation == 'projection':
                row['projection_trace'][10]['au_scale_used'] = 1.5
            else:
                row['au_scale_initial'] = 1.0
            with patch.object(adapter.ADAPTER, '_validate', return_value=invalid), \
                    patch.object(adapter, 'generation_configuration', return_value={}), self.assertRaises(ValueError):
                adapter.validate(e.HERE)

    def test_frozen_metric_functions_and_facenet_loops_unchanged(self):
        def functions(source):
            return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef)}
        original = functions(adapter.ADAPTER.V5_CORE.read_text())
        derived = functions(adapter.ADAPTER.DERIVED_SOURCE)
        for name in ('au_metrics', 'extras', 'aggregate', 'load_openface', 'openface', 'facenet'):
            self.assertEqual(original[name], derived[name], name)
        output = self.output / 'facenet_binding'
        output.mkdir()
        e.save(output / 'items.json', {'items': [], 'test_used': False})
        model_source = (e.POLICY_DIR / 'run_facenet_single_directions.py').read_text()
        for count in (88, 135, 136):
            source = e.derived_facenet_source(output, count)
            self.assertIn(f'len(items) == {count}', source)
            for start, end in [('        detector = MTCNN(', "    save(out / 'protocol.json'"),
                               ('    measurements, embeddings = {}, {}', '    records = [{'),
                               ('    for row in records:\n        source_sha', '    coverage = {')]:
                self.assertEqual(model_source[model_source.index(start):model_source.index(end)],
                                 source[source.index(start):source.index(end)])
        for count in (87, 137):
            with self.assertRaises(ValueError):
                e.derived_facenet_source(output, count)

    def test_group_statistics_common_valid_and_noedit_NA(self):
        rows = []
        for subject in sorted(e.SUBJECTS):
            for group in ('no_edit', 'lowest_available'):
                for method in ('left', 'right'):
                    metrics = e.au_metrics([.2] * 12, [.2] * 12, [0] * 12,
                                           [0] * 12 if group == 'no_edit' else [1] + [0] * 11)
                    rows.append({'case_id': subject + '_' + group, 'subject': subject, 'group': group,
                                 'method': method, **{key: metrics.get(key) for key in e.METRICS},
                                 'identity_cosine': .9})
        stats = adapter.grouped_statistics(rows, ['left', 'right'])
        self.assertEqual(stats['full8']['identity_cosine']['common_cases'], 8)
        self.assertEqual(stats['full8']['target_mae_changed']['common_cases'], 4)
        self.assertEqual(stats['no_edit4']['target_mae_changed']['common_cases'], 0)
        self.assertIsNone(stats['no_edit4']['target_mae_changed']['subject_equal_means']['left'])
        self.assertEqual(stats['edited4']['target_mae_changed']['subject_count'], 4)
        rows[-1]['identity_cosine'] = None
        self.assertEqual(adapter.grouped_statistics(rows, ['left', 'right'])['full8']['identity_cosine']['common_cases'], 7)

    def test_merge136_failure_NA_buffalo_not_run_and_fair_pairs(self):
        cases, generation, run = self.fixtures()
        generation['status'] = 'completed_with_failures'
        generation['records'][0].update(status='failed', output=None, sha256=None)
        items = self.build(cases, generation, run)
        available = [item for item in items if item['image_available']]
        output = self.output / 'synthetic_merge'
        output.mkdir()
        for folder in ('openface', 'facenet'):
            (output / folder).mkdir()
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
            adapter.merge(output)
        report = e.read(output / 'comparison/results.json')
        stats = e.read(output / 'comparison/v9_statistics.json')
        self.assertEqual((len(report['records']), report['logical_records']), (136, 136))
        self.assertEqual(len(report['coverage']), 17)
        self.assertEqual(len(report['pairwise']), 91)
        failed = next(row for row in report['records'] if row['generation_status'] == 'failed')
        self.assertTrue(all(failed[key] is None for key in e.METRICS))
        self.assertTrue(all(row['buffalo_auxiliary']['status'] == 'not_run' for row in report['records']))
        self.assertEqual(len(stats['fair_baseline_pairs']), 3)
        self.assertEqual(len(stats['candidate_vs_a']), 6)
        self.assertEqual(set(stats['by_subject']), e.SUBJECTS)
        self.assertEqual(stats['frozen_comparison_sha256'], e.sha(output / 'comparison/results.json'))

    def test_prepare_all_failed_no_models_and_no_overwrite(self):
        cases, result, _ = self.fixtures()
        for row in result['records']:
            row.update(status='failed', output=None, sha256=None, error='synthetic_no_model_run')
        result['status'] = 'completed_with_failures'
        run = self.output / 'synthetic_all_failed_run'
        run.mkdir()
        config = {'kind': adapter.RESULT_KIND, 'test_data_used': False, 'training_performed': False,
                  'guidance_scale': 7.5, 'ip_scale': .9, 'seed': 101, 'nominal_steps': 20,
                  'configs': [{'name': name, **params} for name, params in adapter.CONFIGS.items()]}
        e.save(run / 'configuration.json', config)
        result['configuration'] = {'path': str(run / 'configuration.json'), 'sha256': e.sha(run / 'configuration.json')}
        e.save(run / 'cases.json', cases)
        e.save(run / 'result.json', result)
        output = self.output / 'synthetic_prepared'
        e.prepare(run, output)
        _, document = e.checked(output)
        self.assertEqual((document['logical_count'], document['measurement_count']), (136, 88))
        self.assertFalse((output / 'openface').exists())
        self.assertFalse((output / 'facenet').exists())
        with self.assertRaises(FileExistsError):
            e.prepare(run, output)
        for name in ('torch', 'facenet_pytorch', 'insightface'):
            self.assertNotIn(name, sys.modules)

    def test_configuration_cfg_bound_to_hash_and_run(self):
        output = self.output / 'synthetic_configuration'
        output.mkdir()
        config = {'kind': adapter.RESULT_KIND, 'test_data_used': False, 'training_performed': False,
                  'guidance_scale': 8, 'ip_scale': .9, 'seed': 101, 'nominal_steps': 20}
        e.save(output / 'configuration.json', config)
        e.save(output / 'result.json', {'configuration': {'path': str(output / 'configuration.json'),
                                                        'sha256': e.sha(output / 'configuration.json')}})
        document = {'generation_manifest_hashes': {str(output / 'result.json'): e.sha(output / 'result.json')}}
        with self.assertRaisesRegex(ValueError, 'guidance_scale'):
            adapter.generation_configuration(output, document)


if __name__ == '__main__':
    unittest.main(verbosity=2)
