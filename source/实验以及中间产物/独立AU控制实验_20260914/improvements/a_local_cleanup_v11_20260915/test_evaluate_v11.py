#!/usr/bin/env python3
"""V11 synthetic metadata and frozen-formula tests; no model inference."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluate_v11 as adapter
e = adapter.CORE


class V11EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = e.HERE / 'synthetic_tests'
        parent.mkdir(exist_ok=True)
        cls.output = Path(tempfile.mkdtemp(prefix='v11_evaluator_', dir=parent))

    def fixtures(self):
        run = self.output / 'not_a_real_generation'
        cases = adapter.frozen_cases()
        supports, _ = adapter.support_bundle()
        mask_refs = {row['case_id']: row for row in supports['cases']}
        previous = e.read(adapter.V10 / 'runs/terminal_202609151940/result.json')
        old = {(row['case_id'], row['method']): row for row in previous['records']}
        records = []
        for case in cases:
            case['cleanup_mask'] = mask_refs[case['case_id']]
            for method in e.METHODS:
                family, config = method.split('_', 1)
                row = copy.deepcopy(old[(case['case_id'], family + '_terminal_last_eps')])
                row.update(method=method, config=config, **adapter.CONFIGS[config], denoising_steps=15,
                           deis_unet_calls=15, total_unet_calls=15,
                           output=str(run / (case['case_id'] + '_' + method + '.png')), sha256='d' * 64)
                row['cleanup_mask'] = copy.deepcopy(case['cleanup_mask'])
                row['original_a_projection_prefix'] = copy.deepcopy(row['projection_trace'][:14])
                row['a_prefix_verified'] = True
                row['terminal_trace'].update(mode=config, filter=adapter.filter_config(row['cleanup_sigma']),
                                             cleanup_mask=copy.deepcopy(case['cleanup_mask']), base_mode='terminal_last_eps',
                                             base_terminal_clean_latent_sha256=row['terminal_trace']['clean_latent_sha256'],
                                             outside_cleanup_clean_exact=True)
                records.append(row)
        return ({'kind': adapter.CASE_KIND, 'test_data_used': False, 'case_count': 8, 'cases': cases},
                {'kind': adapter.RESULT_KIND, 'test_data_used': False, 'status': 'complete', 'records': records}, run)

    def configuration(self):
        return {'kind': adapter.RESULT_KIND, 'test_data_used': False, 'training_performed': False,
                'guidance_scale': 7.5, 'ip_scale': .9, 'seed': 101, 'nominal_steps': 20,
                'configs': [{'name': name, **params} for name, params in adapter.CONFIGS.items()],
                'cleanup_mask_manifest': {'path': str(adapter.MASK_MANIFEST), 'sha256': adapter.MASK_MANIFEST_SHA}}

    def build(self, cases, result, run):
        with patch.object(e, 'checked_image', side_effect=lambda ref: dict(ref)):
            return e.build_items(cases, result, run)

    def test120_rows_32_new_88_references(self):
        report = adapter.metadata()
        self.assertEqual((report['logical_records'], report['new_generated_planned'], report['reference_count']), (120, 32, 88))
        rows = self.build(*self.fixtures())
        self.assertEqual(len(rows), 120)
        self.assertEqual(sum(row['role'] == 'new_generation' for row in rows), 32)
        self.assertEqual(len({row['sample_id'] for row in rows}), 120)
        self.assertEqual(len(e.METRICS), 14)

    def test_invalid_old_case_and_nonterminal_rejected(self):
        cases = adapter.frozen_cases()
        cases[0]['a_bidirectional']['path'] += '.changed'
        with self.assertRaises(ValueError):
            adapter.reference_lineage(cases)
        for mutation in ('running', 'duplicate', 'count48', 'test'):
            cases, result, run = self.fixtures()
            if mutation == 'running':
                result['status'] = 'running'
            elif mutation == 'duplicate':
                result['records'][-1] = copy.deepcopy(result['records'][0])
            elif mutation == 'count48':
                result['records'].extend(copy.deepcopy(result['records'][:16]))
            else:
                cases['cases'][0]['split'] = 'test'
            with self.assertRaises(ValueError):
                self.build(cases, result, run)

    def test_frozen_score_functions_and_model_loop_unchanged(self):
        def functions(source):
            return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef)}
        old, new = functions(adapter.BASE.ADAPTER.DERIVED_SOURCE), functions(adapter.DERIVED_SOURCE)
        for name in ('au_metrics', 'extras', 'aggregate', 'load_openface', 'openface', 'facenet'):
            self.assertEqual(old[name], new[name], name)
        self.assertEqual(old['merge'].split('    methods = REFERENCES')[0], new['merge'].split('    methods = REFERENCES')[0])
        folder = self.output / 'facenet_binding'
        folder.mkdir()
        e.save(folder / 'items.json', {'items': [], 'test_used': False})
        source = (e.POLICY_DIR / 'run_facenet_single_directions.py').read_text()
        for count in (88, 119, 120):
            derived = e.derived_facenet_source(folder, count)
            start, end = '    measurements, embeddings = {}, {}', '    records = [{'
            self.assertEqual(source[source.index(start):source.index(end)], derived[derived.index(start):derived.index(end)])
            self.assertIn(f'len(items) == {count}', derived)
        for count in (87, 121, 136):
            with self.assertRaises(ValueError):
                e.derived_facenet_source(folder, count)

    def test_merge120_66_pairs_two_fair_pairs_and_NA(self):
        cases, result, run = self.fixtures()
        result['status'] = 'completed_with_failures'
        result['records'][0].update(status='failed', output=None, sha256=None)
        items = self.build(cases, result, run)
        available = [row for row in items if row['image_available']]
        output = self.output / 'merge'
        output.mkdir()
        for folder in ('openface', 'facenet'):
            (output / folder).mkdir()
        for name in ('manifest.json', 'items.json', 'input_seal.json'):
            e.save(output / name, {'synthetic': True})
        aus = [{**row, 'vector': [.2] * 12, 'selected_au17': {str(au): .2 for au in e.AUS + e.EXTRA},
                'all_candidates': [], 'raw_rows': [], 'face_count': 1} for row in available]
        faces = [{**row, 'identity_cosine': .9, 'candidates': [], 'face_count': 1} for row in available]
        e.save(output / 'openface/results.json', {'status': 'complete', 'test_used': False, 'policy_sha256': e.POLICY_SHA, 'records': aus})
        e.save(output / 'facenet/results.json', {'summary': {'status': 'complete'}, 'test_used': False, 'policy_sha256': e.POLICY_SHA, 'records': faces})
        with patch.object(e, 'checked', return_value=(output, {'items': items, 'generation_status': result['status']})):
            adapter.merge(output)
        report = e.read(output / 'comparison/results.json')
        stats = e.read(output / 'comparison/v11_statistics.json')
        self.assertEqual((len(report['records']), report['logical_records'], report['new_generated_planned']), (120, 120, 32))
        self.assertEqual(len(report['pairwise']), 66)
        self.assertEqual(len(stats['fair_baseline_pairs']), 2)
        self.assertEqual(len(stats['candidate_vs_a']), 4)
        self.assertEqual(set(stats['by_subject']), e.SUBJECTS)
        failed = next(row for row in report['records'] if row['generation_status'] == 'failed')
        self.assertTrue(all(failed[metric] is None for metric in e.METRICS))
        self.assertTrue(all(row['buffalo_auxiliary']['status'] == 'not_run' for row in report['records']))
        for name in ('torch', 'facenet_pytorch', 'insightface'):
            self.assertNotIn(name, sys.modules)

    def test32_complete_local_terminal_traces(self):
        cases, generation, _ = self.fixtures()
        by_case = {row['case_id']: row for row in cases['cases']}
        _, original = adapter.BASE.frozen_v7()
        results = []
        for row in generation['records']:
            family = row['method'].split('_', 1)[0]
            results.append(adapter.audit_local_terminal(row, original[row['case_id'], family + '_smooth_s075_a150'], by_case[row['case_id']]))
        self.assertEqual(len(results), 32)
        self.assertEqual(sum(row['total_unet_calls'] for row in results), 480)

    def test_source_only_receipt_and_active_AU_negative_cases(self):
        cases, _, _ = self.fixtures()
        case = next(row for row in cases['cases'] if row['group'] != 'no_edit')
        bundle = adapter.support_bundle()
        path = Path(case['cleanup_mask']['receipt']['path'])
        original_read = e.read
        original = original_read(path)
        for mutation in ('target_geometry', 'pixels', 'source_points', 'active_AU', 'mask_bytes'):
            changed = copy.deepcopy(original)
            if mutation == 'target_geometry':
                changed['target_geometry_used'] = True
            elif mutation == 'pixels':
                changed['generated_pixels_used'] = True
            elif mutation == 'source_points':
                changed['source_landmarks_68_512'][0][0] += 1
            elif mutation == 'active_AU':
                changed['active_regions'] = ['chin']
            else:
                changed['mask_bytes_sha256'] = 'a' * 64
            with patch.object(e, 'read', side_effect=lambda p: changed if Path(p) == path else original_read(p)), \
                    self.assertRaises(ValueError, msg=mutation):
                adapter.validate_cleanup_metadata(case, bundle)

    def test_local_sigma_support_and_unet_mutations_rejected(self):
        cases, generation, _ = self.fixtures()
        row = next(row for row in generation['records'] if row['group'] != 'no_edit')
        case = next(case for case in cases['cases'] if case['case_id'] == row['case_id'])
        _, original = adapter.BASE.frozen_v7()
        prior = original[row['case_id'], row['method'].split('_', 1)[0] + '_smooth_s075_a150']
        for mutation in ('extra', 'sigma', 'support', 'filter', 'prefix', 'source', 'epsilon'):
            changed = copy.deepcopy(row)
            if mutation == 'extra':
                changed['terminal_unet_calls'] = 1
            elif mutation == 'sigma':
                changed['cleanup_sigma'] = .75
            elif mutation == 'support':
                changed['cleanup_mask']['sha256'] = 'a' * 64
            elif mutation == 'filter':
                changed['terminal_trace']['filter']['outside_cleanup_exact'] = False
            elif mutation == 'prefix':
                changed['projection_trace'][0]['latent_sha256'] = 'a' * 64
            elif mutation == 'source':
                changed['terminal_trace']['epsilon_source'] = 'extra_t0'
            else:
                changed['terminal_trace']['last_deis_raw_epsilon_sha256'] = 'a' * 64
            with self.assertRaises(ValueError, msg=mutation):
                adapter.audit_local_terminal(changed, prior, case)

    def test_all_failed_prepare120_and_fresh_seal(self):
        cases, result, _ = self.fixtures()
        for row in result['records']:
            row.update(status='failed', output=None, sha256=None)
        result['status'] = 'completed_with_failures'
        run = self.output / 'all_failed'
        run.mkdir()
        e.save(run / 'configuration.json', self.configuration())
        result['configuration'] = {'path': str(run / 'configuration.json'), 'sha256': e.sha(run / 'configuration.json')}
        e.save(run / 'cases.json', cases)
        e.save(run / 'result.json', result)
        output = self.output / 'prepared'
        e.prepare(run, output)
        _, document = e.checked(output)
        self.assertEqual((document['logical_count'], document['measurement_count']), (120, 88))
        self.assertEqual(document['terminal_call_audit'], [])
        self.assertEqual(len(document['cleanup_support_audit']), 8)
        with self.assertRaises(FileExistsError):
            e.prepare(run, output)
        self.assertFalse((output / 'openface').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
