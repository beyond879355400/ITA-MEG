"""Synthetic DEV19 schema/statistics tests; no model or generated image reads."""
import ast
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dev19_evaluate as d
import contract as c


def synthetic_new(prior):
    sources, cases, rows = {}, [], []
    for cid, meta in prior['metadata'].items():
        sid = meta['subject']
        image = {'path': meta['reference_frames']['source']['evaluation_path'], 'sha256': meta['reference_frames']['source']['evaluation_sha256']}
        sources.setdefault(sid, {'source_id': sid, 'subject': sid, 'canonical_source': image})
        cases.append({'case_id': cid, 'subject': sid, 'split': 'dev', 'source_id': sid,
            'source_au_vector_0_5': meta['source_labels_12'], 'target_au_vector_0_5': meta['target_labels_12']})
        for method in c.OURS:
            rows.append({'method': method, 'case_id': cid, 'subject': sid, 'split': 'dev', 'seed': 101,
                'source_aus_0_5': meta['source_labels_12'], 'target_aus_0_5': meta['target_labels_12'],
                'source_real': image, 'status': 'complete', 'execution_id': method + cid, 'execution_started': True,
                'output': '/synthetic/new/' + method + cid + '.png', 'sha256': 'a' * 64})
    result = {'kind': 'selected_v11_existing_dev19_completion', 'split': 'dev', 'status': 'complete',
        'sampling_seeds': [101], 'training_performed': False, 'target_pixels_used': False, 'target_latents_computed': False,
        'planned': 95, 'completed': 95, 'records': rows}
    config = {'kind': result['kind'], 'split': 'dev', 'sampling_seeds': [101], 'case_ids': list(prior['metadata']), 'variants': c.VARIANTS,
        'inputs': {'cases': prior['refs']['labels'], 'comparison': prior['refs']['manifest']}}
    cases_doc = {'kind': 'existing_dev19_source_only_cases', 'split': 'dev', 'case_count': 19,
        'sources': list(sources.values()), 'cases': cases}
    return result, cases_doc, config


class DevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = d.load_prior(verify_images=False)

    def test_fixed_original19_complete_metadata_and_36cells(self):
        self.assertEqual(len(self.prior['metadata']), 19)
        self.assertEqual(len(self.prior['reused_rows']), 114)
        self.assertEqual(len(self.prior['cells']), 36)
        self.assertEqual(sum(r['group'] == 'no_edit' for r in self.prior['metadata'].values()), 4)
        self.assertEqual({r['method'] for r in self.prior['reused_rows']}, set(d.EXTERNAL + ['source_real', 'target_real']))
        self.assertFalse(self.prior['images_sha_verified'])

    def test_new95_schema_exact_labels_single_seed(self):
        result, cases_doc, config = synthetic_new(self.prior)
        plan, records = d.validate_new_generation(result, cases_doc, config, self.prior)
        self.assertEqual((len(plan), len(records)), (95, 95))
        self.assertEqual({r['method'] for r in records.values()}, set(c.OURS))

    def test_reject_four_case_average_and_wrong_seed_or_source(self):
        for mutation in ('four', 'seed', 'source', 'label', 'role'):
            result, cases_doc, config = synthetic_new(self.prior)
            if mutation == 'four': config['case_ids'] = config['case_ids'][:4]
            elif mutation == 'seed': result['records'][0]['seed'] = 202
            elif mutation == 'source': result['records'][0]['source_real'] = {'path': '/bad.png', 'sha256': 'f' * 64}
            elif mutation == 'label': result['records'][0]['target_aus_0_5'] = [5] * 12
            else: result['records'][0]['split'] = c.ROLE
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                d.validate_new_generation(result, cases_doc, config, self.prior)

    def test_failed_and_missing_new_outputs_keep95_denominator(self):
        result, cases_doc, config = synthetic_new(self.prior)
        result['records'].pop()
        result['records'][0]['status'] = 'failed'
        result['completed'] = 93
        original = copy.deepcopy(result)
        plan, records = d.validate_new_generation(result, cases_doc, config, self.prior)
        self.assertEqual(len(plan), 95)
        self.assertEqual(len(records), 94)
        failed = next(r for r in records.values() if r['status'] == 'failed')
        self.assertIsNone(failed['output'])
        self.assertFalse(failed['failed_artifact']['eligible_for_evaluation'])
        self.assertEqual(result, original)

    def test_facenet_role_and_inputcounts_adapted_without_running(self):
        _, text = d.derive_facenet(d.TOOLS / 'synthetic_unused', 'a' * 64, 133, 19)
        tree = ast.parse(text)
        constants = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
        self.assertNotIn(c.ROLE, constants)
        self.assertIn('dev', constants)
        self.assertIn('thresholds=[0.6, 0.7, 0.7]', text)
        self.assertIn('num_classes=8631', text)
        self.assertIn('len(items) == 133', text)
        self.assertIn('len(sources) == 19', text)
        self.assertIn("'test_used': False", text)
        compile(text, '<dev19-test>', 'exec')

    def synthetic_report_rows(self):
        rows = copy.deepcopy(self.prior['reused_rows'])
        external = {(r['case_id'], r['method']): r for r in rows}
        for cid, meta in self.prior['metadata'].items():
            for method in c.OURS:
                row = copy.deepcopy(external[cid, 'magicface'])
                row.update(method=method, original_method=method, buffalo_identity_cosine=None,
                    synthetic_fixture=True, generation_status='complete')
                rows.append(row)
        return rows

    def test_nine_settings_and_same36strata_without_old_internal(self):
        rows = self.synthetic_report_rows()
        report = d.comparison_report(rows, [])
        self.assertEqual(len(report['records']), 209)
        self.assertEqual(len(report['methods']), 9)
        self.assertEqual(len(report['pairwise']), 30)
        self.assertEqual(len(report['au_level_all_methods']), 36)
        self.assertTrue(report['LP_is_one_method_two_settings'])
        self.assertFalse(report['independent_final_test'])
        self.assertTrue(report['no_new_buffalo'])
        self.assertNotIn('current_model', report['methods'])
        for method in d.METHODS:
            self.assertEqual(report['coverage'][method]['all']['planned'], 19)
            self.assertEqual(report['coverage'][method]['edited']['planned'], 15)
            self.assertEqual(report['coverage'][method]['no_edit']['planned'], 4)

    def test_subject_equal_not_image_weight_and_na_coverage(self):
        rows = self.synthetic_report_rows()
        subject_values = dict(zip(sorted({r['subject'] for r in rows}), (1., 2., 3., 4.)))
        for row in rows:
            if row['method'] in d.METHODS:
                row['target_mae_changed'] = None if row['group'] == 'no_edit' else subject_values[row['subject']]
        report = d.comparison_report(rows, [])
        metric = report['all_methods_common_valid']['all']['edited']['target_mae_changed']
        self.assertTrue(all(abs(value-2.5) < 1e-12 for value in metric['subject_equal_means'].values()))
        failed = next(r for r in rows if r['method'] == c.OURS[1] and r['group'] != 'no_edit')
        failed['vector'] = None
        for name in c.SOURCE_METRICS: failed[name] = None
        with_na = d.comparison_report(rows, [])
        self.assertEqual(with_na['coverage'][c.OURS[1]]['edited']['planned'], 15)
        self.assertEqual(with_na['coverage'][c.OURS[1]]['edited']['AU_valid'], 14)

    def test_main_frozen_sources_byte_identical(self):
        d.check_main()
        for name, expected in d.MAIN_PINS.items():
            self.assertEqual(c.sha(d.HERE / name), expected)


if __name__ == '__main__':
    unittest.main()
