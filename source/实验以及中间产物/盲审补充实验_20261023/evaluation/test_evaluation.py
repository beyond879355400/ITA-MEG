"""CPU-only tests: no torch/ML imports, no generation, no reference image reads."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import ablation_evaluation_contract as contract
import evaluate


def fixture():
    cases = {}
    for subject in range(8):
        sid = f'S{subject:02d}'
        for number in range(3):
            cid = f'{sid}_case{number}'
            cases[cid] = {'case_id': cid, 'subject': sid, 'source_labels_12': [0.] * 12,
                          'target_labels_12': [1.] + [0.] * 11,
                          'source_reference': {'status': 'complete', 'image':
                                               {'path': f'/reference/{sid}.png', 'sha256': 'a' * 64}}}
    plan = [{'arm': arm, 'subject': row['subject'], 'case_id': cid, 'seed': seed}
            for arm in contract.ARMS for cid, row in cases.items() for seed in contract.SEEDS]
    rows = [{**slot, 'status': 'success', 'source_path': cases[slot['case_id']]['source_reference']['image']['path'],
             'source_sha256': 'a' * 64, 'source_au12': [0.] * 12, 'target_au12': [1.] + [0.] * 11,
             'output_path': '/new/image.png', 'output_sha256': 'b' * 64} for slot in plan]
    return cases, {'slot_plan': plan, 'records': rows}


class SchemaTests(unittest.TestCase):
    def test_complete_grid(self):
        cases, generation = fixture()
        plan, rows = contract.normalize_generation(generation, cases)
        self.assertEqual(len(plan), 288)
        self.assertEqual(len(rows), 288)

    def test_missing_generation_row_is_kept_in_plan(self):
        cases, generation = fixture()
        generation['records'].pop()
        plan, rows = contract.normalize_generation(generation, cases)
        self.assertEqual(len(plan), 288)
        self.assertEqual(len(rows), 287)

    def test_failed_row_does_not_need_image(self):
        cases, generation = fixture()
        generation['records'][0].update(status='failed', output_path=None, output_sha256=None)
        plan, rows = contract.normalize_generation(generation, cases)
        self.assertEqual(len(plan), 288)
        self.assertEqual(rows[contract.row_key(generation['records'][0])]['status'], 'failed')

    def test_denominator_cannot_shrink(self):
        cases, generation = fixture()
        generation['slot_plan'].pop()
        with self.assertRaisesRegex(ValueError, 'denominator'):
            contract.normalize_generation(generation, cases)

    def test_duplicate_cannot_replace_missing_output(self):
        cases, generation = fixture()
        generation['records'][-1] = generation['records'][0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            contract.normalize_generation(generation, cases)

    def test_labels_cannot_be_changed(self):
        cases, generation = fixture()
        generation['records'][0]['target_au12'] = [0.] * 12
        with self.assertRaisesRegex(ValueError, 'labels changed'):
            contract.normalize_generation(generation, cases)

    def test_source_cannot_be_swapped(self):
        cases, generation = fixture()
        generation['records'][0]['source_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'Canonical source'):
            contract.normalize_generation(generation, cases)

    def test_real_reference_cases_are_frozen(self):
        cases = contract.reference_cases(contract.read(contract.REFERENCES))
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({c['subject'] for c in cases.values()}), 8)
        self.assertEqual(contract.sha(contract.REFERENCES), contract.REFERENCES_SHA)

    def test_subgroup_assignment_is_label_only(self):
        cases = contract.reference_cases(contract.read(contract.REFERENCES))
        plan = contract.subgroup_plan(cases)
        rows = list(plan['case_assignments'].values())
        self.assertEqual(sum(r['intensity_partition'] == 'all_active_intensity1' for r in rows), 7)
        self.assertEqual(sum(r['intensity_partition'] == 'has_intensity_gt1' for r in rows), 17)
        self.assertEqual(sum(r['combination_partition'] == 'multi_active_au' for r in rows), 15)
        self.assertEqual(sum(r['combination_partition'] == 'single_active_au' for r in rows), 9)
        self.assertIn(15, plan['missing_active_AUs'])
        self.assertEqual(plan['active_AU_case_counts']['15'], 0)
        self.assertFalse(plan['new_subjects_or_expanded_AU_coverage'])


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stats = contract.load_legacy('statistics')

    def test_metrics_match_direct_arithmetic(self):
        metric, extras = contract.metric_functions()
        source, target = [0.] * 12, [1., 2.] + [0.] * 10
        measured_source, predicted = [.1] * 12, [.8, 1.5] + [.3] * 10
        result = metric(predicted, measured_source, source, target)
        self.assertAlmostEqual(result['target_mae_changed'], .35)
        self.assertAlmostEqual(result['delta_mae_changed'], .45)
        self.assertAlmostEqual(result['unchanged_source_relative_drift'], .2)
        self.assertEqual(result['direction_correct_fraction'], 1.)
        result = metric(predicted, None, source, target)
        self.assertIsNotNone(result['target_mae_changed'])
        self.assertIsNone(result['unchanged_source_relative_drift'])
        _, extra = extras({str(au): .5 for au in contract.EXTRA}, None, None)
        self.assertIsNone(extra)

    def test_zero_edit_changed_metrics_are_na(self):
        metric, _ = contract.metric_functions()
        result = metric([.1] * 12, [.1] * 12, [0.] * 12, [0.] * 12)
        self.assertIsNone(result['target_mae_changed'])
        self.assertIsNone(result['delta_mae_changed'])
        self.assertEqual(result['unchanged_source_relative_drift'], 0.)

    def test_hierarchical_not_image_weighted(self):
        values = {('S1', 'A', 101): 0., ('S1', 'A', 202): 0.,
                  ('S1', 'B', 101): 10., ('S2', 'C', 101): 20.}
        result = self.stats.subject_aggregate(values)
        self.assertEqual(result['subject_means'], {'S1': 5., 'S2': 20.})
        self.assertEqual(result['mean'], 12.5)

    def test_sign_flip_uses_all_256_patterns(self):
        result = self.stats.subject_inference([1.] * 8, 20261023)
        self.assertEqual(result['sign_patterns'], 256)
        self.assertEqual(result['two_sided_sign_flip_p'], 2 / 256)
        self.assertEqual(result['bootstrap_samples'], 50000)
        self.assertEqual(result['difference_ci95'], [1., 1.])

    def test_holm_family_keeps_na_hypothesis(self):
        rows = [{'id': str(i), 'family': 'f', 'two_sided_sign_flip_p': p}
                for i, p in enumerate((.01, .04, None))]
        self.stats.holm_adjust(rows)
        self.assertAlmostEqual(rows[0]['holm_adjusted_p'], .03)
        self.assertAlmostEqual(rows[1]['holm_adjusted_p'], .08)
        self.assertIsNone(rows[2]['holm_adjusted_p'])
        self.assertTrue(all(row['holm_family_size'] == 3 for row in rows))

    def test_primary_family_is_three_du_tests(self):
        spec = contract.analysis_spec()
        self.assertEqual(len(spec['comparisons']), 3)
        self.assertTrue(all(c['metric'] == contract.PRIMARY for c in spec['comparisons']))
        self.assertEqual({c['reference'] for c in spec['comparisons']}, set(contract.ARMS[1:]))

    def test_facenet_adapter_changes_no_model_parameters(self):
        text = evaluate.derive_facenet(Path('/new/job'), 'a' * 64, 336, 24)
        ast.parse(text)
        self.assertIn('len(items) == 336', text)
        self.assertIn('len(sources) == 24', text)
        self.assertIn("source_sha = sources.get(row['case_id'])", text)
        self.assertNotIn('from run_facenet_v2', text)
        self.assertIn("device='cpu'", text)
        self.assertIn('thresholds=[0.6, 0.7, 0.7]', text)
        self.assertIn('image_size=160', text)
        self.assertIn('max(ranked, key=lambda r: (r[\'area\'], -r[\'index\']))', text)
        self.assertNotIn("'test_used': False", text)

    def test_measurement_sha_mismatch_is_rejected(self):
        item = {'method': 'full', 'subject': 'S1', 'case_id': 'C1', 'seed': 101, 'sha256': 'a' * 64}
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            evaluate.validate_measurements({'items': [item]}, {'records': [{**item, 'sha256': 'b' * 64}]})

    def test_failed_detector_row_is_retained(self):
        item = {'method': 'full', 'subject': 'S1', 'case_id': 'C1', 'seed': 101, 'sha256': 'a' * 64}
        result = evaluate.validate_measurements({'items': [item]}, {'records': [{**item, 'vector': None}]})
        self.assertEqual(len(result), 1)

    def test_complete_merge_preserves_failure_and_metric_na(self):
        cases, generation = fixture()
        generation['records'][0].update(status='failed', output_path=None, output_sha256=None)
        plan, normalized = contract.normalize_generation(generation, cases)
        items, availability = [], []
        for slot in plan:
            row = normalized[contract.row_key(slot)]
            availability.append({'key': list(contract.row_key(slot)),
                                 'status': 'available' if row['status'] == 'success' else 'failed'})
            if row['status'] == 'success':
                items.append({**slot, 'sha256': row['output_sha256']})
        for cid, case in cases.items():
            for method in ('source_real', 'target_real'):
                items.append({'method': method, 'subject': case['subject'], 'case_id': cid,
                              'seed': 0, 'sha256': 'c' * 64})
        document = {'planned': plan, 'normalized_generation_records': generation['records'],
                    'items': items, 'availability': availability, 'case_metadata': cases}
        au = {'records': [{**item, 'vector': [.1] * 12, 'parse_status': 'primary_face_finite',
                           'selected_au17': {str(number): .1 for number in contract.EXTRA}}
                          for item in items]}
        face = {'records': [{**item, 'identity_cosine': .9, 'status': 'valid'} for item in items]}
        # One generated image has a failed AU detector but a valid independent identity score.
        au['records'][0].update(vector=None, selected_au17=None, parse_status='no_face')
        rows = evaluate.merge_metrics(document, au, face)
        self.assertEqual(len(rows), 288)
        self.assertEqual(rows[0]['status'], 'unavailable')
        self.assertIsNone(rows[0]['metrics'][contract.PRIMARY])
        self.assertEqual(rows[1]['status'], 'complete')
        self.assertIsNone(rows[1]['metrics'][contract.PRIMARY])
        self.assertEqual(rows[1]['metrics']['identity_cosine'], .9)
        result = self.stats.analyze({'kind': 'paired_subject_metric_input',
                                    'analysis_spec': contract.analysis_spec(), 'planned': plan, 'records': rows})
        coverage = next(r for r in result['coverage_table'] if r['method'] == 'full' and r['metric'] == contract.PRIMARY)
        self.assertEqual(coverage['planned_seed_outputs'], 72)
        self.assertEqual(coverage['valid_metric_outputs'], 70)
        self.assertEqual(coverage['valid_subjects'], 8)
        self.assertEqual(len(result['comparisons']), 3)
        self.assertTrue(all(row['holm_family_size'] == 3 for row in result['comparisons']))

    def test_factorial_interaction_uses_four_way_common_keys(self):
        rows = []
        for subject in range(8):
            for arm, value in zip(contract.ARMS, (4., 2., 1., 0.)):
                rows.append({'method': arm, 'subject': f'S{subject}', 'case_id': f'C{subject}',
                             'seed': 101, 'status': 'complete', 'metrics': {contract.PRIMARY: value}})
        rows[-1]['metrics'][contract.PRIMARY] = None
        result = evaluate.factorial_exploration(rows, self.stats)[contract.PRIMARY]
        self.assertEqual(result['n_subjects'], 7)
        self.assertEqual(result['mean'], 1.)
        self.assertEqual(len(result['missing_units']), 1)
        self.assertEqual(result['inference']['sign_patterns'], 128)
        self.assertTrue(result['exploratory'])

    def test_subgroup_denominators_and_metric_specific_na(self):
        cases, generation = fixture()
        plan = contract.subgroup_plan(cases)
        planned, _ = contract.normalize_generation(generation, cases)
        rows = [{**slot, 'status': 'complete', 'metrics': {metric: 1. for metric in plan['metrics']}}
                for slot in planned]
        rows[0]['metrics'][contract.PRIMARY] = None
        result = evaluate.subgroup_summary(rows, plan, self.stats)
        self.assertEqual(len(result), 48)
        drift = next(row for row in result if row['method'] == 'full' and row['metric'] == contract.PRIMARY
                     and row['subgroup'] == 'all_active_intensity1')
        identity = next(row for row in result if row['method'] == 'full' and row['metric'] == 'identity_cosine'
                        and row['subgroup'] == 'all_active_intensity1')
        self.assertEqual(drift['planned_cases'], 24)
        self.assertEqual(drift['planned_seed_outputs'], 72)
        self.assertEqual(drift['valid_seed_outputs'], 71)
        self.assertEqual(identity['valid_seed_outputs'], 72)
        self.assertTrue(all(row['descriptive_only'] for row in result))

    def calibration_fixture(self):
        cases, items, measured = {}, [], []
        for cid, subject, target, source_sha, target_sha, prediction in (
                ('C1', 'S1', 1., 'a', 'b', 3.), ('C2', 'S1', 2., 'a', 'c', 4.),
                ('C3', 'S2', 1., 'd', 'e', 1.)):
            case = {'case_id': cid, 'subject': subject, 'source_labels_12': [0.] * 12,
                    'target_labels_12': [target] + [0.] * 11}
            for role, digest, value in (('source', source_sha * 64, 0.), ('target', target_sha * 64, prediction)):
                reference = {'path': '/reference/' + digest + '.png', 'sha256': digest}
                case[role + '_reference'] = {'status': 'complete', 'image': reference}
                item = {'case_id': cid, 'subject': subject, 'method': role + '_real', 'seed': 0, **reference}
                items.append(item)
                measured.append({**item, 'vector': [value] + [0.] * 11, 'parse_status': 'primary_face_finite'})
            cases[cid] = case
        return {'items': items, 'case_metadata': cases, 'real_reference_calibration': contract.calibration_plan()}, {'records': measured}

    def test_real_calibration_deduplicates_and_weights_subjects(self):
        document, au = self.calibration_fixture()
        result = evaluate.real_reference_calibration(document, au, self.stats)
        self.assertEqual(result['unique_reference_images'], 5)
        self.assertEqual(len(result['physical_records'][0]['logical_references']), 2)
        level1 = next(row for row in result['summary'] if row['AU'] == 1 and row['manual_intensity_stratum'] == 1)
        overall = next(row for row in result['summary'] if row['AU'] == 1 and row['manual_intensity_stratum'] == 'all')
        self.assertEqual(level1['subject_equal_MAE'], 1.)
        self.assertEqual(level1['subject_equal_bias'], 1.)
        self.assertEqual(level1['valid_subjects'], 2)
        self.assertAlmostEqual(overall['subject_equal_MAE'], 2 / 3)
        self.assertEqual(overall['valid_unique_images'], 5)

    def test_real_calibration_failure_preserves_planned_denominator(self):
        document, au = self.calibration_fixture()
        au['records'][1].update(vector=None, parse_status='no_face')
        result = evaluate.real_reference_calibration(document, au, self.stats)
        level1 = next(row for row in result['summary'] if row['AU'] == 1 and row['manual_intensity_stratum'] == 1)
        self.assertEqual(level1['planned_unique_images'], 2)
        self.assertEqual(level1['valid_unique_images'], 1)
        self.assertEqual(level1['planned_subjects'], 2)
        self.assertEqual(level1['valid_subjects'], 1)

    def test_real_calibration_label_conflict_is_not_guessed(self):
        document, au = self.calibration_fixture()
        # The same physical source image cannot carry two different manual labels.
        document['case_metadata']['C2']['source_labels_12'] = [1.] + [0.] * 11
        result = evaluate.real_reference_calibration(document, au, self.stats)
        self.assertEqual(len(result['reference_label_conflicts']), 1)
        self.assertIsNone(result['reference_label_conflicts'][0]['manual_labels_12'])
        self.assertIsNone(result['reference_label_conflicts'][0]['measured_12'])

    def test_file_seals_and_exclusive_outputs(self):
        with tempfile.TemporaryDirectory(dir=contract.HERE, prefix='unit_tmp_') as folder:
            path = Path(folder) / 'sample.json'
            contract.save(path, {'value': 1})
            bound = contract.ref(path)
            self.assertEqual(contract.verified(bound), path)
            with self.assertRaises(FileExistsError):
                contract.save(path, {'value': 2})
            with self.assertRaises(ValueError):
                contract.verified({**bound, 'sha256': '0' * 64})

    def test_no_ml_packages_imported(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('facenet_pytorch', sys.modules)
        self.assertNotIn('diffusers', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
