"""Synthetic metadata/CSV/statistical tests only; no new-case image reads."""
import copy
import csv
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import verify_completed as verify
import statistics_check as statistics


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = verify.read(verify.COMPLETION / 'protocol/proposal_v1/cases_label_only.json')['cases']
        cls.cases = {r['case_id']: {'case_id': r['case_id'], 'subject': r['subject'], 'split': verify.ROLE,
            'group': 'no_edit' if r['source_au_vector_0_5'] == r['target_au_vector_0_5'] else 'edited',
            'source_labels_12': r['source_au_vector_0_5'], 'target_labels_12': r['target_au_vector_0_5']} for r in cls.labels}
        cls.contract = module(verify.COMPLETION / 'evaluation/contract.py', 'test_only_completion_contract')
        cls.engine = module(verify.EXP / 'statistics/paired_subject_statistics.py', 'test_only_frozen_statistics_engine')

    def indexes(self):
        indexes = []
        for methods in [verify.OURS, ['magicface'], ['faceadapter'], ['liveportrait']]:
            rows = []
            for method in methods:
                for cid, case in self.cases.items():
                    for seed in verify.SEEDS:
                        actual = 101 if method == 'liveportrait' else seed
                        execution = method + '/' + cid + '_seed' + str(actual)
                        digest = hashlib.sha256(execution.encode()).hexdigest()
                        rows.append({'method': method, 'subject': case['subject'], 'case_id': cid, 'seed': seed,
                            'split': verify.ROLE, 'status': 'complete', 'execution_id': execution, 'execution_started': True,
                            'source_aus_0_5': case['source_labels_12'], 'target_aus_0_5': case['target_labels_12'],
                            'output': {'path': '/synthetic/' + execution + '.png', 'sha256': digest},
                            'alias_of': {'method': method, 'case_id': cid, 'seed': 101} if seed != actual else None})
            indexes.append({'status': 'complete', 'usable_for_evaluation': True, 'records': rows,
                'method_names': methods, 'case_ids': list(self.cases), 'sampling_seeds': verify.SEEDS})
        return indexes

    def test_full704_physical768_logical_with64_lp_aliases(self):
        rows, accounting = verify.normalize_generation(self.indexes(), self.cases)
        self.assertEqual(len(rows), 768)
        self.assertEqual(sum(r['planned_physical_executions'] for r in accounting.values()), 704)
        self.assertEqual(sum(r['executed_physical_calls'] for r in accounting.values()), 704)
        self.assertEqual(accounting['liveportrait']['alias_rows'], 64)
        self.assertEqual(accounting['liveportrait']['successful_physical_calls'], 32)

    def test_alias_output_change_is_rejected(self):
        indexes = self.indexes()
        indexes[-1]['records'][1]['output']['sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            verify.normalize_generation(indexes, self.cases)

    def test_missing_failure_and_nonblind_role_denominators(self):
        indexes = self.indexes()
        lost = indexes[0]['records'].pop()
        indexes[1]['records'][0].update(status='failed', execution_started=False, output=None)
        rows, accounting = verify.normalize_generation(indexes, self.cases)
        self.assertEqual(len(rows), 767)
        self.assertEqual(accounting[lost['method']]['missing_logical_rows'], 1)
        self.assertEqual(sum(r['planned_physical_executions'] for r in accounting.values()), 704)
        indexes[0]['records'][0]['split'] = 'test'
        with self.assertRaises(ValueError):
            verify.normalize_generation(indexes, self.cases)

    def test_failed_artifact_not_treated_as_success(self):
        indexes = self.indexes()
        indexes[0]['records'][0]['status'] = 'failed'
        rows, _ = verify.normalize_generation(indexes, self.cases)
        self.assertIsNone(rows[0]['output'])
        self.assertIn('failed_artifact', rows[0])

    def test_quarantined_index_rejected(self):
        indexes = self.indexes(); indexes[-1]['usable_for_evaluation'] = False
        with self.assertRaises(ValueError):
            verify.normalize_generation(indexes, self.cases)

    def test_missing_terminal_results_never_opens_images(self):
        with mock.patch.object(verify.HashAudit, 'ref') as read_artifact:
            with self.assertRaisesRegex(ValueError, 'Terminal export absent'):
                verify.verify(verify.COMPLETION / 'evaluation/UNCREATED_SYNTHETIC_JOB', verify.COMPLETION / 'frozen_protocol.json', verify.FREEZE_SHA)
            read_artifact.assert_not_called()

    def test_csv_largest_geometry_invalid_is_na_not_alternate_face(self):
        core = verify.independent_core(); reader = core.load_csv_reader()
        columns = ['confidence'] + [f'x_{i}' for i in range(68)] + [f'y_{i}' for i in range(68)] + [f'AU{a:02d}_r' for a in (*verify.AUS, *verify.EXTRA)]
        def face(size, confidence, bad=False):
            row = {'confidence': confidence}
            row.update({f'x_{i}': i * size / 67 for i in range(68)})
            row.update({f'y_{i}': (i % 2) * size for i in range(68)})
            row.update({f'AU{a:02d}_r': 1 for a in (*verify.AUS, *verify.EXTRA)})
            if bad: row['AU01_r'] = 'nan'
            return row
        with tempfile.TemporaryDirectory(dir=verify.HERE) as directory:
            path = Path(directory) / 'synthetic.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
                writer.writerows([face(10, .99), face(20, .01, True)])
            rows, selected, vector, values, confidence = reader(path)
            self.assertEqual(selected, 1)
            self.assertIsNone(vector)
            self.assertIsNone(values['1'])
            self.assertEqual(confidence, .01)
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
                writer.writerows([face(20, .01, True), face(20, .99)])
            self.assertEqual(reader(path)[1], 0)

    def test_au_arithmetic_changed_unchanged_and_extra_na(self):
        core = verify.independent_core()
        source = [0] * 12; target = [0] * 12; target[0] = 2
        prediction = [0] * 12; prediction[0] = 1
        values = core.au_metrics(prediction, source, source, target)
        self.assertEqual(values['target_mae_changed'], 1)
        self.assertEqual(values['delta_mae_changed'], 1)
        self.assertEqual(values['direction_correct_fraction'], 1)
        self.assertEqual(values['unchanged_source_relative_drift'], 0)
        self.assertIsNone(values['extra5_proxy_mae_to_real_target'])
        noedit = core.au_metrics(source, source, source, source)
        self.assertIsNone(noedit['target_mae_changed'])
        self.assertEqual(noedit['target_mae_all12'], 0)

    def statistical_fixture(self):
        indexes = self.indexes()
        # Missing output, missing whole comparator subject, and a metric NA.
        indexes[0]['records'].pop()
        for row in indexes[1]['records']:
            if row['subject'] == 'SN032': row.update(status='failed', output=None, execution_started=False)
        rows, accounting = verify.normalize_generation(indexes, self.cases)
        planned = self.contract.plan_rows(self.cases)
        exported = []
        spec = self.contract.analysis_spec()
        source_metrics = {v['source_metric'] for v in spec['metrics'].values()}
        for p in planned:
            sid = verify.SUBJECTS.index(p['subject']); mid = verify.METHODS.index(p['method'])
            values = {m: (sid + 1) / 10 + mid / 100 + p['seed'] / 10000 for m in source_metrics}
            if p['seed'] == 202 and sid == 2: values['target_mae_changed'] = None
            if self.cases[p['case_id']]['group'] == 'no_edit':
                for m in ('target_mae_changed', 'delta_mae_changed', 'direction_correct_fraction'): values[m] = None
            exported.append({**self.cases[p['case_id']], **p, **values})
        doc = {'protocol_sha256': verify.FREEZE_SHA, 'case_metadata': self.cases, 'planned': planned,
            'normalized_generation_records': rows, 'execution_accounting': accounting}
        inputs = self.contract.statistics_document(doc, exported)
        result = self.contract.enrich_coverage(self.engine.analyze(inputs), inputs, self.cases)
        return inputs, result, planned, spec

    def test_statistics_match_frozen_engine_with_scope_na_and_partial_subjects(self):
        inputs, result, planned, spec = self.statistical_fixture()
        receipt = statistics.verify_statistics(inputs, result, inputs['records'], planned, self.cases, spec)
        self.assertEqual(receipt['raw_method_metric_aggregates'], 336)
        self.assertEqual(receipt['paired_comparisons'], 7)
        self.assertEqual(receipt['family_sizes'], {'external_primary_C': 3, 'internal_primary_C': 4})
        magic = next(r for r in result['comparisons'] if r['reference'] == 'magicface')
        self.assertEqual(magic['n_subjects'], 7)
        self.assertEqual(magic['sign_patterns'], 128)

    def test_bootstrap_ci_tampering_is_detected(self):
        inputs, result, planned, spec = self.statistical_fixture()
        result['comparisons'][0]['difference_ci95'][0] += .01
        with self.assertRaisesRegex(ValueError, 'difference_ci95'):
            statistics.verify_statistics(inputs, result, inputs['records'], planned, self.cases, spec)

    def test_holm_keeps_all_na_hypothesis_in_declared_family(self):
        inputs, _, planned, spec = self.statistical_fixture()
        for row in inputs['records']:
            if row['method'] == 'base_local_lp_s050':
                row['metrics'][self.contract.PRIMARY] = None
                row['metric_reasons'][self.contract.PRIMARY] = 'measurement_or_reference_NA'
        result = self.contract.enrich_coverage(self.engine.analyze(inputs), inputs, self.cases)
        statistics.verify_statistics(inputs, result, inputs['records'], planned, self.cases, spec)
        case = next(r for r in result['comparisons'] if r['reference'] == 'base_local_lp_s050')
        self.assertIsNone(case['holm_adjusted_p'])
        self.assertEqual(case['holm_family_size'], 4)

    def test_inference_zero_one_and_full8_subjects(self):
        self.assertIsNone(statistics.inference([], 20260915)['difference_ci95'])
        self.assertIsNone(statistics.inference([1], 20260915)['two_sided_sign_flip_p'])
        result = statistics.inference([1] * 8, 20260915)
        self.assertEqual(result['sign_patterns'], 256)
        self.assertEqual(result['two_sided_sign_flip_p'], 2 / 256)
        self.assertEqual(result['difference_ci95'], [1., 1.])

    def test_end_to_end_full_synthetic_terminal_artifact_graph(self):
        import synthetic_fixture
        with tempfile.TemporaryDirectory(dir=verify.HERE) as directory:
            root = Path(directory)
            job, freeze, plan = synthetic_fixture.build(root, self.labels, self.indexes(), self.contract, self.engine)
            with mock.patch.object(verify, 'COMPLETION', root), mock.patch.object(verify, 'FREEZE_SHA', freeze['sha256']), \
                    mock.patch.object(verify, 'LABEL_SHA', plan['sha256']):
                report = verify.verify(job, freeze['path'], freeze['sha256'])
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(report['planned_logical_outputs'], 768)
            self.assertEqual(report['actual_started_physical_jobs'], 704)
            self.assertEqual(report['manual_label_files_verified'], 96)
            self.assertEqual(report['raw_method_metric_aggregates'], 336)
            self.assertFalse(report['identity_verification_scope']['facenet']['cosine_independently_recomputed'])


if __name__ == '__main__':
    unittest.main()
