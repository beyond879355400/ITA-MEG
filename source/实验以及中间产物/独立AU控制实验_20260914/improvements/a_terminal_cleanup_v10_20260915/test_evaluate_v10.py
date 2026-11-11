#!/usr/bin/env python3
"""Synthetic v10 metadata/aggregation tests; never run model inference."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluate_v10 as adapter

e = adapter.CORE


class V10EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = e.HERE / 'synthetic_tests'
        parent.mkdir(exist_ok=True)
        cls.output = Path(tempfile.mkdtemp(prefix='v10_evaluator_', dir=parent))

    def configuration(self):
        return {'kind': adapter.RESULT_KIND, 'test_data_used': False, 'training_performed': False,
                'seed': 101, 'nominal_steps': 20, 'guidance_scale': 7.5, 'ip_scale': .9,
                'configs': [{'name': name, **params} for name, params in adapter.CONFIGS.items()],
                'denoising_steps': 15, 'scheduler_sigmas_mutated': False,
                'terminal_scheduler_step_calls': 0, 'terminal_add_noise_calls': 0,
                'terminal_filter': adapter.FILTER}

    def fixtures(self):
        run = self.output / 'not_a_real_generation'
        cases = adapter.BASE.frozen_cases()
        _, old = adapter.BASE.frozen_v7()
        records = []
        for case in cases:
            for method in e.METHODS:
                family, config = method.split('_', 1)
                row = copy.deepcopy(old[(case['case_id'], family + '_smooth_s075_a150')])
                extra = adapter.CONFIGS[config]['terminal_unet_calls']
                projected = row['source_latent_sha256'] if row['group'] == 'no_edit' else 'f' * 64
                trace = [{'phase': 'deis' if index < 15 else 'terminal', 't': timestep, 'au_scale': 1.5,
                          'cfg_batch': 2, 'raw_epsilon_sha256': f'{index + 1:064x}', 'prompt_embeds_sha256': 'e' * 64}
                         for index, timestep in enumerate(adapter.TIMESTEPS + ([0] if extra else []))]
                terminal = {'mode': config, 'last_deis_timestep': 50, 'rho0': .02, 'alpha0': 1.0, 'sigma0': .02,
                            'epsilon_source': 'extra_t0' if extra else 'last_t50', 'input_latent_sha256': 'a' * 64,
                            'epsilon_cfg_sha256': 'b' * 64, 'last_deis_raw_epsilon_sha256': trace[14]['raw_epsilon_sha256'],
                            'terminal_raw_epsilon_sha256': trace[-1]['raw_epsilon_sha256'] if extra else None,
                            'clean_latent_sha256': 'c' * 64,
                            'post_filter_clean_latent_sha256': ('d' if config == 'terminal_t0_lp' else 'c') * 64,
                            'projected_latent_sha256': projected, 'extra_cfg': 7.5 if extra else None,
                            'filter': copy.deepcopy(adapter.FILTER) if config == 'terminal_t0_lp' else None,
                            'scheduler_step_index_before': 20, 'scheduler_step_index_after': 20}
                row['projection_trace'][-1]['latent_sha256'] = projected
                row.update(method=method, config=config, **adapter.CONFIGS[config], denoising_steps=15,
                           deis_unet_calls=15, total_unet_calls=15 + extra, unet_call_trace=trace,
                           terminal_trace=terminal, output=str(run / (case['case_id'] + '_' + method + '.png')),
                           sha256='d' * 64)
                records.append(row)
        return ({'kind': adapter.CASE_KIND, 'test_data_used': False, 'case_count': 8, 'cases': cases},
                {'kind': adapter.RESULT_KIND, 'test_data_used': False, 'status': 'complete', 'records': records}, run)

    def build(self, cases, result, run):
        with patch.object(e, 'checked_image', side_effect=lambda ref: dict(ref)):
            return e.build_items(cases, result, run)

    def test_metadata136_and_full_v9_lineage(self):
        report = adapter.metadata()
        self.assertEqual((report['case_count'], report['reference_count'], report['logical_records']), (8, 88, 136))
        self.assertEqual(len(report['original_metric_columns']), 14)
        self.assertTrue(report['reference_lineage']['all_old_v9_case_fields_preserved'])
        cases = adapter.BASE.frozen_cases()
        cases[0]['a_baseline']['path'] += '.changed'
        with self.assertRaises(ValueError):
            adapter.reference_lineage(cases)

    def test_48_complete_terminal_audits_and_explicit_15_16_counts(self):
        items = self.build(*self.fixtures())
        with patch.object(adapter, '_primitive_validate', return_value={'items': items}), \
                patch.object(adapter.BASE, 'generation_configuration', return_value=self.configuration()):
            document = adapter.validate(e.HERE)
        self.assertEqual(len(document['terminal_call_audit']), 48)
        counts = [row['total_unet_calls'] for row in document['terminal_call_audit']]
        self.assertEqual((counts.count(15), counts.count(16)), (16, 32))

    def test_hidden_extra_call_and_terminal_trace_mutations_rejected(self):
        _, generated, _ = self.fixtures()
        row = next(row for row in generated['records'] if row['method'] == 'baseline_terminal_t0')
        _, old = adapter.BASE.frozen_v7()
        prior = old[(row['case_id'], 'baseline_smooth_s075_a150')]
        for mutation in ('actual_steps', 'denoising_steps', 'deis_unet_calls', 'total_unet_calls', 'terminal_unet_calls',
                         'trace_length', 't0', 'au', 'cfg_batch', 'prompt', 'mode', 'source', 'step_index',
                         'extra_cfg', 'rho', 'filter', 'epsilon', 'prefix', 'projection', 'noedit'):
            wrong = copy.deepcopy(row)
            if mutation in ('actual_steps', 'denoising_steps', 'deis_unet_calls', 'total_unet_calls', 'terminal_unet_calls'):
                wrong[mutation] = 99
            elif mutation == 'trace_length':
                wrong['unet_call_trace'].pop()
            elif mutation == 't0':
                wrong['unet_call_trace'][-1]['t'] = 1
            elif mutation == 'au':
                wrong['unet_call_trace'][-1]['au_scale'] = 0
            elif mutation == 'cfg_batch':
                wrong['unet_call_trace'][-1]['cfg_batch'] = 1
            elif mutation == 'prompt':
                wrong['unet_call_trace'][-1]['prompt_embeds_sha256'] = 'a' * 64
            elif mutation == 'mode':
                wrong['terminal_trace']['mode'] = 'terminal_last_eps'
            elif mutation == 'source':
                wrong['terminal_trace']['epsilon_source'] = 'last_t50'
            elif mutation == 'step_index':
                wrong['terminal_trace']['scheduler_step_index_after'] = 21
            elif mutation == 'extra_cfg':
                wrong['terminal_trace']['extra_cfg'] = 1
            elif mutation == 'rho':
                wrong['terminal_trace']['rho0'] = .1
            elif mutation == 'filter':
                wrong['terminal_trace']['filter'] = adapter.FILTER
            elif mutation == 'epsilon':
                wrong['terminal_trace']['terminal_raw_epsilon_sha256'] = 'b' * 64
            elif mutation == 'prefix':
                wrong['projection_trace'][0]['latent_sha256'] = 'b' * 64
            elif mutation == 'projection':
                wrong['terminal_trace']['projected_latent_sha256'] = 'f' * 64
            else:
                wrong['terminal_trace']['projected_latent_sha256'] = 'f' * 64
                wrong['projection_trace'][-1]['latent_sha256'] = 'f' * 64
            with self.assertRaises(ValueError, msg=mutation):
                adapter.audit_terminal(wrong, prior)

    def test_metric_functions_and_facenet_loop_byte_preserved(self):
        def functions(source):
            return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef)}
        original = functions(adapter.BASE.ADAPTER.V5_CORE.read_text())
        derived = functions(adapter.BASE.ADAPTER.DERIVED_SOURCE)
        for name in ('au_metrics', 'extras', 'aggregate', 'load_openface', 'openface', 'facenet'):
            self.assertEqual(original[name], derived[name], name)
        self.assertEqual(original['merge'].split('    methods = REFERENCES')[0],
                         derived['merge'].split('    methods = REFERENCES')[0])
        folder = self.output / 'facenet_binding'
        folder.mkdir()
        e.save(folder / 'items.json', {'test_used': False, 'items': []})
        original = (e.POLICY_DIR / 'run_facenet_single_directions.py').read_text()
        for count in (88, 136):
            source = e.derived_facenet_source(folder, count)
            self.assertIn(f'len(items) == {count}', source)
            start, end = '    measurements, embeddings = {}, {}', '    records = [{'
            self.assertEqual(original[original.index(start):original.index(end)], source[source.index(start):source.index(end)])

    def test_nonterminal_duplicate_and_test_cases_rejected(self):
        for mutation in ('running', 'duplicate', 'test'):
            cases, result, run = self.fixtures()
            if mutation == 'running':
                result['status'] = 'running'
            elif mutation == 'duplicate':
                result['records'][-1] = copy.deepcopy(result['records'][0])
            else:
                cases['cases'][0]['split'] = 'test'
            with self.assertRaises(ValueError):
                self.build(cases, result, run)

    def test_all_failed_retained_with_zero_actual_calls_and_fresh_prepare(self):
        cases, result, _ = self.fixtures()
        for row in result['records']:
            row.update(status='failed', output=None, sha256=None, denoising_steps=0, deis_unet_calls=0,
                       terminal_unet_calls=0, total_unet_calls=0, unet_call_trace=[], terminal_trace=None)
        result['status'] = 'completed_with_failures'
        run = self.output / 'all_failed_run'
        run.mkdir()
        e.save(run / 'configuration.json', self.configuration())
        result['configuration'] = {'path': str(run / 'configuration.json'), 'sha256': e.sha(run / 'configuration.json')}
        e.save(run / 'cases.json', cases)
        e.save(run / 'result.json', result)
        output = self.output / 'prepared'
        e.prepare(run, output)
        _, document = e.checked(output)
        self.assertEqual((document['logical_count'], document['measurement_count']), (136, 88))
        self.assertEqual(document['terminal_call_audit'], [])
        with self.assertRaises(FileExistsError):
            e.prepare(run, output)
        self.assertFalse((output / 'openface').exists())
        self.assertFalse((output / 'facenet').exists())

    def test_merge136_and_NA_buffalo_grouped_fair_pairs(self):
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
        e.save(output / 'openface/results.json', {'status': 'complete', 'policy_sha256': e.POLICY_SHA,
                                                'test_used': False, 'records': aus})
        e.save(output / 'facenet/results.json', {'summary': {'status': 'complete'}, 'policy_sha256': e.POLICY_SHA,
                                               'test_used': False, 'records': faces})
        with patch.object(e, 'checked', return_value=(output, {'items': items, 'generation_status': result['status']})):
            adapter.merge(output)
        report = e.read(output / 'comparison/results.json')
        stats = e.read(output / 'comparison/v10_statistics.json')
        self.assertEqual(len(report['records']), 136)
        self.assertEqual(len(report['pairwise']), 91)
        self.assertEqual(len(stats['fair_baseline_pairs']), 3)
        failed = next(row for row in report['records'] if row['generation_status'] == 'failed')
        self.assertTrue(all(failed[key] is None for key in e.METRICS))
        self.assertTrue(all(row['buffalo_auxiliary']['status'] == 'not_run' for row in report['records']))
        self.assertEqual(set(stats['by_subject']), e.SUBJECTS)
        for name in ('torch', 'facenet_pytorch', 'insightface'):
            self.assertNotIn(name, sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
