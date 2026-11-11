"""CPU synthetic gate tests for independent existing19DEV completion entry."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image
import common
import run_dev_completion as runner


def synthetic_case(noedit=False):
    case = {'case_id': 'synthetic_dev', 'subject': 'synthetic', 'split': 'dev', 'source_id': 'source', 'prompt': 'fixed prompt',
        'source_au_vector_0_5': [0] * 12, 'target_au_vector_0_5': [0] * 12 if noedit else [1] + [0] * 11,
        'group': 'no_edit' if noedit else 'edited'}
    source = {'source_id': 'source', 'subject': 'synthetic', 'split': 'dev', 'canonical_source': {'path': 'source.png', 'sha256': 'a' * 64}}
    return {'cases': [case], 'sources': [source], 'expected': {}}, {case['case_id']: ()}


def synthetic_receipt():
    return {'condition_hashes': {'au_embedding': 'a'}, 'projection_trace': [{'t': 50, 'latent_sha256': 'b'}],
        'source_latent_sha256': 'c', 'noise_sha256': 'd', 'deis_unet_calls': 15, 'terminal_unet_calls': 0, 'total_unet_calls': 15}


class DevGridTests(unittest.TestCase):
    def setUp(self):
        self.parent = common.HERE / 'synthetic_tests'
        self.parent.mkdir(exist_ok=True)

    def test_exact_five_variants_and_single_seed_are_executed(self):
        inputs, masks = synthetic_case()
        class Runtime:
            def __init__(self):
                self.calls = []
            def prepare(self, variant, *args):
                return {'variant': variant}
            def sample(self, prepared, seed):
                self.calls.append((prepared['variant'], seed))
                return Image.new('RGB', (512, 512), (42, 71, 93)), synthetic_receipt()
        runtime = Runtime()
        with tempfile.TemporaryDirectory(dir=self.parent) as directory, patch('builtins.print'):
            rows = runner.execute_dev_grid(inputs, masks, Path(directory), runtime)
            self.assertEqual(runtime.calls, [(v, 101) for v in common.VARIANTS])
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r['status'] == 'complete' and r['split'] == 'dev' and r['seed'] == 101 for r in rows))

    def test_prepare_failure_retains_all_five_not_started(self):
        inputs, masks = synthetic_case()
        class Runtime:
            def prepare(self, *args):
                raise ValueError('synthetic source failure')
        with tempfile.TemporaryDirectory(dir=self.parent) as directory, patch('builtins.print'):
            rows = runner.execute_dev_grid(inputs, masks, Path(directory), Runtime())
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r['status'] == 'not_started' and r['output'] is None and r['sha256'] is None for r in rows))

    def test_failed_saved_image_is_kept_only_as_ineligible_failed_artifact(self):
        inputs, masks = synthetic_case(noedit=True)
        class Runtime:
            def prepare(self, *args):
                return {'reconstruction': {'sha256': 'wrong'}}
            def sample(self, *args):
                return Image.new('RGB', (512, 512)), synthetic_receipt()
        with tempfile.TemporaryDirectory(dir=self.parent) as directory, patch('builtins.print'):
            rows = runner.execute_dev_grid(inputs, masks, Path(directory), Runtime())
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r['status'] == 'failed' and r['output'] is None and r['sha256'] is None for r in rows))
            for row in rows:
                artifact = row['saved_output_unverified']
                self.assertFalse(artifact['eligible_for_evaluation'])
                self.assertEqual(common.sha(artifact['path']), artifact['sha256'])

    def test_existing_subset_checks_png_conditions_and_entire_trace(self):
        receipt = synthetic_receipt()
        expected = {**receipt, 'sha256': 'expected'}
        self.assertTrue(runner.selected_parity(receipt, 'expected', expected)['passed'])
        for key in ('condition_hashes', 'projection_trace', 'source_latent_sha256', 'noise_sha256'):
            bad = copy.deepcopy(receipt)
            bad[key] = 'changed'
            self.assertFalse(runner.selected_parity(bad, 'expected', expected)['passed'])
        self.assertFalse(runner.selected_parity(receipt, 'changed PNG', expected)['passed'])
        bad = {**receipt, 'total_unet_calls': 16}
        self.assertFalse(runner.selected_parity(bad, 'expected', expected)['passed'])

    def test_actual_dev_labels_preserve19_and36cells(self):
        document = common.read(common.EXP / runner.DEV_PINS['cases'][0])
        cases, coverage = runner.validate_dev_labels(document)
        self.assertEqual(len(cases), 19)
        self.assertEqual(len(coverage), 36)
        for mutate in ('role', 'order', 'label'):
            bad = copy.deepcopy(document)
            if mutate == 'role':
                bad['cases'][0]['split'] = 'postfreeze_case_validation'
            elif mutate == 'order':
                bad['cases'].reverse()
            else:
                bad['cases'][0]['source_au_vector_0_5'][0] = 1
            with self.assertRaises(ValueError):
                runner.validate_dev_labels(bad)

    def test_whitelist_drops_cached_target_latents_and_preserves_features(self):
        inputs, _ = synthetic_case(noedit=True)
        case = inputs['cases'][0]
        inputs['refs'] = {'cache_features': {'path': 'synthetic_archive'}}
        anchor = torch.arange(768.).float()
        sample = {'case_id': case['case_id'], 'subject': case['subject'], 'split': 'dev', 'prompt': case['prompt'],
            'image_patches': torch.ones(49, 512), 'source_identity': torch.ones(512), 'base_faceid': torch.ones(512),
            'text_tokens': torch.ones(77, 768), 'text_pooled': anchor, 'source_aus': torch.zeros(12), 'target_aus': torch.zeros(12),
            'target_latent': object(), 'target_image_path': '/forbidden/target.png'}
        stub = SimpleNamespace(make_prompt=lambda _: case['prompt'])
        with patch.dict(sys.modules, {'prepare_training_cache': stub}), patch.object(runner, 'load_anchor', return_value=(anchor, {'binding': 'synthetic'}, None)), patch.object(torch, 'load', return_value={'samples': [sample]}):
            features, cache, receipts = runner.source_only_features(inputs, torch)
        self.assertEqual(set(features), {'source_anchor', 'sources', 'conditions'})
        self.assertEqual(set(features['sources']['source']), {'image_patches', 'source_identity', 'base_faceid'})
        self.assertEqual(set(features['conditions'][case['case_id']]), {'text_tokens', 'text_pooled', 'source_aus', 'target_aus', 'prompt'})
        self.assertTrue(torch.equal(features['conditions'][case['case_id']]['text_pooled'], anchor))

    def test_new_entry_does_not_change_frozen_core_spec(self):
        self.assertEqual(common.read(common.HERE / 'generation_spec_candidate_v3.json'), common.execution_spec())
        self.assertEqual(runner.DEV_SEEDS, [101])
        self.assertEqual(common.SEEDS, [101, 202, 303])

    def test_core_delta_from_v2_is_failure_accounting_only(self):
        old = common.read(common.HERE / 'generation_spec_candidate_v2.json')
        current = common.read(common.HERE / 'generation_spec_candidate_v3.json')
        block = ("                            if row['output'] is not None:\n"
                 "                                row['saved_output_unverified'] = {'path': row['output'], 'sha256': row['sha256'], 'eligible_for_evaluation': False}\n"
                 "                            row['output'], row['sha256'] = None, None\n")
        source = (common.HERE / 'run_grid.py').read_text()
        self.assertEqual(source.count(block), 1)
        self.assertEqual(hashlib.sha256(source.replace(block, '').encode()).hexdigest(), old['execution_sources']['run_grid.py']['sha256'])
        for name in ('common.py', 'geometry.py', 'prepare_cache.py'):
            self.assertEqual(old['execution_sources'][name], current['execution_sources'][name])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
