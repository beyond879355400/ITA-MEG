"""Synthetic-only contract tests: no image/model or new-case pixels accessed."""
import copy
import csv
import tempfile
from pathlib import Path
import unittest
from unittest import mock
import contract as c
import evaluate as e


def cases():
    result = {}
    for subject in range(8):
        for n in range(4):
            cid = f'S{subject:02d}_case{n}'
            result[cid] = {'case_id': cid, 'subject': f'S{subject:02d}', 'split': c.ROLE,
                'group': 'no_edit' if n == 0 else 'edited', 'source_labels_12': [0] * 12,
                'target_labels_12': [int(n > 0)] + [0] * 11}
    return result


def indexes(meta):
    result = []
    for m in c.METHODS:
        rows = []
        for cid, row in meta.items():
            for seed in c.SEEDS:
                actual = 101 if m == 'liveportrait' else seed
                item = {**row, 'method': m, 'seed': seed, 'status': 'complete',
                    'source_aus_0_5': row['source_labels_12'], 'target_aus_0_5': row['target_labels_12'],
                    'execution_id': f'{m}/{cid}/{actual}', 'execution_started': True,
                    'output': f'/synthetic/{m}/{cid}/{actual}.png',
                    'sha256': ('%064x' % (len(rows) // 3 + 1)) if m == 'liveportrait' else '%064x' % (len(rows) + 1)}
                if m == 'liveportrait' and seed != 101:
                    item['alias_of'] = {'method': m, 'case_id': cid, 'seed': 101}
                rows.append(item)
        result.append({'status': 'complete', 'method_names': [m], 'case_ids': list(meta), 'sampling_seeds': c.SEEDS, 'records': rows})
    return result


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.meta = cases()
        self.indexes = indexes(self.meta)

    def test_full_plan_and_alias_accounting(self):
        plan, rows, counts = c.normalize_records(self.meta, self.indexes)
        self.assertEqual(len(plan), 768)
        self.assertEqual(len(rows), 768)
        self.assertEqual(counts['liveportrait']['executed_physical_calls'], 32)
        self.assertEqual(counts['liveportrait']['alias_rows'], 64)
        self.assertEqual(counts['liveportrait']['planned_logical_seed_outputs'], 96)
        self.assertEqual(counts[c.OURS[0]]['executed_physical_calls'], 96)

    def test_duplicate_and_cross_subject_rejected(self):
        self.indexes[0]['records'].append(copy.deepcopy(self.indexes[0]['records'][0]))
        with self.assertRaises(ValueError): c.normalize_records(self.meta, self.indexes)
        fresh = indexes(self.meta)
        fresh[0]['records'][0]['subject'] = 'other'
        with self.assertRaises(ValueError): c.normalize_records(self.meta, fresh)

    def test_role_and_manual_labels_immutable(self):
        for field, value in [('split', 'dev'), ('split', 'test'), ('target_aus_0_5', [5] * 12)]:
            fresh = indexes(self.meta)
            fresh[0]['records'][0][field] = value
            with self.assertRaises(ValueError): c.normalize_records(self.meta, fresh)

    def test_alias_cannot_change_image_or_pose_as_stochastic(self):
        self.indexes[-1]['records'][1]['sha256'] = 'f' * 64
        with self.assertRaises(ValueError): c.normalize_records(self.meta, self.indexes)
        fresh = indexes(self.meta)
        fresh[0]['records'][1]['alias_of'] = {'method': c.OURS[0], 'case_id': list(self.meta)[0], 'seed': 101}
        with self.assertRaises(ValueError): c.normalize_records(self.meta, fresh)

    def test_missing_failed_outputs_stay_in_denominator(self):
        self.indexes[0]['records'].pop()
        row = self.indexes[0]['records'][0]
        row.update(status='failed', output=None, sha256=None)
        plan, rows, counts = c.normalize_records(self.meta, self.indexes)
        self.assertEqual(len(plan), 768)
        self.assertEqual(counts[c.OURS[0]]['missing_logical_rows'], 1)
        self.assertEqual(counts[c.OURS[0]]['successful_physical_calls'], 94)

    def test_families_and_primary_are_fixed_before_results(self):
        spec = c.analysis_spec()
        self.assertEqual(len(spec['comparisons']), 7)
        self.assertTrue(all(r['metric'] == c.PRIMARY for r in spec['comparisons']))
        self.assertEqual(sum(r['family'] == 'external_primary_C' for r in spec['comparisons']), 3)
        self.assertEqual(sum(r['family'] == 'internal_primary_C' for r in spec['comparisons']), 4)
        self.assertEqual(len(spec['metrics']), len(c.SOURCE_METRICS) * 3)

    def test_failure_and_scope_not_replaced_by_zero(self):
        row = self.indexes[0]['records'][3]
        row.update(status='failed', output=None, sha256=None)
        plan, records, counts = c.normalize_records(self.meta, self.indexes)
        doc = {'planned': plan, 'normalized_generation_records': list(records.values()), 'execution_accounting': counts, 'protocol_sha256': 'a' * 64}
        metrics = [{**self.meta[r['case_id']], **r, **{name: .2 for name in c.SOURCE_METRICS}} for r in plan]
        result = c.statistics_document(doc, metrics)
        failure = result['records'][3]
        self.assertIsNone(failure['metrics'][c.PRIMARY])
        self.assertEqual(failure['metric_reasons'][c.PRIMARY], 'generation_failed')
        noedit = result['records'][0]
        self.assertIsNone(noedit['metrics'][c.PRIMARY])
        self.assertEqual(noedit['metric_reasons'][c.PRIMARY], 'out_of_predeclared_scope')
        self.assertEqual(result['metadata']['data_role'], c.ROLE)
        self.assertFalse(result['metadata']['independent_final_test'])

    def test_seed_case_subject_weights_and_exact256(self):
        stats = c.legacy('statistics')
        values = {('a', 'a1', 101): 0., ('a', 'a1', 202): 2., ('a', 'a2', 101): 5., ('b', 'b1', 101): 9.}
        aggregate = stats.subject_aggregate(values)
        self.assertEqual(aggregate['subject_means'], {'a': 3., 'b': 9.})
        self.assertEqual(aggregate['mean'], 6.)
        inference = stats.subject_inference([1.] * 8, seed=20260915)
        self.assertEqual(inference['sign_patterns'], 256)
        self.assertEqual(inference['bootstrap_samples'], 50000)
        self.assertAlmostEqual(inference['two_sided_sign_flip_p'], 2 / 256)

    def test_holm_na_retains_prespecified_family(self):
        stats = c.legacy('statistics')
        rows = [{'id': 'a', 'family': 'external', 'two_sided_sign_flip_p': .01},
                {'id': 'b', 'family': 'external', 'two_sided_sign_flip_p': .04},
                {'id': 'c', 'family': 'external', 'two_sided_sign_flip_p': None}]
        stats.holm_adjust(rows)
        self.assertAlmostEqual(rows[0]['holm_adjusted_p'], .03)
        self.assertAlmostEqual(rows[1]['holm_adjusted_p'], .08)
        self.assertIsNone(rows[2]['holm_adjusted_p'])
        self.assertTrue(all(r['holm_family_size'] == 3 for r in rows))

    def test_old_merge_exact_metrics_and_extra_na(self):
        meta = self.meta[list(self.meta)[1]]
        method = c.OURS[1]
        plan = [{k: v for k, v in {**meta, 'method': method, 'seed': 101}.items() if k in ('method', 'subject', 'case_id', 'seed')}]
        source = {**meta, 'method': 'source_real', 'seed': 0, 'vector': [.2] + [0.] * 11, 'selected_au17': {str(a): .3 for a in (7, 10, 14, 23, 45)}}
        target = {**source, 'method': 'target_real', 'vector': [1.] + [0.] * 11, 'selected_au17': {str(a): .6 for a in (7, 10, 14, 23, 45)}}
        output = {**source, 'method': method, 'seed': 101, 'vector': [.5] + [0.] * 11, 'selected_au17': {str(a): .5 for a in (7, 10, 14, 23, 45)}}
        identity = {**output, 'identity_cosine': .8}
        doc = {'planned': plan, 'case_metadata': {meta['case_id']: meta}}
        merge = c.legacy('evaluation_export').merge_measurements
        rows = merge(doc, {'records': [source, target, output]}, {'records': [identity]}, {'records': []})
        self.assertAlmostEqual(rows[0]['target_mae_changed'], .5)
        self.assertAlmostEqual(rows[0]['delta_mae_changed'], .7)
        self.assertEqual(rows[0]['direction_correct_fraction'], 1.)
        self.assertAlmostEqual(rows[0]['extra5_proxy_mae_to_real_target'], .1)
        self.assertEqual(rows[0]['identity_cosine'], .8)
        missing = merge(doc, {'records': [source, output]}, {'records': [identity]}, {'records': []})
        self.assertIsNone(missing[0]['extra5_proxy_mae_to_real_target'])
        self.assertTrue(all(missing[0][f'extra_AU{a}_error_to_real_target'] is None for a in (7, 10, 14, 23, 45)))

    def test_openface_largest_geometry_no_score_fallback(self):
        module = c.legacy('openface')
        fields = ['face', 'confidence'] + [f'{axis}_{i}' for axis in ('x', 'y') for i in range(68)] + [f'AU{a:02d}_r' for a in module.AUS17]
        def row(index, scale, bad=False):
            value = {'face': str(index), 'confidence': '0.01' if index == 1 else '.99'}
            value.update({f'x_{i}': str((i % 10) * scale) for i in range(68)})
            value.update({f'y_{i}': str((i // 10) * scale) for i in range(68)})
            value.update({f'AU{a:02d}_r': '1.0' for a in module.AUS17})
            if bad: value['AU01_r'] = 'nan'
            return value
        with tempfile.TemporaryDirectory(dir=c.HERE) as directory:
            path = Path(directory) / 'synthetic.csv'
            with path.open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows([row(0, 1), row(1, 2, bad=True)])
            parsed = module.parse(path)
        self.assertEqual(parsed['selected_row'], 1)
        self.assertIsNone(parsed['vector'])
        self.assertEqual(len(parsed['raw_rows']), 2)

    def test_identity_derivation_role_and_seed_uniqueness_only(self):
        for kind in ('facenet', 'buffalo'):
            _, text = e.derive_identity(kind, c.HERE / 'synthetic_not_run', 'a' * 64, 832, 32)
            compile(text, '<synthetic-identity>', 'exec')
            self.assertIn(c.ROLE, text)
            self.assertNotIn("['split'] == 'test'", text)
            self.assertNotIn("['split'] == 'dev'", text)
            self.assertIn('independent_final_test', text)
            if kind == 'facenet':
                self.assertIn("row['case_id'], row['method'], row['seed']", text)
                self.assertIn('thresholds=[0.6, 0.7, 0.7]', text)
                self.assertIn('num_classes=8631', text)

    def test_source_items_only_canonical_and_missing_records(self):
        rows = list(self.meta.values())
        cache = {'cases': [{**r, 'source_id': r['subject']} for r in rows]}
        sources = {r['subject']: {'status': 'complete', 'canonical_source': {'path': '/synthetic/source.png', 'sha256': 'a' * 64}} for r in rows}
        sources['S00'] = {'status': 'failed', 'error': 'no source'}
        control = {'cases': self.meta}
        items, unavailable = e.source_items(control, cache, sources)
        self.assertEqual((len(items), len(unavailable)), (28, 4))
        self.assertTrue(all(r['method'] == 'source_real' and r['seed'] == 0 for r in items + unavailable))
        self.assertFalse(any('target_reference_image' in r for r in items))

    def test_spec_metadata_pins_without_models(self):
        spec = c.execution_spec()
        self.assertEqual(spec['liveportrait']['flag_relative_motion'], False)
        self.assertEqual(spec['analysis_spec']['bootstrap_samples'], 50000)
        self.assertEqual(spec['data_role'], c.ROLE)

    def test_declared_physical_count_must_match(self):
        self.indexes[-1]['physical_executions'] = 96
        with self.assertRaises(ValueError): c.normalize_records(self.meta, self.indexes)

    def test_scope_coverage_and_full_paired_analysis(self):
        plan, rows, counts = c.normalize_records(self.meta, self.indexes)
        doc = {'planned': plan, 'normalized_generation_records': list(rows.values()), 'execution_accounting': counts,
            'protocol_sha256': 'a' * 64, 'case_metadata': self.meta}
        measured = [{**self.meta[p['case_id']], **p, **{name: .2 for name in c.SOURCE_METRICS}} for p in plan]
        input_doc = c.statistics_document(doc, measured)
        result = c.enrich_coverage(c.legacy('statistics').analyze(input_doc), input_doc, self.meta)
        coverage = next(r for r in result['coverage_table'] if r['method'] == c.OURS[1] and r['metric'] == c.PRIMARY)
        self.assertEqual(coverage['planned_applicable_seed_outputs'], 72)
        self.assertEqual(coverage['out_of_scope_seed_outputs'], 24)
        self.assertEqual(coverage['applicable_generation_or_measurement_unavailable'], 0)
        self.assertEqual(len(result['comparisons']), 7)
        self.assertTrue(all(r['sign_patterns'] == 256 and r['all_eight_subjects_paired'] for r in result['comparisons']))

    def test_native_generated_size_is_preserved_and_real_reference_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=c.HERE) as directory:
            path = Path(directory) / 'native_lp.png'
            Image.new('RGB', (1024, 768), (17, 23, 91)).save(path)
            original = path.read_bytes()
            image_ref = c.ref(path)
            item = e.inspect_image_item({'method': 'liveportrait'}, image_ref, 'generated')
            self.assertEqual(item['image_size'], [1024, 768])
            with self.assertRaises(ValueError): e.inspect_image_item({}, image_ref, 'real_source')
            with self.assertRaises(ValueError): e.inspect_image_item({}, image_ref, 'real_target')
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(c.sha(path), image_ref['sha256'])

    def test_prepare_native_lp_enters_measurement_and_wrong_real_is_unavailable(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=c.HERE) as directory:
            root = Path(directory)
            image = root / 'native_lp.png'
            Image.new('RGB', (1024, 768), (23, 71, 41)).save(image)
            original_sha = c.sha(image)
            image_ref = c.ref(image)
            tiny_indexes = indexes(self.meta)
            for index in tiny_indexes[:-1]: index['records'] = []
            tiny_indexes[-1]['records'] = tiny_indexes[-1]['records'][:3]
            for row in tiny_indexes[-1]['records']:
                row['output'], row['sha256'] = str(image), original_sha
            refs = {'kind': 'completion_evaluation_references', 'status': 'complete', 'data_role': c.ROLE,
                'protocol_sha256': 'a' * 64, 'measurement_policy_sha256': c.POLICY_SHA,
                'source_cache_manifest': {}, 'raw_manifest': {}, 'cases': []}
            for meta in self.meta.values():
                refs['cases'].append({**meta, 'source_reference': {'status': 'failed'}, 'target_reference': {'status': 'failed'}})
            refs['cases'][0]['source_reference'] = {'status': 'complete', 'image': image_ref}
            manifest = root / 'references.json'
            c.save(manifest, refs)
            refs['source_cache_manifest'] = refs['raw_manifest'] = c.ref(manifest)
            with mock.patch.object(e, 'read', return_value=refs), mock.patch.object(e, 'load_indexes', return_value=tiny_indexes):
                job = e.prepare_job({'cases': self.meta, 'protocol_sha256': 'a' * 64, 'freeze': {}}, c.ref(manifest), [], root / 'job')
            doc = c.read(job / 'items.json')
            self.assertEqual(len(doc['items']), 3)
            self.assertTrue(all(r['method'] == 'liveportrait' and r['image_size'] == [1024, 768] for r in doc['items']))
            bad = next(r for r in doc['availability'] if r['key'][0] == 'source_real' and r['key'][2] == refs['cases'][0]['case_id'])
            self.assertEqual(bad['status'], 'unavailable')
            self.assertEqual(c.sha(image), original_sha)

    def test_openface_expansion_keeps_actual_native_size(self):
        item = {**next(iter(self.meta.values())), 'method': 'liveportrait', 'seed': 101,
            'sha256': 'a' * 64, 'path': '/synthetic/native_lp.png', 'image_size': [1024, 768]}
        physical = [{'sha256': 'a' * 64, 'path': item['path'], 'raw_rows': [{'x_0': '801'}], 'vector': [0.] * 12}]
        expanded = e.expand_openface_records([item], physical)
        self.assertEqual(expanded[0]['image_size'], [1024, 768])
        self.assertEqual(expanded[0]['raw_rows'][0]['x_0'], '801')

    def test_failed_targets_with_none_sha_keep_distinct_case_provenance(self):
        a = {'status': 'failed', 'path': '/raw/a.png', 'sha256': None, 'error': 'decode_a', 'label_frame_1based': 10}
        b = {'status': 'failed', 'path': '/raw/b.png', 'sha256': None, 'error': 'decode_b', 'label_frame_1based': 20}
        cache, align = {}, mock.Mock()
        ra = e.resolve_target_reference('case_a', a, {}, cache, align)
        rb = e.resolve_target_reference('case_b', b, {}, cache, align)
        self.assertEqual(len(cache), 2)
        self.assertEqual(ra['raw_image'], a)
        self.assertEqual(rb['raw_image'], b)
        self.assertEqual((ra['error'], rb['error']), ('decode_a', 'decode_b'))
        self.assertFalse(ra['target_alignment_attempted'])
        align.assert_not_called()

    def test_successful_target_sha_reuse_retains_each_raw_path(self):
        a = {'status': 'complete', 'path': '/raw/a.png', 'sha256': 'a' * 64}
        b = {'status': 'complete', 'path': '/raw/b.png', 'sha256': 'a' * 64}
        align = mock.Mock(return_value={'status': 'complete', 'raw_image': a, 'image': {'path': '/canon/a.png', 'sha256': 'b' * 64}})
        cache = {}
        ra = e.resolve_target_reference('case_a', a, {}, cache, align)
        rb = e.resolve_target_reference('case_b', b, {}, cache, align)
        self.assertEqual(align.call_count, 1)
        self.assertEqual(rb['raw_image'], b)
        self.assertEqual(rb['measurement_raw_image'], a)
        self.assertEqual(ra['image'], rb['image'])

    def test_failed_artifact_never_measured_or_counted_and_original_unchanged(self):
        self.indexes[0]['records'][0]['status'] = 'failed'
        original = copy.deepcopy(self.indexes)
        _, rows, counts = c.normalize_records(self.meta, self.indexes)
        failed = rows[c.row_key(self.indexes[0]['records'][0])]
        self.assertIsNone(failed['output'])
        self.assertIsNone(failed['sha256'])
        self.assertEqual(failed['failed_artifact']['output'], original[0]['records'][0]['output'])
        self.assertEqual(counts[c.OURS[0]]['successful_physical_calls'], 95)
        self.assertEqual(self.indexes, original)


if __name__ == '__main__':
    unittest.main()
