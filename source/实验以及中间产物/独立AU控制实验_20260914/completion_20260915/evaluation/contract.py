"""Post-freeze case validation contract. Metadata only; no pixel/model imports."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
COMPLETION = HERE.parent
EXP = COMPLETION.parent
ROLE = 'postfreeze_case_validation'
POLICY_SHA = 'd29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c'
LABEL_SHA = '93ca535e58fed243b9002369b647eb4d46887fcceebfbd0cd01db0314ff3d0b7'
V2 = EXP / 'development_evaluation/evaluation_policy_v2'
SEEDS = [101, 202, 303]
AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
VARIANTS = ['base', 'bidirectional', 'condition_reads_image', 'image_reads_condition', 'mlp']
OURS = [v + '_local_lp_s050' for v in VARIANTS]
EXTERNAL = ['magicface', 'faceadapter', 'liveportrait']
METHODS = OURS + EXTERNAL
PRIMARY = 'target_mae_changed__edited'
SOURCE_METRICS = {'target_mae_changed': 'lower', 'delta_mae_changed': 'lower',
    'unchanged_source_relative_drift': 'lower', 'target_mae_all12': 'lower',
    'delta_mae_all12': 'lower', 'direction_correct_fraction': 'higher',
    'identity_cosine': 'higher', 'buffalo_identity_cosine': 'higher',
    'extra5_proxy_mae_to_real_target': 'lower',
    **{f'extra_AU{a}_error_to_real_target': 'lower' for a in (7, 10, 14, 23, 45)}}
PINS = {
    'evaluation_measurements': ('final_test/evaluation_measurements.py', '8c31892cdc7b5f003b82837ecb7d65e69f0f76c5ecb82cd128dec7d943e2e273'),
    'evaluation_export': ('final_test/evaluation_export.py', '7473028eed74c432953cbfc2323fa640c29e1671fa73f85029a444407da5baac'),
    'evaluation_contract': ('final_test/evaluation_contract.py', '7513ace1edbf3b7fd3c6da065882f1c4f4886c829e9f27c373962cc83f6285dc'),
    'evaluation_references': ('final_test/evaluation_references.py', '25bf4e41122d49476bf88416522b4645031e7a6cd142cdbb122dce22503fa5c5'),
    'statistics': ('statistics/paired_subject_statistics.py', 'f0f1baaf08464ac2e46bfe607ae7ea34ae1c49f200288b57b45e32df2a44d4ed'),
    'openface': ('development_evaluation/evaluation_policy_v2/openface_v2.py', '5446237ff2fba672bd06ec62820bf671d9deb4c776d83a28d18060a581b9f908'),
    'facenet': ('development_evaluation/evaluation_policy_v2/run_facenet_single_directions.py', '5d8e1916bb5bcc9afc307980438b5f2f3fb0610f5dde12141f90e85efa7ebc2d'),
    'buffalo': ('development_evaluation/evaluation_policy_v2/append_buffalo.py', '96daabea1e618a993de22ef684a4cbdcc6822911191a424fc457ea19012af88a'),
}
DEPENDENCIES = [
    'development_evaluation/evaluation_policy_v2/summarize_v2.py',
    'development_evaluation/evaluation_policy_v2/run_facenet_v2.py',
    'development_evaluation/evaluation_policy_v2/INPUT_COORDINATE_NOTE.md',
    'development_evaluation/evaluate_patch_variant.py',
    'development_evaluation/v3_global_raw_dev64/evaluate.py',
    'development_evaluation/v1/evaluate_development.py',
    'development_evaluation/v4_slot_2k_mlp200_dev19/evaluate.py',
    'development_evaluation/real_dev_calibration/calibration_metrics.py',
]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def verified(value):
    need(isinstance(value, dict) and re.fullmatch(r'[0-9a-f]{64}', str(value.get('sha256', ''))), 'Explicit path/SHA reference required')
    path = Path(value['path']).resolve()
    need(path.is_relative_to(EXP) and path.is_file() and sha(path) == value['sha256'], 'Reference path/SHA changed')
    return path


def new_directory(path):
    path = Path(path).resolve()
    need(path.is_relative_to(HERE) and not path.exists(), 'New evaluation subdirectory required')
    path.mkdir(parents=True)
    return path


def legacy(name):
    relative, expected = PINS[name]
    path = EXP / relative
    need(sha(path) == expected, 'Frozen algorithm changed: ' + name)
    sys.path.insert(0, str(EXP / 'final_test'))
    sys.path.insert(0, str(V2))
    spec = importlib.util.spec_from_file_location('completion_reuse_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analysis_spec():
    metrics = {name + '__' + scope: {'source_metric': name, 'scope': scope, 'direction': direction}
        for name, direction in SOURCE_METRICS.items() for scope in ('all', 'edited', 'no_edit')}
    chosen = 'bidirectional_local_lp_s050'
    comparisons = [{'id': 'chosen_vs_' + method, 'method': chosen, 'reference': method,
        'metric': PRIMARY, 'family': 'external_primary_C'} for method in EXTERNAL]
    comparisons += [{'id': 'chosen_vs_' + method, 'method': chosen, 'reference': method,
        'metric': PRIMARY, 'family': 'internal_primary_C'} for method in OURS if method != chosen]
    return {'metrics': metrics, 'comparisons': comparisons, 'bootstrap_samples': 50000,
        'bootstrap_seed': 20260915, 'confidence_level': .95}


def execution_spec():
    # Fixed methods/endpoint/families are declared before opening any new images.
    pins = {}
    for name, (relative, expected) in PINS.items():
        need(sha(EXP / relative) == expected, 'Frozen algorithm changed: ' + name)
        pins[name] = ref(EXP / relative)
    need(sha(V2 / 'policy.json') == POLICY_SHA, 'Frozen v2 policy changed')
    return {'version': 'completion_postfreeze_evaluation_v2', 'data_role': ROLE,
        'independent_final_test': False, 'previously_exposed_subjects': True,
        'measurement_policy': ref(V2 / 'policy.json'), 'methods': METHODS, 'sampling_seeds': SEEDS,
        'primary_metric': PRIMARY, 'primary_family': 'external_primary_C',
        'families': {'external_primary_C': 3, 'internal_primary_C': 4},
        'method_settings': {m: {'input_group': 'source_real_driver' if m in ('faceadapter', 'liveportrait') else 'source_manual_au' if m == 'magicface' else 'source_text_au',
            'seed_execution': 'deterministic_alias' if m == 'liveportrait' else 'stochastic_each_seed'} for m in METHODS},
        'analysis_spec': analysis_spec(), 'algorithm_sources': pins,
        'algorithm_dependencies': {name: ref(EXP / name) for name in DEPENDENCIES},
        'liveportrait': {'flag_relative_motion': False, 'mode': 'absolute', 'physical_calls': 32, 'alias_rows': 64, 'logical_rows': 96},
        'execution_sources': {name: ref(HERE / name) for name in ('contract.py', 'evaluate.py')},
        'reference_space': 'source_cache_canonical512_and_independent_target_canonical512',
        'generated_image_space': 'native RGB at original positive dimensions; no crop, resize, padding, enhancement or recompression before frozen evaluators',
        'image_dimensions': 'record actual width,height; only real source/target canonical references require512x512',
        'failure_policy': 'all planned rows retained; invalid selected face/reference gives NA, never zero or alternate face',
        'aggregation': 'seed then case then subject; subjects equal; paired common case+seed keys',
        'inference': '50000 whole-subject bootstrap; exact 2^n sign flips (256 with all8); fixed-family Holm; pointwise unadjusted CI'}


def validate_labels(document):
    cases = document['cases']
    need(len(cases) == 32 and len({r['case_id'] for r in cases}) == 32, 'Complete32-case plan required')
    subjects = {r['subject'] for r in cases}
    need(len(subjects) == 8, 'Exactly8 subjects required')
    for row in cases:
        need(row['split'] == ROLE and row['au_order'] == AUS, 'Case role/AU order changed')
        a, b = row['source_au_vector_0_5'], row['target_au_vector_0_5']
        need(len(a) == len(b) == 12 and not any(a), 'Known zero-source12AU required')
        need(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 5 for v in a + b), 'Invalid labels')
    for subject in subjects:
        rows = [r for r in cases if r['subject'] == subject]
        need(len(rows) == 4 and sum(r['source_au_vector_0_5'] == r['target_au_vector_0_5'] for r in rows) == 1, 'Each subject requires one noedit and three edited cases')
    return cases


def load_control(path, expected):
    need(sha(path) == expected, 'Freeze SHA mismatch')
    frozen = read(path)
    need(frozen.get('kind') == 'completion_v11_frozen_protocol' and frozen.get('status') == 'frozen', 'Explicit completion freeze required')
    need(frozen.get('evaluation') == execution_spec(), 'Frozen evaluation sources/metrics/families changed')
    need(frozen['label_plan']['sha256'] == LABEL_SHA, 'Approved32-case label plan changed')
    labels = validate_labels(read(verified(frozen['label_plan'])))
    need(frozen['case_ids'] == [r['case_id'] for r in labels], 'Frozen case order changed')
    metadata = {r['case_id']: {'case_id': r['case_id'], 'subject': r['subject'], 'split': ROLE,
        'group': 'no_edit' if r['source_au_vector_0_5'] == r['target_au_vector_0_5'] else 'edited',
        'source_labels_12': r['source_au_vector_0_5'], 'target_labels_12': r['target_au_vector_0_5']} for r in labels}
    return {'freeze': ref(path), 'protocol_sha256': expected, 'frozen': frozen, 'labels': labels, 'cases': metadata}


def load_cache(path, expected, control):
    cache = read(verified({'path': str(path), 'sha256': expected}))
    need(cache.get('kind') == 'completion_v11_source_feature_cache' and cache.get('status') in ('complete', 'completed_with_failures'), 'Source cache not terminal')
    need(cache.get('protocol_sha256') == control['protocol_sha256'] and cache.get('case_ids') == list(control['cases']) and cache.get('sampling_seeds') == SEEDS, 'Source cache binding changed')
    need(cache.get('target_pixels_read') is False and cache.get('target_latents_computed') is False and cache.get('vae_loaded') is False, 'Source cache target boundary changed')
    sources = {r['source_id']: r for r in cache['sources']}
    need(len(sources) == len(cache['sources']) == 8 and len(cache['cases']) == 32, 'Source cache count mismatch')
    for c, label in zip(cache['cases'], control['labels']):
        need(all(c.get(k) == v for k, v in label.items()), 'Source cache labels changed')
        source = sources[c['source_id']]
        need(source['subject'] == c['subject'] and source['split'] == ROLE and source['source_frame_1based'] == label['source_frame_1based'], 'Canonical source owner mismatch')
        need(source['status'] in ('complete', 'failed'), 'Incomplete source record')
        if source['status'] == 'complete':
            image = source['canonical_source']
            need((image['width'], image['height'], image['mode']) == (512, 512, 'RGB'), 'Canonical source format changed')
    return cache, sources


def row_key(row):
    return row['method'], row['subject'], row['case_id'], row['seed']


def plan_rows(cases):
    return [{'method': m, 'subject': row['subject'], 'case_id': cid, 'seed': seed}
        for m in METHODS for cid, row in cases.items() for seed in SEEDS]


def normalize_records(cases, indexes):
    """Pure schema check; missing planned rows remain in the denominator."""
    plan = plan_rows(cases)
    expected = {row_key(p) for p in plan}
    records, methods = {}, set()
    for index in indexes:
        need(index.get('status') in ('complete', 'completed_with_failures') and index.get('usable_for_evaluation', True), 'Index incomplete/quarantined')
        names = index['method_names']
        need(names and len(set(names)) == len(names) and not methods.intersection(names) and set(names) <= set(METHODS), 'Duplicate/unplanned method index')
        methods.update(names)
        need(index['case_ids'] == list(cases) and index['sampling_seeds'] == SEEDS, 'Index case/seed plan changed')
        for original in index['records']:
            row = copy.deepcopy(original)
            key = row_key(row)
            need(key in expected and key not in records and row['method'] in names, 'Duplicate/unplanned generation row')
            meta = cases[row['case_id']]
            need(row['split'] == ROLE and row['status'] in ('complete', 'failed', 'not_started'), 'Invalid generation role/status')
            for key2, source in (('source_aus_0_5', 'source_labels_12'), ('target_aus_0_5', 'target_labels_12')):
                need(row.get(key2) == meta[source], 'Generated row AU labels differ from fixed case')
            need(isinstance(row.get('execution_id'), str) and row['execution_id'] and type(row.get('execution_started')) is bool, 'Explicit physical execution identity required')
            need(row['status'] != 'not_started' or row['execution_started'] is False, 'Not-started row cannot claim execution')
            if row['status'] == 'complete':
                need(row['execution_started'], 'Complete output requires actual execution')
                if isinstance(row.get('output'), str):
                    row['output'] = {'path': row['output'], 'sha256': row.get('sha256')}
                output = row.get('output')
                need(isinstance(output, dict) and isinstance(output.get('path'), str) and re.fullmatch(r'[0-9a-f]{64}', str(output.get('sha256', ''))), 'Complete output lacks path/SHA')
            else:
                if row.get('output') is not None or row.get('sha256') is not None:
                    artifact = {'output': copy.deepcopy(row.get('output')), 'sha256': row.get('sha256'),
                        'reason': 'retained unaccepted artifact from failed generation; not measured or counted as success'}
                    need('failed_artifact' not in row or row['failed_artifact'] == artifact, 'Conflicting failed artifact provenance')
                    row['failed_artifact'] = artifact
                    row['output'], row['sha256'] = None, None
            records[key] = row
        actual = [r for r in index['records'] if r.get('alias_of') is None]
        if 'physical_executions' in index:
            need(index['physical_executions'] == sum(r['execution_started'] for r in actual), 'Declared physical execution count differs')
        if 'unique_output_images' in index:
            digests = {r['output']['sha256'] if isinstance(r['output'], dict) else r['sha256'] for r in index['records'] if r['status'] == 'complete'}
            need(index['unique_output_images'] == len(digests), 'Declared unique image count differs')
    need(methods == set(METHODS), 'All8 methods need terminal indexes; missing rows are retained as missing')
    execution_ids = set()
    for key, row in records.items():
        method, subject, cid, seed = key
        alias = row.get('alias_of')
        if method != 'liveportrait' or seed == 101:
            need(alias is None, 'Stochastic/actual output cannot be aliased')
            need(row['execution_id'] not in execution_ids, 'Physical execution reused across stochastic draws/cases')
            execution_ids.add(row['execution_id'])
        else:
            need(alias == {'method': method, 'case_id': cid, 'seed': 101}, 'LivePortrait must explicitly alias seed101')
            base = records.get((method, subject, cid, 101))
            need(base is not None and all(row.get(k) == base.get(k) for k in ('execution_id', 'execution_started', 'status', 'output')), 'Alias differs from physical output')
    accounting = {}
    for method in METHODS:
        rows = [r for r in records.values() if r['method'] == method]
        actual = [r for r in rows if r.get('alias_of') is None]
        accounting[method] = {'planned_logical_seed_outputs': 96, 'observed_logical_rows': len(rows), 'missing_logical_rows': 96 - len(rows),
            'planned_physical_executions': 32 if method == 'liveportrait' else 96,
            'executed_physical_calls': sum(r['execution_started'] for r in actual), 'successful_physical_calls': sum(r['status'] == 'complete' for r in actual),
            'alias_rows': len(rows) - len(actual), 'unique_successful_image_SHAs': len({r['output']['sha256'] for r in rows if r['status'] == 'complete'})}
    return plan, records, accounting


def statistics_document(doc, metric_rows):
    rows = {row_key(r): r for r in metric_rows}
    need(len(rows) == len(metric_rows), 'Duplicate merged metric row')
    generation = {row_key(r): r for r in doc['normalized_generation_records']}
    records = []
    spec = analysis_spec()
    for p in doc['planned']:
        key = row_key(p)
        g, measured = generation.get(key), rows.get(key)
        if g is None:
            continue
        values, reasons = {}, {}
        for name, definition in spec['metrics'].items():
            value = None
            if g['status'] != 'complete': reason = 'generation_' + g['status']
            elif measured is None: reason = 'measurement_missing'
            elif definition['scope'] != 'all' and measured['group'] != definition['scope']: reason = 'out_of_predeclared_scope'
            else:
                value = measured.get(definition['source_metric'])
                need(value is None or type(value) in (int, float) and math.isfinite(value), 'Nonfinite metric')
                reason = None if value is not None else 'measurement_or_reference_NA'
            values[name], reasons[name] = value, reason
        records.append({**p, 'status': g['status'], 'metrics': values, 'metric_reasons': reasons,
            'execution_id': g['execution_id'], 'alias_of': g.get('alias_of'), 'output': g.get('output'),
            'failed_artifact': g.get('failed_artifact')})
    return {'kind': 'paired_subject_metric_input', 'metadata': {'data_role': ROLE, 'independent_final_test': False,
        'previously_exposed_subjects': True, 'protocol_sha256': doc['protocol_sha256'], 'measurement_policy_sha256': POLICY_SHA,
        'physical_execution_accounting': doc['execution_accounting'], 'primary_metric': PRIMARY,
        'primary_family': 'external_primary_C', 'primary_families': ['external_primary_C'],
        'alias_seed_rows_are_not_independent_generations': True}, 'analysis_spec': spec,
        'planned': doc['planned'], 'records': records}


def enrich_coverage(result, input_document, case_metadata):
    """Separate known out-of-scope rows from generation/evaluation failure counts."""
    records = {row_key(r): r for r in input_document['records']}
    for row in result['coverage_table']:
        definition = input_document['analysis_spec']['metrics'][row['metric']]
        applicable = [p for p in input_document['planned'] if p['method'] == row['method'] and
            (definition['scope'] == 'all' or case_metadata[p['case_id']]['group'] == definition['scope'])]
        reasons = {}
        for p in applicable:
            r = records.get(row_key(p))
            reason = 'missing_output' if r is None else r['metric_reasons'].get(row['metric'])
            if reason is not None:
                reasons[reason] = reasons.get(reason, 0) + 1
        row.update(predeclared_scope=definition['scope'], planned_applicable_seed_outputs=len(applicable),
            out_of_scope_seed_outputs=row['planned_seed_outputs'] - len(applicable),
            applicable_unavailable_by_reason=reasons,
            applicable_generation_or_measurement_unavailable=len(applicable) - row['valid_metric_outputs'])
    for row in result['comparisons']:
        row['all_eight_subjects_paired'] = row['n_subjects'] == 8
        row['expected_sign_patterns_at_full_coverage'] = 256
    return result
