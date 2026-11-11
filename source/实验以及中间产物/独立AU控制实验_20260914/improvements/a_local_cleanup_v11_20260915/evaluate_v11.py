#!/usr/bin/env python3
"""V11 dev8 local cleanup: frozen scoring, 32 new generations + 88 references."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import struct
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V10 = HERE.parent / 'a_terminal_cleanup_v10_20260915'
V10_EVALUATOR = V10 / 'evaluate_v10.py'
V10_EVALUATOR_SHA = '71c18dd6027c8945d48b47f8889b54568df166666113342437eccac078f58133'
V10_CASES = V10 / 'runs/terminal_202609151940/cases.json'
V10_CASES_SHA = '227a371225b21066656863cabcede26a3ababb7c6ca785645e02515387ed6982'
CONFIGS = {name: {'strength': .75, 'au_scale': 1.5, 'terminal_unet_calls': 0, 'cleanup_sigma': sigma}
           for name, sigma in [('local_lp_s050', .5), ('local_lp_s075', .75)]}
METHODS = [family + '_' + config for config in CONFIGS for family in ('baseline', 'bidirectional')]
CASE_KIND = 'v11_development_local_cleanup_cases'
RESULT_KIND = 'v11_local_cleanup_diagnostic'
MASK_MANIFEST = HERE / 'masks/first8/manifest.json'
MASK_CODE = HERE / 'cleanup_mask.py'
MASK_MANIFEST_SHA = '0969554b7720c4b0508426559450e57118200df069857f4cb5b8e09dbbafa48c'
MASK_CODE_SHA = '913623ca6c2dd76f0a3bca5a5316a6fb10339a3fe0d506944286fba32ee41be9'
MASK_POLICY_SHA = 'd461ef42704c1c5c09ee79fa86bc275fd1ee071f7c85c4dc24bc723de6541459'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


require(sha(V10_EVALUATOR) == V10_EVALUATOR_SHA, 'Frozen V10 evaluator changed')
_spec = importlib.util.spec_from_file_location('v11_frozen_v10_evaluator', V10_EVALUATOR)
FROZEN_V10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FROZEN_V10)
BASE = FROZEN_V10.BASE
valid_sha = FROZEN_V10.valid_sha
REFERENCES, TIMESTEPS = BASE.REFERENCES, BASE.TIMESTEPS


def derive_core():
    """Transform only final V10 CORE method/kind/count metadata."""
    source = BASE.ADAPTER.DERIVED_SOURCE
    changes = {
        'METHODS = ' + repr(FROZEN_V10.METHODS): 'METHODS = ' + repr(METHODS),
        repr(FROZEN_V10.CASE_KIND): repr(CASE_KIND),
        repr(FROZEN_V10.RESULT_KIND): repr(RESULT_KIND),
        "len(records) == 48, 'Expected all 48 planned generation records, including failures'":
            "len(records) == 32, 'Expected all 32 planned generation records, including failures'",
        "len(items) == 136, 'Duplicate logical record'": "len(items) == 120, 'Duplicate logical record'",
        "'logical_count': 136": "'logical_count': 120",
        '88 <= count <= 136': '88 <= count <= 120',
        "'new_generated_planned': 48": "'new_generated_planned': 32",
        "'logical_records': 136": "'logical_records': 120",
    }
    for old, new in changes.items():
        require(source.count(old) == 1, 'Ambiguous V11 metadata binding: ' + old)
        source = source.replace(old, new)
    return source, [{'old': old, 'new': new} for old, new in changes.items()]


DERIVED_SOURCE, ADAPTATIONS = derive_core()
CORE = types.ModuleType('v11_frozen_metric_core')
CORE.__file__ = str(Path(__file__).resolve())
exec(compile(DERIVED_SOURCE, str(BASE.ADAPTER.V5_CORE), 'exec'), CORE.__dict__)
_primitive_validate = CORE.validate
_frozen_merge = CORE.merge


def check_policy():
    require(sha(V10_EVALUATOR) == V10_EVALUATOR_SHA, 'Frozen V10 evaluator changed')
    return FROZEN_V10.check_policy()


def frozen_cases():
    check_policy()
    require(sha(V10_CASES) == V10_CASES_SHA, 'Frozen V10 cases changed')
    document = CORE.read(V10_CASES)
    require(document['kind'] == FROZEN_V10.CASE_KIND and document['test_data_used'] is False and
            document['case_count'] == len(document['cases']) == 8, 'Invalid V10 case source')
    FROZEN_V10.reference_lineage(document['cases'])
    return document['cases']


def reference_lineage(cases):
    lineage = FROZEN_V10.reference_lineage(cases)
    old = {case['case_id']: case for case in frozen_cases()}
    require(len(cases) == 8 and {case['case_id'] for case in cases} == set(old), 'V11 dev8 case IDs changed')
    for case in cases:
        for key, value in old[case['case_id']].items():
            require(case.get(key) == value, 'V11 changed frozen V10 case field: ' + key)
    lineage.update(v10_case_manifest={'path': str(V10_CASES), 'sha256': V10_CASES_SHA},
                   all_old_v10_case_fields_preserved=True)
    return lineage


def provenance():
    return {'v10_evaluator': str(V10_EVALUATOR), 'v10_evaluator_sha256': V10_EVALUATOR_SHA,
            'v10_final_core_sha256': hashlib.sha256(BASE.ADAPTER.DERIVED_SOURCE.encode()).hexdigest(),
            'v11_entry_sha256': sha(__file__), 'v11_core_sha256': hashlib.sha256(DERIVED_SOURCE.encode()).hexdigest(),
            'metadata_adaptations': ADAPTATIONS, 'metric_model_selection_standardization_changed': False,
            'frozen_policy_sha256': CORE.POLICY_SHA,
            'cleanup_manifest_sha256': MASK_MANIFEST_SHA, 'cleanup_code_sha256': MASK_CODE_SHA,
            'cleanup_policy_sha256': MASK_POLICY_SHA}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def read_mask64(path):
    """Read the finite little-endian float32 NPY payload without NumPy/ML imports."""
    data = Path(path).read_bytes()
    require(data[:6] == b'\x93NUMPY' and data[6:8] in (b'\x01\x00', b'\x02\x00'), 'Unsupported mask NPY header')
    start = 10 if data[6] == 1 else 12
    length = struct.unpack('<H' if start == 10 else '<I', data[8:start])[0]
    header = ast.literal_eval(data[start:start + length].decode('latin1').strip())
    require(header == {'descr': '<f4', 'fortran_order': False, 'shape': (64, 64)}, 'Expected C-order float32 mask64')
    body = data[start + length:]
    require(len(body) == 4096 * 4, 'Unexpected mask byte length')
    values = struct.unpack('<4096f', body)
    require(all(math.isfinite(value) and 0 <= value <= 1 for value in values), 'Mask must be finite in [0,1]')
    return values, hashlib.sha256(body).hexdigest()


def support_bundle():
    require(all(isinstance(value, str) and len(value) == 64 for value in (MASK_MANIFEST_SHA, MASK_CODE_SHA, MASK_POLICY_SHA)),
            'V11 source68 masks have not yet been frozen')
    require(sha(MASK_MANIFEST) == MASK_MANIFEST_SHA and sha(MASK_CODE) == MASK_CODE_SHA, 'Frozen cleanup support changed')
    document = CORE.read(MASK_MANIFEST)
    require(document['kind'] == 'v11_source_only_cleanup_masks' and document['status'] == 'complete' and
            document['case_count'] == len(document['cases']) == 8, 'Invalid cleanup mask manifest')
    require(document['policy_sha256'] == MASK_POLICY_SHA == canonical_sha(document['policy']), 'Cleanup policy changed')
    for container in (document, document['policy']):
        require(container.get('source_only') is True and container.get('test_used') is False and
                container.get('target_geometry_used') is False and container.get('generated_pixels_used') is False,
                'Cleanup geometry is not strictly source-only')
    require(document['policy'].get('subject_exceptions') is False and document['policy'].get('outcome_tuning') is False,
            'Subject exceptions/outcome tuning forbidden')
    require(document['code'] == {'path': str(MASK_CODE), 'sha256': MASK_CODE_SHA}, 'Cleanup code binding differs')
    require(document['inputs']['cases'] == {'path': str(V10_CASES), 'sha256': V10_CASES_SHA}, 'Cleanup case input differs')
    for ref in document['inputs'].values():
        CORE.checked_image(ref)
    source = CORE.read(document['inputs']['source_measurements']['path'])
    records = {(row['case_id'], row['method']): row for row in source['records']}
    return document, records


def validate_cleanup_metadata(case, bundle=None):
    document, sources = bundle or support_bundle()
    entries = {entry['case_id']: entry for entry in document['cases']}
    ref = case.get('cleanup_mask')
    require(ref == entries.get(case['case_id']), 'Case cleanup support differs from frozen manifest')
    for item in (ref, ref['receipt'], ref['source_debug']):
        CORE.checked_image(item)
    require(ref['source_sha256'] == case['source_real']['sha256'] and ref['policy_sha256'] == MASK_POLICY_SHA,
            'Cleanup source or policy binding differs')
    receipt = CORE.read(ref['receipt']['path'])
    require(receipt['case_id'] == case['case_id'] and receipt['policy_sha256'] == MASK_POLICY_SHA,
            'Cleanup receipt case/policy differs')
    require(receipt['source_only'] is True and receipt['test_used'] is False and receipt['target_geometry_used'] is False and
            receipt['generated_pixels_used'] is False and receipt['source_image_unchanged'] is True, 'Non-source support receipt')
    require(receipt['cleanup_mask'] == {'path': ref['path'], 'sha256': ref['sha256']} and
            receipt['source_debug'] == ref['source_debug'], 'Cleanup file receipt differs')
    require(receipt['edit_mask'] == {key: case['region_mask'][key] for key in ('path', 'sha256')}, 'Original edit mask changed')
    changed = [au for au, before, after in zip(CORE.AUS, case['source_aus_0_5'], case['target_aus_0_5']) if before != after]
    active = (['glabella'] if set(changed).intersection((1, 2, 4)) else []) + (['chin'] if 17 in changed else [])
    require(receipt['changed_aus'] == ref['changed_aus'] == changed and receipt['active_regions'] == ref['active_regions'] == active,
            'Cleanup active AU/region rules differ')
    require(receipt['source_aus_0_5'] == case['source_aus_0_5'] and receipt['target_aus_0_5'] == case['target_aus_0_5'],
            'Cleanup AU controls changed')
    source = sources[(case['case_id'], 'source_real')]
    require(source['method'] == 'source_real' and source['split'] == case['split'] == 'dev' and
            source['path'] == case['source_real']['path'] and source['sha256'] == receipt['source_sha256'] == ref['source_sha256'],
            'Cleanup landmarks not bound to source_real')
    selected = source['selected_row']
    require(type(selected) is int and 0 <= selected < len(source['raw_rows']) and receipt['selected_source_row'] == selected,
            'Source selected row changed')
    raw = source['raw_rows'][selected]
    points = [[float(raw[f'x_{index}']), float(raw[f'y_{index}'])] for index in range(68)]
    require(receipt['source_landmarks_68_512'] == points and receipt['source_landmarks_sha256'] == canonical_sha(points),
            'Cleanup landmarks differ from original source measurement')
    require(all(math.isfinite(value) for point in points for value in point) and
            math.isclose(receipt['face_scale_px'], math.dist(points[0], points[16]), abs_tol=1e-10) and
            receipt['source_confidence'] == source['confidence'] >= .5, 'Source geometry scale/confidence changed')
    values, digest = read_mask64(ref['path'])
    edit, _ = read_mask64(case['region_mask']['path'])
    require(receipt['mask_bytes_sha256'] == digest and all(value == 0 for value, support in zip(values, edit) if support <= 0),
            'Cleanup mask bytes or edit support intersection changed')
    area = {'alpha_sum_64': math.fsum(values), 'alpha_mean_64': math.fsum(values) / 4096,
            'positive_fraction_64': sum(value > 0 for value in values) / 4096}
    require(receipt['mask_area'] == ref['mask_area'] and all(math.isclose(area[key], receipt['mask_area'][key], abs_tol=1e-10)
            for key in area), 'Cleanup mask area differs')
    require(receipt['raster_protected_pixels_overlap'] == 0 and receipt['empty_support'] == (not any(values)),
            'Protected region or empty-support receipt differs')
    if not active or case['group'] == 'no_edit':
        require(not any(values), 'Unrelated/no-edit AU case must have zero cleanup support')
    return {'case_id': case['case_id'], 'source_sha256': ref['source_sha256'], 'policy_sha256': MASK_POLICY_SHA,
            'mask_sha256': ref['sha256'], 'active_regions': active, 'changed_aus': changed, 'mask_area': area,
            'source68_matches_original_measurement': True, 'source_only': True}


def filter_config(sigma):
    radius = 2 if sigma == .5 else 3
    return {'sigma_latent': sigma, 'radius': radius, 'kernel_size': radius * 2 + 1, 'padding': 'reflect',
            'normalization': 'conv(editmask*delta)/conv(editmask)', 'denominator_epsilon': 1e-8,
            'denominator_fallback': 'original_delta',
            'local_blend': 'clean + cleanupmask * (filtered_delta - original_delta)',
            'outside_cleanup_exact': True, 'final_projection': 'original_editmask_once'}


def audit_local_terminal(row, old, case):
    config = row['config']
    require(row['cleanup_mask'] == case['cleanup_mask'], 'Generated cleanup support differs from case')
    for key in ('prompt', 'condition_hashes', 'region_mask', 'source_latent_sha256', 'noise_sha256'):
        require(row.get(key) == old[key], 'Generated A binding changed: ' + key)
    for key, value in {'strength': .75, 'au_scale': 1.5, 'ip_scale': .9, 'nominal_steps': 20,
                       'cleanup_sigma': CONFIGS[config]['cleanup_sigma']}.items():
        require(row.get(key) == value, 'Generated config changed: ' + key)
    if row['status'] != 'complete':
        return None
    counts = {'actual_steps': 15, 'denoising_steps': 15, 'deis_unet_calls': 15, 'terminal_unet_calls': 0, 'total_unet_calls': 15}
    require(all(type(row.get(key)) is int and row[key] == value for key, value in counts.items()), 'Actual UNet count mismatch')
    require(row['actual_timesteps'] == TIMESTEPS and row['projection_trace'][:14] == old['projection_trace'][:14] and
            len(row['projection_trace']) == 15 and row['projection_trace'][-1]['next_t'] is None, 'Original A DEIS prefix changed')
    require(row.get('original_a_projection_prefix') == old['projection_trace'][:14] and row.get('a_prefix_verified') is True,
            'Original A prefix receipt is missing or differs')
    calls = row['unet_call_trace']
    require(len(calls) == 15 and all(call['phase'] == 'deis' and call['t'] == t and call['au_scale'] == 1.5 and
            call['cfg_batch'] == 2 and FROZEN_V10.valid_sha(call.get('raw_epsilon_sha256')) and
            FROZEN_V10.valid_sha(call.get('prompt_embeds_sha256')) for call, t in zip(calls, TIMESTEPS)), 'Invalid actual UNet trace')
    require(len({call['prompt_embeds_sha256'] for call in calls}) == 1, 'Prompt embeddings changed during calls')
    trace = row['terminal_trace']
    require(trace['mode'] == config and trace['epsilon_source'] == 'last_t50' and trace['extra_cfg'] is None and
            trace['last_deis_timestep'] == 50 and trace['terminal_raw_epsilon_sha256'] is None and
            trace['last_deis_raw_epsilon_sha256'] == calls[-1]['raw_epsilon_sha256'], 'Terminal epsilon path changed')
    require(trace['scheduler_step_index_before'] == trace['scheduler_step_index_after'] == 20, 'Cleanup advanced scheduler')
    require(trace['filter'] == filter_config(row['cleanup_sigma']) and trace['cleanup_mask'] == row['cleanup_mask'], 'Local filter metadata changed')
    require(trace.get('base_mode') == 'terminal_last_eps' and trace.get('base_terminal_clean_latent_sha256') == trace['clean_latent_sha256'] and
            trace.get('outside_cleanup_clean_exact') is True, 'Base clean or outside-cleanup exactness receipt changed')
    for key in ('input_latent_sha256', 'epsilon_cfg_sha256', 'clean_latent_sha256', 'post_filter_clean_latent_sha256', 'projected_latent_sha256'):
        require(FROZEN_V10.valid_sha(trace.get(key)), 'Missing cleanup tensor hash: ' + key)
    require(all(type(trace.get(key)) in (int, float) and math.isfinite(trace[key]) for key in ('rho0', 'alpha0', 'sigma0')) and
            trace['alpha0'] > 0 and trace['sigma0'] >= 0 and
            math.isclose(trace['rho0'], trace['sigma0'] / trace['alpha0'], rel_tol=1e-6, abs_tol=1e-8), 'Terminal coefficients inconsistent')
    require(trace['projected_latent_sha256'] == row['projection_trace'][-1]['latent_sha256'], 'Final projection hash differs')
    if not row['cleanup_mask']['mask_area']['alpha_sum_64']:
        require(trace['post_filter_clean_latent_sha256'] == trace['clean_latent_sha256'], 'Empty cleanup changed clean latent')
    if row['group'] == 'no_edit':
        require(trace['projected_latent_sha256'] == row['source_latent_sha256'], 'No-edit did not restore source latent')
    return {'case_id': row['case_id'], 'method': row['method'], 'config': config, **counts, 'A_prefix_states_preserved': 14}


def validate(run):
    document = _primitive_validate(run)
    run = Path(document['run'])
    result = CORE.read(run / 'result.json')
    require(sha(run / 'result.json') == document['generation_manifest_hashes'][str(run / 'result.json')], 'Generation result changed')
    ref = CORE.checked_image(result['configuration'])
    require(Path(ref['path']) == run / 'configuration.json', 'Configuration outside this run')
    configuration = CORE.read(ref['path'])
    require(sha(ref['path']) == ref['sha256'], 'Configuration changed during read')
    for key, value in {'kind': RESULT_KIND, 'test_data_used': False, 'training_performed': False, 'guidance_scale': 7.5,
                       'ip_scale': .9, 'seed': 101, 'nominal_steps': 20}.items():
        require(configuration.get(key) == value, 'Shared configuration changed: ' + key)
    require(configuration['configs'] == [{'name': name, **params} for name, params in CONFIGS.items()], 'Cleanup configs differ')
    require(configuration['cleanup_mask_manifest'] == {'path': str(MASK_MANIFEST), 'sha256': MASK_MANIFEST_SHA}, 'Cleanup manifest binding differs')
    cases = {case['case_id']: case for case in CORE.read(run / 'cases.json')['cases']}
    bundle = support_bundle()
    supports = [validate_cleanup_metadata(case, bundle) for case in cases.values()]
    _, old = BASE.frozen_v7()
    audits = []
    for item in document['items']:
        if item['role'] == 'new_generation':
            row = item['generation_record']
            family, config = row['method'].split('_', 1)
            require(row['config'] == config, 'Method/config mismatch')
            result = audit_local_terminal(row, old[row['case_id'], family + '_smooth_s075_a150'], cases[row['case_id']])
            if result is not None:
                audits.append(result)
    document.update(configs=CONFIGS, cleanup_support_audit=supports, terminal_call_audit=audits,
                    adapter_provenance=provenance(), source_only_cleanup=True)
    return document


def metadata():
    cases = frozen_cases()
    lineage = reference_lineage(cases)
    for case in cases:
        for method in REFERENCES:
            CORE.checked_image(case[method])
    bundle = support_bundle()
    entries = {entry['case_id']: entry for entry in bundle[0]['cases']}
    support = [validate_cleanup_metadata({**case, 'cleanup_mask': entries[case['case_id']]}, bundle) for case in cases]
    return {'status': 'metadata_verified_no_models_run', 'generator_schema_ready': True,
            'test_used': False, 'case_count': 8, 'reference_count': 88, 'new_generated_planned': 32,
            'logical_records': 120, 'case_kind': CASE_KIND, 'result_kind': RESULT_KIND,
            'methods': METHODS, 'references': REFERENCES, 'configs': CONFIGS,
            'actual_timesteps': TIMESTEPS, 'original_metric_columns': CORE.METRICS,
            'policy_sha256': CORE.POLICY_SHA, 'reference_lineage': lineage, 'cleanup_support_audit': support,
            'adapter_provenance': provenance()}


def grouped_statistics(rows, methods):
    split = CORE.aggregate(rows, methods)
    full = CORE.aggregate([{**row, 'group': 'lowest_available'} for row in rows], methods)['edited']
    return {'full8': full, 'no_edit4': split['no_edit'], 'edited4': split['edited']}


def supplemental_statistics(report):
    rows = report['records']
    all_methods = REFERENCES + METHODS
    return {'status': 'complete', 'test_used': False, 'logical_records': 120,
            'original_metric_columns': CORE.METRICS, 'buffalo_status': 'not_run',
            'statistics_policy': 'Frozen per-metric common-valid intersection, case then subject-equal means; NA retained.',
            'group_note': 'full8 does not make no-edit changed-AU metrics defined; their valid count remains at most 4.',
            'per_method': {method: grouped_statistics(rows, [method]) for method in all_methods},
            'by_subject': {subject: {method: grouped_statistics(
                [row for row in rows if row['subject'] == subject], [method]) for method in all_methods}
                for subject in sorted(CORE.SUBJECTS)},
            'fair_baseline_pairs': {config: grouped_statistics(rows, ['baseline_' + config, 'bidirectional_' + config])
                                    for config in CONFIGS},
            'candidate_vs_a': {method + '_vs_a_' + method.split('_', 1)[0]: grouped_statistics(
                rows, [method, 'a_' + method.split('_', 1)[0]]) for method in METHODS},
            'terminal_counts': CONFIGS, 'conclusion_scope': report['conclusion_scope'],
            'cleanup_scope': 'Frozen source-only support, no per-subject exceptions; V11 hypothesis was informed by prior development images. Zero protected overlap describes the 512 mask raster, not final RGB invariance.',
            'compute_budget_note': 'Every candidate uses 15 DEIS UNet calls and zero terminal UNet calls; total 15.'}


def merge(output):
    _frozen_merge(output)
    output = CORE.scoped_path(output, HERE)
    report = CORE.read(output / 'comparison/results.json')
    stats = supplemental_statistics(report)
    stats.update(frozen_comparison_sha256=CORE.sha(output / 'comparison/results.json'), adapter_provenance=provenance())
    CORE.save(output / 'comparison/v11_statistics.json', stats)


CORE.check_policy = check_policy
CORE.reference_lineage = reference_lineage
CORE.validate = validate
CORE.merge = merge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('metadata', 'validate', 'prepare', 'openface', 'facenet', 'merge', 'all'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase in ('metadata', 'validate'):
        require(args.phase != 'validate' or args.run is not None, '--run required')
        report = metadata() if args.phase == 'metadata' else validate(args.run)
        print(json.dumps({key: value for key, value in report.items() if key != 'items'}, ensure_ascii=False))
        return
    require(args.output is not None, 'Fresh V11 --output required')
    if args.phase in ('prepare', 'all'):
        require(args.run is not None, '--run required')
        CORE.prepare(args.run, args.output)
    if args.phase in ('openface', 'all'):
        CORE.openface(args.output)
    if args.phase in ('facenet', 'all'):
        CORE.facenet(args.output)
    if args.phase in ('merge', 'all'):
        merge(args.output)


if __name__ == '__main__':
    main()
