#!/usr/bin/env python3
"""Dev8 late-AU diagnostic: frozen CPU scoring, 136 logical rows.

Import, metadata, validate, prepare and tests use no model runtime. OpenFace and
FaceNet run only in explicit openface/facenet/all phases after generation ends.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
KERNEL = HERE.parent / 'region_preserving_v6_20260915/evaluate_v6.py'
KERNEL_SHA = 'e72173eb31f4101eb7e5b586e59cea5c3aae68ece44455c77fa211814e65ad67'
V7_RUN = HERE.parent / 'smooth_regions_v7_20260915/runs/smooth_202609151710'
V7_INPUTS = {
    'cases.json': '5335ff4c2b41585b4ab83ddfe1253edf1de127b01ddb9190cea542c0078c3c90',
    'result.json': 'e59ce11ad87ff6bba91c8131019997833fee0f482a3df364e2d6f28e80aa079d',
}
CONFIGS = {
    name: {'strength': .75, 'au_scale_initial': 1.5, 'au_scale_late': late,
           'au_late_max_t': 250}
    for name, late in [('late_au100', 1.0), ('late_au050', .5), ('late_au000', 0.0)]
}
A_REFERENCES = ['a_baseline', 'a_bidirectional']
CASE_KIND = 'v9_development_late_au_cases'
RESULT_KIND = 'v9_late_au_diagnostic'
TIMESTEPS = [749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def derive_v6_adapter():
    """SHA-gated metadata bindings; no metric or model implementation changes."""
    raw = KERNEL.read_bytes()
    require(sha_bytes(raw) == KERNEL_SHA, 'Frozen v6 evaluator changed')
    source = raw.decode('utf-8')
    old_configs = "CONFIGS = {\n    'masked_s055_a100': {'strength': .55, 'au_scale': 1.0},\n    'masked_s075_a100': {'strength': .75, 'au_scale': 1.0},\n    'masked_s075_a150': {'strength': .75, 'au_scale': 1.5},\n}"
    changes = {
        old_configs: 'CONFIGS = ' + repr(CONFIGS),
        'REFERENCES = BASE_REFERENCES + IMG2IMG_REFERENCES':
            'REFERENCES = BASE_REFERENCES + IMG2IMG_REFERENCES + ' + repr(A_REFERENCES),
        "'v6_development_region_preserving_cases'": repr(CASE_KIND),
        "'v6_region_preserving_diagnostic'": repr(RESULT_KIND),
        '"len(items) == 120, \'Duplicate logical record\'"':
            '"len(items) == 136, \'Duplicate logical record\'"',
        '"\'logical_count\': 120"': '"\'logical_count\': 136"',
        "'72 <= count <= 120'": "'88 <= count <= 136'",
        "repr(IMG2IMG_REFERENCES) + ' + METHODS'":
            "repr(IMG2IMG_REFERENCES + " + repr(A_REFERENCES) + ") + ' + METHODS'",
        '"\'logical_records\': 120"': '"\'logical_records\': 136"',
        "'reference_count': 72": "'reference_count': 88",
        "'logical_records': 120": "'logical_records': 136",
        "expected_steps = 11 if CONFIGS[config]['strength'] == .55 else 15": 'expected_steps = 15',
        "require(not any(isinstance(node, ast.Constant) and type(node.value) is int and node.value == 88\n"
        "                    for node in ast.walk(ast.parse(source))), 'Stale v5 count remains')":
            "require(not any(text in source for text in (\"len(items) == 88,\", \"'logical_count': 88\", "
            "\"'logical_records': 88\")), 'Stale v5 logical count remains')",
    }
    for old, new in changes.items():
        require(source.count(old) == 1, 'Ambiguous v9 metadata binding: ' + old)
        source = source.replace(old, new)
    return source, [{'old': old, 'new': new} for old, new in changes.items()]


V6_DERIVED_SOURCE, V9_ADAPTATIONS = derive_v6_adapter()
ADAPTER = types.ModuleType('v9_sha_pinned_v6_adapter')
ADAPTER.__file__ = str(Path(__file__).resolve())
exec(compile(V6_DERIVED_SOURCE, str(KERNEL), 'exec'), ADAPTER.__dict__)
CORE = ADAPTER.CORE
METHODS, REFERENCES = ADAPTER.METHODS, ADAPTER.REFERENCES
_base_check_policy = ADAPTER.check_policy
_base_frozen_cases = ADAPTER.frozen_cases
_base_reference_lineage = ADAPTER.reference_lineage
_base_validate = ADAPTER.validate
_base_provenance = ADAPTER.provenance
_base_merge = CORE.merge


def check_policy():
    require(CORE.sha(KERNEL) == KERNEL_SHA, 'Frozen v6 evaluator changed')
    return _base_check_policy()


def frozen_v7():
    check_policy()
    for name, digest in V7_INPUTS.items():
        require(CORE.sha(V7_RUN / name) == digest, 'Frozen v7 input changed: ' + name)
    cases, result = CORE.read(V7_RUN / 'cases.json'), CORE.read(V7_RUN / 'result.json')
    require(cases['kind'] == 'v7_development_region_preserving_cases' and
            result['kind'] == 'v7_region_preserving_diagnostic', 'Wrong v7 reference manifest')
    require(cases['test_data_used'] is False and result['test_data_used'] is False and
            result['status'] == 'complete', 'V7 references must be completed dev data')
    mapping = {(row['case_id'], row['method']): row for row in result['records']}
    require(len(mapping) == len(result['records']) == 48, 'Duplicate or missing v7 record')
    return cases['cases'], mapping


def frozen_cases():
    """Preserve every old v7 case field, adding only the two user-selected A refs."""
    previous = {case['case_id']: case for case in _base_frozen_cases()}
    cases, records = frozen_v7()
    require(len(cases) == 8 and {case['case_id'] for case in cases} == set(previous),
            'V7 does not match frozen v5 dev8')
    for case in cases:
        for key, value in previous[case['case_id']].items():
            require(case.get(key) == value, 'V7 changed historical v5 case field: ' + key)
        for family in ('baseline', 'bidirectional'):
            row = records[(case['case_id'], family + '_smooth_s075_a150')]
            require(row['status'] == 'complete', 'A reference generation incomplete')
            for key in ('subject', 'split', 'group', 'source_aus_0_5', 'target_aus_0_5', 'prompt', 'region_mask'):
                require(row[key] == case[key], 'A reference/case mismatch: ' + key)
            require(row['strength'] == .75 and row['au_scale'] == 1.5 and row['ip_scale'] == .9 and
                    row['nominal_steps'] == 20 and row['actual_steps'] == 15 and
                    row['actual_timesteps'] == TIMESTEPS, 'A generation config changed')
            case['a_' + family] = {'path': row['output'], 'sha256': row['sha256']}
    return cases


def reference_lineage(cases):
    lineage = _base_reference_lineage(cases)
    lineage['v5_reference_bindings'] = [binding for binding in lineage['v5_reference_bindings']
                                       if binding['role'] not in A_REFERENCES]
    bindings = [{'case_id': case['case_id'], 'role': role, **case[role]}
                for case in cases for role in REFERENCES]
    # Byte-bind the unchanged region mask and its receipt, not only their paths.
    for case in cases:
        CORE.checked_image(case['region_mask'])
        CORE.checked_image(case['region_mask']['receipt'])
    lineage.update(v7_inputs={name: {'path': str(V7_RUN / name), 'sha256': digest}
                              for name, digest in V7_INPUTS.items()},
                   v7_reference_bindings=bindings,
                   a_reference_methods={'a_' + family: family + '_smooth_s075_a150'
                                        for family in ('baseline', 'bidirectional')},
                   all_old_v7_case_fields_preserved=True)
    return lineage


def provenance():
    return {**_base_provenance(), 'v6_kernel': str(KERNEL), 'v6_kernel_sha256': KERNEL_SHA,
            'v9_entry_sha256': CORE.sha(__file__), 'v9_metadata_adaptations': V9_ADAPTATIONS,
            'v9_derived_v6_sha256': sha_bytes(V6_DERIVED_SOURCE.encode('utf-8')),
            'v7_run': str(V7_RUN), 'v7_input_sha256': V7_INPUTS,
            'metric_model_selection_standardization_changed': False}


def generation_configuration(run, document):
    run = CORE.scoped_path(run, HERE)
    path = run / 'result.json'
    result = CORE.read(path)
    require(CORE.sha(path) == document['generation_manifest_hashes'][str(path)],
            'Generation result changed while reading configuration')
    ref = CORE.checked_image(result['configuration'])
    require(Path(ref['path']) == run / 'configuration.json', 'Configuration must belong to this run')
    config = CORE.read(ref['path'])
    require(CORE.sha(ref['path']) == ref['sha256'], 'Configuration changed while reading')
    require(config.get('kind') == RESULT_KIND and config.get('test_data_used') is False and
            config.get('training_performed') is False, 'Configuration is not an untrained dev diagnostic')
    for key, value in {'guidance_scale': 7.5, 'ip_scale': .9, 'seed': 101, 'nominal_steps': 20}.items():
        require(config.get(key) == value, 'Shared generation config changed: ' + key)
    require(config.get('configs') == [{'name': name, **params} for name, params in CONFIGS.items()],
            'Configuration candidate list differs from record schedule')
    return config


def validate(run):
    document = _base_validate(run)
    generation_configuration(run, document)
    _, old_records = frozen_v7()
    audit = []
    for item in document['items']:
        if item['role'] != 'new_generation':
            continue
        row = item['generation_record']
        family, config = row['method'].split('_', 1)
        old = old_records[(row['case_id'], family + '_smooth_s075_a150')]
        require(row.get('ip_scale') == .9, 'IP scale changed from A')
        for key in ('condition_hashes', 'region_mask', 'prompt'):
            require(row.get(key) == old[key], 'Generated A binding changed: ' + key)
        if row['status'] != 'complete':
            continue  # Failed generations stay NA and are not evaluated.
        for key in ('source_latent_sha256', 'noise_sha256'):
            require(row.get(key) == old[key], 'Generated A binding changed: ' + key)
        require(row.get('actual_timesteps') == TIMESTEPS, 'Actual timestep schedule differs from A')
        expected = [{'t': t, 'au_scale': CONFIGS[config]['au_scale_late'] if t <= 250 else 1.5}
                    for t in TIMESTEPS]
        trace = row.get('au_scale_trace')
        require(isinstance(trace, list) and len(trace) == 15, 'Expected 15 actual UNet AU scales')
        require(all(isinstance(entry, dict) and entry.get('t') == target['t'] and
                    entry.get('au_scale') == target['au_scale'] for entry, target in zip(trace, expected)),
                'Actual UNet AU scale trace disagrees with schedule')
        projection = row.get('projection_trace')
        require(isinstance(projection, list) and len(projection) == 15 and
                all(isinstance(entry, dict) and entry.get('t') == target['t'] and
                    entry.get('au_scale_used') == target['au_scale'] for entry, target in zip(projection, expected)),
                'Projection trace and actual UNet AU scales disagree')
        audit.append({'case_id': row['case_id'], 'method': row['method'], 'config': config,
                      'actual_steps': 15, 'initial_steps': 10, 'late_steps': 5,
                      'a_condition_and_noise_preserved': True})
    document.update(adapter_provenance=provenance(), schedule_audit=audit,
                    fair_baseline_pairs={config: ['baseline_' + config, 'bidirectional_' + config]
                                         for config in CONFIGS})
    return document


def metadata():
    cases = frozen_cases()
    lineage = reference_lineage(cases)
    for case in cases:
        for method in REFERENCES:
            CORE.checked_image(case[method])
    return {'status': 'metadata_verified_no_models_run', 'test_used': False,
            'case_count': 8, 'reference_count': 88, 'new_generated_planned': 48,
            'logical_records': 136, 'case_kind': CASE_KIND, 'result_kind': RESULT_KIND,
            'methods': METHODS, 'references': REFERENCES, 'configs': CONFIGS,
            'actual_timesteps': TIMESTEPS, 'original_metric_columns': CORE.METRICS,
            'policy_sha256': CORE.POLICY_SHA, 'reference_lineage': lineage,
            'adapter_provenance': provenance()}


def grouped_statistics(rows, methods):
    """Use frozen subject-equal aggregation unchanged; include a full8 view."""
    split = CORE.aggregate(rows, methods)
    # Only grouping metadata changes; null metrics are retained, so full8 has
    # four valid edited cases for changed-AU metrics and up to eight otherwise.
    full = CORE.aggregate([{**row, 'group': 'lowest_available'} for row in rows], methods)['edited']
    return {'full8': full, 'no_edit4': split['no_edit'], 'edited4': split['edited']}


def supplemental_statistics(report):
    rows = report['records']
    all_methods = REFERENCES + METHODS
    fairness = {config: grouped_statistics(rows, ['baseline_' + config, 'bidirectional_' + config])
                for config in CONFIGS}
    return {'status': 'complete', 'test_used': False, 'logical_records': 136,
            'original_metric_columns': CORE.METRICS, 'buffalo_status': 'not_run',
            'statistics_policy': 'Frozen per-metric common-valid intersection, case then subject-equal means; NA retained.',
            'group_note': 'full8 does not make no-edit changed-AU metrics defined; their valid count remains at most 4.',
            'per_method': {method: grouped_statistics(rows, [method]) for method in all_methods},
            'by_subject': {subject: {method: grouped_statistics(
                [row for row in rows if row['subject'] == subject], [method]) for method in all_methods}
                for subject in sorted(CORE.SUBJECTS)},
            'fair_baseline_pairs': fairness,
            'candidate_vs_a': {method + '_vs_a_' + method.split('_', 1)[0]: grouped_statistics(
                rows, [method, 'a_' + method.split('_', 1)[0]]) for method in METHODS},
            'conclusion_scope': report['conclusion_scope']}


def merge(output):
    _base_merge(output)
    output = CORE.scoped_path(output, HERE)
    report_path = output / 'comparison/results.json'
    report = CORE.read(report_path)
    supplement = supplemental_statistics(report)
    supplement['frozen_comparison_sha256'] = CORE.sha(report_path)
    supplement['adapter_provenance'] = provenance()
    CORE.save(output / 'comparison/v9_statistics.json', supplement)


ADAPTER.check_policy = CORE.check_policy = check_policy
ADAPTER.frozen_cases = frozen_cases
ADAPTER.reference_lineage = CORE.reference_lineage = reference_lineage
ADAPTER.provenance = provenance
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
        document = metadata() if args.phase == 'metadata' else validate(args.run)
        print(json.dumps({key: value for key, value in document.items() if key != 'items'}, ensure_ascii=False))
        return
    require(args.output is not None, '--output required; fresh directory below v9 diagnostic')
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
