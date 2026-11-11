#!/usr/bin/env python3
"""Fresh four-arm output evaluation. Explicit --execute is required for models.

No generator model is imported. No paper files or old measurements are modified.
Each evaluation run is immutable; reruns must use a new job directory.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from ablation_evaluation_contract import (ARMS, AUS, EXP, EXTRA, HERE, METRICS, PINS, POLICY_SHA, PRIMARY,
                      REFERENCES, REFERENCES_SHA, ROLE, SUPPLEMENT, V2,
                      analysis_spec, calibration_plan, code_bindings, load_legacy, metric_functions,
                      normalize_generation, read, ref, reference_cases, require,
                      row_key, save, sha, subgroup_plan, verified)

sys.dont_write_bytecode = True


def image_item(metadata, image_ref, role):
    from PIL import Image
    path = verified(image_ref)
    with Image.open(path) as image:
        image.load()
        require(image.mode == 'RGB' and image.size == (512, 512),
                'Native RGB 512x512 required; no crop/resize/conversion during evaluation')
        dimensions = list(image.size)
    require(sha(path) == image_ref['sha256'], 'Image changed during inspection')
    return {**metadata, 'path': str(path), 'sha256': image_ref['sha256'],
            'image_size': dimensions, 'image_role': role, 'split': ROLE}


def freeze_analysis(output):
    """Seal hypotheses, subgroup definitions and code before any new evaluation."""
    path = Path(output).resolve()
    require(path.is_relative_to(HERE) and not path.exists(), 'New analysis-plan file under evaluation/ required')
    cases = reference_cases(read(verified({'path': str(REFERENCES), 'sha256': REFERENCES_SHA})))
    payload = {
        'kind': 'frozen_supplementary_ablation_analysis_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'data_role': ROLE, 'analysis_spec': analysis_spec(), 'subgroup_plan': subgroup_plan(cases),
        'real_reference_calibration': calibration_plan(),
        'reference_manifest': ref(REFERENCES), 'algorithm_bindings': code_bindings(),
        'source_bindings': {p.name: ref(p) for p in (Path(__file__), HERE / 'ablation_evaluation_contract.py')},
        'plan_count': 288, 'new_subject_test': False, 'no_new_output_images_or_scores_read': True,
    }
    save(path, payload)
    return ref(path)


def prepare(generation_result, output, analysis_plan):
    generation_result = Path(generation_result).resolve()
    require(generation_result.is_relative_to(SUPPLEMENT / 'generation'),
            'Only newly generated supplementary outputs may be evaluated')
    result_ref = ref(generation_result)
    generation = read(generation_result)
    require(generation.get('kind') == 'preservation_factorial_result' and
            generation.get('smoke') is False, 'Only the prespecified full 288-slot run may enter formal evaluation')
    run_root = Path(generation['run_root']).resolve()
    require(run_root.is_relative_to(SUPPLEMENT / 'generation') and
            generation_result.is_relative_to(run_root / 'sessions'), 'Generation result/run root mismatch')
    freeze_path = verified(generation['freeze'])
    require(freeze_path.is_relative_to(SUPPLEMENT / 'generation'), 'Supplementary generation freeze required')
    references = read(verified({'path': str(REFERENCES), 'sha256': REFERENCES_SHA}))
    cases = reference_cases(references)
    analysis_plan = Path(analysis_plan).resolve()
    require(analysis_plan.is_relative_to(HERE), 'Analysis plan outside evaluation/')
    frozen_analysis = read(analysis_plan)
    require(frozen_analysis['kind'] == 'frozen_supplementary_ablation_analysis_v1' and
            frozen_analysis['analysis_spec'] == analysis_spec() and
            frozen_analysis['subgroup_plan'] == subgroup_plan(cases) and
            frozen_analysis['real_reference_calibration'] == calibration_plan(), 'Frozen analysis specification changed')
    for binding in frozen_analysis['source_bindings'].values():
        verified(binding)
    plan, records = normalize_generation(generation, cases)
    require(sha(V2 / 'policy.json') == POLICY_SHA, 'Frozen measurement policy changed')
    algorithms = code_bindings()
    output = Path(output).resolve()
    require(output.is_relative_to(HERE) and output != HERE and not output.exists(),
            'Evaluation requires a new directory under evaluation/')
    items, availability = [], []

    def add(metadata, image_ref, role):
        key = [metadata['method'], metadata['subject'], metadata['case_id'], metadata['seed']]
        try:
            item = image_item(metadata, image_ref, role)
            items.append(item)
            availability.append({'key': key, 'status': 'available', 'role': role, 'image': image_ref})
        except Exception as error:
            availability.append({'key': key, 'status': 'unavailable', 'role': role,
                                 'image': image_ref, 'error': str(error), 'error_type': type(error).__name__})

    for cid, case in cases.items():
        meta = {'case_id': cid, 'subject': case['subject'], 'source_labels_12': case['source_labels_12'],
                'target_labels_12': case['target_labels_12'], 'group': 'edited'}
        for role in ('source', 'target'):
            reference = case[role + '_reference']
            key = [role + '_real', case['subject'], cid, 0]
            if reference['status'] == 'complete':
                add({**meta, 'method': role + '_real', 'seed': 0}, reference['image'], 'real_' + role)
            else:
                availability.append({'key': key, 'status': 'reference_failed', 'role': 'real_' + role,
                                     'reference': reference})
    for slot in plan:
        key = row_key(slot)
        row = records.get(key)
        if row is None or row['status'] != 'success':
            availability.append({'key': list(key), 'status': 'missing_output' if row is None else row['status'],
                                 'role': 'generated', 'generation_record': row})
            continue
        path = Path(row['output_path']).resolve()
        require(path.is_relative_to(run_root / 'attempts'), 'Output outside this new generation run')
        case = cases[slot['case_id']]
        meta = {**slot, 'source_labels_12': case['source_labels_12'],
                'target_labels_12': case['target_labels_12'], 'group': 'edited'}
        add(meta, {'path': str(path), 'sha256': row['output_sha256']}, 'generated')
    output.mkdir(parents=True)
    payload = {
        'kind': 'supplementary_ablation_evaluation_inputs', 'data_role': ROLE,
        'independent_final_test': False, 'previously_exposed_subjects': True, 'test_used': True,
        'generation_result': result_ref, 'reference_manifest': ref(REFERENCES),
        'generation_freeze': ref(freeze_path), 'generation_run_root': str(run_root),
        'generation_status': generation['status'],
        'same_protocol_complete': generation.get('same_protocol_complete') is True,
        'all_full_arm_parity_passed': generation.get('all_full_arm_parity_passed') is True,
        'measurement_policy': ref(V2 / 'policy.json'), 'measurement_policy_sha256': POLICY_SHA,
        'algorithm_bindings': algorithms, 'analysis_spec': analysis_spec(),
        'analysis_plan': ref(analysis_plan), 'subgroup_plan': frozen_analysis['subgroup_plan'],
        'real_reference_calibration': frozen_analysis['real_reference_calibration'],
        'planned': plan, 'normalized_generation_records': list(records.values()),
        'case_metadata': cases, 'items': items, 'availability': availability,
        'generated_slots': 288, 'images_expected_if_all_valid': 336,
        'aggregation': 'seed -> case -> equal-weight subject; paired common case/seed keys',
        'failure_policy': 'retain all slots; failed detection/reference yields NA, never imputed zero',
        'no_old_generated_measurements_used': True,
        'source_bindings': {p.name: ref(p) for p in (Path(__file__), HERE / 'ablation_evaluation_contract.py')},
    }
    save(output / 'items.json', payload)
    save(output / 'items_seal.json', {'sha256': sha(output / 'items.json')})
    require(sha(generation_result) == result_ref['sha256'], 'Generation manifest changed during preparation')
    return output


def job_control(job):
    job = Path(job).resolve()
    require(job.is_relative_to(HERE) and job != HERE, 'Job outside the new evaluation directory')
    document = read(job / 'items.json')
    require(sha(job / 'items.json') == read(job / 'items_seal.json')['sha256'], 'Prepared job changed')
    require(document['data_role'] == ROLE and document['analysis_spec'] == analysis_spec(),
            'Wrong role/analysis specification')
    require(document['measurement_policy_sha256'] == POLICY_SHA, 'Measurement policy changed')
    for reference in (document['generation_result'], document['generation_freeze'],
                      document['reference_manifest'], document['measurement_policy'], document['analysis_plan']):
        verified(reference)
    for reference in document['algorithm_bindings'].values():
        verified(reference)
    for reference in document['source_bindings'].values():
        verified(reference)
    require(len(document['planned']) == 288, 'Wrong evaluation denominator')
    return job, document


def check_tool_assets(kind):
    policy = read(V2 / 'policy.json')
    require(sha(V2 / 'policy.json') == POLICY_SHA, 'Policy changed')
    root = EXP / 'tools' / kind
    seal_ref = 'openface' if kind == 'openface' else 'identity_primary'
    require(sha(root / 'tool_seal.json') == policy[seal_ref]['tool_seal_sha256'], 'Tool seal changed')
    seal = read(root / 'tool_seal.json')
    assets = [seal['binary'], seal['launcher'], *seal['model_files']] if kind == 'openface' else seal['source_files'] + seal['weights']
    for asset in assets:
        require(sha(root / asset['path']) == asset['sha256'], 'Tool asset changed: ' + asset['path'])
    return root, seal


def measure_openface(job):
    job, document = job_control(job)
    core = load_legacy('openface')
    check_tool_assets('openface')
    output = job / 'openface'
    require(not output.exists(), 'Never overwrite an OpenFace measurement run')
    for item in document['items']:
        verified(item)
    output.mkdir()
    unique = {}
    for item in document['items']:
        unique.setdefault(item['sha256'], {**item, 'sample_id': 'sha_' + item['sha256']})
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    physical = []
    with (output / 'measurements.jsonl').open('x', encoding='utf-8', buffering=1) as journal:
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = {workers.submit(core.measure, item, output): item for item in unique.values()}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {**item, 'parse_status': 'wrapper_error', 'vector': None,
                              'selected_au17': None, 'error': str(error), 'error_type': type(error).__name__}
                physical.append(result)
                journal.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                print(json.dumps({'evaluator': 'openface', 'completed': len(physical), 'planned': len(unique)}), flush=True)
    by_sha = {row['sha256']: row for row in physical}
    records = [{**by_sha[item['sha256']], **item,
                'measurement_source_path': by_sha[item['sha256']]['path']} for item in document['items']]
    save(output / 'results.json', {'status': 'complete', 'data_role': ROLE, 'policy_sha256': POLICY_SHA,
         'input_sha256': sha(job / 'items.json'), 'records': records, 'unique_measurements': physical,
         'algorithm': document['algorithm_bindings']['openface']})


def derive_facenet(job, input_sha, count, source_count):
    """Original model/geometry/normalization code; schema and job-path adaptation only."""
    path, expected = PINS['facenet']
    require(sha(path) == expected, 'Frozen FaceNet algorithm changed')
    text = path.read_text(encoding='utf-8')
    replacements = {
        'from run_facenet_v2 import HERE, ROOT, POLICY_SHA, sha, save':
            f"from ablation_evaluation_contract import sha, save\nHERE = Path({str(V2)!r})\n"
            f"ROOT = Path({str(EXP)!r})\nPOLICY_SHA = {POLICY_SHA!r}",
        "root = HERE / 'single_directions'": f'root = Path({str(job)!r})',
        "INPUT_SHA = 'cd7365706c5b04540f6d97ff90f4f8d3ade5513b08f57179479395e702f18eb5'": f"INPUT_SHA = '{input_sha}'",
        "not document['test_used']": f"document['data_role'] == {ROLE!r}",
        "(row['case_id'], row['method'])": "(row['case_id'], row['method'], row['seed'])",
        "source_sha = sources[row['case_id']]": "source_sha = sources.get(row['case_id'])",
        "HERE / 'shared_cache/matplotlib'": "root / 'cache/matplotlib'",
        "HERE / 'shared_cache/xdg'": "root / 'cache/xdg'",
    }
    for old, new in replacements.items():
        require(text.count(old) == 1, 'Ambiguous FaceNet metadata adaptation: ' + old)
        text = text.replace(old, new)

    class MetadataAdapter(ast.NodeTransformer):
        def visit_Dict(self, node):
            self.generic_visit(node)
            for index, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == 'test_used':
                    node.values[index] = ast.Constant(value=True)
            return node

        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 57:
                node.value = count
            elif type(node.value) is int and node.value == 19:
                node.value = source_count
            elif node.value == 'dev':
                node.value = ROLE
            return node

    return ast.unparse(ast.fix_missing_locations(MetadataAdapter().visit(ast.parse(text)))) + '\n'


def measure_facenet(job):
    job, document = job_control(job)
    require(sys.flags.optimize == 0, 'FaceNet frozen adapter uses assertions; do not run Python -O')
    require(not (job / 'facenet').exists(), 'Never overwrite a FaceNet measurement run')
    for item in document['items']:
        verified(item)
    check_tool_assets('facenet')
    text = derive_facenet(job, sha(job / 'items.json'), len(document['items']),
                          sum(row['method'] == 'source_real' for row in document['items']))
    adapter = job / 'facenet_adapter.py'
    with adapter.open('x', encoding='utf-8') as stream:
        stream.write(text)
    namespace = {'__name__': 'supplementary_frozen_facenet', '__file__': str(adapter)}
    exec(compile(text, str(adapter), 'exec'), namespace)
    namespace['main']()
    save(job / 'facenet/binding.json', {
        'data_role': ROLE, 'policy_sha256': POLICY_SHA, 'input_sha256': sha(job / 'items.json'),
        'algorithm': document['algorithm_bindings']['facenet'], 'adapter': ref(adapter),
        'adaptations': 'input path/SHA/count, seed-key uniqueness, supplementary role, missing-source NA, cache paths only',
        'device': 'cpu', 'model_geometry_and_normalization_changed': False,
    })


def logical_key(row):
    return row['method'], row['subject'], row['case_id'], row['seed']


def validate_measurements(document, measured):
    available = {logical_key(row): row for row in document['items']}
    require(len(available) == len(document['items']), 'Duplicate prepared input')
    result = {}
    for row in measured['records']:
        key = logical_key(row)
        require(key in available and key not in result, 'Unexpected/duplicate measured image')
        require(row['sha256'] == available[key]['sha256'], 'Measured image SHA differs from sealed input')
        result[key] = row
    require(set(result) == set(available), 'Evaluator must retain all available images including failed detections')
    return result


def merge_metrics(document, au_document, face_document):
    au, faces = validate_measurements(document, au_document), validate_measurements(document, face_document)
    au_metrics, extra_errors = metric_functions()
    available = {tuple(row['key']): row for row in document['availability']}
    generated = {row_key(row): row for row in document['normalized_generation_records']}
    rows = []
    for slot in document['planned']:
        key = row_key(slot)
        case = document['case_metadata'][slot['case_id']]
        original = generated.get(key)
        a, f = au.get(key), faces.get(key)
        source = au.get(('source_real', slot['subject'], slot['case_id'], 0))
        target = au.get(('target_real', slot['subject'], slot['case_id'], 0))
        values = au_metrics(a.get('vector') if a else None, source.get('vector') if source else None,
                            case['source_labels_12'], case['target_labels_12'])
        values['identity_cosine'] = f.get('identity_cosine') if f else None
        extra, values['extra5_proxy_mae_to_real_target'] = extra_errors(
            a.get('selected_au17') if a else None, target.get('selected_au17') if target else None,
            source.get('selected_au17') if source else None)
        for number in EXTRA:
            values[f'extra_AU{number}_error_to_real_target'] = extra[str(number)]['absolute_error_to_target']
        image_available = available.get(key, {}).get('status') == 'available'
        rows.append({**slot, 'status': 'complete' if image_available else 'unavailable',
                     'generation_status': original['status'] if original else 'missing_output',
                     'output_sha256': original.get('output_sha256') if original else None,
                     'source_au12': case['source_labels_12'], 'target_au12': case['target_labels_12'],
                     'metrics': {name: values.get(name) for name in METRICS},
                     'AU_diagnostics': values, 'extra5': extra,
                     'AU_status': a.get('parse_status') if a else 'unavailable',
                     'FaceNet_status': f.get('status') if f else 'unavailable',
                     'availability': available.get(key)})
    return rows


def factorial_exploration(rows, stats):
    """All four arms must be valid at the same case/seed; no confirmatory claim."""
    by_key = {logical_key(row): row for row in rows}
    units = sorted({key[1:] for key in by_key})
    result = {}
    for metric in METRICS:
        values = {}
        missing = []
        for unit in units:
            parts = {arm: stats.finite_value(by_key.get((arm,) + unit), metric)[0] for arm in ARMS}
            if any(value is None for value in parts.values()):
                missing.append(list(unit))
                continue
            # Y11 - Y01 - Y10 + Y00 on the raw metric scale.
            values[unit] = parts['full'] - parts['no_projection'] - parts['no_smoothing'] + parts['both_off']
        aggregate = stats.subject_aggregate(values)
        result[metric] = {**aggregate, 'planned_case_seed_units': len(units), 'missing_units': missing,
                          'difference_definition': 'full - no_projection - no_smoothing + both_off',
                          'inference': stats.subject_inference(aggregate['subject_means'].values(), 20261023),
                          'exploratory': True, 'multiple_testing_adjusted': False}
    return result


def write_csv(path, rows, fields):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def subgroup_summary(rows, plan, stats):
    """Pure descriptive summaries for prespecified manual-label groups."""
    assignments = plan['case_assignments']
    output = []
    for partition, groups in plan['partitions'].items():
        for group in groups:
            case_ids = {cid for cid, case in assignments.items() if case[partition] == group}
            subjects = {assignments[cid]['subject'] for cid in case_ids}
            for arm in ARMS:
                selected = [row for row in rows if row['method'] == arm and row['case_id'] in case_ids]
                for metric in plan['metrics']:
                    valid = {}
                    for row in selected:
                        value, _ = stats.finite_value(row, metric)
                        if value is not None:
                            valid[(row['subject'], row['case_id'], row['seed'])] = value
                    aggregate = stats.subject_aggregate(valid)
                    output.append({'partition': partition, 'subgroup': group, 'method': arm, 'metric': metric,
                                   'subject_equal_mean': aggregate['mean'], 'planned_subjects': len(subjects),
                                   'planned_cases': len(case_ids), 'planned_seed_outputs': len(selected),
                                   'valid_subjects': aggregate['n_subjects'], 'valid_cases': aggregate['n_cases'],
                                   'valid_seed_outputs': aggregate['n_seed_outputs'], 'descriptive_only': True})
    return output


def real_reference_calibration(document, au_document, stats):
    """Deduplicate real reference images, retain failures, never rate generated images."""
    au = validate_measurements(document, au_document)
    grouped = {}
    unavailable_without_sha = []
    for cid, case in document['case_metadata'].items():
        for role in ('source', 'target'):
            reference = case[role + '_reference']
            if reference['status'] != 'complete':
                unavailable_without_sha.append({'case_id': cid, 'subject': case['subject'], 'role': role,
                                                'reason': 'reference_failed', 'reference': reference})
                continue
            image = reference['image']
            labels = case[role + '_labels_12']
            logical = {'case_id': cid, 'subject': case['subject'], 'role': role,
                       'manual_labels_12': labels, 'path': image['path']}
            key = (role + '_real', case['subject'], cid, 0)
            measured = au.get(key)
            grouped.setdefault(image['sha256'], []).append((logical, measured))
    physical = []
    for digest, rows in sorted(grouped.items()):
        subjects = {meta['subject'] for meta, _ in rows}
        labels = {tuple(meta['manual_labels_12']) for meta, _ in rows}
        conflict = len(subjects) != 1 or len(labels) != 1
        candidates = [measured for _, measured in rows if measured is not None]
        # The original OpenFace wrapper SHA-deduplicates before measurement.
        vectors = {json.dumps(row.get('vector'), allow_nan=False) for row in candidates}
        require(len(vectors) <= 1, 'Identical real-image SHA has inconsistent AU measurements')
        measured = candidates[0] if candidates else None
        physical.append({'sha256': digest, 'subject': next(iter(subjects)) if len(subjects) == 1 else None,
                         'manual_labels_12': list(next(iter(labels))) if not conflict else None,
                         'measured_12': None if conflict or measured is None else measured.get('vector'),
                         'status': 'reference_label_conflict' if conflict else
                                   measured.get('parse_status', 'unknown') if measured else 'reference_unavailable',
                         'logical_references': [meta for meta, _ in rows]})
    summaries = []
    for index, au_number in enumerate(AUS):
        for level in document['real_reference_calibration']['intensity_strata']:
            eligible = [row for row in physical if row['manual_labels_12'] is not None and
                        (level == 'all' or row['manual_labels_12'][index] == level)]
            valid = [row for row in eligible if row['measured_12'] is not None]
            measures = {}
            for name in ('absolute_error', 'signed_error', 'manual_intensity', 'estimated_intensity'):
                values = {}
                for row in valid:
                    manual, estimate = row['manual_labels_12'][index], row['measured_12'][index]
                    value = {'absolute_error': abs(estimate - manual), 'signed_error': estimate - manual,
                             'manual_intensity': manual, 'estimated_intensity': estimate}[name]
                    values[(row['subject'], row['sha256'], 0)] = value
                measures[name] = stats.subject_aggregate(values)['mean']
            summaries.append({'AU': au_number, 'manual_intensity_stratum': level,
                              'planned_unique_images': len(eligible),
                              'planned_subjects': len({r['subject'] for r in eligible}),
                              'valid_unique_images': len(valid), 'valid_subjects': len({r['subject'] for r in valid}),
                              'subject_equal_MAE': measures['absolute_error'],
                              'subject_equal_bias': measures['signed_error'],
                              'subject_equal_manual_intensity': measures['manual_intensity'],
                              'subject_equal_estimated_intensity': measures['estimated_intensity'],
                              'descriptive_only': True})
    return {'plan': document['real_reference_calibration'], 'unique_reference_images': len(physical),
            'physical_records': physical, 'unavailable_without_sha': unavailable_without_sha,
            'reference_label_conflicts': [row for row in physical if row['status'] == 'reference_label_conflict'],
            'summary': summaries}


def export(job):
    job, document = job_control(job)
    au = read(job / 'openface/results.json')
    face = read(job / 'facenet/results.json')
    face_binding = read(job / 'facenet/binding.json')
    input_sha = sha(job / 'items.json')
    require(au['status'] == 'complete' and au['input_sha256'] == input_sha and
            au['policy_sha256'] == POLICY_SHA, 'OpenFace completion/binding mismatch')
    require(face['summary']['status'] == 'complete' and face['policy_sha256'] == POLICY_SHA and
            face_binding['input_sha256'] == input_sha and face_binding['data_role'] == ROLE,
            'FaceNet completion/binding mismatch')
    for item in document['items']:
        verified(item)
    rows = merge_metrics(document, au, face)
    stats = load_legacy('statistics')
    payload = {'kind': 'paired_subject_metric_input', 'analysis_spec': analysis_spec(),
               'planned': document['planned'], 'records': rows,
               'metadata': {'data_role': ROLE, 'independent_final_test': False,
                            'previously_exposed_subjects': True, 'generation_result': document['generation_result'],
                            'items_sha256': input_sha, 'evaluation_policy_sha256': POLICY_SHA,
                            'generation_status': document['generation_status'],
                            'same_protocol_complete': document['same_protocol_complete'],
                            'all_full_arm_parity_passed': document['all_full_arm_parity_passed'],
                            'primary_metric': PRIMARY, 'primary_family_size': 3,
                            'no_old_generated_measurements_used': True}}
    statistics = stats.analyze(payload)
    interaction = factorial_exploration(rows, stats)
    subgroups = subgroup_summary(rows, document['subgroup_plan'], stats)
    calibration = real_reference_calibration(document, au, stats)
    output = job / 'statistics'
    require(not output.exists(), 'Never overwrite statistics')
    output.mkdir()
    save(output / 'all_case_seed_metrics.json', {'records': rows, 'planned_outputs': 288})
    save(output / 'paired_subject_input.json', payload)
    save(output / 'paired_subject_results.json', statistics)
    save(output / 'factorial_interaction_exploratory.json', interaction)
    save(output / 'subgroup_plan.json', document['subgroup_plan'])
    save(output / 'real_reference_calibration.json', calibration)
    write_csv(output / 'real_reference_calibration.csv', calibration['summary'],
              ['AU', 'manual_intensity_stratum', 'planned_unique_images', 'planned_subjects',
               'valid_unique_images', 'valid_subjects', 'subject_equal_MAE', 'subject_equal_bias',
               'subject_equal_manual_intensity', 'subject_equal_estimated_intensity', 'descriptive_only'])
    write_csv(output / 'subgroup_summary.csv', subgroups,
              ['partition', 'subgroup', 'method', 'metric', 'subject_equal_mean', 'planned_subjects',
               'planned_cases', 'planned_seed_outputs', 'valid_subjects', 'valid_cases', 'valid_seed_outputs',
               'descriptive_only'])
    write_csv(output / 'all_case_seed_metrics.csv',
              [{**row, **row['metrics']} for row in rows],
              ['method', 'subject', 'case_id', 'seed', 'status', 'generation_status', 'output_sha256',
               'AU_status', 'FaceNet_status', *METRICS])
    summary = []
    for arm in ARMS:
        summary.append({'method': arm, **{metric: statistics['raw_method_statistics'][arm][metric]['mean']
                                         for metric in METRICS}})
    write_csv(output / 'all_method_summary.csv', summary, ['method', *METRICS])
    write_csv(output / 'coverage.csv', statistics['coverage_table'],
              ['method', 'metric', 'planned_seed_outputs', 'observed_output_rows', 'valid_metric_outputs',
               'planned_cases', 'planned_subjects', 'valid_cases', 'valid_subjects', 'unavailable_by_reason'])
    write_csv(output / 'primary_DU_comparisons.csv', statistics['comparisons'],
              ['id', 'method', 'reference', 'metric', 'paired_method_mean', 'paired_reference_mean',
               'mean_difference', 'difference_ci95', 'two_sided_sign_flip_p', 'holm_adjusted_p',
               'holm_family_size', 'n_subjects', 'n_cases', 'n_seed_pairs', 'sign_patterns'])
    save(output / 'receipt.json', {'status': 'complete', 'measurement_sources':
         {'openface': ref(job / 'openface/results.json'), 'facenet': ref(job / 'facenet/results.json')},
         'statistics_algorithm': ref(PINS['statistics'][0]), 'input': ref(output / 'paired_subject_input.json'),
         'output_count': len(rows), 'all_generation_failures_and_metric_NAs_retained': True})
    return output


def inventory():
    """Stat only for large model assets; never hydrate or import ML runtimes."""
    paths = [path for path, _ in PINS.values()] + [V2 / 'policy.json', REFERENCES,
             V2 / 'run_facenet_v2.py', V2.parent / 'real_dev_calibration/calibration_metrics.py',
             V2 / 'real_dev_openface/summary.json', V2 / 'facenet_real12/summary.json',
             V2 / 'INPUT_COORDINATE_NOTE.md']
    for case in reference_cases(read(REFERENCES)).values():
        for role in ('source', 'target'):
            reference = case[role + '_reference']
            if reference['status'] == 'complete':
                paths.append(Path(reference['image']['path']))
    for name in ('openface', 'facenet'):
        root = EXP / 'tools' / name
        paths.append(root / 'tool_seal.json')
        seal = read(root / 'tool_seal.json')
        assets = [seal['binary'], seal['launcher'], *seal['model_files']] if name == 'openface' else seal['source_files'] + seal['weights']
        paths.extend(root / asset['path'] for asset in assets)
        if name == 'facenet':
            # MTCNN resolves runtime/data symlinks, not only the sealed targets.
            paths.extend(root / 'runtime/facenet_pytorch/data' / file for file in ('pnet.pt', 'rnet.pt', 'onet.pt'))
    rows = []
    for path in sorted(set(paths)):
        try:
            stat = path.stat()
            rows.append({'path': str(path), 'exists': True, 'bytes': stat.st_size,
                         'dataless': bool(getattr(stat, 'st_flags', 0) & 0x40000000),
                         'allocated_bytes': getattr(stat, 'st_blocks', 0) * 512})
        except OSError as error:
            rows.append({'path': str(path), 'exists': False, 'error': str(error)})
    return {'model_imported': False, 'large_file_contents_read': False, 'files': rows,
            'missing_or_dataless': [r for r in rows if not r['exists'] or r.get('dataless')]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('inventory', 'freeze', 'prepare', 'openface', 'facenet', 'export'))
    parser.add_argument('--generation-result', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--analysis-plan', type=Path, default=HERE / 'analysis_plan_v1.json')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.phase == 'inventory':
        result = inventory()
        if args.output:
            path = args.output.resolve()
            require(path.is_relative_to(HERE), 'Inventory output must be under evaluation/')
            save(path, result)
            print(json.dumps({'output': str(path), 'files': len(result['files']),
                              'missing_or_dataless': len(result['missing_or_dataless']),
                              'bytes_to_hydrate': sum(r.get('bytes', 0) for r in result['missing_or_dataless']),
                              'models_loaded': False}, ensure_ascii=False))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.phase == 'freeze':
        require(args.output is not None, '--output required')
        print(json.dumps(freeze_analysis(args.output), ensure_ascii=False))
    elif args.phase == 'prepare':
        require(args.generation_result is not None and args.output is not None, '--generation-result and --output required')
        print(prepare(args.generation_result, args.output, args.analysis_plan))
    elif args.phase == 'export':
        require(args.job is not None, '--job required')
        print(export(args.job))
    else:
        require(args.execute and args.job is not None, 'Model execution requires --execute --job')
        {'openface': measure_openface, 'facenet': measure_facenet}[args.phase](args.job)


if __name__ == '__main__':
    main()
