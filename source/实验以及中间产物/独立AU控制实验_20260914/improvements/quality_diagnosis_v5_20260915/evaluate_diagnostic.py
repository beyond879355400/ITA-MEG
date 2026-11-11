#!/usr/bin/env python3
"""Dev8 diagnostic adapter; fixed OpenFace v2 + independent FaceNet, CPU only.

Preparation and --help import only the standard library. Real evaluators start
only in explicit openface/facenet/all phases, after a terminal generation run.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
POLICY_DIR = EXP / 'development_evaluation/evaluation_policy_v2'
POLICY_SHA = 'd29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c'
PINNED = {
    'openface_v2.py': '5446237ff2fba672bd06ec62820bf671d9deb4c776d83a28d18060a581b9f908',
    'run_facenet_v2.py': 'd8bd902fb2a360eae00de39d4fe4ea2e01ce41a819b16f74fe1881efc3fd2754',
    'run_facenet_single_directions.py': '5d8e1916bb5bcc9afc307980438b5f2f3fb0610f5dde12141f90e85efa7ebc2d',
}
AU_CORE = POLICY_DIR.parent / 'v3_global_raw_dev64/evaluate.py'
AU_CORE_SHA = '5a8b8944375f5b7ef1ebca3bbfe46cc2513beb2a110f27d5842d10f286d1e3dd'
AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
EXTRA = [7, 10, 14, 23, 45]
SUBJECTS = {'SN004', 'SN012', 'SN017', 'SN018'}
GROUPS = {'no_edit', 'lowest_available'}
METHODS = [f'{method}_img2img_s{strength}' for strength in ('035', '055', '075')
           for method in ('baseline', 'bidirectional')]
REFERENCES = ['source_real', 'target_real', 'baseline_text2img',
              'bidirectional_text2img', 'vae_reconstruction']
METRICS = ['target_mae_changed', 'delta_mae_changed', 'unchanged_source_relative_drift',
           'target_mae_all12', 'delta_mae_all12', 'direction_correct_fraction',
           'identity_cosine', 'buffalo_identity_cosine', 'extra5_proxy_mae_to_real_target'] + [
               f'extra_AU{au}_error_to_real_target' for au in EXTRA]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def mean(values):
    return sum(values) / len(values) if values else None


def scoped_path(path, root=EXP):
    path = Path(path).resolve()
    require(path.is_relative_to(root.resolve()), 'Path outside experiment scope')
    require(not any(part == 'test' or part.startswith('final_test') for part in path.parts),
            'Test path forbidden before file access')
    return path


def checked_image(reference):
    require(isinstance(reference, dict), 'Image reference must include path and SHA')
    path = scoped_path(reference['path'])
    digest = reference['sha256']
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Invalid image SHA')
    stat = path.stat()
    require(not getattr(stat, 'st_flags', 0) & 0x40000000 and stat.st_blocks > 0,
            'Image must be a locally materialized file')
    require(sha(path) == digest, 'Image SHA mismatch: ' + str(path))
    return {'path': str(path), 'sha256': digest}


def check_policy():
    require(sha(POLICY_DIR / 'policy.json') == POLICY_SHA, 'Frozen policy changed')
    for name, digest in PINNED.items():
        require(sha(POLICY_DIR / name) == digest, 'Frozen evaluator changed: ' + name)
    require(sha(AU_CORE) == AU_CORE_SHA, 'Frozen AU metric implementation changed')
    return read(POLICY_DIR / 'policy.json')


def au_metrics(pred, source_pred, source_gt, target_gt):
    """Execute only the pinned original pure metric function, not its module."""
    require(sha(AU_CORE) == AU_CORE_SHA, 'Frozen AU metric implementation changed')
    tree = ast.parse(AU_CORE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'au_metrics')
    scope = {'mean': mean, 'AUS': AUS}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(AU_CORE), 'exec'), scope)
    return scope['au_metrics'](pred, source_pred, source_gt, target_gt)


def labels(values):
    require(isinstance(values, list) and len(values) == 12, 'Expected 12 AU labels')
    require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 5 for v in values),
            'AU labels must be finite in [0,5]')
    return values


def reference_lineage(cases):
    """Bind all old references/labels to the already frozen dev19 image set."""
    policy = check_policy()
    path = POLICY_DIR / 'dev19_items.json'
    expected_sha = policy['inputs']['dev19_items.json']['sha256']
    require(sha(path) == expected_sha, 'Frozen dev19 reference manifest changed')
    document = read(path)
    require(document['test_used'] is False and all(row['split'] == 'dev' for row in document['items']),
            'Reference manifest is not strictly dev')
    mapping = {(row['case_id'], row['method']): row for row in document['items']}
    roles = {'source_real': 'source_real', 'target_real': 'target_real',
             'baseline_text2img': 'baseline', 'bidirectional_text2img': 'current_model'}
    bindings = []
    for case in cases:
        for role, old_method in roles.items():
            key = case['case_id'], old_method
            require(key in mapping, 'Case not in frozen dev19 reference set')
            old = mapping[key]
            require(case['subject'] == old['subject'] and case['split'] == old['split'] == 'dev',
                    'Reference subject/split changed')
            require(case[role]['sha256'] == old['sha256'], 'Reference image differs from frozen dev19: ' + role)
            require(case['source_aus_0_5'] == old['source_labels_12'] and
                    case['target_aus_0_5'] == old['target_labels_12'], 'Reference labels differ from frozen dev19')
            bindings.append({'case_id': case['case_id'], 'role': role, 'old_method': old_method,
                             'old_original_method': old['original_method'], 'sha256': old['sha256']})
    return {'manifest': str(path), 'sha256': expected_sha, 'bindings': bindings,
            'same_sha_different_path_allowed': True}


def build_items(case_document, result_document, run):
    """Closed dev8 x six methods; failures retained without fake image data."""
    require(case_document.get('kind') == 'v5_development_diagnostic_cases', 'Wrong case manifest kind')
    require(result_document.get('kind') == 'v5_source_initialization_diagnostic', 'Wrong generation result kind')
    require(case_document.get('test_data_used') is False and result_document.get('test_data_used') is False,
            'Explicit dev-only declarations required')
    require(result_document.get('status') in ('complete', 'completed_with_failures'), 'Generation is not terminal')
    cases = case_document['cases']
    require(len(cases) == case_document.get('case_count') == 8, 'Expected exactly eight cases')
    require(len({case['case_id'] for case in cases}) == 8, 'Duplicate case ID')
    require({(case['subject'], case['group']) for case in cases} == set(itertools.product(SUBJECTS, GROUPS)),
            'Expected four named dev subjects x two groups')
    items, by_case = [], {}
    for case in cases:
        require(case.get('split') == 'dev', 'Non-dev case forbidden')
        require(re.fullmatch('[A-Za-z0-9_.-]+', case['case_id']), 'Unsafe case ID')
        source, target = labels(case['source_aus_0_5']), labels(case['target_aus_0_5'])
        require((source == target) == (case['group'] == 'no_edit'), 'Group and AU labels disagree')
        base = {key: case[key] for key in ('case_id', 'subject', 'split', 'group')}
        base.update(source_labels_12=source, target_labels_12=target)
        by_case[case['case_id']] = base
        for method in REFERENCES:
            ref = checked_image(case[method])
            if method == 'vae_reconstruction':
                require(Path(ref['path']).is_relative_to(run), 'VAE reconstruction must belong to this run')
            items.append({**base, **ref, 'method': method, 'sample_id': case['case_id'] + '__' + method,
                          'generation_status': 'reference', 'image_available': True,
                          'role': 'real_reference' if method.endswith('_real') else 'reconstruction_reference'
                          if method == 'vae_reconstruction' else 'frozen_generation_reference'})
    records = result_document['records']
    require(len(records) == 48, 'Expected all 48 planned generation records, including failures')
    expected = set(itertools.product(by_case, METHODS))
    require({(row['case_id'], row['method']) for row in records} == expected, 'Missing or unexpected generated case/method')
    for row in records:
        base = by_case[row['case_id']]
        require(all(row.get(key) == base[key] for key in ('subject', 'split', 'group')), 'Generated case metadata differs')
        require(row.get('seed') == 101, 'Only frozen seed 101 is allowed')
        require(row.get('source_aus_0_5') == base['source_labels_12'] and
                row.get('target_aus_0_5') == base['target_labels_12'], 'Generated AU labels differ')
        status = row.get('status')
        require(status in ('complete', 'failed', 'error'), 'A generation record is not terminal')
        available = status == 'complete'
        if available:
            ref = checked_image({'path': row['output'], 'sha256': row['sha256']})
            require(Path(ref['path']).is_relative_to(run), 'New output is outside this run')
        else:
            # Do not open or evaluate a partial artifact from a failed generation.
            ref = {'path': row.get('output'), 'sha256': row.get('sha256')}
        items.append({**base, **ref, 'method': row['method'], 'sample_id': row['case_id'] + '__' + row['method'],
                      'seed': 101, 'generation_status': status, 'generation_record': row,
                      'role': 'new_generation', 'image_available': available})
    failed = sum(item['role'] == 'new_generation' and not item['image_available'] for item in items)
    require((result_document['status'] == 'complete') == (failed == 0), 'Run status disagrees with generation records')
    require(len({item['sample_id'] for item in items}) == len(items) == 88, 'Duplicate logical record')
    return sorted(items, key=lambda item: (item['case_id'], item['method']))


def validate(run):
    check_policy()
    run = scoped_path(run, HERE)
    cases_path, result_path = run / 'cases.json', run / 'result.json'
    before = {str(path): sha(path) for path in (cases_path, result_path)}
    cases, result = read(cases_path), read(result_path)
    lineage = reference_lineage(cases['cases'])
    items = build_items(cases, result, run)
    require(all(sha(path) == digest for path, digest in before.items()), 'Generation manifests changed during validation')
    return {'status': 'prepared', 'run': str(run), 'generation_status': result['status'],
            'policy_sha256': POLICY_SHA, 'test_used': False, 'logical_count': 88,
            'reference_lineage': lineage,
            'measurement_count': sum(item['image_available'] for item in items),
            'generation_manifest_hashes': before, 'items': items}


def prepare(run, output):
    document = validate(run)
    output = scoped_path(output, HERE)
    require(output != Path(document['run']) and not output.is_relative_to(Path(document['run'])),
            'Evaluation output must be separate from generation run')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'manifest.json', document)
    save(output / 'items.json', {'test_used': False, 'items': [item for item in document['items'] if item['image_available']]})
    save(output / 'input_seal.json', {'manifest_sha256': sha(output / 'manifest.json'),
                                    'items_sha256': sha(output / 'items.json'), 'evaluator_sha256': sha(__file__),
                                    'policy_sha256': POLICY_SHA})
    return output


def checked(output):
    check_policy()
    output = scoped_path(output, HERE)
    seal = read(output / 'input_seal.json')
    require(seal['policy_sha256'] == POLICY_SHA and seal['evaluator_sha256'] == sha(__file__), 'Prepared evaluator changed')
    require(sha(output / 'manifest.json') == seal['manifest_sha256'] and
            sha(output / 'items.json') == seal['items_sha256'], 'Prepared input manifest changed')
    document = read(output / 'manifest.json')
    require(document == validate(document['run']), 'Generation or reference input changed after preparation')
    return output, document


def load_openface():
    spec = importlib.util.spec_from_file_location('v5_fixed_openface', POLICY_DIR / 'openface_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def openface(output):
    output, document = checked(output)
    policy, helper = check_policy(), load_openface()
    seal = read(helper.TOOLS / 'tool_seal.json')
    require(sha(helper.TOOLS / 'tool_seal.json') == policy['openface']['tool_seal_sha256'], 'OpenFace seal changed')
    for asset in [seal['binary'], seal['launcher'], *seal['model_files']]:
        require(sha(helper.TOOLS / asset['path']) == asset['sha256'], 'OpenFace asset changed')
    for folder in ('real_dev_openface', 'facenet_real12'):
        prior = read(POLICY_DIR / folder / 'summary.json')
        require(prior['status'] == 'complete' and prior['policy_sha256'] == POLICY_SHA, 'Frozen calibration incomplete')
    folder = output / 'openface'
    folder.mkdir(exist_ok=False)
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    items = [item for item in document['items'] if item['image_available']]
    unique = {}
    for item in items:
        unique.setdefault(item['sha256'], item)
    save(folder / 'protocol.json', {'policy_sha256': POLICY_SHA, 'manifest_sha256': sha(output / 'manifest.json'),
                                  'helper_sha256': PINNED['openface_v2.py'], 'test_used': False,
                                  'logical_records': len(items), 'unique_images': len(unique), 'workers': 2,
                                  'reuse': 'same SHA measured once; all logical records retained'})
    measurements = {}
    with (folder / 'per_image.jsonl').open('x', encoding='utf-8', buffering=1) as stream, ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(helper.measure, item, folder) for item in unique.values()]
        for job in as_completed(jobs):
            row = job.result()
            measurements[row['sha256']] = row
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            print(json.dumps({'phase': 'openface', 'done': len(measurements), 'planned': len(unique)}), flush=True)
    records = [{**measurements[item['sha256']], **item,
                'measurement_source_path': measurements[item['sha256']]['path']} for item in items]
    checked(output)
    save(folder / 'results.json', {'status': 'complete', 'policy_sha256': POLICY_SHA, 'test_used': False,
                                 'records': records, 'unique_measurements': list(measurements.values())})


def derived_facenet_source(output, count):
    """Only input binding/count/cache substitutions; entire model loop unchanged."""
    require(type(count) is int and 40 <= count <= 88, 'Invalid diagnostic measurement count')
    core = POLICY_DIR / 'run_facenet_single_directions.py'
    require(sha(core) == PINNED[core.name], 'Pinned FaceNet core changed')
    source = core.read_text()
    substitutions = {
        "INPUT_SHA = 'cd7365706c5b04540f6d97ff90f4f8d3ade5513b08f57179479395e702f18eb5'":
            'INPUT_SHA = ' + repr(sha(output / 'items.json')),
        "root = HERE / 'single_directions'": 'root = Path(' + repr(str(output)) + ')',
        'len(items) == 57': f'len(items) == {count}',
        "len({(row['case_id'], row['method']) for row in items}) == 57":
            "len({(row['case_id'], row['method']) for row in items}) == " + str(count),
        "'input_logical_count': 57": f"'input_logical_count': {count}",
        "'logical_records': 57": f"'logical_records': {count}",
        'len(sources) == 19': 'len(sources) == 8',
        "HERE / 'shared_cache/matplotlib'": "out / 'cache/matplotlib'",
        "HERE / 'shared_cache/xdg'": "out / 'cache/xdg'",
    }
    for old, new in substitutions.items():
        require(source.count(old) == 1, 'Ambiguous FaceNet adapter binding: ' + old)
        source = source.replace(old, new)
    tree = ast.parse(source)
    # If31 of48 generations failed,57 is the actual requested logical count.
    require(count == 57 or not any(isinstance(node, ast.Constant) and type(node.value) is int and node.value == 57
                                  for node in ast.walk(tree)), 'Stale original FaceNet logical-count constant remains')
    return source


def facenet(output):
    output, document = checked(output)
    source = derived_facenet_source(output, document['measurement_count'])
    require(not (output / 'facenet').exists(), 'FaceNet output already exists')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(POLICY_DIR))
    scope = {'__name__': 'v5_pinned_facenet', '__file__': str(Path(__file__).resolve())}
    exec(compile(source, str(POLICY_DIR / 'run_facenet_single_directions.py'), 'exec'), scope)
    scope['main']()
    checked(output)
    with (output / 'facenet/derived_evaluator_snapshot.py').open('x') as stream:
        stream.write(source)
    save(output / 'facenet/adaptation.json', {'policy_sha256': POLICY_SHA,
         'frozen_core_sha256': PINNED['run_facenet_single_directions.py'],
         'derived_source_sha256': sha(output / 'facenet/derived_evaluator_snapshot.py'),
         'changes': 'input directory/hash/count, source case count, cache output; model loop byte-preserved',
         'selection_and_standardization_changed': False, 'test_used': False})


def extras(pred, target, source):
    detail = {}
    for au in EXTRA:
        p = pred.get(str(au)) if pred is not None else None
        t = target.get(str(au)) if target is not None else None
        s = source.get(str(au)) if source is not None else None
        detail[str(au)] = {'output': p, 'real_target_measured': t, 'real_source_measured': s,
            'absolute_error_to_target': abs(p - t) if p is not None and t is not None else None,
            'output_minus_source': p - s if p is not None and s is not None else None,
            'target_minus_source': t - s if t is not None and s is not None else None}
    errors = [entry['absolute_error_to_target'] for entry in detail.values()]
    return detail, mean(errors) if all(value is not None for value in errors) else None


def aggregate(rows, methods):
    """Per-metric complete-case intersection; cases then subject-equal means."""
    mapping = {(row['case_id'], row['method']): row for row in rows}
    result = {}
    for group in ('edited', 'no_edit'):
        cases = sorted({row['case_id'] for row in rows if row['method'] == methods[0]
                        and (row['group'] == 'no_edit') == (group == 'no_edit')})
        stats = {}
        for key in METRICS:
            common = [case for case in cases if all((case, method) in mapping and
                      mapping[(case, method)].get(key) is not None for method in methods)]
            subjects = sorted({mapping[(case, methods[0])]['subject'] for case in common})
            per_subject = {subject: {method: mean([mapping[(case, method)][key] for case in common
                           if mapping[(case, method)]['subject'] == subject]) for method in methods} for subject in subjects}
            stats[key] = {'planned_cases': len(cases), 'common_cases': len(common), 'subject_count': len(subjects),
                          'case_ids': common, 'subjects': subjects, 'per_subject': per_subject,
                          'subject_equal_means': {method: mean([entry[method] for entry in per_subject.values()]) for method in methods}}
        result[group] = stats
    return result


def merge(output):
    output, document = checked(output)
    au, fn = read(output / 'openface/results.json'), read(output / 'facenet/results.json')
    require(au['status'] == fn['summary']['status'] == 'complete', 'Evaluation is not complete')
    require(all(data['policy_sha256'] == POLICY_SHA and data['test_used'] is False for data in (au, fn)), 'Mixed policy')
    am = {(row['case_id'], row['method']): row for row in au['records']}
    fm = {(row['case_id'], row['method']): row for row in fn['records']}
    available = {(item['case_id'], item['method']) for item in document['items'] if item['image_available']}
    require(set(am) == set(fm) == available and len(am) == len(au['records']) == len(fn['records']), 'Evaluator row mismatch')
    rows = []
    for item in document['items']:
        key = item['case_id'], item['method']
        source, target = am[(item['case_id'], 'source_real')], am[(item['case_id'], 'target_real')]
        if item['image_available']:
            require(am[key]['sha256'] == fm[key]['sha256'] == item['sha256'], 'Cross-evaluator SHA mismatch')
            row = {**am[key], **item, 'facenet_primary': fm[key], 'identity_cosine': fm[key]['identity_cosine']}
        else:
            row = {**item, 'vector': None, 'selected_au17': None, 'parse_status': 'generation_failed_not_evaluated',
                   'all_candidates': [], 'raw_rows': [], 'selected_row': None, 'face_count': None,
                   'facenet_primary': {'status': 'generation_failed_not_evaluated', 'candidates': [], 'face_count': None},
                   'identity_cosine': None}
        row.update(au_metrics(row['vector'], source['vector'], item['source_labels_12'], item['target_labels_12']))
        row['extra5'], row['extra5_proxy_mae_to_real_target'] = extras(row['selected_au17'], target['selected_au17'], source['selected_au17'])
        for au_number in EXTRA:
            row[f'extra_AU{au_number}_error_to_real_target'] = row['extra5'][str(au_number)]['absolute_error_to_target']
        row.update(buffalo_identity_cosine=None, buffalo_auxiliary={'status': 'not_run', 'reason': 'out_of_diagnostic_scope_not_detection_failure'},
                   identity_metric='independent FaceNet VGGFace2; buffalo not run')
        require(all(metric in row for metric in METRICS), 'Missing original metric column')
        rows.append(row)
    methods = REFERENCES + METHODS
    coverage = {}
    for method in methods:
        coverage[method] = {}
        for group in ('edited', 'no_edit'):
            part = [row for row in rows if row['method'] == method and (row['group'] == 'no_edit') == (group == 'no_edit')]
            planned = len(part)
            counts = {'image_available': sum(row['image_available'] for row in part),
                      'AU_valid': sum(row['vector'] is not None for row in part),
                      'FaceNet_valid': sum(row['identity_cosine'] is not None for row in part)}
            coverage[method][group] = {'planned': planned, **counts,
                **{name + '_rate': value / planned if planned else None for name, value in counts.items()},
                'buffalo_status': 'not_run', 'buffalo_valid_rate': None,
                'metric_valid_counts': {metric: sum(row[metric] is not None for row in part) for metric in METRICS}}
    generation_methods = ['baseline_text2img', 'bidirectional_text2img'] + METHODS
    pairs = [(a, b) for a, b in itertools.combinations(generation_methods, 2)]
    source_copy = [{**{key: row[key] for key in ('case_id', 'subject', 'group', 'path', 'sha256')},
                    'role': 'copy_source_reference_not_generation_method', 'source_measured_vector': row['vector'],
                    'metrics': {metric: row[metric] for metric in METRICS}} for row in rows if row['method'] == 'source_real']
    report = {'status': 'complete', 'generation_status': document['generation_status'], 'policy_sha256': POLICY_SHA,
              'test_used': False, 'case_count': 8, 'new_generated_planned': 48, 'logical_records': 88,
              'original_metric_columns': METRICS, 'records': rows, 'coverage': coverage,
              'per_method_subject_equal': {method: aggregate(rows, [method]) for method in methods},
              'all_generation_methods_common_valid': aggregate(rows, generation_methods),
              'pairwise': {a + '_vs_' + b: aggregate(rows, [a, b]) for a, b in pairs},
              'source_copy_references': source_copy,
              'source_copy_subject_equal': aggregate(rows, ['source_real']),
              'buffalo_policy': 'column retained as NA; not run, not a detector failure; no surrogate substituted',
              'conclusion_scope': 'dev4 subjects x no-edit/lowest-available diagnostic only; no significance test, quality threshold or SOTA ranking',
              'bindings': {name: sha(output / name) for name in ('manifest.json', 'items.json', 'input_seal.json',
                                                               'openface/results.json', 'facenet/results.json')}}
    folder = output / 'comparison'
    folder.mkdir(exist_ok=False)
    save(folder / 'results.json', report)
    save(folder / 'summary.json', {key: value for key, value in report.items() if key not in ('records', 'pairwise')})
    print(json.dumps({'status': 'complete', 'records': len(rows), 'result': str(folder / 'results.json')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('validate', 'prepare', 'openface', 'facenet', 'merge', 'all'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase == 'validate':
        require(args.run is not None, '--run required')
        document = validate(args.run)
        print(json.dumps({key: value for key, value in document.items() if key != 'items'}, ensure_ascii=False))
        return
    require(args.output is not None, '--output required; must be a fresh directory below this diagnostic')
    if args.phase in ('prepare', 'all'):
        require(args.run is not None, '--run required')
        prepare(args.run, args.output)
    if args.phase in ('openface', 'all'):
        openface(args.output)
    if args.phase in ('facenet', 'all'):
        facenet(args.output)
    if args.phase in ('merge', 'all'):
        merge(args.output)


if __name__ == '__main__':
    main()
