"""Unique-name bindings and pure schema checks for the new four-arm experiment."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SUPPLEMENT = HERE.parent
EXP = SUPPLEMENT.parent / '独立AU控制实验_20260914'
V2 = EXP / 'development_evaluation/evaluation_policy_v2'
ROLE = 'supplementary_projection_smoothing_ablation'
ARMS = ('full', 'no_projection', 'no_smoothing', 'both_off')
SEEDS = (101, 202, 303)
AUS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
EXTRA = (7, 10, 14, 23, 45)
POLICY_SHA = 'd29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c'
REFERENCES = EXP / 'completion_20260915/evaluation/references_v1/references.json'
REFERENCES_SHA = 'e93c575050f2fa0a3b1e312830fe4b622bc33d0a3fa76efeaf77c7484a168761'
PINS = {
    'openface': (V2 / 'openface_v2.py', '5446237ff2fba672bd06ec62820bf671d9deb4c776d83a28d18060a581b9f908'),
    'facenet': (V2 / 'run_facenet_single_directions.py', '5d8e1916bb5bcc9afc307980438b5f2f3fb0610f5dde12141f90e85efa7ebc2d'),
    'au_metrics': (EXP / 'development_evaluation/v3_global_raw_dev64/evaluate.py', '5a8b8944375f5b7ef1ebca3bbfe46cc2513beb2a110f27d5842d10f286d1e3dd'),
    'extra_metrics': (V2 / 'summarize_v2.py', 'e9aa7cc6263e1126a03a2aba103a5852fb999b9a4e227470852105018a0b0c12'),
    'statistics': (EXP / 'statistics/paired_subject_statistics.py', 'f0f1baaf08464ac2e46bfe607ae7ea34ae1c49f200288b57b45e32df2a44d4ed'),
}
PRIMARY = 'unchanged_source_relative_drift'
SUBGROUP_METRICS = ('target_mae_changed', PRIMARY, 'identity_cosine')
METRICS = {
    'target_mae_changed': 'lower', 'delta_mae_changed': 'lower',
    PRIMARY: 'lower', 'target_mae_all12': 'lower', 'delta_mae_all12': 'lower',
    'direction_correct_fraction': 'higher', 'identity_cosine': 'higher',
    'extra5_proxy_mae_to_real_target': 'lower',
    **{f'extra_AU{au}_error_to_real_target': 'lower' for au in EXTRA},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def verified(reference):
    require(isinstance(reference, dict), 'Path/SHA reference required')
    require(re.fullmatch(r'[0-9a-f]{64}', str(reference.get('sha256', ''))) is not None,
            'Explicit SHA256 required')
    path = Path(reference['path']).resolve()
    require(path.is_file() and sha(path) == reference['sha256'], f'File missing or changed: {path}')
    return path


def row_key(row):
    arm = row.get('arm', row.get('method'))
    require(arm in ARMS, 'Unknown experimental arm')
    require(type(row.get('seed')) is int and row['seed'] in SEEDS, 'Unexpected sampling seed')
    for field in ('subject', 'case_id'):
        require(isinstance(row.get(field), str) and bool(row[field]), f'Missing {field}')
    return arm, row['subject'], row['case_id'], row['seed']


def reference_cases(document):
    cases = {}
    for row in document['cases']:
        if row['source_labels_12'] == row['target_labels_12']:
            continue
        cid = row['case_id']
        require(cid not in cases, 'Duplicate reference case')
        for key in ('source_labels_12', 'target_labels_12'):
            values = row[key]
            require(len(values) == 12 and all(type(v) in (int, float) and math.isfinite(v)
                    and 0 <= v <= 5 for v in values), 'Invalid manually annotated AU vector')
        cases[cid] = row
    require(len(cases) == 24, 'Exactly 24 fixed nonzero requests required')
    subjects = {row['subject'] for row in cases.values()}
    require(len(subjects) == 8 and all(sum(r['subject'] == s for r in cases.values()) == 3
            for s in subjects), 'Exactly 8 subjects, 3 nonzero requests each required')
    return cases


def normalize_generation(document, cases):
    """Keep all 288 slots, including missing/failed attempts; never use old scores."""
    plan = [{'arm': arm, 'method': arm, 'subject': case['subject'], 'case_id': cid, 'seed': seed}
            for arm in ARMS for cid, case in cases.items() for seed in SEEDS]
    expected = {row_key(row) for row in plan}
    supplied = [row_key(row) for row in document['slot_plan']]
    require(len(supplied) == len(set(supplied)) == 288 and set(supplied) == expected,
            'Generation plan differs from the fixed 288-slot denominator')
    rows = {}
    for row in document['records']:
        key = row_key(row)
        require(key in expected and key not in rows, 'Unplanned/duplicate generation record')
        case = cases[row['case_id']]
        require(row['subject'] == case['subject'], 'Generation subject differs from frozen case')
        require(row.get('source_au12') == case['source_labels_12'] and
                row.get('target_au12') == case['target_labels_12'], 'Generation AU labels changed')
        require(row.get('status') in ('success', 'failed', 'not_started'), 'Nonterminal generation status')
        source = case['source_reference']['image']
        require(row.get('source_sha256') == source['sha256'] and
                Path(row['source_path']).resolve() == Path(source['path']).resolve(),
                'Canonical source does not match frozen reference')
        if row['status'] == 'success':
            require(isinstance(row.get('output_path'), str) and bool(row['output_path']) and
                    re.fullmatch(r'[0-9a-f]{64}', str(row.get('output_sha256', ''))) is not None,
                    'Successful generation lacks output path/SHA')
        rows[key] = row
    return plan, rows


def analysis_spec():
    return {
        'metrics': {key: {'direction': direction} for key, direction in METRICS.items()},
        'comparisons': [{'id': 'full_vs_' + arm, 'method': 'full', 'reference': arm,
                         'metric': PRIMARY, 'family': 'projection_smoothing_primary_DU'}
                        for arm in ARMS[1:]],
        'bootstrap_samples': 50000, 'bootstrap_seed': 20261023, 'confidence_level': .95,
    }


def subgroup_plan(cases):
    """Label-only, two mutually exclusive partitions; never use predicted AU."""
    rows = {}
    for cid, case in cases.items():
        source, target = case['source_labels_12'], case['target_labels_12']
        require(not any(source), 'These predefined target-intensity strata require the frozen zero-AU sources')
        active = [(au, value) for au, value in zip(AUS, target) if value > 0]
        require(bool(active) and all(value >= 1 for _, value in active), 'Nonzero integer-grade request required')
        rows[cid] = {
            'subject': case['subject'], 'active_aus': [au for au, _ in active],
            'intensity_partition': 'all_active_intensity1' if all(value == 1 for _, value in active) else 'has_intensity_gt1',
            'combination_partition': 'single_active_au' if len(active) == 1 else 'multi_active_au',
        }
    coverage = {str(au): sum(au in row['active_aus'] for row in rows.values()) for au in AUS}
    return {
        'assignment_basis': 'frozen manual target AU labels only; source labels are all zero',
        'case_assignments': rows, 'active_AU_case_counts': coverage,
        'missing_active_AUs': [au for au in AUS if coverage[str(au)] == 0],
        'metrics': list(SUBGROUP_METRICS), 'descriptive_only': True,
        'new_subjects_or_expanded_AU_coverage': False,
        'partitions': {
            'intensity_partition': ['all_active_intensity1', 'has_intensity_gt1'],
            'combination_partition': ['single_active_au', 'multi_active_au'],
        },
    }


def calibration_plan():
    return {
        'kind': 'automatic_AU_estimator_bias_check_on_real_manually_labeled_frames',
        'references': ['source_real', 'target_real'], 'manual_label_source': ref(REFERENCES),
        'AU_order': list(AUS), 'intensity_strata': ['all', 0, 1, 2, 3, 4, 5],
        'metrics': ['absolute_error', 'signed_error', 'manual_intensity', 'estimated_intensity'],
        'signed_error': 'OpenFace estimate minus manual label',
        'deduplication': 'same image SHA once; conflicting subject/labels excluded and explicitly listed',
        'aggregation': 'unique real frames averaged within subject then subjects equally weighted',
        'failure_policy': 'retain planned image count; invalid measured 12AU vector yields NA',
        'descriptive_only': True, 'generated_image_human_evaluation': False,
        'independent_new_subject_validation': False,
    }


def load_legacy(name):
    path, expected = PINS[name]
    require(sha(path) == expected, 'Frozen algorithm changed: ' + name)
    spec = importlib.util.spec_from_file_location('ablation_legacy_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_functions():
    """Execute only the original pure functions, without model/evaluator imports."""
    namespace = {'AUS': AUS, 'EXTRA': EXTRA}
    for name, selected in (('au_metrics', {'mean', 'au_metrics'}),
                           ('extra_metrics', {'extra_errors'})):
        path, expected = PINS[name]
        require(sha(path) == expected, 'Frozen metric algorithm changed: ' + name)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        require({node.name for node in nodes} == selected, 'Missing metric function')
        code = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(code, str(path), 'exec'), namespace)
    return namespace['au_metrics'], namespace['extra_errors']


def code_bindings():
    result = {}
    for name, (path, expected) in PINS.items():
        require(sha(path) == expected, 'Frozen algorithm changed: ' + name)
        result[name] = ref(path)
    return result
