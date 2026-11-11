"""Independent completion statistics arithmetic; no evaluator/statistics engine calls."""
import collections
import math


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def close(actual, expected, where='value'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(where + ': keys differ')
        for key in expected:
            close(actual[key], expected[key], where + '.' + str(key))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(where + ': lengths differ')
        for index, (left, right) in enumerate(zip(actual, expected)):
            close(left, right, where + '[' + str(index) + ']')
    elif type(expected) in (int, float):
        if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > 1e-12:
            raise ValueError(where + ': numeric mismatch ' + repr((actual, expected)))
    elif actual != expected:
        raise ValueError(where + ': value mismatch ' + repr((actual, expected)))


def key(row):
    return tuple(row[name] for name in ('method', 'subject', 'case_id', 'seed'))


def mapped(rows):
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate logical key')
    return result


def aggregate(values):
    cases = collections.defaultdict(list)
    for (subject, case, seed), value in sorted(values.items()):
        cases[subject, case].append(value)
    subjects = collections.defaultdict(list)
    for (subject, case), samples in cases.items():
        subjects[subject].append(mean(samples))
    by_subject = {subject: mean(v) for subject, v in sorted(subjects.items())}
    return {'mean': mean(by_subject.values()), 'n_subjects': len(subjects), 'n_cases': len(cases),
        'n_seed_outputs': len(values), 'subject_means': by_subject,
        'case_means': [{'subject': subject, 'case_id': case, 'seed_mean': mean(samples), 'n_seeds': len(samples)}
                      for (subject, case), samples in sorted(cases.items())]}


def inference(values, seed):
    """Recompute50k PCG64 draws and manual linear quantiles; exact sign enumeration."""
    import itertools
    import numpy as np
    values = list(values); n = len(values)
    if n > 8 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Expected at most8 finite subject differences')
    result = {'n_subjects': n, 'mean_difference': mean(values), 'difference_ci95': None,
        'two_sided_sign_flip_p': None, 'sign_patterns': None, 'bootstrap_samples': 0, 'bootstrap_seed': seed,
        'inference_status': 'no_paired_subjects' if not n else 'descriptive_only_one_subject' if n == 1 else 'computed'}
    if n < 2:
        return result
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.asarray(values, dtype=np.float64)[rng.integers(0, n, size=(50000, n))]
    draws = np.sort(samples.sum(axis=1) / n)
    def quantile(p):
        position = (len(draws) - 1) * p
        lo = math.floor(position); hi = math.ceil(position)
        return float(draws[lo] + (draws[hi] - draws[lo]) * (position - lo))
    observed = abs(mean(values)); tolerance = 1e-12 * max(1., observed)
    extreme = sum(abs(mean(sign * value for sign, value in zip(signs, values))) >= observed - tolerance
                  for signs in itertools.product((-1, 1), repeat=n))
    result.update(difference_ci95=[quantile((1 - .95) / 2), quantile(1 - (1 - .95) / 2)],
        two_sided_sign_flip_p=extreme / (2 ** n), sign_patterns=2 ** n,
        bootstrap_samples=50000, sign_flip_extreme_patterns=extreme)
    return result


def verify_statistics(document, result, rows, planned, cases, spec):
    close(document['analysis_spec'], spec, 'input analysis spec')
    close(result['analysis_spec'], spec, 'result analysis spec')
    close(mapped(document['planned']), mapped(planned), 'planned statistics rows')
    close(mapped(document['records']), mapped(rows), 'independently derived statistical input')
    records = mapped(rows); allkeys = sorted(mapped(planned)); methods = sorted({k[0] for k in allkeys})
    metrics = spec['metrics']; values = {}; reasons = {}; coverage = []
    close(set(result['raw_method_statistics']), set(methods), 'method statistics keys')
    for method in methods:
        own = [k for k in allkeys if k[0] == method]
        close(set(result['raw_method_statistics'][method]), set(metrics), 'metric definitions')
        for metric, definition in metrics.items():
            valid = {}; missing = []
            for k in own:
                row = records.get(k)
                value = row['metrics'][metric] if row is not None and row['status'] == 'complete' else None
                reason = ('missing_output' if row is None else 'row_status:' + row['status'] if row['status'] != 'complete'
                          else 'metric_NA' if value is None else None)
                values[k, metric] = value; reasons[k, metric] = reason
                if reason is None:
                    valid[k[1:]] = value
                else:
                    missing.append({'subject': k[1], 'case_id': k[2], 'seed': k[3], 'reason': reason})
            stat = aggregate(valid)
            close(result['raw_method_statistics'][method][metric], stat, method + '.' + metric)
            applicable = [k for k in own if definition['scope'] == 'all' or cases[k[2]]['group'] == definition['scope']]
            applicable_reasons = collections.Counter()
            for k in applicable:
                row = records.get(k)
                reason = 'missing_output' if row is None else row['metric_reasons'][metric]
                if reason is not None:
                    applicable_reasons[reason] += 1
            coverage.append({'method': method, 'metric': metric, 'planned_seed_outputs': len(own),
                'observed_output_rows': sum(k in records for k in own), 'valid_metric_outputs': len(valid),
                'planned_cases': len({k[1:3] for k in own}), 'planned_subjects': len({k[1] for k in own}),
                'valid_cases': stat['n_cases'], 'valid_subjects': stat['n_subjects'],
                'unavailable_by_reason': dict(collections.Counter(r['reason'] for r in missing)), 'unavailable_outputs': missing,
                'predeclared_scope': definition['scope'], 'planned_applicable_seed_outputs': len(applicable),
                'out_of_scope_seed_outputs': len(own) - len(applicable), 'applicable_unavailable_by_reason': dict(applicable_reasons),
                'applicable_generation_or_measurement_unavailable': len(applicable) - len(valid)})
    close(result['coverage_table'], coverage, 'scope-aware coverage')
    comparisons = {r['id']: r for r in result['comparisons']}
    if len(comparisons) != len(result['comparisons']) or set(comparisons) != {r['id'] for r in spec['comparisons']}:
        raise ValueError('Comparison roster changed')
    families = collections.defaultdict(list)
    for comparison in spec['comparisons']:
        actual = comparisons[comparison['id']]
        method, reference, metric = (comparison[k] for k in ('method', 'reference', 'metric'))
        common = [k[1:] for k in allkeys if k[0] == method]
        valid = [k for k in common if values[(method,) + k, metric] is not None and values[(reference,) + k, metric] is not None]
        left = aggregate({k: values[(method,) + k, metric] for k in valid})
        right = aggregate({k: values[(reference,) + k, metric] for k in valid})
        subject_rows = [{'subject': s, 'method_mean': value, 'reference_mean': right['subject_means'][s],
            'difference': value - right['subject_means'][s]} for s, value in left['subject_means'].items()]
        expected = {**comparison, 'metric_direction': metrics[metric]['direction'],
            'difference_definition': 'method minus reference, no direction flip', 'paired_method_mean': left['mean'],
            'paired_reference_mean': right['mean'], **inference([r['difference'] for r in subject_rows], spec['bootstrap_seed']),
            'n_cases': left['n_cases'], 'n_seed_pairs': len(valid), 'subject_statistics': subject_rows,
            'case_seed_means': {'method': left['case_means'], 'reference': right['case_means']},
            'coverage': {'planned_common_seed_outputs': len(common), 'valid_common_seed_outputs': len(valid),
                'method_only_planned_keys': [], 'reference_only_planned_keys': [],
                'planned_common_cases': len({k[:2] for k in common}), 'planned_common_subjects': len({k[0] for k in common}),
                'unavailable_pairs': [{'subject': k[0], 'case_id': k[1], 'seed': k[2], 'method_reason': reasons[(method,) + k, metric],
                    'reference_reason': reasons[(reference,) + k, metric]} for k in common if k not in valid]},
            'all_eight_subjects_paired': left['n_subjects'] == 8, 'expected_sign_patterns_at_full_coverage': 256}
        for field, value in expected.items():
            close(actual[field], value, comparison['id'] + '.' + field)
        families[comparison['family']].append(actual)
    for family, rows_in_family in families.items():
        ordered = sorted(rows_in_family, key=lambda r: (1. if r['two_sided_sign_flip_p'] is None else r['two_sided_sign_flip_p'], r['id']))
        running = 0.
        for rank, row in enumerate(ordered):
            p = row['two_sided_sign_flip_p']
            running = max(running, min(1., (len(ordered) - rank) * (1. if p is None else p)))
            close(row['holm_adjusted_p'], None if p is None else running, 'Holm ' + row['id'])
            close(row['holm_family_size'], len(ordered), 'Holm fixed family size')
    close(result['declared_holm_families'], {f: [r['id'] for r in rows_in_family] for f, rows_in_family in sorted(families.items())}, 'declared families')
    return {'raw_method_metric_aggregates': len(methods) * len(metrics), 'paired_comparisons': len(comparisons),
        'bootstrap_recomputed_samples_per_comparison': 50000,
        'family_sizes': {f: len(rows_in_family) for f, rows_in_family in sorted(families.items())}}
