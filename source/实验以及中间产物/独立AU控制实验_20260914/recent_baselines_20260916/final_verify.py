"""Independent final arithmetic check from full per-image CSV; no model calls."""
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
EVAL = ROOT/'evaluation_20260916_v1'
TABLE = EVAL/'formal128_v1/statistics/all_per_image.csv'
PAIR = EVAL/'paired_descriptives_v1.json'
STATS = EVAL/'formal128_v1/statistics/paired_subject_results.json'
OUR = 'bidirectional_local_lp_s050'
METRICS = ['target_mae_changed', 'identity_cosine', 'unchanged_source_relative_drift']


def read_bytes(path):
    path = Path(path)
    before = path.stat()
    with path.open('rb', buffering=0) as f:
        chunks, left = [], before.st_size
        while left:
            part = f.read(min(left, 1024*1024))
            assert part, f'Short read: {path}'
            chunks.append(part)
            left -= len(part)
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    return b''.join(chunks)


def main():
    raw = read_bytes(TABLE)
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    counts = Counter(r['method'] for r in rows)
    assert len(rows) == 896 and len({(r['method'], r['case_id'], r['seed']) for r in rows}) == 896
    assert counts['swiftedit'] == 96 and counts['performrecast'] == 32
    assert all(n == 96 for m, n in counts.items() if m != 'performrecast')
    assert all(r['generation_status'] == 'complete' and r['input_availability'] == 'available' for r in rows)
    assert all(r['seed'] == '' and r['seed_not_applicable'] == 'True' for r in rows if r['method'] == 'performrecast')
    assert all(r['seed'] in ('101','202','303') for r in rows if r['method'] != 'performrecast')
    buckets = defaultdict(list)
    for r in rows:
        if r['group'] != 'edited':
            continue
        for metric in METRICS:
            if r[metric] != '':
                value = float(r[metric])
                assert math.isfinite(value)
                buckets[(r['method'], r['subject'], r['case_id'], metric)].append(value)
    means = {k: mean(v) for k, v in buckets.items()}
    reported = json.loads(read_bytes(PAIR))['rows']
    checks = []
    for target in reported:
        metric, other = target['metric'], target['reference']
        subjects = defaultdict(lambda: [[],[]])
        case_count = 0
        for (m, subject, case, met), ours in means.items():
            if m != OUR or met != metric:
                continue
            key = (other, subject, case, met)
            if key in means:
                subjects[subject][0].append(ours)
                subjects[subject][1].append(means[key])
                case_count += 1
        ours = mean(mean(v[0]) for v in subjects.values())
        other_value = mean(mean(v[1]) for v in subjects.values())
        assert abs(ours-target['ours']) < 1e-12
        assert abs(other_value-target['reference_mean']) < 1e-12
        assert case_count == target['n_cases'] and len(subjects) == target['n_subjects']
        reduction = None if metric == 'identity_cosine' else 1-ours/other_value
        if reduction is not None:
            assert abs(reduction-target['relative_reduction_if_lower_better']) < 1e-12
        checks.append({'reference': other, 'metric': metric, 'ours': ours, 'other': other_value,
                       'relative_reduction': reduction, 'cases': case_count, 'subjects': len(subjects)})
    comparisons = json.loads(read_bytes(STATS))['comparisons']
    assert len(comparisons) == 5
    running = 0
    for rank, r in enumerate(sorted(comparisons, key=lambda v: v['two_sided_sign_flip_p'])):
        running = max(running, min(1, (5-rank)*r['two_sided_sign_flip_p']))
        assert abs(running-r['holm_adjusted_p']) < 1e-12
    overall = {}
    for method in [OUR, 'magicface', 'faceadapter', 'liveportrait', 'swiftedit', 'performrecast']:
        grouped = defaultdict(list)
        for (m, subject, case, metric), value in means.items():
            if m == method and metric == 'unchanged_source_relative_drift':
                grouped[subject].append(value)
        overall[method] = mean(mean(v) for v in grouped.values())
    assert overall[OUR] == min(overall.values())
    report = {'status': 'pass', 'source_csv_sha256': hashlib.sha256(raw).hexdigest(),
              'source_rows': len(rows), 'method_counts': dict(counts), 'paired_checks': checks,
              'non_target_drift_subject_means': overall, 'holm5_independent_recompute': True,
              'algorithm': 'CSV -> valid seeds per case -> common cases per metric -> equal subject means',
              'models_called': False, 'source_files_modified': False}
    with (ROOT/'final_verification_v1.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'status':'pass','rows':len(rows),'paired_checks':len(checks),'holm5':True}))


if __name__ == '__main__':
    main()
