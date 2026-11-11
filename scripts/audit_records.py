"""Recompute archived per-image measurement summaries without model inference."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]

def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def number(value):
    if value in ("", None):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite stored measurement")
    return parsed

def aggregate(data, method, metric, group="all"):
    by_case = defaultdict(list)
    seen = set()
    for row in data:
        if row["method"] != method or (group != "all" and row.get("group") != group):
            continue
        key = (row["subject"], row["case_id"], row["seed"])
        if key in seen:
            raise ValueError("Duplicate case/seed measurement")
        seen.add(key)
        v = number(row.get(metric))
        if v is not None:
            by_case[key[:2]].append(v)
    by_subject = defaultdict(list)
    for (subject, _), values in by_case.items():
        by_subject[subject].append(statistics.mean(values))
    means = [statistics.mean(values) for values in by_subject.values()]
    return (statistics.mean(means) if means else None, len(means), len(by_case))

def check(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{label}: {actual!r} != {expected!r}")
    elif not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{label}: {actual!r} != {expected!r}")

def holm(values):
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    result, running = [0.0] * len(values), 0.0
    for rank, index in enumerate(ordered):
        running = max(running, (len(values)-rank)*values[index])
        result[index] = min(1.0, running)
    return result

def paired_primary(data, comparisons):
    results = []
    for item in comparisons:
        metric, group = item['metric'].rsplit('__', 1)
        means = []
        for method in (item['method'], item['reference']):
            by_case = defaultdict(list)
            for row in data:
                if row['method'] == method and row['group'] == group:
                    value = number(row.get(metric))
                    if value is not None:
                        by_case[(row['subject'],row['case_id'])].append(value)
            means.append({key:statistics.mean(v) for key,v in by_case.items()})
        common = sorted(set(means[0]) & set(means[1]))
        by_subject = defaultdict(list)
        for key in common:
            by_subject[key[0]].append(means[0][key]-means[1][key])
        differences = [statistics.mean(v) for _,v in sorted(by_subject.items())]
        if not differences:
            raise ValueError('No paired subjects')
        observed = statistics.mean(differences)
        count = sum(abs(statistics.mean(sign*x for sign,x in zip(signs,differences))) >= abs(observed)-1e-12
                    for signs in itertools.product((-1,1),repeat=len(differences)))
        p = count/(2**len(differences))
        check(observed,item['mean_difference'],item['id']+'/difference')
        check(p,item['two_sided_sign_flip_p'],item['id']+'/exact_p')
        check(len(differences),item['n_subjects'],item['id']+'/subjects')
        check(len(common),item['n_cases'],item['id']+'/cases')
        results.append(p)
    for actual,item in zip(holm(results),comparisons):
        check(actual,item['holm_adjusted_p'],item['id']+'/holm')
    return len(comparisons)

def verify_manifest(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        raw = root / item["file"]
        p = raw.resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or raw.is_symlink():
            raise ValueError("Invalid manifest path: " + item["file"])
        if p.stat().st_size != item["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("File changed: " + item["file"])
    return len(manifest["files"])

def audit(root=ROOT):
    files = verify_manifest(root)
    data = rows(root / "records/main_per_image.csv")
    expected = rows(root / "records/main_summary.csv")
    if len(data) != 896:
        raise ValueError("Expected 896 logical rows; aliases are not independent subjects")
    external = json.loads((root/'records/main_paired_statistics.json').read_text())
    internal = json.loads((root/'records/internal_paired_statistics.json').read_text())
    ext = [c for c in external['comparisons'] if c['family']=='all_five_external_primary_C']
    inter = [c for c in internal['comparisons'] if c['family']=='internal_primary_C']
    if len(ext)!=5 or len(inter)!=4:
        raise ValueError('Expected five external and four internal primary contrasts')
    paired_checks = paired_primary(data,ext)+paired_primary(data,inter)
    summaries = 0
    for item in expected:
        metric, group = item["metric"].rsplit("__", 1)
        value, subjects, cases = aggregate(data, item["method"], metric, group)
        check(value, number(item["mean"]), item["method"] + "/" + item["metric"])
        check(subjects, number(item["n_subjects"]), "subjects")
        check(cases, number(item["n_cases"]), "cases")
        summaries += 1
    ablation = rows(root / "records/ablation_per_image.csv")
    if len(ablation) != 288:
        raise ValueError("Expected 288 ablation slots")
    ab_checks = 0
    for item in rows(root / "records/ablation_summary.csv"):
        for metric, expected_value in item.items():
            if metric == "method":
                continue
            value, subjects, cases = aggregate(ablation, item["method"], metric)
            check(value, number(expected_value), item["method"] + "/" + metric)
            check(subjects, 8, "ablation subjects")
            check(cases, 24, "ablation cases")
            ab_checks += 1
    show = []
    for method in ["bidirectional_local_lp_s050", "magicface", "faceadapter", "liveportrait", "swiftedit", "performrecast"]:
        values = {metric: aggregate(data, method, metric, "edited") for metric in
                  ["target_mae_changed", "unchanged_source_relative_drift", "identity_cosine"]}
        show.append({"method": method, "values": values})
    full = aggregate(ablation, "full", "unchanged_source_relative_drift")[0]
    no_projection = aggregate(ablation, "no_projection", "unchanged_source_relative_drift")[0]
    no_smoothing = aggregate(ablation, "no_smoothing", "unchanged_source_relative_drift")[0]
    return {"status": "PASS", "verified_files": files, "main_measurement_rows": len(data),
            "main_summaries_recomputed": summaries, "ablation_measurement_rows": len(ablation),
            "ablation_summaries_recomputed": ab_checks,
            "paired_primary_tests_recomputed": paired_checks,
            "projection_DU_reduction_percent": (no_projection-full)/no_projection*100,
            "smoothing_DU_reduction_percent": (no_smoothing-full)/no_smoothing*100,
            "external_comparison": show,
            "scope": "Recomputed seed/case/subject means from archived measurements. No training, inference, AU/FaceNet rerun or bootstrap rerun.",
            "models_loaded": False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional NEW receipt file; existing file is never overwritten")
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)
    return 0

if __name__ == "__main__":
    sys.exit(main())
