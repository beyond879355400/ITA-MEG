"""Read-only, standard-library audit; never imports experiment aggregators.

Recalculate metrics from measurements, seed/case/subject means, exact paired
sign flips, Holm adjustment, manual-label subgroups and real-frame calibration.
Only the explicitly requested new receipt file is written.
"""

import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone


ROUND = Path(__file__).resolve().parents[1]
THESIS = ROUND.parents[1]
JOB = ROUND / "evaluation/jobs/formal_v1"
STATS = JOB / "statistics"
OLD_SUMMARY = THESIS / (
    "实验以及中间产物/独立AU控制实验_20260914/recent_baselines_20260916/"
    "evaluation_20260916_v1/formal128_v1/statistics/all_method_summary.csv"
)
ARMS = ("full", "no_projection", "no_smoothing", "both_off")
CORE = (
    "target_mae_changed", "delta_mae_changed",
    "unchanged_source_relative_drift", "identity_cosine",
)
AUS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
DU = "unchanged_source_relative_drift"


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def number(value):
    return None if value in (None, "") else float(value)


def key(row):
    return row["method"], row["subject"], row["case_id"], int(row["seed"])


class Audit:
    def __init__(self):
        self.checks = []
        self.inputs = {}

    def load(self, path, is_csv=False):
        path = Path(path).resolve()
        raw = path.read_bytes()
        self.inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        if is_csv:
            import io
            return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        return json.loads(raw)

    def check(self, label, actual, expected, tolerance=1e-11):
        if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
            passed = math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= tolerance
        else:
            passed = actual == expected
        self.checks.append({"check": label, "pass": passed, "actual": actual, "expected": expected})


def hierarchical(rows, metric):
    """Average valid seeds within case, cases within subject, then subjects."""
    cases = collections.defaultdict(list)
    for row in rows:
        value = number(row.get(metric))
        if row["status"] == "complete" and value is not None and math.isfinite(value):
            cases[(row["subject"], row["case_id"])].append(value)
    subjects = collections.defaultdict(list)
    for (subject, _case), values in cases.items():
        subjects[subject].append(mean(values))
    subject_means = {subject: mean(values) for subject, values in subjects.items()}
    return {
        "mean": mean(subject_means.values()), "valid_outputs": sum(map(len, cases.values())),
        "valid_cases": len(cases), "valid_subjects": len(subjects),
        "subject_means": subject_means,
    }


def run(audit):
    rows = audit.load(STATS / "all_case_seed_metrics.csv", is_csv=True)
    raw = audit.load(STATS / "all_case_seed_metrics.json")["records"]
    reported = {r["method"]: r for r in audit.load(STATS / "all_method_summary.csv", is_csv=True)}
    au = audit.load(JOB / "openface/results.json")["records"]
    face = audit.load(JOB / "facenet/results.json")["records"]
    am = {key(r): r for r in au}
    fm = {key(r): r for r in face}
    csvmap = {key(r): r for r in rows}
    audit.check("CSV rows", len(rows), 288)
    audit.check("CSV unique keys", len(csvmap), len(rows))
    audit.check("JSON unique keys", len({key(r) for r in raw}), len(raw))
    audit.check("JSON/CSV key sets", sorted(map(key, raw)), sorted(csvmap))
    audit.check("OpenFace unique keys", len(am), len(au))
    audit.check("FaceNet unique keys", len(fm), len(face))

    actual = {}
    for arm in ARMS:
        selected = [r for r in rows if r["method"] == arm]
        audit.check(arm + " planned outputs", len(selected), 72)
        audit.check(arm + " generation success", sum(r["generation_status"] == "success" for r in selected), 72)
        audit.check(arm + " complete", sum(r["status"] == "complete" for r in selected), 72)
        audit.check(arm + " seed set", sorted({int(r["seed"]) for r in selected}), [101, 202, 303])
        actual[arm] = {}
        for metric in CORE:
            result = hierarchical(selected, metric)
            actual[arm][metric] = result
            audit.check(arm + "/" + metric + " mean", result["mean"], number(reported[arm][metric]))
            audit.check(arm + "/" + metric + " counts", [result["valid_outputs"], result["valid_cases"], result["valid_subjects"]], [72, 24, 8])

    for row in raw:
        k = key(row)
        measured = am[k]
        source = am[("source_real", row["subject"], row["case_id"], 0)]
        pred, source_pred = measured["vector"], source["vector"]
        source_label, target_label = row["source_au12"], row["target_au12"]
        changed = [i for i in range(12) if source_label[i] != target_label[i]]
        unchanged = [i for i in range(12) if source_label[i] == target_label[i]]
        calculated = {
            "target_mae_changed": mean(abs(pred[i] - target_label[i]) for i in changed),
            "delta_mae_changed": mean(abs(pred[i] - source_pred[i] - (target_label[i] - source_label[i])) for i in changed),
            DU: mean(abs(pred[i] - source_pred[i]) for i in unchanged),
            "identity_cosine": fm[k]["identity_cosine"],
        }
        for metric, value in calculated.items():
            audit.check(str(k) + "/raw/" + metric, value, row["metrics"][metric])
            audit.check(str(k) + "/CSV/" + metric, value, number(csvmap[k][metric]))
        audit.check(str(k) + "/AU image SHA", measured["sha256"], row["output_sha256"])
        audit.check(str(k) + "/FaceNet image SHA", fm[k]["sha256"], row["output_sha256"])

    primary_csv = {r["reference"]: r for r in audit.load(STATS / "primary_DU_comparisons.csv", is_csv=True)}
    comparisons = []
    full = {(r["subject"], r["case_id"], int(r["seed"])): r for r in rows if r["method"] == "full"}
    for arm in ARMS[1:]:
        reference = {(r["subject"], r["case_id"], int(r["seed"])): r for r in rows if r["method"] == arm}
        paired = []
        for k in sorted(full.keys() & reference.keys()):
            left, right = full[k], reference[k]
            a, b = number(left[DU]), number(right[DU])
            if left["status"] == right["status"] == "complete" and a is not None and b is not None:
                paired.append({"subject": k[0], "case_id": k[1], "status": "complete", "difference": a - b})
        result = hierarchical(paired, "difference")
        delta = [result["subject_means"][sid] for sid in sorted(result["subject_means"])]
        observed = abs(mean(delta))
        signs = itertools.product((-1, 1), repeat=len(delta))
        extreme = sum(abs(mean(d * sign for d, sign in zip(delta, pattern))) >= observed - 1e-12 for pattern in signs)
        p = extreme / (2 ** len(delta))
        result.update(reference=arm, two_sided_sign_flip_p=p, sign_patterns=2 ** len(delta),
                      negative_subjects=sum(d < 0 for d in delta), zero_subjects=sum(d == 0 for d in delta))
        comparisons.append(result)
        expected = primary_csv[arm]
        audit.check(arm + "/DU difference", result["mean"], number(expected["mean_difference"]))
        audit.check(arm + "/DU p", p, number(expected["two_sided_sign_flip_p"]))
        for label, value in (("n_subjects", result["valid_subjects"]), ("n_cases", result["valid_cases"]), ("n_seed_pairs", result["valid_outputs"]), ("sign_patterns", 2 ** len(delta))):
            audit.check(arm + "/" + label, value, int(expected[label]))
    running = 0.0
    for rank, result in enumerate(sorted(comparisons, key=lambda r: r["two_sided_sign_flip_p"])):
        running = max(running, min(1.0, (len(comparisons) - rank) * result["two_sided_sign_flip_p"]))
        result["holm_adjusted_p"] = running
        audit.check(result["reference"] + "/Holm", running, number(primary_csv[result["reference"]]["holm_adjusted_p"]))

    old = {r["metric"]: r for r in audit.load(OLD_SUMMARY, is_csv=True) if r["method"] == "bidirectional_local_lp_s050"}
    old_comparison = {}
    for metric, text in reported["full"].items():
        if metric == "method":
            continue
        new_value, old_value = number(text), number(old[metric + "__edited"]["mean"])
        old_comparison[metric] = {"new": new_value, "old": old_value, "absolute_difference": abs(new_value - old_value)}
        audit.check("old full edited/" + metric, new_value, old_value, tolerance=0)

    metadata = {}
    for row in raw:
        value = {"subject": row["subject"], "target": row["target_au12"], "source": row["source_au12"]}
        if row["case_id"] in metadata:
            audit.check("case label agreement/" + str(key(row)), value, metadata[row["case_id"]])
        metadata[row["case_id"]] = value
    assignments = {}
    for cid, value in metadata.items():
        active = [v for v in value["target"] if v > 0]
        audit.check("nonzero target/" + cid, bool(active), True)
        audit.check("zero source labels/" + cid, value["source"], [0] * 12)
        assignments[cid] = {
            "intensity_partition": "all_active_intensity1" if max(active) == 1 else "has_intensity_gt1",
            "combination_partition": "single_active_au" if len(active) == 1 else "multi_active_au",
        }
    subgroup_plan = audit.load(STATS / "subgroup_plan.json")
    for cid, groups in assignments.items():
        for partition, group in groups.items():
            audit.check(cid + "/" + partition, group, subgroup_plan["case_assignments"][cid][partition])
    subgroup_counts = {}
    for reported_row in audit.load(STATS / "subgroup_summary.csv", is_csv=True):
        partition, group = reported_row["partition"], reported_row["subgroup"]
        selected = [r for r in rows if r["method"] == reported_row["method"] and assignments[r["case_id"]][partition] == group]
        result = hierarchical(selected, reported_row["metric"])
        prefix = "subgroup/" + "/".join(reported_row[c] for c in ("partition", "subgroup", "method", "metric"))
        audit.check(prefix + "/mean", result["mean"], number(reported_row["subject_equal_mean"]))
        counts = {"planned_subjects": len({r["subject"] for r in selected}), "planned_cases": len({r["case_id"] for r in selected}),
                  "planned_seed_outputs": len(selected), "valid_subjects": result["valid_subjects"],
                  "valid_cases": result["valid_cases"], "valid_seed_outputs": result["valid_outputs"]}
        for name, value in counts.items():
            audit.check(prefix + "/" + name, value, int(reported_row[name]))
        audit.check(prefix + "/descriptive only", reported_row["descriptive_only"], "True")
        subgroup_counts[group] = {"subjects": counts["planned_subjects"], "cases": counts["planned_cases"]}

    items = audit.load(JOB / "items.json")
    dedup = {}
    for row in items["items"]:
        if row["method"] not in ("source_real", "target_real"):
            continue
        labels = row["source_labels_12"] if row["method"] == "source_real" else row["target_labels_12"]
        value = {"subject": row["subject"], "labels": labels, "prediction": am[key(row)]["vector"]}
        if row["sha256"] in dedup:
            audit.check("same real image metadata/" + str(key(row)), value, dedup[row["sha256"]])
        dedup[row["sha256"]] = value
    calibration = audit.load(STATS / "real_reference_calibration.json")
    audit.check("unique real reference frames", len(dedup), 32)
    audit.check("reported unique real frames", calibration["unique_reference_images"], len(dedup))
    audit.check("real label conflicts", len(calibration["reference_label_conflicts"]), 0)
    audit.check("real refs without SHA", len(calibration["unavailable_without_sha"]), 0)
    for row in calibration["physical_records"]:
        audit.check("calibration raw/" + row["sha256"], {"subject": row["subject"], "labels": row["manual_labels_12"], "prediction": row["measured_12"]}, dedup[row["sha256"]])
    calibration_csv = audit.load(STATS / "real_reference_calibration.csv", is_csv=True)
    for row in calibration_csv:
        idx = AUS.index(int(row["AU"]))
        stratum = row["manual_intensity_stratum"]
        selected = [r for r in dedup.values() if stratum == "all" or r["labels"][idx] == int(stratum)]
        valid = [r for r in selected if r["prediction"] is not None]
        subjects = collections.defaultdict(list)
        for item in valid:
            subjects[item["subject"]].append(item)
        functions = {
            "subject_equal_MAE": lambda r: abs(r["prediction"][idx] - r["labels"][idx]),
            "subject_equal_bias": lambda r: r["prediction"][idx] - r["labels"][idx],
            "subject_equal_manual_intensity": lambda r: r["labels"][idx],
            "subject_equal_estimated_intensity": lambda r: r["prediction"][idx],
        }
        prefix = "calibration/" + row["AU"] + "/" + stratum
        for name, function in functions.items():
            value = mean(mean(function(r) for r in group) for group in subjects.values())
            audit.check(prefix + "/" + name, value, number(row[name]))
        for name, value in (("planned_unique_images", len(selected)), ("planned_subjects", len({r["subject"] for r in selected})), ("valid_unique_images", len(valid)), ("valid_subjects", len(subjects))):
            audit.check(prefix + "/" + name, value, int(row[name]))
        audit.check(prefix + "/descriptive only", row["descriptive_only"], "True")
    return {
        "method_metrics": actual, "primary_DU_comparisons": comparisons,
        "old_full_parity": old_comparison, "subgroup_counts": subgroup_counts,
        "real_reference_unique_images": len(dedup),
        "real_reference_12AU_macro_MAE": mean(number(r["subject_equal_MAE"]) for r in calibration_csv if r["manual_intensity_stratum"] == "all"),
        "scope": {"independent_final_test": items["independent_final_test"],
                  "previously_exposed_subjects": items["previously_exposed_subjects"],
                  "generated_image_human_evaluation": calibration["plan"]["generated_image_human_evaluation"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROUND / "validation") or output.exists():
        raise ValueError("Receipt must be a new file within this round's validation directory")
    started = datetime.now(timezone.utc).isoformat()
    audit = Audit()
    result = run(audit)
    # Check the inputs again so receipt hashes bind the actual unmodified inputs.
    for path, expected in list(audit.inputs.items()):
        audit.check("input unchanged/" + path, hashlib.sha256(Path(path).read_bytes()).hexdigest(), expected)
    failed = [check for check in audit.checks if not check["pass"]]
    receipt = {
        "status": "PASS" if not failed else "FAIL", "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "python": sys.version, "input_sha256": audit.inputs,
        "original_aggregation_imported": False, "models_loaded": False,
        "source_files_modified": False, "checks": len(audit.checks),
        "passed": len(audit.checks) - len(failed), "failed": failed,
        "results": result, "check_details": audit.checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "checks": receipt["checks"], "failed": len(failed), "output": str(output)}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
