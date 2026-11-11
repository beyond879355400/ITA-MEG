#!/usr/bin/env python3
"""Freeze subject split and plan label-balanced real frames; NO frame extraction."""
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

from prepare_disfa_pilot import AUS, file_state, sha, save, labels_for_subject, contiguous_runs

HERE = Path(__file__).resolve().parent
OUT = HERE / "au_supervision_v2"
RAW = Path.home() / "Desktop/Cursor/geme-main/DISFA"
SEED = 20260914
TRAIN_LOCK = {"SN016", "SN029"}
DEV_LOCK = {"SN012", "SN018"}
LEGACY_TEST_EXCLUDE = {"SN002", "SN010", "SN017", "SN023", "SN025", "SN026", "SN027", "SN028", "SN029"}
MIN_GAP = 8
NEUTRAL_FRACTION = .10


def rank(*parts):
    return hashlib.sha256(("au-supervision-v2|20260914|" + "|".join(map(str, parts))).encode()).hexdigest()


def all_runs(vectors):
    return {(a, v): contiguous_runs([f for f, x in vectors.items() if x[f"au{a}"] == v]) for a in AUS for v in range(6)}


def choose_split(subjects, runs):
    positive_bins = [(a, v) for a in AUS for v in range(1, 6)]
    features = {s: [len(runs[s][b]) for b in positive_bins] for s in subjects}
    total = [sum(features[s][i] for s in subjects) for i in range(len(positive_bins))]
    allowed_test = sorted(set(subjects) - LEGACY_TEST_EXCLUDE - TRAIN_LOCK - DEV_LOCK)
    rng, best = random.Random(SEED), None
    for _ in range(12000):
        test = set(rng.sample(allowed_test, 8))
        dev_pool = sorted(set(subjects) - test - TRAIN_LOCK - DEV_LOCK)
        dev = DEV_LOCK | set(rng.sample(dev_pool, 2))
        train = set(subjects) - test - dev
        assert len(train) == 15 and len(dev) == 4 and len(test) == 8 and TRAIN_LOCK <= train
        groups = {"train": sorted(train), "dev": sorted(dev), "test": sorted(test)}
        score = 0.
        for name, group in groups.items():
            for i, (_, level) in enumerate(positive_bins):
                if not total[i]:
                    continue
                count = sum(features[s][i] for s in group)
                if not count:
                    score += 1_000_000 if name == "train" else (10_000 if level <= 3 else 10)
                score += ((count / total[i]) - len(group) / len(subjects)) ** 2
        key = (score, tuple(groups["test"]), tuple(groups["dev"]))
        if best is None or key < best[0]:
            best = (key, groups)
    return best[1], best[0][0]


def make_candidates(subject, vectors, runs):
    candidates = {}
    for (au, level), intervals in runs.items():
        if level == 0:
            continue
        for first, last in intervals:
            frame = (first + last) // 2
            row = candidates.setdefault(frame, {"subject": subject, "frame": frame, "labels": vectors[frame], "representative_events": []})
            row["representative_events"].append({"au": au, "intensity": level, "start": first, "end": last})
    neutral = contiguous_runs([f for f, vals in vectors.items() if not any(vals.values())])
    for first, last in neutral:
        frame = (first + last) // 2
        candidates[frame] = {"subject": subject, "frame": frame, "labels": vectors[frame],
                             "representative_events": [{"type": "all_12_au_zero", "start": first, "end": last}]}
    for row in candidates.values():
        row["all_zero"] = not any(row["labels"].values())
    return candidates


def select_frames(name, subjects, candidates, limit, runs):
    quota_order = sorted(subjects, key=lambda s: rank("quota", name, s))
    base, extra = divmod(limit, len(subjects))
    quotas = {s: base + (i < extra) for i, s in enumerate(quota_order)}
    chosen = {s: [] for s in subjects}
    counts = collections.Counter()
    chosen_rows = []
    def eligible(s, row):
        return all(abs(row["frame"] - other) >= MIN_GAP for other in chosen[s])
    def append(s, row):
        chosen[s].append(row["frame"])
        chosen_rows.append({**row, "split": name})
        counts.update((a, row["labels"][f"au{a}"]) for a in AUS)
    # At most 10% per subject are wholly neutral. All other AU labels stay real.
    for s in quota_order:
        neutral = sorted((x for x in candidates[s].values() if x["all_zero"]), key=lambda x: rank("neutral", s, x["frame"]))
        needed = math.floor(quotas[s] * NEUTRAL_FRACTION)
        for row in neutral:
            if not needed:
                break
            if eligible(s, row):
                append(s, row)
                needed -= 1
    event_totals = {(a, v): sum(len(runs[s][(a, v)]) for s in subjects) for a in AUS for v in range(1, 6)}
    # Subject round-robin prevents one rich subject taking most of the budget.
    while True:
        added = 0
        for s in quota_order:
            if len(chosen[s]) >= quotas[s]:
                continue
            pool = [x for x in candidates[s].values() if not x["all_zero"] and eligible(s, x)]
            if not pool:
                continue
            def priority(row):
                bins = [(a, row["labels"][f"au{a}"]) for a in AUS if row["labels"][f"au{a}"] > 0]
                new = sum((1000. / math.sqrt(max(1, event_totals[b]))) for b in bins if counts[b] == 0)
                balance = sum(1. / ((1 + counts[b]) * math.sqrt(max(1, event_totals[b]))) for b in bins)
                return (-new, -balance, rank(name, s, row["frame"]))
            append(s, min(pool, key=priority))
            added += 1
        if not added:
            break
    chosen_rows.sort(key=lambda x: (x["subject"], x["frame"]))
    assert len(chosen_rows) <= limit
    for s in subjects:
        assert len(chosen[s]) <= quotas[s]
        assert all(b - a >= MIN_GAP for a, b in zip(sorted(chosen[s]), sorted(chosen[s])[1:]))
    return chosen_rows, quotas


def coverage(subjects, vectors, rows=None):
    counts = {f"au{a}": {str(v): 0 for v in range(6)} for a in AUS}
    values = (row["labels"] for row in rows) if rows is not None else (lab for s in subjects for lab in vectors[s].values())
    for labels in values:
        for a in AUS:
            counts[f"au{a}"][str(labels[f"au{a}"])] += 1
    return counts


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    label_root = RAW / "ActionUnit_Labels"
    subjects = sorted(p.name for p in label_root.iterdir() if p.is_dir() and p.name.startswith("SN") and p.name[2:].isdigit())
    assert len(subjects) == 27
    vectors, runs, sources = {}, {}, []
    for s in subjects:
        vectors[s], _, records = labels_for_subject(label_root, s)
        runs[s] = all_runs(vectors[s])
        sources.extend(records)
    save(OUT / "raw_manual_label_sources.json", sources)
    groups, score = choose_split(subjects, runs)
    assert TRAIN_LOCK <= set(groups["train"]) and DEV_LOCK <= set(groups["dev"])
    assert not (set(groups["test"]) & LEGACY_TEST_EXCLUDE)
    freeze = {"protocol": "independent_au_supervision_subjects_v2", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "seed": SEED, "groups": groups, "counts": {k: len(v) for k, v in groups.items()},
              "forced_train": sorted(TRAIN_LOCK), "forced_dev": sorted(DEV_LOCK), "test_excluded_legacy_subjects": sorted(LEGACY_TEST_EXCLUDE),
              "test_locked": True, "test_frames_prepared": False,
              "selection_basis": "original human AU label event coverage only; no new or old model predictions/outputs inspected",
              "search_iterations": 12000, "score": score, "au_order": list(AUS),
              "label_source_manifest_sha256": sha(OUT / "raw_manual_label_sources.json"), "planner_sha256": sha(__file__),
              "supersedes_subject_manifest_sha256": sha(HERE / "au_supervision_v1/subjects_frozen.json"),
              "revision_reason": "before any test extraction/output: increase independent test subjects 5 to 8; no model scores used",
              "exact_two_sided_sign_flip_minimum_p": 2 / (2 ** 8)}
    save(OUT / "subjects_frozen.json", freeze)
    freeze_hash = sha(OUT / "subjects_frozen.json")
    print(json.dumps({"subjects": groups, "frozen_sha256": freeze_hash}, ensure_ascii=False), flush=True)

    # Camera choice is only based on frame-count integrity, not model performance.
    original_inventory = json.loads((HERE / "pilot_v1_02/raw_video_inventory.json").read_text())
    camera_sources = {}
    for s in subjects:
        choices = [x for x in original_inventory if x["subject"] == s and x["frame_count_matches_labels"] and file_state(x["path"])["resident"]]
        choices.sort(key=lambda x: (x["camera"] != "right", x["path"]))
        if not choices:
            raise RuntimeError("No resident frame-count-matched camera: " + s)
        row = choices[0]
        current = file_state(row["path"])
        if current["size_bytes"] != row["size_bytes"] or current["mtime_ns"] != row["mtime_ns"]:
            raise RuntimeError("Raw video changed after metadata audit: " + s)
        camera_sources[s] = {**current, "camera": row["camera"], "video_stream": row["video_stream"],
                             "manual_label_frames": row["manual_label_frames"],
                             "camera_reason": "right matches labels" if row["camera"] == "right" else "right header mismatch; same-subject left header matches labels; no index shift"}
    save(OUT / "camera_sources.json", camera_sources)
    candidates = {s: make_candidates(s, vectors[s], runs[s]) for s in groups["train"] + groups["dev"]}
    manifests, summaries, decode_jobs = {}, {}, []
    for name, limit in (("train", 600), ("dev", 240)):
        rows, quotas = select_frames(name, groups[name], candidates, limit, runs)
        for s in groups[name]:
            camera_sources[s]["sha256"] = sha(camera_sources[s]["path"])
        for row in rows:
            s, f = row["subject"], row["frame"]
            planned = OUT / "raw_frames" / name / s / f"{s}_{camera_sources[s]['camera']}_{f:05d}.png"
            assert not planned.exists()
            row.update(sample_id=f"au_sup_v2_{name}_{s}_f{f:05d}", label_frame=f, video_frame=f - 1,
                       camera=camera_sources[s]["camera"], video_path=camera_sources[s]["path"],
                       original_video_sha256=camera_sources[s]["sha256"],
                       image_path=str(planned), frame_path=str(planned), relative_image_path=str(planned.relative_to(OUT.parent.parent)),
                       au_order=list(AUS), au_vector_0_5=[row["labels"][f"au{a}"] for a in AUS],
                       au_vector_0_1=[row["labels"][f"au{a}"] / 5 for a in AUS], extracted=False)
        full_cov, selected_cov = coverage(groups[name], vectors), coverage(groups[name], vectors, rows)
        missing_present = [f"AU{a}={v}" for a in AUS for v in range(6) if full_cov[f"au{a}"][str(v)] and not selected_cov[f"au{a}"][str(v)]]
        unavailable = [f"AU{a}={v}" for a in AUS for v in range(6) if not full_cov[f"au{a}"][str(v)]]
        summary = {"limit": limit, "planned_frames": len(rows), "subjects": groups[name], "subject_quotas": quotas,
                   "frames_by_subject": dict(collections.Counter(r["subject"] for r in rows)),
                   "all_zero_frames": sum(r["all_zero"] for r in rows), "min_gap_frames_within_subject": MIN_GAP,
                   "selected_label_counts": selected_cov, "raw_split_label_counts": full_cov,
                   "available_but_not_sampled_bins": missing_present, "unavailable_bins_in_split": unavailable,
                   "unique_representative_positive_events": len({(r["subject"], e["au"], e["intensity"], e["start"], e["end"])
                        for r in rows for e in r["representative_events"] if "au" in e})}
        manifest = {"protocol": "au_supervision_v2", "split": name, "subjects_frozen_sha256": freeze_hash,
                    "count": len(rows), "frames_extracted": 0, "samples": rows}
        save(OUT / f"{name}_manifest_planned.json", manifest)
        save(OUT / f"{name}_sampling_summary.json", summary)
        manifests[name], summaries[name] = manifest, summary
        for s in groups[name]:
            subset = [r for r in rows if r["subject"] == s]
            decode_jobs.append({"split": name, "subject": s, "camera": camera_sources[s]["camera"],
                                "video_path": camera_sources[s]["path"], "original_video_sha256": camera_sources[s]["sha256"],
                                "label_frames": [r["label_frame"] for r in subset], "video_frames": [r["video_frame"] for r in subset],
                                "outputs": [r["image_path"] for r in subset],
                                "decode_policy": "one sequential decode per video; select exact indices; full original resolution PNG; no crop/resize/enhancement"})
    save(OUT / "train_dev_video_sources_hashed.json", {s: camera_sources[s] for s in groups["train"] + groups["dev"]})
    save(OUT / "decode_jobs_planned.json", {"jobs": decode_jobs, "count": len(decode_jobs), "test_jobs": 0, "executed": False})
    global_cov = coverage(subjects, vectors)
    summary = {"status": "manifests_only_waiting_for_extraction_authorization", "subjects_frozen_sha256": freeze_hash,
               "groups": groups, "train_count": manifests["train"]["count"], "dev_count": manifests["dev"]["count"],
               "all_zero_counts": {k: v["all_zero_frames"] for k, v in summaries.items()},
               "available_but_not_sampled_bins": {k: v["available_but_not_sampled_bins"] for k, v in summaries.items()},
               "globally_unavailable_bins": [f"AU{a}={v}" for a in AUS for v in range(6) if not global_cov[f"au{a}"][str(v)]],
               "left_camera_subjects": [s for s in groups["train"] + groups["dev"] if camera_sources[s]["camera"] == "left"],
               "new_frames_extracted": 0, "test_frames_extracted": 0, "model_runs": 0}
    save(OUT / "planning_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
