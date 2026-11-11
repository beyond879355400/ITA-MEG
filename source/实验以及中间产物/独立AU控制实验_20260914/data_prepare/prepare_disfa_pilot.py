#!/usr/bin/env python3
"""New subject-isolated DISFA pilot from original manual labels, never old outputs."""
import argparse
import collections
import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
AUS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
SEED = 20260914
TAG = "independent-au-control-20260914-v1"


def file_state(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False, "resident": False}
    s = path.stat()
    return {"path": str(path), "exists": True, "resident": not bool(s.st_flags & 0x40000000) and s.st_blocks > 0,
            "size_bytes": s.st_size, "blocks": s.st_blocks, "flags": hex(s.st_flags), "mtime_ns": s.st_mtime_ns}


def sha(path):
    if not file_state(path)["resident"]:
        raise RuntimeError("Refusing nonresident source: " + str(path))
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            d.update(b)
    return d.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def stable(*parts):
    return hashlib.sha256((TAG + "|" + "|".join(map(str, parts))).encode()).hexdigest()


def contiguous_runs(frames):
    if not frames:
        return []
    runs, first, previous = [], frames[0], frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            runs.append((first, previous))
            first = frame
        previous = frame
    runs.append((first, previous))
    return runs


def labels_for_subject(label_root, subject):
    data, records, counts = {}, [], {}
    common = None
    for au in AUS:
        path = label_root / subject / f"{subject}_au{au}.txt"
        if not file_state(path)["resident"]:
            raise RuntimeError("Manual label unavailable: " + str(path))
        rows = {}
        with path.open(newline="", encoding="utf-8-sig") as f:
            for cells in csv.reader(f):
                if not cells:
                    continue
                if len(cells) != 2:
                    raise ValueError("Malformed raw label row")
                frame, value = map(int, cells)
                if frame < 1 or value not in range(6) or frame in rows:
                    raise ValueError("Invalid raw frame/intensity/duplicate")
                rows[frame] = value
        indices = set(rows)
        if common is not None and indices != common:
            raise ValueError("Different frame sets among AUs: " + subject)
        common = indices
        frames = sorted(rows)
        if frames != list(range(1, frames[-1] + 1)):
            raise ValueError("Raw label frame sequence is not contiguous")
        values = collections.Counter(rows.values())
        counts[f"au{au}"] = {"frames_by_intensity": {str(i): values[i] for i in range(6)},
                             "events_by_exact_intensity": {str(i): len(contiguous_runs([n for n in frames if rows[n] == i])) for i in (1, 2, 3)},
                             "positive_events": len(contiguous_runs([n for n in frames if rows[n] > 0])),
                             "weak_1_2_events": len(contiguous_runs([n for n in frames if 1 <= rows[n] <= 2]))}
        records.append({**file_state(path), "sha256": sha(path), "subject": subject, "au": au, "frames": len(frames)})
        data[au] = rows
    vectors = {n: {f"au{a}": data[a][n] for a in AUS} for n in sorted(common)}
    return vectors, counts, records


def probe_video(path):
    state = file_state(path)
    if not state["resident"]:
        return state
    command = [shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(path)]
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=15)
        if p.returncode:
            state["probe_error"] = p.stderr
        else:
            state["video_stream"] = json.loads(p.stdout)["streams"][0]
    except subprocess.TimeoutExpired:
        state["probe_error"] = "15-second metadata timeout"
    return state


def candidate_split(subjects, coverage):
    # Labels only; no pixel model, old generated image or predicted score enters.
    features = {s: [coverage[s][f"au{a}"]["events_by_exact_intensity"][str(v)] for a in AUS for v in (1, 2, 3)] for s in subjects}
    totals = [sum(features[s][i] for s in subjects) for i in range(36)]
    rng, best = random.Random(SEED), None
    for _ in range(4000):
        order = list(subjects)
        rng.shuffle(order)
        groups = {"train": sorted(order[:17]), "dev": sorted(order[17:22]), "test_candidate": sorted(order[22:])}
        score = 0.0
        for split, group in groups.items():
            for i, total in enumerate(totals):
                n = sum(features[s][i] for s in group)
                if total:
                    score += ((n / total) - len(group) / len(subjects)) ** 2
                    if not n:
                        score += 25 if split != "train" else 50
        key = (score, tuple(groups["test_candidate"]), tuple(groups["dev"]))
        if best is None or key < best[0]:
            best = (key, groups)
    return best[1], best[0][0]


def source_anchor(vectors):
    last = max(vectors)
    pool = [n for n, values in vectors.items() if 100 <= n <= last // 3 and n % 13 == 7 and not any(values.values())]
    if not pool:
        raise ValueError("No neutral early-third grid-13 anchor; do not relax after seeing outputs")
    return min(pool, key=lambda n: stable("source", n))


def eligible_events(vectors, source):
    result = {}
    for au in AUS:
        for value in (1, 2, 3):
            frames = [n for n, vals in vectors.items() if vals[f"au{au}"] == value and max(vals.values()) <= 3 and abs(n - source) >= 24]
            result[(au, value)] = contiguous_runs(frames)
    return result


def choose_pilot_subjects(group, bins):
    eligible = [s for s in group if s in bins and all(any(bins[s][(a, v)] for a in AUS) for v in (1, 2, 3))]
    choices = []
    for pair in itertools.combinations(eligible, 2):
        cover = {(a, v) for s in pair for a in AUS for v in (1, 2, 3) if bins[s][(a, v)]}
        choices.append((-len(cover), stable("pilot_subjects", *pair), pair))
    if not choices:
        raise RuntimeError("No two pilot subjects with required manual-label coverage")
    return list(min(choices)[2])


def select_pairs(groups, bins, anchors, vectors):
    pairs = []
    for split in ("train", "dev"):
        subjects = choose_pilot_subjects(groups[split], bins)
        used_au = set()
        for subject in subjects:
            selected = []
            for slot, intensity in enumerate((1, 1, 2, 3)):
                choices = []
                for au in AUS:
                    for first, last in bins[subject][(au, intensity)]:
                        frame = (first + last) // 2
                        if any(abs(frame - old) < 24 for old in selected):
                            continue
                        choices.append((au in used_au, stable(split, subject, intensity, au, first, last), au, frame, first, last))
                if not choices:
                    raise RuntimeError("No prespecified pilot stratum: " + str((split, subject, intensity)))
                _, _, au, frame, first, last = min(choices)
                selected.append(frame)
                used_au.add(au)
                source = anchors[subject]
                pairs.append({"case_id": f"fresh_v1_{split}_{subject}_s{source:05d}_t{frame:05d}_AU{au:02d}_{intensity}",
                              "split": split, "subject": subject, "source_frame": source, "target_frame": frame,
                              "source_video_frame": source - 1, "target_video_frame": frame - 1,
                              "source_aus": vectors[subject][source], "target_aus": vectors[subject][frame],
                              "focus_au": au, "focus_intensity": intensity, "focus_event": [first, last],
                              "au_order": list(AUS), "target_au_vector_0_5": [vectors[subject][frame][f"au{a}"] for a in AUS],
                              "target_au_vector_0_1": [vectors[subject][frame][f"au{a}"] / 5 for a in AUS]})
    assert len(pairs) == 16
    assert {p["subject"] for p in pairs if p["split"] == "train"}.isdisjoint({p["subject"] for p in pairs if p["split"] == "dev"})
    return pairs


def extract_frame(video, frame, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if not file_state(video)["resident"]:
        raise RuntimeError("Video no longer resident")
    vf = f"select=eq(n\\,{frame - 1})"
    command = [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(video), "-vf", vf,
               "-frames:v", "1", "-fps_mode", "vfr", "-n", str(target)]
    p = subprocess.run(command, capture_output=True, text=True, timeout=45)
    if p.returncode or not target.exists():
        raise RuntimeError(p.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path.home() / "Desktop/Cursor/geme-main/DISFA")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    if EXPERIMENT not in out.parents:
        raise ValueError("All output must stay in the new experiment")
    out.mkdir(parents=True, exist_ok=False)
    labels = args.raw_root / "ActionUnit_Labels"
    subjects = sorted(p.name for p in labels.iterdir() if p.is_dir() and re.fullmatch(r"SN\d{3}", p.name))
    if len(subjects) != 27:
        raise ValueError("Unexpected subject count; inspect raw data first")
    vectors, coverage, label_records, video_records, videos, matching_videos = {}, {}, [], [], {}, set()
    for subject in subjects:
        v, c, records = labels_for_subject(labels, subject)
        vectors[subject], coverage[subject] = v, c
        label_records += records
        for camera, dirname in (("right", "Video_RightCamera"), ("left", "Videos_LeftCamera")):
            matches = [p for p in (args.raw_root / dirname).iterdir() if p.suffix.lower() == ".avi" and subject.lower() in p.name.lower()]
            if len(matches) != 1:
                raise ValueError("Cannot uniquely bind original video")
            record = {**probe_video(matches[0]), "subject": subject, "camera": camera}
            stream = record.get("video_stream", {})
            nb = stream.get("nb_frames", "N/A")
            record["manual_label_frames"] = len(v)
            record["frame_count_matches_labels"] = nb != "N/A" and int(nb) == len(v)
            video_records.append(record)
            if camera == "right":
                videos[subject] = matches[0]
                if record["frame_count_matches_labels"] and record["resident"]:
                    matching_videos.add(subject)
    save(out / "raw_label_sources.json", label_records)
    save(out / "raw_video_inventory.json", video_records)
    save(out / "au_coverage_by_subject.json", coverage)
    aggregate = {f"au{a}": {"frames_by_intensity": {str(v): sum(coverage[s][f"au{a}"]["frames_by_intensity"][str(v)] for s in subjects) for v in range(6)},
                           "events_by_exact_intensity": {str(v): sum(coverage[s][f"au{a}"]["events_by_exact_intensity"][str(v)] for s in subjects) for v in (1, 2, 3)},
                           "positive_events": sum(coverage[s][f"au{a}"]["positive_events"] for s in subjects)} for a in AUS}
    save(out / "au_coverage_aggregate.json", aggregate)
    groups, score = candidate_split(subjects, coverage)
    anchors, bins, exclusions = {}, {}, {}
    for subject in subjects:
        if subject not in matching_videos:
            exclusions[subject] = "Right-camera video header frame count differs from labels; no offset guessed; excluded from pilot"
            continue
        try:
            anchors[subject] = source_anchor(vectors[subject])
            bins[subject] = eligible_events(vectors[subject], anchors[subject])
        except ValueError as e:
            exclusions[subject] = str(e)
    pairs = select_pairs(groups, bins, anchors, vectors)
    protocol = {"protocol_id": TAG, "status": "candidate_subject_split_not_final_test_lock", "seed": SEED,
        "groups": groups, "split_search": {"iterations": 4000, "score": score, "basis": "original manual-label event coverage only"},
        "au_order": list(AUS), "pilot_pairs": 16, "pilot_train_subjects": sorted({x["subject"] for x in pairs if x["split"] == "train"}),
        "pilot_dev_subjects": sorted({x["subject"] for x in pairs if x["split"] == "dev"}),
        "source_rule": "manual all-zero AU; frames>=100 in first third; frame%13==7; stable SHA rank",
        "target_rule": "per-subject strata 1,1,2,3; all AU values <=3; exact-level event midpoint; spacing24; stable SHA and AU coverage",
        "pilot_source_exclusions": exclusions, "test_frames_extracted": 0, "old_generated_or_predicted_data_read": False,
        "preprocessing": "exact original right-camera full-resolution decoded frame, label index minus one; lossless PNG serialization; no crop, resize, detection model or enhancement"}
    save(out / "split_candidate.json", protocol)
    save(out / "pilot_pairs_before_extraction.json", pairs)
    # Hash only the four resident original videos needed for this 16-pair pilot.
    used = sorted({x["subject"] for x in pairs})
    hashes = {s: {**file_state(videos[s]), "sha256": sha(videos[s])} for s in used}
    save(out / "pilot_original_video_hashes.json", hashes)
    requests = {}
    for row in pairs:
        for role in ("source", "target"):
            frame = row[role + "_frame"]
            path = out / "frames" / row["split"] / row["subject"] / f"{row['subject']}_{frame:05d}.png"
            requests[(row["subject"], frame)] = path
            row[role + "_path"] = str(path)
            row[role + "_image_path"] = str(path.relative_to(EXPERIMENT))
        row["original_video_sha256"] = hashes[row["subject"]]["sha256"]
        row["original_video_path"] = str(videos[row["subject"]])
        row["camera"] = "right"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(extract_frame, videos[s], frame, path) for (s, frame), path in requests.items()]
        for f in futures:
            f.result()
    for row in pairs:
        for role in ("source", "target"):
            row[role + "_sha256"] = sha(row[role + "_path"])
    save(out / "pilot_pairs.json", {"protocol_id": TAG, "root": str(EXPERIMENT), "pairs": pairs, "count": len(pairs), "test_pairs": 0})
    save(out / "receipt.json", {"utc": datetime.now(timezone.utc).isoformat(), "source": "original DISFA manual labels and resident right-camera videos",
         "label_file_count": len(label_records), "subject_count": len(subjects), "frames_per_au": sum(len(x) for x in vectors.values()),
         "pilot_pair_count": 16, "new_extracted_frames": len(requests), "split_sha256": sha(out / "split_candidate.json"),
         "pilot_sha256": sha(out / "pilot_pairs.json"), "old_generated_or_predicted_data_read": False})
    print(json.dumps({"out": str(out), "groups": groups, "pilot_train": protocol["pilot_train_subjects"],
                      "pilot_dev": protocol["pilot_dev_subjects"], "pairs": len(pairs), "frames": len(requests)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
