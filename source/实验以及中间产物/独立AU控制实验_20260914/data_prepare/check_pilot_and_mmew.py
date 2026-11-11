#!/usr/bin/env python3
"""Verify new pilot pixels/labels; inventory only real MMEW source frames."""
import collections
import hashlib
import json
from pathlib import Path
import re

from PIL import Image
from prepare_disfa_pilot import AUS, EXPERIMENT, file_state, save, sha

HERE = Path(__file__).resolve().parent
PILOT = HERE / "pilot_v1_02"
doc = json.loads((PILOT / "pilot_pairs.json").read_text())
rows = doc["pairs"]
assert len(rows) == 16
assert collections.Counter(x["split"] for x in rows) == {"train": 8, "dev": 8}
subjects = {split: sorted({x["subject"] for x in rows if x["split"] == split}) for split in ("train", "dev")}
assert set(subjects["train"]).isdisjoint(subjects["dev"])
images = {}
for row in rows:
    assert row["source_video_frame"] == row["source_frame"] - 1
    assert row["target_video_frame"] == row["target_frame"] - 1
    assert set(row["source_aus"]) == {f"au{a}" for a in AUS}
    assert set(row["target_aus"]) == {f"au{a}" for a in AUS}
    assert max(row["target_aus"].values()) <= 3
    assert row["target_aus"][f"au{row['focus_au']}"] == row["focus_intensity"]
    for role in ("source", "target"):
        path = Path(row[role + "_path"])
        assert EXPERIMENT in path.parents
        assert sha(path) == row[role + "_sha256"]
        if str(path) not in images:
            with Image.open(path) as image:
                image.load()
                images[str(path)] = {"width": image.width, "height": image.height, "mode": image.mode}
                assert (image.width, image.height) == (1024, 768), "Expected original full frame, not central crop"
                assert image.mode == "RGB"
qa = {"status": "pass", "pairs": len(rows), "unique_full_frames": len(images), "subjects": subjects,
      "intensity_counts": dict(collections.Counter(x["focus_intensity"] for x in rows)),
      "focus_aus": sorted({x["focus_au"] for x in rows}), "frame_dimensions": [1024, 768],
      "all_target_labels_max_3": True, "source_labels_read_not_imputed": True,
      "models_loaded": False, "test_frames_extracted": 0, "images": images}
save(PILOT / "qa.json", qa)
print(json.dumps({k: v for k, v in qa.items() if k != "images"}, ensure_ascii=False, indent=2))

# This is a fresh filesystem inventory, not an old experiment prediction/feature table.
old_data = EXPERIMENT.parent / "视频级微表情识别实验/data"
source = old_data / "processed/source/MMEW"
archive = old_data / "raw/peerj_cs_1140/raw_data3.rar"
sequence_rows, frame_rows = [], []
for category in sorted(source.iterdir()):
    if not category.is_dir():
        continue
    for seq in sorted(category.iterdir()):
        if not seq.is_dir():
            continue
        match = re.fullmatch(r"(S\d+)-\d+-\d+", seq.name)
        if not match:
            raise ValueError("Unknown real-source sequence naming: " + seq.name)
        files = sorted(seq.glob("*.jpg"), key=lambda p: int(p.stem))
        assert [int(p.stem) for p in files] == list(range(1, 17))
        states = [file_state(p) for p in files]
        for state in states:
            state.update(subject=match[1], sequence=seq.name, category=category.name)
        first = states[0]
        entry = {"subject": match[1], "sequence": seq.name, "category_from_source_folder": category.name,
                 "source_folder": str(seq), "frames": len(files), "resident_frames": sum(s["resident"] for s in states),
                 "first_frame": first, "numeric_au_ground_truth": None}
        if first["resident"]:
            entry["first_frame_sha256"] = sha(first["path"])
        sequence_rows.append(entry)
        frame_rows.extend(states)
by_subject = {}
for subject in sorted({x["subject"] for x in sequence_rows}):
    subset = [x for x in sequence_rows if x["subject"] == subject]
    by_subject[subject] = {"sequences": len(subset), "frames": sum(x["frames"] for x in subset),
                           "resident_frames": sum(x["resident_frames"] for x in subset),
                           "resident_first_frames": sum(x["first_frame"]["resident"] for x in subset)}
inventory = {"source_root": str(source), "archive": file_state(archive),
    "archive_provenance_locator": "PeerJ Computer Science cs-1140 supplementary raw_data3.rar (per stored source configuration)",
    "asset_kind": "public preprocessed 16-frame sequences, not located original full videos",
    "counts": {"sequences": len(sequence_rows), "subjects": len(by_subject), "frames": len(frame_rows),
               "resident_frames": sum(x["resident"] for x in frame_rows),
               "resident_first_frames": sum(x["first_frame"]["resident"] for x in sequence_rows),
               "subjects_with_resident_first": sum(x["resident_first_frames"] > 0 for x in by_subject.values())},
    "subject_identity_basis": "Sxx prefix of original extracted sequence-folder name; no model identity prediction",
    "numeric_au_labels_found": False, "generated_image_ground_truth_claimed": False,
    "test_selection_frozen": False, "test_images_copied_or_generated": False,
    "old_prediction_or_generated_files_read": False, "subjects": by_subject, "sequences": sequence_rows}
save(HERE / "mmew_source_inventory.json", inventory)
save(HERE / "mmew_frame_stat.json", frame_rows)
print(json.dumps({"MMEW": inventory["counts"], "archive": inventory["archive"]}, ensure_ascii=False, indent=2))
