#!/usr/bin/env python3
"""Execute approved real train/dev frame plan, one sequential decode per video."""
import collections
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from PIL import Image
from prepare_disfa_pilot import AUS, file_state, sha, labels_for_subject

HERE = Path(__file__).resolve().parent
PLAN = HERE / "au_supervision_v2"
EXPECTED_SPLIT = "b0837ba1ae1b30eec0dd33c48f2300809625a09437b2c161ac4576dad521fe6d"
RAW = Path.home() / "Desktop/Cursor/geme-main/DISFA"


def save_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def update_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def frame_info(path, expected_width, expected_height):
    with Image.open(path) as im:
        im.load()
        if im.mode != "RGB" or (im.width, im.height) != (expected_width, expected_height):
            raise ValueError("Decoded image is not the full original RGB frame: " + str(path))
    return {"image_sha256": sha(path), "width": expected_width, "height": expected_height,
            "mode": "RGB", "size_bytes": Path(path).stat().st_size}


def decode(job, video_source):
    split, subject = job["split"], job["subject"]
    paths = [Path(p).resolve() for p in job["outputs"]]
    frames = job["video_frames"]
    if frames != sorted(set(frames)) or any(f < 0 for f in frames):
        raise ValueError("Video frame plan is not sorted and unique")
    if any(PLAN.resolve() not in p.parents for p in paths):
        raise ValueError("Frame output escaped approved plan directory")
    video = Path(job["video_path"])
    if not file_state(video)["resident"]:
        raise RuntimeError("Original video not resident")
    if sha(video) != job["original_video_sha256"]:
        raise ValueError("Original video changed since planning")
    receipt_path = PLAN / "decode_receipts" / f"{split}_{subject}.json"
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if prior["video_sha256"] != job["original_video_sha256"] or prior["video_frames"] != frames:
            raise ValueError("Saved decode receipt has different source/frames")
        for record in prior["frames"]:
            if sha(record["image_path"]) != record["image_sha256"]:
                raise ValueError("Previously extracted frame changed")
        return prior
    temp_root = PLAN / ".decode_tmp"
    temp_root.mkdir(exist_ok=True)
    width = int(video_source["video_stream"]["width"])
    height = int(video_source["video_stream"]["height"])
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"{split}_{subject}_", dir=temp_root) as temp:
        temp = Path(temp)
        vf = "select=" + "+".join(f"eq(n\\,{frame})" for frame in frames)
        command = [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-threads", "1",
                   "-i", str(video), "-vf", vf, "-frames:v", str(len(frames)), "-fps_mode", "vfr",
                   "-start_number", "1", "-n", str(temp / "frame_%06d.png")]
        p = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if p.returncode:
            raise RuntimeError(f"ffmpeg failed {subject}: {p.stderr}")
        actual = sorted(temp.glob("frame_*.png"))
        if len(actual) != len(frames):
            raise ValueError(f"Frame count mismatch {subject}: {len(actual)} vs {len(frames)}")
        records = []
        for source, destination, label_frame, video_frame in zip(actual, paths, job["label_frames"], frames):
            info = frame_info(source, width, height)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha(destination) != info["image_sha256"]:
                    raise ValueError("Do not overwrite an existing different raw frame")
            else:
                with source.open("rb") as src, destination.open("xb") as target:
                    shutil.copyfileobj(src, target)
            if sha(destination) != info["image_sha256"]:
                raise ValueError("Frame copy checksum mismatch")
            records.append({"split": split, "subject": subject, "camera": job["camera"],
                            "label_frame": label_frame, "video_frame": video_frame, "image_path": str(destination), **info})
    result = {"status": "pass", "split": split, "subject": subject, "camera": job["camera"], "video_path": str(video),
              "video_sha256": job["original_video_sha256"], "video_frames": frames, "frame_count": len(records),
              "sequential_decode_passes": 1, "frames": records, "elapsed_seconds": time.monotonic() - start,
              "command": command[:-1] + ["<task-local-temporary-dir>/frame_%06d.png"]}
    save_new(receipt_path, result)
    print(json.dumps({"split": split, "subject": subject, "frames": len(records), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


def main():
    if sha(PLAN / "subjects_frozen.json") != EXPECTED_SPLIT:
        raise ValueError("Frozen subject split changed")
    subjects = json.loads((PLAN / "subjects_frozen.json").read_text())["groups"]
    assert [len(subjects[k]) for k in ("train", "dev", "test")] == [15, 4, 8]
    manifests = {s: json.loads((PLAN / f"{s}_manifest_planned.json").read_text()) for s in ("train", "dev")}
    jobs = json.loads((PLAN / "decode_jobs_planned.json").read_text())["jobs"]
    sources = json.loads((PLAN / "train_dev_video_sources_hashed.json").read_text())
    for job in jobs:
        if job["split"] not in ("train", "dev") or job["subject"] not in subjects[job["split"]] or job["subject"] in subjects["test"]:
            raise ValueError("Test/unknown subject in decoding plan")
    if len(jobs) != 19:
        raise ValueError("Expected one decode job per 19 train/dev subjects")
    # Re-read raw manual labels and validate every planned label before extraction.
    raw_labels = {s: labels_for_subject(RAW / "ActionUnit_Labels", s)[0] for s in subjects["train"] + subjects["dev"]}
    for split, manifest in manifests.items():
        expected = 600 if split == "train" else 240
        assert manifest["subjects_frozen_sha256"] == EXPECTED_SPLIT and manifest["count"] == expected
        assert len(manifest["samples"]) == expected
        for sample in manifest["samples"]:
            assert sample["subject"] in subjects[split] and sample["split"] == split
            assert sample["video_frame"] == sample["label_frame"] - 1
            assert sample["labels"] == raw_labels[sample["subject"]][sample["label_frame"]]
            assert sample["au_vector_0_5"] == [sample["labels"][f"au{a}"] for a in AUS]
    binding = {"subjects_frozen_sha256": EXPECTED_SPLIT, "planner_sha256": sha(HERE / "plan_au_supervision_v2.py"),
               "extractor_sha256": sha(__file__), "planned_manifest_sha256": {s: sha(PLAN / f"{s}_manifest_planned.json") for s in manifests},
               "decode_jobs_sha256": sha(PLAN / "decode_jobs_planned.json"),
               "ffmpeg_version": subprocess.check_output([shutil.which("ffmpeg"), "-version"], text=True).splitlines()[0],
               "test_subjects": subjects["test"], "test_decode_jobs": 0, "workers": 2,
               "pixel_processing": "full-frame video decode to RGB PNG only; no resize/crop/enhancement/model"}
    bind_path = PLAN / "extraction_binding.json"
    if bind_path.exists():
        if json.loads(bind_path.read_text()) != binding:
            raise ValueError("Extraction binding changed; inspect rather than overwrite")
    else:
        save_new(bind_path, binding)
    lock = PLAN / ".extract.lock"
    with lock.open("x") as f:
        json.dump({"pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat()}, f)
    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(decode, job, sources[job["subject"]]) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                update_json(PLAN / "extraction_progress.json", {"completed_jobs": len(results), "total_jobs": len(jobs),
                            "completed_frames": sum(x["frame_count"] for x in results), "test_frames": 0})
        records = {(x["subject"], frame["label_frame"]): frame for x in results for frame in x["frames"]}
        for split, manifest in manifests.items():
            ready = []
            for sample in manifest["samples"]:
                extracted = records[(sample["subject"], sample["label_frame"])]
                assert Path(sample["image_path"]).resolve() == Path(extracted["image_path"]).resolve()
                ready.append({**sample, **extracted, "extracted": True})
            final = {**manifest, "frames_extracted": len(ready), "samples": ready,
                     "planned_manifest_sha256": binding["planned_manifest_sha256"][split],
                     "extraction_binding_sha256": sha(bind_path)}
            final_path = PLAN / f"{split}_manifest_extracted.json"
            if final_path.exists():
                if json.loads(final_path.read_text()) != final:
                    raise ValueError("Existing extracted manifest differs")
            else:
                save_new(final_path, final)
        receipt = {"status": "pass", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "subjects_frozen_sha256": EXPECTED_SPLIT,
                   "train_frames": 600, "dev_frames": 240, "test_frames": 0, "decoded_videos": len(results), "sequential_decode_passes": len(results),
                   "frame_count": len(records), "label_source": "original manual AU files rechecked for all 840 frames",
                   "train_manifest_sha256": sha(PLAN / "train_manifest_extracted.json"), "dev_manifest_sha256": sha(PLAN / "dev_manifest_extracted.json"),
                   "raw_labels_preserved": True, "original_data_modified": False, "model_runs": 0}
        save_new(PLAN / "extraction_receipt.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    except Exception as error:
        update_json(PLAN / "extraction_failure.json", {"error_type": type(error).__name__, "error": str(error),
                    "completed_jobs": len(results), "test_frames": 0})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
