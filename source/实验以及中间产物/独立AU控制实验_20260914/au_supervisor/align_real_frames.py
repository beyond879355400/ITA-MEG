#!/usr/bin/env python3
"""CPU detector-only canonical112-scaled512 alignment of the fixed 840 frames."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DATA = EXPERIMENT / "data_prepare/au_supervision_v2"
SPLIT_SHA = "b0837ba1ae1b30eec0dd33c48f2300809625a09437b2c161ac4576dad521fe6d"
INPUT_SHA = {"train": "96fed05699d294f43d157fc9e02d460d2f28e52d0c363d522adec84aea1445ff",
             "dev": "9c87420076417b7eaed6d41b4c011bc0265e5adcc9685f3f34411ab071e616e1"}
DETECTOR_SHA = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"


def environment():
    for key, sub in {"PYTHONPYCACHEPREFIX": "pycache", "TORCH_HOME": "torch", "HF_HOME": "huggingface",
                     "MPLCONFIGDIR": "matplotlib", "XDG_CACHE_HOME": "xdg", "TMPDIR": "tmp"}.items():
        p = HERE / "cache" / sub
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      NO_ALBUMENTATIONS_UPDATE="1", PYDANTIC_DISABLE_PLUGINS="__all__", MPLBACKEND="Agg")
    sys.dont_write_bytecode = True
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, data, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with path.open("x", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    else:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)


def copy_exact(source, target, expected):
    if sha(source) != expected:
        raise ValueError("Source checksum mismatch")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with Path(source).open("rb") as inp, target.open("xb") as out:
            shutil.copyfileobj(inp, out)
    if sha(target) != expected:
        raise ValueError("Destination checksum mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=HERE / "alignment_v1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    output = args.output.resolve()
    if HERE not in output.parents:
        raise ValueError("Alignment output outside AU supervisor")
    environment()
    import numpy as np
    import cv2
    from PIL import Image, ImageOps
    from insightface.app import FaceAnalysis
    from insightface.utils.face_align import estimate_norm
    if sha(DATA / "subjects_frozen.json") != SPLIT_SHA:
        raise ValueError("Subject lock mismatch")
    split = json.loads((DATA / "subjects_frozen.json").read_text())["groups"]
    rows = []
    for group in ("train", "dev"):
        file = DATA / f"{group}_manifest_extracted.json"
        if sha(file) != INPUT_SHA[group]:
            raise ValueError("Approved input manifest changed")
        part = json.loads(file.read_text())["samples"]
        for row in part:
            if row["split"] != group or row["subject"] not in split[group] or row["subject"] in split["test"]:
                raise ValueError("Subject leak or unexpected split")
        rows.extend(part)
    assert len(rows) == 840
    detector_source = EXPERIMENT.parent / "baselines/ganimation_emotionnet_20260718/evaluation_models/insightface/models/buffalo_l/det_10g.onnx"
    model_root = HERE / "models/detector_only"
    detector_path = model_root / "models/buffalo_l/det_10g.onnx"
    copy_exact(detector_source, detector_path, DETECTOR_SHA)
    if sorted(p.name for p in detector_path.parent.glob("*.onnx")) != ["det_10g.onnx"]:
        raise ValueError("Detector directory must not contain identity or other networks")
    recipe = {"subject_split_sha256": SPLIT_SHA, "input_manifest_sha256": INPUT_SHA, "aligner_sha256": sha(__file__),
              "reference_cache_builder_sha256": sha(EXPERIMENT / "prepare_training_cache.py"),
              "detector_sha256": DETECTOR_SHA, "detector_path": str(detector_path),
              "alignment": "estimate_norm(kps,112) affine * (512/112); warp original BGR to512 INTER_LINEAR",
              "detection": {"provider": "CPUExecutionProvider", "input_size": [640, 640], "thresholds": [.5, .3, .15, .1],
                            "variants": ["original", "autocontrast_cutoff1_detection_only"], "selection": "maximum_bbox_area"},
              "loaded_model_roles": ["detection"], "vae_loaded": False, "identity_network_loaded": False,
              "packages": {n: importlib.metadata.version(n) for n in ("numpy", "Pillow", "opencv-python", "insightface", "onnxruntime")}}
    if output.exists():
        if not args.resume or json.loads((output / "binding.json").read_text()) != recipe:
            raise ValueError("Existing alignment requires exact resume")
    else:
        output.mkdir(parents=True)
        write_json(output / "binding.json", recipe)
        copy_exact(__file__, output / "aligner_snapshot.py", recipe["aligner_sha256"])
        for group in ("train", "dev"):
            copy_exact(DATA / f"{group}_manifest_extracted.json", output / f"input_{group}.json", INPUT_SHA[group])
        copy_exact(DATA / "subjects_frozen.json", output / "subjects_frozen.json", SPLIT_SHA)
    face = FaceAnalysis(name="buffalo_l", root=str(model_root), allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    face.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=.15)
    if set(face.models) != {"detection"}:
        raise RuntimeError("Unexpected loaded model role")
    results = []
    started = time.monotonic()
    for index, row in enumerate(rows):
        sample = row["sample_id"]
        if not re.fullmatch(r"au_sup_v2_(train|dev)_SN\d{3}_f\d+", sample):
            raise ValueError("Unsafe sample id")
        record_path = output / "records" / (sample + ".json")
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record["input"] != row:
                raise ValueError("Resumed sample changed")
            if record["status"] == "aligned" and sha(record["aligned_path"]) != record["aligned_sha256"]:
                raise ValueError("Resumed aligned image changed")
            results.append(record)
            continue
        raw = Path(row["image_path"]).resolve()
        if EXPERIMENT not in raw.parents or sha(raw) != row["image_sha256"]:
            raise ValueError("Raw input changed: " + sample)
        record = {"input": row, "sample_id": sample, "split": row["split"], "subject": row["subject"],
                  "raw_path": str(raw), "raw_sha256": row["image_sha256"], "detection_attempts": []}
        try:
            rgb = np.asarray(Image.open(raw).convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            detected = []
            for variant in ("original", "autocontrast_cutoff1_detection_only"):
                detector_rgb = rgb if variant == "original" else np.asarray(ImageOps.autocontrast(Image.fromarray(rgb), cutoff=1))
                for threshold in (.5, .3, .15, .1):
                    face.det_model.det_thresh = threshold
                    found = face.get(cv2.cvtColor(detector_rgb, cv2.COLOR_RGB2BGR))
                    record["detection_attempts"].append({"variant": variant, "threshold": threshold, "candidates": len(found)})
                    if found:
                        detected = found
                        break
                if detected:
                    break
            if not detected:
                raise RuntimeError("No face detected; fixed sample retained as failed")
            chosen = max(detected, key=lambda x: float((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])))
            matrix = estimate_norm(chosen.kps, image_size=112) * (512. / 112.)
            if not np.isfinite(matrix).all():
                raise ValueError("Nonfinite alignment")
            aligned = cv2.warpAffine(bgr, matrix, (512, 512), flags=cv2.INTER_LINEAR)
            path = output / "aligned" / row["split"] / (sample + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise ValueError("Unrecorded existing aligned output; do not overwrite")
            if not cv2.imwrite(str(path), aligned):
                raise RuntimeError("PNG save failed")
            if sha(raw) != row["image_sha256"]:
                raise ValueError("Raw input altered during alignment")
            record.update(status="aligned", aligned_path=str(path), aligned_sha256=sha(path),
                          affine_original_to_512=matrix.tolist(), face_bbox=chosen.bbox.tolist(),
                          face_kps=chosen.kps.tolist(), detection_score=float(chosen.det_score), width=512, height=512)
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        finally:
            face.det_model.det_thresh = .15
        write_json(record_path, record)
        results.append(record)
        if (index + 1) % 25 == 0:
            counts = {g: {s: sum(x["split"] == g and x["status"] == s for x in results) for s in ("aligned", "failed")} for g in ("train", "dev")}
            write_json(output / "progress.json", {"completed": len(results), "total": 840, "counts": counts}, replace=True)
            print(json.dumps({"completed": len(results), "counts": counts}), flush=True)
    manifest = {"status": "complete", "binding_sha256": sha(output / "binding.json"), "subject_split_sha256": SPLIT_SHA,
                "input_manifest_sha256": INPUT_SHA, "count": 840, "test_count": 0,
                "counts": {g: {s: sum(x["split"] == g and x["status"] == s for x in results) for s in ("aligned", "failed")} for g in ("train", "dev")},
                "elapsed_seconds": time.monotonic() - started, "samples": results}
    write_json(output / "aligned_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
