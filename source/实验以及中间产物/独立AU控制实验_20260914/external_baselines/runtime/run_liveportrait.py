#!/usr/bin/env python3
"""Pinned LivePortrait human image-driven inference; preflight is stdlib-only."""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from liveportrait_detection_policy import DetectorPolicy, THRESHOLDS

ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent.parent
COMMIT = "9b294b3d0536135442ea73cb01e6cb3ca7029dd3"
MODEL_REVISION = "82a4fa6735ca58432b6ce39301b4b9ee066dea47"
ASSET_MANIFEST = ROOT / "second_baseline_review/liveportrait_required_assets.json"
PARAMETERS = (
    "flag_use_half_precision", "flag_crop_driving_video", "flag_force_cpu", "flag_normalize_lip",
    "flag_source_video_eye_retargeting", "flag_eye_retargeting", "flag_lip_retargeting", "flag_stitching",
    "flag_relative_motion", "flag_pasteback", "flag_do_crop", "driving_option", "driving_multiplier",
    "driving_smooth_observation_variance", "animation_region", "det_thresh", "scale", "vx_ratio", "vy_ratio",
    "flag_do_rot", "source_max_dim", "source_division", "scale_crop_driving_video", "vx_ratio_crop_driving_video",
    "vy_ratio_crop_driving_video", "flag_do_torch_compile", "device_id",
)
PACKAGES = ("torch", "torchvision", "numpy", "scipy", "Pillow", "opencv-python", "onnxruntime", "onnx",
            "scikit-image", "imageio", "PyYAML", "rich", "tyro", "pykalman")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temporary, path)


class Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile = stream, logfile
    def write(self, value):
        self.stream.write(value)
        self.logfile.write(value)
        self.logfile.flush()
        return len(value)
    def flush(self):
        self.stream.flush()
        self.logfile.flush()
    def isatty(self):
        return False


def result_exit_code(results):
    return 1 if any(x.get("status") != "success" for x in results) else 0


def readiness(path):
    path = Path(path)
    if not path.is_file():
        return "missing"
    s = path.stat()
    if s.st_size == 0:
        return "empty"
    if getattr(s, "st_flags", 0) & 0x40000000 or (s.st_size > 0 and s.st_blocks == 0):
        return "dataless"
    if path.name.endswith((".part", ".tmp", ".incomplete")):
        return "partial"
    return None


def snapshot(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if readiness(source) or sha(source) != expected:
        raise ValueError("Snapshot source not ready or changed: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != expected:
            raise ValueError("Existing snapshot changed: " + str(destination))
        return
    with source.open("rb") as src, destination.open("xb") as out:
        shutil.copyfileobj(src, out)
    if sha(destination) != expected:
        raise ValueError("Snapshot copy mismatch")


def load_cases(path, selected=None):
    path = Path(path)
    doc = json.loads(path.read_text())
    rows = doc["cases"] if isinstance(doc, dict) else doc
    seen, result = set(), []
    for original in rows:
        row = copy.deepcopy(original)
        name = row.get("case_id", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
            raise ValueError("Unsafe case id")
        if name in seen:
            raise ValueError("Duplicate case id: " + name)
        seen.add(name)
        if selected and name not in selected:
            continue
        for key, original_key in (("source", "source_image_path"), ("target", "target_image_path")):
            raw = row.get(key, row.get(original_key))
            if not isinstance(raw, str):
                raise ValueError("Missing input path")
            file = Path(raw)
            if not file.is_absolute():
                file = (THESIS if original_key in row else path.parent) / file
            file = file.resolve()
            if readiness(file):
                raise ValueError("Input not resident: " + str(file))
            actual = sha(file)
            expected = row.get(key + "_sha256") or row.get(key + "_sha256_expected")
            if expected and expected != actual:
                raise ValueError("Input hash mismatch: " + name + "/" + key)
            row[key], row[key + "_sha256"] = str(file), actual
        row["seed"] = int(row.get("seed", 424))
        result.append(row)
    if selected and set(selected) - seen:
        raise ValueError("Selected case absent")
    if not result:
        raise ValueError("Empty case selection")
    return result


def author_parameters(repo, device="mps"):
    tree = ast.parse((Path(repo) / "src/config/argument_config.py").read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "ArgumentConfig")
    values = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in PARAMETERS:
            values[node.target.id] = ast.literal_eval(node.value)
    if set(values) != set(PARAMETERS):
        raise ValueError("Author argument defaults changed")
    # Supplied driving PNGs are whole-frame square crops, not aligned portraits.
    values["flag_crop_driving_video"] = True
    values["flag_do_torch_compile"] = False
    if device == "cpu":
        values["flag_force_cpu"] = True
        values["flag_use_half_precision"] = False
    return values


def git_check(repo):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if commit != COMMIT or dirty:
        raise ValueError("Author repo must be clean and pinned")
    return commit


def model_inventory(checkpoint):
    doc = json.loads(ASSET_MANIFEST.read_text())
    if doc["official_model_revision"] != MODEL_REVISION or len(doc["assets"]) != 8:
        raise ValueError("Official model plan changed")
    records, missing = [], []
    for entry in doc["assets"]:
        name = entry["filename"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe asset path")
        p = Path(checkpoint) / name
        issue = readiness(p)
        if issue:
            missing.append({"path": str(p), "reason": issue})
            continue
        actual = sha(p)
        if p.stat().st_size != entry["expected_bytes"] or actual != entry["sha256"]:
            raise ValueError("Model SHA/size mismatch: " + name)
        records.append({"relative_name": name, "path": str(p), "size_bytes": p.stat().st_size, "sha256": actual})
    return records, missing


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def source_files(repo):
    required = ("src/live_portrait_pipeline.py", "src/live_portrait_wrapper.py", "src/config/models.yaml",
                "src/config/argument_config.py", "src/config/inference_config.py", "src/config/crop_config.py",
                "src/utils/resources/mask_template.png", "src/utils/resources/lip_array.pkl",
                "src/utils/human_landmark_runner.py", "src/utils/face_analysis_diy.py")
    records = []
    for name in required:
        p = Path(repo) / name
        if readiness(p):
            raise ValueError("Required author source/resource missing: " + str(p))
        records.append({"relative_name": name, "path": str(p), "size_bytes": p.stat().st_size, "sha256": sha(p)})
    return records


class CaptureOutput:
    """Save the author's pre-JPEG pixels, never decode/select a concatenation."""
    def __init__(self, module, native_path, png_path, output_dir):
        self.module, self.native = module, Path(native_path).resolve()
        self.png, self.directory = Path(png_path).resolve(), Path(output_dir).resolve()
        self.captured = 0

    def __enter__(self):
        self.original = self.module.imwrite
        def write(path, array, *args, **kwargs):
            path = Path(path).resolve()
            if self.directory not in path.parents:
                raise ValueError("Author image write escaped case directory")
            if path == self.native:
                if self.captured or self.png.exists():
                    raise ValueError("Repeated primary output")
                if not self.original(str(self.png), array):
                    raise RuntimeError("Could not serialize raw generated PNG")
                self.captured += 1
            saved = self.original(str(path), array, *args, **kwargs)
            if not saved:
                raise RuntimeError("Author image save failed: " + str(path))
            return saved
        self.module.imwrite = write
        return self

    def __exit__(self, *exc):
        self.module.imwrite = self.original


def synchronize(torch, device):
    if device == "mps":
        torch.mps.synchronize()


def bind_cpu_fp32_context(wrapper, device, half_precision):
    if device == "cpu" and not half_precision:
        # torch 2.1 validates an unsupported CPU fp16 dtype even when disabled.
        # This keeps the requested fp32 path without any autocast operation.
        wrapper.inference_ctx = contextlib.nullcontext
        return True
    return False


def initialize(repo, checkpoint, parameters, detection_policy, device="mps"):
    # Imports and all model initialization occur only in --execute.
    sys.path.insert(0, str(repo))
    import torch
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    import src.live_portrait_pipeline as author_pipeline
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("This wrapper is bound to the author's Mac/MPS route")
    original_args = ArgumentConfig()
    for key, value in parameters.items():
        setattr(original_args, key, value)
    ic = InferenceConfig(**{k: v for k, v in vars(original_args).items() if hasattr(InferenceConfig, k)})
    cc = CropConfig(**{k: v for k, v in vars(original_args).items() if hasattr(CropConfig, k)})
    for key, name in (("checkpoint_F", "appearance_feature_extractor.pth"), ("checkpoint_M", "motion_extractor.pth"),
                      ("checkpoint_G", "spade_generator.pth"), ("checkpoint_W", "warping_module.pth")):
        setattr(ic, key, str(checkpoint / "liveportrait/base_models" / name))
    ic.checkpoint_S = str(checkpoint / "liveportrait/retargeting_models/stitching_retargeting_module.pth")
    cc.insightface_root = str(checkpoint / "insightface")
    cc.landmark_ckpt_path = str(checkpoint / "liveportrait/landmark.onnx")
    if device == "cpu":
        # Upstream Pipeline does not forward crop_cfg.flag_force_cpu to Cropper.
        # Pass the author's existing CPU option; no crop or model math changes.
        cropper_class = author_pipeline.Cropper
        def cpu_cropper(*args, **kwargs):
            kwargs["flag_force_cpu"] = True
            return cropper_class(*args, **kwargs)
        with patch.object(author_pipeline, "Cropper", cpu_cropper):
            pipeline = author_pipeline.LivePortraitPipeline(inference_cfg=ic, crop_cfg=cc)
    else:
        pipeline = author_pipeline.LivePortraitPipeline(inference_cfg=ic, crop_cfg=cc)
    if pipeline.live_portrait_wrapper.device != device:
        raise RuntimeError("Unexpected generation device")
    cpu_fp32_context = bind_cpu_fp32_context(pipeline.live_portrait_wrapper, device, parameters["flag_use_half_precision"])
    detector = DetectorPolicy(pipeline.cropper.face_analysis_wrapper.det_model, detection_policy)
    pipeline.cropper.face_analysis_wrapper.det_model = detector
    runtime = {"generation_device": device,
               "generation_dtype": str(next(pipeline.live_portrait_wrapper.appearance_feature_extractor.parameters()).dtype),
               "cpu_fp32_nullcontext": cpu_fp32_context,
               "detection_providers": detector.session.get_providers(),
               "landmark_providers": pipeline.cropper.human_landmark_runner.session.get_providers(),
               "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}
    return pipeline, detector, original_args, torch, runtime


def execute_case(row, output, pipeline, detector, base_args, torch, device="mps"):
    case_dir = output / "cases" / row["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (case_dir / f"attempt_{attempt:03d}").exists():
        attempt += 1
    work = case_dir / f"attempt_{attempt:03d}"
    inputs, author_output = work / "inputs", work / "author_output"
    inputs.mkdir(parents=True)
    author_output.mkdir()
    staged = {}
    for key, stem in (("source", "source"), ("target", "driving")):
        path = inputs / (stem + Path(row[key]).suffix.lower())
        snapshot(row[key], path, row[key + "_sha256"])
        staged[key] = path
    args = copy.deepcopy(base_args)
    args.source, args.driving, args.output_dir = str(staged["source"]), str(staged["target"]), str(author_output)
    expected_native = author_output / f"{staged['source'].stem}--{staged['target'].stem}.jpg"
    raw_png = author_output / "generated.png"
    detector.begin_case(row["case_id"])
    result = {"input": row, "attempt": attempt, "case_id": row["case_id"]}
    start = time.monotonic()
    try:
        import cv2
        from PIL import Image
        synchronize(torch, device)
        with CaptureOutput(cv2, expected_native, raw_png, author_output) as capture:
            native, concat = pipeline.execute(args)
        synchronize(torch, device)
        if Path(native).resolve() != expected_native.resolve() or capture.captured != 1:
            raise ValueError("Missing or unexpected genuine non-concat image")
        with Image.open(raw_png) as image:
            image.load()
            size = list(image.size)
            if len(image.getbands()) != 3:
                raise ValueError("Generated image is not RGB")
        with Image.open(staged["source"]) as source_image:
            if size != list(source_image.size):
                raise ValueError("Output size is not one source-sized pasteback image; reject possible concat")
        for key in ("source", "target"):
            if sha(row[key]) != row[key + "_sha256"] or sha(staged[key]) != row[key + "_sha256"]:
                raise ValueError("Input was modified during generation")
        generated = case_dir / "generated.png"
        snapshot(raw_png, generated, sha(raw_png))
        result.update(status="success", output=str(generated), output_sha256=sha(generated), output_size=size,
                      native_output=str(native), concat_output=str(concat), driver_template=str(staged["target"].with_suffix(".pkl")))
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    result["elapsed_seconds"] = time.monotonic() - start
    result["detection_events"] = detector.events
    save_json(work / "result.json", result)
    save_json(case_dir / "result.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-id", action="append")
    p.add_argument("--detection-policy", choices=("author_default", "shared_fallback"), default="shared_fallback")
    p.add_argument("--device", choices=("cpu", "mps"), default="mps")
    p.add_argument("--resume", action="store_true")
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("repo", "checkpoint", "manifest", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if ROOT not in args.output.parents:
        raise ValueError("Output must stay in this experiment folder")
    git_check(args.repo)
    rows = load_cases(args.manifest, args.case_id)
    parameters = author_parameters(args.repo, args.device)
    files = source_files(args.repo)
    models, missing = model_inventory(args.checkpoint)
    versions = package_versions()
    report = {"method": "liveportrait", "case_count": len(rows), "missing_models": missing,
              "packages": versions, "missing_packages": [k for k, v in versions.items() if v is None],
              "author_macos_torch": "2.3.0", "actual_torch": versions["torch"], "parameters": parameters}
    if args.preflight:
        args.output.mkdir(parents=True, exist_ok=False)
        save_json(args.output / "preflight.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if not missing and not report["missing_packages"] else 2
    if missing or report["missing_packages"]:
        raise RuntimeError("Preflight has missing inputs/dependencies: " + json.dumps(report, ensure_ascii=False))
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        raise RuntimeError("Use the author's PYTORCH_ENABLE_MPS_FALLBACK=1")
    helper = ROOT / "liveportrait_detection_policy.py"
    configuration = {"method": "liveportrait", "upstream_commit": COMMIT, "model_revision": MODEL_REVISION,
        "wrapper_sha256": sha(__file__), "helper_sha256": sha(helper), "manifest_sha256": sha(args.manifest),
        "asset_manifest_sha256": sha(ASSET_MANIFEST), "model_files": models, "source_files": files, "cases": rows,
        "randomness_policy": "deterministic; seed is frozen-pairing key only", "packages": versions,
        "policy": {"mode": "human_static_image_driven", "device": args.device, "effective_author_cli_parameters": parameters,
            "cpu_device_binding": "forward author's flag_force_cpu to Cropper; fp32 and no MPS synchronization" if args.device == "cpu" else None,
            "cpu_fp32_context": "nullcontext replaces disabled fp16 autocast on torch2.1; no dtype conversion" if args.device == "cpu" else None,
            "input_preprocessing_adaptation": "author driving crop enabled for square whole-frame targets",
            "detection_policy": args.detection_policy, "fallback_thresholds": list(THRESHOLDS),
            "contrast_use": "detection only; landmark and generation receive original RGB",
            "serialization": "lossless PNG from pre-JPEG non-concat author pasteback pixels; author JPEG retained",
            "timing": "single-case author preprocessing/inference/PNG and JPEG save; excludes loading/evaluation",
            "training": False, "torch_compile": False}}
    config_hash = signature(configuration)
    if args.output.exists():
        if not args.resume:
            raise ValueError("Output exists; choose a new run or explicit resume")
        old = json.loads((args.output / "configuration.json").read_text())
        if old != configuration:
            raise ValueError("Resume source/model/parameter/code/runtime mismatch")
    else:
        args.output.mkdir(parents=True)
        save_json(args.output / "configuration.json", configuration)
    snapshots = args.output / "snapshots"
    snapshot(__file__, snapshots / "run_liveportrait.py", configuration["wrapper_sha256"])
    snapshot(helper, snapshots / helper.name, configuration["helper_sha256"])
    snapshot(args.manifest, snapshots / "frozen_cases.json", configuration["manifest_sha256"])
    snapshot(ASSET_MANIFEST, snapshots / "model_assets.json", configuration["asset_manifest_sha256"])
    lock = args.output / ".run.lock"
    with lock.open("x") as f:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, f)
    results = []
    streams = contextlib.ExitStack()
    logfile = streams.enter_context((args.output / "run.log").open("a" if args.resume else "x", encoding="utf-8"))
    streams.enter_context(contextlib.redirect_stdout(Tee(sys.stdout, logfile)))
    streams.enter_context(contextlib.redirect_stderr(Tee(sys.stderr, logfile)))
    try:
        started = time.monotonic()
        pipeline, detector, base_args, torch, runtime = initialize(args.repo, args.checkpoint, parameters, args.detection_policy, args.device)
        save_json(args.output / "runtime.json", {**runtime, "initialization_seconds": time.monotonic() - started,
                  "configuration_sha256": config_hash, "python": sys.version, "python_executable": sys.executable})
        for row in rows:
            result_file = args.output / "cases" / row["case_id"] / "result.json"
            if args.resume and result_file.exists():
                existing = json.loads(result_file.read_text())
                if existing.get("input") != row:
                    raise ValueError("Resumed case input differs")
                if existing.get("status") == "success":
                    if readiness(existing["output"]) or sha(existing["output"]) != existing["output_sha256"]:
                        raise ValueError("Resumed output changed")
                    results.append(existing)
                    continue
            result = execute_case(row, args.output, pipeline, detector, base_args, torch, args.device)
            results.append(result)
            summary = {"method": "liveportrait", "case_count": len(rows), "completed_count": len(results),
                       "success_count": sum(x["status"] == "success" for x in results),
                       "failed_count": sum(x["status"] == "failed" for x in results), "results": results}
            save_json(args.output / "summary.json", summary)
            print(json.dumps({"case_id": row["case_id"], "status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        save_json(args.output / "summary.json", {"method": "liveportrait", "case_count": len(rows), "completed_count": len(results),
                  "success_count": sum(x["status"] == "success" for x in results),
                  "failed_count": sum(x["status"] == "failed" for x in results), "results": results})
    except Exception as error:
        save_json(args.output / "run_failure.json", {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        print(traceback.format_exc(), file=sys.stderr)
        raise
    finally:
        streams.close()
        lock.unlink(missing_ok=True)
    return result_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
