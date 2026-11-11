#!/usr/bin/env python3
"""Fixed-version Face-Adapter reenactment, without modifying upstream files.

All network/model operations happen only with --execute. --preflight is stdlib-only.
Manifest JSONL: case_id, source, target, optional seed (default 0), other metadata.
Source/target paths may be absolute or relative to the manifest's parent.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from types import ModuleType, SimpleNamespace

from faceadapter_detection_policy import DetectionPolicy, THRESHOLDS

ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "6e313885188e05740a7c40e12c1b9015c748bf08"
POLICY = {
    "branch": "reenactment", "scheduler": "EulerDiscreteScheduler", "steps": 25,
    "guidance_scale": 5.0, "crop_ratio": 0.81, "image_size": 512,
    "condition_seed": 999, "diffusion_seed_default": 0,
    "token_encoder_mode": "upstream_training_mode_preserved",
    "seed_policy": "reset condition_seed before each case; diffusion seed from row",
    "timing_scope": "single-case preprocessing, encoding, inference, PNG save; excludes model loading and evaluation",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def json_signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def snapshot_file(source, destination, expected_sha256):
    """Capture exactly the bytes bound to a run, never replace an older snapshot."""
    payload = Path(source).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Snapshot source drifted: " + str(source))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError("Existing snapshot differs: " + str(destination))
        return
    with destination.open("xb") as handle:
        handle.write(payload)


def dependency_versions():
    versions = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "huggingface-hub", "safetensors", "insightface", "onnxruntime", "numpy", "scipy", "opencv-python", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python_executable": sys.executable, "python": sys.version, "packages": versions}


def git_state(repo):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if commit != UPSTREAM_COMMIT or dirty:
        raise ValueError("Upstream must be clean and pinned to " + UPSTREAM_COMMIT)
    return commit


def load_cases(manifest, selected=None):
    cases, ids = [], set()
    manifest = Path(manifest).resolve()
    if manifest.suffix.lower() == ".json":
        document = json.loads(manifest.read_text())
        raw_rows = document["cases"] if isinstance(document, dict) else document
    else:
        raw_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    for line_no, raw in enumerate(raw_rows, 1):
        row = dict(raw)
        # Existing frozen comparison contracts use paths relative to thesis root.
        if "source_image_path" in row and "source" not in row:
            thesis_root = ROOT.parent.parent
            row["source"] = str(thesis_root / row["source_image_path"])
            row["target"] = str(thesis_root / row["target_image_path"])
        case_id = row.get("case_id", "")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", case_id):
            raise ValueError(f"Line {line_no}: invalid case_id")
        if case_id in ids:
            raise ValueError("Duplicate case_id: " + case_id)
        ids.add(case_id)
        if selected and case_id not in selected:
            continue
        row["seed"] = int(row.get("seed", 0))
        for field in ("source", "target"):
            path = Path(row[field])
            if not path.is_absolute():
                path = manifest.parent / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            row[field] = str(path)
            actual_hash = sha256(path)
            declared_hash = row.get(field + "_sha256") or row.get(field + "_image_sha256") or row.get(field + "_sha256_expected")
            if declared_hash and declared_hash != actual_hash:
                raise ValueError("Frozen input hash mismatch: " + case_id + " / " + field)
            row[field + "_sha256"] = actual_hash
        cases.append(row)
    if not cases:
        raise ValueError("Empty manifest")
    if selected and set(selected) - ids:
        raise ValueError("Requested case ids absent: " + ", ".join(sorted(set(selected) - ids)))
    return cases


def file_readiness(path):
    """Never read a macOS dataless placeholder or a partial download."""
    path = Path(path)
    if path.name.endswith((".part", ".tmp", ".incomplete")):
        return "partial download"
    try:
        info = path.stat()
    except FileNotFoundError:
        return "missing"
    if not path.is_file():
        return "not a regular file"
    if getattr(info, "st_flags", 0) & 0x40000000:
        return "dataless / not resident locally"
    if info.st_size <= 0:
        return "empty file"
    return None


def checked_json(path, files, missing):
    problem = file_readiness(path)
    if problem:
        missing.append(f"{path} ({problem})")
        return None
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
    except (ValueError, UnicodeError, OSError) as error:
        missing.append(f"{path} (invalid JSON: {error})")
        return None
    files.add(Path(path))
    return value


def checked_file(path, files, missing):
    problem = file_readiness(path)
    if problem:
        missing.append(f"{path} ({problem})")
    else:
        files.add(Path(path))


def component_inventory(directory, kind="diffusers"):
    """Select normal loader filenames, not arbitrary *.bin or download receipts.

    Config-only/partial-only directories fail. A selected sharded index must
    contain a nonempty weight_map and every referenced resident shard.
    """
    directory = Path(directory)
    files, missing = set(), []
    checked_json(directory / "config.json", files, missing)
    names = ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin") if kind == "diffusers" else ("model.safetensors", "pytorch_model.bin")
    selected = None
    for filename in names:
        direct = directory / filename
        index = directory / (filename + ".index.json")
        if direct.exists():
            checked_file(direct, files, missing)
            selected = direct
            break
        if index.exists():
            selected = index
            data = checked_json(index, files, missing)
            weight_map = data.get("weight_map") if data else None
            if not isinstance(weight_map, dict) or not weight_map:
                missing.append(f"{index} (missing or empty weight_map)")
                break
            shard_names = list(weight_map.values())
            if not all(isinstance(name, str) for name in shard_names):
                missing.append(f"{index} (non-string shard name)")
                break
            for shard in sorted(set(shard_names)):
                relative = Path(shard)
                if relative.is_absolute() or ".." in relative.parts or relative.suffix not in (".safetensors", ".bin"):
                    missing.append(f"{index} (invalid shard path: {shard})")
                    continue
                checked_file(directory / relative, files, missing)
            break
    if selected is None:
        missing.append(f"{directory} (missing complete weight file or shard index: {', '.join(names)})")
    return files, missing


def resource_inventory(repo, checkpoint, base_model, vae_model, hash_content=False, preload_final_components=False, base_model_kind="stable-diffusion-v1-5"):
    repo, checkpoint, base_model, vae_model = map(Path, (repo, checkpoint, base_model, vae_model))
    files, missing = set(), []
    required_files = [repo / "empty_prompt_embedding.pth"]
    required_files += [checkpoint / name for name in (
        "net_image2token.pth", "net_id2token.pth", "net_seg_res18.pth",
        "third_party/d3dfr_res50_nofc.pth", "third_party/BFM_model_front.mat",
        "third_party/insightface_glint360k.pth",
    )]
    required_files += [checkpoint / "third_party/models/antelopev2" / name for name in (
        "1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "glintr100.onnx", "scrfd_10g_bnkps.onnx",
    )]
    for path in required_files:
        checked_file(path, files, missing)
    model_index = checked_json(base_model / "model_index.json", files, missing)
    checked_json(base_model / "scheduler/scheduler_config.json", files, missing)
    checked_json(base_model / "tokenizer/tokenizer_config.json", files, missing)
    checked_json(base_model / "tokenizer/vocab.json", files, missing)
    checked_file(base_model / "tokenizer/merges.txt", files, missing)
    for name in ("special_tokens_map.json", "added_tokens.json", "tokenizer.json"):
        path = base_model / "tokenizer" / name
        if path.exists():
            checked_json(path, files, missing)
    components = [(checkpoint / "controlnet", "diffusers"), (checkpoint / "vision_encoder", "transformers"), (vae_model, "diffusers"), (base_model / "text_encoder", "transformers")]
    final_unet = checkpoint / "pretrained_unet"
    if base_model_kind == "stable-diffusion-v1-5" and (preload_final_components or final_unet.exists()):
        components.append((final_unet, "diffusers"))
    if not preload_final_components:
        # Original loading still reads these before replacing the final weights.
        components.extend([(base_model / "unet", "diffusers"), (base_model / "vae", "diffusers")])
        for name in ("safety_checker", "feature_extractor"):
            declared = model_index.get(name) if model_index else None
            if isinstance(declared, list) and any(value is not None for value in declared):
                if name == "safety_checker":
                    components.append((base_model / name, "transformers"))
                else:
                    checked_json(base_model / name / "preprocessor_config.json", files, missing)
    for directory, kind in components:
        component_files, problems = component_inventory(directory, kind)
        files.update(component_files)
        missing.extend(problems)
    inventory = []
    for path in sorted(files):
        info = path.stat()
        row = {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "resident": True}
        if hash_content:
            row["sha256"] = sha256(path)
        inventory.append(row)
    return {"missing": missing, "files": inventory, "content_hashed": hash_content, "policy": "actual loader config and selected complete weights only; partial/metadata excluded; dataless rejected"}


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)


class Compatibility(ast.NodeTransformer):
    """Only explicit device/seed/resource adaptations, not architecture edits."""
    def visit_Constant(self, node):
        if node.value == "stabilityai/sd-vae-ft-mse":
            return ast.copy_location(ast.Attribute(ast.Name("args", ast.Load()), "vae_model", ast.Load()), node)
        if node.value == "empty_prompt_embedding.pth":
            return ast.copy_location(ast.Attribute(ast.Name("args", ast.Load()), "empty_prompt", ast.Load()), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "FaceAnalysis":
            for keyword in node.keywords:
                if keyword.arg == "providers":
                    keyword.value = ast.Name("onnx_providers", ast.Load())
        if isinstance(node.func, ast.Attribute) and node.func.attr == "prepare":
            for keyword in node.keywords:
                if keyword.arg == "ctx_id":
                    keyword.value = ast.Name("onnx_ctx_id", ast.Load())
        if isinstance(node.func, ast.Attribute) and node.func.attr == "manual_seed":
            node.args = [ast.Name("case_seed", ast.Load())]
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        # Absolute snapshot paths lose the upstream model-id string; preserve
        # SD1.5's original optional pretrained_unet branch via explicit model kind.
        if isinstance(node.left, ast.Constant) and node.left.value == "stable-diffusion" and len(node.comparators) == 1:
            node.comparators[0] = ast.Attribute(ast.Name("args", ast.Load()), "base_model_kind", ast.Load())
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if assigned(node, "pipe"):
            expression = "load_pipeline_components(StableDiffusionFaceAdapterPipeline, UNet2DConditionModel, AutoencoderKL, args, controlnet, weight_dtype, device)"
            node.value = ast.parse(expression, mode="eval").body
        if assigned(node, "vae_ft_mse"):
            node.value = ast.IfExp(test=ast.Attribute(ast.Name("args", ast.Load()), "preload_final_components", ast.Load()), body=ast.Attribute(ast.Name("pipe", ast.Load()), "vae", ast.Load()), orelse=node.value)
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        # The only upstream conditional that assigns pipe.unet is the optional
        # final UNet replacement. Skip duplicate loading after direct injection.
        replaces_unet = any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "pipe" and target.attr == "unet" for target in child.targets) for child in node.body)
        if replaces_unet:
            node.test = ast.BoolOp(op=ast.And(), values=[ast.UnaryOp(op=ast.Not(), operand=ast.Attribute(ast.Name("args", ast.Load()), "preload_final_components", ast.Load())), node.test])
        return node


def split_upstream(source):
    """Compile original initialization and one pair; remove only swapping/output glue."""
    tree = ast.parse(source)
    infer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "infer")
    src_for = next(node for node in infer.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "src_im_file")
    drive_for = next(node for node in src_for.body if isinstance(node, ast.For))
    stop = next(i for i, node in enumerate(drive_for.body) if assigned(node, "accurate_mask_path"))
    initialization = []
    collecting = False
    for node in infer.body:
        if assigned(node, "src_img_list"):
            break
        # Skip upstream output directory creation; wrapper owns unique output paths.
        if assigned(node, "controlnet"):
            collecting = True
        if not collecting:
            continue
        initialization.append(node)
    pair = []
    for node in src_for.body:
        if node is drive_for:
            break
        pair.append(node)
    pair.extend(drive_for.body[:stop])
    # Replace the last save call so repeated sources cannot overwrite each other.
    save = pair[-1]
    if not (isinstance(save, ast.Expr) and isinstance(save.value, ast.Call) and isinstance(save.value.func, ast.Attribute) and save.value.func.attr == "save"):
        raise ValueError("Unexpected upstream reenactment boundary")
    save.value.args = [ast.Name("output_path", ast.Load())]
    imports = []
    for node in tree.body:
        if node is infer or isinstance(node, ast.If):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "third_party" and any(alias.name == "model_parsing" for alias in node.names):
            continue
        imports.append(node)
    result = []
    for nodes in (imports, initialization, pair):
        module = Compatibility().visit(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
        ast.fix_missing_locations(module)
        result.append(compile(module, "<fixed-upstream-FaceAdapter>", "exec"))
    return tuple(result)


def compile_pipeline_without_unused_fastai(source):
    tree = ast.parse(source)
    unused = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "fastai.basics"]
    if len(unused) != 1 or [alias.name for alias in unused[0].names] != ["*"]:
        raise ValueError("Unexpected upstream fastai import; re-audit compatibility")
    tree.body.remove(unused[0])
    ast.fix_missing_locations(tree)
    return compile(tree, "<fixed-upstream-FaceAdapter-pipeline>", "exec")


def load_compatible_pipeline(repo):
    path = Path(repo) / "face_adapter_pipline.py"
    module = ModuleType("face_adapter_pipline")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile_pipeline_without_unused_fastai(path.read_text()), module.__dict__)
    return module


class TorchLoadProxy:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def load(self, *args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        kwargs.setdefault("weights_only", True)
        return self._module.load(*args, **kwargs)


def load_pipeline_components(pipeline_class, unet_class, vae_class, args, controlnet, dtype, device):
    kwargs = dict(controlnet=controlnet, torch_dtype=dtype, cache_dir=args.cache_dir if args.use_cache else None, local_files_only=args.use_cache, requires_safety_checker=False)
    if args.preload_final_components:
        final_unet = Path(args.checkpoint) / "pretrained_unet"
        if args.base_model_kind != "stable-diffusion-v1-5" or not final_unet.is_dir():
            raise ValueError("Final-component preload requires SD1.5 and author pretrained_unet")
        kwargs["unet"] = unet_class.from_pretrained(str(final_unet), torch_dtype=dtype).to(device)
        kwargs["vae"] = vae_class.from_pretrained(args.vae_model, cache_dir=args.cache_dir if args.use_cache else None, torch_dtype=dtype, local_files_only=args.use_cache).to(device)
        # The upstream custom __call__ comments out run_safety_checker entirely;
        # these components do not participate in the reproduced computation.
        kwargs["safety_checker"] = None
        kwargs["feature_extractor"] = None
    return pipeline_class.from_pretrained(args.base_model, **kwargs).to(device)


def synchronize(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--base-model", type=Path, required=True, help="Downloaded local Diffusers model directory")
    result.add_argument("--base-model-kind", choices=("stable-diffusion-v1-5", "community"), default="stable-diffusion-v1-5")
    result.add_argument("--vae-model", type=Path, required=True, help="Downloaded local sd-vae-ft-mse directory")
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--case-id", action="append", help="Run only these frozen case ids; repeat for multiple cases")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    result.add_argument("--dtype", choices=("float32", "float16"), default=None)
    result.add_argument("--preload-final-components", action="store_true", help="Inject the identical final author UNet and FT-VAE; skip downloading unused base UNet/VAE")
    result.add_argument("--detection-policy", choices=("author_default", "shared_fallback"), default="author_default", help="author_default keeps original threshold; shared_fallback is a separately recorded input detector policy")
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    result.add_argument("--resume", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for field in ("repo", "checkpoint", "base_model", "vae_model", "manifest", "output"):
        setattr(args, field, getattr(args, field).resolve())
    if ROOT not in args.output.parents:
        raise ValueError("Output must be a subdirectory of " + str(ROOT))
    commit = git_state(args.repo)
    cases = load_cases(args.manifest, set(args.case_id or []))
    source = (args.repo / "infer.py").read_text()
    module_code, init_code, case_code = split_upstream(source)
    inventory = resource_inventory(args.repo, args.checkpoint, args.base_model, args.vae_model, args.execute, args.preload_final_components, args.base_model_kind)
    if args.preload_final_components and (args.base_model_kind != "stable-diffusion-v1-5" or not (args.checkpoint / "pretrained_unet").is_dir()):
        inventory["missing"].append("--preload-final-components requires SD1.5 + checkpoint/pretrained_unet")
    dtype = args.dtype or ("float16" if args.device == "cuda" else "float32")
    if args.device == "cpu" and dtype != "float32":
        raise ValueError("CPU execution requires float32")
    configuration = {
        "upstream_commit": commit, "upstream_infer_sha256": sha256(args.repo / "infer.py"),
        "upstream_pipeline_sha256": sha256(args.repo / "face_adapter_pipline.py"),
        "compatibility": "remove unused fastai.basics wildcard import in memory; explicit load map_location=cpu; device/providers; deterministic case seeds",
        "wrapper_sha256": sha256(__file__), "policy": POLICY, "device": args.device, "dtype": dtype,
        "detection_policy": args.detection_policy,
        "detection_policy_source_sha256": sha256(ROOT / "faceadapter_detection_policy.py"),
        "detection_parameters": {"thresholds": list(THRESHOLDS) if args.detection_policy == "shared_fallback" else "author detector threshold", "preprocess_order": ["original", "autocontrast_cutoff_1"] if args.detection_policy == "shared_fallback" else ["original"], "generation_rgb_modified": False},
        "base_model": str(args.base_model), "base_model_kind": args.base_model_kind, "vae_model": str(args.vae_model),
        "checkpoint": str(args.checkpoint), "preload_final_components": args.preload_final_components, "manifest_sha256": sha256(args.manifest),
        "cases": cases, "resources": inventory,
        "python": sys.version, "platform": platform.platform(),
    }
    if args.preflight:
        print(json.dumps({"static_ok": True, "case_count": len(cases), "missing_resources": inventory["missing"], "configuration": configuration}, ensure_ascii=False, indent=2))
        return 0 if not inventory["missing"] else 2
    if inventory["missing"]:
        raise FileNotFoundError("Missing resources: " + ", ".join(inventory["missing"]))
    if args.output.exists() and not args.resume:
        raise FileExistsError("Output exists; choose new output or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / ".run.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("Output locked; inspect running process before manually removing stale lock")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        config_path = args.output / "configuration.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text())
            if json_signature(previous) != json_signature(configuration):
                raise ValueError("Resume refused: inputs, weights, code or execution configuration changed")
        else:
            atomic_json(config_path, configuration)
        snapshot_file(__file__, args.output / "snapshots/run_faceadapter.py", configuration["wrapper_sha256"])
        snapshot_file(ROOT / "faceadapter_detection_policy.py", args.output / "snapshots/faceadapter_detection_policy.py", configuration["detection_policy_source_sha256"])
        snapshot_file(args.repo / "infer.py", args.output / "snapshots/upstream_infer.py", configuration["upstream_infer_sha256"])
        snapshot_file(args.repo / "face_adapter_pipline.py", args.output / "snapshots/upstream_face_adapter_pipline.py", configuration["upstream_pipeline_sha256"])
        dependency_record = dependency_versions()
        dependency_path = args.output / "dependency_versions.json"
        if dependency_path.exists():
            if json.loads(dependency_path.read_text()) != dependency_record:
                raise ValueError("Dependency versions changed since this run")
        else:
            atomic_json(dependency_path, dependency_record)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
        os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
        os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg")
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
        sys.dont_write_bytecode = True
        sys.pycache_prefix = str(ROOT / "cache/pycache")
        # Imported upstream modules must not create untracked __pycache__ files,
        # otherwise the deliberately strict clean-repo resume check would fail.
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(args.repo))
        namespace = {"__name__": "faceadapter_frozen_runtime", "__file__": str(args.repo / "infer.py")}
        with (args.output / "runtime.log").open("a", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            load_compatible_pipeline(args.repo)
            exec(module_code, namespace)
            torch = namespace["torch"]
            if args.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS is unavailable")
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            proxy = TorchLoadProxy(torch)
            namespace["torch"] = proxy
            namespace["model_insightface_backbone"].torch = proxy
            namespace["device"] = args.device
            namespace["weight_dtype"] = getattr(torch, dtype)
            namespace["onnx_providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
            namespace["onnx_ctx_id"] = 0 if args.device == "cuda" else -1
            runtime_args = SimpleNamespace(checkpoint=str(args.checkpoint), base_model=str(args.base_model), base_model_kind=args.base_model_kind, vae_model=str(args.vae_model), empty_prompt=str(args.repo / "empty_prompt_embedding.pth"), use_cache=True, cache_dir=str(ROOT / "cache/huggingface"), crop_ratio=POLICY["crop_ratio"], preload_final_components=args.preload_final_components)
            namespace["args"] = runtime_args
            namespace["load_pipeline_components"] = load_pipeline_components
            namespace["set_seed"](POLICY["condition_seed"])
            start = time.monotonic()
            exec(init_code, namespace)
            namespace["app"] = DetectionPolicy(namespace["app"], args.detection_policy)
            synchronize(torch, args.device)
            atomic_json(args.output / "environment.json", {"torch": torch.__version__, "numpy": namespace["np"].__version__, "device": args.device, "dtype": dtype, "model_initialization_seconds": time.monotonic() - start, "token_encoder_training": {key: namespace[key].training for key in ("net_image2token", "net_id2token")}})
            failures = 0
            for row in cases:
                case_dir = args.output / "cases" / row["case_id"]
                case_dir.mkdir(parents=True, exist_ok=True)
                result_path, image_path = case_dir / "result.json", case_dir / "generated.png"
                if result_path.exists():
                    previous = json.loads(result_path.read_text())
                    if previous.get("status") == "success":
                        if previous["input"] != row or not image_path.is_file() or sha256(image_path) != previous["output_sha256"]:
                            raise ValueError("Corrupt or mismatched resumed case: " + row["case_id"])
                        continue
                if image_path.exists():
                    raise FileExistsError("Unsealed image exists; inspect before rerun: " + str(image_path))
                # A fresh local namespace removes previous-case tensors and file variables.
                local = dict(namespace)
                runtime_args.source = str(Path(row["source"]).parent)
                runtime_args.target = str(Path(row["target"]).parent)
                local.update(src_im_file=Path(row["source"]).name, drive_im_file=Path(row["target"]).name, output_path=str(image_path), case_seed=row["seed"])
                namespace["app"].begin_case(row["case_id"])
                namespace["set_seed"](POLICY["condition_seed"])
                synchronize(torch, args.device)
                start = time.monotonic()
                try:
                    exec(case_code, local)
                    synchronize(torch, args.device)
                    with namespace["Image"].open(image_path) as generated:
                        if generated.size != (512, 512):
                            raise ValueError("Unexpected generated image dimensions")
                        generated.verify()
                    record = {"status": "success", "input": row, "output": str(image_path), "output_sha256": sha256(image_path), "elapsed_seconds": time.monotonic() - start, "timing_scope": POLICY["timing_scope"]}
                except Exception as error:
                    failures += 1
                    record = {"status": "failed", "input": row, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - start}
                record["detection_policy"] = args.detection_policy
                record["detection_calls"] = list(namespace["app"].events)
                attempt = 1
                while (case_dir / f"attempt_{attempt:03d}.json").exists():
                    attempt += 1
                atomic_json(case_dir / f"attempt_{attempt:03d}.json", record)
                atomic_json(result_path, record)
                print(json.dumps({"case_id": row["case_id"], "status": record["status"]}), flush=True)
                del local
            records = [json.loads((args.output / "cases" / row["case_id"] / "result.json").read_text()) for row in cases]
            atomic_json(args.output / "summary.json", {"case_count": len(cases), "success_count": sum(record["status"] == "success" for record in records), "failed_count": sum(record["status"] == "failed" for record in records), "results": records})
            return 1 if failures else 0
    except BaseException as error:
        number = 1
        while (args.output / f"run_failure_{number:03d}.json").exists():
            number += 1
        atomic_json(args.output / f"run_failure_{number:03d}.json", {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
