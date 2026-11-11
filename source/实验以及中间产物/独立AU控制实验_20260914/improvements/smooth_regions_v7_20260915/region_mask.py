#!/usr/bin/env python3
"""Fixed v7 smoothing of SHA-pinned v6 source-only latent masks, not image retouching."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V6_PATH = HERE.parent / "region_preserving_v6_20260915" / "region_mask.py"
V6_CODE_SHA256 = "75f7995e3addb7656e0935da62e920f9d43b540ffa9b5e127093835afb3e438e"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_v6(name):
    if file_sha(V6_PATH) != V6_CODE_SHA256:
        raise ValueError("Frozen v6 source geometry implementation changed")
    spec = importlib.util.spec_from_file_location(name, V6_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# This geometry module is never rebound or monkey-patched by prepare().
BASE_V6 = _load_v6("fixed_v6_geometry_for_v7")
AUS = BASE_V6.AUS
canonical_sha = BASE_V6.canonical_sha
_allowed_path = BASE_V6._allowed_path
VERSION = "source68_v6_fixed_maxpool7_gaussian_s1p5_t3_v7"
SMOOTHING = {
    "input": "exact float32 64x64 output of pinned v6 build_mask",
    "maxpool": {"kernel_size": [7, 7], "stride": 1, "padding": [3, 3],
                "padding_mode": "constant_zero", "context_radius_latent_px": 3,
                "context_radius_image_px": 24},
    "gaussian": {"sigma_latent_px": 1.5, "truncate_sigma": 3.0,
                 "radius_rounding": "int(truncate_sigma * sigma_latent_px + 0.5)",
                 "radius_latent_px": 5, "kernel_length": 11,
                 "boundary_mode": "half_sample_symmetric (scipy reflect; numpy symmetric)",
                 "implementation": "normalized float64 separable convolution, horizontal then vertical"},
    "output": "clip to [0,1], cast to float32 64x64",
    "no_edit": "explicit all-zero output; no dilation or Gaussian for an empty v6 mask",
    "parameter_grid": False,
    "applied_to": "latent editing mask during denoising, never generated image pixels",
}
POLICY = {
    "version": VERSION,
    "interpretation": "Engineering smooth edit-support prior, not a strict causal AU partition",
    "source_only": True, "test_used": False, "target_geometry_used": False,
    "base_v6_code": {"path": str(V6_PATH), "sha256": V6_CODE_SHA256},
    "base_v6_policy": copy.deepcopy(BASE_V6.POLICY),
    "base_v6_policy_sha256": canonical_sha(BASE_V6.POLICY),
    "smoothing": SMOOTHING,
    "source_size": [512, 512], "latent_size": [64, 64],
    "protocol": "One fixed development engineering revision, not a mask parameter search",
    "outcome_note": "Revision motivated by v6 boundary artifacts; no per-case/result-adaptive geometry or mask tuning",
}


def _mask_sha(mask):
    return hashlib.sha256(mask.tobytes(order="C")).hexdigest()


def _area(mask):
    return {"alpha_sum_64": float(mask.sum(dtype=np.float64)),
            "alpha_mean_64": float(mask.mean(dtype=np.float64)),
            "positive_fraction_64": float(np.mean(mask > 0)),
            "at_least_half_fraction_64": float(np.mean(mask >= .5)),
            "alpha_equivalent_pixels_512": float(mask.sum(dtype=np.float64) * 64)}


def _support_bbox(mask):
    y, x = np.nonzero(mask > 0)
    return None if len(x) == 0 else [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]


def _kernel():
    sigma, radius = 1.5, 5
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (x / sigma) ** 2)
    return weights / weights.sum(dtype=np.float64)


def smooth_mask(base_mask):
    """Fixed mask-only morphology/convolution, with no external information."""
    base_mask = np.asarray(base_mask)
    if base_mask.shape != (64, 64) or base_mask.dtype != np.float32:
        raise ValueError("Expected the exact v6 float32 64x64 latent mask")
    if not np.isfinite(base_mask).all() or base_mask.min() < 0 or base_mask.max() > 1:
        raise ValueError("Invalid v6 mask range or nonfinite value")
    if not np.any(base_mask):
        return np.zeros((64, 64), dtype=np.float32)
    padded = np.pad(base_mask.astype(np.float64), 3, mode="constant", constant_values=0)
    dilated = np.lib.stride_tricks.sliding_window_view(padded, (7, 7)).max(axis=(-1, -2))
    weights = _kernel()
    horizontal = np.pad(dilated, ((0, 0), (5, 5)), mode="symmetric")
    horizontal = (np.lib.stride_tricks.sliding_window_view(horizontal, 11, axis=1)
                  * weights).sum(axis=-1, dtype=np.float64)
    vertical = np.pad(horizontal, ((5, 5), (0, 0)), mode="symmetric")
    blurred = (np.lib.stride_tricks.sliding_window_view(vertical, 11, axis=0)
               * weights).sum(axis=-1, dtype=np.float64)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def build_mask(case, source_record):
    """Return (float32[64,64], receipt), preserving v6 source-only validation."""
    base_mask, base_receipt = BASE_V6.build_mask(case, source_record)
    if _mask_sha(base_mask) != base_receipt["mask_bytes_sha256"]:
        raise ValueError("Pinned v6 mask byte hash is internally inconsistent")
    mask = smooth_mask(base_mask)
    receipt = copy.deepcopy(base_receipt)
    receipt.update({
        "version": VERSION, "policy_sha256": canonical_sha(POLICY),
        "interpretation": POLICY["interpretation"],
        "base_v6": {"code": POLICY["base_v6_code"],
                    "policy_sha256": base_receipt["policy_sha256"],
                    "mask_bytes_sha256": _mask_sha(base_mask),
                    "mask_area": base_receipt["mask_area"],
                    "receipt_without_artifacts_sha256": canonical_sha(base_receipt)},
        "smoothing": copy.deepcopy(SMOOTHING), "gaussian_kernel_float64": _kernel().tolist(),
        "regions_describe": "Original v6 pre-smoothing regions only, not the final v7 support extent",
        "base_v6_support_bbox_64": _support_bbox(base_mask),
        "smoothed_support_bbox_64": _support_bbox(mask),
        "mask_area": _area(mask), "mask_bytes_sha256": _mask_sha(mask),
        "mask_array_source_only": True, "generated_images_modified": False,
    })
    return mask, receipt


def prepare(cases_path, measurements_path, output):
    # Reuse the fixed IO/validation/receipt-writing implementation through a
    # separately loaded module. Its globals are local in-memory bindings only;
    # the BASE_V6 geometry module and every v6 file remain untouched.
    io = _load_v6("fixed_v6_io_adapted_for_v7")
    io.build_mask = build_mask
    io.POLICY = POLICY
    io.VERSION = VERSION
    io.__file__ = str(Path(__file__).resolve())  # manifest inputs.code_sha256 is the v7 entry point.
    return io.prepare(cases_path, measurements_path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--cases", required=True)
    parser.add_argument("--source-measurements", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = prepare(args.cases, args.source_measurements, args.output)
    print(json.dumps({"status": manifest["status"], "case_count": manifest["case_count"],
                      "unique_source_count": manifest["unique_source_count"],
                      "source_unchanged": manifest["source_unchanged"],
                      "policy_sha256": manifest["policy_sha256"],
                      "manifest": str(Path(args.output) / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
