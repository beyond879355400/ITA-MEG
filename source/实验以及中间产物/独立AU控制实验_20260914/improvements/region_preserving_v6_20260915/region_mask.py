#!/usr/bin/env python3
"""Fixed source-only geometric AU edit support; not a causal AU segmentation.

Only source_real selected OpenFace 68-point geometry and the declared source /
target AU control vectors influence the mask. Target images, target landmarks,
prompts, generated images, AU detector outputs and outcome scores are not used.
Numpy creates all masks. Pillow is only used for debug PNGs / source overlays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np


AUS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
VERSION = "source68_fixed_geometric_prior_v1"
REGION_BY_AU = {
    1: ("brow_left", "brow_right"),
    2: ("brow_left", "brow_right"),
    4: ("brow_left", "brow_right"),
    5: ("eye_left", "eye_right"),
    6: ("cheek_left", "cheek_right"),
    9: ("nose",),
    12: ("mouth",),
    15: ("mouth",),
    17: ("mouth_chin",),
    20: ("mouth",),
    25: ("mouth",),
    26: ("mouth_chin",),
}
POLICY = {
    "version": VERSION,
    "interpretation": "Engineering geometric edit-support prior, not strict AU causal partition",
    "au_order": AUS,
    "region_by_au": REGION_BY_AU,
    "landmark_indices": "OpenFace 68-point zero-based indices; left/right are image sides",
    "source_size": [512, 512],
    "latent_size": [64, 64],
    "face_scale": "Euclidean distance between source landmarks 0 and 16",
    "brow_padding_scale_xy": [0.035, 0.040, 0.025],
    "eye_padding_scale_xy": [0.022, 0.022, 0.022],
    "nose_padding_scale_xy": [0.020, 0.018, 0.018],
    "mouth_padding_scale_xy": [0.050, 0.025, 0.045],
    "mouth_chin_padding_scale_xy": [0.040, 0.025, 0.025],
    "mouth_upper_exclusion": "support y >= max(source nose 31:36 y) + 0.010 * scale",
    "cheek_support": "lower eyelid to nose base, between outer jaw and ipsilateral nose wing",
    "cheek_offset_scale": 0.010,
    "soft_edge_scale": 0.025,
    "edge": "compact inward smoothstep, exactly zero outside the declared support boxes",
    "union": "pointwise max, regardless of AU sign or magnitude",
    "downsample": "exact 8x8 area average from 512 to 64, float32",
    "no_edit": "exactly zero when source and target AU control vectors are equal",
    "landmark_crop_tolerance_px": 64,
    "minimum_source_confidence": 0.5,
    "test_used": False,
    "target_geometry_used": False,
    "outcome_tuning": False,
}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _allowed_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Path must be absolute: {value}")
    resolved = path.resolve()
    if any(re.search(r"(^|[_-])(test|final_test)([_-]|$)", part.lower())
           for part in resolved.parts):
        raise ValueError(f"Test paths are forbidden: {value}")
    return resolved


def _png_size(path):
    # Read only the PNG header. No image transform participates in geometry.
    with Path(path).open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Source must be a PNG with a valid IHDR header")
    return tuple(struct.unpack(">II", header[16:24]))


def _controls(case):
    if case.get("split") not in ("train", "dev"):
        raise ValueError("Only explicit train/dev cases are allowed")
    order = tuple(case.get("au_order", AUS))
    if any(au not in REGION_BY_AU for au in order):
        raise ValueError("Unknown AU in au_order")
    if order != AUS:
        raise ValueError("AU order must exactly match the frozen 12-AU order")
    source = np.asarray(case["source_aus_0_5"], dtype=np.float64)
    target = np.asarray(case["target_aus_0_5"], dtype=np.float64)
    if source.shape != (12,) or target.shape != (12,):
        raise ValueError("Expected exactly 12 source and target AU control values")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("AU controls must be finite")
    if np.any(source < 0) or np.any(source > 5) or np.any(target < 0) or np.any(target > 5):
        raise ValueError("AU control values must be in [0, 5]")
    changed = [au for au, before, after in zip(AUS, source, target) if before != after]
    if "changed_aus" in case:
        declared = case["changed_aus"]
        if any(au not in REGION_BY_AU for au in declared):
            raise ValueError("Unknown AU in changed_aus")
        if sorted(declared) != sorted(changed):
            raise ValueError("Declared changed_aus conflicts with AU controls")
    if case.get("group") == "no_edit" and changed:
        raise ValueError("no_edit case has nonzero AU change")
    return source, target, changed


def _source_geometry(case, record):
    if record.get("method") != "source_real":
        raise ValueError("Only method=source_real geometry is allowed")
    if record.get("split") not in ("train", "dev") or record["split"] != case["split"]:
        raise ValueError("Source record must match the train/dev case split")
    source = case["source_real"]
    if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))):
        raise ValueError("Source SHA-256 is required")
    if record.get("sha256") != source["sha256"]:
        raise ValueError("Source geometry SHA-256 mismatch")
    path = _allowed_path(source["path"])
    if _allowed_path(record["path"]) != path:
        raise ValueError("Source geometry path mismatch")
    if record.get("measurement_source_path") is not None:
        if _allowed_path(record["measurement_source_path"]) != path:
            raise ValueError("Measurement source path mismatch")
    size = tuple(record["image_size"]) if "image_size" in record else _png_size(path)
    if size != (512, 512):
        raise ValueError("Source image must be exactly 512x512; no geometry resize is allowed")
    if record.get("returncode", 0) != 0 or record.get("parse_status") != "primary_face_finite":
        raise ValueError("Source geometry did not pass the existing OpenFace parser")
    confidence = float(record.get("confidence", 0))
    if not np.isfinite(confidence) or confidence < POLICY["minimum_source_confidence"]:
        raise ValueError("Source face confidence is invalid or below the fixed threshold")
    row_index = record.get("selected_row")
    rows = record.get("raw_rows", [])
    if type(row_index) is not int or row_index < 0 or row_index >= len(rows):
        raise ValueError("Invalid selected source landmark row")
    row = rows[row_index]
    try:
        points = np.asarray([[float(row[f"x_{i}"]), float(row[f"y_{i}"])]
                             for i in range(68)], dtype=np.float64)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Invalid source 68 landmarks") from exc
    if not np.isfinite(points).all():
        raise ValueError("Source landmarks must all be finite")
    if np.any(points < -64) or np.any(points > 576):
        raise ValueError("Source landmarks exceed fixed 64px crop tolerance")
    scale = float(np.linalg.norm(points[16] - points[0]))
    if not 64 <= scale <= 512 or np.ptp(points[:, 1]) < 64:
        raise ValueError("Degenerate source face geometry")
    if points[36:42, 0].mean() >= points[42:48, 0].mean():
        raise ValueError("Invalid landmark topology: image-side eye order")
    if points[48:68, 1].mean() <= points[36:48, 1].mean():
        raise ValueError("Invalid landmark topology: mouth above eyes")
    return points, scale, row_index, confidence


def _regions(points, scale):
    """All boxes are fixed source-derived support bounds, with inward soft edges."""
    regions = {}

    def add(name, indices, padding, lower_y=None):
        q = points[indices]
        px, up, down = np.asarray(padding) * scale
        bounds = [float(q[:, 0].min() - px), float(q[:, 1].min() - up),
                  float(q[:, 0].max() + px), float(q[:, 1].max() + down)]
        if lower_y is not None:
            bounds[1] = max(bounds[1], float(lower_y))
        regions[name] = {"shape": "soft_bbox", "source_landmark_indices": list(indices),
                         "support_bbox_512_unclipped": bounds,
                         "support_bbox_512_clipped": np.clip(bounds, 0, 512).tolist(),
                         "padding_face_scale_xy_up_down": list(padding)}

    add("brow_left", range(17, 22), (0.035, 0.040, 0.025))
    add("brow_right", range(22, 27), (0.035, 0.040, 0.025))
    add("eye_left", range(36, 42), (0.022, 0.022, 0.022))
    add("eye_right", range(42, 48), (0.022, 0.022, 0.022))
    add("nose", range(27, 36), (0.020, 0.018, 0.018))
    nasal_floor = points[31:36, 1].max() + 0.010 * scale
    add("mouth", range(48, 68), (0.050, 0.025, 0.045), lower_y=nasal_floor)
    add("mouth_chin", [*range(48, 68), 6, 7, 8, 9, 10],
        (0.040, 0.025, 0.025), lower_y=nasal_floor)
    for side, indices, bounds in [
        ("left", [2, 31, 40, 41], [points[2, 0], points[[40, 41], 1].mean() - .010 * scale,
                                   points[31, 0] - .010 * scale, points[31, 1] + .010 * scale]),
        ("right", [14, 35, 46, 47], [points[35, 0] + .010 * scale,
                                     points[[46, 47], 1].mean() - .010 * scale,
                                     points[14, 0], points[35, 1] + .010 * scale]),
    ]:
        regions[f"cheek_{side}"] = {
            "shape": "soft_bbox", "source_landmark_indices": indices,
            "support_bbox_512_unclipped": [float(v) for v in bounds],
            "support_bbox_512_clipped": np.clip(bounds, 0, 512).tolist(),
            "source_formula": "See fixed cheek_support and cheek_offset_scale in policy",
        }
    for region in regions.values():
        x0, y0, x1, y1 = region["support_bbox_512_clipped"]
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Source geometry produced an empty/reversed region")
    return regions


def _soft_box(bounds, edge_width):
    x0, y0, x1, y1 = bounds
    # Pixel centers, exactly compact support; no Gaussian tail into protected areas.
    y, x = np.mgrid[:512, :512].astype(np.float64) + 0.5
    distance = np.minimum.reduce([x - x0, x1 - x, y - y0, y1 - y])
    alpha = np.clip(distance / edge_width, 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def build_mask(case, source_record):
    """Return (numpy.float32[64,64], receipt) using source-only geometry.

    A zero means source-locked / protected; one means editable. The source image
    size is verified from its PNG header unless a verified image_size is supplied.
    File hashes are checked by prepare(); this pure geometry API binds declared
    source hashes and records the selected raw geometry for reproducibility.
    """
    source, target, changed = _controls(case)
    points, scale, selected_row, confidence = _source_geometry(case, source_record)
    all_regions = _regions(points, scale)
    names = sorted({region for au in changed for region in REGION_BY_AU[au]})
    mask512 = np.zeros((512, 512), dtype=np.float64)
    edge_width = POLICY["soft_edge_scale"] * scale
    for name in names:
        # Keep the edge at its original support boundary, clipping by image grid only.
        mask512 = np.maximum(mask512, _soft_box(all_regions[name]["support_bbox_512_unclipped"],
                                               edge_width))
    mask = mask512.reshape(64, 8, 64, 8).mean(axis=(1, 3)).astype(np.float32)
    if changed and not np.any(mask):
        raise ValueError("Nonzero AU edit produced an empty mask")
    receipt = {
        "version": VERSION, "policy_sha256": canonical_sha(POLICY),
        "case_id": case["case_id"], "split": case["split"], "group": case.get("group"),
        "source_only": True, "test_used": False, "target_geometry_used": False,
        "source_sha256": case["source_real"]["sha256"],
        "source_path": str(_allowed_path(case["source_real"]["path"])),
        "source_landmarks_68_512": points.tolist(), "selected_source_row": selected_row,
        "source_landmarks_sha256": canonical_sha(points.tolist()),
        "source_confidence": confidence, "face_scale_px": scale,
        "landmarks_outside_crop_indices": np.flatnonzero(np.any((points < 0) | (points >= 512), axis=1)).tolist(),
        "source_aus_0_5": source.tolist(), "target_aus_0_5": target.tolist(),
        "changed_aus": changed, "active_regions": names, "soft_edge_px": edge_width,
        "regions": {name: all_regions[name] for name in names},
        "per_au": [{"au": au, "source": float(source[AUS.index(au)]),
                    "target": float(target[AUS.index(au)]), "regions": list(REGION_BY_AU[au]),
                    "geometry": {name: all_regions[name] for name in REGION_BY_AU[au]}}
                   for au in changed],
        "mask_area": {"alpha_sum_64": float(mask.sum(dtype=np.float64)),
                      "alpha_mean_64": float(mask.mean(dtype=np.float64)),
                      "positive_fraction_64": float(np.mean(mask > 0)),
                      "at_least_half_fraction_64": float(np.mean(mask >= .5)),
                      "alpha_equivalent_pixels_512": float(mask.sum(dtype=np.float64) * 64)},
        "mask_bytes_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
        "interpretation": POLICY["interpretation"],
    }
    return mask, receipt


def prepare(cases_path, measurements_path, output):
    cases_path = _allowed_path(cases_path)
    measurements_path = _allowed_path(measurements_path)
    output = _allowed_path(output)
    cases_doc = json.loads(cases_path.read_text())
    measurements_doc = json.loads(measurements_path.read_text())
    if cases_doc.get("test_data_used") is not False or measurements_doc.get("test_used") is not False:
        raise ValueError("Both inputs must explicitly certify no test data used")
    cases = cases_doc["cases"]
    if len(cases) != 8 or cases_doc.get("case_count") != 8:
        raise ValueError("This frozen diagnostic preparation expects exactly 8 cases")
    if len({case["case_id"] for case in cases}) != 8:
        raise ValueError("Duplicate case IDs")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", case["case_id"]) for case in cases):
        raise ValueError("Unsafe case ID")
    # Filter solely on role; never inspect target/generated landmark fields.
    source_records = [r for r in measurements_doc["records"] if r.get("method") == "source_real"]
    by_case = {}
    for record in source_records:
        if record["case_id"] in by_case:
            raise ValueError("Duplicate source_real record for a case")
        by_case[record["case_id"]] = record
    unique_sources = {}
    prepared = []
    geometry_by_sha = {}
    for case in cases:
        _controls(case)
        path = _allowed_path(case["source_real"]["path"])
        sha = case["source_real"]["sha256"]
        if sha not in unique_sources:
            if len(unique_sources) >= 4:
                raise ValueError("At most 4 unique source images may be reused")
            if _png_size(path) != (512, 512):
                raise ValueError("Source image must be exactly 512x512")
            if file_sha(path) != sha:
                raise ValueError("Actual source file SHA-256 mismatch")
            unique_sources[sha] = str(path)
        elif unique_sources[sha] != str(path):
            raise ValueError("One source SHA has conflicting source paths")
        if case["case_id"] not in by_case:
            raise ValueError("Missing matching source_real measurement")
        record = dict(by_case[case["case_id"]], image_size=[512, 512])
        mask, receipt = build_mask(case, record)
        geometry_sha = receipt["source_landmarks_sha256"]
        if sha in geometry_by_sha and geometry_by_sha[sha] != geometry_sha:
            raise ValueError("Repeated source image has inconsistent selected landmarks")
        geometry_by_sha[sha] = geometry_sha
        prepared.append((case, mask, receipt))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty; existing mask artifacts are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw
    debug_sources = {}
    entries = []
    for case, mask, receipt in prepared:
        case_id = case["case_id"]
        npy = output / f"{case_id}.mask64.npy"
        np.save(npy, mask, allow_pickle=False)
        # Debug visualization of the EXACT 64x64 mask, nearest-neighbor expanded.
        mask512 = np.repeat(np.repeat(mask, 8, axis=0), 8, axis=1)
        png = output / f"{case_id}.mask512.png"
        Image.fromarray(np.rint(mask512 * 255).astype(np.uint8)).save(png)
        sha = receipt["source_sha256"]
        if sha not in debug_sources:
            with Image.open(unique_sources[sha]) as image:
                debug_sources[sha] = np.asarray(image.convert("RGB")).copy()
        source = debug_sources[sha].astype(np.float64)
        alpha = 0.5 * mask512[..., None]
        preview = np.rint(source * (1 - alpha) + np.array([255, 32, 32]) * alpha).astype(np.uint8)
        overlay = Image.fromarray(preview)
        draw = ImageDraw.Draw(overlay)
        for x, y in receipt["source_landmarks_68_512"]:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 255, 255))
        overlay_path = output / f"{case_id}.source_overlay.png"
        overlay.save(overlay_path)
        receipt_path = output / f"{case_id}.receipt.json"
        artifacts = {"mask64": {"path": str(npy), "sha256": file_sha(npy)},
                     "mask512_debug": {"path": str(png), "sha256": file_sha(png)},
                     "source_overlay_debug": {"path": str(overlay_path), "sha256": file_sha(overlay_path)}}
        receipt["artifacts"] = artifacts
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        entries.append({"case_id": case_id, "split": case["split"], "group": case.get("group"), "source_sha256": sha,
                        "changed_aus": receipt["changed_aus"], "active_regions": receipt["active_regions"],
                        "mask_area": receipt["mask_area"], "receipt_path": str(receipt_path),
                        "receipt_sha256": file_sha(receipt_path), **artifacts})
    manifest = {"kind": "source_only_geometric_region_mask_preparation", "status": "complete",
                "policy": POLICY, "policy_sha256": canonical_sha(POLICY), "source_only": True,
                "test_used": False, "target_geometry_used": False, "models_run": False,
                "unique_source_count": len(unique_sources), "case_count": len(entries),
                "source_unchanged": all(file_sha(path) == sha for sha, path in unique_sources.items()),
                "inputs": {"cases": {"path": str(cases_path), "sha256": file_sha(cases_path)},
                           "source_measurements": {"path": str(measurements_path), "sha256": file_sha(measurements_path)},
                           "code_sha256": file_sha(__file__)}, "cases": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


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
                      "manifest": str(Path(args.output) / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
