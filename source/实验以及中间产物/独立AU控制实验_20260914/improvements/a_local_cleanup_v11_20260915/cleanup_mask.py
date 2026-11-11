"""Frozen V11 source68 engineering cleanup support, never anatomical causality."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V6 = HERE.parent / 'region_preserving_v6_20260915/region_mask.py'
V10_CASES = HERE.parent / 'a_terminal_cleanup_v10_20260915/runs/terminal_202609151940/cases.json'
V7_MANIFEST = HERE.parent / 'smooth_regions_v7_20260915/masks/first8/manifest.json'
V6_SHA = '75f7995e3addb7656e0935da62e920f9d43b540ffa9b5e127093835afb3e438e'
CASES_SHA = '227a371225b21066656863cabcede26a3ababb7c6ca785645e02515387ed6982'
V7_MANIFEST_SHA = 'f314521f0d98ecfd0c393d6a3b36464edd2df6fd24a01ee46d524d215b3e086c'
VERSION = 'source68_local_glabella_chin_v11_fixed_v1'
POLICY = {
    'version': VERSION, 'interpretation': 'Engineering local latent regularization support, not anatomical or causal AU segmentation',
    'source_only': True, 'test_used': False, 'target_geometry_used': False,
    'generated_pixels_used': False, 'subject_exceptions': False, 'outcome_tuning': False,
    'au_order': [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26],
    'coordinate_frame': 'origin=mean eye centers; u=left36:42 to right42:48 unit vector; v=perpendicular down',
    'scale': 'Euclidean source landmark0 to16 distance',
    'glabella': {'trigger': [1, 2, 4], 'u': 'u21+0.015S to u22-0.015S',
                 'v': 'min(v21,v22)-0.04S to v27+0.015S'},
    'chin': {'trigger': [17], 'u': 'central70percent of mouth48:54, clipped to jaw6:10',
             'v': 'max(v55:60)+0.035S to v8-0.035S'},
    'protected': {'brows': 'source17:22 and22:27 polylines buffered0.025S',
                  'eyes': 'source36:42 and42:48 closed polygons plus0.025S edge buffers',
                  'lips': 'source48:60 closed outer polygon plus0.025S edge buffer'},
    'soft_edge': 'compact inward smoothstep0.015S',
    'downsample': 'exact8x8 area average512to64 then intersect(editmask64>0), no dilation',
    'cropping': '512pixel raster clips support; landmarks outside frame within frozen64px tolerance are recorded',
    'no_edit': 'zero when source and target AU controls match; unrelated AU changes also yield zero cleanup support',
    'activation': 'trigger AU differs between original source and target controls; magnitude/sign do not reshape support',
    'filter_configs': [{'name': 'local_lp_s050', 'sigma_latent': .5, 'radius': 2},
                       {'name': 'local_lp_s075', 'sigma_latent': .75, 'radius': 3}],
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


need(sha(V6) == V6_SHA, 'Frozen V6 source geometry validator changed')
spec = importlib.util.spec_from_file_location('v11_fixed_source68_validator', V6)
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)
canonical_sha = BASE.canonical_sha


def _segment_distance(u, v, a, b):
    delta = b - a
    length2 = float(delta @ delta)
    if length2 <= 1e-12:
        return np.hypot(u - a[0], v - a[1])
    t = np.clip(((u - a[0]) * delta[0] + (v - a[1]) * delta[1]) / length2, 0, 1)
    return np.hypot(u - (a[0] + t * delta[0]), v - (a[1] + t * delta[1]))


def _protected_shape(u, v, points, radius, closed):
    pairs = list(zip(points[:-1], points[1:]))
    if closed:
        pairs.append((points[-1], points[0]))
    distance = np.full(u.shape, np.inf)
    inside = np.zeros(u.shape, dtype=bool)
    for a, b in pairs:
        distance = np.minimum(distance, _segment_distance(u, v, a, b))
        if closed:
            crossing = (a[1] > v) != (b[1] > v)
            denominator = b[1] - a[1]
            if abs(denominator) > 1e-12:
                inside ^= crossing & (u < a[0] + (v - a[1]) * (b[0] - a[0]) / denominator)
    return (distance <= radius) | inside


def geometry_mask(points, scale, changed, edit_mask):
    """Pure geometry; no subject, text, target landmark or generated-pixel input."""
    points, edit_mask = np.asarray(points, dtype=np.float64), np.asarray(edit_mask)
    need(points.shape == (68, 2) and np.isfinite(points).all() and math.isfinite(scale) and scale > 0,
         'Invalid source geometry')
    need(edit_mask.shape == (64, 64) and edit_mask.dtype == np.float32 and np.isfinite(edit_mask).all()
         and edit_mask.min() >= 0 and edit_mask.max() <= 1, 'Expected fixed finitefloat32 editmask64')
    left, right = points[36:42].mean(0), points[42:48].mean(0)
    direction = right - left
    need(np.linalg.norm(direction) > 1, 'Degenerate eye coordinate frame')
    ex = direction / np.linalg.norm(direction)
    ey = np.asarray([-ex[1], ex[0]])
    need(ey[1] > 0, 'Expected down-facing source local coordinate axis')
    origin, basis = (left + right) / 2, np.stack((ex, ey))
    p = (points - origin) @ basis.T
    yy, xx = np.mgrid[:512, :512].astype(np.float64) + .5
    u = (xx - origin[0]) * ex[0] + (yy - origin[1]) * ex[1]
    v = (xx - origin[0]) * ey[0] + (yy - origin[1]) * ey[1]
    protection = np.zeros((512, 512), dtype=bool)
    for indices, closed in ((range(17, 22), False), (range(22, 27), False),
                            (range(36, 42), True), (range(42, 48), True), (range(48, 60), True)):
        protection |= _protected_shape(u, v, p[list(indices)], .025 * scale, closed)
    mouth_left, mouth_right = float(p[48, 0]), float(p[54, 0])
    need(mouth_right > mouth_left, 'Invalid source mouth corner order')
    mouth_width = mouth_right - mouth_left
    boxes = {
        'glabella': [p[21, 0] + .015 * scale, min(p[21, 1], p[22, 1]) - .04 * scale,
                     p[22, 0] - .015 * scale, p[27, 1] + .015 * scale],
        'chin': [max(p[6, 0], mouth_left + .15 * mouth_width), p[55:60, 1].max() + .035 * scale,
                 min(p[10, 0], mouth_right - .15 * mouth_width), p[8, 1] - .035 * scale],
    }
    active = []
    if set(changed).intersection((1, 2, 4)):
        active.append('glabella')
    if 17 in changed:
        active.append('chin')
    result512 = np.zeros((512, 512), dtype=np.float64)
    regions = {}
    for name in active:
        x0, y0, x1, y1 = [float(value) for value in boxes[name]]
        valid = x1 > x0 and y1 > y0
        if valid:
            distance = np.minimum.reduce([u - x0, x1 - u, v - y0, y1 - v])
            edge = np.clip(distance / (.015 * scale), 0., 1.)
            part = edge * edge * (3 - 2 * edge)
            part[protection] = 0
            result512 = np.maximum(result512, part)
        corners_local = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        corners_image = corners_local @ basis + origin
        regions[name] = {'local_box_unclipped': [x0, y0, x1, y1], 'image_corners_unclipped': corners_image.tolist(),
                         'box_valid': valid, 'source_support_clipped_by_frame': bool(np.any(corners_image < 0) or np.any(corners_image > 512))}
    result = result512.reshape(64, 8, 64, 8).mean(axis=(1, 3)).astype(np.float32)
    result[edit_mask <= 0] = 0
    need(np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1, 'Invalid cleanup output')
    details = {'active_regions': active, 'regions': regions, 'frame_origin_512': origin.tolist(),
               'frame_basis_uv': basis.tolist(), 'protected_radius_px': .025 * scale, 'soft_edge_px': .015 * scale,
               'source_landmarks_outside_frame': np.flatnonzero(np.any((points < 0) | (points >= 512), axis=1)).tolist(),
               'mask_area': {'alpha_sum_64': float(result.sum(dtype=np.float64)),
                             'alpha_mean_64': float(result.mean(dtype=np.float64)),
                             'positive_fraction_64': float(np.mean(result > 0))},
               'empty_support': not bool(np.any(result)),
               'raster_protected_pixels_overlap': int(np.count_nonzero(result512[protection]))}
    return result, details, protection


def build_mask(case, source_record, edit_mask):
    source, target, changed = BASE._controls(case)
    points, scale, selected, confidence = BASE._source_geometry(case, source_record)
    mask, receipt, protected = geometry_mask(points, scale, changed, edit_mask)
    receipt.update(version=VERSION, policy_sha256=canonical_sha(POLICY), case_id=case['case_id'],
        source_only=True, test_used=False, target_geometry_used=False, generated_pixels_used=False,
        source_sha256=case['source_real']['sha256'], source_landmarks_68_512=points.tolist(),
        source_landmarks_sha256=canonical_sha(points.tolist()), selected_source_row=selected,
        source_confidence=confidence, face_scale_px=scale, source_aus_0_5=source.tolist(),
        target_aus_0_5=target.tolist(), changed_aus=changed, mask_bytes_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    return mask, receipt, protected


def verified_documents():
    need(sha(V10_CASES) == CASES_SHA and sha(V7_MANIFEST) == V7_MANIFEST_SHA, 'Frozen V10 cases or V7 mask manifest changed')
    cases_doc = json.loads(V10_CASES.read_text())
    need(cases_doc['kind'] == 'v10_development_terminal_cases' and cases_doc['test_data_used'] is False
         and len(cases_doc['cases']) == 8, 'Original8dev required')
    edit_doc = json.loads(V7_MANIFEST.read_text())
    item = edit_doc['inputs']['source_measurements']
    path = BASE._allowed_path(item['path'])
    need(sha(path) == item['sha256'], 'Frozen source measurement changed')
    records = {r['case_id']: r for r in json.loads(path.read_text())['records'] if r['method'] == 'source_real'}
    entries = {row['case_id']: row for row in edit_doc['cases']}
    return cases_doc['cases'], records, entries, item


def save_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def prepare(output):
    from PIL import Image, ImageDraw
    output = Path(output).resolve()
    need(output.parent == HERE / 'masks' and not output.exists(), 'Exclusive V11 masks/name output required')
    cases, records, edits, measurements = verified_documents()
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for case in cases:
        source_path = BASE._allowed_path(case['source_real']['path'])
        need(sha(source_path) == case['source_real']['sha256'], 'Source image changed')
        old = edits[case['case_id']]['mask64']
        need(sha(old['path']) == old['sha256'], 'Fixed V7 editmask changed')
        edit = np.load(old['path'], allow_pickle=False)
        mask, receipt, protected = build_mask(case, records[case['case_id']], edit)
        folder = output / case['case_id']; folder.mkdir()
        path = folder / 'cleanup64.npy'
        with path.open('xb') as stream:
            np.save(stream, mask, allow_pickle=False)
        Image.fromarray(np.round(mask * 255).astype(np.uint8)).save(folder / 'cleanup64.png')
        image = Image.open(source_path).convert('RGB'); need(image.size == (512, 512), 'Expected frozen512 source')
        source_pixels = np.asarray(image).astype(np.float64)
        display_mask = np.repeat(np.repeat(mask, 8, 0), 8, 1)
        overlay = source_pixels * (1 - .65 * display_mask[..., None]) + np.asarray([40., 225., 105.]) * (.65 * display_mask[..., None])
        overlay[protected] = .82 * overlay[protected] + .18 * np.asarray([225., 50., 90.])
        canvas = Image.new('RGB', (1536, 548), 'white')
        canvas.paste(image, (0, 36)); canvas.paste(Image.fromarray(np.round(overlay).astype(np.uint8)), (512, 36))
        canvas.paste(Image.fromarray(np.round(display_mask * 255).astype(np.uint8)).convert('RGB'), (1024, 36))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), 'SOURCE ONLY | ' + case['case_id'], fill='black')
        draw.text((520, 8), 'GREEN cleanup | RED protected | ' + ','.join(receipt['active_regions']), fill='black')
        draw.text((1032, 8), 'Actual64 mask enlarged8x; not generated image', fill='black')
        for i in (8, 17, 21, 22, 26, 27, 36, 39, 42, 45, 48, 54, 57):
            x, y = receipt['source_landmarks_68_512'][i]
            draw.ellipse((512 + x - 2, 36 + y - 2, 512 + x + 2, 36 + y + 2), fill='yellow')
        debug = folder / 'source_support_debug.png'; canvas.save(debug)
        receipt.update(cleanup_mask={'path': str(path), 'sha256': sha(path)}, edit_mask=old,
                       source_debug={'path': str(debug), 'sha256': sha(debug)}, source_image_unchanged=sha(source_path) == case['source_real']['sha256'])
        receipt_path = folder / 'receipt.json'; save_new(receipt_path, receipt)
        entries.append({'case_id': case['case_id'], 'path': str(path), 'sha256': sha(path),
                        'source_sha256': case['source_real']['sha256'], 'active_regions': receipt['active_regions'],
                        'changed_aus': receipt['changed_aus'], 'policy_sha256': receipt['policy_sha256'],
                        'mask_area': receipt['mask_area'], 'receipt': {'path': str(receipt_path), 'sha256': sha(receipt_path)},
                        'source_debug': {'path': str(debug), 'sha256': sha(debug)}})
    document = {'kind': 'v11_source_only_cleanup_masks', 'status': 'complete', 'case_count': 8,
        'source_only': True, 'test_used': False, 'target_geometry_used': False, 'generated_pixels_used': False,
        'policy': POLICY, 'policy_sha256': canonical_sha(POLICY), 'code': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        'inputs': {'cases': {'path': str(V10_CASES), 'sha256': CASES_SHA},
                   'edit_masks': {'path': str(V7_MANIFEST), 'sha256': V7_MANIFEST_SHA}, 'source_measurements': measurements},
        'cases': entries}
    save_new(output / 'manifest.json', document)
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    doc = prepare(args.output)
    print(json.dumps({'status': doc['status'], 'cases': len(doc['cases']), 'policy_sha256': doc['policy_sha256']}))
