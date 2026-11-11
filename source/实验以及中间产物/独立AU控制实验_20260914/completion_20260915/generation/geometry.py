"""General-case V7/V11 source geometry: identical math, explicit new split IO."""
import numpy as np
from pathlib import Path
from common import AUS, COMPLETION, canonical_sha, module, need, sha

V7 = module('v7_geometry')
V11 = module('v11_geometry')
V6 = V7.BASE_V6


def build_masks(case, source, measurement):
    """Target AU labels are allowed; only SOURCE measurements/pixels are bound."""
    need(case['split'] in ('postfreeze_case_validation', 'dev'), 'Explicit validation/development role required')
    need(measurement['method'] == 'source_real', 'Only source_real geometry accepted')
    need(measurement['split'] == case['split'], 'Measurement role mismatch')
    canonical = source['canonical_source']
    path = Path(canonical['path']).resolve()
    need(COMPLETION in path.parents or case['split'] == 'dev', 'Validation source must be in completion directory')
    need(Path(measurement['path']).resolve() == path and measurement['sha256'] == canonical['sha256'], 'Source geometry binding mismatch')
    need(measurement.get('measurement_source_path', str(path)) == str(path), 'Measurement source mismatch')
    need(tuple(measurement.get('image_size', (512, 512))) == (512, 512), 'Geometry must be canonical512')
    need(measurement.get('returncode', 0) == 0 and measurement['parse_status'] == 'primary_face_finite', 'Source geometry measurement failed')
    confidence = float(measurement['confidence'])
    need(np.isfinite(confidence) and confidence >= V6.POLICY['minimum_source_confidence'], 'Low confidence source geometry')
    selected = measurement['selected_row']
    need(type(selected) is int and 0 <= selected < len(measurement['raw_rows']), 'Invalid geometry row')
    row = measurement['raw_rows'][selected]
    points = np.asarray([[float(row[f'x_{i}']), float(row[f'y_{i}'])] for i in range(68)], dtype=np.float64)
    need(np.isfinite(points).all() and not np.any(points < -64) and not np.any(points > 576), 'Nonfinite/out-of-range source geometry')
    scale = float(np.linalg.norm(points[16] - points[0]))
    need(64 <= scale <= 512 and np.ptp(points[:, 1]) >= 64, 'Degenerate source geometry')
    need(points[36:42, 0].mean() < points[42:48, 0].mean() and points[48:68, 1].mean() > points[36:48, 1].mean(), 'Invalid source topology')
    before = np.asarray(case['source_au_vector_0_5'], dtype=np.float64)
    after = np.asarray(case['target_au_vector_0_5'], dtype=np.float64)
    need(case['au_order'] == AUS and before.shape == after.shape == (12,), 'InvalidAU shapes/order')
    need(np.isfinite(before).all() and np.isfinite(after).all() and min(before.min(), after.min()) >= 0 and max(before.max(), after.max()) <= 5, 'InvalidAU range')
    changed = [au for au, a, b in zip(AUS, before, after) if a != b]
    regions = V6._regions(points, scale)
    active = sorted({name for au in changed for name in V6.REGION_BY_AU[au]})
    raster = np.zeros((512, 512), dtype=np.float64)
    for name in active:
        raster = np.maximum(raster, V6._soft_box(regions[name]['support_bbox_512_unclipped'], V6.POLICY['soft_edge_scale'] * scale))
    base = raster.reshape(64, 8, 64, 8).mean(axis=(1, 3)).astype(np.float32)
    need(not changed or np.any(base), 'Nonzero edit with empty source support')
    edit = V7.smooth_mask(base)
    cleanup, details, _ = V11.geometry_mask(points, scale, changed, edit)
    receipt = {'kind': 'completion_fixed_v7_v11_source_geometry', 'case_id': case['case_id'],
        'split': case['split'], 'source_only': True, 'target_geometry_used': False, 'generated_pixels_used': False,
        'source': canonical, 'source_au_vector_0_5': before.tolist(), 'target_au_vector_0_5': after.tolist(),
        'changed_aus': changed, 'source_landmarks_68_512': points.tolist(), 'source_landmarks_sha256': canonical_sha(points.tolist()),
        'selected_source_row': selected, 'source_confidence': confidence, 'face_scale_px': scale,
        'v7_policy_sha256': V7.canonical_sha(V7.POLICY), 'v11_policy_sha256': V11.canonical_sha(V11.POLICY),
        'edit_mask_tensor_sha256': __import__('hashlib').sha256(edit.tobytes()).hexdigest(),
        'cleanup_mask_tensor_sha256': __import__('hashlib').sha256(cleanup.tobytes()).hexdigest(),
        'active_edit_regions': active, 'cleanup': details,
        'scope_note': 'Frozen development-derived geometry applied unchanged; not an independent-subject claim'}
    return edit, cleanup, receipt
