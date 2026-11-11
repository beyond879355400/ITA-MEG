#!/usr/bin/env python3
"""Append the two single-direction variants under the unchanged v2 policy."""
import contextlib
import json
import math
import os
from pathlib import Path
import sys

from run_facenet_v2 import HERE, ROOT, POLICY_SHA, sha, save

INPUT_SHA = 'cd7365706c5b04540f6d97ff90f4f8d3ade5513b08f57179479395e702f18eb5'


def main():
    policy_path = HERE / 'policy.json'
    assert sha(policy_path) == POLICY_SHA
    policy = json.loads(policy_path.read_text())
    for folder in ('real_dev_openface', 'facenet_real12'):
        summary = json.loads((HERE / folder / 'summary.json').read_text())
        assert summary['status'] == 'complete' and summary['policy_sha256'] == POLICY_SHA
    root = HERE / 'single_directions'
    source = root / 'items.json'
    assert sha(source) == INPUT_SHA
    document = json.loads(source.read_text())
    items = document['items']
    assert len(items) == 57 and not document['test_used']
    assert all(row['split'] == 'dev' for row in items)
    assert len({(row['case_id'], row['method']) for row in items}) == 57
    out = root / 'facenet'
    out.mkdir(parents=True, exist_ok=False)
    unique = {}
    for row in items:
        assert sha(row['path']) == row['sha256']
        unique.setdefault(row['sha256'], row)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1',
                      MPLCONFIGDIR=str(HERE / 'shared_cache/matplotlib'), XDG_CACHE_HOME=str(HERE / 'shared_cache/xdg'),
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    face_root = ROOT / 'tools/facenet'
    assert sha(face_root / 'tool_seal.json') == policy['identity_primary']['tool_seal_sha256']
    tool = json.loads((face_root / 'tool_seal.json').read_text())
    for asset in tool['source_files'] + tool['weights']:
        assert sha(face_root / asset['path']) == asset['sha256']
    sys.path.insert(0, str(face_root / 'runtime'))
    import requests
    import urllib.request
    def no_network(*args, **kwargs):
        raise RuntimeError('Offline evaluation only')
    requests.sessions.Session.request = no_network
    urllib.request.urlopen = no_network
    import numpy as np
    import torch
    from PIL import Image
    from facenet_pytorch import MTCNN, InceptionResnetV1
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    with (out / 'runtime.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        detector = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[.6, .7, .7], factor=.709,
                         post_process=True, keep_all=True, device='cpu').eval()
        model = InceptionResnetV1(pretrained=None, classify=True, num_classes=8631).eval()
        model.load_state_dict(torch.load(face_root / 'models/20180402-114759-vggface2.pt', map_location='cpu', weights_only=True), strict=True)
        model.classify = False
        model.requires_grad_(False)
    save(out / 'protocol.json', {'policy_sha256': POLICY_SHA, 'input_sha256': INPUT_SHA, 'script_sha256': sha(__file__),
              'input_logical_count': 57, 'unique_image_count': len(unique), 'input_coordinate_note_sha256': sha(HERE / 'INPUT_COORDINATE_NOTE.md'),
              'geometry_policy': policy['identity_primary'], 'realdev_openface_summary_sha256': sha(HERE / 'real_dev_openface/summary.json'),
              'real12_facenet_summary_sha256': sha(HERE / 'facenet_real12/summary.json'), 'test_used': False,
              'reuse': 'SHA-deduplicate within this manifest; preserve all logical rows; no re-alignment or padding'})
    (out / 'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out / 'policy_snapshot.json').write_bytes(policy_path.read_bytes())
    measurements, embeddings = {}, {}
    for digest, item in unique.items():
        record = {'sha256': digest, 'measurement_source_path': item['path'], 'policy_sha256': POLICY_SHA,
                  'face_count': None, 'candidates': [], 'selected_index': None, 'status': 'pending'}
        try:
            image = Image.open(item['path']).convert('RGB')
            image.load()
            record['input_image_size'] = list(image.size)
            boxes, probs, points = detector.detect(image, landmarks=True)
            record['face_count'] = 0 if boxes is None else len(boxes)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    raw = [float(v) for v in box]
                    valid = all(math.isfinite(v) for v in raw)
                    area = (raw[2] - raw[0]) * (raw[3] - raw[1]) if valid else None
                    valid = valid and math.isfinite(area) and area > 0
                    record['candidates'].append({'index': i, 'bbox': [v if math.isfinite(v) else None for v in raw],
                        'area': area if valid else None, 'geometry_rankable': valid,
                        'probability': float(probs[i]) if math.isfinite(float(probs[i])) else None,
                        'landmarks': [[float(v) if math.isfinite(float(v)) else None for v in xy] for xy in points[i]]})
            ranked = [row for row in record['candidates'] if row['geometry_rankable']]
            selected = max(ranked, key=lambda r: (r['area'], -r['index'])) if ranked else None
            record['selected_index'] = selected['index'] if selected else None
            record['status'] = 'no_face' if not record['face_count'] else 'no_valid_geometry' if not selected else 'selected'
            if selected is not None:
                feature = model(detector.extract(image, boxes[selected['index']:selected['index'] + 1], save_path=None))[0]
                norm = float(feature.norm())
                if feature.shape == (512,) and torch.isfinite(feature).all() and math.isfinite(norm) and norm > 0:
                    embeddings[digest] = feature.double().numpy()
                    record.update(status='valid', embedding_norm=norm)
                else:
                    record['status'] = 'selected_feature_invalid'
        except Exception as error:
            record.update(status='error', error_type=type(error).__name__, error=str(error))
        assert sha(item['path']) == digest
        measurements[digest] = record
    records = [{**row, **measurements[row['sha256']], 'path': row['path'], 'identity_cosine': None, 'policy_sha256': POLICY_SHA} for row in items]
    sources = {row['case_id']: row['sha256'] for row in records if row['method'] == 'source_real'}
    assert len(sources) == 19
    for row in records:
        source_sha = sources[row['case_id']]
        if row['sha256'] in embeddings and source_sha in embeddings:
            a, b = embeddings[row['sha256']], embeddings[source_sha]
            row['identity_cosine'] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    coverage = {m: {'planned': sum(r['method'] == m for r in records),
                     'valid_identity': sum(r['method'] == m and r['identity_cosine'] is not None for r in records)}
                for m in sorted({r['method'] for r in records})}
    summary = {'status': 'complete', 'policy_sha256': POLICY_SHA, 'logical_records': 57, 'unique_images': len(unique),
               'valid_unique_images': len(embeddings), 'coverage': coverage, 'test_used': False}
    save(out / 'results.json', {'policy_sha256': POLICY_SHA, 'records': records, 'unique_measurements': list(measurements.values()),
                               'summary': summary, 'test_used': False, 'embeddings_saved': False})
    save(out / 'summary.json', summary)
    assert sha(policy_path) == POLICY_SHA and sha(source) == INPUT_SHA
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
