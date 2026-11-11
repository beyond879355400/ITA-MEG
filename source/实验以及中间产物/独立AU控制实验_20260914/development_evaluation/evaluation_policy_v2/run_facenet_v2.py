#!/usr/bin/env python3
"""FaceNet stages bound to the frozen evaluation_policy_v2; CPU, geometry-only."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
POLICY_SHA = 'd29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c'
sys.path.insert(0, str(HERE.parent / 'real_dev_calibration'))
from calibration_metrics import quantile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, payload):
    with path.open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('real12', 'dev19'), required=True)
    args = p.parse_args()
    policy_path = HERE / 'policy.json'
    assert sha(policy_path) == POLICY_SHA
    policy = json.loads(policy_path.read_text())
    cfg = policy['identity_primary']
    assert not policy['final_test_accessed'] and cfg['image_size'] == 160 and cfg['margin'] == 0
    assert cfg['thresholds'] == [.6, .7, .7] and cfg['factor'] == .709 and cfg['device'] == 'cpu'
    assert cfg['identity_acceptance_threshold'] is None
    if args.stage == 'dev19':
        of_path = HERE / 'real_dev_openface/summary.json'
        if not of_path.exists():
            raise RuntimeError('Await complete OpenFace real-dev calibration; do not run dev19 yet')
        of_summary = json.loads(of_path.read_text())
        assert of_summary['status'] == 'complete' and of_summary['policy_sha256'] == POLICY_SHA
        own = json.loads((HERE / 'facenet_real12/summary.json').read_text())
        assert own['status'] == 'complete' and own['policy_sha256'] == POLICY_SHA
    input_name = 'facenet_real12_items.json' if args.stage == 'real12' else 'dev19_items.json'
    input_path = HERE / input_name
    assert sha(input_path) == policy['inputs'][input_name]['sha256']
    document = json.loads(input_path.read_text())
    assert not document['test_used']
    rows = document['images'] if args.stage == 'real12' else document['items']
    assert len(rows) == (12 if args.stage == 'real12' else 95)
    if args.stage == 'dev19':
        assert all(r['split'] == 'dev' for r in rows)
        assert len({(r['case_id'], r['method']) for r in rows}) == 95
    by_sha = {}
    for row in rows:
        path = Path(row['path'])
        stat = path.stat()
        assert not getattr(stat, 'st_flags', 0) & 0x40000000 and stat.st_blocks > 0 and sha(path) == row['sha256']
        by_sha.setdefault(row['sha256'], row)
    out = HERE / ('facenet_real12' if args.stage == 'real12' else 'facenet_dev19')
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1',
                      MPLCONFIGDIR=str(out / 'cache/matplotlib'), XDG_CACHE_HOME=str(out / 'cache/xdg'),
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    face_root = ROOT / 'tools/facenet'
    assert sha(face_root / 'tool_seal.json') == cfg['tool_seal_sha256']
    seal = json.loads((face_root / 'tool_seal.json').read_text())
    for asset in seal['source_files'] + seal['weights']:
        assert sha(face_root / asset['path']) == asset['sha256']
    sys.path.insert(0, str(face_root / 'runtime'))
    import requests
    import urllib.request
    def no_network(*args, **kwargs):
        raise RuntimeError('FaceNet v2 evaluation is offline')
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
                         keep_all=True, post_process=True, device='cpu').eval()
        model = InceptionResnetV1(pretrained=None, classify=True, num_classes=8631).eval()
        model.load_state_dict(torch.load(face_root / 'models/20180402-114759-vggface2.pt', map_location='cpu', weights_only=True), strict=True)
        model.classify = False
        model.requires_grad_(False)
    protocol = {'policy_sha256': POLICY_SHA, 'stage': args.stage, 'input_sha256': sha(input_path),
                'input_manifest': str(input_path), 'tool_seal_sha256': cfg['tool_seal_sha256'], 'script_sha256': sha(__file__),
                'planned_logical_records': len(rows), 'unique_image_sha_count': len(by_sha),
                'duplicate_measurement_reuse': 'same image SHA measured once; every logical row preserved',
                'device': 'cpu', 'threads': 2, 'primary_face_selection': cfg['selection'],
                'input': 'sealed original PIL RGB, no pre-alignment/padding; only real references are canonical512',
                'coordinate_note_sha256': sha(HERE / 'INPUT_COORDINATE_NOTE.md'),
                'standardization': cfg['standardization'], 'identity_threshold_chosen': False, 'test_used': False,
                'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow')}}
    if args.stage == 'dev19':
        protocol['real_dev_openface_summary_sha256'] = sha(HERE / 'real_dev_openface/summary.json')
        protocol['real12_facenet_summary_sha256'] = sha(HERE / 'facenet_real12/summary.json')
    save(out / 'protocol.json', protocol)
    (out / 'policy_snapshot.json').write_bytes(policy_path.read_bytes())
    (out / 'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    measurements, embeddings = {}, {}
    for digest, row in by_sha.items():
        result = {'sha256': digest, 'measurement_source_path': row['path'], 'policy_sha256': POLICY_SHA,
                  'face_count': None, 'candidates': [], 'selected_index': None, 'status': 'pending'}
        try:
            image = Image.open(row['path']).convert('RGB')
            image.load()
            result['input_image_size'] = list(image.size)
            boxes, probs, landmarks = detector.detect(image, landmarks=True)
            result['face_count'] = 0 if boxes is None else len(boxes)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    values = [float(v) for v in box]
                    valid = all(math.isfinite(v) for v in values)
                    area = (values[2] - values[0]) * (values[3] - values[1]) if valid else None
                    rankable = valid and math.isfinite(area) and area > 0
                    result['candidates'].append({'index': i, 'bbox': [v if math.isfinite(v) else None for v in values],
                        'bbox_nonfinite': not valid, 'area': area if rankable else None, 'geometry_rankable': rankable,
                        'probability': float(probs[i]) if math.isfinite(float(probs[i])) else None,
                        'landmarks': [[float(v) if math.isfinite(float(v)) else None for v in xy] for xy in landmarks[i]]})
            rankable = [r for r in result['candidates'] if r['geometry_rankable']]
            selected = max(rankable, key=lambda r: (r['area'], -r['index'])) if rankable else None
            result['selected_index'] = selected['index'] if selected else None
            result['status'] = 'no_face' if not result['face_count'] else 'no_valid_geometry' if not selected else 'selected'
            if selected is not None:
                crop = detector.extract(image, boxes[selected['index']:selected['index'] + 1], save_path=None)
                feature = model(crop)[0]
                norm = float(feature.norm())
                if feature.shape != (512,) or not torch.isfinite(feature).all() or not math.isfinite(norm) or norm <= 0:
                    result['status'] = 'selected_feature_invalid'
                else:
                    embeddings[digest] = feature.double().numpy()
                    result.update(embedding_norm=norm, status='valid')
        except Exception as error:
            result.update(status='error', error_type=type(error).__name__, error=str(error))
        assert sha(row['path']) == digest
        measurements[digest] = result
    records = [{**row, **measurements[row['sha256']], 'path': row['path'], 'identity_cosine': None, 'policy_sha256': POLICY_SHA} for row in rows]
    if args.stage == 'real12':
        comparisons = []
        for a, b in itertools.combinations(records, 2):
            score = None
            if a['sha256'] in embeddings and b['sha256'] in embeddings:
                x, y = embeddings[a['sha256']], embeddings[b['sha256']]
                score = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
            comparisons.append({'left_sha256': a['sha256'], 'right_sha256': b['sha256'],
                                'genuine': a['subject'] == b['subject'], 'cosine': score})
        genuine = [r['cosine'] for r in comparisons if r['genuine'] and r['cosine'] is not None]
        impostor = [r['cosine'] for r in comparisons if not r['genuine'] and r['cosine'] is not None]
        summary = {'status': 'complete', 'policy_sha256': POLICY_SHA, 'planned_images': 12, 'valid_images': len(embeddings),
                   'genuine_pairs': len(genuine), 'impostor_pairs': len(impostor), 'threshold_chosen': False,
                   'genuine_distribution': {str(q): quantile(genuine, q) for q in (0, .05, .5, .95, 1)},
                   'impostor_distribution': {str(q): quantile(impostor, q) for q in (0, .5, .95, .99, 1)}, 'test_used': False}
        save(out / 'results.json', {'policy_sha256': POLICY_SHA, 'records': records, 'comparisons': comparisons,
                                   'embeddings_saved': False, 'test_used': False})
    else:
        sources = {r['case_id']: r['sha256'] for r in records if r['method'] == 'source_real'}
        assert len(sources) == 19
        for record in records:
            source = sources[record['case_id']]
            if record['sha256'] in embeddings and source in embeddings:
                x, y = embeddings[record['sha256']], embeddings[source]
                record['identity_cosine'] = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        coverage = {m: {'planned': sum(r['method'] == m for r in records),
                         'valid_identity': sum(r['method'] == m and r['identity_cosine'] is not None for r in records)}
                    for m in sorted({r['method'] for r in records})}
        summary = {'status': 'complete', 'policy_sha256': POLICY_SHA, 'logical_records': 95, 'unique_images': len(by_sha),
                   'valid_unique_images': len(embeddings), 'coverage': coverage, 'test_used': False}
        save(out / 'results.json', {'policy_sha256': POLICY_SHA, 'records': records, 'unique_measurements': list(measurements.values()),
                                   'summary': summary, 'embeddings_saved': False, 'test_used': False})
    assert sha(policy_path) == POLICY_SHA and sha(input_path) == protocol['input_sha256']
    save(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
