#!/usr/bin/env python3
"""Single-case diagnostic: same canonical reference, OpenFace and independent FaceNet."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys

from prepare_dev_contracts import HERE, EXPERIMENT, sha, save


def main():
    source_case = 'fresh_v2_dev_SN004_s00745_t04266'
    cache_path = EXPERIMENT / 'training_cache/train2k_batched_202609141853/manifest.json'
    main_evaluation = EXPERIMENT / 'development_evaluation/v4_slot_2k_step200_dev19'
    main_results = json.loads((main_evaluation / 'results.json').read_text())
    main_protocol = json.loads((main_evaluation / 'protocol.json').read_text())
    cache = json.loads(cache_path.read_text())
    assert not main_results['test_used'] and not cache['test_data_used']
    assert sha(cache_path) == main_protocol['cache_manifest_sha256'] and cache['cache_sha256'] == main_protocol['cache_sha256']
    pair = next(x for x in cache['pairs'] if x['case_id'] == source_case)
    assert pair['split'] == 'dev'
    original = {x['method']: x for x in main_results['records'] if x['case_id'] == source_case}
    assert original['source_real']['sha256'] == pair['aligned_source']['aligned_sha256']
    source_labels, target_labels = pair['source_au_vector_0_5'], pair['target_au_vector_0_5']
    magic_dir = HERE / 'runs/magicface_smoke_native_202609142002'
    magic = json.loads((magic_dir / 'generation.json').read_text())
    live_dir = HERE / 'runtime/runs/liveportrait_smoke_raw_native_202609142014'
    live = json.loads((live_dir / 'summary.json').read_text())['results'][0]
    assert magic['status'] == 'complete' and live['status'] == 'success'
    assert magic['request_case_id'] == source_case and magic['target_aus'] == target_labels and magic['source_aus'] == source_labels
    assert live['input']['request_case_id'] == source_case
    items = []
    for method in ('source_real', 'baseline', 'current_model'):
        row = original[method]
        assert sha(row['path']) == row['sha256']
        items.append({'case_id': source_case, 'subject': row['subject'], 'method': method, 'path': row['path'], 'sha256': row['sha256'],
                      'split': 'dev', 'group': row['group'], 'source_labels_12': source_labels, 'target_labels_12': target_labels})
    for method, path, expected in [('magicface', magic['output'], magic['output_sha256']), ('liveportrait', live['output'], live['output_sha256'])]:
        assert sha(path) == expected
        items.append({'case_id': source_case, 'subject': pair['subject'], 'method': method, 'path': path, 'sha256': expected,
                      'split': 'dev', 'group': original['source_real']['group'], 'source_labels_12': source_labels, 'target_labels_12': target_labels})
    base = HERE / ('evaluation/smoke_pair_' + datetime.now().strftime('%Y%m%d%H%M'))
    out = base
    i = 2
    while out.exists():
        out = base.with_name(base.name + f'-{i:02d}')
        i += 1
    out.mkdir(parents=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1',
                      MPLCONFIGDIR=str(HERE / 'cache/matplotlib'), XDG_CACHE_HOME=str(HERE / 'cache/xdg'),
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(EXPERIMENT / 'development_evaluation'))
    from evaluate_patch_variant import e
    seal = json.loads((e.TOOLS / 'tool_seal.json').read_text())
    for row in [seal['binary'], seal['launcher'], *seal['model_files']]:
        assert sha(e.TOOLS / row['path']) == row['sha256']
    records = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for task in as_completed([executor.submit(e.openface, item, out) for item in items]):
            records.append(task.result())
    measured = {row['method']: row for row in records}
    for name in ('source_real', 'baseline', 'current_model'):
        assert measured[name]['vector'] == original[name]['vector'], 'Main OpenFace measurements did not reproduce'
    for row in records:
        row.update(e.au_metrics(row['vector'], measured['source_real']['vector'], source_labels, target_labels))
        row['all_measured_au_r'] = {k: float(v) for k, v in row['raw_rows'][0].items() if k.startswith('AU') and k.endswith('_r')} if row['vector'] is not None else None
    face_root = EXPERIMENT / 'tools/facenet'
    face_seal = json.loads((face_root / 'tool_seal.json').read_text())
    for row in face_seal['source_files'] + face_seal['weights']:
        assert sha(face_root / row['path']) == row['sha256']
    sys.path.insert(0, str(face_root / 'runtime'))
    import numpy as np
    import torch
    from PIL import Image
    from facenet_pytorch import MTCNN, InceptionResnetV1
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    with (out / 'facenet_runtime.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[.6, .7, .7], factor=.709,
                      post_process=True, keep_all=True, device='cpu').eval()
        model = InceptionResnetV1(pretrained=None, classify=True, num_classes=8631).eval()
        model.load_state_dict(torch.load(face_root / 'models/20180402-114759-vggface2.pt', map_location='cpu', weights_only=True), strict=True)
        model.classify = False
        model.requires_grad_(False)
    embeddings, face_records = {}, {}
    for item in items:
        image = Image.open(item['path']).convert('RGB')
        boxes, probs, points = mtcnn.detect(image, landmarks=True)
        count = 0 if boxes is None else len(boxes)
        row = {'image_size': list(image.size), 'face_count': count, 'status': 'no_face' if count == 0 else 'multiple_faces' if count != 1 else 'single_face',
               'boxes': None if boxes is None else boxes.tolist(), 'probabilities': None if probs is None else probs.tolist(), 'identity_cosine': None}
        if count == 1:
            crop = mtcnn.extract(image, boxes, save_path=None)
            feature = model(crop)[0]
            if feature.shape != (512,) or not torch.isfinite(feature).all():
                row['status'] = 'invalid_feature'
            else:
                row['norm'] = float(feature.norm())
                embeddings[item['method']] = feature.double().numpy()
        face_records[item['method']] = row
    for row in records:
        method = row['method']
        if method in embeddings and 'source_real' in embeddings:
            a, b = embeddings[method], embeddings['source_real']
            face_records[method]['identity_cosine'] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        row['facenet'] = face_records[method]
        assert sha(row['path']) == row['sha256']
    records.sort(key=lambda row: ['source_real', 'baseline', 'current_model', 'magicface', 'liveportrait'].index(row['method']))
    save(out / 'protocol.json', {'case_id': source_case, 'items': items, 'cache_manifest_sha256': sha(cache_path),
           'main_evaluation_sha256': sha(main_evaluation / 'results.json'), 'magicface_generation_sha256': sha(magic_dir / 'generation.json'),
           'liveportrait_summary_sha256': sha(live_dir / 'summary.json'), 'script_sha256': sha(__file__),
           'openface_seal_sha256': sha(e.TOOLS / 'tool_seal.json'), 'facenet_seal_sha256': sha(face_root / 'tool_seal.json'),
           'source_reference': 'exact canonical source from train2k cache and main dev19 evaluation for both R(source) and identity',
           'output_preprocess': 'unchanged native saved outputs; no beautification, resampling or chosen crop before measurement',
           'openface': 'fixed CLNF/static, singleton only, AU_r 0..5; full raw rows kept including AU45_r',
           'facenet': 'fixed author MTCNN160, margin0, thresholds .6/.7/.7, fixed standardization, no padding; singleton only',
           'comparison_scope': 'one-case engineering diagnostic; image-driven LivePortrait gets additional real driver, not same-input SOTA rank',
           'test_used': False, 'generation_run': False, 'device': 'cpu'})
    save(out / 'results.json', {'records': records, 'scope': 'one development case only', 'test_used': False, 'sota_conclusion': False})
    (out / 'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'output': str(out), 'records': [{k: row.get(k) for k in ('method', 'target_mae_changed', 'delta_mae_changed', 'unchanged_source_relative_drift', 'direction_correct_fraction')} |
                     {'AU45_r': row['all_measured_au_r'].get('AU45_r') if row['all_measured_au_r'] else None, 'facenet': row['facenet']} for row in records]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
