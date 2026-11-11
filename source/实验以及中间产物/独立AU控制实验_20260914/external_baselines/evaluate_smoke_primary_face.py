#!/usr/bin/env python3
"""Six-image geometry-only primary-face sensitivity check, preserving every candidate."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from prepare_dev_contracts import HERE, EXPERIMENT, sha, save

AUS12 = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
AUS17 = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
EXTRA = [7, 10, 14, 23, 45]


def landmark_geometry(row):
    try:
        x = [float(row[f'x_{i}']) for i in range(68)]
        y = [float(row[f'y_{i}']) for i in range(68)]
        if not all(math.isfinite(v) for v in x + y):
            raise ValueError('Nonfinite landmarks')
        box = [min(x), min(y), max(x), max(y)]
        area = (box[2] - box[0]) * (box[3] - box[1])
        return {'bbox': box, 'area': area if area > 0 else None}
    except (KeyError, ValueError):
        return {'bbox': None, 'area': None}


def main():
    prior_dir = HERE / 'evaluation/smoke_pair_202609142019'
    prior = json.loads((prior_dir / 'protocol.json').read_text())
    prior_results = json.loads((prior_dir / 'results.json').read_text())
    items = prior['items'][:]
    main_eval = EXPERIMENT / 'development_evaluation/v4_slot_2k_step200_dev19/results.json'
    main_rows = json.loads(main_eval.read_text())['records']
    target = next(r for r in main_rows if r['case_id'] == prior['case_id'] and r['method'] == 'target_real')
    items.append({k: target[k] for k in ('case_id', 'subject', 'method', 'path', 'sha256', 'split', 'group', 'source_labels_12', 'target_labels_12')})
    order = ['source_real', 'target_real', 'baseline', 'current_model', 'magicface', 'liveportrait']
    items.sort(key=lambda r: order.index(r['method']))
    assert len(items) == 6 and all(r['split'] == 'dev' for r in items)
    for item in items:
        assert sha(item['path']) == item['sha256']
    base = HERE / ('evaluation/primary_face_sensitivity_' + datetime.now().strftime('%Y%m%d%H%M'))
    out = base
    n = 2
    while out.exists():
        out = base.with_name(base.name + f'-{n:02d}')
        n += 1
    out.mkdir(parents=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1', MPLCONFIGDIR=str(HERE / 'cache/matplotlib'),
                      XDG_CACHE_HOME=str(HERE / 'cache/xdg'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(EXPERIMENT / 'development_evaluation'))
    from evaluate_patch_variant import e
    tool_seal = json.loads((e.TOOLS / 'tool_seal.json').read_text())
    for row in [tool_seal['binary'], tool_seal['launcher'], *tool_seal['model_files']]:
        assert sha(e.TOOLS / row['path']) == row['sha256']
    def openface(item, flag):
        folder = out / 'openface' / item['method'] / ('with_2Dfp' if flag else 'without_2Dfp')
        folder.mkdir(parents=True)
        command = [str(e.TOOLS / 'FaceLandmarkImg'), '-f', item['path'], '-aus', '-mloc', str(e.TOOLS / 'build/bin/model/main_clnf_general.txt')]
        if flag:
            command += ['-2Dfp']
        command += ['-out_dir', str(folder)]
        with (folder / 'run.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        files = list(folder.glob('*.csv'))
        rows = []
        if result.returncode == 0 and len(files) == 1:
            with files[0].open() as stream:
                rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
        payload = {'method': item['method'], 'flag_2Dfp': flag, 'returncode': result.returncode, 'command': command,
                   'csv_files': [str(p) for p in files], 'raw_rows': rows}
        save(folder / 'parsed.json', payload)
        return payload
    measured = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(openface, item, flag) for item in items for flag in (False, True)]
        for future in as_completed(futures):
            row = future.result()
            measured[(row['method'], row['flag_2Dfp'])] = row
    records = []
    for item in items:
        plain, extra = measured[(item['method'], False)], measured[(item['method'], True)]
        assert len(plain['raw_rows']) == len(extra['raw_rows']) > 0
        # Readback verifies all 17 AU values, not just the controlled subset.
        unchanged = all({f'AU{a:02d}_r': p[f'AU{a:02d}_r'] for a in AUS17} ==
                        {f'AU{a:02d}_r': q[f'AU{a:02d}_r'] for a in AUS17}
                        for p, q in zip(plain['raw_rows'], extra['raw_rows']))
        assert unchanged, '-2Dfp changed AU readback'
        candidates = [{**landmark_geometry(row), 'row_index': i, 'face_id': row.get('face'), 'confidence': row.get('confidence')}
                      for i, row in enumerate(extra['raw_rows'])]
        eligible = [c for c in candidates if c['area'] is not None]
        selected = max(eligible, key=lambda c: (c['area'], -c['row_index'])) if eligible else None
        au17 = None
        if selected is not None:
            values = [float(extra['raw_rows'][selected['row_index']][f'AU{a:02d}_r']) for a in AUS17]
            if all(math.isfinite(v) and 0 <= v <= 5 for v in values):
                au17 = dict(zip(AUS17, values))
        areas = sorted((c['area'] for c in eligible), reverse=True)
        records.append({**item, 'openface': {'all_candidates': candidates, 'selected_row': selected['row_index'] if selected else None,
                                            'selection': 'maximum 68-landmark bounding-box area, tie original row order',
                                            'second_to_first_area_ratio': areas[1] / areas[0] if len(areas) > 1 else None,
                                            'all_17au_unchanged_by_2Dfp': unchanged, 'selected_au17': au17}})
    source = next(r for r in records if r['method'] == 'source_real')['openface']['selected_au17']
    target_au = next(r for r in records if r['method'] == 'target_real')['openface']['selected_au17']
    assert source is not None and target_au is not None
    for row in records:
        au = row['openface']['selected_au17']
        if au is not None:
            row['metrics12'] = e.au_metrics([au[a] for a in AUS12], [source[a] for a in AUS12], row['source_labels_12'], row['target_labels_12'])
            row['uncontrolled5'] = {str(a): {'output': au[a], 'source': source[a], 'real_target': target_au[a],
                                            'output_minus_target': au[a] - target_au[a], 'output_minus_source': au[a] - source[a],
                                            'target_minus_source': target_au[a] - source[a]} for a in EXTRA}
            row['uncontrolled5_proxy_mae_to_real_target'] = sum(abs(au[a] - target_au[a]) for a in EXTRA) / len(EXTRA)
        else:
            row['metrics12'], row['uncontrolled5'] = None, None
    face_root = EXPERIMENT / 'tools/facenet'
    face_seal = json.loads((face_root / 'tool_seal.json').read_text())
    for entry in face_seal['source_files'] + face_seal['weights']:
        assert sha(face_root / entry['path']) == entry['sha256']
    sys.path.insert(0, str(face_root / 'runtime'))
    import numpy as np
    import torch
    from PIL import Image
    from facenet_pytorch import MTCNN, InceptionResnetV1
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    with (out / 'facenet_runtime.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        detector = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[.6, .7, .7], factor=.709,
                         keep_all=True, post_process=True, device='cpu').eval()
        model = InceptionResnetV1(pretrained=None, classify=True, num_classes=8631).eval()
        model.load_state_dict(torch.load(face_root / 'models/20180402-114759-vggface2.pt', map_location='cpu', weights_only=True), strict=True)
        model.classify = False
    embeddings = {}
    for row in records:
        image = Image.open(row['path']).convert('RGB')
        boxes, probabilities, points = detector.detect(image, landmarks=True)
        candidates = [] if boxes is None else [{'index': i, 'bbox': box.tolist(), 'area': float((box[2] - box[0]) * (box[3] - box[1])),
                                                'probability': float(probabilities[i])} for i, box in enumerate(boxes)]
        finite = [c for c in candidates if math.isfinite(c['area']) and c['area'] > 0]
        chosen = max(finite, key=lambda c: (c['area'], -c['index'])) if finite else None
        record = {'all_candidates': candidates, 'selected_index': chosen['index'] if chosen else None,
                  'selection': 'maximum detected bounding-box area, author default largest-face policy; no identity score used',
                  'identity_cosine': None, 'status': 'no_valid_box' if chosen is None else 'selected'}
        areas = sorted([c['area'] for c in finite], reverse=True)
        record['second_to_first_area_ratio'] = areas[1] / areas[0] if len(areas) > 1 else None
        if chosen is not None:
            tensor = detector.extract(image, boxes[chosen['index']:chosen['index'] + 1], save_path=None)
            feature = model(tensor)[0]
            if feature.shape == (512,) and torch.isfinite(feature).all():
                embeddings[row['method']] = feature.double().numpy()
            else:
                record['status'] = 'selected_feature_invalid'
        row['facenet'] = record
        assert sha(row['path']) == row['sha256']
    for row in records:
        if row['method'] in embeddings and 'source_real' in embeddings:
            a, b = embeddings[row['method']], embeddings['source_real']
            row['facenet']['identity_cosine'] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    protocol = {'items': items, 'source_reference': 'same canonical source from 2k main evaluation',
                'prior_singleton_results_sha256': sha(prior_dir / 'results.json'), 'prior_singleton_report_sha256': sha(prior_dir / 'REPORT.md'),
                'main_evaluation_sha256': sha(main_eval), 'script_sha256': sha(__file__), 'test_used': False,
                'selection_policy': 'FaceNet maximum bbox area; OpenFace maximum 68-landmark bbox area; candidates retained; no score-dependent choice',
                'extra_aus_complete': EXTRA, 'full_au_order': AUS17, 'unchanged_output_images': True,
                'openface_flag_validation': 'all six files run both -aus and -aus -2Dfp; every row all17AU identical',
                'analysis_scope': 'single-case sensitivity, not a new primary policy or SOTA ranking'}
    save(out / 'protocol.json', protocol)
    save(out / 'results.json', {'records': records, 'scope': 'six-image single-case sensitivity only', 'test_used': False, 'sota_conclusion': False})
    (out / 'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'output': str(out), 'records': [{'method': row['method'], 'openface_faces': len(row['openface']['all_candidates']),
                   'facenet_faces': len(row['facenet']['all_candidates']), 'facenet_identity': row['facenet']['identity_cosine'],
                   'uncontrolled5': row['uncontrolled5']} for row in records]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
