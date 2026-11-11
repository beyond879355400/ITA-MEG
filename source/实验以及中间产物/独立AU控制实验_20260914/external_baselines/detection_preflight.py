#!/usr/bin/env python3
"""CPU detector-only input readiness; no crop, background, generation or scoring."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

from prepare_dev_contracts import HERE, OLD, resident, sha, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    document = json.loads(args.manifest.read_text())
    assert not document['test_used']
    images = {}
    for case in document['cases']:
        assert case['split'] == 'dev'
        for role in ('source', 'target'):
            path, digest = case[role], case[role + '_sha256']
            assert resident(path)['status'] == 'resident' and sha(path) == digest
            images.setdefault(digest, {'path': path, 'sha256': digest, 'uses': []})['uses'].append({'case_id': case['case_id'], 'role': role})
    assert len(images) == 12
    args.output.mkdir(parents=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', NO_ALBUMENTATIONS_UPDATE='1',
                      MPLCONFIGDIR=str(HERE / 'cache/matplotlib'), XDG_CACHE_HOME=str(HERE / 'cache/xdg'),
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE / 'runtime'))
    from liveportrait_detection_policy import DetectorPolicy
    import cv2
    import onnxruntime as ort
    from insightface.model_zoo import get_model
    models = {'magicface_and_faceadapter': (OLD / 'models/face_adapter/third_party/models/antelopev2/scrfd_10g_bnkps.onnx', .5),
              'liveportrait': (OLD / 'models/liveportrait_author/insightface/models/buffalo_l/det_10g.onnx', .15)}
    result = {'scope': '12 unique fixed dev source/target inputs; detector-only, before author crop',
              'manifest_sha256': sha(args.manifest), 'script_sha256': sha(__file__), 'records': [],
              'test_used': False, 'generation_run': False, 'image_preprocessing': 'original canonical512, no padding',
              'policy_note': 'default threshold versus already-defined shared fallback, not a new threshold search'}
    for name, (path, threshold) in models.items():
        assert resident(path)['status'] == 'resident'
        session = ort.SessionOptions()
        session.intra_op_num_threads = session.inter_op_num_threads = 1
        with (args.output / (name + '.log')).open('x') as log, contextlib.redirect_stdout(log):
            detector = get_model(str(path), providers=['CPUExecutionProvider'], sess_options=session)
            detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=threshold)
        for digest, item in images.items():
            bgr = cv2.imread(item['path'], cv2.IMREAD_COLOR)
            assert bgr.shape == (512, 512, 3)
            for policy in ('author_default', 'shared_fallback'):
                wrapped = DetectorPolicy(detector, policy=policy)
                wrapped.begin_case(digest)
                boxes, _ = wrapped.detect(bgr, max_num=0, metric='default')
                result['records'].append({'model': name, 'model_path': str(path), 'model_sha256': sha(path),
                                          'image': item, 'policy': policy, 'face_count': len(boxes), 'events': wrapped.events})
            assert sha(item['path']) == digest
    result['summary'] = {name: {policy: {'planned_unique_images': 12,
                             'detected': sum(r['model'] == name and r['policy'] == policy and r['face_count'] > 0 for r in result['records']),
                             'singleton': sum(r['model'] == name and r['policy'] == policy and r['face_count'] == 1 for r in result['records'])}
                              for policy in ('author_default', 'shared_fallback')} for name in models}
    save(args.output / 'results.json', result)
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
