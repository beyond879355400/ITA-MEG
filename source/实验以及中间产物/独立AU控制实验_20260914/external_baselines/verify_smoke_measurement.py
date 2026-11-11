#!/usr/bin/env python3
"""Read-only check of actual smoke outputs, source binding and all AU calculations."""
import json
from pathlib import Path
import sys

from prepare_dev_contracts import HERE, EXPERIMENT, sha


def main():
    folder = HERE / 'evaluation/smoke_pair_202609142019'
    result = json.loads((folder / 'results.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    rows = {r['method']: r for r in result['records']}
    assert len(rows) == 5 and rows['source_real']['sha256'] == '5dcbecd54f40a282a34b8eea0003553ba1306485517dc49236c75e63676b5ef3'
    assert sha(folder / 'evaluator_snapshot.py') == protocol['script_sha256']
    sys.path.insert(0, str(EXPERIMENT / 'development_evaluation'))
    from evaluate_patch_variant import e
    from verify_training_steps import same
    source = rows['source_real']
    report = (folder / 'REPORT.md').read_text()
    for row in rows.values():
        assert sha(row['path']) == row['sha256']
        assert e.helper.parse_openface(Path(row['csv_files'][0]))['vector'] == row['vector']
        recalculated = e.au_metrics(row['vector'], source['vector'], row['source_labels_12'], row['target_labels_12'])
        assert all(same(row[k], v) for k, v in recalculated.items())
    for i, au in enumerate((1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)):
        values = [rows[name]['vector'][i] for name in ('source_real', 'baseline', 'current_model', 'magicface', 'liveportrait')]
        line = f"|{au}|{int(source['target_labels_12'][i])}|" + '|'.join(f'{v:.2f}' for v in values) + '|'
        assert line in report
    assert rows['magicface']['all_measured_au_r']['AU45_r'] == 4.01
    assert rows['magicface']['facenet']['face_count'] == rows['liveportrait']['facenet']['face_count'] == 2
    assert rows['magicface']['facenet']['identity_cosine'] is None and rows['liveportrait']['facenet']['identity_cosine'] is None
    magic = HERE / 'runs/magicface_smoke_native_202609142002'
    generation = json.loads((magic / 'generation.json').read_text())
    assert generation['inference_steps'] == 50 and generation['seed'] == 101 and not generation['target_image_read']
    assert sha(magic / 'runner_successful.py') == generation['runner_sha256']
    live = json.loads((HERE / 'runtime/runs/liveportrait_smoke_raw_native_202609142014/motion_probe.json').read_text())
    assert live['returncode'] == 0 and not live['pixel_exact_source_copy']
    assert len(live['warp_decode_observations']) == 1 and live['warp_decode_observations'][0]['changed_scalar_count'] == 63
    print(json.dumps({'status': 'PASS', 'images': 5, 'all_12au_rows_verified': 12, 'same_canonical_reference': True,
                      'magicface_real_generation': True, 'liveportrait_nonzero_motion': True, 'test_used': False}))


if __name__ == '__main__':
    main()
