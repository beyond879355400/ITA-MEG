#!/usr/bin/env python3
"""Read-only check of six-image geometric selection and unchanged singleton report."""
import csv
import json
from pathlib import Path

from prepare_dev_contracts import HERE, sha
from evaluate_smoke_primary_face import AUS17, EXTRA, landmark_geometry


def main():
    folder = HERE / 'evaluation/primary_face_sensitivity_202609142028'
    protocol = json.loads((folder / 'protocol.json').read_text())
    result = json.loads((folder / 'results.json').read_text())
    prior = HERE / 'evaluation/smoke_pair_202609142019'
    assert sha(prior / 'results.json') == protocol['prior_singleton_results_sha256']
    assert sha(prior / 'REPORT.md') == protocol['prior_singleton_report_sha256']
    assert sha(folder / 'evaluator_snapshot.py') == protocol['script_sha256']
    assert protocol['extra_aus_complete'] == EXTRA and protocol['full_au_order'] == AUS17
    rows = {r['method']: r for r in result['records']}
    order = ['source_real', 'target_real', 'baseline', 'current_model', 'magicface', 'liveportrait']
    assert set(rows) == set(order)
    for name in order:
        row = rows[name]
        assert sha(row['path']) == row['sha256']
        f = row['facenet']
        chosen = max(f['all_candidates'], key=lambda x: (x['area'], -x['index']))
        assert f['selected_index'] == chosen['index']
        assert f['identity_cosine'] is not None
        o = row['openface']
        expected = max(o['all_candidates'], key=lambda x: (x['area'], -x['row_index']))
        assert o['selected_row'] == expected['row_index']
        without = json.loads((folder / 'openface' / name / 'without_2Dfp/parsed.json').read_text())
        with_landmarks = json.loads((folder / 'openface' / name / 'with_2Dfp/parsed.json').read_text())
        assert len(without['raw_rows']) == len(with_landmarks['raw_rows'])
        for old, new in zip(without['raw_rows'], with_landmarks['raw_rows']):
            assert all(old[f'AU{a:02d}_r'] == new[f'AU{a:02d}_r'] for a in AUS17)
        assert landmark_geometry(with_landmarks['raw_rows'][expected['row_index']])['area'] == expected['area']
        assert set(row['uncontrolled5']) == {str(a) for a in EXTRA}
    report = (folder / 'REPORT.md').read_text()
    for au in AUS17:
        line = '|' + str(au) + '|' + '|'.join(f"{rows[name]['openface']['selected_au17'][str(au)]:.2f}" for name in order) + '|'
        assert line in report
    assert not protocol['test_used'] and not result['test_used']
    print(json.dumps({'status': 'PASS', 'images': 6, 'au_flag_pairs_verified': 102,
                      'complete_17au_report_rows': 17, 'geometric_selections_verified': 12,
                      'old_singleton_result_and_report_unchanged': True, 'test_used': False}))


if __name__ == '__main__':
    main()
