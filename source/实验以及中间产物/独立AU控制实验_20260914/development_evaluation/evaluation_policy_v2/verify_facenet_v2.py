#!/usr/bin/env python3
"""Read-only verification of policy binding, geometry choice and logical row retention."""
import json
from pathlib import Path

from run_facenet_v2 import HERE, POLICY_SHA, sha


def main():
    assert sha(HERE / 'policy.json') == POLICY_SHA
    policy = json.loads((HERE / 'policy.json').read_text())
    for directory, input_name, count, key in [('facenet_real12', 'facenet_real12_items.json', 12, 'images'),
                                               ('facenet_dev19', 'dev19_items.json', 95, 'items')]:
        out = HERE / directory
        protocol = json.loads((out / 'protocol.json').read_text())
        result = json.loads((out / 'results.json').read_text())
        summary = json.loads((out / 'summary.json').read_text())
        assert protocol['policy_sha256'] == result['policy_sha256'] == summary['policy_sha256'] == POLICY_SHA
        assert sha(out / 'policy_snapshot.json') == POLICY_SHA and sha(out / 'evaluator_snapshot.py') == protocol['script_sha256']
        input_path = HERE / input_name
        assert sha(input_path) == policy['inputs'][input_name]['sha256'] == protocol['input_sha256']
        inputs = json.loads(input_path.read_text())[key]
        assert len(result['records']) == len(inputs) == count
        for source, row in zip(inputs, result['records']):
            assert row['path'] == source['path'] and row['sha256'] == source['sha256'] and sha(row['path']) == row['sha256']
            assert row['policy_sha256'] == POLICY_SHA
            ranked = [x for x in row['candidates'] if x['geometry_rankable']]
            chosen = max(ranked, key=lambda x: (x['area'], -x['index'])) if ranked else None
            assert row['selected_index'] == (chosen['index'] if chosen else None)
            if key == 'items':
                assert row['case_id'] == source['case_id'] and row['method'] == source['method']
        assert summary['status'] == 'complete' and not summary['test_used'] and not result['embeddings_saved']
    data = json.loads((HERE / 'facenet_dev19/results.json').read_text())
    assert len(data['unique_measurements']) == len({r['sha256'] for r in data['records']}) == 68
    assert all(r['identity_cosine'] is not None for r in data['records'])
    assert len({(r['case_id'], r['method']) for r in data['records']}) == 95
    assert json.loads((HERE / 'real_dev_openface/summary.json').read_text())['policy_sha256'] == POLICY_SHA
    print(json.dumps({'status': 'PASS', 'policy_sha256': POLICY_SHA, 'real_calibration': 12,
                      'dev19_logical_records': 95, 'dev19_unique_measurements': 68,
                      'geometry_only_selection': True, 'test_used': False}))


if __name__ == '__main__':
    main()
