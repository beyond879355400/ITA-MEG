#!/usr/bin/env python3
"""Read-only extension audit: no models, no metric recomputation."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY_SHA = 'd29cd8e67df6f552fb343d9ee0db42d294c2027c0e2da822ee5514920f8f289c'
GROUPS = {
    'single_directions': (57, 'cd7365706c5b04540f6d97ff90f4f8d3ade5513b08f57179479395e702f18eb5'),
    'external_driver': (57, 'bbeaf8d447b710060aaaf0d4a78391cefbb55606f2921485f598b7aa6bad02a4'),
    'external_magicface': (38, 'cd1e07bf6c8f9395440a5d3b4858a0d22748127fe8d0af454c30fc221555b531'),
    'external_liveportrait_absolute': (38, '5c6fecc82650b1ce27d69ba9dc3ef4eca522384223dde70c41d9505231ba0eab'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(HERE / 'policy.json') == POLICY_SHA
    original = json.loads((HERE / 'facenet_dev19/results.json').read_text())
    original_by_key = {(r['case_id'], r['sha256']): r for r in original['records']}
    reports = {}
    for group, (count, input_sha) in GROUPS.items():
        root = HERE / group
        assert sha(root / 'items.json') == input_sha
        inputs = json.loads((root / 'items.json').read_text())['items']
        result = json.loads((root / 'facenet/results.json').read_text())
        protocol = json.loads((root / 'facenet/protocol.json').read_text())
        summary = json.loads((root / 'facenet/summary.json').read_text())
        assert result['policy_sha256'] == protocol['policy_sha256'] == summary['policy_sha256'] == POLICY_SHA
        assert protocol['input_sha256'] == input_sha
        assert summary['status'] == 'complete' and not summary['test_used']
        assert not result['embeddings_saved']
        assert len(inputs) == len(result['records']) == count
        assert len({(r['case_id'], r['method']) for r in result['records']}) == count
        shared = 0
        for source, row in zip(inputs, result['records']):
            for key in ['case_id', 'method', 'path', 'sha256']:
                assert row[key] == source[key]
            assert sha(row['path']) == row['sha256'] and row['policy_sha256'] == POLICY_SHA
            ranked = [x for x in row['candidates'] if x['geometry_rankable']]
            chosen = max(ranked, key=lambda x: (x['area'], -x['index'])) if ranked else None
            assert row['selected_index'] == (chosen['index'] if chosen else None)
            ref = original_by_key.get((row['case_id'], row['sha256']))
            if ref:
                assert abs(row['identity_cosine'] - ref['identity_cosine']) < 1e-12
                shared += 1
        unique_count = len({r['sha256'] for r in result['records']})
        assert len(result['unique_measurements']) == unique_count
        reports[group] = {'logical_records': count, 'unique_images': unique_count,
                          'shared_readbacks_match_1e-12': shared,
                          'valid_identity': sum(r['identity_cosine'] is not None for r in result['records'])}
    print(json.dumps({'status': 'PASS', 'policy_sha256': POLICY_SHA, 'groups': reports, 'test_used': False}))


if __name__ == '__main__':
    main()
