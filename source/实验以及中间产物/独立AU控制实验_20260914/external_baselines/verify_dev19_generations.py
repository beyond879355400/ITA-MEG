#!/usr/bin/env python3
"""Integrity-only checks for the three new native external dev19 batches."""
import argparse
import json
from pathlib import Path

from prepare_dev_contracts import HERE, sha


def verify_driver(method, folder, contract):
    summary = json.loads((folder / 'summary.json').read_text())
    configuration = json.loads((folder / 'configuration.json').read_text())
    expected = {r['case_id']: r for r in contract['cases']}
    assert summary['case_count'] == len(expected) == 19
    assert len(summary['results']) == 19
    assert {r['case_id'] if 'case_id' in r else r['input']['case_id'] for r in summary['results']} == set(expected)
    for row in summary['results']:
        case = row['input']
        original = expected[case['case_id']]
        for key in ('source', 'target', 'source_sha256', 'target_sha256', 'seed', 'request_case_id'):
            assert case[key] == original[key]
        assert sha(case['source']) == case['source_sha256'] and sha(case['target']) == case['target_sha256']
        if row['status'] == 'success':
            assert sha(row['output']) == row['output_sha256']
        assert case['seed'] == 101
    if method == 'liveportrait':
        motion = json.loads((folder / 'motion_summary.json').read_text())
        assert motion['all_fresh_execution'] and not motion['prior_single_case_reused'] and len(motion['records']) == 19
        assert len([r for r in motion['records'] if r['status'] == 'success']) == summary['success_count']
        for row in motion['records']:
            if row['status'] == 'success':
                assert len(row['warp_decode_observations']) == 1
                assert sha(row['output']) == row['output_sha256']
    else:
        assert configuration['policy']['steps'] == 25 and configuration['policy']['scheduler'] == 'EulerDiscreteScheduler'
        assert configuration['policy']['branch'] == 'reenactment'
    return {'case_count': 19, 'success_count': summary['success_count'], 'failed_count': summary['failed_count']}


def verify_magic(folder, contract):
    configuration = json.loads((folder / 'configuration.json').read_text())
    assert configuration['cases'] == contract['cases'] and not configuration['old_single_case_outputs_reused']
    assert configuration['case_count'] == 19 and configuration['steps'] == 50
    summary = json.loads((folder / 'generate_summary.json').read_text())
    assert summary['processed_count'] == 19 and len(summary['records']) == 19
    pids = []
    for case, record in zip(contract['cases'], summary['records']):
        assert record['case_id'] == case['case_id']
        if record['status'] == 'complete':
            output = folder / 'cases' / case['case_id']
            generation = json.loads((output / 'generation.json').read_text())
            assert generation['case_id'] == case['case_id'] and generation['source_raw_sha256'] == case['source_sha256']
            assert generation['au_variation_sent'] == case['au_delta_native'] and generation['seed'] == 101
            assert generation['inference_steps'] == 50 and generation['scheduler'] == 'UniPCMultistepScheduler'
            assert generation['guidance_scale'] == 7.5 and generation['image_guidance_scale'] == 1.5
            assert not generation['target_image_read'] and not generation['test_used']
            assert sha(output / 'generated.png') == generation['output_sha256']
            assert sha(output / 'source_author_crop.png') == generation['source_crop_sha256']
            assert sha(output / 'source_background_contour.png') == generation['background_sha256']
            if 'process_id' in record:
                pids.append(record['process_id'])
    assert len(set(pids)) == len(pids)
    return {'case_count': 19, 'success_count': summary['success_count'], 'failed_count': summary['failed_count'],
            'unique_success_process_ids': len(set(pids))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--include-magicface', action='store_true')
    args = p.parse_args()
    root = HERE / 'contracts/raw_dev19_v3_202609142040'
    native = json.loads((root / 'magicface_dev19.json').read_text())
    driver = json.loads((root / 'image_driver_dev19.json').read_text())
    old = HERE / 'contracts/raw_dev12_v2_202609141929'
    assert native['cases'][:12] == json.loads((old / 'magicface_dev12.json').read_text())['cases']
    assert driver['cases'][:12] == json.loads((old / 'image_driver_dev12.json').read_text())['cases']
    result = {'test_used': False, 'old_12_exactly_preserved': True,
              'faceadapter': verify_driver('faceadapter', HERE / 'runtime/runs/faceadapter_dev19_raw_native_202609142058', driver),
              'liveportrait': verify_driver('liveportrait', HERE / 'runtime/runs/liveportrait_dev19_raw_native_202609142041', driver)}
    if args.include_magicface:
        result['magicface'] = verify_magic(HERE / 'runs/magicface_dev19_raw_native_202609142101', native)
    print(json.dumps({'status': 'PASS', **result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
