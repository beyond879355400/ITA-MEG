#!/usr/bin/env python3
"""Append exactly seven frozen dev requests while preserving the original 12 rows."""
import copy
from datetime import datetime
import json
from pathlib import Path

from prepare_dev_contracts import HERE, EXPERIMENT, AUS, resident, sha, save


def main():
    old_root = HERE / 'contracts/raw_dev12_v2_202609141929'
    old_native = json.loads((old_root / 'magicface_dev12.json').read_text())
    old_driver = json.loads((old_root / 'image_driver_dev12.json').read_text())
    request_path = EXPERIMENT / 'development_previews/v4_slot_2k_step200_dev19_202609141941/frozen_cases.json'
    cache_path = EXPERIMENT / 'training_cache/train2k_batched_202609141853/manifest.json'
    requests, cache = json.loads(request_path.read_text()), json.loads(cache_path.read_text())
    assert not requests['test_data_used'] and not cache['test_data_used'] and not requests['model_outputs_used_for_selection']
    assert requests['sampling_seeds'] == [101] and requests['total_case_count'] == len(requests['cases']) == 19
    assert requests['cache_sha256'] == cache['cache_sha256']
    pairs = {row['case_id']: row for row in cache['pairs']}
    original_ids = [row['request_case_id'] for row in old_native['cases']]
    assert set(original_ids) == set(requests['original_case_ids']) and len(original_ids) == 12
    from PIL import Image
    checked = {}
    records = {}
    for case in requests['cases']:
        pair = pairs[case['case_id']]
        assert pair['split'] == case['split'] == 'dev'
        assert pair['source_au_vector_0_5'] == case['source_au_vector_0_5'] and pair['target_au_vector_0_5'] == case['target_au_vector_0_5']
        for role in ('source', 'target'):
            raw = pair['aligned_' + role]
            path, digest = Path(raw['raw_path']), raw['raw_sha256']
            assert resident(path)['status'] == 'resident' and sha(path) == digest
            if digest not in checked:
                with Image.open(path) as image:
                    image.load()
                    assert image.size == (1024, 768) and image.mode == 'RGB'
                checked[digest] = {'path': str(path), 'sha256': digest, 'size': [1024, 768], 'mode': 'RGB'}
        records[case['case_id']] = (case, pair)
    native, driver = copy.deepcopy(old_native['cases']), copy.deepcopy(old_driver['cases'])
    for old_n, old_d in zip(native, driver):
        case, pair = records[old_n['request_case_id']]
        assert old_n['source_au_vector_0_5'] == case['source_au_vector_0_5'] and old_n['target_au_vector_0_5'] == case['target_au_vector_0_5']
        assert old_n['source_sha256'] == old_d['source_sha256'] == pair['aligned_source']['raw_sha256']
        assert old_d['target_sha256'] == pair['aligned_target']['raw_sha256']
    for case in requests['cases']:
        if case['case_id'] in original_ids:
            continue
        pair = pairs[case['case_id']]
        row_n = copy.deepcopy(native[0])
        row_n.update(case_id=case['case_id'] + '_seed101', request_case_id=case['case_id'], subject=case['subject'], group=case['group'],
                     source=pair['aligned_source']['raw_path'], source_sha256=pair['aligned_source']['raw_sha256'],
                     source_au_vector_0_5=case['source_au_vector_0_5'], target_au_vector_0_5=case['target_au_vector_0_5'],
                     au_delta_native=[t - s for s, t in zip(case['source_au_vector_0_5'], case['target_au_vector_0_5'])])
        row_d = copy.deepcopy(driver[0])
        for key in ('case_id', 'request_case_id', 'subject', 'group', 'source', 'source_sha256', 'source_au_vector_0_5', 'target_au_vector_0_5'):
            row_d[key] = row_n[key]
        row_d.update(target=pair['aligned_target']['raw_path'], target_sha256=pair['aligned_target']['raw_sha256'])
        native.append(row_n)
        driver.append(row_d)
    assert len(native) == len(driver) == 19 and native[:12] == old_native['cases'] and driver[:12] == old_driver['cases']
    assert {r['request_case_id'] for r in native[12:]} == set(requests['additional_case_ids'])
    assert all('target' not in r and 'target_image_path' not in r for r in native)
    base = HERE / ('contracts/raw_dev19_v3_' + datetime.now().strftime('%Y%m%d%H%M'))
    out = base
    i = 2
    while out.exists():
        out = base.with_name(base.name + f'-{i:02d}')
        i += 1
    out.mkdir(parents=True)
    common = {'schema_version': 3, 'scope': 'development_only', 'test_used': False, 'sampling_seeds': [101],
              'source_space': 'original_raw_1024x768', 'author_native_preprocessing_allowed': True,
              'request_manifest_sha256': sha(request_path), 'cache_manifest_sha256': sha(cache_path),
              'original_case_count': 12, 'appended_case_count': 7, 'total_case_count': 19,
              'old_native_contract_sha256': sha(old_root / 'magicface_dev12.json'),
              'old_driver_contract_sha256': sha(old_root / 'image_driver_dev12.json'),
              'old_12_rows_exactly_preserved': True, 'generated_outputs_used_for_selection': False, 'builder_sha256': sha(__file__)}
    save(out / 'magicface_dev19.json', {**common, 'input_group': 'source_plus_manual_au', 'cases': native})
    save(out / 'image_driver_dev19.json', {**common, 'input_group': 'source_plus_real_driver', 'cases': driver})
    save(out / 'raw_input_checks.json', {'images': list(checked.values()), 'test_used': False})
    (out / 'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'output': str(out), 'cases': 19, 'old_12_unchanged': True, 'additional_cases': 7, 'raw_images': len(checked)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
