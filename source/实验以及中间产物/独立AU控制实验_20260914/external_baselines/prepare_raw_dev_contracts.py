#!/usr/bin/env python3
"""Version2: the same dev requests, original raw frames and author-native crops."""
from datetime import datetime
import json
from pathlib import Path

from prepare_dev_contracts import HERE, EXPERIMENT, AUS, sha, resident, save


def main():
    requests_path = EXPERIMENT / 'development_previews/dev12_manifest.json'
    cache_path = EXPERIMENT / 'training_cache/expanded_v2_202609141711/manifest.json'
    requests, cache = json.loads(requests_path.read_text()), json.loads(cache_path.read_text())
    assert not requests['test_data_used'] and not cache['test_data_used'] and requests['cache_sha256'] == cache['cache_sha256']
    assert len(requests['cases']) == 12 and requests['sampling_seeds'] == [101]
    pairs = {r['case_id']: r for r in cache['pairs']}
    from PIL import Image
    images, native, driver = {}, [], []
    for case in requests['cases']:
        pair = pairs[case['case_id']]
        assert case['split'] == pair['split'] == 'dev'
        source_au, target_au = case['source_au_vector_0_5'], case['target_au_vector_0_5']
        assert len(source_au) == len(target_au) == 12 and pair['target_au_vector_0_5'] == target_au
        expected_source = pair.get('source_au_vector_0_5', [pair.get('source_aus', {}).get(f'au{a}') for a in AUS])
        assert source_au == expected_source
        raw = {}
        for role in ('source', 'target'):
            aligned = pair['aligned_' + role]
            path, digest = Path(aligned['raw_path']).resolve(), aligned['raw_sha256']
            assert EXPERIMENT in path.parents and resident(path)['status'] == 'resident' and sha(path) == digest
            if digest not in images:
                with Image.open(path) as image:
                    image.load()
                    assert image.mode == 'RGB' and image.size == (1024, 768)
                images[digest] = {'path': str(path), 'sha256': digest, 'width': 1024, 'height': 768, 'mode': 'RGB'}
            raw[role] = images[digest]
        for seed in requests['sampling_seeds']:
            common = {'case_id': case['case_id'] + f'_seed{seed}', 'request_case_id': case['case_id'], 'subject': case['subject'],
                      'split': 'dev', 'group': case['group'], 'seed': seed, 'au_order': AUS,
                      'source': raw['source']['path'], 'source_sha256': raw['source']['sha256'],
                      'source_width': 1024, 'source_height': 768, 'source_space': 'same_original_raw_frame_as_our_encoder',
                      'source_au_vector_0_5': source_au, 'target_au_vector_0_5': target_au,
                      'preprocessing_policy': 'author_native_crop; do not feed tightly aligned canonical512 directly'}
            native.append({**common, 'au_delta_native': [t - s for s, t in zip(source_au, target_au)],
                           'input_group': 'source_plus_manual_au', 'background_policy': 'derive_from_author_crop_of_this_raw_source_only',
                           'prompt': 'A close up of a person.', 'native_inference_steps': 50})
            driver.append({**common, 'target': raw['target']['path'], 'target_sha256': raw['target']['sha256'],
                           'target_width': 1024, 'target_height': 768, 'input_group': 'source_plus_real_driver',
                           'target_role': 'same_original_raw_observed_target_frame; never_generated'})
    assert len(images) == len(native) == len(driver) == 12
    stamp = HERE / ('contracts/raw_dev12_v2_' + datetime.now().strftime('%Y%m%d%H%M'))
    output = stamp
    i = 2
    while output.exists():
        output = stamp.with_name(stamp.name + f'-{i:02d}')
        i += 1
    output.mkdir(parents=True)
    provenance = {'schema_version': 2, 'scope': 'development_only', 'test_used': False,
                  'request_manifest_sha256': sha(requests_path), 'pair_cache_manifest_sha256': sha(cache_path),
                  'sampling_seeds': [101], 'source_space': 'original_raw_1024x768',
                  'author_native_preprocessing_allowed': True, 'old_canonical_contract_preserved': True,
                  'reason': 'same original frame information; author detector context, not generated-outcome selection',
                  'old_generated_outputs_or_metrics_used': False, 'builder_sha256': sha(__file__)}
    save(output / 'magicface_dev12.json', {**provenance, 'input_group': 'source_plus_manual_au', 'cases': native})
    save(output / 'image_driver_dev12.json', {**provenance, 'input_group': 'source_plus_real_driver', 'cases': driver})
    save(output / 'input_checks.json', {'unique_raw_images': list(images.values()), 'test_used': False, 'generation_run': False})
    (output / 'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'output': str(output), 'requests': 12, 'raw_images': 12, 'generation_run': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
