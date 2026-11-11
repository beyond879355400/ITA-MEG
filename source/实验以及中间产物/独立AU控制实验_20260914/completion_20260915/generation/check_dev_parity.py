"""Main-owned MPS gate: two existing DEV cases vs selected V11 bytes. Default no models."""
import argparse
import json
from pathlib import Path
import sys

from common import *
import run_grid


def source_pixel_parity(new_path, old_reference):
    """Source preprocessing parity concerns decoded pixels, not PNG encoder bytes."""
    from PIL import Image
    import numpy as np
    old_path = Path(old_reference['path'])
    need(sha(old_path) == old_reference['sha256'], 'Original DEV source file changed')
    with Image.open(old_path) as old_image, Image.open(new_path) as new_image:
        need(old_image.mode == new_image.mode == 'RGB' and old_image.size == new_image.size == (512, 512),
             'DEV source decoded mode/size changed')
        old_pixels, new_pixels = np.asarray(old_image), np.asarray(new_image)
        need(old_pixels.shape == new_pixels.shape == (512, 512, 3) and old_pixels.dtype == new_pixels.dtype == np.uint8,
             'DEV source decoded array shape/dtype changed')
        need(np.array_equal(old_pixels, new_pixels), 'New source runtime changed DEV canonical source pixels')
        return {'old_png': ref(old_path), 'new_png': ref(new_path),
            'png_bytes_equal': sha(old_path) == sha(new_path), 'decoded_pixels_equal': True,
            'mode': 'RGB', 'shape': [512, 512, 3], 'dtype': 'uint8',
            'decoded_pixel_sha256': hashlib.sha256(new_pixels.tobytes()).hexdigest(),
            'gate': 'exact decoded mode/shape/dtype/pixels; PNG compression bytes not a source parity requirement'}


def plan():
    old = EXP / 'improvements/a_local_cleanup_v11_20260915/runs/local_202609152016'
    result = read(old / 'result.json')
    need(sha(old / 'result.json') == '37c239afa6d4dac1ca1e5517b60de420a58f72a833ff6193b99340791a3a7451', 'Selected V11 result changed')
    cases = read(old / 'cases.json')['cases']
    # Fixed lexicographically first prior DEV subject, one noedit and one edit.
    subject = sorted({c['subject'] for c in cases})[0]
    cases = [c for c in cases if c['subject'] == subject]
    need(len(cases) == 2 and {c['group'] for c in cases} == {'no_edit', 'lowest_available'}, 'Expected exact original two DEV cases')
    expected = {(r['case_id'], r['method']): r for r in result['records']
                if r['case_id'] in {c['case_id'] for c in cases} and r['method'] in ('baseline_local_lp_s050', 'bidirectional_local_lp_s050')}
    need(len(expected) == 4 and all(r['status'] == 'complete' for r in expected.values()), 'Missing four exact prior DEV outputs')
    for row in expected.values():
        need(sha(row['output']) == row['sha256'], 'Prior DEV output changed')
    return cases, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cases, expected = plan()
    if not args.execute:
        print(json.dumps({'status': 'prepared_no_models', 'cases': [c['case_id'] for c in cases], 'images': 4,
            'source_runtime_checks': 1, 'device': 'mps', 'new_validation_pixels_read': False}))
        return
    need(args.output is not None and not args.output.exists() and COMPLETION in args.output.resolve().parents, 'Explicit new completion dev-parity output required')
    out = args.output.resolve()
    out.mkdir(parents=True)
    torch = environment('mps')
    from prepare_training_cache import make_prompt
    from PIL import Image
    from geometry import build_masks
    anchor, binding, payload = load_anchor()
    feature_path = EXP / 'training_cache/train2k_batched_202609141853/features.pt'
    samples = torch.load(feature_path, map_location='cpu', weights_only=True)['samples']
    need(all(c['split'] in ('train', 'dev') for c in samples), 'Training cache contained new validation cases')
    samples = {c['case_id']: c for c in samples}
    old_case = cases[0]
    original_alignment = old_case['source_alignment']
    need(sha(original_alignment['raw_path']) == original_alignment['raw_sha256'], 'Prior DEV raw source changed')
    source_runtime = module('source_runtime').SourceRuntime('mps', payload)
    with Image.open(original_alignment['raw_path']) as image:
        canonical, source_features, alignment = source_runtime.source(image.convert('RGB'))
    old_source = old_case['source_real']
    canonical_path = out / 'canonical_source.png'
    canonical.save(canonical_path)
    source_parity = source_pixel_parity(canonical_path, old_source)
    save(out / 'source_pixel_parity.json', source_parity)
    source_record = {'source_id': 'existing_dev_source', 'subject': old_case['subject'], 'split': 'dev', 'status': 'complete',
        'canonical_source': {**ref(canonical_path), 'width': 512, 'height': 512, 'mode': 'RGB'}}
    conditions, feature_checks = {}, []
    mapped_cases = []
    for case in cases:
        sample = samples[case['case_id']]
        need(sample['split'] == 'dev', 'Only prior dev source allowed')
        prompt = make_prompt(case['target_aus_0_5'])
        tokens, pooled = source_runtime.text(prompt)
        checks = {k: bool(torch.equal(source_features[k], sample[k])) for k in source_features}
        checks.update(text_tokens=bool(torch.equal(tokens, sample['text_tokens'])), text_pooled=bool(torch.equal(pooled, sample['text_pooled'])))
        feature_checks.append({'case_id': case['case_id'], 'exact_tensor_checks': checks})
        need(all(checks.values()), 'Source/text feature parity not exact: ' + str(checks))
        conditions[case['case_id']] = {'text_tokens': tokens, 'text_pooled': pooled, 'source_aus': sample['source_aus'],
            'target_aus': sample['target_aus'], 'prompt': prompt}
        mapped_cases.append({'case_id': case['case_id'], 'subject': case['subject'], 'split': 'dev', 'source_id': source_record['source_id'],
            'source_au_vector_0_5': case['source_aus_0_5'], 'target_au_vector_0_5': case['target_aus_0_5'], 'au_order': AUS, 'condition_status': 'complete'})
    del source_runtime
    # Existing source geometry is re-bound to the verified exactly same decoded source pixels.
    geometry_doc = read(EXP / 'improvements/smooth_regions_v7_20260915/masks/first8/manifest.json')
    source_measurements = read(verified_ref(geometry_doc['inputs']['source_measurements'], 'existing DEV source measurement'))
    source_map = {r['case_id']: r for r in source_measurements['records'] if r['method'] == 'source_real'}
    masks = {}
    for case in mapped_cases:
        measurement = dict(source_map[case['case_id']])
        measurement.update(path=str(canonical_path), measurement_source_path=str(canonical_path), sha256=sha(canonical_path))
        edit, cleanup, receipt = build_masks(case, source_record, measurement)
        mask_path = out / (case['case_id'] + '_geometry.json')
        save(mask_path, receipt)
        masks[case['case_id']] = edit, cleanup, ref(mask_path)
    features = {'source_anchor': anchor, 'sources': {source_record['source_id']: source_features}, 'conditions': conditions}
    cache = {'source_anchor': binding, 'sources': [source_record]}
    runtime = run_grid.Runtime(features, cache, out, torch)
    comparisons = []
    try:
        for case in mapped_cases:
            for variant, old_method in (('base', 'baseline_local_lp_s050'), ('bidirectional', 'bidirectional_local_lp_s050')):
                prepared = runtime.prepare(variant, case, source_record, masks[case['case_id']])
                image, receipt = runtime.sample(prepared, 101)
                path = out / (case['case_id'] + '_' + variant + '.png')
                image.save(path)
                old = expected[case['case_id'], old_method]
                checks = {'output_sha256': sha(path) == old['sha256'], 'conditions': receipt['condition_hashes'] == old['condition_hashes'],
                    'projection_trace': receipt['projection_trace'] == old['projection_trace'],
                    'source_latent': receipt['source_latent_sha256'] == old['source_latent_sha256'],
                    'noise': receipt['noise_sha256'] == old['noise_sha256'], 'calls_15_0_15':
                    (receipt['deis_unet_calls'], receipt['terminal_unet_calls'], receipt['total_unet_calls']) == (15, 0, 15)}
                comparisons.append({'case_id': case['case_id'], 'variant': variant, 'output': ref(path), 'expected_sha256': old['sha256'],
                    'checks': checks, 'passed': all(checks.values()), 'receipt': receipt})
                save(out / (case['case_id'] + '_' + variant + '.json'), comparisons[-1])
        result = {'kind': 'selected_v11_dev_parity', 'status': 'complete', 'passed': all(r['passed'] for r in comparisons),
            'case_ids': [c['case_id'] for c in cases], 'feature_checks': feature_checks, 'records': comparisons,
            'source_pixel_parity': source_parity,
            'new_validation_pixels_read': False, 'training_performed': False, 'policy': POLICY}
        save(out / 'result.json', result)
        print(json.dumps({'status': 'complete', 'passed': result['passed'], 'result': ref(out / 'result.json')}))
        return 0 if result['passed'] else 2
    finally:
        runtime.close()


if __name__ == '__main__':
    raise SystemExit(main())
