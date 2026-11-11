"""Complete fixed selected V11 on existing19 DEV cases; no new cases or target pixels."""
import argparse
import json
from pathlib import Path
import time
import traceback
from common import *
import run_grid

DEV_PINS = {
    'cases': ('development_evaluation/dev_coverage_audit/dev19_manifest_2k.json', '90a0307b053d369e49286ea63ad3f6e731c8213b3786fbd4591706c1b7f69a28'),
    'comparison': ('development_evaluation/evaluation_policy_v2/comparison_nine/case_output_manifest.json', '9fa2105acb99e319f6adac2a329f7b6a3ca4c324e8920f7d5f5f2c64261ca4d8'),
    'cache_manifest': ('training_cache/train2k_batched_202609141853/manifest.json', '4a378e75a478cf241ef08b3de3727bfa3e6da25e505b85ca1bc9cee74c57b413'),
    'cache_features': ('training_cache/train2k_batched_202609141853/features.pt', '7ede6afdfc6948ba4da845309033edc700466d80fc91bb5921410b780e8133e0'),
    'source_measurements': ('development_evaluation/evaluation_policy_v2/dev19_openface/results.json', 'c676f1aef38bb86b80cc4fc7495cbb2bbca6fd7d36d59f6350d9535234c4734b'),
    'selected_cases': ('improvements/a_local_cleanup_v11_20260915/runs/local_202609152016/cases.json', '87715a940a1e93de0c876710a74ab575add747f8ad340f818b041dcbc3a15344'),
    'selected_result': ('improvements/a_local_cleanup_v11_20260915/runs/local_202609152016/result.json', '37c239afa6d4dac1ca1e5517b60de420a58f72a833ff6193b99340791a3a7451'),
    'full_constructor_parity': ('completion_20260915/generation/dev_parity_202609152317/result.json', '5f415cb6a70e64ef0277fa20287b621d3e0843f08f91469b3c6a2d88f02c27d9'),
}
DEV_SEEDS = [101]
SUBJECTS = ['SN004', 'SN012', 'SN017', 'SN018']


def validate_dev_labels(document):
    need(document['test_data_used'] is False and document['status'] == 'frozen_additional_development', 'Must use frozen existing development labels')
    cases = document['cases']
    need(len(cases) == 19 and len({c['case_id'] for c in cases}) == 19, 'Exactly19 unique dev cases required')
    need([c['case_id'] for c in cases] == document['original_case_ids'] + document['additional_case_ids'], 'Original frozen dev case order changed')
    need(sorted({c['subject'] for c in cases}) == SUBJECTS, 'Frozen four dev subjects required')
    coverage = set()
    noedit_count = 0
    for case in cases:
        need(case['split'] == 'dev', 'Dev cases must retain actual dev role')
        source, target = case['source_au_vector_0_5'], case['target_au_vector_0_5']
        need(len(source) == len(target) == 12 and not any(source), 'Known zeroAU source required')
        need(all(type(value) in (int, float) and 0 <= value <= 5 for value in target), 'Invalid frozen labels')
        if not any(target):
            noedit_count += 1
            need(case['group'] == 'no_edit', 'Noedit group mismatch')
        for au, before, after in zip(AUS, source, target):
            if before != after and after in (1, 2, 3):
                coverage.add((au, after))
    need(noedit_count == 4 and len(coverage) == 36, 'Original noedit/36AU-level coverage changed')
    return cases, sorted(coverage)


def plan():
    refs = {}
    for role, (relative, expected) in DEV_PINS.items():
        path = EXP / relative
        need(sha(path) == expected, 'Frozen dev input changed: ' + role)
        refs[role] = ref(path)
    need(read(HERE / 'generation_spec_candidate_v3.json') == execution_spec(), 'Four core generation files/spec changed')
    need(read(refs['full_constructor_parity']['path'])['passed'] is True, 'Full V11 constructor parity must pass')
    labels, coverage = validate_dev_labels(read(refs['cases']['path']))
    comparison = read(refs['comparison']['path'])
    need(comparison['status'] == 'frozen_complete' and comparison['test_used'] is False and comparison['case_count'] == 19, 'Comparison reference is not frozen dev19')
    comp = {c['case_id']: c for c in comparison['cases']}
    need(set(comp) == {c['case_id'] for c in labels}, 'Comparison case set mismatch')
    cache_meta = read(refs['cache_manifest']['path'])
    need(cache_meta['test_data_used'] is False and cache_meta['cache_sha256'] == refs['cache_features']['sha256'], 'Original cache binding mismatch')
    pairs = {r['case_id']: r for r in cache_meta['pairs']}
    measurement_doc = read(refs['source_measurements']['path'])
    need(measurement_doc['status'] == 'complete' and measurement_doc['test_used'] is False, 'Existing source measurements incomplete')
    measurements = {r['case_id']: r for r in measurement_doc['records'] if r['method'] == 'source_real'}
    need(len(measurements) == 19 and set(measurements) == set(comp), 'One original source measurement per dev case required')
    sources, cases = {}, []
    for label in labels:
        cid = label['case_id']
        pair, original = pairs[cid], comp[cid]
        need(pair['split'] == original['split'] == 'dev' and pair['subject'] == original['subject'] == label['subject'], 'Existing dev identity/role changed')
        # Read source refs and original target AU labels only. No target path/stat/open.
        source_ref = original['reference_frames']['source']
        need(source_ref['manual_AU_0_5'] == label['source_au_vector_0_5'], 'Reference sourceAU changed')
        need(original['reference_frames']['target']['manual_AU_0_5'] == label['target_au_vector_0_5'], 'Reference targetAU labels changed')
        need(pair['source_au_vector_0_5'] == label['source_au_vector_0_5'] and pair['target_au_vector_0_5'] == label['target_au_vector_0_5'], 'Cached pairAU changed')
        canonical = {'path': source_ref['evaluation_path'], 'sha256': source_ref['evaluation_sha256'], 'width': 512, 'height': 512, 'mode': 'RGB'}
        verified_ref(canonical, 'comparison canonical SOURCE')
        need(pair['aligned_source']['aligned_sha256'] == canonical['sha256'], 'Training/comparison canonical SOURCE differs')
        measurement = measurements[cid]
        need(measurement['sha256'] == canonical['sha256'] and Path(measurement['path']).resolve() == Path(canonical['path']).resolve(), 'Source measurement path/SHA changed')
        sid = label['subject'] + '_existing_source_f' + str(source_ref['label_frame']).zfill(5)
        source = {'source_id': sid, 'subject': label['subject'], 'split': 'dev', 'status': 'complete',
            'canonical_source': canonical, 'source_frame_1based': source_ref['label_frame'], 'camera': source_ref['camera']}
        # Cases use different historical copies of exactly the same PNG; runtime encodes one per subject.
        if sid in sources:
            need(sources[sid]['canonical_source']['sha256'] == canonical['sha256'], 'Shared dev source changed')
        else:
            sources[sid] = source
        cases.append({**label, 'source_id': sid, 'au_order': AUS, 'prompt': pair['prompt'], 'condition_status': 'complete',
            'case_canonical_source': canonical})
    need(len(sources) == 4, 'Four unique canonical dev sources required')
    old_cases = {c['case_id']: c for c in read(refs['selected_cases']['path'])['cases']}
    need(len(old_cases) == 8 and set(old_cases) <= set(comp), 'Selected V11 subset changed')
    expected = {}
    for row in read(refs['selected_result']['path'])['records']:
        if row['method'] not in ('baseline_local_lp_s050', 'bidirectional_local_lp_s050'):
            continue
        need(row['status'] == 'complete' and row['seed'] == 101, 'Selected V11 expected record invalid')
        need(sha(row['output']) == row['sha256'], 'Existing selected V11 output changed')
        expected[row['case_id'], 'base' if row['method'].startswith('baseline_') else 'bidirectional'] = row
    need(len(expected) == 16, 'Exact16 old8case×2methods parity rows required')
    return {'refs': refs, 'cases': cases, 'sources': list(sources.values()), 'measurements': measurements,
        'old_cases': old_cases, 'expected': expected, 'coverage_cells': coverage,
        'core_generation': execution_spec(), 'sampling_seeds': DEV_SEEDS}


def prepare_masks(inputs, output=None):
    import numpy as np
    from geometry import build_masks
    sources = {s['source_id']: s for s in inputs['sources']}
    masks, records = {}, []
    for case in inputs['cases']:
        source = sources[case['source_id']]
        measurement = dict(inputs['measurements'][case['case_id']])
        original_ref = {'path': measurement['path'], 'sha256': measurement['sha256']}
        # All source-copy SHAs were checked in plan; rebind the identical image measurement to the shared source path.
        measurement.update(path=source['canonical_source']['path'], measurement_source_path=source['canonical_source']['path'])
        edit, cleanup, receipt = build_masks(case, source, measurement)
        receipt.update(original_case_source_measurement=original_ref, original_measurement_case_id=case['case_id'])
        if case['case_id'] in inputs['old_cases']:
            old = inputs['old_cases'][case['case_id']]
            old_edit, old_cleanup = old['region_mask'], old['cleanup_mask']
            verified_ref(old_edit, 'old V11 editmask')
            verified_ref(old_cleanup, 'old V11 cleanupmask')
            need(np.array_equal(edit, np.load(old_edit['path'], allow_pickle=False)), 'Old8 editmask parity failed')
            need(np.array_equal(cleanup, np.load(old_cleanup['path'], allow_pickle=False)), 'Old8 cleanupmask parity failed')
            receipt['old_v11_mask_parity'] = {'edit_exact': True, 'cleanup_exact': True}
        if output is not None:
            folder = output / 'masks' / case['case_id']
            folder.mkdir(parents=True)
            for name, value in (('edit64', edit), ('cleanup64', cleanup)):
                with (folder / (name + '.npy')).open('xb') as stream:
                    np.save(stream, value, allow_pickle=False)
            receipt.update(edit_mask=ref(folder / 'edit64.npy'), cleanup_mask=ref(folder / 'cleanup64.npy'))
            save(folder / 'receipt.json', receipt)
            stored_receipt = ref(folder / 'receipt.json')
        else:
            stored_receipt = {'receipt_sha256': canonical_sha(receipt), 'prepared_no_output': True}
        masks[case['case_id']] = edit, cleanup, stored_receipt
        records.append({'case_id': case['case_id'], 'status': 'complete', 'receipt': stored_receipt,
            'edit_nonzero_cells': int(np.count_nonzero(edit)), 'cleanup_nonzero_cells': int(np.count_nonzero(cleanup)),
            'old_v11_mask_parity': receipt.get('old_v11_mask_parity')})
    return masks, records


def source_only_features(inputs, torch):
    """Whitelist only source features/conditions from the pinned existing TRAIN/DEV tensor archive."""
    from prepare_training_cache import make_prompt
    anchor, binding, _ = load_anchor()
    archive = torch.load(inputs['refs']['cache_features']['path'], map_location='cpu', weights_only=True)['samples']
    need(all(s['split'] in ('train', 'dev') for s in archive), 'Original archive includes an unexpected role')
    samples = {s['case_id']: s for s in archive}
    features = {'source_anchor': anchor, 'sources': {}, 'conditions': {}}
    receipts = []
    for case in inputs['cases']:
        sample = samples[case['case_id']]
        need(sample['subject'] == case['subject'] and sample['split'] == 'dev', 'Whitelisted sample role changed')
        need(sample['source_aus'].tolist() == case['source_au_vector_0_5'] and sample['target_aus'].tolist() == case['target_au_vector_0_5'], 'Whitelisted manualAU changed')
        need(sample['prompt'] == case['prompt'] == make_prompt(case['target_au_vector_0_5']), 'Original label-derived prompt changed')
        source = {k: sample[k].detach().cpu().clone() for k in ('image_patches', 'source_identity', 'base_faceid')}
        sid = case['source_id']
        if sid in features['sources']:
            need(all(torch.equal(source[k], features['sources'][sid][k]) for k in source), 'Same-source cached features differ')
        features['sources'][sid] = source
        condition = {k: sample[k].detach().cpu().clone() for k in ('text_tokens', 'text_pooled', 'source_aus', 'target_aus')}
        condition['prompt'] = sample['prompt']
        need(any(case['target_au_vector_0_5']) or torch.equal(condition['text_pooled'], anchor), 'Noedit anchor mismatch')
        features['conditions'][case['case_id']] = condition
        receipts.append({'case_id': case['case_id'], 'source_feature_hashes': {k: tensor_sha(v) for k, v in source.items()},
            'condition_feature_hashes': {k: tensor_sha(v) for k, v in condition.items() if isinstance(v, torch.Tensor)}})
    del archive, samples
    return features, {'source_anchor': binding, 'sources': inputs['sources'], 'cases': inputs['cases']}, receipts


def selected_parity(receipt, output_sha, expected):
    checks = {'png_sha256': output_sha == expected['sha256'], 'conditions': receipt['condition_hashes'] == expected['condition_hashes'],
        'projection_trace': receipt['projection_trace'] == expected['projection_trace'],
        'source_latent': receipt['source_latent_sha256'] == expected['source_latent_sha256'],
        'noise': receipt['noise_sha256'] == expected['noise_sha256'],
        'calls_15_0_15': (receipt['deis_unet_calls'], receipt['terminal_unet_calls'], receipt['total_unet_calls']) == (15, 0, 15)}
    return {'expected_output_sha256': expected['sha256'], 'checks': checks, 'passed': all(checks.values())}


def retain_failed_artifact(row):
    if row.get('output') is not None:
        row['saved_output_unverified'] = {'path': row['output'], 'sha256': row.get('sha256'), 'eligible_for_evaluation': False}
    row['output'], row['sha256'] = None, None


def execute_dev_grid(inputs, masks, output, runtime):
    sources = {s['source_id']: s for s in inputs['sources']}
    records = []
    with (output / 'generation_records.jsonl').open('x', buffering=1) as log:
        for case in inputs['cases']:
            source = sources[case['source_id']]
            for variant in VARIANTS:
                method = variant + '_local_lp_s050'
                row = {'case_id': case['case_id'], 'subject': case['subject'], 'split': 'dev', 'method': method, 'variant': variant,
                    'seed': 101, 'execution_id': method + '::' + case['case_id'] + '::101', 'execution_started': False,
                    'group': 'no_edit' if not any(case['target_au_vector_0_5']) else 'nonzero_target', 'development_group': case['group'],
                    'source_aus_0_5': case['source_au_vector_0_5'], 'target_aus_0_5': case['target_au_vector_0_5'],
                    'source_real': source['canonical_source'], 'prompt': case['prompt'], 'output': None, 'sha256': None}
                started = time.monotonic()
                try:
                    need(case['case_id'] in masks, 'Source geometry preparation failed')
                    prepared = runtime.prepare(variant, case, source, masks[case['case_id']])
                    row['execution_started'] = True
                    image, receipt = runtime.sample(prepared, 101)
                    row.update(receipt)
                    need(image.mode == 'RGB' and image.size == (512, 512), 'Native512RGB output required')
                    folder = output / 'images' / case['case_id']
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / (method + '_seed101.png')
                    need(not path.exists(), 'Refusing dev output overwrite')
                    image.save(path)
                    row.update(output=str(path), sha256=sha(path))
                    if row['group'] == 'no_edit':
                        need(row['sha256'] == prepared['reconstruction']['sha256'], 'Noedit source VAE equality failed')
                    expected = inputs['expected'].get((case['case_id'], variant))
                    if expected is not None:
                        row['old_v11_parity'] = selected_parity(receipt, row['sha256'], expected)
                        need(row['old_v11_parity']['passed'], 'Existing V11 subset strict output/condition/trace parity failed')
                    row['status'] = 'complete'
                except Exception as error:
                    row.update(getattr(error, 'generation_receipt', {}))
                    row.update(status='failed' if row['execution_started'] else 'not_started', error=str(error), traceback=traceback.format_exc())
                    retain_failed_artifact(row)
                row['seconds'] = time.monotonic() - started
                records.append(row)
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                print(json.dumps({k: row[k] for k in ('case_id', 'method', 'status', 'seconds')}), flush=True)
    need(len(records) == len(inputs['cases']) * len(VARIANTS), 'Development grid accounting mismatch')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    inputs = plan()
    masks, mask_records = prepare_masks(inputs)
    if not args.execute:
        print(json.dumps({'status': 'prepared_no_models', 'split': 'dev', 'case_count': 19, 'source_count': 4,
            'coverage_cells': len(inputs['coverage_cells']), 'variants': VARIANTS, 'seeds': DEV_SEEDS, 'planned': 95,
            'mask_count': len(mask_records), 'old_v11_mask_cases_exact': sum(r['old_v11_mask_parity'] is not None for r in mask_records),
            'existing_output_parity_rows': len(inputs['expected']), 'models_loaded': False, 'target_pixels_read': False}))
        return
    need(args.output is not None, 'Explicit new dev19 output required')
    output = args.output.resolve()
    need(output.parent == COMPLETION / 'dev19_runs' and not output.exists(), 'Output must be new completion/dev19_runs/name')
    output.mkdir(parents=True)
    save(output / 'configuration.json', {'kind': 'selected_v11_existing_dev19_completion', 'split': 'dev',
        'inputs': inputs['refs'], 'core_generation': inputs['core_generation'], 'runner': ref(__file__),
        'case_ids': [c['case_id'] for c in inputs['cases']], 'variants': VARIANTS, 'sampling_seeds': DEV_SEEDS, 'planned': 95,
        'coverage_cells': inputs['coverage_cells'], 'training_performed': False, 'hyperparameters_changed': False,
        'target_pixels_read': False, 'target_latents_supplied_to_runtime': False,
        'interpretation': 'Complete already exposed development coverage, not independent final test'} )
    save(output / 'cases.json', {'kind': 'existing_dev19_source_only_cases', 'split': 'dev', 'case_count': 19,
        'sources': inputs['sources'], 'cases': inputs['cases'], 'target_pixels_read': False})
    masks, mask_records = prepare_masks(inputs, output)
    save(output / 'mask_manifest.json', {'source_only': True, 'records': mask_records})
    torch = environment('mps')
    features, cache, feature_receipts = source_only_features(inputs, torch)
    save(output / 'source_feature_receipts.json', {'source_anchor': cache['source_anchor'], 'records': feature_receipts,
        'target_latents_supplied_to_runtime': False})
    runtime = None
    try:
        runtime = run_grid.Runtime(features, cache, output, torch)
        records = execute_dev_grid(inputs, masks, output, runtime)
        result = {'kind': 'selected_v11_existing_dev19_completion', 'status': 'complete' if all(r['status'] == 'complete' for r in records) else 'completed_with_failures',
            'split': 'dev', 'planned': 95, 'completed': sum(r['status'] == 'complete' for r in records), 'records': records,
            'configuration': ref(output / 'configuration.json'), 'runtime': ref(output / 'runtime.json'),
            'sampling_seeds': DEV_SEEDS, 'all_planned_cases_and_seeds_retained': True,
            'old_v11_parity_rows': sum('old_v11_parity' in r for r in records),
            'target_pixels_used': False, 'target_latents_computed': False, 'training_performed': False}
        save(output / 'result.json', result)
        print(json.dumps({'status': result['status'], 'result': ref(output / 'result.json')}), flush=True)
        return 0 if result['status'] == 'complete' else 2
    except Exception as error:
        save(output / 'fatal_error.json', {'error': str(error), 'traceback': traceback.format_exc(), 'result_is_incomplete': True})
        raise
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == '__main__':
    raise SystemExit(main())
