"""Build a new frozen source-only cache with the existing training-compatible runtime."""
import argparse
from pathlib import Path
import traceback
from common import *


def raw_source_plan(labels, raw, freeze_sha):
    need(raw['protocol_sha256'] == freeze_sha, 'Raw source manifest freeze mismatch')
    need(raw['status'] in ('raw_frames_extracted', 'completed_with_failures'), 'Raw manifest integrity/incomplete status rejected')
    rows = {r['case_id']: r for r in raw['cases']}
    need(set(rows) == {c['case_id'] for c in labels} and len(rows) == len(raw['cases']), 'Raw case set changed')
    sources, cases = {}, []
    for label in labels:
        row = rows[label['case_id']]
        need(all(row.get(key) == label[key] for key in ('subject', 'split', 'camera')), 'Raw case identity/role/camera mismatch')
        inputs = row['generation_input']
        need(inputs['source_au_vector_0_5'] == label['source_au_vector_0_5'] and
             inputs['target_au_vector_0_5'] == label['target_au_vector_0_5'], 'Raw AU label mismatch')
        raw_source = inputs['source_image']
        need(raw_source['label_frame_1based'] == label['source_frame_1based'], 'Raw source frame mismatch')
        need(all(raw_source.get(key) == label[key] for key in ('subject', 'split', 'camera')), 'Raw source identity/role/camera mismatch')
        need(raw_source['video_frame_zero_based'] == label['source_frame_1based'] - 1, 'Raw source zero-based frame mismatch')
        need(raw_source['original_video_path'] == label['video_path'], 'Raw source video provenance mismatch')
        need(re.fullmatch(r'[0-9a-f]{64}', raw_source['original_video_sha256']), 'Raw source video SHA required')
        source_id = label['subject'] + '_source_f' + str(label['source_frame_1based']).zfill(5)
        source = {'source_id': source_id, 'subject': label['subject'], 'split': label['split'],
                  'source_frame_1based': label['source_frame_1based'], 'raw_source': dict(raw_source)}
        need(source_id not in sources or sources[source_id] == source, 'Inconsistent shared source')
        sources[source_id] = source
        cases.append({**label, 'source_id': source_id})
    need(len(sources) == 8, 'Eight shared sources required')
    return list(sources.values()), cases


def build(labels, raw, freeze_sha, output, runtime, anchor, anchor_binding, torch):
    from PIL import Image
    from prepare_training_cache import make_prompt
    sources, cases = raw_source_plan(labels, raw, freeze_sha)
    need(not output.exists() and COMPLETION in output.resolve().parents, 'New completion source-cache path required')
    output.mkdir(parents=True)
    (output / 'canonical_sources').mkdir()
    features = {'source_anchor': anchor.cpu(), 'sources': {}, 'conditions': {}}
    check_tensor = module('source_runtime').check_tensor
    for source in sources:
        try:
            raw_source = source['raw_source']
            need(raw_source['status'] == 'complete', 'Raw source extraction failed')
            path = verified_ref(raw_source, 'raw SOURCE')
            with Image.open(path) as image:
                need(image.mode == 'RGB' and image.size == (raw_source['width'], raw_source['height']), 'Raw source image dimensions changed')
                canonical, tensors, alignment = runtime.source(image.copy())
            need(canonical.mode == 'RGB' and canonical.size == (512, 512), 'Canonical source must be512RGB')
            shapes = {'image_patches': (49, 512), 'source_identity': (512,), 'base_faceid': (512,)}
            need(set(tensors) == set(shapes), 'Unexpected/target source tensor')
            for key, shape in shapes.items():
                check_tensor(tensors[key], shape, key)
            target = output / 'canonical_sources' / (source['source_id'] + '.png')
            canonical.save(target)
            need(sha(path) == raw_source['sha256'], 'Raw source changed during preparation')
            features['sources'][source['source_id']] = tensors
            source.update(status='complete', canonical_source={**ref(target), 'width': 512, 'height': 512, 'mode': 'RGB'}, alignment=alignment)
        except Exception as error:
            source.update(status='failed', canonical_source=None, error=str(error), traceback=traceback.format_exc())
    prompts = {}
    for case in cases:
        case['prompt'] = make_prompt(case['target_au_vector_0_5'])
        try:
            if case['prompt'] not in prompts:
                prompts[case['prompt']] = runtime.text(case['prompt'])
            tokens, pooled = prompts[case['prompt']]
            check_tensor(tokens, (77, 768), 'text_tokens')
            check_tensor(pooled, (768,), 'text_pooled')
            need(any(case['target_au_vector_0_5']) or torch.equal(pooled, anchor), 'Noedit text must equal TRAIN source anchor exactly')
            features['conditions'][case['case_id']] = {'text_tokens': tokens, 'text_pooled': pooled,
                'source_aus': torch.tensor(case['source_au_vector_0_5'], dtype=torch.float32),
                'target_aus': torch.tensor(case['target_au_vector_0_5'], dtype=torch.float32), 'prompt': case['prompt']}
            case['condition_status'] = 'complete'
        except Exception as error:
            case.update(condition_status='failed', error=str(error), traceback=traceback.format_exc())
    path = output / 'features.pt'
    with path.open('xb') as stream:
        torch.save(features, stream)
    document = {'kind': 'completion_v11_source_feature_cache', 'protocol_sha256': freeze_sha,
        'status': 'complete' if all(s['status'] == 'complete' for s in sources) and all(c['condition_status'] == 'complete' for c in cases) else 'completed_with_failures',
        'sources': sources, 'cases': cases, 'case_ids': [c['case_id'] for c in cases], 'sampling_seeds': SEEDS,
        'source_anchor': anchor_binding, 'features_file': ref(path),
        'target_pixels_read': False, 'target_latents_computed': False, 'vae_loaded': False,
        'device': str(runtime.device), 'dtype': runtime.dtype, 'source_assets': runtime.assets,
        'environment': runtime.environment, 'source_cache_code': ref(__file__)}
    save(output / 'manifest.json', document)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha', required=True)
    parser.add_argument('--raw-manifest', type=Path, required=True)
    parser.add_argument('--raw-manifest-sha', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    protocol, labels = load_freeze(args.freeze, args.freeze_sha)
    need(sha(args.raw_manifest) == args.raw_manifest_sha, 'Raw manifest SHA mismatch')
    raw = read(args.raw_manifest)
    need(raw['label_plan_sha256'] == protocol['label_plan']['sha256'], 'Raw extraction label plan changed')
    sources, cases = raw_source_plan(labels, raw, args.freeze_sha)
    if not args.execute:
        print(json.dumps({'status': 'prepared_no_models', 'source_count': len(sources), 'case_count': len(cases)}))
        return
    need(args.output is not None, '--output required with --execute')
    torch = environment(POLICY['source_device'])
    anchor, binding, payload = load_anchor()
    runtime = module('source_runtime').SourceRuntime(POLICY['source_device'], payload)
    result = build(labels, raw, args.freeze_sha, args.output.resolve(), runtime, anchor, binding, torch)
    print(json.dumps({'status': result['status'], 'manifest': str(args.output / 'manifest.json')}))


if __name__ == '__main__':
    main()
