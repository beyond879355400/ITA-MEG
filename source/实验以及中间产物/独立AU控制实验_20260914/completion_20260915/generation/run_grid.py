"""Frozen complete V11 source-only generation; default metadata preflight, no model."""
import argparse
from pathlib import Path
import time
import traceback
from common import *


def load_cache(path, expected, labels, freeze_sha, load_tensors=False):
    need(sha(path) == expected, 'Source cache manifest SHA mismatch')
    cache = read(path)
    need(cache['kind'] == 'completion_v11_source_feature_cache' and cache['status'] in ('complete', 'completed_with_failures'), 'Source cache invalid')
    need(cache['protocol_sha256'] == freeze_sha and cache['case_ids'] == [c['case_id'] for c in labels] and cache['sampling_seeds'] == SEEDS, 'Cache freeze/case/seed mismatch')
    need(cache['target_pixels_read'] is False and cache['target_latents_computed'] is False and cache['vae_loaded'] is False, 'Cache source-only boundary failed')
    need(cache['source_cache_code']['sha256'] == sha(HERE / 'prepare_cache.py'), 'Cache implementation mismatch')
    need(cache['device'] == POLICY['source_device'] and cache['dtype'] == POLICY['dtype'], 'Cache precision/device mismatch')
    sources = {s['source_id']: s for s in cache['sources']}
    need(len(sources) == 8 and len(sources) == len(cache['sources']), 'Eight unique source rows required')
    need(len(cache['cases']) == len(labels), 'Cache case count mismatch')
    for actual, label in zip(cache['cases'], labels):
        need(all(actual.get(k) == v for k, v in label.items()), 'Cache case labels changed')
        source = sources[actual['source_id']]
        need(source['subject'] == label['subject'] and source['split'] == label['split'] and source['source_frame_1based'] == label['source_frame_1based'], 'Cache source binding mismatch')
        need(actual['condition_status'] in ('complete', 'failed'), 'Incomplete cache condition')
    for source in sources.values():
        need(source['status'] in ('complete', 'failed'), 'Incomplete cache source')
        if source['status'] == 'complete':
            verified_ref(source['canonical_source'], 'canonical SOURCE')
            need((source['canonical_source']['width'], source['canonical_source']['height'], source['canonical_source']['mode']) == (512, 512, 'RGB'), 'Canonical source format changed')
    feature_path = verified_ref(cache['features_file'], 'source feature tensors')
    if not load_tensors:
        return cache, None
    import torch
    features = torch.load(feature_path, map_location='cpu', weights_only=True)
    need(set(features) == {'source_anchor', 'sources', 'conditions'}, 'Unexpected features; target latents forbidden')
    check = module('source_runtime').check_tensor
    check(features['source_anchor'], (768,), 'anchor')
    need(tensor_sha(features['source_anchor']) == cache['source_anchor']['text_pooled_sha256'], 'Source anchor tensor changed')
    need(set(features['sources']) == {s['source_id'] for s in sources.values() if s['status'] == 'complete'}, 'Source feature success set changed')
    for values in features['sources'].values():
        shapes = {'image_patches': (49, 512), 'source_identity': (512,), 'base_faceid': (512,)}
        need(set(values) == set(shapes), 'Unexpected source feature fields')
        for key, shape in shapes.items():
            check(values[key], shape, key)
    need(set(features['conditions']) == {c['case_id'] for c in cache['cases'] if c['condition_status'] == 'complete'}, 'Condition feature success set changed')
    for case in cache['cases']:
        if case['condition_status'] != 'complete':
            continue
        values = features['conditions'][case['case_id']]
        shapes = {'text_tokens': (77, 768), 'text_pooled': (768,), 'source_aus': (12,), 'target_aus': (12,)}
        need(set(values) == set(shapes) | {'prompt'}, 'Unexpected condition/target latent fields')
        for key, shape in shapes.items():
            check(values[key], shape, key)
        need(values['source_aus'].tolist() == case['source_au_vector_0_5'] and values['target_aus'].tolist() == case['target_au_vector_0_5'], 'Cached labels changed')
        need(values['prompt'] == case['prompt'], 'Prompt changed')
        need(any(case['target_au_vector_0_5']) or torch.equal(values['text_pooled'], features['source_anchor']), 'Noedit source anchor changed')
    return cache, features


def fixed_controller(v11, unet, source, edit, cross, torch, cleanup):
    class Controller(v11.v10.TerminalController):
        def __init__(self):
            super().__init__(unet, 'terminal_last_eps', source, edit, cross, torch)

        def correct(self, latent, pipeline, prompt_embeds):
            clean = super().correct(latent, pipeline, prompt_embeds)
            result = v11.local_filtered_clean(clean, source, edit, cleanup, .5, torch)
            self.terminal_trace.update(mode='local_lp_s050', base_mode='terminal_last_eps',
                base_terminal_clean_latent_sha256=tensor_sha(clean), post_filter_clean_latent_sha256=tensor_sha(result),
                filter=v11.filter_metadata(.5), outside_cleanup_clean_exact=True)
            return result
    return Controller()


def sampling_callback(controller, source, noise, edit, torch):
    history = []
    def step_end(pipeline, index, timestep, kwargs):
        latents = kwargs['latents']
        need(torch.isfinite(latents).all().item(), 'Nonfinite denoising latent')
        times = pipeline.scheduler.timesteps[5:]
        need([int(v) for v in times] == TIMESTEPS and int(timestep) == int(times[index]), 'Frozen timestep trajectory changed')
        next_t = times[index + 1] if index + 1 < len(times) else None
        if next_t is None:
            latents = controller.correct(latents, pipeline, kwargs['prompt_embeds'])
        reference = source if next_t is None else pipeline.scheduler.add_noise(source, noise, next_t.reshape(1))
        projected = edit * latents + (1 - edit) * reference
        need(torch.isfinite(projected).all().item(), 'Nonfinite source projection')
        if next_t is None:
            controller.finalize_projection(projected)
        history.append({'t': int(timestep), 'next_t': None if next_t is None else int(next_t), 'latent_sha256': tensor_sha(projected)})
        return {'latents': projected}
    return step_end, history


def conditions_for_variant(backbone, adapter, trust, source, condition, anchor, device, torch):
    """Base branch is exactly the original backbone; no adapter or residual operation."""
    aus, identity = condition['target_aus'][None].to(device), source['base_faceid'][None].to(device)
    base_au, base_ip = backbone.baseline_conditions(aus, identity)
    null_au, null_ip = backbone.baseline_conditions(torch.zeros_like(aus), torch.zeros_like(identity))
    values = {'au_embedding': base_au, 'image_embedding': base_ip}
    if adapter is not None:
        from train_centered_adapter import centered_batch_conditions
        batch = {k: condition[k][None].to(device) for k in ('text_pooled', 'target_aus', 'source_aus')}
        batch['image_patches'] = source['image_patches'][None].to(device)
        _, _, values = centered_batch_conditions(adapter, batch, anchor.to(device), base_au, base_ip, trust)
        if torch.equal(condition['source_aus'], condition['target_aus']):
            need(torch.equal(values['au_embedding'], base_au) and torch.equal(values['image_embedding'], base_ip), 'Noedit learned residual must be zero')
    need(torch.equal(values['au_embedding'][..., :12], base_au[..., :12]), 'Explicit12AU prefix changed')
    return {'au_embedding': torch.cat((null_au, values['au_embedding'])),
            'image_embedding': torch.cat((null_ip, values['image_embedding'])), 'au_scale': 1.5, 'ip_scale': .9}


class Runtime:
    def __init__(self, features, cache, output, torch):
        import numpy as np
        from PIL import Image
        from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL, DEISMultistepScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        from modeling.frozen_backbone import FrozenConditionedDenoiser, RV
        from modeling.safe_token_residual_adapter import RelativeRMSConfig
        from train_centered_adapter import load_checkpoint, adapter_from_checkpoint, source_and_asset_bindings, verify_source_anchor
        self.torch, self.features, self.device, self.output = torch, features, 'mps', output
        self.v11 = module('v11')
        anchor, binding, anchor_payload = load_anchor()
        need(cache['source_anchor'] == binding and torch.equal(features['source_anchor'], anchor), 'Training/source cache anchor differs')
        current, _ = source_and_asset_bindings({})
        need(current['public_assets'] == anchor_payload['bindings']['public_assets'], 'Public asset binding mismatch')
        self.adapters, self.checkpoints = {'base': (None, None)}, {}
        for variant, (folder, expected) in CKPTS.items():
            path = EXP / 'runs' / folder / 'checkpoint_00000200.pt'
            need(sha(path) == expected, 'Early200 checkpoint changed')
            state = load_checkpoint(path)
            need(state['variant'] == variant and state['global_step'] == 200 and state['diagnostic'] is False, 'Wrong checkpoint family/step')
            need(state['bindings']['public_assets'] == current['public_assets'] and state['bindings']['cache'] == anchor_payload['bindings']['cache'], 'Checkpoint base/cache mismatch')
            verify_source_anchor(anchor, binding, state['bindings']['source_anchor'])
            for relative, value in state['bindings']['sources'].items():
                need(sha(EXP / relative) == value, 'Frozen trained source changed: ' + relative)
            adapter = adapter_from_checkpoint(state, self.device).eval().requires_grad_(False)
            effective = state['effective_config']
            trust = RelativeRMSConfig(enabled=effective['trust_region'], rho_au=effective['rho_au'], rho_ip=effective['rho_ip'], reference_rms_floor=effective['reference_rms_floor'])
            self.adapters[variant] = (adapter, trust)
            self.checkpoints[variant] = {**ref(path), 'global_step': 200, 'variant': variant,
                'parameter_count': sum(p.numel() for p in adapter.parameters()), 'model_config': state['model_config'], 'effective_config': effective}
        self.backbone = FrozenConditionedDenoiser(self.device, checkpointing=False)
        vae = AutoencoderKL.from_pretrained(str(RV / 'vae'), local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False)
        text = CLIPTextModel.from_pretrained(str(RV / 'text_encoder'), local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False)
        self.pipe = StableDiffusionImg2ImgPipeline(vae=vae, text_encoder=text,
            tokenizer=CLIPTokenizer.from_pretrained(str(RV / 'tokenizer'), local_files_only=True), unet=self.backbone.unet,
            scheduler=DEISMultistepScheduler.from_pretrained(str(RV / 'scheduler'), local_files_only=True),
            safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        def finite_decode(module, args, result):
            need(torch.isfinite(result).all().item(), 'Nonfinite VAE decode')
        self.decode_guard = vae.decoder.register_forward_hook(finite_decode)
        self.sources, receipts = {}, []
        with torch.no_grad():
            for source in cache['sources']:
                receipt = {'source_id': source['source_id'], 'source': source['canonical_source']}
                try:
                    need(source['status'] == 'complete', 'Source cache failed')
                    path = verified_ref(source['canonical_source'], 'canonical SOURCE for VAE')
                    with Image.open(path) as image:
                        need(image.size == (512, 512) and image.mode == 'RGB', 'VAE source must be512RGB')
                        value = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float().to(self.device) / 127.5 - 1
                    latent = vae.encode(value).latent_dist.mean * vae.config.scaling_factor
                    need(torch.isfinite(latent).all().item(), 'Nonfinite source VAE latent')
                    decoded = vae.decode(latent / vae.config.scaling_factor).sample
                    reconstructed = self.pipe.image_processor.postprocess(decoded, output_type='pil', do_denormalize=[True])[0]
                    path = output / 'source_reconstructions' / (source['source_id'] + '.png')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    reconstructed.save(path)
                    self.sources[source['source_id']] = (latent, ref(path))
                    receipt.update(status='complete', latent_sha256=tensor_sha(latent), reconstruction=ref(path))
                except Exception as error:
                    receipt.update(status='failed', error=str(error), traceback=traceback.format_exc())
                receipts.append(receipt)
        latents_path = output / 'source_latents.pt'
        with latents_path.open('xb') as stream:
            torch.save({key: val[0].cpu() for key, val in self.sources.items()}, stream)
        self.receipt = {'device': self.device, 'dtype': 'torch.float32', 'torch': torch.__version__,
            'checkpoints': self.checkpoints, 'public_assets': current['public_assets'], 'source_anchor': binding,
            'scheduler': type(self.pipe.scheduler).__name__, 'scheduler_config': dict(self.pipe.scheduler.config),
            'source_latents': ref(latents_path), 'source_vae_records': receipts, 'target_encoder_calls': 0}
        save(output / 'runtime.json', self.receipt)

    def prepare(self, variant, case, source, mask):
        torch = self.torch
        need(source['source_id'] in self.sources, 'Source VAE preparation failed')
        condition = self.features['conditions'][case['case_id']]
        with torch.no_grad():
            cross = conditions_for_variant(self.backbone, *self.adapters[variant],
                self.features['sources'][source['source_id']], condition, self.features['source_anchor'], self.device, torch)
        latent, reconstruction = self.sources[source['source_id']]
        return {'prompt': condition['prompt'], 'cross': cross, 'source': latent,
            'reconstruction': reconstruction, 'edit': torch.from_numpy(mask[0])[None, None].to(self.device),
            'cleanup': torch.from_numpy(mask[1])[None, None].to(self.device), 'mask_receipt': mask[2],
            'condition_hashes': {k: tensor_sha(v) for k, v in cross.items() if isinstance(v, torch.Tensor)}}

    def sample(self, prepared, seed):
        torch, source = self.torch, prepared['source']
        noise = torch.randn(source.shape, generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32).to(self.device)
        controller = fixed_controller(self.v11, self.backbone.unet, source, prepared['edit'], prepared['cross'], torch, prepared['cleanup'])
        callback, history = sampling_callback(controller, source, noise, prepared['edit'], torch)
        receipt = {'condition_hashes': prepared['condition_hashes'], 'source_latent_sha256': tensor_sha(source),
            'noise_sha256': tensor_sha(noise), 'mask_receipt': prepared['mask_receipt'], 'source_vae_reconstruction': prepared['reconstruction']}
        try:
            with torch.no_grad():
                image = self.pipe(prompt=prepared['prompt'], image=source, strength=.75, num_inference_steps=20,
                    guidance_scale=7.5, negative_prompt='', generator=torch.Generator(device='cpu').manual_seed(seed),
                    cross_attention_kwargs=prepared['cross'], callback_on_step_end=callback,
                    callback_on_step_end_tensor_inputs=['latents', 'prompt_embeds']).images[0]
            controller.assert_complete()
            need(len(history) == 15, 'Actual callback count differs')
            return image, {**receipt, **controller.record(), 'projection_trace': history}
        except Exception as error:
            error.generation_receipt = {**receipt, **controller.record(), 'projection_trace': history}
            raise
        finally:
            controller.close()

    def close(self):
        self.decode_guard.remove()


def execute_grid(cache, measurements, output, runtime, masks):
    """All five methods, all cases, all seeds retained, including preparation failures."""
    results = []
    sources = {s['source_id']: s for s in cache['sources']}
    with (output / 'generation_records.jsonl').open('x', buffering=1) as log:
        for case in cache['cases']:
            source = sources[case['source_id']]
            for variant in VARIANTS:
                error, prepared = None, None
                try:
                    need(source['status'] == 'complete' and case['condition_status'] == 'complete', 'Source/text cache failed')
                    need(case['case_id'] in masks, 'Source geometry/mask failed')
                    prepared = runtime.prepare(variant, case, source, masks[case['case_id']])
                except Exception as exc:
                    error = str(exc)
                for seed in SEEDS:
                    method = variant + '_local_lp_s050'
                    row = {'case_id': case['case_id'], 'subject': case['subject'], 'split': case['split'],
                        'method': method, 'variant': variant, 'seed': seed, 'execution_id': f'{method}::{case["case_id"]}::{seed}',
                        'source_aus_0_5': case['source_au_vector_0_5'], 'target_aus_0_5': case['target_au_vector_0_5'],
                        'group': 'no_edit' if not any(case['target_au_vector_0_5']) else 'nonzero_target',
                        'source_real': source['canonical_source'], 'output': None, 'sha256': None, 'execution_started': False}
                    started = time.monotonic()
                    if error:
                        row.update(status='not_started', error=error)
                    else:
                        try:
                            row['execution_started'] = True
                            image, receipt = runtime.sample(prepared, seed)
                            row.update(receipt)
                            need(image.mode == 'RGB' and image.size == (512, 512), 'Native512RGB output required')
                            path = output / 'images' / case['case_id'] / (method + '_seed' + str(seed) + '.png')
                            path.parent.mkdir(parents=True, exist_ok=True)
                            need(not path.exists(), 'Refusing image overwrite')
                            image.save(path)
                            row.update(output=str(path), sha256=sha(path))
                            if row['group'] == 'no_edit':
                                need(row['sha256'] == prepared['reconstruction']['sha256'], 'Zeroedit must equal exact source VAE reconstruction')
                            row['status'] = 'complete'
                        except Exception as exc:
                            row.update(getattr(exc, 'generation_receipt', {}))
                            row.update(status='failed', error=str(exc), traceback=traceback.format_exc())
                            if row['output'] is not None:
                                row['saved_output_unverified'] = {'path': row['output'], 'sha256': row['sha256'], 'eligible_for_evaluation': False}
                            row['output'], row['sha256'] = None, None
                    row['seconds'] = time.monotonic() - started
                    results.append(row)
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    print(json.dumps({k: row[k] for k in ('case_id', 'method', 'seed', 'status', 'seconds')}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-spec', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--freeze-sha')
    parser.add_argument('--source-cache', type=Path)
    parser.add_argument('--source-cache-sha')
    parser.add_argument('--source-measurements', type=Path)
    parser.add_argument('--source-measurements-sha')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.emit_spec:
        save(args.emit_spec, execution_spec())
        print(json.dumps({'status': 'generation_spec_only', 'path': str(args.emit_spec)}))
        return
    need(all((args.freeze, args.freeze_sha, args.source_cache, args.source_cache_sha,
              args.source_measurements, args.source_measurements_sha)), 'Explicit frozen references required')
    _, labels = load_freeze(args.freeze, args.freeze_sha)
    torch = environment('mps') if args.execute else None
    cache, features = load_cache(args.source_cache, args.source_cache_sha, labels, args.freeze_sha, load_tensors=args.execute)
    need(sha(args.source_measurements) == args.source_measurements_sha, 'Source measurements SHA mismatch')
    measurements = read(args.source_measurements)
    need(measurements['protocol_sha256'] == args.freeze_sha and measurements['source_cache_sha256'] == args.source_cache_sha, 'Source geometry protocol/cache binding changed')
    need(measurements['source_only'] is True, 'Only source geometry allowed')
    if not args.execute:
        print(json.dumps({'status': 'prepared_no_models', 'cases': len(labels), 'variants': VARIANTS, 'seeds': SEEDS, 'planned': 480, 'models_loaded': False}))
        return
    need(args.output is not None, 'New --output required')
    out = args.output.resolve()
    need(COMPLETION in out.parents and not out.exists(), 'New completion output required')
    out.mkdir(parents=True)
    save(out / 'configuration.json', {'kind': 'completion_v11_generation', 'policy': POLICY,
        'freeze': ref(args.freeze), 'source_cache': ref(args.source_cache), 'source_measurements': ref(args.source_measurements),
        'generation': execution_spec(), 'planned': 480, 'target_pixels_used': False, 'target_latents_computed': False})
    from geometry import build_masks
    import numpy as np
    mapping = {r['case_id']: r for r in measurements['records'] if r.get('method') == 'source_real'}
    need(len(mapping) == len(measurements['records']) and set(mapping) == set(cache['case_ids']), 'Unique source measurement per planned case required')
    sources = {s['source_id']: s for s in cache['sources']}
    masks, mask_records = {}, []
    for case in cache['cases']:
        try:
            edit, cleanup, receipt = build_masks(case, sources[case['source_id']], mapping[case['case_id']])
            folder = out / 'masks' / case['case_id']
            folder.mkdir(parents=True)
            for name, array in (('edit64', edit), ('cleanup64', cleanup)):
                with (folder / (name + '.npy')).open('xb') as stream:
                    np.save(stream, array, allow_pickle=False)
            receipt.update(edit_mask=ref(folder / 'edit64.npy'), cleanup_mask=ref(folder / 'cleanup64.npy'))
            save(folder / 'receipt.json', receipt)
            masks[case['case_id']] = edit, cleanup, ref(folder / 'receipt.json')
            mask_records.append({'case_id': case['case_id'], 'status': 'complete', 'receipt': masks[case['case_id']][2]})
        except Exception as exc:
            mask_records.append({'case_id': case['case_id'], 'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
    save(out / 'mask_manifest.json', {'source_only': True, 'records': mask_records})
    runtime = None
    try:
        runtime = Runtime(features, cache, out, torch)
        records = execute_grid(cache, measurements, out, runtime, masks)
        result = {'kind': 'completion_v11_generation', 'status': 'complete' if all(r['status'] == 'complete' for r in records) else 'completed_with_failures',
            'planned': 480, 'completed': sum(r['status'] == 'complete' for r in records), 'records': records,
            'configuration': ref(out / 'configuration.json'), 'runtime': ref(out / 'runtime.json'),
            'all_planned_cases_and_seeds_retained': True, 'target_pixels_used': False, 'target_latents_computed': False}
        need(len(records) == 480, 'Grid accounting incomplete')
        save(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'result': ref(out / 'result.json')}), flush=True)
    except Exception as exc:
        save(out / 'fatal_error.json', {'error': str(exc), 'traceback': traceback.format_exc(), 'result_is_incomplete': True})
        raise
    finally:
        if runtime:
            runtime.close()


if __name__ == '__main__':
    main()
