"""Dev-only paired epsilon residual sampling; --execute is explicitly required.

Private model-loading path copied from the SHA-bound V6 kernel. No training,
downloads, scheduler.step, target pixels, test images, or post-hoc RGB blending.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
V5 = HERE.parent / 'quality_diagnosis_v5_20260915'
V5RUN = V5 / 'runs/source_init_202609151525'
V6 = HERE.parent / 'region_preserving_v6_20260915'
V7 = HERE.parent / 'smooth_regions_v7_20260915'
MASK_MANIFEST = V7 / 'masks/first8/manifest.json'
GAINS = (('paired_g025', .25), ('paired_g050', .5), ('paired_g100', 1.))
METHODS = ('baseline', 'bidirectional')
NOMINAL_STEPS, STRENGTH, SEED = 20, .75, 101
CFG, AU_SCALE, IP_SCALE = 7.5, 1., .9
V5_SHA = '6296388a3ffeafd6cc413589f0f498966e113c8c0e183e36e30017fce96edd89'
V6_SHA = '85236dfc8803fb66d8f2440a240475a10c59f189ab3df059a44389f13db1fcec'
V7_MASK_SHA = 'ca173d310425c6c97ba4350a21cdd5b3d4f08a961fdb87b9c134e1b3029ef824'
MASK_MANIFEST_SHA = 'f314521f0d98ecfd0c393d6a3b36464edd2df6fd24a01ee46d524d215b3e086c'


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load_pinned_module(name, path, digest):
    need(_sha(path) == digest, 'Frozen helper changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v5 = load_pinned_module('v8_frozen_v5_helpers', V5 / 'run_diagnostic.py', V5_SHA)
sha, save, ref, read, tensor_sha = v5.sha, v5.save, v5.ref, v5.read, v5.tensor_sha


def finite(torch, *values):
    need(all(torch.isfinite(value).all().item() for value in values), 'Nonfinite paired sampling tensor')


def rms(value):
    # CPU float64 prevents squared finite float32 values overflowing in logs.
    return float(value.detach().cpu().double().square().mean().sqrt())


def residual_step(d, delta, mask, rho, next_rho, gain, torch):
    """Single soft-mask multiplication on the new increment; no soft projection."""
    need(math.isfinite(rho) and math.isfinite(next_rho) and rho > next_rho >= 0,
         'Noise levels must decrease to an explicit nonnegative endpoint')
    need(math.isfinite(gain) and gain >= 0, 'Invalid residual gain')
    finite(torch, d, delta, mask)
    need(d.shape == delta.shape and mask.shape == (d.shape[0], 1, d.shape[2], d.shape[3]),
         'Residual/mask shape mismatch')
    need(mask.min().item() >= 0 and mask.max().item() <= 1, 'Mask outside [0,1]')
    value = d + (next_rho - rho) * gain * mask * delta
    value = torch.where(mask == 0, torch.zeros_like(value), value)
    finite(torch, value)
    return value


def source_target_inputs(source, noise, d, level, torch):
    alpha = source.new_tensor(level['alpha'])
    sigma = source.new_tensor(level['sigma'])
    source_noisy = alpha * source + sigma * noise
    target_noisy = source_noisy + alpha * d
    finite(torch, source_noisy, target_noisy)
    return source_noisy, target_noisy


def time_levels(scheduler):
    """DEIS's actual20/.75 query points; clean final endpoint, no DEIS step call."""
    scheduler.set_timesteps(NOMINAL_STEPS, device='cpu')
    need(scheduler.order == 1, 'Unexpected pipeline indexing order')
    need(scheduler.config.prediction_type == 'epsilon' and not scheduler.config.thresholding,
         'Paired field requires unthresholded epsilon prediction')
    need(not scheduler.config.use_karras_sigmas and scheduler.config.timestep_spacing == 'linspace',
         'Original DEIS grid required')
    start = NOMINAL_STEPS - int(NOMINAL_STEPS * STRENGTH)
    points = []
    for i in range(start, len(scheduler.timesteps)):
        rho = scheduler.sigmas[i]
        alpha, sigma = scheduler._sigma_to_alpha_sigma_t(rho)
        points.append({'t': int(scheduler.timesteps[i]), 'schedule_index': i,
                       'alpha': float(alpha), 'sigma': float(sigma), 'rho': float(rho)})
    need(len(points) == 15 and points[-1]['t'] == 50, 'Original15 DEIS query points required')
    for i, point in enumerate(points):
        nxt = points[i + 1] if i + 1 < len(points) else {'t': None, 'alpha': 1., 'sigma': 0., 'rho': 0.}
        point.update(next_t=nxt['t'], next_alpha=nxt['alpha'], next_sigma=nxt['sigma'], next_rho=nxt['rho'])
        need(point['rho'] > point['next_rho'] >= 0, 'Nonmonotonic actual noise grid')
    return points


def cfg_prediction(unet, latent, timestep, packed_text, cross, torch):
    """Two complete CFG samples per branch, matching installed Img2Img order."""
    finite(torch, latent, packed_text, cross['au_embedding'], cross['image_embedding'])
    need(packed_text.shape[0] == 2 and cross['au_embedding'].shape[0] == 2 and
         cross['image_embedding'].shape[0] == 2, 'Expected [unconditional,conditional] batch2')
    prediction = unet(torch.cat((latent, latent)), timestep,
                      encoder_hidden_states=packed_text, timestep_cond=None,
                      cross_attention_kwargs=cross, added_cond_kwargs=None, return_dict=False)[0]
    finite(torch, prediction)
    uncond, cond = prediction.chunk(2)
    result = uncond + CFG * (cond - uncond)
    finite(torch, result)
    return result


def source_cache_key(row, timestep):
    """Cache is local to one frozen runtime; full source inputs/scales are bound."""
    import hashlib
    payload = {'source_latent': row['source_latent_sha256'], 'noise': row['noise_sha256'],
               'source_cross': row['source_condition_hashes'], 'source_text': row['source_text_tokens_sha256'],
               'timestep': int(timestep), 'cfg': CFG, 'au_scale': AU_SCALE, 'ip_scale': IP_SCALE}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verified_inputs():
    need(sha(V6 / 'run_region_diagnostic.py') == V6_SHA, 'V6 model-loading reference changed')
    pins = {V5RUN / 'cases.json': '7daad59d5540d7b49243cb03da38f7aaed5d2c4294c8cb46ac7ce58b46d43145',
            V5RUN / 'result.json': '4e92651f47abadfefdf0be2a91600544c1142cbbd99ebab411b2bea0b22177ce',
            V5RUN / 'vae_check.json': 'a25bc2564362b749dd5080d5822806a7387d6648050896afcd398c52e5c04d26',
            MASK_MANIFEST: MASK_MANIFEST_SHA, V7 / 'region_mask.py': V7_MASK_SHA}
    for path, digest in pins.items():
        need(sha(path) == digest, 'Frozen input changed: ' + str(path))
    expected = v5.preflight()  # pinned train/dev cache and existing dev-only references
    cases = read(V5RUN / 'cases.json')['cases']
    v5_records = {(row['case_id'], row['method']): row for row in read(V5RUN / 'result.json')['records']}
    need(len(cases) == len(expected) == 8 and cases[0]['group'] == 'no_edit', 'Exact8 and first noedit required')
    for case, original in zip(cases, expected):
        need(all(case[key] == value for key, value in original.items()), 'V5 case binding changed')
        need(case['split'] == 'dev' and not any(case['source_aus_0_5']), 'Known source-zero dev only')
        need(sha(case['vae_reconstruction']['path']) == case['vae_reconstruction']['sha256'], 'VAE reference changed')
        for method in ('baseline_img2img_s055', 'bidirectional_img2img_s055',
                       'baseline_img2img_s075', 'bidirectional_img2img_s075'):
            row = v5_records[case['case_id'], method]
            need(row['status'] == 'complete' and sha(row['output']) == row['sha256'], 'V5 img2img reference changed')
            case[method] = {'path': row['output'], 'sha256': row['sha256']}
    import numpy as np
    masks_module = load_pinned_module('v8_frozen_v7_mask', V7 / 'region_mask.py', V7_MASK_SHA)
    doc = read(MASK_MANIFEST)
    need(doc['status'] == 'complete' and doc['source_only'] and doc['source_unchanged'] and
         not doc['test_used'] and not doc['target_geometry_used'], 'Invalid source-only mask declaration')
    for key in ('cases', 'source_measurements'):
        item = doc['inputs'][key]
        need(sha(item['path']) == item['sha256'], 'Mask input binding changed')
    source_records = {r['case_id']: r for r in read(doc['inputs']['source_measurements']['path'])['records']
                      if r['method'] == 'source_real'}
    entries = {r['case_id']: r for r in doc['cases']}
    need(set(entries) == {c['case_id'] for c in cases}, 'Mask case set mismatch')
    for case in cases:
        entry = entries[case['case_id']]
        item = entry['mask64']
        need(sha(item['path']) == item['sha256'] and entry['source_sha256'] == case['source_real']['sha256'],
             'Mask/source SHA mismatch')
        mask = np.load(item['path'], allow_pickle=False)
        reproduced, _ = masks_module.build_mask(case, source_records[case['case_id']])
        need(np.array_equal(mask, reproduced), 'Mask differs from exact frozen V7 source68 rule')
        need((not np.any(mask)) == (case['group'] == 'no_edit'), 'Noedit mask mismatch')
        need(sha(entry['receipt_path']) == entry['receipt_sha256'], 'Mask receipt changed')
        case['region_mask'] = {**item, 'source_sha256': entry['source_sha256'], 'area': entry['mask_area'],
                               'receipt': {'path': entry['receipt_path'], 'sha256': entry['receipt_sha256']}}
    info = read(V5RUN / 'vae_check.json')
    need(sha(info['latents_file']['path']) == info['latents_file']['sha256'], 'Frozen source latents changed')
    return cases, info


def load_runtime(torch, info):
    """Private copy of V6 pinned loader; invoked only by --execute."""
    from diffusers import AutoencoderKL, DEISMultistepScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from modeling.frozen_backbone import FrozenConditionedDenoiser, RV
    from modeling.safe_token_residual_adapter import RelativeRMSConfig
    from train_centered_adapter import (load_checkpoint, adapter_from_checkpoint, resolve_source_anchor,
                                       verify_source_anchor, source_and_asset_bindings)
    payload = load_checkpoint(v5.CKPT)
    need(payload['global_step'] == 200 and payload['variant'] == 'bidirectional', 'Frozen early200 bidirectional required')
    for path, digest in payload['bindings']['sources'].items():
        need(sha(EXP / path) == digest, 'Trained source changed')
    assets, _ = source_and_asset_bindings({})
    need(assets['public_assets'] == payload['bindings']['public_assets'], 'Public assets changed')
    data = torch.load(v5.CACHE / 'features.pt', map_location='cpu', weights_only=True)['samples']
    need(all(row['split'] in ('train', 'dev') for row in data), 'No test data permitted')
    samples = {row['case_id']: row for row in data}
    anchor, binding = resolve_source_anchor([row for row in data if row['split'] == 'train'],
                                             [row for row in data if row['split'] == 'dev'])
    verify_source_anchor(anchor, binding, payload['bindings']['source_anchor'])
    source_tokens = samples[binding['anchor_train_case_id']]['text_tokens']
    need(source_tokens.shape == (77, 768) and source_tokens.dtype == torch.float32, 'Expected cached neutral77x768')
    for row in data:
        if not torch.count_nonzero(row['target_aus']).item():
            need(torch.equal(row['text_tokens'], source_tokens), 'Cached neutral text tokens must match exactly')
    adapter = adapter_from_checkpoint(payload, 'mps').eval().requires_grad_(False)
    e = payload['effective_config']
    trust = RelativeRMSConfig(enabled=e['trust_region'], rho_au=e['rho_au'], rho_ip=e['rho_ip'],
                              reference_rms_floor=e['reference_rms_floor'])
    backbone = FrozenConditionedDenoiser('mps', checkpointing=False)
    backbone.eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(str(RV / 'vae'), local_files_only=True,
                                       torch_dtype=torch.float32).eval().requires_grad_(False).to('mps')
    text = CLIPTextModel.from_pretrained(str(RV / 'text_encoder'), local_files_only=True,
                                        torch_dtype=torch.float32).eval().requires_grad_(False).to('mps')
    tokenizer = CLIPTokenizer.from_pretrained(str(RV / 'tokenizer'), local_files_only=True)
    negative_ids = tokenizer([''], padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids
    negative_tokens = text(negative_ids.to('mps'))[0]
    scheduler = DEISMultistepScheduler.from_pretrained(str(RV / 'scheduler'), local_files_only=True)
    levels = time_levels(scheduler)
    latents = torch.load(info['latents_file']['path'], map_location='cpu', weights_only=True)
    finite(torch, source_tokens, negative_tokens, *latents.values())
    return {'samples': samples, 'anchor': anchor.to('mps'), 'binding': binding,
            'source_tokens': source_tokens[None].to('mps'), 'negative_tokens': negative_tokens,
            'adapter': adapter, 'trust': trust, 'backbone': backbone, 'vae': vae, 'latents': latents,
            'scheduler_config': dict(scheduler.config), 'levels': levels, 'assets': assets['public_assets']}


def pack_case(runtime, case, torch):
    from train_centered_adapter import centered_batch_conditions
    sample = runtime['samples'][case['case_id']]
    need(sample['split'] == 'dev' and sample['subject'] == case['subject'] and sample['prompt'] == case['prompt'],
         'Cached dev identity/prompt mismatch')
    for key, field in (('source_aus', 'source_aus_0_5'), ('target_aus', 'target_aus_0_5')):
        need(sample[key].tolist() == case[field], 'Original absolute AU labels changed')
    backbone = runtime['backbone']
    target_aus = sample['target_aus'][None].to('mps')
    source_aus = sample['source_aus'][None].to('mps')
    identity = sample['base_faceid'][None].to('mps')
    base_au, base_ip = backbone.baseline_conditions(target_aus, identity)
    source_au, source_ip = backbone.baseline_conditions(source_aus, identity)
    null_au, null_ip = backbone.baseline_conditions(torch.zeros_like(target_aus), torch.zeros_like(identity))
    batch = {key: sample[key][None].to('mps') for key in ('image_patches', 'text_pooled', 'target_aus', 'source_aus')}
    _, _, adapted = centered_batch_conditions(runtime['adapter'], batch, runtime['anchor'], base_au, base_ip, runtime['trust'])

    def cross(au, ip):
        return {'au_embedding': torch.cat((null_au, au)), 'image_embedding': torch.cat((null_ip, ip)),
                'au_scale': AU_SCALE, 'ip_scale': IP_SCALE}

    source_cross = cross(source_au, source_ip)
    targets = {'baseline': cross(base_au, base_ip),
               'bidirectional': cross(adapted['au_embedding'], adapted['image_embedding'])}
    target_tokens = sample['text_tokens'][None].to('mps')
    need(target_tokens.shape == runtime['source_tokens'].shape == (1, 77, 768), 'Cached text token shape changed')
    source_text = torch.cat((runtime['negative_tokens'], runtime['source_tokens']))
    target_text = torch.cat((runtime['negative_tokens'], target_tokens))
    finite(torch, source_text, target_text)
    target_hashes = {}
    for method, target in targets.items():
        hashes = {key: tensor_sha(value) for key, value in target.items() if isinstance(value, torch.Tensor)}
        need(hashes == case[method + '_text2img_condition_hashes'], 'Target AU/FaceID condition hashes changed')
        need(torch.equal(target['au_embedding'][1:, :, :12], base_au[:, :, :12]), 'Explicit absolute AU prefix changed')
        target_hashes[method] = hashes
        if case['group'] == 'no_edit':
            need(torch.equal(source_text, target_text) and all(torch.equal(source_cross[key], target[key])
                 for key in ('au_embedding', 'image_embedding')), 'Noedit complete conditions must be identical')
    return source_cross, targets, source_text, target_text, target_hashes


def execute(out):
    need(out.parent == HERE / 'runs' and not out.exists(), 'Exclusive new V8 run path required')
    cases, info = verified_inputs()
    out.mkdir(parents=True, exist_ok=False)
    configuration = {'kind': 'v8_paired_residual_diagnostic', 'test_data_used': False, 'training_performed': False,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'script': ref(__file__),
        'v5_helpers': ref(V5 / 'run_diagnostic.py'), 'v6_loader_reference': ref(V6 / 'run_region_diagnostic.py'),
        'v7_mask_code': ref(V7 / 'region_mask.py'), 'mask_manifest': ref(MASK_MANIFEST),
        'source_latents': info['latents_file'], 'v5_cases': ref(V5RUN / 'cases.json'),
        'gains': [{'name': name, 'gain': gain} for name, gain in GAINS], 'methods': list(METHODS),
        'seed': SEED, 'nominal_steps': NOMINAL_STEPS, 'strength': STRENGTH, 'actual_steps': 15,
        'guidance_scale': CFG, 'au_scale': AU_SCALE, 'ip_scale': IP_SCALE,
        'solver': 'explicit_first_order_rho_on_DEIS_query_grid', 'scheduler_step_calls': 0,
        'update': 'd += (rho_next-rho)*gain*mask*(CFG_target-CFG_source)', 'final_rho': 0.,
        'source_condition': 'frozen neutral text tokens + recorded source AU0 + same base FaceID',
        'target_condition': 'unchanged cached text_tokens + SHA-identical original target AU/IP conditions',
        'mask_application': 'once on residual increment; zero support only is snapped to0; no soft projection',
        'source_cache': 'run-local frozen-UNet CFG epsilon by source latent/noise/source condition/text/timestep/scales; target never cached',
        'post_generation_pixel_blend': False, 'target_pixels_supplied': False,
        'noedit_semantics': 'algebraic source VAE reconstruction, not a learned preservation claim'}
    save(out / 'configuration.json', configuration)
    save(out / 'cases.json', {'kind': 'v8_development_paired_cases', 'test_data_used': False, 'case_count': 8, 'cases': cases})
    records, fatal = [], None
    try:
        torch = v5.environment()
        import numpy as np
        from PIL import Image
        with torch.no_grad():
            runtime = load_runtime(torch, info)
            save(out / 'runtime.json', {'scheduler': 'DEISMultistepScheduler_grid_only',
                'scheduler_config': runtime['scheduler_config'], 'time_levels': runtime['levels'],
                'checkpoint': ref(v5.CKPT), 'public_assets': runtime['assets'], 'source_anchor': runtime['binding'],
                'source_text_tokens_sha256': tensor_sha(runtime['source_tokens']),
                'negative_text_tokens_sha256': tensor_sha(runtime['negative_tokens']),
                'torch': torch.__version__, 'device': 'mps', 'dtype': 'float32', 'scheduler_step_calls': 0})
            source_cache = {}
            with (out / 'generation_records.jsonl').open('x', buffering=1) as log:
                for case_index, case in enumerate(cases):
                    source_cross, target_crosses, source_text, target_text, target_hashes = pack_case(runtime, case, torch)
                    source = runtime['latents'][case['source_real']['sha256']].to('mps')
                    mask = torch.from_numpy(np.load(case['region_mask']['path'], allow_pickle=False))[None, None].to('mps')
                    need(source.shape == (1, 4, 64, 64) and mask.shape == (1, 1, 64, 64), 'Original latent shapes required')
                    for method in METHODS:
                        for config_name, gain in GAINS:
                            name = method + '_' + config_name
                            image_path = out / 'images' / case['case_id'] / (name + '.png')
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            row = {key: case[key] for key in ('case_id', 'subject', 'split', 'group', 'source_aus_0_5', 'target_aus_0_5', 'prompt')}
                            row.update(method=name, config=config_name, gain=gain, seed=SEED, strength=STRENGTH,
                                nominal_steps=NOMINAL_STEPS, guidance_scale=CFG, au_scale=AU_SCALE, ip_scale=IP_SCALE,
                                output=str(image_path), status='pending', condition_hashes=target_hashes[method],
                                source_condition_hashes={key: tensor_sha(value) for key, value in source_cross.items() if isinstance(value, torch.Tensor)},
                                source_text_tokens_sha256=tensor_sha(source_text), target_text_tokens_sha256=tensor_sha(target_text),
                                source_latent_sha256=tensor_sha(source), region_mask=case['region_mask'],
                                source_unet_calls=0, source_cache_reuses=0, source_prediction_requests=0,
                                target_unet_calls=0, vae_decode_calls=0, scheduler_step_calls=0,
                                zeroedit_required=case['group'] == 'no_edit', finite=False, step_trace=[])
                            start = time.monotonic()
                            try:
                                noise = torch.randn(source.shape, generator=torch.Generator(device='cpu').manual_seed(SEED), dtype=torch.float32).to('mps')
                                row['noise_sha256'] = tensor_sha(noise)
                                d = torch.zeros_like(source)
                                for level in runtime['levels']:
                                    source_noisy, target_noisy = source_target_inputs(source, noise, d, level, torch)
                                    t = torch.tensor(level['t'], device='mps', dtype=torch.long)
                                    row['source_prediction_requests'] += 1
                                    cache_key = source_cache_key(row, level['t'])
                                    reused = cache_key in source_cache
                                    if reused:
                                        es = source_cache[cache_key]
                                        row['source_cache_reuses'] += 1
                                    else:
                                        row['source_unet_calls'] += 1
                                        es = cfg_prediction(runtime['backbone'].unet, source_noisy, t, source_text, source_cross, torch)
                                        source_cache[cache_key] = es.detach().clone()
                                    row['target_unet_calls'] += 1
                                    et = cfg_prediction(runtime['backbone'].unet, target_noisy, t, target_text, target_crosses[method], torch)
                                    delta = et - es
                                    finite(torch, delta)
                                    zeroedit = torch.equal(es, et)
                                    if row['zeroedit_required']:
                                        need(zeroedit and torch.count_nonzero(d).item() == 0, 'Noedit epsilon/d identity failed before mask')
                                    d = residual_step(d, delta, mask, level['rho'], level['next_rho'], gain, torch)
                                    row['step_trace'].append({**level, 'source_eps_rms': rms(es), 'target_eps_rms': rms(et),
                                        'delta_rms': rms(delta), 'masked_delta_rms': rms(mask * delta), 'residual_rms': rms(d),
                                        'source_prediction_reused': reused, 'source_cache_key_sha256': cache_key,
                                        'epsilon_exact_equal': zeroedit, 'd_exact_zero': torch.count_nonzero(d).item() == 0, 'finite': True})
                                output_latent = source + d
                                finite(torch, output_latent)
                                if row['zeroedit_required']:
                                    need(torch.equal(output_latent, source), 'Noedit final latent differs from clean source')
                                row['final_latent_sha256'] = tensor_sha(output_latent)
                                row['vae_decode_calls'] += 1
                                decoded = runtime['vae'].decode(output_latent / runtime['vae'].config.scaling_factor).sample
                                finite(torch, decoded)
                                pixels = ((decoded / 2 + .5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
                                Image.fromarray(pixels).save(image_path)
                                output_sha = sha(image_path)
                                if row['zeroedit_required']:
                                    need(output_sha == case['vae_reconstruction']['sha256'], 'Noedit PNG differs from frozen VAE reconstruction')
                                need(len(row['step_trace']) == 15 and row['source_prediction_requests'] == row['target_unet_calls'] == 15 and
                                     row['source_unet_calls'] + row['source_cache_reuses'] == 15 and row['vae_decode_calls'] == 1,
                                     'Bounded call-count contract failed')
                                row.update(status='complete', sha256=output_sha, finite=True,
                                           noedit_gate='PASS' if row['zeroedit_required'] else 'not_applicable')
                            except Exception as error:
                                row.update(status='failed', sha256=sha(image_path) if image_path.exists() else None,
                                           error=str(error), traceback=traceback.format_exc())
                            row.update(seconds=time.monotonic() - start, actual_steps=len(row['step_trace']),
                                       source_calls=row['source_unet_calls'], target_calls=row['target_unet_calls'],
                                       actual_timesteps=[step['t'] for step in row['step_trace']])
                            records.append(row)
                            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                            print(json.dumps({key: row[key] for key in ('case_id', 'method', 'status', 'seconds')}, ensure_ascii=False), flush=True)
                    if case_index == 0:
                        passed = len(records) == 6 and all(row['status'] == 'complete' and row['noedit_gate'] == 'PASS' for row in records)
                        save(out / 'first_case_engineering_gate.json', {'status': 'PASS' if passed else 'FAIL',
                            'generated': len(records), 'requires_exact_paired_epsilon_before_mask': True,
                            'requires_clean_source_latent_and_vae_png': True, 'quality_acceptance': False})
                        need(passed, 'First noedit case failed; stop before additional dev cases')
    except Exception as error:
        fatal = {'error': str(error), 'traceback': traceback.format_exc()}
        save(out / 'run_failure.json', fatal)
    finally:
        completed = sum(row['status'] == 'complete' for row in records)
        save(out / 'result.json', {'kind': 'v8_paired_residual_diagnostic', 'test_data_used': False,
            'status': 'stopped_with_failure' if fatal else 'complete' if completed == 48 else 'completed_with_failures',
            'planned': 48, 'attempted': len(records), 'completed': completed, 'records': records,
            'actual_source_unet_calls': sum(row['source_unet_calls'] for row in records),
            'actual_target_unet_calls': sum(row['target_unet_calls'] for row in records),
            'source_cache_reuses': sum(row['source_cache_reuses'] for row in records),
            'actual_vae_decode_calls': sum(row['vae_decode_calls'] for row in records),
            'cases': ref(out / 'cases.json'), 'configuration': ref(out / 'configuration.json'), 'fatal': fatal})
    need(fatal is None, 'V8 stopped; see exclusive run_failure.json and result.json')
    print(json.dumps({'status': 'v8_complete', 'run': str(out), 'completed': completed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        cases, _ = verified_inputs()
        print(json.dumps({'status': 'verified_inputs_not_executed', 'cases': len(cases),
                          'planned': 48, 'gains': [gain for _, gain in GAINS], 'models_loaded': False}))
        return
    need(args.run is not None, '--execute requires an explicit new --run path')
    execute(args.run.resolve())


if __name__ == '__main__':
    main()
