"""Bounded development-only source initialization diagnosis; no training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
PY = Path.home() / 'Desktop/Cursor/geme-main/sota_baselines_2026/MagicFace/.venv_magicface/bin/python'
CACHE = EXP / 'training_cache/train2k_batched_202609141853'
DEV = EXP / 'development_evaluation/dev_coverage_audit/dev19_manifest_2k.json'
CKPT = EXP / 'runs/v4_slot_2k_bidir_step200_202609141920/checkpoint_00000200.pt'
OLD = EXP / 'development_previews/v4_slot_2k_step200_dev19_202609141941'
SUBJECTS = ['SN004', 'SN012', 'SN017', 'SN018']
STRENGTHS = [.35, .55, .75]
PINS = {CACHE / 'manifest.json': '4a378e75a478cf241ef08b3de3727bfa3e6da25e505b85ca1bc9cee74c57b413',
        CACHE / 'features.pt': '7ede6afdfc6948ba4da845309033edc700466d80fc91bb5921410b780e8133e0',
        DEV: '90a0307b053d369e49286ea63ad3f6e731c8213b3786fbd4591706c1b7f69a28',
        CKPT: '7fcc25b704fca8c0a4e46a171096df8fea940b40a246d4939193e64f533b4e30'}

def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def save(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def ref(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p)}

def preflight():
    for path, expected in PINS.items():
        need(sha(path) == expected, 'Pinned input changed: ' + str(path))
    meta, dev = read(CACHE / 'manifest.json'), read(DEV)
    need(meta['test_data_used'] is False and dev['test_data_used'] is False, 'Development only')
    cases = [c for c in dev['cases'] if c['group'] in ('no_edit', 'lowest_available')]
    need(len(cases) == 8 and sorted({c['subject'] for c in cases}) == SUBJECTS, 'Exact 4x2 development cases required')
    old = [json.loads(line) for line in (OLD / 'generation_records.jsonl').read_text().splitlines()]
    pairs = {p['case_id']: p for p in meta['pairs']}
    output = []
    for c in cases:
        p = pairs[c['case_id']]
        need(c['split'] == p['split'] == 'dev' and p['subject'] == c['subject'], 'Development role mismatch')
        row = {'case_id': c['case_id'], 'subject': c['subject'], 'split': 'dev', 'group': c['group'],
               'source_aus_0_5': c['source_au_vector_0_5'], 'target_aus_0_5': c['target_au_vector_0_5'], 'prompt': p['prompt']}
        for role in ('source', 'target'):
            a = p['aligned_' + role]
            need(a['split'] == a['input']['split'] == 'dev' and a['subject'] == c['subject'], 'Alignment mapping mismatch')
            need(sha(a['raw_path']) == a['raw_sha256'] and sha(a['aligned_path']) == a['aligned_sha256'], 'Source/target SHA mismatch')
            need(a['input']['au_vector_0_5'] == row[role + '_aus_0_5'], 'Raw label binding mismatch')
            row[role + '_real'] = {'path': a['aligned_path'], 'sha256': a['aligned_sha256']}
            row[role + '_alignment'] = {k: a[k] for k in ('raw_path', 'raw_sha256', 'face_kps', 'affine_original_to_512')}
        for method, name in [('baseline', 'baseline_text2img'), ('source_centered_bidirectional_step200_7fcc25b7', 'bidirectional_text2img')]:
            rows = [r for r in old if r['case_id'] == c['case_id'] and r['method'] == method and r['seed'] == 101]
            need(len(rows) == 1 and rows[0]['status'] == 'complete', 'Missing exact original development reference')
            r = rows[0]
            need(sha(r['output']) == r['sha256'], 'Original generation changed')
            row[name] = {'path': r['output'], 'sha256': r['sha256']}
            row[name + '_condition_hashes'] = r['condition_hashes']
        output.append(row)
    return output

def environment():
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      NO_ALBUMENTATIONS_UPDATE='1', HF_HOME=str(EXP / 'cache/huggingface'),
                      TORCH_HOME=str(EXP / 'cache/torch'), MPLCONFIGDIR=str(EXP / 'cache/matplotlib'))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(EXP))
    import torch
    need(torch.__version__.split('+')[0] == '2.1.1' and torch.backends.mps.is_available(), 'Original torch/MPS required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return torch

def prepare(out):
    cases = preflight()
    need(not out.exists() and out.parent == HERE / 'runs', 'New v5 runs directory required')
    out.mkdir(parents=True)
    save(out / 'configuration.json', {'kind': 'v5_source_initialization_diagnostic', 'test_data_used': False,
        'training_performed': False, 'script': ref(__file__), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_bindings': [ref(p) for p in PINS], 'case_ids': [c['case_id'] for c in cases],
        'strengths': STRENGTHS, 'seed': 101, 'nominal_steps': 20, 'guidance_scale': 7.5, 'au_scale': 1., 'ip_scale': .9,
        'initialization': 'source VAE posterior mean times scaling_factor, no target latent; same CPU seeded noise',
        'comparison_note': 'Changing strength changes initial noise and denoising depth together; diagnostic only'})
    torch = environment()
    from diffusers import AutoencoderKL
    from modeling.frozen_backbone import RV
    from PIL import Image
    import numpy as np
    vae = AutoencoderKL.from_pretrained(str(RV / 'vae'), local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False).to('mps')
    unique = {}
    (out / 'vae_reconstructions').mkdir()
    with torch.no_grad():
        for c in cases:
            s = c['source_real']
            if s['sha256'] not in unique:
                image = Image.open(s['path']).convert('RGB')
                need(image.size == (512, 512), 'Expected original canonical512 source')
                x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float().to('mps') / 127.5 - 1
                z = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
                decoded = vae.decode(z / vae.config.scaling_factor).sample
                need(torch.isfinite(z).all().item() and torch.isfinite(decoded).all().item(), 'Nonfinite VAE source latent/reconstruction')
                pixel = ((decoded / 2 + .5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
                path = out / 'vae_reconstructions' / (c['subject'] + '.png')
                Image.fromarray(pixel).save(path)
                unique[s['sha256']] = {'latent': z.cpu(), 'reconstruction': ref(path),
                    'source': s, 'pixel_MAE_0_255': float(np.abs(pixel.astype(float) - np.asarray(image).astype(float)).mean()),
                    'latent_sha256': tensor_sha(z)}
            c['vae_reconstruction'] = unique[s['sha256']]['reconstruction']
    torch.save({key: value['latent'] for key, value in unique.items()}, out / 'source_latents.pt')
    save(out / 'vae_check.json', {'test_data_used': False, 'vae_source': str(RV / 'vae'),
         'sources': [{k: v for k, v in val.items() if k != 'latent'} for val in unique.values()],
         'latents_file': ref(out / 'source_latents.pt'), 'encoder_calls': 4, 'target_encoder_calls': 0})
    save(out / 'cases.json', {'kind': 'v5_development_diagnostic_cases', 'test_data_used': False, 'case_count': 8, 'cases': cases})
    print(json.dumps({'status': 'prepared_4_source_reconstructions', 'run': str(out), 'cases': 8}), flush=True)

def generate(out):
    conf = read(out / 'configuration.json')
    need(conf['script']['sha256'] == sha(__file__), 'Generator changed after prepare')
    expected = preflight()
    cases = read(out / 'cases.json')['cases']
    for c, e in zip(cases, expected):
        need({k: c[k] for k in e} == e, 'Prepared case metadata changed')
    need(len(cases) == 8 and not (out / 'generation_records.jsonl').exists(), 'No overwrite/retry of generation')
    check = read(out / 'vae_check.json')
    need(sha(check['latents_file']['path']) == check['latents_file']['sha256'], 'Source latent file changed')
    torch = environment()
    from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL, DEISMultistepScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from modeling.frozen_backbone import FrozenConditionedDenoiser, RV
    from modeling.safe_token_residual_adapter import RelativeRMSConfig
    from train_centered_adapter import (load_checkpoint, adapter_from_checkpoint, centered_batch_conditions,
                                       resolve_source_anchor, verify_source_anchor, source_and_asset_bindings)
    payload = load_checkpoint(CKPT)
    need(payload['global_step'] == 200 and payload['variant'] == 'bidirectional', 'Early200 bidirectional only')
    for path, expected_sha in payload['bindings']['sources'].items():
        need(sha(EXP / path) == expected_sha, 'Original model source changed')
    current, _ = source_and_asset_bindings({})
    need(current['public_assets'] == payload['bindings']['public_assets'], 'Public assets changed')
    data = torch.load(CACHE / 'features.pt', map_location='cpu', weights_only=True)['samples']
    need(all(d['split'] in ('train', 'dev') for d in data), 'Test data forbidden')
    samples = {s['case_id']: s for s in data}
    anchor, binding = resolve_source_anchor([d for d in data if d['split'] == 'train'], [d for d in data if d['split'] == 'dev'])
    verify_source_anchor(anchor, binding, payload['bindings']['source_anchor'])
    model = adapter_from_checkpoint(payload, 'mps').eval().requires_grad_(False)
    e = payload['effective_config']
    trust = RelativeRMSConfig(enabled=e['trust_region'], rho_au=e['rho_au'], rho_ip=e['rho_ip'], reference_rms_floor=e['reference_rms_floor'])
    backbone = FrozenConditionedDenoiser('mps', checkpointing=False)
    vae = AutoencoderKL.from_pretrained(str(RV / 'vae'), local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False)
    text = CLIPTextModel.from_pretrained(str(RV / 'text_encoder'), local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False)
    pipe = StableDiffusionImg2ImgPipeline(vae=vae, text_encoder=text,
        tokenizer=CLIPTokenizer.from_pretrained(str(RV / 'tokenizer'), local_files_only=True),
        unet=backbone.unet, scheduler=DEISMultistepScheduler.from_pretrained(str(RV / 'scheduler'), local_files_only=True),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False).to('mps')
    pipe.set_progress_bar_config(disable=True)
    latents = torch.load(out / 'source_latents.pt', map_location='cpu', weights_only=True)
    need(all(torch.isfinite(z).all().item() for z in latents.values()), 'Nonfinite source latent')
    def finite_decode_hook(module, args, result):
        tensor = result if isinstance(result, torch.Tensor) else result[0] if isinstance(result, tuple) else result.sample
        need(torch.isfinite(tensor).all().item(), 'Nonfinite raw VAE decode')
    decode_guard = vae.decoder.register_forward_hook(finite_decode_hook)
    save(out / 'runtime.json', {'scheduler': type(pipe.scheduler).__name__, 'scheduler_config': dict(pipe.scheduler.config),
         'pipe_class': type(pipe).__name__, 'torch': torch.__version__, 'device': 'mps', 'dtype': 'float32',
         'checkpoint': ref(CKPT), 'public_assets': current['public_assets'], 'source_anchor': binding,
         'target_pixels_passed_to_generator': False, 'source_latents_precomputed': True})
    records = []
    with torch.no_grad(), (out / 'generation_records.jsonl').open('x', buffering=1) as log:
        for c in cases:
            s = samples[c['case_id']]
            need(s['split'] == 'dev' and s['subject'] == c['subject'] and s['prompt'] == c['prompt'], 'Cache source identity/text mismatch')
            for key, field in [('target_aus', 'target_aus_0_5'), ('source_aus', 'source_aus_0_5')]:
                need(s[key].tolist() == c[field], 'Cached AU changed')
            aus, identity = s['target_aus'][None].to('mps'), s['base_faceid'][None].to('mps')
            base_au, base_ip = backbone.baseline_conditions(aus, identity)
            null_au, null_ip = backbone.baseline_conditions(torch.zeros_like(aus), torch.zeros_like(identity))
            batch = {k: s[k][None].to('mps') for k in ('image_patches', 'text_pooled', 'target_aus', 'source_aus')}
            _, _, adapted = centered_batch_conditions(model, batch, anchor.to('mps'), base_au, base_ip, trust)
            values = {'baseline': {'au_embedding': base_au, 'image_embedding': base_ip}, 'bidirectional': adapted}
            for method, v in values.items():
                need(torch.equal(v['au_embedding'][..., :12], base_au[..., :12]), 'AU prefix changed')
                cross = {'au_embedding': torch.cat((null_au, v['au_embedding'])), 'image_embedding': torch.cat((null_ip, v['image_embedding'])), 'au_scale': 1., 'ip_scale': .9}
                hashes = {k: tensor_sha(t) for k, t in cross.items() if isinstance(t, torch.Tensor)}
                need(hashes == c[method + '_text2img_condition_hashes'], 'New conditions differ from original text2img; not a pure initialization diagnosis')
                for strength in STRENGTHS:
                    name = method + '_img2img_s' + f'{round(strength * 100):03d}'
                    path = out / 'images' / c['case_id'] / (name + '.png')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    row = {k: c[k] for k in ('case_id', 'subject', 'split', 'group', 'source_aus_0_5', 'target_aus_0_5', 'prompt')}
                    row.update(method=name, seed=101, strength=strength, nominal_steps=20, condition_hashes=hashes,
                               source_latent_sha256=tensor_sha(latents[c['source_real']['sha256']]), output=str(path), status='pending')
                    steps = []
                    def callback(step, timestep, tensor):
                        steps.append(int(timestep))
                        need(torch.isfinite(tensor).all().item(), 'Nonfinite denoising latent')
                    start = time.monotonic()
                    try:
                        image = pipe(prompt=c['prompt'], image=latents[c['source_real']['sha256']].to('mps'),
                            strength=strength, num_inference_steps=20, guidance_scale=7.5, negative_prompt='',
                            generator=torch.Generator(device='cpu').manual_seed(101), cross_attention_kwargs=cross,
                            callback=callback, callback_steps=1).images[0]
                        image.save(path)
                        need(len(steps) == int(20 * strength), 'Unexpected actual denoising step count')
                        row.update(status='complete', sha256=sha(path))
                    except Exception as error:
                        row.update(status='failed', sha256=None, error=str(error), traceback=traceback.format_exc())
                    row['seconds'] = time.monotonic() - start
                    row.update(actual_steps=len(steps), actual_timesteps=steps)
                    records.append(row)
                    log.write(json.dumps(row, ensure_ascii=False) + '\n')
                    print(json.dumps({k: row[k] for k in ('case_id', 'method', 'status', 'seconds')}), flush=True)
            # First-case acceptance is engineering only: complete finite outputs
            # and exact noedit conditions/PNGs. No quality-based case selection.
            if len(records) == 6:
                need(all(r['status'] == 'complete' for r in records), 'First-case pipeline failed; stop bounded trial')
                if c['group'] == 'no_edit':
                    need(all(records[i]['sha256'] == records[i + 3]['sha256'] for i in range(3)), 'Noedit source-centered regression failed')
                save(out / 'first_case_engineering_gate.json', {'status': 'PASS', 'generated': 6, 'quality_acceptance': False})
    save(out / 'result.json', {'kind': 'v5_source_initialization_diagnostic', 'test_data_used': False,
         'status': 'complete' if all(r['status'] == 'complete' for r in records) else 'completed_with_failures',
         'records': records, 'planned': 48, 'completed': sum(r['status'] == 'complete' for r in records),
         'cases': ref(out / 'cases.json'), 'configuration': ref(out / 'configuration.json')})
    decode_guard.remove()
    print(json.dumps({'status': 'diagnostic_complete', 'generated': len(records), 'run': str(out)}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['preflight', 'prepare', 'generate'])
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.phase == 'preflight':
        print(json.dumps({'status': 'verified_dev_metadata', 'cases': len(preflight()), 'models_loaded': False}))
    else:
        need(args.run is not None, 'Explicit new run path required')
        output = args.run.resolve()
        need(output.parent == HERE / 'runs', 'Must remain in new v5 runs directory')
        (prepare if args.phase == 'prepare' else generate)(output)
