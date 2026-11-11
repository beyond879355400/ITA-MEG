"""Real train/dev pipeline pilot. Not a final SOTA experiment or AU-trained model."""
import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                  NO_ALBUMENTATIONS_UPDATE='1', HF_HOME=str(ROOT / 'cache/huggingface'),
                  TORCH_HOME=str(ROOT / 'cache/torch'), MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'))
sys.dont_write_bytecode = True
sys.pycache_prefix = str(ROOT / 'cache/pycache')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--identity-weight', type=float, default=0.1)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    p.add_argument('--variant', choices=('bidirectional', 'condition_reads_image', 'image_reads_condition', 'mlp'), default='bidirectional')
    args = p.parse_args()
    if not 2 <= args.steps <= 128 or not 0 < args.lr <= 0.001 or not 0 <= args.identity_weight <= 1:
        raise ValueError('This entry is a bounded pilot only')
    import torch
    from modeling.cross_modal_adapter import AdapterConfig, CrossModalResidualAdapter, PooledMLPResidualAdapter, augment_condition_embeddings
    from modeling.frozen_backbone import FrozenConditionedDenoiser, FrozenImageTextEncoders, sha256
    cache_manifest = json.loads((args.cache / 'manifest.json').read_text())
    cache_path = args.cache / 'features.pt'
    if sha256(cache_path) != cache_manifest['cache_sha256'] or cache_manifest['test_data_used']:
        raise ValueError('Cache integrity or test exclusion failed')
    samples = torch.load(cache_path, map_location='cpu', weights_only=True)['samples']
    train = [x for x in samples if x['split'] == 'train']
    dev = [x for x in samples if x['split'] == 'dev']
    if not train or not dev or len(train)+len(dev) != len(samples):
        raise ValueError('Explicit train/dev only inputs required')
    if {x['subject'] for x in train} & {x['subject'] for x in dev}:
        raise ValueError('Subject leakage')
    name = f"pilot_{args.variant}_{datetime.now().strftime('%Y%m%d%H%M')}"
    output = ROOT / 'runs' / name
    number = 2
    while output.exists():
        output = ROOT / 'runs' / (name + f'-{number:02d}')
        number += 1
    output.mkdir(parents=True)
    print('TRAIN_OUTPUT=' + str(output), flush=True)
    checkpoint = output / 'adapter_final.pt'
    result = {'status': 'running', 'stage': 'train_dev_pipeline_pilot', 'formal_sota_evidence': False,
              'au_supervision_enabled': False, 'reason': 'train-only AU supervisor is not trained yet',
              'old_generated_images_or_metrics_used': False, 'test_data_used': False,
              'source_cache_sha256': cache_manifest['cache_sha256'], 'args': vars(args) | {'cache': str(args.cache.resolve())}}
    source_paths = [Path(__file__), ROOT / 'modeling/cross_modal_adapter.py', ROOT / 'modeling/frozen_backbone.py']
    (output / 'snapshots').mkdir()
    for path in source_paths:
        shutil.copyfile(path, output / 'snapshots' / path.name)
    result['source_sha256'] = {path.name: sha256(path) for path in source_paths}
    try:
        with (output / 'run.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            torch.manual_seed(20260914)
            device = torch.device(args.device)
            sync = torch.mps.synchronize if args.device == 'mps' else lambda: None
            backbone = FrozenConditionedDenoiser(device=args.device, checkpointing=True)
            encoders = FrozenImageTextEncoders(device=args.device) if args.identity_weight else None
            torch.manual_seed(20260914)
            cfg = AdapterConfig(direction=args.variant if args.variant != 'mlp' else 'bidirectional',
                                ff_multiplier=3 if args.variant in ('condition_reads_image', 'image_reads_condition') else 2)
            adapter = (PooledMLPResidualAdapter(cfg, 1032) if args.variant == 'mlp' else CrossModalResidualAdapter(cfg)).to(device)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
            result.update(adapter_config=asdict(cfg), trainable_parameters=adapter.parameter_counts()['trainable'],
                          train_case_ids=[x['case_id'] for x in train], dev_case_ids=[x['case_id'] for x in dev],
                          train_subjects=sorted({x['subject'] for x in train}), dev_subjects=sorted({x['subject'] for x in dev}),
                          timestep_range_inclusive=[50, 500], gradient_clip_norm=1.0,
                          residual_regularization_weight=0.0001, optimizer='AdamW', weight_decay=0.01)
            rng = torch.Generator(device='cpu').manual_seed(20260915)
            fields = ('image_patches', 'text_pooled', 'text_tokens', 'target_aus', 'base_faceid', 'source_identity', 'latent_mean', 'latent_std')

            def batch(row):
                return {key: row[key].unsqueeze(0).to(device) for key in fields}

            def noise_input(b, generator, timestep=None):
                latent_noise = torch.randn(b['latent_mean'].shape, generator=generator).to(device)
                clean = b['latent_mean'] + b['latent_std'] * latent_noise
                target_noise = torch.randn(clean.shape, generator=generator).to(device)
                time_index = torch.randint(50, 501, (1,), generator=generator).to(device) if timestep is None else torch.tensor([timestep], device=device)
                return clean, target_noise, time_index, backbone.noise_scheduler.add_noise(clean, target_noise, time_index)

            def predict(b, noisy, t, enabled=True):
                base_au, base_image = backbone.baseline_conditions(b['target_aus'], b['base_faceid'])
                adapted = adapter(b['image_patches'], b['text_pooled'], b['target_aus'])
                values = augment_condition_embeddings(base_au, base_image, adapted) if enabled else {'au_embedding': base_au, 'image_embedding': base_image}
                predicted = backbone(noisy, t, b['text_tokens'], values['au_embedding'], values['image_embedding'])
                return predicted, adapted

            def validation():
                adapter.eval()
                records = []
                with torch.no_grad():
                    for i, row in enumerate(dev):
                        b = batch(row)
                        generator = torch.Generator(device='cpu').manual_seed(20261914 + i)
                        clean, noise, t, noisy = noise_input(b, generator, 250)
                        prediction, _ = predict(b, noisy, t)
                        records.append({'case_id': row['case_id'], 'noise_mse': float((prediction-noise).square().mean().cpu())})
                adapter.train()
                return records

            with torch.no_grad():
                b = batch(train[0])
                _, _, t, noisy = noise_input(b, torch.Generator(device='cpu').manual_seed(999), 250)
                before, _ = predict(b, noisy, t, False)
                after, _ = predict(b, noisy, t, True)
                result['zero_init_denoiser_max_abs_difference'] = float((before-after).abs().max().cpu())
                if not torch.equal(before, after):
                    raise RuntimeError('Zero initialization did not exactly preserve denoiser output')
            result['dev_before'] = validation()
            trace = []
            started = time.monotonic()
            with (output / 'training_trace.jsonl').open('x', buffering=1) as trace_file:
                for step in range(args.steps):
                    row = train[step % len(train)]
                    b = batch(row)
                    optimizer.zero_grad(set_to_none=True)
                    _, noise, t, noisy = noise_input(b, rng)
                    sync()
                    begin = time.monotonic()
                    predicted, adapted = predict(b, noisy, t)
                    diffusion_loss = (predicted-noise).square().mean()
                    identity_loss = torch.zeros((), device=device)
                    if encoders is not None:
                        alpha = backbone.noise_scheduler.alphas_cumprod.to(device)[t].view(1, 1, 1, 1)
                        x0_latent = (noisy-(1-alpha).sqrt()*predicted) / alpha.sqrt()
                        decoded = encoders.decode_latents(x0_latent).clamp(-1, 1)
                        identity = encoders.identity_embedding(decoded)
                        identity_loss = (1-torch.nn.functional.cosine_similarity(identity, b['source_identity'])).mean()
                    residual = adapted.au_delta.square().mean()+adapted.image_delta.square().mean()
                    loss = diffusion_loss+args.identity_weight*identity_loss+0.0001*residual
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                    if not torch.isfinite(loss) or not torch.isfinite(gradient):
                        raise RuntimeError('Nonfinite pilot loss or gradient')
                    directional_grads = {}
                    if hasattr(adapter, 'blocks'):
                        block = adapter.blocks[0]
                        for name in ('condition_from_image', 'image_from_condition'):
                            module = getattr(block, name)
                            if module is not None:
                                directional_grads[name] = {part: float(getattr(module, part).weight.grad.norm().cpu()) for part in ('q', 'k', 'v')}
                    if any(p.grad is not None for p in backbone.parameters()):
                        raise RuntimeError('Frozen backbone unexpectedly has parameter gradients')
                    optimizer.step()
                    sync()
                    record = {'step': step+1, 'case_id': row['case_id'], 'timestep': int(t.cpu()),
                              'loss': float(loss.detach().cpu()), 'diffusion_loss': float(diffusion_loss.detach().cpu()),
                              'identity_loss': float(identity_loss.detach().cpu()), 'residual_penalty': float(residual.detach().cpu()),
                              'unclipped_gradient_norm': float(gradient.cpu()), 'directional_gradient_norms': directional_grads,
                              'seconds': time.monotonic()-begin}
                    trace.append(record)
                    trace_file.write(json.dumps(record)+'\n')
                    print(json.dumps(record), flush=True)
                    if (step+1) % 8 == 0:
                        torch.save({'adapter': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                                    'optimizer': optimizer.state_dict(), 'config': asdict(cfg), 'step': step+1,
                                    'stage': result['stage'], 'cache_sha256': result['source_cache_sha256']}, output / f'checkpoint_{step+1:04d}.pt')
                    del loss, predicted, adapted, diffusion_loss, identity_loss, residual
                    if encoders is not None:
                        del decoded, identity, x0_latent
            result['training_seconds'] = time.monotonic()-started
            result['dev_after'] = validation()
            result['dev_noise_mse_before'] = sum(x['noise_mse'] for x in result['dev_before'])/len(dev)
            result['dev_noise_mse_after'] = sum(x['noise_mse'] for x in result['dev_after'])/len(dev)
            torch.save({'adapter': {k: v.detach().cpu() for k, v in adapter.state_dict().items()}, 'config': asdict(cfg),
                        'step': args.steps, 'stage': result['stage'], 'cache_sha256': result['source_cache_sha256']}, checkpoint)
            result.update(status='complete', checkpoint_sha256=sha256(checkpoint), steps_completed=len(trace),
                          frozen_denoiser_parameters_unchanged=True,
                          note='Development pipeline check only; denoising MSE is not final AU or SOTA performance.')
    except Exception as error:
        result.update(status='fail', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('dev_before', 'dev_after')}, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
