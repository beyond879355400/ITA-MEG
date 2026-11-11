"""New development-only baseline/adapter images with equal fixed sampling seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                  NO_ALBUMENTATIONS_UPDATE='1', HF_HOME=str(ROOT / 'cache/huggingface'),
                  TORCH_HOME=str(ROOT / 'cache/torch'), MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'))
sys.dont_write_bytecode = True
sys.pycache_prefix = str(ROOT / 'cache/pycache')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--residual-scale', type=float, default=1.0)
    p.add_argument('--au-residual-scale', type=float)
    p.add_argument('--ip-residual-scale', type=float)
    p.add_argument('--max-subjects', type=int, default=2)
    p.add_argument('--repeat-baseline', action='store_true')
    p.add_argument('--branch-isolation', action='store_true')
    args = p.parse_args()
    if not 10 <= args.steps <= 30:
        raise ValueError('Bounded development preview only')
    if not 0 <= args.residual_scale <= 1 or args.max_subjects not in (1, 2):
        raise ValueError('Invalid development diagnostic controls')
    au_scale = args.residual_scale if args.au_residual_scale is None else args.au_residual_scale
    ip_scale = args.residual_scale if args.ip_residual_scale is None else args.ip_residual_scale
    if not 0 <= au_scale <= 1 or not 0 <= ip_scale <= 1:
        raise ValueError('Invalid branch residual scale')
    args.output = args.output.resolve()
    if ROOT not in args.output.parents or args.output.exists():
        raise ValueError('New in-project output required')
    import torch
    from diffusers import StableDiffusionPipeline
    from modeling.frozen_backbone import FrozenConditionedDenoiser, FrozenImageTextEncoders, RV, sha256
    from modeling.cross_modal_adapter import AdapterConfig, CrossModalResidualAdapter, PooledMLPResidualAdapter, augment_condition_embeddings
    metadata = json.loads((args.cache / 'manifest.json').read_text())
    if sha256(args.cache / 'features.pt') != metadata['cache_sha256'] or metadata['test_data_used']:
        raise ValueError('Bad cache or test leakage')
    cache = torch.load(args.cache / 'features.pt', map_location='cpu', weights_only=True)['samples']
    chosen, seen = [], set()
    for sample in cache:
        if sample['split'] == 'dev' and sample['subject'] not in seen:
            chosen.append(sample)
            seen.add(sample['subject'])
    if len(chosen) != 2:
        raise ValueError('Pilot expects exactly two development subjects')
    chosen = chosen[:args.max_subjects]
    actual_pairs = {x['case_id']: x for x in metadata['pairs']}
    models = [('baseline', None, None, 0.0, 0.0)]
    for path in args.run:
        result = json.loads((path / 'result.json').read_text())
        if result['status'] != 'complete' or result['source_cache_sha256'] != metadata['cache_sha256']:
            raise ValueError('Trained run does not bind to this development cache')
        if sha256(path / 'adapter_final.pt') != result['checkpoint_sha256']:
            raise ValueError('Checkpoint hash changed')
        state = torch.load(path / 'adapter_final.pt', map_location='cpu', weights_only=True)
        config = AdapterConfig(**state['config'])
        variant = result['args']['variant']
        model = PooledMLPResidualAdapter(config, 1032) if variant == 'mlp' else CrossModalResidualAdapter(config)
        model.load_state_dict(state['adapter'], strict=True)
        model.eval().requires_grad_(False).to('mps')
        if args.branch_isolation:
            models.extend((variant+suffix, model, str(path.resolve()), a, i) for suffix, a, i in
                          (('_au_only', 1.0, 0.0), ('_ip_only', 0.0, 1.0), ('_small_both', 0.1, 0.1)))
        else:
            models.append((variant, model, str(path.resolve()), au_scale, ip_scale))
    if args.repeat_baseline:
        models.append(('baseline_repeat', None, None, 0.0, 0.0))
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / 'preview_source.py')
    backbone = FrozenConditionedDenoiser('mps', checkpointing=False)
    encoders = FrozenImageTextEncoders('mps')
    pipe = StableDiffusionPipeline.from_pretrained(str(RV), unet=backbone.unet, vae=encoders.vae,
        text_encoder=encoders.text_encoder, tokenizer=encoders.tokenizer, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False, local_files_only=True).to('mps')
    records = []
    with torch.no_grad():
        for sample in chosen:
            pair = actual_pairs[sample['case_id']]
            directory = args.output / sample['case_id']
            directory.mkdir()
            for key in ('source', 'target'):
                original = Path(pair['aligned_' + key]['aligned_path'])
                shutil.copyfile(original, directory / (key + '_real.png'))
            patches = sample['image_patches'].unsqueeze(0).to('mps')
            text = sample['text_pooled'].unsqueeze(0).to('mps')
            aus = sample['target_aus'].unsqueeze(0).to('mps')
            source_id = sample['base_faceid'].unsqueeze(0).to('mps')
            au_cond, id_cond = backbone.baseline_conditions(aus, source_id)
            au_null, id_null = backbone.baseline_conditions(torch.zeros_like(aus), torch.zeros_like(source_id))
            for name, adapter, run, au_scale, ip_scale in models:
                if adapter is None:
                    edited = {'au_embedding': au_cond, 'image_embedding': id_cond}
                else:
                    edited = augment_condition_embeddings(au_cond, id_cond, adapter(patches, text, aus))
                    edited = {'au_embedding': au_cond + au_scale*(edited['au_embedding']-au_cond),
                              'image_embedding': id_cond + ip_scale*(edited['image_embedding']-id_cond)}
                    if au_scale == 0 and ip_scale == 0 and (not torch.equal(edited['au_embedding'], au_cond) or not torch.equal(edited['image_embedding'], id_cond)):
                        raise RuntimeError('Zero residual did not preserve conditions')
                cross = {'au_embedding': torch.cat((au_null, edited['au_embedding'])), 'au_scale': 1.0,
                         'image_embedding': torch.cat((id_null, edited['image_embedding'])), 'ip_scale': 0.9}
                start = time.monotonic()
                condition_hashes = {key: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                                    for key, value in cross.items() if isinstance(value, torch.Tensor)}
                image = pipe(sample['prompt'], negative_prompt='', height=512, width=512,
                             num_inference_steps=args.steps, guidance_scale=7.5,
                             generator=torch.Generator(device='cpu').manual_seed(args.seed),
                             cross_attention_kwargs=cross).images[0]
                path = directory / (name + '.png')
                image.save(path)
                record = {'case_id': sample['case_id'], 'subject': sample['subject'], 'split': 'dev',
                          'method': name, 'training_run': run, 'output': str(path), 'sha256': sha256(path),
                          'prompt': sample['prompt'], 'target_aus_0_5': sample['target_aus'].tolist(),
                          'seed': args.seed, 'steps': args.steps, 'residual_scale': args.residual_scale,
                          'au_residual_scale': au_scale, 'ip_residual_scale': ip_scale,
                          'condition_hashes': condition_hashes, 'seconds': time.monotonic()-start}
                records.append(record)
                (args.output / 'generation_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
                print(json.dumps(record), flush=True)
    (args.output / 'result.json').write_text(json.dumps({'status': 'complete', 'images': len(records),
        'sampling_policy': 'first fixed case of each dev subject; one equal seed for every method; no selection by score',
        'test_data_used': False, 'formal_sota_evidence': False, 'cache_sha256': metadata['cache_sha256']}, indent=2)+'\n')


if __name__ == '__main__':
    main()
