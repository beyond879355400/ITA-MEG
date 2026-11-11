"""Measure v1 development condition changes without a generator or test data."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.pycache_prefix = str(ROOT / 'cache/pycache')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from modeling.frozen_backbone import FF, IP, fineface_classes, sha256
    from modeling.cross_modal_adapter import AdapterConfig, CrossModalResidualAdapter, PooledMLPResidualAdapter
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('Require a new experiment output directory')
    meta = json.loads((args.cache / 'manifest.json').read_text())
    if meta['test_data_used'] or sha256(args.cache / 'features.pt') != meta['cache_sha256']:
        raise ValueError('Cache integrity or test exclusion failed')
    samples = torch.load(args.cache / 'features.pt', map_location='cpu', weights_only=True)['samples']
    if any(s['split'] not in ('train', 'dev') for s in samples):
        raise ValueError('Diagnostic accepts only train/dev')
    _, _, AUEncoder, IPProjection = fineface_classes()
    au_encoder = AUEncoder(hidden_dim=756, clip_dim=768, pad_zeros=False)
    au_encoder.load_state_dict(torch.load(FF / 'sd15_au_encoder.ckpt', weights_only=True, map_location='cpu'))
    projection = IPProjection(cross_attention_dim=768, id_embeddings_dim=512, num_tokens=16)
    projection.load_state_dict(torch.load(IP / 'ip-adapter-faceid-portrait-v11_sd15.bin', weights_only=True, map_location='cpu')['image_proj'])
    au_encoder.eval().requires_grad_(False)
    projection.eval().requires_grad_(False)
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / 'diagnostic_source.py')
    rms = lambda x: float(x.square().mean().sqrt())
    records = []
    with torch.no_grad():
        for run in args.run:
            run_result = json.loads((run / 'result.json').read_text())
            if run_result['source_cache_sha256'] != meta['cache_sha256'] or sha256(run / 'adapter_final.pt') != run_result['checkpoint_sha256']:
                raise ValueError('Run/checkpoint identity mismatch')
            state = torch.load(run / 'adapter_final.pt', map_location='cpu', weights_only=True)
            cfg = AdapterConfig(**state['config'])
            variant = run_result['args']['variant']
            adapter = PooledMLPResidualAdapter(cfg, 1032) if variant == 'mlp' else CrossModalResidualAdapter(cfg)
            adapter.load_state_dict(state['adapter'], strict=True)
            adapter.eval().requires_grad_(False)
            for s in samples:
                au, ip = au_encoder(s['target_aus'][None]), projection(s['base_faceid'][None])
                residual = adapter(s['image_patches'][None], s['text_pooled'][None], s['target_aus'][None])
                au_tail_rms = rms(au[..., 12:])
                ip_rms = rms(ip)
                record = {'case_id': s['case_id'], 'subject': s['subject'], 'split': s['split'], 'method': variant,
                          'base_au_tail_rms': au_tail_rms, 'au_tail_residual_rms': rms(residual.au_delta[..., 12:]),
                          'au_tail_relative_rms': rms(residual.au_delta[..., 12:])/max(1e-12, au_tail_rms),
                          'explicit_au_residual_max_abs': float(residual.au_delta[..., :12].abs().max()),
                          'explicit_au_before': au[0, 0, :12].tolist(),
                          'explicit_au_after': (au+residual.au_delta)[0, 0, :12].tolist(),
                          'base_ip_rms': ip_rms, 'ip_residual_rms': rms(residual.image_delta),
                          'ip_relative_rms': rms(residual.image_delta)/max(1e-12, ip_rms),
                          'ip_residual_slots': residual.image_delta.shape[1],
                          'checkpoint_sha256': run_result['checkpoint_sha256']}
                records.append(record)
    summary = []
    for method in sorted({r['method'] for r in records}):
        for split in ('train', 'dev'):
            rows = [r for r in records if r['method'] == method and r['split'] == split]
            summary.append({'method': method, 'split': split, 'cases': len(rows), **{
                key: {'mean': sum(r[key] for r in rows)/len(rows), 'max': max(r[key] for r in rows)}
                for key in ('au_tail_relative_rms', 'explicit_au_residual_max_abs', 'ip_relative_rms')}})
    (output / 'records.json').write_text(json.dumps(records, indent=2)+'\n')
    result = {'status': 'complete', 'scope': 'v1 train/dev condition-scale diagnostic, not final generation performance',
              'test_data_used': False, 'cache_sha256': meta['cache_sha256'], 'summary': summary,
              'interpretation_limit': 'Condition-space magnitude alone does not establish causal origin of image drift; branch-isolated sampling is needed.'}
    (output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
