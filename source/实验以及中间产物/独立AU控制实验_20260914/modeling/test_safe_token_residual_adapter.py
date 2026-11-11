"""Development-v3 structural safety checks; no training or image generation."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.pycache_prefix = str(HERE / 'test_pycache')

import torch
from token_residual_adapter import TokenAdapterConfig, TokenCrossModalResidualAdapter
from safe_token_residual_adapter import SafeTokenCrossModalResidualAdapter, SafeTokenPooledMLPResidualAdapter, RelativeRMSConfig, augment_condition_embeddings, condition_residuals, relative_rms_trust_region, suggest_matched_mlp_hidden_dim

SEED = 20260914
CACHE_DIR = None
OBSERVATIONS = {}


def small(**kwargs):
    return replace(TokenAdapterConfig(), image_dim=32, text_dim=40, hidden_dim=32, au_hidden_dim=16, au_feature_dim=32, condition_dim=64, **kwargs)


def inputs(config, device='cpu'):
    return (torch.randn(2, 49, config.image_dim).to(device), torch.randn(2, config.text_dim).to(device), (torch.rand(2, config.num_aus) * 3).to(device))


class SafeReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(SEED)

    def test_v1_and_v2_sources_unchanged(self):
        for name, expected in (('cross_modal_adapter.py', 'dab32f9c0969682281f7d51559c3ccefea05dbc93edd54fa8561ed5c802d40f4'), ('token_residual_adapter.py', '42b0dd3e4bb1287e4484d9700cfcf3798fe4064cb9f8691468944e79409bf146')):
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), expected)

    def test_au_head_only_learns_remaining_dimensions(self):
        model = SafeTokenCrossModalResidualAdapter()
        self.assertEqual(model.au_output.weight.shape, (756, 128))
        self.assertIsNone(model.au_output.bias)
        old = TokenCrossModalResidualAdapter()
        self.assertEqual(old.parameter_counts()['total'] - model.parameter_counts()['total'], 1536)
        self.assertFalse(any('rho' in name or 'trust' in name for name, _ in model.named_parameters()))

    def test_explicit_au_prefix_unchanged_with_nonzero_tail(self):
        cfg = small()
        model = SafeTokenCrossModalResidualAdapter(cfg)
        torch.nn.init.normal_(model.au_output.weight, std=0.1)
        torch.nn.init.normal_(model.image_output.weight, std=0.1)
        base_au, base_image = torch.randn(2, 1, 64), torch.randn(2, 16, 64)
        base_au[..., :12] = torch.arange(12).remainder(6)
        out = model(*inputs(cfg))
        self.assertTrue(torch.equal(out.au_delta[..., :12], torch.zeros(2, 1, 12)))
        for config in (RelativeRMSConfig(), RelativeRMSConfig(enabled=True, rho_au=0.05, rho_ip=0.05)):
            changed = augment_condition_embeddings(base_au, base_image, out, trust_region=config)
            self.assertTrue(torch.equal(changed['au_embedding'][..., :12], base_au[..., :12]))
        out.au_delta = out.au_delta.clone()
        out.au_delta[0, 0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, 'prefix'):
            augment_condition_embeddings(base_au, base_image, out)

    def test_initial_conditions_exact_for_all_independent_factors(self):
        cfg = small()
        args = inputs(cfg)
        base_au, base_ip = torch.randn(2, 1, 64), torch.randn(2, 16, 64)
        for readout in ('global', 'slot'):
            for scale in ('raw', 'unit_mean'):
                model = SafeTokenCrossModalResidualAdapter(replace(cfg, readout=readout, patch_scale=scale))
                output = model(*args)
                for trust in (RelativeRMSConfig(), RelativeRMSConfig(enabled=True, rho_au=0.1, rho_ip=0.1)):
                    result = augment_condition_embeddings(base_au, base_ip, output, trust_region=trust)
                    self.assertTrue(torch.equal(result['au_embedding'], base_au))
                    self.assertTrue(torch.equal(result['image_embedding'], base_ip))

    def test_relative_rms_bound_per_slot_uses_shared_reference_scalar(self):
        raw = torch.randn(2, 16, 64, dtype=torch.float64) * 100
        reference = torch.randn(2, 16, 64, dtype=torch.float64) * torch.arange(1, 17).reshape(1, 16, 1)
        bounded = relative_rms_trust_region(raw, reference, 0.07)
        bound = 0.07 * reference.square().mean((1, 2), keepdim=True).sqrt()
        observed = bounded.square().mean(-1, keepdim=True).sqrt()
        self.assertTrue(torch.all(observed <= bound * (1 + 1e-12)))

    def test_zero_raw_residual_has_finite_identity_derivative(self):
        raw = torch.zeros(2, 16, 8, dtype=torch.float64, requires_grad=True)
        reference = torch.zeros_like(raw)
        output = relative_rms_trust_region(raw, reference, 0.1)
        grad = torch.autograd.grad(output.sum(), raw)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.allclose(grad, torch.ones_like(grad), atol=1e-12, rtol=1e-12))

    def test_rho_zero_diagnostic_returns_zero_without_nan(self):
        raw = torch.full((1, 1, 8), float('nan'))
        result = relative_rms_trust_region(raw, torch.zeros(1, 16, 8), 0.0)
        self.assertTrue(torch.equal(result, torch.zeros_like(raw)))
        self.assertTrue(torch.isfinite(result).all())

    def test_training_config_requires_explicit_positive_rhos(self):
        self.assertFalse(RelativeRMSConfig().enabled)
        with self.assertRaisesRegex(ValueError, 'explicit positive'):
            RelativeRMSConfig(enabled=True)
        with self.assertRaisesRegex(ValueError, 'diagnostic-only'):
            RelativeRMSConfig(enabled=True, rho_au=0, rho_ip=0.1)
        with self.assertRaisesRegex(ValueError, 'positive'):
            RelativeRMSConfig(enabled=True, rho_au=-0.1, rho_ip=0.1)

    def test_global_residual_not_turned_into_different_token_updates(self):
        raw = torch.randn(2, 1, 32)
        reference = torch.randn(2, 16, 32) * torch.arange(1, 17).reshape(1, 16, 1)
        global_value = relative_rms_trust_region(raw, reference, 0.1)
        repeated = relative_rms_trust_region(raw.expand(-1, 16, -1), reference, 0.1)
        self.assertEqual(global_value.shape, (2, 1, 32))
        self.assertTrue(torch.equal(global_value.expand(-1, 16, -1), repeated))

    def test_trust_disabled_preserves_unbounded_v3_residuals(self):
        model = SafeTokenCrossModalResidualAdapter(small())
        torch.nn.init.normal_(model.au_output.weight, std=0.03)
        torch.nn.init.normal_(model.image_output.weight, std=0.03)
        output = model(*inputs(model.config))
        effective = condition_residuals(torch.randn(2, 1, 64), torch.randn(2, 16, 64), output)
        self.assertTrue(torch.equal(effective['au_delta'], output.au_delta))
        self.assertTrue(torch.equal(effective['image_delta'], output.image_delta))

    def first_gradient_probe(self, device):
        cfg = small(readout='slot')
        model = SafeTokenCrossModalResidualAdapter(cfg).to(device)
        args = inputs(cfg, device)
        base_au, base_ip = torch.randn(2, 1, 64).to(device), torch.randn(2, 16, 64).to(device)
        target_au, target_ip = torch.randn(2, 1, 64).to(device), torch.randn(2, 16, 64).to(device)
        values = []
        for config in (RelativeRMSConfig(), RelativeRMSConfig(enabled=True, rho_au=0.05, rho_ip=0.08)):
            model.zero_grad(set_to_none=True)
            output = model(*args)
            result = augment_condition_embeddings(base_au, base_ip, output, trust_region=config)
            loss = (result['au_embedding'] - target_au).square().mean() + (result['image_embedding'] - target_ip).square().mean()
            loss.backward()
            a, i = model.au_output.weight.grad, model.image_output.weight.grad
            self.assertTrue(torch.isfinite(a).all() and torch.isfinite(i).all())
            self.assertGreater(float(a.norm().detach().cpu()), 0)
            self.assertGreater(float(i.norm().detach().cpu()), 0)
            values.append((a.detach().clone(), i.detach().clone()))
        self.assertTrue(torch.allclose(values[0][0], values[1][0], atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(values[0][1], values[1][1], atol=1e-6, rtol=1e-5))
        # Nonzero-input limiter backward on the actual device is also finite.
        raw = torch.randn(2, 16, 64).to(device).requires_grad_(True)
        limited = relative_rms_trust_region(raw, base_ip, 0.08)
        limited.square().mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.norm().detach().cpu()), 0)
        OBSERVATIONS['first_gradient_' + device] = {'au_head_norm': float(values[1][0].norm().cpu()), 'image_head_norm': float(values[1][1].norm().cpu()), 'matches_unbounded_at_zero': True, 'nonzero_raw_gradient_norm': float(raw.grad.norm().detach().cpu())}

    def test_cpu_gradient_alive_with_zero_heads_and_positive_radius(self):
        self.first_gradient_probe('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_gradient_alive_with_zero_heads_and_positive_radius(self):
        self.first_gradient_probe('mps')

    def test_parameter_matched_controls_really_remove_au_prefix_weights(self):
        counts = {}
        for readout in ('global', 'slot'):
            config = TokenAdapterConfig(readout=readout)
            reference = SafeTokenCrossModalResidualAdapter(config)
            width = suggest_matched_mlp_hidden_dim(reference)
            controls = {'bidirectional': reference, 'condition_reads_image': SafeTokenCrossModalResidualAdapter(replace(config, direction='condition_reads_image', ff_multiplier=3)), 'image_reads_condition': SafeTokenCrossModalResidualAdapter(replace(config, direction='image_reads_condition', ff_multiplier=3)), 'mlp': SafeTokenPooledMLPResidualAdapter(config, width)}
            for model in controls.values():
                self.assertEqual(model.au_output.weight.numel(), 756 * 128)
                self.assertLess(abs(model.parameter_counts()['total'] - reference.parameter_counts()['total']), 300)
            counts[readout] = {name: model.parameter_counts()['total'] for name, model in controls.items()}
        self.assertEqual(counts['global'], counts['slot'])
        OBSERVATIONS['parameter_counts'] = counts

    def test_real_public_au_prefix_survives_nonzero_protected_residual(self):
        if CACHE_DIR is None:
            self.skipTest('No explicit new training cache supplied')
        sys.path.insert(0, str(HERE.parent))
        from modeling.frozen_backbone import FF, IP, fineface_classes, sha256
        manifest = json.loads((CACHE_DIR / 'manifest.json').read_text())
        self.assertFalse(manifest['test_data_used'])
        self.assertEqual(sha256(CACHE_DIR / 'features.pt'), manifest['cache_sha256'])
        samples = torch.load(CACHE_DIR / 'features.pt', map_location='cpu', weights_only=True)['samples']
        sample = next(row for row in samples if row['split'] == 'train')
        _, _, AUEncoder, MLPProjModel = fineface_classes()
        au_encoder = AUEncoder(hidden_dim=756, clip_dim=768, pad_zeros=False).requires_grad_(False)
        au_encoder.load_state_dict(torch.load(FF / 'sd15_au_encoder.ckpt', map_location='cpu', weights_only=True))
        ip = MLPProjModel(cross_attention_dim=768, id_embeddings_dim=512, num_tokens=16).requires_grad_(False)
        ip.load_state_dict(torch.load(IP / 'ip-adapter-faceid-portrait-v11_sd15.bin', map_location='cpu', weights_only=True)['image_proj'])
        raw_aus = sample['target_aus'].unsqueeze(0)
        base_au, base_ip = au_encoder(raw_aus), ip(sample['base_faceid'].unsqueeze(0))
        self.assertTrue(torch.equal(base_au[..., :12], raw_aus.unsqueeze(1)))
        model = SafeTokenCrossModalResidualAdapter()
        torch.nn.init.normal_(model.au_output.weight, std=0.1)
        torch.nn.init.normal_(model.image_output.weight, std=0.1)
        out = model(sample['image_patches'].unsqueeze(0), sample['text_pooled'].unsqueeze(0), raw_aus)
        updated = augment_condition_embeddings(base_au, base_ip, out, trust_region=RelativeRMSConfig(enabled=True, rho_au=0.05, rho_ip=0.05))
        self.assertTrue(torch.equal(updated['au_embedding'][..., :12], raw_aus.unsqueeze(1)))
        OBSERVATIONS['real_au_prefix'] = {'case_id': sample['case_id'], 'cache_sha256': manifest['cache_sha256'], 'exact_original_0_to_5_values_preserved': True, 'training_performed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    CACHE_DIR = args.cache_dir
    start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SafeReadoutTests))
    report = {'seed': SEED, 'torch': torch.__version__, 'python': sys.version, 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped), 'seconds': time.monotonic() - start, 'scope': 'structural RMS/gradient/condition tests only; no training or generation', 'observations': OBSERVATIONS}
    if args.report:
        with args.report.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(not result.wasSuccessful())
