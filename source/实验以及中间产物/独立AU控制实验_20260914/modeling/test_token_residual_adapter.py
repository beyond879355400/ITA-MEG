"""Development-v2 structural tests; no dataset fitting or full model training."""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

import torch
from torch.nn import functional as F
from token_residual_adapter import TokenAdapterConfig, TokenCrossModalResidualAdapter, TokenPooledMLPResidualAdapter, augment_condition_embeddings, spatial_slot_pool, suggest_matched_mlp_hidden_dim

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.pycache_prefix = str(HERE / 'test_pycache')
SEED = 20260914
CACHE_DIR = None
OBSERVATIONS = {}
V1_SHA = 'dab32f9c0969682281f7d51559c3ccefea05dbc93edd54fa8561ed5c802d40f4'


def small(**kwargs):
    return replace(TokenAdapterConfig(), image_dim=32, text_dim=40, hidden_dim=32, au_hidden_dim=16, au_feature_dim=32, condition_dim=64, **kwargs)


def inputs(config, device='cpu'):
    return (torch.randn(2, 49, config.image_dim).to(device), torch.randn(2, config.text_dim).to(device), (torch.rand(2, config.num_aus) * 3).to(device))


def open_diagnostic_heads(model):
    # Deliberate synthetic test state, NOT a fitted model or saved checkpoint.
    torch.nn.init.normal_(model.au_output.weight, std=0.01)
    torch.nn.init.normal_(model.image_output.weight, std=0.01)


class TokenAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(SEED)

    def test_v1_source_remains_frozen(self):
        self.assertEqual(hashlib.sha256((HERE / 'cross_modal_adapter.py').read_bytes()).hexdigest(), V1_SHA)

    def test_pool_is_adaptive_average_pool_forward_and_backward(self):
        tokens = torch.randn(2, 49, 5, dtype=torch.float64, requires_grad=True)
        pooled = spatial_slot_pool(tokens)
        reference = F.adaptive_avg_pool2d(tokens.transpose(1, 2).reshape(2, 5, 7, 7), (4, 4)).flatten(2).transpose(1, 2)
        self.assertTrue(torch.allclose(pooled, reference, atol=1e-12, rtol=1e-12))
        a = torch.autograd.grad(pooled.square().sum(), tokens, retain_graph=True)[0]
        b = torch.autograd.grad(reference.square().sum(), tokens)[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-12, rtol=1e-12))

    def test_shapes_and_zero_condition_equivalence_all_factors(self):
        cfg = small()
        args = inputs(cfg)
        base_au, base_image = torch.randn(2, 1, 64), torch.randn(2, 16, 64)
        for readout in ('global', 'slot'):
            for scale in ('raw', 'unit_mean'):
                model = TokenCrossModalResidualAdapter(replace(cfg, readout=readout, patch_scale=scale))
                output = model(*args, return_attention=True)
                self.assertEqual(output.image_slots.shape, (2, 16, 32))
                self.assertEqual(output.image_delta.shape, (2, 1 if readout == 'global' else 16, 64))
                self.assertEqual(output.au_delta.shape, (2, 1, 64))
                self.assertEqual(output.attention['condition_from_image'].shape, (2, 4, 2, 49))
                updated = augment_condition_embeddings(base_au, base_image, output)
                self.assertTrue(torch.equal(updated['au_embedding'], base_au))
                self.assertTrue(torch.equal(updated['image_embedding'], base_image))

    def test_readout_factor_uses_identical_parameters_and_exact_slot_mean(self):
        config = small(readout='slot')
        slot = TokenCrossModalResidualAdapter(config).eval()
        open_diagnostic_heads(slot)
        global_model = TokenCrossModalResidualAdapter(replace(config, readout='global')).eval()
        global_model.load_state_dict(slot.state_dict(), strict=True)
        args = inputs(config)
        a, b = slot(*args), global_model(*args)
        self.assertEqual(slot.parameter_counts(), global_model.parameter_counts())
        self.assertTrue(torch.equal(a.fusion, b.fusion))
        self.assertTrue(torch.equal(a.image_delta.mean(1, keepdim=True), b.image_delta))
        self.assertTrue(torch.equal(a.au_delta, b.au_delta))

    def test_scale_factor_is_exact_and_independent_of_readout(self):
        cfg = small(patch_scale='raw')
        raw = TokenCrossModalResidualAdapter(cfg)
        unit = TokenCrossModalResidualAdapter(replace(cfg, patch_scale='unit_mean'))
        unit.load_state_dict(raw.state_dict(), strict=True)
        args = inputs(cfg)
        a = raw._encode_scaled(*args, None)
        b = unit._encode_scaled(*args, None)
        self.assertTrue(torch.equal(a[0] * 49, b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertTrue(torch.equal(a[2], b[2]))
        self.assertTrue(torch.equal(b[3], torch.full((2,), 49.0)))
        self.assertEqual(raw.parameter_counts(), unit.parameter_counts())

    def test_slot_residuals_are_different_and_condition_changes_are_slot_specific(self):
        model = TokenCrossModalResidualAdapter(small()).eval()
        open_diagnostic_heads(model)
        patches, text, aus = inputs(model.config)
        original = model(patches, text, aus).image_delta
        changed_au = aus.clone()
        changed_au[:, 0] += 1
        changed = model(patches, text, changed_au).image_delta
        self.assertGreater(float((original - original.mean(1, keepdim=True)).abs().max().detach()), 1e-6)
        effect = changed - original
        self.assertGreater(float((effect - effect.mean(1, keepdim=True)).abs().max().detach()), 1e-7)

    def test_common_key_shift_cancels_and_value_shift_is_global(self):
        q = torch.randn(2, 4, 5, 8, dtype=torch.float64)
        k = torch.randn(2, 4, 16, 8, dtype=torch.float64)
        v = torch.randn_like(k)
        dk, dv = torch.randn(2, 4, 1, 8, dtype=torch.float64), torch.randn(2, 4, 1, 8, dtype=torch.float64)
        weights = (q @ k.transpose(-2, -1) / 8**0.5).softmax(-1)
        shifted = (q @ (k + dk).transpose(-2, -1) / 8**0.5).softmax(-1)
        self.assertTrue(torch.allclose(weights, shifted, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(shifted @ (v + dv) - weights @ v, dv.expand(-1, -1, 5, -1), atol=1e-12, rtol=1e-12))

    def test_nonuniform_key_residual_can_change_token_attention(self):
        q = torch.tensor([[[1.0, 0.0]]])
        keys = torch.zeros(1, 16, 2)
        altered = keys.clone()
        altered[:, 0, 0] = 2
        original = (q @ keys.transpose(-2, -1)).softmax(-1)
        new = (q @ altered.transpose(-2, -1)).softmax(-1)
        self.assertGreater(float(new[0, 0, 0] - original[0, 0, 0]), 0.1)

    def test_existing_ip_length_is_strictly_sixteen(self):
        model = TokenCrossModalResidualAdapter(small())
        output = model(*inputs(model.config))
        for count in (1, 15, 17, 32):
            with self.assertRaisesRegex(ValueError, 'exactly 16'):
                augment_condition_embeddings(torch.zeros(2, 1, 64), torch.zeros(2, count, 64), output)

    def test_masked_pool_and_unit_gain_use_only_valid_patches(self):
        cfg = small(patch_scale='unit_mean')
        model = TokenCrossModalResidualAdapter(cfg)
        patches, text, aus = inputs(cfg)
        mask = torch.zeros(2, 49, dtype=torch.bool)
        mask[:, :3] = True
        result = model(patches, text, aus, image_padding=mask)
        changed = patches.clone()
        changed[:, :3] = 1000
        again = model(changed, text, aus, image_padding=mask)
        self.assertTrue(torch.equal(result.image_slots, again.image_slots))
        self.assertTrue(torch.equal(result.patch_gain, torch.full((2,), 46.0)))

    def gradient_probe(self, device):
        cfg = small()
        model = TokenCrossModalResidualAdapter(cfg).to(device)
        open_diagnostic_heads(model)
        args = inputs(cfg, device)
        out = model(*args)
        loss_weights = torch.randn(2, 16, 64).to(device)
        loss = (out.image_delta * loss_weights).sum() + out.au_delta.square().sum()
        loss.backward()
        grads = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for path in ('condition_from_image', 'image_from_condition'):
            for part in ('q', 'k', 'v'):
                name = f'blocks.0.{path}.{part}.weight'
                grad = dict(model.named_parameters())[name].grad
                self.assertIsNotNone(grad)
                norm = float(grad.detach().norm().cpu())
                self.assertGreater(norm, 0)
                grads[name] = norm
        self.assertTrue((model.slot_position.grad.norm(dim=-1) > 0).all())
        self.assertGreater(float(model.slot_readout[0].weight.grad.norm().cpu()), 0)
        OBSERVATIONS['gradient_' + device] = grads

    def test_cpu_gradient_reaches_both_paths_and_all_readout_slots(self):
        self.gradient_probe('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_gradient_reaches_both_paths_and_all_readout_slots(self):
        self.gradient_probe('mps')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_cpu_mps_nonzero_slot_readout_agrees(self):
        model = TokenCrossModalResidualAdapter(small()).eval()
        open_diagnostic_heads(model)
        mps = copy.deepcopy(model).to('mps')
        args = inputs(model.config)
        a, b = model(*args), mps(*(x.to('mps') for x in args))
        self.assertTrue(torch.allclose(a.image_delta, b.image_delta.cpu(), atol=2e-4, rtol=2e-3))
        OBSERVATIONS['cpu_mps_max_image_delta_error'] = float((a.image_delta - b.image_delta.cpu()).abs().max().detach())

    def test_parameter_matched_controls_share_same_readout(self):
        counts = {}
        for readout in ('global', 'slot'):
            cfg = TokenAdapterConfig(readout=readout)
            bi = TokenCrossModalResidualAdapter(cfg)
            width = suggest_matched_mlp_hidden_dim(bi)
            variants = {'bidirectional': bi, 'condition_reads_image': TokenCrossModalResidualAdapter(replace(cfg, direction='condition_reads_image', ff_multiplier=3)), 'image_reads_condition': TokenCrossModalResidualAdapter(replace(cfg, direction='image_reads_condition', ff_multiplier=3)), 'mlp': TokenPooledMLPResidualAdapter(cfg, width)}
            reference = bi.parameter_counts()['total']
            for name, model in variants.items():
                self.assertLess(abs(model.parameter_counts()['total'] - reference) / reference, 0.001)
                self.assertEqual(sum(p.numel() for p in model.slot_readout.parameters()), sum(p.numel() for p in bi.slot_readout.parameters()))
                self.assertEqual(model.slot_position.numel(), bi.slot_position.numel())
            counts[readout] = {name: model.parameter_counts()['total'] for name, model in variants.items()}
            counts[readout]['mlp_hidden_dim'] = width
        self.assertEqual(counts['global'], counts['slot'])
        OBSERVATIONS['canonical_parameter_counts'] = counts

    def test_mlp_retains_raw_local_stream_for_shared_readout(self):
        cfg = small(patch_scale='unit_mean')
        reference = TokenCrossModalResidualAdapter(cfg)
        model = TokenPooledMLPResidualAdapter(cfg, suggest_matched_mlp_hidden_dim(reference))
        args = inputs(cfg)
        raw = model._encode_scaled(*args, None)
        output = model(*args)
        self.assertTrue(torch.equal(output.image_tokens, raw[0]))
        self.assertTrue(torch.equal(output.image_slots, spatial_slot_pool(raw[0])))
        self.assertFalse(hasattr(model, 'blocks'))

    def test_all_variants_use_every_shared_readout_slot_in_both_modes(self):
        for readout in ('global', 'slot'):
            cfg = small(readout=readout)
            reference = TokenCrossModalResidualAdapter(cfg)
            models = [reference,
                      TokenCrossModalResidualAdapter(replace(cfg, direction='condition_reads_image', ff_multiplier=3)),
                      TokenCrossModalResidualAdapter(replace(cfg, direction='image_reads_condition', ff_multiplier=3)),
                      TokenPooledMLPResidualAdapter(cfg, suggest_matched_mlp_hidden_dim(reference))]
            for model in models:
                open_diagnostic_heads(model)
                output = model(*inputs(cfg))
                loss = output.image_delta.square().sum() + output.au_delta.square().sum()
                loss.backward()
                self.assertTrue((model.slot_position.grad.norm(dim=-1) > 0).all())
                self.assertGreater(float(model.slot_readout[0].weight.grad.norm()), 0)
                self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_zero_init_matches_real_public_conditions_from_new_training_cache(self):
        if CACHE_DIR is None:
            self.skipTest('No explicit new train cache supplied')
        sys.path.insert(0, str(HERE.parent))
        from modeling.frozen_backbone import FF, IP, fineface_classes, sha256
        manifest = json.loads((CACHE_DIR / 'manifest.json').read_text())
        self.assertFalse(manifest['test_data_used'])
        cache_file = CACHE_DIR / 'features.pt'
        self.assertEqual(sha256(cache_file), manifest['cache_sha256'])
        samples = torch.load(cache_file, map_location='cpu', weights_only=True)['samples']
        sample = next(x for x in samples if x['split'] == 'train')
        _, _, AUEncoder, MLPProjModel = fineface_classes()
        encoder = AUEncoder(hidden_dim=756, clip_dim=768, pad_zeros=False).requires_grad_(False)
        encoder.load_state_dict(torch.load(FF / 'sd15_au_encoder.ckpt', map_location='cpu', weights_only=True))
        ip_state = torch.load(IP / 'ip-adapter-faceid-portrait-v11_sd15.bin', map_location='cpu', weights_only=True)
        projection = MLPProjModel(cross_attention_dim=768, id_embeddings_dim=512, num_tokens=16).requires_grad_(False)
        projection.load_state_dict(ip_state['image_proj'])
        raw_au, faceid = sample['target_aus'].unsqueeze(0), sample['base_faceid'].unsqueeze(0)
        base_au, base_image = encoder(raw_au), projection(faceid)
        for readout in ('global', 'slot'):
            for scale in ('raw', 'unit_mean'):
                model = TokenCrossModalResidualAdapter(TokenAdapterConfig(readout=readout, patch_scale=scale))
                out = model(sample['image_patches'].unsqueeze(0), sample['text_pooled'].unsqueeze(0), raw_au)
                actual = augment_condition_embeddings(base_au, base_image, out)
                self.assertTrue(torch.equal(actual['au_embedding'], base_au))
                self.assertTrue(torch.equal(actual['image_embedding'], base_image))
        OBSERVATIONS['real_condition_equivalence'] = {'case_id': sample['case_id'], 'split': 'train', 'cache_sha256': manifest['cache_sha256'], 'all_four_factor_settings_exact': True, 'public_au_input_scale': '0-5 unchanged', 'base_faceid': 'buffalo_l unchanged'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    CACHE_DIR = args.cache_dir
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TokenAdapterTests))
    report = {'seed': SEED, 'torch': torch.__version__, 'python': sys.version, 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped), 'seconds': time.monotonic() - started, 'scope': 'module structure, gradients and zero-condition check; no training or generation', 'default_config': asdict(TokenAdapterConfig()), 'observations': OBSERVATIONS}
    if args.report:
        with args.report.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(not result.wasSuccessful())
