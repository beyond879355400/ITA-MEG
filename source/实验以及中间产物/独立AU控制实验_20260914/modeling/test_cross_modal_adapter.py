"""Synthetic structural/gradient tests only; no images, pretrained weights, or training corpus."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import unittest

import torch
from cross_modal_adapter import AdapterConfig, CrossModalResidualAdapter, PooledMLPResidualAdapter, augment_condition_embeddings, suggest_matched_mlp_hidden_dim

SEED = 20260914
OBSERVATIONS = {}


def inputs(config, batch=2, patches=7, device='cpu'):
    # Generate on CPU for fixed source randomness, then transfer unchanged.
    return (torch.randn(batch, patches, config.image_dim).to(device), torch.randn(batch, config.text_dim).to(device), (torch.rand(batch, config.num_aus) * 3).to(device))


def small_config(**kwargs):
    return replace(AdapterConfig(), image_dim=32, text_dim=40, hidden_dim=32, au_hidden_dim=16, au_feature_dim=32, condition_dim=64, max_patch_tokens=13, **kwargs)


def gradient_norm(model, name):
    grad = dict(model.named_parameters())[name].grad
    return None if grad is None else float(grad.detach().norm().cpu())


def two_step_probe(device='cpu', canonical=False):
    torch.manual_seed(SEED)
    config = AdapterConfig() if canonical else small_config()
    count = 49 if canonical else 7
    model = CrossModalResidualAdapter(config).to(device)
    patches, text, aus = inputs(config, patches=count, device=device)
    base_au = torch.randn(2, 1, config.condition_dim).to(device)
    base_image = torch.randn(2, 16, config.condition_dim).to(device)
    weight_au = (torch.randn(config.condition_dim, 8) * 0.05).to(device)
    weight_image = (torch.randn(config.condition_dim, 8) * 0.05).to(device)
    target = torch.randn(2, 8).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    names = [f'blocks.0.{direction}.{projection}.weight' for direction in ('condition_from_image', 'image_from_condition') for projection in ('q', 'k', 'v')]
    stages = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(patches, text, aus)
        embeddings = augment_condition_embeddings(base_au, base_image, output)
        # Fixed differentiable consumer represents conditioning into a frozen
        # network. This is NOT a diffusion-training result or quality test.
        prediction = embeddings['au_embedding'][:, 0] @ weight_au + embeddings['image_embedding'].mean(1) @ weight_image
        loss = (prediction - target).square().mean()
        loss.backward()
        stages.append({'loss': float(loss.detach().cpu()), 'au_head': gradient_norm(model, 'au_output.weight'), 'image_head': gradient_norm(model, 'image_output.weight'), 'cross_gradients': {name: gradient_norm(model, name) for name in names}})
        optimizer.step()
    OBSERVATIONS['two_step_' + device + ('_canonical' if canonical else '')] = stages
    return model, (patches, text, aus), stages


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(SEED)

    def test_canonical_shapes_and_asymmetric_attention_axes(self):
        model = CrossModalResidualAdapter()
        output = model(*inputs(model.config, patches=49), return_attention=True)
        self.assertEqual(output.condition_tokens.shape, (2, 2, 128))
        self.assertEqual(output.image_tokens.shape, (2, 49, 128))
        self.assertEqual(output.au_delta.shape, (2, 1, 768))
        self.assertEqual(output.image_delta.shape, (2, 1, 768))
        self.assertEqual(output.attention['condition_from_image'].shape, (2, 4, 2, 49))
        self.assertEqual(output.attention['image_from_condition'].shape, (2, 4, 49, 2))
        for weights in output.attention.values():
            self.assertTrue(torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6))

    def test_zero_initialization_preserves_existing_embeddings_exactly(self):
        model = CrossModalResidualAdapter()
        output = model(*inputs(model.config, patches=49))
        au, image = torch.randn(2, 1, 768), torch.randn(2, 16, 768)
        augmented = augment_condition_embeddings(au, image, output)
        self.assertTrue(torch.equal(augmented['au_embedding'], au))
        self.assertTrue(torch.equal(augmented['image_embedding'], image))
        frozen = torch.nn.Linear(768, 8).requires_grad_(False)
        before = frozen(au[:, 0] + image.mean(1))
        after = frozen(augmented['au_embedding'][:, 0] + augmented['image_embedding'].mean(1))
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(after.requires_grad)

    def test_first_step_opens_heads_second_step_reaches_both_directions_cpu(self):
        _, _, stages = two_step_probe('cpu', canonical=True)
        self.assertGreater(stages[0]['au_head'], 0)
        self.assertGreater(stages[0]['image_head'], 0)
        self.assertTrue(all(v == 0 for v in stages[0]['cross_gradients'].values()))
        self.assertTrue(all(v is not None and v > 0 for v in stages[1]['cross_gradients'].values()))

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_first_step_opens_heads_second_step_reaches_both_directions_mps(self):
        _, _, stages = two_step_probe('mps', canonical=True)
        self.assertGreater(stages[0]['au_head'], 0)
        self.assertGreater(stages[0]['image_head'], 0)
        self.assertTrue(all(v == 0 for v in stages[0]['cross_gradients'].values()))
        self.assertTrue(all(v is not None and v > 0 for v in stages[1]['cross_gradients'].values()))

    def test_both_cross_directions_change_nonzero_adapter_output(self):
        model, args, _ = two_step_probe()
        complete = model(*args).au_delta.detach()
        block = model.blocks[0]
        for name in ('condition_from_image', 'image_from_condition'):
            saved = getattr(block, name)
            setattr(block, name, None)
            ablated = model(*args).au_delta.detach()
            setattr(block, name, saved)
            self.assertGreater(float((complete - ablated).abs().max()), 1e-7, name)

    def test_distinct_inputs_reach_trainable_residual(self):
        model, (patches, text, aus), _ = two_step_probe()
        reference = model(patches, text, aus).au_delta.detach()
        changed_au = aus.clone()
        changed_au[:, 0] += 1
        changed_image = patches.clone()
        changed_image[:, 0, :4] += 0.5
        changed_text = text.clone()
        changed_text[:, :4] += 0.5
        for alternative in ((changed_image, text, aus), (patches, changed_text, aus), (patches, text, changed_au)):
            difference = float((model(*alternative).au_delta.detach() - reference).abs().max())
            self.assertGreater(difference, 1e-7)

    def test_padding_masks_keys_queries_and_pooling(self):
        model = CrossModalResidualAdapter(small_config())
        patches, text, aus = inputs(model.config)
        mask = torch.zeros(2, 7, dtype=torch.bool)
        mask[:, -2:] = True
        first = model(patches, text, aus, image_padding=mask, return_attention=True)
        changed = patches.clone()
        changed[:, -2:] += 100
        second = model(changed, text, aus, image_padding=mask, return_attention=True)
        self.assertTrue(torch.equal(first.fusion, second.fusion))
        self.assertTrue(torch.equal(first.image_tokens[:, -2:], torch.zeros_like(first.image_tokens[:, -2:])))
        self.assertTrue(torch.equal(first.attention['condition_from_image'][..., -2:], torch.zeros_like(first.attention['condition_from_image'][..., -2:])))
        self.assertTrue(torch.equal(first.attention['image_from_condition'][:, :, -2:], torch.zeros_like(first.attention['image_from_condition'][:, :, -2:])))
        self.assertTrue(torch.allclose(first.image_weights.sum(-1), torch.ones(2)))

    def test_all_padded_image_is_rejected(self):
        model = CrossModalResidualAdapter(small_config())
        with self.assertRaisesRegex(ValueError, 'valid image patch'):
            model(*inputs(model.config), image_padding=torch.ones(2, 7, dtype=torch.bool))

    def test_au_direct_concat_is_exactly_raw_over_five(self):
        model = CrossModalResidualAdapter(small_config())
        aus = torch.rand(2, 12) * 5
        embedding = model.au_encoder(aus)
        self.assertTrue(torch.equal(embedding[:, :12], aus / 5))

    def test_invalid_au_range_and_shape_are_rejected(self):
        model = CrossModalResidualAdapter(small_config())
        patches, text, aus = inputs(model.config)
        with self.assertRaisesRegex(ValueError, 'AU shape'):
            model(patches, text, aus[:, :-1])
        aus[0, 0] = 6
        with self.assertRaisesRegex(ValueError, r'\[0,5\]'):
            model(patches, text, aus)

    def test_nonmatching_hidden_head_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'divisible'):
            AdapterConfig(hidden_dim=31, num_heads=4)

    def test_multiple_layers_and_non_square_patch_count(self):
        model = CrossModalResidualAdapter(small_config(num_layers=2))
        output = model(*inputs(model.config, batch=3, patches=13), return_attention=True)
        self.assertEqual(output.image_tokens.shape, (3, 13, 32))
        self.assertEqual(output.attention['image_from_condition'].shape, (3, 4, 13, 2))

    def test_single_direction_controls_remove_only_one_cross_path(self):
        config = small_config()
        bidirectional = CrossModalResidualAdapter(config)
        condition_reads = CrossModalResidualAdapter(replace(config, direction='condition_reads_image'))
        image_reads = CrossModalResidualAdapter(replace(config, direction='image_reads_condition'))
        self.assertIsNone(condition_reads.blocks[0].image_from_condition)
        self.assertIsNone(image_reads.blocks[0].condition_from_image)
        expected_difference = 4 * config.hidden_dim * config.hidden_dim + 4 * config.hidden_dim
        self.assertEqual(bidirectional.parameter_counts()['total'] - condition_reads.parameter_counts()['total'], expected_difference)
        self.assertEqual(condition_reads.parameter_counts()['total'], image_reads.parameter_counts()['total'])
        for model, expected in ((condition_reads, 'condition_from_image'), (image_reads, 'image_from_condition')):
            result = model(*inputs(config), return_attention=True)
            self.assertEqual(set(result.attention), {expected})

    def test_mlp_control_nearly_matches_parameter_count(self):
        model = CrossModalResidualAdapter()
        width = suggest_matched_mlp_hidden_dim(model)
        control = PooledMLPResidualAdapter(model.config, width)
        difference = abs(model.parameter_counts()['total'] - control.parameter_counts()['total'])
        self.assertLess(difference / model.parameter_counts()['total'], 0.005)
        result = control(*inputs(model.config, patches=49))
        self.assertTrue(torch.equal(result.au_delta, torch.zeros_like(result.au_delta)))
        OBSERVATIONS['canonical_parameter_counts'] = {'bidirectional': model.parameter_counts(), 'condition_reads_image': CrossModalResidualAdapter(replace(model.config, direction='condition_reads_image')).parameter_counts(), 'image_reads_condition': CrossModalResidualAdapter(replace(model.config, direction='image_reads_condition')).parameter_counts(), 'matched_mlp_hidden_dim': width, 'matched_mlp': control.parameter_counts(), 'mlp_count_absolute_difference': difference}

    def test_wider_single_direction_controls_nearly_match_parameters(self):
        config = AdapterConfig()
        reference = CrossModalResidualAdapter(config).parameter_counts()['total']
        counts = {}
        for direction in ('condition_reads_image', 'image_reads_condition'):
            control = CrossModalResidualAdapter(replace(config, direction=direction, ff_multiplier=3))
            total = control.parameter_counts()['total']
            self.assertLess(abs(total - reference) / reference, 0.001)
            counts[direction] = total
        OBSERVATIONS['parameter_matched_single_direction_ff_multiplier_3'] = counts

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_cpu_mps_nonzero_forward_agreement(self):
        model = CrossModalResidualAdapter(small_config()).eval()
        with torch.no_grad():
            model.au_output.weight.normal_(std=0.01)
            model.image_output.weight.normal_(std=0.01)
        mps_model = copy.deepcopy(model).to('mps')
        args = inputs(model.config)
        cpu = model(*args)
        mps = mps_model(*(x.to('mps') for x in args))
        self.assertTrue(torch.allclose(cpu.au_delta, mps.au_delta.cpu(), atol=2e-4, rtol=2e-3))
        self.assertTrue(torch.allclose(cpu.image_delta, mps.image_delta.cpu(), atol=2e-4, rtol=2e-3))
        OBSERVATIONS['cpu_mps_max_au_delta_error'] = float((cpu.au_delta - mps.au_delta.cpu()).abs().max().detach())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AdapterTests))
    report = {'seed': SEED, 'torch': torch.__version__, 'python': sys.version, 'mps_available': torch.backends.mps.is_available(), 'scope': 'synthetic module/gradient checks only; no dataset, pretrained models, or full diffusion training', 'default_config': asdict(AdapterConfig()), 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped), 'seconds': time.monotonic() - started, 'observations': OBSERVATIONS}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'observations'}, ensure_ascii=False, indent=2))
    raise SystemExit(not result.wasSuccessful())
