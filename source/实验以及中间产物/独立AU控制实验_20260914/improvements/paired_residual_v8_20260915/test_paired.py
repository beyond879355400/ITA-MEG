"""CPU-only analytic tests of the actual V8 helpers. No weights/images/model init."""
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import torch
from diffusers import DEISMultistepScheduler
import run_paired as runner


class PairedFormulaTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2) / 8
        self.noise = torch.randn(self.source.shape, generator=torch.Generator().manual_seed(101))
        self.mask = torch.tensor([[[[0., .25], [.75, 1.]]]])
        self.scheduler = DEISMultistepScheduler(beta_start=.00085, beta_end=.012, beta_schedule='scaled_linear',
                                                prediction_type='epsilon', thresholding=False,
                                                timestep_spacing='linspace', use_karras_sigmas=False)

    def test_actual15_grid_and_explicit_clean_endpoint(self):
        levels = runner.time_levels(self.scheduler)
        self.assertEqual([r['t'] for r in levels], [749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50])
        for row, next_row in zip(levels, levels[1:]):
            self.assertEqual(row['next_rho'], next_row['rho'])
            self.assertEqual(row['next_t'], next_row['t'])
        self.assertEqual((levels[-1]['next_rho'], levels[-1]['next_alpha'], levels[-1]['next_sigma']), (0., 1., 0.))
        self.assertIsNone(levels[-1]['next_t'])
        self.assertGreater(float(self.scheduler.sigmas[-1]), 0.)
        self.assertIsNone(self.scheduler.step_index)
        self.assertEqual(self.scheduler.model_outputs, [None, None])

    def test_actual_noisy_source_equals_installed_add_noise(self):
        for row in runner.time_levels(self.scheduler):
            xs, xt = runner.source_target_inputs(self.source, self.noise, torch.zeros_like(self.source), row, torch)
            expected = self.scheduler.add_noise(self.source, self.noise, torch.tensor([row['t']]))
            torch.testing.assert_close(xs, expected, atol=0, rtol=0)
            self.assertTrue(torch.equal(xs, xt))

    def test_target_uses_alpha_times_clean_residual(self):
        row = runner.time_levels(self.scheduler)[0]
        d = self.source + .3
        xs, xt = runner.source_target_inputs(self.source, self.noise, d, row, torch)
        torch.testing.assert_close(xt, xs + row['alpha'] * d, rtol=0, atol=0)

    def test_constant_difference_telescopes_with_correct_sign(self):
        levels = runner.time_levels(self.scheduler)
        delta = torch.ones_like(self.source) * .3
        d = torch.zeros_like(self.source)
        for row in levels:
            d = runner.residual_step(d, delta, self.mask, row['rho'], row['next_rho'], .5, torch)
        expected = -levels[0]['rho'] * .5 * self.mask * delta
        torch.testing.assert_close(d, expected, atol=2e-7, rtol=2e-6)
        self.assertTrue((d <= 0).all())

    def test_mask_applied_once_without_previous_residual_decay(self):
        d = torch.ones_like(self.source)
        d[:, :, 0, 0] = 0
        delta = torch.ones_like(d) * 2
        actual = runner.residual_step(d, delta, self.mask, 2., 1., .25, torch)
        expected = d - .5 * self.mask
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertNotEqual(float(actual[0, 0, 0, 1]), float((self.mask * d - .5 * self.mask.square())[0, 0, 0, 1]))

    def test_all_one_mask_noedit_is_exact_without_forcing_empty_mask(self):
        d = torch.zeros_like(self.source)
        for row in runner.time_levels(self.scheduler):
            xs, xt = runner.source_target_inputs(self.source, self.noise, d, row, torch)
            es, et = xs.square() + 3, xt.square() + 3  # deterministic identical fake denoiser
            self.assertTrue(torch.equal(es, et))
            d = runner.residual_step(d, et - es, torch.ones_like(self.mask), row['rho'], row['next_rho'], 1., torch)
            self.assertEqual(torch.count_nonzero(d).item(), 0)
        self.assertTrue(torch.equal(self.source + d, self.source))

    def test_zero_gain_and_zero_mask(self):
        for mask, gain in ((self.mask, 0.), (torch.zeros_like(self.mask), 1.)):
            d = torch.zeros_like(self.source)
            for row in runner.time_levels(self.scheduler):
                d = runner.residual_step(d, torch.ones_like(d), mask, row['rho'], row['next_rho'], gain, torch)
            self.assertTrue(torch.equal(self.source + d, self.source))

    def test_equivalent_composite_epsilon_ddim_step(self):
        levels = runner.time_levels(self.scheduler)
        for row in (levels[0], levels[5], levels[-1]):
            d = self.source * .01
            d[:, :, 0, 0] = 0
            delta = self.noise * .03
            _, xt = runner.source_target_inputs(self.source, self.noise, d, row, torch)
            composite = self.noise + .5 * self.mask * delta
            expected = row['next_alpha'] / row['alpha'] * xt + (row['next_sigma'] - row['next_alpha'] / row['alpha'] * row['sigma']) * composite
            next_d = runner.residual_step(d, delta, self.mask, row['rho'], row['next_rho'], .5, torch)
            actual = row['next_alpha'] * (self.source + next_d) + row['next_sigma'] * self.noise
            torch.testing.assert_close(actual, expected, atol=4e-7, rtol=2e-6)

    def test_nonfinite_inputs_and_invalid_steps_are_rejected(self):
        d = torch.zeros_like(self.source)
        for which in ('delta', 'mask'):
            delta, mask = d.clone(), self.mask.clone()
            (delta if which == 'delta' else mask).flatten()[0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                runner.residual_step(d, delta, mask, 1., 0., 1., torch)
        for rho, next_rho in ((1., 1.), (0., 1.), (1., -1.), (float('inf'), 0.)):
            with self.assertRaises(ValueError):
                runner.residual_step(d, d, self.mask, rho, next_rho, 1., torch)

    def test_no_input_mutation(self):
        d, delta = torch.zeros_like(self.source), torch.ones_like(self.source)
        before = [value.clone() for value in (d, delta, self.mask)]
        runner.residual_step(d, delta, self.mask, 1., 0., 1., torch)
        for original, now in zip(before, (d, delta, self.mask)):
            self.assertTrue(torch.equal(original, now))

    def test_cfg_branches_use_each_latent_and_same_unconditional_order(self):
        calls = []
        def fake_unet(latent, t, **kwargs):
            calls.append((latent.clone(), t.clone(), kwargs))
            return (latent + kwargs['encoder_hidden_states'][:, :1, :1].reshape(2, 1, 1, 1),)
        text = torch.tensor([[[2.]], [[5.]]])
        cross = {'au_embedding': torch.zeros(2, 1, 3), 'image_embedding': torch.ones(2, 2, 3)}
        first = runner.cfg_prediction(fake_unet, self.source, torch.tensor(749), text, cross, torch)
        second = runner.cfg_prediction(fake_unet, self.source + 1, torch.tensor(749), text, cross, torch)
        torch.testing.assert_close(first, self.source + 2 + runner.CFG * 3, rtol=0, atol=0)
        torch.testing.assert_close(second - first, torch.ones_like(self.source), rtol=0, atol=0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0].shape[0], 2)
        self.assertIs(calls[0][2]['cross_attention_kwargs'], cross)

    def test_source_cache_key_is_gain_method_independent_but_complete(self):
        row = {'source_latent_sha256': 'a', 'noise_sha256': 'b', 'source_condition_hashes': {'au': 'c', 'ip': 'd'},
               'source_text_tokens_sha256': 'e', 'method': 'baseline', 'gain': .25}
        key = runner.source_cache_key(row, 749)
        other = copy.deepcopy(row)
        other.update(method='bidirectional', gain=1.)
        self.assertEqual(key, runner.source_cache_key(other, 749))
        for field in ('source_latent_sha256', 'noise_sha256', 'source_text_tokens_sha256'):
            other = copy.deepcopy(row)
            other[field] += 'changed'
            self.assertNotEqual(key, runner.source_cache_key(other, 749))
        other = copy.deepcopy(row)
        other['source_condition_hashes']['ip'] += 'changed'
        self.assertNotEqual(key, runner.source_cache_key(other, 749))
        self.assertNotEqual(key, runner.source_cache_key(row, 699))

    def test_actual_runner_has_no_scheduler_step_or_pipeline_generation(self):
        tree = ast.parse(Path(runner.__file__).read_text())
        forbidden = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
                     isinstance(node.func, ast.Attribute) and node.func.attr in ('step', 'backward')]
        self.assertEqual(forbidden, [])
        execute_source = ast.get_source_segment(Path(runner.__file__).read_text(),
            next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute'))
        self.assertIn("row['target_unet_calls'] += 1", execute_source)
        self.assertIn("need(zeroedit and torch.count_nonzero(d).item() == 0", execute_source)
        self.assertIn("output_sha == case['vae_reconstruction']['sha256']", execute_source)
        self.assertNotIn("['vae'].encode(", execute_source)


if __name__ == '__main__':
    unittest.main()
