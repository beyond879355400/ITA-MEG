"""Small CPU tensors only: terminal algebra, masking, CFG, hooks, and derivation."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('terminal_under_test', HERE / 'run_terminal.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TinyScheduler:
    def __init__(self):
        self.config = SimpleNamespace(prediction_type='epsilon', thresholding=False)
        self.sigmas = torch.tensor([.3, .02916716])
        self.model_outputs = [torch.ones(1), torch.zeros(1)]
        self.step_index = 20

    def _sigma_to_alpha_sigma_t(self, rho):
        alpha = 1 / (rho ** 2 + 1) ** .5
        return alpha, rho * alpha

    def scale_model_input(self, sample, t):
        return sample

    def step(self, *args, **kwargs):
        raise AssertionError('Must not advance scheduler during cleanup')

    def add_noise(self, *args, **kwargs):
        raise AssertionError('Must not call add_noise(t0)')


class TinyUNet(torch.nn.Module):
    def forward(self, value, t, encoder_hidden_states, timestep_cond, cross_attention_kwargs,
                added_cond_kwargs, return_dict=False):
        # Make eps(t0) visibly different from eps(t50), with distinct CFG halves.
        result = torch.zeros_like(value)
        result[0] = int(t) / 100 + 1
        result[1] = int(t) / 100 + 2
        return (result,)


def fixture(mode, mask_value=1.):
    src = torch.full((1, 4, 8, 8), .25)
    mask = torch.full((1, 1, 8, 8), mask_value)
    cross = {'au_scale': 1.5, 'ip_scale': .9}
    text = torch.arange(12.).reshape(2, 2, 3)
    unet = TinyUNet()
    pipe = SimpleNamespace(unet=unet, scheduler=TinyScheduler(), do_classifier_free_guidance=True,
                           guidance_scale=7.5, cross_attention_kwargs=cross)
    control = runner.TerminalController(unet, mode, src, mask, cross, torch)
    for t in runner.TIMESTEPS:
        unet(torch.cat([src, src]), t, encoder_hidden_states=text, timestep_cond=None,
             cross_attention_kwargs=cross, added_cond_kwargs=None, return_dict=False)
    return src, mask, text, pipe, control


class TerminalTests(unittest.TestCase):
    def test_nonzero_terminal_coefficients_and_exact_algebra(self):
        source = torch.arange(24.).reshape(1, 4, 2, 3) / 10
        epsilon = torch.full_like(source, .7)
        alpha, sigma, rho = runner.terminal_coefficients(TinyScheduler(), source, torch)
        latent = alpha * source + sigma * epsilon
        clean = runner.estimate_clean(latent, epsilon, alpha, sigma, torch)
        self.assertGreater(float(sigma), 0)
        self.assertGreater(float(rho), 0)
        self.assertTrue(torch.allclose(clean, source, atol=3e-7, rtol=1e-6))

    def test_last_epsilon_uses_complete_cfg_once(self):
        ref = torch.zeros(1, 4, 2, 2)
        raw = torch.cat([torch.ones_like(ref), torch.ones_like(ref) * 3])
        self.assertTrue(torch.equal(runner.cfg_epsilon(raw, ref, torch), torch.full_like(ref, 16.)))
        with self.assertRaisesRegex(ValueError, 'batch/shape'):
            runner.cfg_epsilon(raw[:1], ref, torch)

    def test_shape_and_nonfinite_rejection(self):
        src = torch.zeros(1, 4, 8, 8)
        for bad in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                runner.estimate_clean(src + bad, src, torch.tensor(1.), torch.tensor(.03), torch)
            with self.assertRaises(ValueError):
                runner.filtered_clean_residual(src, src, torch.ones(1, 1, 8, 8) * bad, torch)
        with self.assertRaises(ValueError):
            runner.estimate_clean(src, src[:, :1], torch.tensor(1.), torch.tensor(.03), torch)
        with self.assertRaises(ValueError):
            runner.estimate_clean(src, src, torch.tensor(0.), torch.tensor(.03), torch)
        with self.assertRaises(ValueError):
            runner.filtered_clean_residual(src, src, torch.ones_like(src), torch)

    def test_zero_mask_returns_exact_source(self):
        for mode in runner.MODES:
            src, mask, text, pipe, control = fixture(mode, 0.)
            try:
                clean = control.correct(src + .1, pipe, text)
                projected = runner.core['project_latents'](clean, src, torch.ones_like(src), mask, pipe.scheduler, None, torch)
                control.finalize_projection(projected)
                control.assert_complete()
                self.assertTrue(torch.equal(projected, src))
            finally:
                control.close()

    def test_constant_delta_is_preserved_with_soft_mask_normalization(self):
        src = torch.zeros(1, 4, 8, 8)
        mask = torch.linspace(.1, .9, 64).reshape(1, 1, 8, 8)
        filtered = runner.filtered_clean_residual(src + 3., src, mask, torch)
        self.assertTrue(torch.allclose(filtered, src + 3., atol=1e-6))
        zero = runner.filtered_clean_residual(src + 3., src, mask * 0, torch)
        self.assertTrue(torch.equal(zero, src))

    def test_filter_does_not_import_values_outside_mask(self):
        src = torch.zeros(1, 4, 8, 8)
        mask = torch.zeros(1, 1, 8, 8); mask[:, :, 2:6, 2:6] = .4
        clean = torch.where(mask > 0, torch.ones_like(src) * 2, torch.ones_like(src) * 1000)
        filtered = runner.filtered_clean_residual(clean, src, mask, torch)
        self.assertTrue(torch.equal(filtered[mask.expand_as(src) == 0], src[mask.expand_as(src) == 0]))
        self.assertTrue(torch.allclose(filtered[mask.expand_as(src) > 0], torch.full((64,), 2.), atol=1e-6))

    def test_extra_hook_cannot_overwrite_t50_and_call_counts_reset(self):
        for _ in range(2):
            for mode in runner.MODES:
                src, mask, text, pipe, control = fixture(mode)
                try:
                    last = control.last_raw.clone()
                    snapshot = copy.deepcopy(pipe.scheduler.__dict__)
                    clean = control.correct(src + .1, pipe, text)
                    control.finalize_projection(clean)
                    control.assert_complete()
                    record = control.record()
                    extra = int(mode != 'terminal_last_eps')
                    self.assertEqual(record['denoising_steps'], 15)
                    self.assertEqual(record['terminal_unet_calls'], extra)
                    self.assertEqual(record['total_unet_calls'], 15 + extra)
                    self.assertTrue(torch.equal(control.last_raw, last))
                    self.assertTrue(torch.equal(pipe.scheduler.sigmas, snapshot['sigmas']))
                    self.assertEqual(pipe.scheduler.step_index, 20)
                    if extra:
                        self.assertFalse(torch.equal(control.terminal_raw, last))
                        self.assertEqual(control.calls[-1]['t'], 0)
                        self.assertEqual(control.calls[-1]['prompt_embeds_sha256'], control.calls[-2]['prompt_embeds_sha256'])
                finally:
                    control.close()
                self.assertEqual(len(pipe.unet._forward_hooks), 0)

    def test_mismatched_prompt_or_extra_condition_rejected_and_hook_removed(self):
        for mutate in ('text', 'condition'):
            src, mask, text, pipe, control = fixture('terminal_t0')
            try:
                if mutate == 'text':
                    text = text + 1
                else:
                    control.last_kwargs['added_cond_kwargs'] = {'unexpected': True}
                with self.assertRaises(ValueError):
                    control.correct(src, pipe, text)
            finally:
                control.close()
            self.assertEqual(len(pipe.unet._forward_hooks), 0)

    def test_actual_derived_callback_cleans_before_last_source_projection(self):
        tree = ast.parse(runner.source)
        callback = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'step_end')
        for mode in runner.MODES:
            for mask_value in (0., .3, 1.):
                src, mask, text, pipe, old_control = fixture(mode, mask_value)
                old_control.close()
                control = runner.TerminalController(pipe.unet, mode, src, mask, pipe.cross_attention_kwargs, torch)
                pipe.scheduler.timesteps = torch.tensor([999, 949, 899, 849, 799] + runner.TIMESTEPS)
                next_noise_calls = []
                def add_noise(source, noise, t):
                    next_noise_calls.append(int(t[0]))
                    self.assertGreater(int(t[0]), 0)
                    return source + noise * t[0] / 1000
                pipe.scheduler.add_noise = add_noise
                history = []
                namespace = dict(torch=torch, need=runner.need, strength=.75, controller=control, src=src,
                    mask=mask, noise=torch.ones_like(src), history=history,
                    project_latents=runner.core['project_latents'], v5=SimpleNamespace(tensor_sha=runner.tensor_sha))
                exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(callback)], type_ignores=[])),
                             '<actual-derived-v10-callback>', 'exec'), namespace)
                latent = src.clone()
                try:
                    for i, t in enumerate(runner.TIMESTEPS):
                        raw = pipe.unet(torch.cat([latent, latent]), t, encoder_hidden_states=text, timestep_cond=None,
                                       cross_attention_kwargs=pipe.cross_attention_kwargs, added_cond_kwargs=None)[0]
                        predicted = latent - runner.cfg_epsilon(raw, latent, torch) * .001
                        before = runner.tensor_sha(predicted)
                        latent = namespace['step_end'](pipe, i, t, {'latents': predicted, 'prompt_embeds': text})['latents']
                    control.assert_complete()
                    self.assertEqual(len(history), 15)
                    self.assertEqual(next_noise_calls, runner.TIMESTEPS[1:])
                    self.assertEqual(control.terminal_trace['input_latent_sha256'], before)
                    self.assertEqual(control.terminal_trace['projected_latent_sha256'], history[-1]['latent_sha256'])
                    self.assertEqual(history[-1]['next_t'], None)
                    if mask_value == 0:
                        self.assertTrue(torch.equal(latent, src))
                finally:
                    control.close()

    def test_extra_forward_nonfinite_failure_does_not_leave_hook_or_phase(self):
        src, mask, text, pipe, control = fixture('terminal_t0')
        original_forward = pipe.unet.forward
        def bad_forward(value, t, **kwargs):
            return (torch.full_like(value, float('nan')),) if int(t) == 0 else original_forward(value, t, **kwargs)
        pipe.unet.forward = bad_forward
        try:
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                control.correct(src, pipe, text)
            self.assertEqual(control.phase, 'deis')
            self.assertIsNone(control.terminal_raw)
        finally:
            control.close()
        self.assertEqual(len(pipe.unet._forward_hooks), 0)

    def test_frozen_projection_and_cli_derivation(self):
        original, derived = ast.parse(runner.raw_kernel), ast.parse(runner.source)
        get = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'project_latents')
        self.assertEqual(ast.dump(get(original)), ast.dump(get(derived)))
        self.assertEqual(runner.core['CONFIGS'], [(n, .75, 1.5) for n in runner.MODES])
        self.assertIn("callback_on_step_end_tensor_inputs=['latents','prompt_embeds']", runner.source)
        self.assertIn('finally:\n                        controller.close()', runner.source)
        self.assertIn("if c['group']=='no_edit': need(row['sha256']==c['vae_reconstruction']['sha256']", runner.source)
        self.assertNotIn('scheduler.step(', Path(runner.__file__).read_text())
        self.assertNotIn('add_noise(', Path(runner.__file__).read_text())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
