"""CPU tensor-only tests; no model initialization, weights, or real images.

Run with the existing torch/diffusers environment and PYTHONDONTWRITEBYTECODE=1.
The callback is extracted from the actual runner AST so its indexing logic, not
a reimplementation, is exercised with a tiny fake scheduler/pipeline.
"""

import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True  # Importing the v5 helper must not write into v5.

import torch

import run_region_diagnostic as runner

try:
    from diffusers import DEISMultistepScheduler, StableDiffusionImg2ImgPipeline
except ImportError:
    DEISMultistepScheduler = StableDiffusionImg2ImgPipeline = None


class FakeScheduler:
    order = 1

    def __init__(self):
        self.timesteps = torch.arange(999, 0, -50, dtype=torch.int64)
        self.calls = []

    def add_noise(self, source, noise, timestep):
        self.calls.append((source.clone(), noise.clone(), timestep.clone()))
        # Deliberately distinct levels make a current-versus-next error obvious.
        return source + noise * timestep.to(source.dtype).reshape(-1, 1, 1, 1) / 1000


def actual_callback(*, source, noise, mask, history, strength):
    tree = ast.parse(Path(runner.__file__).read_text())
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "step_end"]
    if len(matches) != 1:
        raise AssertionError("Expected one callback definition in the actual runner")
    module = ast.Module(body=[copy.deepcopy(matches[0])], type_ignores=[])
    namespace = {"torch": torch, "need": runner.need, "project_latents": runner.project_latents,
                 "v5": runner.v5, "src": source, "noise": noise, "mask": mask,
                 "history": history, "strength": strength}
    exec(compile(ast.fix_missing_locations(module), str(runner.__file__), "exec"), namespace)
    return namespace["step_end"]


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2) / 8
        self.predicted = -self.source - 1
        self.noise = torch.randn(self.source.shape, generator=torch.Generator().manual_seed(101))
        self.scheduler = FakeScheduler()

    def project(self, mask, next_t):
        return runner.project_latents(self.predicted, self.source, self.noise, mask,
                                      self.scheduler, next_t, torch)

    def test_mask_zero_at_intermediate_step_is_noisy_source_at_next_t(self):
        next_t = torch.tensor(349)
        actual = self.project(torch.zeros(1, 1, 2, 2), next_t)
        expected = self.source + self.noise * .349
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(self.scheduler.calls[-1][2].tolist(), [349])

    def test_mask_zero_final_is_exact_clean_source_and_does_not_add_noise(self):
        actual = self.project(torch.zeros(1, 1, 2, 2), None)
        self.assertTrue(torch.equal(actual, self.source))
        self.assertEqual(self.scheduler.calls, [])

    def test_mask_one_preserves_predicted_exactly_at_both_step_types(self):
        for timestep in (torch.tensor(349), None):
            actual = self.project(torch.ones(1, 1, 2, 2), timestep)
            self.assertTrue(torch.equal(actual, self.predicted))

    def test_soft_mixed_mask_broadcasts_channels_and_is_exact_formula(self):
        mask = torch.tensor([[[[0., .25], [.75, 1.]]]])
        for timestep in (torch.tensor(349), None):
            actual = self.project(mask, timestep)
            reference = self.source if timestep is None else self.source + self.noise * .349
            expected = mask * self.predicted + (1 - mask) * reference
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.shape, self.source.shape)
            self.assertTrue(torch.equal(actual[:, :, 0, 0], reference[:, :, 0, 0]))
            self.assertTrue(torch.equal(actual[:, :, 1, 1], self.predicted[:, :, 1, 1]))

    def test_projection_does_not_mutate_input_tensors(self):
        originals = [tensor.clone() for tensor in (self.predicted, self.source, self.noise)]
        self.project(torch.full((1, 1, 2, 2), .5), torch.tensor(349))
        for original, actual in zip(originals, (self.predicted, self.source, self.noise)):
            self.assertTrue(torch.equal(original, actual))

    def test_nonfinite_projected_latent_rejected(self):
        self.predicted[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite region projection"):
            self.project(torch.ones(1, 1, 2, 2), None)

    def test_actual_callback_uses_next_not_current_timestep(self):
        for strength in (.55, .75):
            history = []
            callback = actual_callback(source=self.source, noise=self.noise,
                                       mask=torch.zeros(1, 1, 2, 2), history=history, strength=strength)
            pipeline = SimpleNamespace(scheduler=FakeScheduler())
            timesteps = pipeline.scheduler.timesteps[20 - int(20 * strength):]
            current = timesteps[0]
            result = callback(pipeline, 0, current, {"latents": self.predicted})["latents"]
            expected = pipeline.scheduler.add_noise(self.source, self.noise, timesteps[1].reshape(1))
            wrong = pipeline.scheduler.add_noise(self.source, self.noise, current.reshape(1))
            self.assertTrue(torch.equal(result, expected))
            self.assertFalse(torch.equal(result, wrong))
            self.assertEqual(history[0]["t"], int(current))
            self.assertEqual(history[0]["next_t"], int(timesteps[1]))

    def test_actual_no_edit_callback_entire_chain_finishes_exact_source(self):
        for strength in (.55, .75):
            history = []
            pipeline = SimpleNamespace(scheduler=FakeScheduler())
            callback = actual_callback(source=self.source, noise=self.noise,
                                       mask=torch.zeros(1, 1, 2, 2), history=history, strength=strength)
            timesteps = pipeline.scheduler.timesteps[20 - int(20 * strength):]
            for index, timestep in enumerate(timesteps):
                # Stand in for arbitrary finite post-scheduler denoising results.
                predicted = self.predicted + index
                result = callback(pipeline, index, timestep, {"latents": predicted})["latents"]
            self.assertTrue(torch.equal(result, self.source))
            self.assertEqual(len(history), int(20 * strength))
            self.assertIsNone(history[-1]["next_t"])
            self.assertEqual(len(pipeline.scheduler.calls), len(timesteps) - 1)

    def test_actual_callback_rejects_wrong_timestep_or_nonfinite_predicted(self):
        callback = actual_callback(source=self.source, noise=self.noise,
                                   mask=torch.zeros(1, 1, 2, 2), history=[], strength=.55)
        pipeline = SimpleNamespace(scheduler=self.scheduler)
        with self.assertRaisesRegex(ValueError, "Callback index mismatch"):
            callback(pipeline, 0, torch.tensor(999), {"latents": self.predicted})
        bad = self.predicted.clone()
        bad[0, 0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "Nonfinite denoising latent"):
            callback(pipeline, 0, self.scheduler.timesteps[9], {"latents": bad})

    def test_fixed_v5_source_and_metadata_hashes_unchanged(self):
        pins = {runner.V5 / "run_diagnostic.py": "6296388a3ffeafd6cc413589f0f498966e113c8c0e183e36e30017fce96edd89",
                runner.V5RUN / "cases.json": "7daad59d5540d7b49243cb03da38f7aaed5d2c4294c8cb46ac7ce58b46d43145",
                runner.V5RUN / "result.json": "4e92651f47abadfefdf0be2a91600544c1142cbbd99ebab411b2bea0b22177ce"}
        for path, sha in pins.items():
            self.assertEqual(runner.sha(path), sha)

    @unittest.skipUnless(StableDiffusionImg2ImgPipeline is not None, "diffusers optional")
    def test_installed_prepare_latents_uses_same_cpu_noise_with_four_channels(self):
        fake = SimpleNamespace(scheduler=self.scheduler)
        timestep = torch.tensor([349])
        actual = StableDiffusionImg2ImgPipeline.prepare_latents(
            fake, self.source, timestep, 1, 1, torch.float32, torch.device("cpu"),
            generator=torch.Generator(device="cpu").manual_seed(101))
        self.assertTrue(torch.equal(self.scheduler.calls[-1][1], self.noise))
        expected = self.source + self.noise * .349
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(DEISMultistepScheduler is not None, "diffusers optional")
    def test_installed_deis_next_level_and_multistep_state_reset(self):
        scheduler = DEISMultistepScheduler()
        for strength in (.55, .75):
            scheduler.set_timesteps(20, device="cpu")
            self.assertEqual(scheduler.order, 1)
            self.assertIsNone(scheduler.step_index)
            self.assertEqual(scheduler.lower_order_nums, 0)
            timesteps = scheduler.timesteps[20 - int(20 * strength):]
            latent = scheduler.add_noise(self.source, self.noise, timesteps[:1])
            for index, timestep in enumerate(timesteps):
                predicted = scheduler.step(torch.zeros_like(latent), timestep, latent, return_dict=False)[0]
                next_t = timesteps[index + 1] if index + 1 < len(timesteps) else None
                latent = runner.project_latents(predicted, self.source, self.noise,
                                                torch.zeros(1, 1, 2, 2), scheduler, next_t, torch)
                expected = self.source if next_t is None else scheduler.add_noise(
                    self.source, self.noise, next_t.reshape(1))
                self.assertTrue(torch.equal(latent, expected))
            self.assertTrue(torch.equal(latent, self.source))


if __name__ == "__main__":
    unittest.main()
