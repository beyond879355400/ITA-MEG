"""Synthetic AU-supervisor mechanics, not AU accuracy or fitting experiments."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import torch
from au_regressor import AURegressorConfig, AUIntensityRegressor, IMAGENET_RESNET18_SHA256, au_intensity_mse, normalize_au_targets, preprocess_for_au

SEED = 20260914
IMAGENET_PATH = None
OBSERVATIONS = {}
HERE = Path(__file__).resolve().parent


class AURegressorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        (HERE / 'test_tmp').mkdir(exist_ok=True)

    def setUp(self):
        torch.manual_seed(SEED)

    def test_tensor_preprocessing_matches_known_midgray(self):
        value = preprocess_for_au(torch.zeros(1, 3, 24, 40))
        expected = (torch.full((3,), 0.5) - torch.tensor((0.485, 0.456, 0.406))) / torch.tensor((0.229, 0.224, 0.225))
        self.assertEqual(value.shape, (1, 3, 224, 224))
        self.assertTrue(torch.allclose(value[0, :, 50, 50], expected))

    def test_zero_one_and_vae_ranges_have_identical_preprocessing(self):
        image = torch.rand(2, 3, 40, 32)
        from_vae = preprocess_for_au(image * 2 - 1)
        from_dataset = preprocess_for_au(image, input_range='zero_one')
        self.assertTrue(torch.allclose(from_vae, from_dataset, atol=1e-6))

    def test_preprocessing_retains_image_gradient(self):
        image = torch.zeros(1, 3, 20, 20, requires_grad=True)
        preprocess_for_au(image).square().mean().backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.abs().sum()), 0)

    def test_prediction_range_and_raw_scale(self):
        model = AUIntensityRegressor().eval()
        image = torch.zeros(1, 3, 32, 32)
        normalized = model.forward_normalized(image)
        raw = model(image)
        self.assertEqual(raw.shape, (1, 12))
        self.assertTrue(torch.all(raw >= 0) and torch.all(raw <= 5))
        self.assertTrue(torch.equal(raw, normalized * 5))
        OBSERVATIONS['parameter_counts_unfrozen'] = model.parameter_counts()

    def test_label_scaling_and_normalized_mse(self):
        raw_target = torch.full((2, 12), 2)
        prediction = torch.full((2, 12), 4.0, requires_grad=True)
        self.assertTrue(torch.equal(normalize_au_targets(raw_target), torch.full((2, 12), 0.4)))
        loss = au_intensity_mse(prediction, raw_target)
        self.assertAlmostEqual(float(loss.detach()), 0.16, places=6)
        loss.backward()
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_training_loss_reaches_new_au_head(self):
        model = AUIntensityRegressor().eval()
        image = torch.rand(1, 3, 32, 32)
        loss = model.training_loss(image, torch.zeros(1, 12), input_range='zero_one')
        loss.backward()
        self.assertGreater(float(model.head[2].weight.grad.norm()), 0)

    def freeze_probe(self, device):
        model = AUIntensityRegressor().to(device)
        next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
        model.freeze_for_generator_loss()
        # This mimics an outer generator module being put in train mode.
        model.train()
        self.assertFalse(model.training)
        self.assertFalse(model.backbone.bn1.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        before_weight = model.backbone.conv1.weight.detach().clone()
        before_running = model.backbone.bn1.running_mean.detach().clone()
        image = (torch.rand(1, 3, 40, 40) * 0.2 - 0.1).to(device).requires_grad_(True)
        prediction = model(image)
        loss = au_intensity_mse(prediction, torch.zeros(1, 12, device=device))
        loss.backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.detach().norm().cpu()), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        self.assertTrue(torch.equal(before_weight, model.backbone.conv1.weight))
        self.assertTrue(torch.equal(before_running, model.backbone.bn1.running_mean))
        OBSERVATIONS['frozen_' + device] = {'input_gradient_norm': float(image.grad.detach().norm().cpu()), 'parameter_counts': model.parameter_counts(), 'training_mode': model.training}

    def test_frozen_supervisor_preserves_input_gradients_cpu(self):
        self.freeze_probe('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_frozen_supervisor_preserves_input_gradients_mps(self):
        self.freeze_probe('mps')

    def test_safe_state_save_load_prediction_roundtrip(self):
        model = AUIntensityRegressor().eval()
        image = torch.zeros(1, 3, 32, 32)
        expected = model(image).detach()
        with tempfile.TemporaryDirectory(dir=HERE / 'test_tmp') as directory:
            checkpoint = Path(directory) / 'synthetic_untrained_au.pth'
            torch.save({'config': asdict(model.config), 'state_dict': model.state_dict(), 'scope': 'synthetic untrained test only'}, checkpoint)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            restored = AUIntensityRegressor(AURegressorConfig(**payload['config'])).eval()
            restored.load_state_dict(payload['state_dict'], strict=True)
            actual = restored(image).detach()
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(payload['scope'], 'synthetic untrained test only')

    def test_fixed_imagenet_checkpoint_loads_without_au_weights(self):
        if IMAGENET_PATH is None:
            self.skipTest('No explicit local ImageNet checkpoint supplied')
        model = AUIntensityRegressor(imagenet_weights=IMAGENET_PATH)
        self.assertEqual(model.backbone_initialization['sha256'], IMAGENET_RESNET18_SHA256)
        self.assertEqual(model.backbone_initialization['kind'], 'imagenet')
        self.assertEqual(model.backbone_initialization['au_head_initialization'], 'random_requires_new_train_split_fitting')
        self.assertEqual(model.head[0].weight.shape, (128, 512))
        self.assertEqual(model.head[2].weight.shape, (12, 128))
        OBSERVATIONS['imagenet_initialization'] = model.backbone_initialization

    def test_unknown_checkpoint_rejected_before_loading(self):
        with tempfile.TemporaryDirectory(dir=HERE / 'test_tmp') as directory:
            path = Path(directory) / 'not_imagenet.pth'
            path.write_bytes(b'not a model')
            with self.assertRaisesRegex(ValueError, 'fixed ImageNet'):
                AUIntensityRegressor(imagenet_weights=path)

    def test_invalid_input_or_label_shapes_and_ranges(self):
        with self.assertRaisesRegex(ValueError, 'shape'):
            preprocess_for_au(torch.zeros(3, 32, 32))
        with self.assertRaisesRegex(ValueError, 'floating'):
            preprocess_for_au(torch.zeros(1, 3, 32, 32, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, 'intensity range'):
            normalize_au_targets(torch.full((1, 12), 6.0))
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            au_intensity_mse(torch.zeros(1, 12), torch.zeros(1, 11))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-weights', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    IMAGENET_PATH = args.imagenet_weights
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AURegressorTests))
    report = {'seed': SEED, 'torch': torch.__version__, 'python': sys.version, 'mps_available': torch.backends.mps.is_available(), 'scope': 'synthetic mechanics only; no AU training or final evaluation', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped), 'seconds': time.monotonic() - started, 'observations': OBSERVATIONS}
    if args.report:
        with args.report.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(not result.wasSuccessful())
