"""New train-only AU intensity supervisor: ImageNet ResNet18 + 512/128/12 head.

The head is randomly initialized and is NOT an AU evaluator until fitted on
the new training subjects and checked on development subjects. This module
does not load any all-DISFA AU model or an independent final evaluator.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Literal, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import resnet18


InputRange = Literal['minus_one_one', 'zero_one']
IMAGENET_RESNET18_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'


@dataclass(frozen=True)
class AURegressorConfig:
    num_aus: int = 12
    hidden_dim: int = 128
    image_size: int = 224
    intensity_scale: float = 5.0

    def __post_init__(self):
        if min(self.num_aus, self.hidden_dim, self.image_size) <= 0:
            raise ValueError('AU count, hidden dimension and image size must be positive')
        if not math.isfinite(self.intensity_scale) or self.intensity_scale <= 0:
            raise ValueError('intensity_scale must be finite and positive')


def preprocess_for_au(images: Tensor, *, input_range: InputRange = 'minus_one_one', image_size: int = 224) -> Tensor:
    """Tensor-only differentiable image preparation, including resize backward.

    VAE outputs use minus_one_one; decoded dataset images use zero_one.
    Clamp gives zero gradients outside the valid pixel interval by definition.
    Bilinear resize uses align_corners=False and antialias=False for the pinned
    CPU/MPS runtime; training and generator-loss paths must use this same rule.
    """
    if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] < 1 or min(images.shape[-2:]) < 1:
        raise ValueError('images must have shape [B,3,H,W] with nonempty dimensions')
    if not torch.is_floating_point(images):
        raise ValueError('images must be floating point, not uint8/PIL inputs')
    if image_size <= 0:
        raise ValueError('image_size must be positive')
    if input_range == 'minus_one_one':
        pixels = (images + 1.0) / 2.0
    elif input_range == 'zero_one':
        pixels = images
    else:
        raise ValueError('Unknown input_range')
    pixels = pixels.clamp(0.0, 1.0)
    pixels = F.interpolate(pixels, size=(image_size, image_size), mode='bilinear', align_corners=False, antialias=False)
    mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (pixels - mean) / std


def normalize_au_targets(raw_targets: Tensor, intensity_scale: float = 5.0) -> Tensor:
    if not isinstance(raw_targets, Tensor) or raw_targets.ndim != 2:
        raise ValueError('raw AU labels must be a [B,n_AU] tensor')
    if not math.isfinite(intensity_scale) or intensity_scale <= 0:
        raise ValueError('intensity_scale must be finite and positive')
    targets = raw_targets if torch.is_floating_point(raw_targets) else raw_targets.float()
    if not torch.isfinite(targets).all().item() or ((targets < 0) | (targets > intensity_scale)).any().item():
        raise ValueError('raw AU labels must be finite and inside the raw intensity range')
    return targets / intensity_scale


def au_intensity_mse(predicted_raw: Tensor, target_raw: Tensor, *, intensity_scale: float = 5.0, reduction: str = 'mean') -> Tensor:
    """Normalized MSE between two explicitly RAW 0--5 intensity tensors.

    Its magnitude is raw MSE / 25 for the default scale. Do not feed already
    normalized predictions or labels here; loss weights must record this scale.
    """
    if predicted_raw.shape != target_raw.shape or predicted_raw.ndim != 2:
        raise ValueError('prediction/label shape mismatch')
    normalized_target = normalize_au_targets(target_raw.to(device=predicted_raw.device, dtype=predicted_raw.dtype), intensity_scale)
    return F.mse_loss(predicted_raw / intensity_scale, normalized_target, reduction=reduction)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class AUIntensityRegressor(nn.Module):
    def __init__(self, config: AURegressorConfig = AURegressorConfig(), *, imagenet_weights: Optional[str | Path] = None):
        super().__init__()
        self.config = config
        self._frozen_for_generator_loss = False
        self.backbone = resnet18(weights=None)  # never downloads implicitly
        self.backbone_initialization = {'kind': 'random', 'au_head_initialization': 'random_requires_new_train_split_fitting'}
        if imagenet_weights is not None:
            path = Path(imagenet_weights).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = _sha256_file(path)
            if actual != IMAGENET_RESNET18_SHA256:
                raise ValueError('Expected the fixed ImageNet ResNet18 checkpoint, not an AU-trained model')
            state = torch.load(path, map_location='cpu', weights_only=True)
            if not isinstance(state, dict) or 'fc.weight' not in state or state['fc.weight'].shape != (1000, 512):
                raise ValueError('Invalid ImageNet classifier checkpoint structure')
            self.backbone.load_state_dict(state, strict=True)
            self.backbone_initialization = {'kind': 'imagenet', 'path': str(path), 'sha256': actual, 'au_head_initialization': 'random_requires_new_train_split_fitting'}
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(512, config.hidden_dim), nn.ReLU(inplace=False), nn.Linear(config.hidden_dim, config.num_aus), nn.Sigmoid())

    def forward_normalized(self, images: Tensor, *, input_range: InputRange = 'minus_one_one') -> Tensor:
        normalized_images = preprocess_for_au(images, input_range=input_range, image_size=self.config.image_size)
        return self.head(self.backbone(normalized_images))

    def forward(self, images: Tensor, *, input_range: InputRange = 'minus_one_one') -> Tensor:
        return self.forward_normalized(images, input_range=input_range) * self.config.intensity_scale

    def training_loss(self, images: Tensor, raw_labels: Tensor, *, input_range: InputRange = 'zero_one') -> Tensor:
        prediction = self(images, input_range=input_range)
        return au_intensity_mse(prediction, raw_labels, intensity_scale=self.config.intensity_scale)

    def freeze_for_generator_loss(self) -> 'AUIntensityRegressor':
        """Freeze weights and BN state while retaining the image-input graph."""
        self._frozen_for_generator_loss = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None  # remove stale gradients from AU-supervisor fitting
        self.train(False)
        return self

    def train(self, mode: bool = True):
        # A parent generator.train() must not resume BN updates in this frozen
        # supervisor. This does not alter global grad mode or detach any input.
        return super().train(False if self._frozen_for_generator_loss else mode)

    def parameter_counts(self):
        return {'total': sum(x.numel() for x in self.parameters()), 'trainable': sum(x.numel() for x in self.parameters() if x.requires_grad)}
