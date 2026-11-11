"""Trainable bidirectional residual conditioning prototype (not an old checkpoint).

Inputs are spatial image features, a pooled text feature, and 0--5 AU values.
Zero-initialized output heads modify EXISTING FineFace AU/FaceID embeddings by
addition: no extra zero token is appended to an attention softmax denominator.
This file loads no pretrained models and never disables autograd.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn


Direction = Literal['bidirectional', 'condition_reads_image', 'image_reads_condition']


@dataclass(frozen=True)
class AdapterConfig:
    image_dim: int = 512
    text_dim: int = 768
    num_aus: int = 12
    au_hidden_dim: int = 64
    au_feature_dim: int = 128
    hidden_dim: int = 128
    condition_dim: int = 768
    num_heads: int = 4
    num_layers: int = 1
    ff_multiplier: int = 2
    max_patch_tokens: int = 49
    dropout: float = 0.0
    direction: Direction = 'bidirectional'
    self_attention: bool = True
    residual_scale: float = 1.0

    def __post_init__(self):
        for name in ('image_dim', 'text_dim', 'num_aus', 'au_hidden_dim', 'hidden_dim', 'condition_dim', 'num_heads', 'num_layers', 'ff_multiplier', 'max_patch_tokens'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.au_feature_dim <= self.num_aus:
            raise ValueError('au_feature_dim must leave room for the nonlinear AU branch')
        if self.hidden_dim % self.num_heads:
            raise ValueError('hidden_dim must be divisible by num_heads')
        if self.direction not in ('bidirectional', 'condition_reads_image', 'image_reads_condition'):
            raise ValueError('Unknown interaction direction')
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError('dropout must be in [0,1)')
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0:
            raise ValueError('residual_scale must be finite and nonnegative')


@dataclass
class ResidualAdapterOutput:
    au_delta: Tensor                 # [B,1,D_condition]
    image_delta: Tensor              # [B,1,D_condition], broadcast over existing ID tokens
    fusion: Tensor                   # [B,D_hidden]
    image_tokens: Tensor             # [B,P,D_hidden]
    condition_tokens: Tensor         # [B,2,D_hidden]
    image_weights: Tensor            # [B,P], initial patch scores, not anatomical masks
    attention: Dict[str, Tensor]     # last block; diagnostic weights with gradient intact


def _zero_padding(tokens: Tensor, mask: Optional[Tensor]) -> Tensor:
    return tokens if mask is None else tokens.masked_fill(mask.unsqueeze(-1), 0.0)


def _pool(tokens: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    count = (~mask).sum(dim=1, keepdim=True).to(tokens.dtype)
    return _zero_padding(tokens, mask).sum(dim=1) / count


class DirectConcatAUEncoder(nn.Module):
    """Keep y/5 explicitly alongside a learned nonlinear joint AU representation."""
    def __init__(self, num_aus: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.nonlinear = nn.Sequential(nn.Linear(num_aus, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, feature_dim - num_aus), nn.GELU())

    def forward(self, values: Tensor) -> Tensor:
        normalized = values / 5.0
        return torch.cat((normalized, self.nonlinear(normalized)), dim=-1)


class CrossAttention(nn.Module):
    """Separate Q/K/V; query and context sequence lengths need not be equal."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, context: Tensor, *, context_padding: Optional[Tensor] = None, query_padding: Optional[Tensor] = None, return_attention: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        batch, nq, width = query.shape
        nk = context.shape[1]
        q = self.q(query).reshape(batch, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(batch, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(batch, nk, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if context_padding is not None:
            scores = scores.masked_fill(context_padding[:, None, None, :], float('-inf'))
        weights = scores.softmax(dim=-1)
        if query_padding is not None:
            weights = weights.masked_fill(query_padding[:, None, :, None], 0.0)
        output = torch.matmul(self.dropout(weights), v).transpose(1, 2).reshape(batch, nq, width)
        output = _zero_padding(self.output(output), query_padding)
        return output, weights if return_attention else None


class BidirectionalInteractionBlock(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        d = config.hidden_dim
        self.condition_from_image = None if config.direction == 'image_reads_condition' else CrossAttention(d, config.num_heads, config.dropout)
        self.image_from_condition = None if config.direction == 'condition_reads_image' else CrossAttention(d, config.num_heads, config.dropout)
        self.condition_cross_norm = nn.LayerNorm(d)
        self.image_cross_norm = nn.LayerNorm(d)
        self.condition_self = CrossAttention(d, config.num_heads, config.dropout) if config.self_attention else None
        self.image_self = CrossAttention(d, config.num_heads, config.dropout) if config.self_attention else None
        self.condition_self_norm = nn.LayerNorm(d) if config.self_attention else nn.Identity()
        self.image_self_norm = nn.LayerNorm(d) if config.self_attention else nn.Identity()
        def ffn():
            return nn.Sequential(nn.Linear(d, d * config.ff_multiplier), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d * config.ff_multiplier, d))
        self.condition_ffn = ffn()
        self.image_ffn = ffn()
        self.condition_ffn_norm = nn.LayerNorm(d)
        self.image_ffn_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, image: Tensor, condition: Tensor, *, image_padding: Optional[Tensor] = None, return_attention: bool = False):
        # Both directions consume the SAME pre-update I and O (parallel update).
        attention = {}
        condition_message = torch.zeros_like(condition)
        image_message = torch.zeros_like(image)
        if self.condition_from_image is not None:
            condition_message, weights = self.condition_from_image(condition, image, context_padding=image_padding, return_attention=return_attention)
            if weights is not None:
                attention['condition_from_image'] = weights
        if self.image_from_condition is not None:
            image_message, weights = self.image_from_condition(image, condition, query_padding=image_padding, return_attention=return_attention)
            if weights is not None:
                attention['image_from_condition'] = weights
        condition = self.condition_cross_norm(condition + self.dropout(condition_message))
        image = _zero_padding(self.image_cross_norm(image + self.dropout(image_message)), image_padding)
        if self.condition_self is not None:
            message, _ = self.condition_self(condition, condition)
            condition = self.condition_self_norm(condition + self.dropout(message))
            message, _ = self.image_self(image, image, context_padding=image_padding, query_padding=image_padding)
            image = _zero_padding(self.image_self_norm(image + self.dropout(message)), image_padding)
        condition = self.condition_ffn_norm(condition + self.dropout(self.condition_ffn(condition)))
        image = _zero_padding(self.image_ffn_norm(image + self.dropout(self.image_ffn(image))), image_padding)
        return image, condition, attention


class _ResidualConditioningBase(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.image_projection = nn.Linear(config.image_dim, d)
        self.text_projection = nn.Linear(config.text_dim, d)
        self.au_encoder = DirectConcatAUEncoder(config.num_aus, config.au_hidden_dim, config.au_feature_dim)
        self.au_projection = nn.Linear(config.au_feature_dim, d)
        self.image_position = nn.Parameter(torch.empty(1, config.max_patch_tokens, d))
        self.condition_type = nn.Parameter(torch.empty(1, 2, d))
        nn.init.normal_(self.image_position, std=0.02)
        nn.init.normal_(self.condition_type, std=0.02)
        self.image_input_norm = nn.LayerNorm(d)
        self.condition_input_norm = nn.LayerNorm(d)
        self.image_score = nn.Linear(d, 1, bias=False)
        self.fusion_norm = nn.LayerNorm(d)
        self.au_output = nn.Linear(d, config.condition_dim, bias=False)
        self.image_output = nn.Linear(d, config.condition_dim, bias=False)
        nn.init.zeros_(self.au_output.weight)
        nn.init.zeros_(self.image_output.weight)

    def _encode(self, patches: Tensor, text: Tensor, aus: Tensor, image_padding: Optional[Tensor]):
        c = self.config
        if patches.ndim != 3 or patches.shape[-1] != c.image_dim:
            raise ValueError('patches must have shape [B,P,image_dim]')
        batch, count, _ = patches.shape
        if count < 1 or count > c.max_patch_tokens:
            raise ValueError('patch token count is out of configured range')
        if text.shape != (batch, c.text_dim) or aus.shape != (batch, c.num_aus):
            raise ValueError('text or AU shape mismatch')
        if text.device != patches.device or aus.device != patches.device:
            raise ValueError('all input tensors must be on the same device')
        if any(not torch.is_floating_point(x) for x in (patches, text, aus)):
            raise ValueError('inputs must be floating-point features/AU values')
        if not torch.isfinite(aus).all().item() or ((aus < 0) | (aus > 5)).any().item():
            raise ValueError('AU values must be finite and in [0,5]')
        if image_padding is not None:
            if image_padding.shape != (batch, count) or image_padding.dtype != torch.bool or image_padding.device != patches.device:
                raise ValueError('image_padding must be a same-device boolean [B,P] mask')
            if image_padding.all(dim=1).any().item():
                raise ValueError('each example needs at least one valid image patch')
        image = self.image_input_norm(self.image_projection(patches) + self.image_position[:, :count])
        logits = self.image_score(image).squeeze(-1)
        if image_padding is not None:
            logits = logits.masked_fill(image_padding, float('-inf'))
        weights = logits.softmax(dim=-1)
        image = _zero_padding(image * weights.unsqueeze(-1), image_padding)
        text_token = self.text_projection(text)
        au_token = self.au_projection(self.au_encoder(aus))
        condition = self.condition_input_norm(torch.stack((text_token, au_token), dim=1) + self.condition_type)
        return image, condition, weights

    def _result(self, image: Tensor, condition: Tensor, weights: Tensor, fusion: Tensor, attention: Dict[str, Tensor]):
        fusion = self.fusion_norm(fusion)
        scale = self.config.residual_scale
        return ResidualAdapterOutput(self.au_output(fusion).unsqueeze(1) * scale, self.image_output(fusion).unsqueeze(1) * scale, fusion, image, condition, weights, attention)

    def parameter_counts(self) -> Dict[str, int]:
        return {'total': sum(x.numel() for x in self.parameters()), 'trainable': sum(x.numel() for x in self.parameters() if x.requires_grad)}


class CrossModalResidualAdapter(_ResidualConditioningBase):
    def __init__(self, config: AdapterConfig = AdapterConfig()):
        super().__init__(config)
        self.blocks = nn.ModuleList(BidirectionalInteractionBlock(config) for _ in range(config.num_layers))

    def forward(self, image_patches: Tensor, text_pooled: Tensor, au_values: Tensor, *, image_padding: Optional[Tensor] = None, return_attention: bool = False) -> ResidualAdapterOutput:
        image, condition, weights = self._encode(image_patches, text_pooled, au_values, image_padding)
        attention = {}
        for block in self.blocks:
            image, condition, attention = block(image, condition, image_padding=image_padding, return_attention=return_attention)
        fusion = _pool(image, image_padding) + condition.mean(dim=1)
        return self._result(image, condition, weights, fusion, attention)


class PooledMLPResidualAdapter(_ResidualConditioningBase):
    """Same encoders and injection heads; nonlinear pooled fusion without attention."""
    def __init__(self, config: AdapterConfig, mlp_hidden_dim: int):
        super().__init__(config)
        if mlp_hidden_dim < 1:
            raise ValueError('mlp_hidden_dim must be positive')
        d = config.hidden_dim
        self.fusion_mlp = nn.Sequential(nn.Linear(2 * d, mlp_hidden_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(mlp_hidden_dim, d))

    def forward(self, image_patches: Tensor, text_pooled: Tensor, au_values: Tensor, *, image_padding: Optional[Tensor] = None, return_attention: bool = False) -> ResidualAdapterOutput:
        image, condition, weights = self._encode(image_patches, text_pooled, au_values, image_padding)
        fusion = self.fusion_mlp(torch.cat((_pool(image, image_padding), condition.mean(dim=1)), dim=-1))
        return self._result(image, condition, weights, fusion, {})


def suggest_matched_mlp_hidden_dim(reference: CrossModalResidualAdapter) -> int:
    """Nearest width by trainable parameter count; does not instantiate another model."""
    interaction_count = sum(x.numel() for x in reference.blocks.parameters())
    d = reference.config.hidden_dim
    return max(1, round((interaction_count - d) / (3 * d + 1)))


def augment_condition_embeddings(base_au: Tensor, base_image: Tensor, output: ResidualAdapterOutput) -> Dict[str, Tensor]:
    """Add to existing slots, preserving sequence lengths and a zero-init baseline."""
    for name, base, delta in (('au', base_au, output.au_delta), ('image', base_image, output.image_delta)):
        if base.ndim != 3 or base.shape[1] < 1 or delta.shape != (base.shape[0], 1, base.shape[2]):
            raise ValueError(name + ' baseline embedding shape mismatch')
        if base.device != delta.device:
            raise ValueError(name + ' baseline/delta device mismatch')
    return {'au_embedding': base_au + output.au_delta.to(base_au.dtype), 'image_embedding': base_image + output.image_delta.to(base_image.dtype)}
