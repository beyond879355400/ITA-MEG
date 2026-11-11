"""Development v2: spatially distinct residuals on existing 16 IP tokens.

Legacy v1 is imported unchanged. Patch scaling and output readout are separate
factors. All fusion variants share this SAME trainable readout architecture.
No dataset, pretrained model or full training routine is loaded here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
from torch import Tensor, nn

if __package__:
    from .cross_modal_adapter import AdapterConfig, BidirectionalInteractionBlock, _ResidualConditioningBase, _pool, suggest_matched_mlp_hidden_dim as _matched_width
else:
    from cross_modal_adapter import AdapterConfig, BidirectionalInteractionBlock, _ResidualConditioningBase, _pool, suggest_matched_mlp_hidden_dim as _matched_width


@dataclass(frozen=True)
class TokenAdapterConfig(AdapterConfig):
    readout: Literal['global', 'slot'] = 'slot'
    patch_scale: Literal['raw', 'unit_mean'] = 'raw'

    def __post_init__(self):
        super().__post_init__()
        if self.max_patch_tokens != 49:
            raise ValueError('v2 uses exactly 49 tokens in a 7x7 image grid')
        if self.readout not in ('global', 'slot'):
            raise ValueError('Unknown readout factor')
        if self.patch_scale not in ('raw', 'unit_mean'):
            raise ValueError('Unknown patch-scale factor')


@dataclass
class TokenResidualOutput:
    au_delta: Tensor                       # [B,1,D_condition]
    image_delta: Tensor                    # actual injection: [B,1,D] or [B,16,D]
    fusion: Tensor                         # [B,D_hidden]
    image_tokens: Tensor                   # [B,49,D_hidden]
    condition_tokens: Tensor               # [B,2,D_hidden]
    image_weights: Tensor                  # original alpha, sum=1 over valid patches
    image_slots: Tensor                    # [B,16,D_hidden], spatial 7x7 -> 4x4 pooling
    candidate_slot_deltas: Tensor          # [B,16,D], before optional global averaging
    patch_gain: Tensor                     # [B], 1 or count of valid patches
    attention: Dict[str, Tensor]


def spatial_slot_pool(image_tokens: Tensor, image_padding: Optional[Tensor] = None) -> Tensor:
    """7x7 -> 4x4 adaptive average pooling, Tensor-only on CPU and MPS.

    Bin boundaries are floor(i*7/4), ceil((i+1)*7/4), identical to PyTorch
    adaptive_avg_pool2d. Explicit means avoid the older MPS kernel's
    non-divisible input/output restriction while retaining autograd.
    """
    if image_tokens.ndim != 3 or image_tokens.shape[1] != 49:
        raise ValueError('spatial pooling requires [B,49,D]')
    batch, _, width = image_tokens.shape
    grid = image_tokens.reshape(batch, 7, 7, width)
    validity = None
    if image_padding is not None:
        if image_padding.shape != (batch, 49) or image_padding.dtype != torch.bool or image_padding.device != image_tokens.device:
            raise ValueError('padding must be a same-device boolean [B,49] mask')
        validity = (~image_padding).to(image_tokens.dtype).reshape(batch, 7, 7, 1)
    slots = []
    for row in range(4):
        y0, y1 = row * 7 // 4, ((row + 1) * 7 + 3) // 4
        for col in range(4):
            x0, x1 = col * 7 // 4, ((col + 1) * 7 + 3) // 4
            region = grid[:, y0:y1, x0:x1]
            if validity is None:
                pooled = region.mean(dim=(1, 2))
            else:
                valid = validity[:, y0:y1, x0:x1]
                pooled = (region * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1.0)
            slots.append(pooled)
    return torch.stack(slots, dim=1)


class _TokenResidualBase(_ResidualConditioningBase):
    def __init__(self, config: TokenAdapterConfig):
        super().__init__(config)
        d = config.hidden_dim
        self.slot_position = nn.Parameter(torch.empty(1, 16, d))
        nn.init.normal_(self.slot_position, std=0.02)
        self.slot_readout = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d))
        # The inherited image_output is the shared FINAL zero-initialized map.
        # It is used for every slot, not replicated into unused parameter banks.

    def _encode_scaled(self, patches, text, aus, image_padding):
        if patches.ndim != 3 or patches.shape[1] != 49:
            raise ValueError('v2 requires exactly 49 image tokens')
        image, condition, weights = self._encode(patches, text, aus, image_padding)
        gain = image.new_ones((image.shape[0], 1, 1))
        if self.config.patch_scale == 'unit_mean':
            if image_padding is None:
                gain = gain * 49
            else:
                gain = (~image_padding).sum(dim=1).to(image.dtype).view(-1, 1, 1)
        return image * gain, condition, weights, gain[:, 0, 0]

    def _token_result(self, image, condition, weights, gain, fusion, image_padding, attention):
        fusion = self.fusion_norm(fusion)
        slots = spatial_slot_pool(image, image_padding)
        batch = image.shape[0]
        readout_input = torch.cat((slots, fusion.unsqueeze(1).expand(-1, 16, -1), self.slot_position.expand(batch, -1, -1)), dim=-1)
        candidate = self.image_output(self.slot_readout(readout_input)) * self.config.residual_scale
        # Exactly the same parameters participate in both factors. Global mode
        # averages the 16 active candidates; slot mode keeps their differences.
        image_delta = candidate.mean(dim=1, keepdim=True) if self.config.readout == 'global' else candidate
        au_delta = self.au_output(fusion).unsqueeze(1) * self.config.residual_scale
        return TokenResidualOutput(au_delta, image_delta, fusion, image, condition, weights, slots, candidate, gain, attention)


class TokenCrossModalResidualAdapter(_TokenResidualBase):
    def __init__(self, config: TokenAdapterConfig = TokenAdapterConfig()):
        super().__init__(config)
        self.blocks = nn.ModuleList(BidirectionalInteractionBlock(config) for _ in range(config.num_layers))

    def forward(self, image_patches: Tensor, text_pooled: Tensor, au_values: Tensor, *, image_padding: Optional[Tensor] = None, return_attention: bool = False) -> TokenResidualOutput:
        image, condition, weights, gain = self._encode_scaled(image_patches, text_pooled, au_values, image_padding)
        attention = {}
        for block in self.blocks:
            image, condition, attention = block(image, condition, image_padding=image_padding, return_attention=return_attention)
        fusion = _pool(image, image_padding) + condition.mean(dim=1)
        return self._token_result(image, condition, weights, gain, fusion, image_padding, attention)


class TokenPooledMLPResidualAdapter(_TokenResidualBase):
    """No cross-attention fusion; retain raw local I for the shared slot readout."""
    def __init__(self, config: TokenAdapterConfig, mlp_hidden_dim: int):
        super().__init__(config)
        if mlp_hidden_dim <= 0:
            raise ValueError('mlp_hidden_dim must be positive')
        d = config.hidden_dim
        self.fusion_mlp = nn.Sequential(nn.Linear(2 * d, mlp_hidden_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(mlp_hidden_dim, d))

    def forward(self, image_patches: Tensor, text_pooled: Tensor, au_values: Tensor, *, image_padding: Optional[Tensor] = None, return_attention: bool = False) -> TokenResidualOutput:
        image, condition, weights, gain = self._encode_scaled(image_patches, text_pooled, au_values, image_padding)
        fusion = self.fusion_mlp(torch.cat((_pool(image, image_padding), condition.mean(dim=1)), dim=-1))
        return self._token_result(image, condition, weights, gain, fusion, image_padding, {})


def suggest_matched_mlp_hidden_dim(reference: TokenCrossModalResidualAdapter) -> int:
    return _matched_width(reference)


def augment_condition_embeddings(base_au: Tensor, base_image: Tensor, output: TokenResidualOutput) -> Dict[str, Tensor]:
    """Accept global or slot residuals, requiring the original ONE/16 token layout."""
    if base_au.ndim != 3 or base_au.shape[1] != 1:
        raise ValueError('base AU must have exactly one token')
    if base_image.ndim != 3 or base_image.shape[1] != 16:
        raise ValueError('base FaceID must have exactly 16 tokens')
    batch, _, width = base_au.shape
    if base_image.shape != (batch, 16, width):
        raise ValueError('base AU/FaceID batch or width mismatch')
    if output.au_delta.shape != (batch, 1, width):
        raise ValueError('AU residual shape mismatch')
    if output.image_delta.shape not in ((batch, 1, width), (batch, 16, width)):
        raise ValueError('image residual must have one or 16 tokens')
    if base_au.device != output.au_delta.device or base_image.device != output.image_delta.device or base_au.device != base_image.device:
        raise ValueError('base conditions and residuals must share device')
    return {'au_embedding': base_au + output.au_delta.to(base_au.dtype), 'image_embedding': base_image + output.image_delta.to(base_image.dtype)}
