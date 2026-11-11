"""Development v3: preserve explicit AU entries and optionally bound residual RMS.

Frozen v1/v2 sources are not modified. Trust region is OFF by default. Active
training configurations require positive explicit rho values; the low-level
rho=0 branch is an intentional diagnostic bypass, not a trainable zero gate.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn

if __package__:
    from .token_residual_adapter import TokenAdapterConfig, TokenCrossModalResidualAdapter, TokenPooledMLPResidualAdapter, TokenResidualOutput, suggest_matched_mlp_hidden_dim
else:
    from token_residual_adapter import TokenAdapterConfig, TokenCrossModalResidualAdapter, TokenPooledMLPResidualAdapter, TokenResidualOutput, suggest_matched_mlp_hidden_dim


@dataclass(frozen=True)
class RelativeRMSConfig:
    enabled: bool = False
    rho_au: Optional[float] = None
    rho_ip: Optional[float] = None
    reference_rms_floor: float = 1e-6

    def __post_init__(self):
        if not math.isfinite(self.reference_rms_floor) or self.reference_rms_floor <= 0:
            raise ValueError('reference_rms_floor must be positive and finite')
        for name in ('rho_au', 'rho_ip'):
            rho = getattr(self, name)
            if self.enabled and rho is None:
                raise ValueError('Enabled trust region requires explicit positive ' + name)
            if rho is not None and (not math.isfinite(rho) or rho <= 0):
                raise ValueError(name + ' must be positive; rho=0 is diagnostic-only in the low-level limiter')


@dataclass
class ProtectedTokenResidualOutput(TokenResidualOutput):
    explicit_au_dims: int


class _ProtectExplicitAUHead:
    def _install_protected_au_head(self):
        learned = self.config.condition_dim - self.config.num_aus
        if learned <= 0:
            raise ValueError('Condition width must exceed explicit AU dimensions')
        # The discarded inherited head is unregistered, not masked or retained.
        self.au_output = nn.Linear(self.config.hidden_dim, learned, bias=False)
        nn.init.zeros_(self.au_output.weight)

    def _token_result(self, *args, **kwargs):
        result = super()._token_result(*args, **kwargs)
        learned = result.au_delta
        zeros = learned.new_zeros((learned.shape[0], 1, self.config.num_aus))
        payload = dict(vars(result))
        payload['au_delta'] = torch.cat((zeros, learned), dim=-1)
        return ProtectedTokenResidualOutput(**payload, explicit_au_dims=self.config.num_aus)


class SafeTokenCrossModalResidualAdapter(_ProtectExplicitAUHead, TokenCrossModalResidualAdapter):
    def __init__(self, config: TokenAdapterConfig = TokenAdapterConfig()):
        super().__init__(config)
        self._install_protected_au_head()


class SafeTokenPooledMLPResidualAdapter(_ProtectExplicitAUHead, TokenPooledMLPResidualAdapter):
    def __init__(self, config: TokenAdapterConfig, mlp_hidden_dim: int):
        super().__init__(config, mlp_hidden_dim)
        self._install_protected_au_head()


def relative_rms_trust_region(raw_delta: Tensor, reference: Tensor, rho: float, *, reference_rms_floor: float = 1e-6) -> Tensor:
    """Bound each residual slot using one shared per-example reference radius.

    Reference RMS is over ALL its slots and channels. In particular, the IP
    reference is [B,16,D] even when raw_delta is [B,1,D]; it never creates
    artificial slot differences from different reference-token norms.

    delta = r*u / sqrt(r^2 + mean_channels(u^2)), r=rho*max(RMS(reference),floor).
    At u=0 and r>0 the derivative is one. Do NOT implement this as sqrt(mean
    u^2) followed by squaring, which introduces a singular intermediate at zero.
    """
    if raw_delta.ndim != 3 or reference.ndim != 3 or raw_delta.shape[0] != reference.shape[0] or raw_delta.shape[-1] != reference.shape[-1]:
        raise ValueError('Residual/reference must share [B,*,D] batch and channel dimensions')
    if raw_delta.shape[1] < 1 or reference.shape[1] < 1 or raw_delta.shape[-1] < 1:
        raise ValueError('Residual/reference dimensions must be nonempty')
    if raw_delta.device != reference.device or not torch.is_floating_point(raw_delta) or not torch.is_floating_point(reference):
        raise ValueError('Residual/reference must be floating tensors on the same device')
    if not math.isfinite(rho) or rho < 0:
        raise ValueError('rho must be finite and nonnegative')
    if rho == 0:
        # Explicit bypass avoids 0/0 and even 0*NaN; no trainable gate is created.
        return torch.zeros_like(raw_delta)
    if not math.isfinite(reference_rms_floor) or reference_rms_floor <= 0:
        raise ValueError('reference_rms_floor must be finite and positive')
    calculation_dtype = torch.float64 if raw_delta.dtype == torch.float64 else torch.float32
    raw = raw_delta.to(calculation_dtype)
    anchor = reference.detach().to(calculation_dtype)
    if not torch.isfinite(raw).all().item() or not torch.isfinite(anchor).all().item():
        raise ValueError('Nonfinite residual or reference')
    anchor_rms = anchor.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(reference_rms_floor)
    radius = anchor_rms * rho
    if not torch.isfinite(radius).all().item() or (radius < math.sqrt(torch.finfo(calculation_dtype).tiny)).any().item():
        raise ValueError('Trust radius is not representable; choose a finite positive rho/floor')
    mean_square = raw.square().mean(dim=-1, keepdim=True)
    if not torch.isfinite(mean_square).all().item():
        raise ValueError('Residual square overflowed; abort instead of producing a nonfinite update')
    denominator = (radius.square() + mean_square).sqrt()
    bounded = raw * (radius / denominator)
    return bounded.to(raw_delta.dtype)


def condition_residuals(base_au: Tensor, base_image: Tensor, output: ProtectedTokenResidualOutput, *, trust_region: RelativeRMSConfig = RelativeRMSConfig()) -> Dict[str, Tensor]:
    if not isinstance(output, ProtectedTokenResidualOutput):
        raise ValueError('v3 augmentation requires its protected-AU output, not a v1/v2 output')
    if base_au.ndim != 3 or base_au.shape[1] != 1 or base_image.ndim != 3 or base_image.shape[1] != 16:
        raise ValueError('Base conditions require exactly one AU token and 16 IP tokens')
    batch, _, width = base_au.shape
    explicit = output.explicit_au_dims
    if not 0 < explicit < width:
        raise ValueError('Explicit AU prefix is invalid')
    if base_image.shape != (batch, 16, width) or output.au_delta.shape != (batch, 1, width) or output.image_delta.shape not in ((batch, 1, width), (batch, 16, width)):
        raise ValueError('Condition/residual shape mismatch')
    if any(t.device != base_au.device for t in (base_image, output.au_delta, output.image_delta)):
        raise ValueError('All condition tensors must share device')
    if torch.count_nonzero(output.au_delta[..., :explicit]).item() != 0:
        raise ValueError('Protected AU prefix must be exactly zero')
    tail = output.au_delta[..., explicit:]
    image_delta = output.image_delta
    if trust_region.enabled:
        tail = relative_rms_trust_region(tail, base_au[..., explicit:], trust_region.rho_au, reference_rms_floor=trust_region.reference_rms_floor)
        image_delta = relative_rms_trust_region(image_delta, base_image, trust_region.rho_ip, reference_rms_floor=trust_region.reference_rms_floor)
    return {'au_delta': torch.cat((tail.new_zeros((batch, 1, explicit)), tail), dim=-1), 'image_delta': image_delta}


def augment_condition_embeddings(base_au: Tensor, base_image: Tensor, output: ProtectedTokenResidualOutput, *, trust_region: RelativeRMSConfig = RelativeRMSConfig()) -> Dict[str, Tensor]:
    effective = condition_residuals(base_au, base_image, output, trust_region=trust_region)
    explicit = output.explicit_au_dims
    # Copy the original prefix directly instead of re-computing its values.
    au = torch.cat((base_au[..., :explicit], base_au[..., explicit:] + effective['au_delta'][..., explicit:].to(base_au.dtype)), dim=-1)
    return {'au_embedding': au, 'image_embedding': base_image + effective['image_delta'].to(base_image.dtype)}
