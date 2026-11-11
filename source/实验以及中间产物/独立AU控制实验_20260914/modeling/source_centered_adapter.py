"""V4 candidate: shared source-centered raw residuals, with zero new parameters.

This helper does not train, load weights, alter a v3 adapter, apply a trust
region, or change the public absolute AU/FaceID conditioning. Pass its result
to the existing v3 augmentation ONCE, using the TARGET base AU reference.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

if __package__:
    from .safe_token_residual_adapter import ProtectedTokenResidualOutput, SafeTokenCrossModalResidualAdapter, SafeTokenPooledMLPResidualAdapter
else:
    from safe_token_residual_adapter import ProtectedTokenResidualOutput, SafeTokenCrossModalResidualAdapter, SafeTokenPooledMLPResidualAdapter


@dataclass
class SourceCenteredResidualOutput(ProtectedTokenResidualOutput):
    # Keep both actual graphs for diagnostics; neither branch is detached.
    # Inherited hidden features/attention describe the TARGET branch, not
    # differences of representations. candidate_slot_deltas IS differenced.
    target_output: ProtectedTokenResidualOutput
    source_output: ProtectedTokenResidualOutput


def forward_source_centered(
    adapter,
    image_patches: Tensor,
    target_text_pooled: Tensor,
    target_aus: Tensor,
    source_text_pooled: Tensor,
    source_aus: Tensor,
    *,
    image_padding: Optional[Tensor] = None,
    return_attention: bool = False,
) -> SourceCenteredResidualOutput:
    """F_theta(I, target) - F_theta(I, source), before the existing single TR.

    Complete identical conditions give zero added residual, i.e. the public
    baseline conditions, NOT reconstruction of the source image. Equal AU
    values alone are not a no-edit gate when text differs. Known nonzero source
    AUs are supported by this mathematical interface; synthetic tests are not
    empirical validation on real nonzero-source images.
    """
    if not isinstance(adapter, (SafeTokenCrossModalResidualAdapter, SafeTokenPooledMLPResidualAdapter)):
        raise TypeError('Source centering requires an existing v3 protected-AU adapter')
    if adapter.config.dropout != 0 or any(isinstance(module, nn.Dropout) and module.p != 0 for module in adapter.modules()):
        raise ValueError('Source centering requires dropout=0, including eval-mode adapters')
    if target_text_pooled.shape != source_text_pooled.shape or target_aus.shape != source_aus.shape:
        raise ValueError('Target and source condition shapes must match')
    target = adapter(image_patches, target_text_pooled, target_aus, image_padding=image_padding, return_attention=return_attention)
    source = adapter(image_patches, source_text_pooled, source_aus, image_padding=image_padding, return_attention=return_attention)
    if target.explicit_au_dims != source.explicit_au_dims:
        raise ValueError('Source/target AU protection differs')
    explicit = target.explicit_au_dims
    if torch.count_nonzero(target.au_delta[..., :explicit]).item() or torch.count_nonzero(source.au_delta[..., :explicit]).item():
        raise ValueError('The v3 explicit AU prefix must be exactly protected')
    payload = dict(vars(target))
    payload.update(au_delta=target.au_delta-source.au_delta,
                   image_delta=target.image_delta-source.image_delta,
                   candidate_slot_deltas=target.candidate_slot_deltas-source.candidate_slot_deltas)
    return SourceCenteredResidualOutput(**payload, target_output=target, source_output=source)
