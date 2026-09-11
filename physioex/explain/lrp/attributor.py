"""Zennit-based LRP attribution for feed-forward / CNN PhysioEx models.

For architectures made of ``Conv``/``Linear``/pooling/activations (Tsinalis,
Chambon2018, …) Zennit's composites give full rule configurability (ε, γ, w²,
z-box, flat, αβ) with BatchNorm canonization.  Models containing recurrent,
attention or transformer blocks must use
:class:`~physioex.explain.lrp.model.ModelLRP` instead (Zennit cannot see inside
fused ``nn.LSTM``/``nn.MultiheadAttention``).

The target neuron is seeded with its **logit value** so ``Σ R ≈ f_c(x)``
(minus what biases absorb).  ``forward(x)`` returns relevance shaped like ``x``.

Requires the ``explain`` extra: ``pip install "physioex[explain]"``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from physioex.explain.lrp._functional import target_seed
from physioex.explain.lrp.diagnostics import ConservationReport


class LRP(nn.Module):
    """LRP attribution over a trained feed-forward PhysioEx model (Zennit).

    Args:
        model: trained model, ``(B, L, C, T) -> (B, L, n_classes)`` or
            ``(B, n_classes)``.  Evaluated in ``eval()`` mode.
        in_index: sequence epoch to explain (central-epoch models emit ``L=1``).
        out_index: class-logit index to explain.
        composite: a Zennit composite; default
            :func:`~physioex.explain.lrp.composites.physioex_composite`
            (ε dense / γ conv / w² first layer) with the BatchNorm canonizer.
        canonizers: used only when ``composite`` is ``None``.

    Example:
        >>> relevance = LRP(model, out_index=2)(x)
        >>> relevance, report = LRP(model, out_index=2)(x, return_report=True)
    """

    def __init__(self, model, in_index=0, out_index=0, composite=None, canonizers=None):
        super().__init__()
        self.model = model
        self.in_index = in_index
        self.out_index = out_index
        if composite is None:
            from physioex.explain.lrp.canonizers import default_canonizers
            from physioex.explain.lrp.composites import physioex_composite

            if canonizers is None:
                canonizers = default_canonizers(model)
            composite = physioex_composite(canonizers=canonizers)
        self.composite = composite

    def forward(self, x: torch.Tensor, return_report: bool = False):
        from zennit.attribution import Gradient

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                if x.dim() == 3:  # (L, C, T) -> (1, L, C, T)
                    x = x.unsqueeze(0)
                x = x.detach().requires_grad_(True)
                # hook-free forward: sizes the seed and materialises LazyLinear
                # modules before Zennit hooks them
                with torch.no_grad():
                    out = self.model(x)
                seed, target = target_seed(out, self.in_index, self.out_index)
                with Gradient(model=self.model, composite=self.composite) as attributor:
                    _, relevance = attributor(x, seed)
        finally:
            self.model.train(was_training)
        if return_report:
            return relevance, ConservationReport.from_relevance(target, relevance)
        return relevance
