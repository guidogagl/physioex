"""Layer-wise Relevance Propagation (LRP) attribution for PhysioEx models.

Unlike the gradient methods in :mod:`physioex.explain.posthoc` (which wrap the
model into a scalar function ``f`` via :class:`~physioex.explain.posthoc.\
functionizer.SeqFunct`), LRP needs the **layered ``nn.Module`` graph** — it
attaches a rule to every module and redistributes relevance through the
backward pass.  So :class:`LRP` wraps the model directly and selects the target
neuron with a one-hot seed on the model output, mirroring Zennit's
``attributor(input, torch.eye(n)[[class]])`` convention.

PhysioEx models map ``(B, L, C, T) -> (B, L, n_classes)`` (sequence-to-sequence;
central-epoch models like Tsinalis / Chambon2018 emit ``L=1``).  ``forward(x)``
returns relevance with the **same shape as x**.

Requires the ``explain`` extra: ``pip install "physioex[explain]"``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LRP(nn.Module):
    """LRP attribution over a trained PhysioEx model.

    Args:
        model: trained model, ``(B, L, C, T) -> (B, L, n_classes)``.
        in_index: index of the sequence epoch to explain (default ``0``;
            central-epoch models have ``L=1`` so ``0`` is the only option).
        out_index: class-logit index to explain (the target class).
        composite: a Zennit composite mapping module types to LRP rules.  If
            ``None``, :func:`physioex_composite` (ε dense / γ conv / w² first)
            with the default BatchNorm canonizer is built automatically.
        canonizers: canonizer list, used only when ``composite is None``
            (defaults to
            :func:`physioex.explain.lrp.canonizers.default_canonizers`).

    Example:
        >>> from physioex.explain.lrp import LRP
        >>> expl = LRP(model, out_index=2)          # explain class 2
        >>> relevance = expl(x)                       # (B, L, C, T)
    """

    def __init__(
        self,
        model: nn.Module,
        in_index: int = 0,
        out_index: int = 0,
        composite=None,
        canonizers=None,
    ):
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

    def _target_seed(self, out: torch.Tensor) -> torch.Tensor:
        """Relevance seed selecting the target neuron.

        Seeds the *output relevance* with the target **logit value**
        ``f_c(x) = out[:, in_index, out_index]`` (zero elsewhere), so total
        relevance conserves to the class evidence: ``Σ R ≈ f_c(x)`` — the
        defining LRP property.  (Seeding with a bare ``1.0`` one-hot would
        instead normalise the relevance to sum to 1.)
        """
        seed = torch.zeros_like(out)
        if out.dim() == 3:  # (B, L, n_classes)
            seed[:, self.in_index, self.out_index] = out[
                :, self.in_index, self.out_index
            ].detach()
        elif out.dim() == 2:  # (B, n_classes)
            seed[:, self.out_index] = out[:, self.out_index].detach()
        else:
            raise ValueError(
                f"Unexpected model output rank {out.dim()}; expected 2 or 3."
            )
        return seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from zennit.attribution import Gradient

        if x.dim() == 3:  # (L, C, T) -> (1, L, C, T)
            x = x.unsqueeze(0)
        x = x.detach().requires_grad_(True)

        # A hook-free forward to (a) learn the output shape for the one-hot
        # seed and (b) materialise any LazyLinear modules before Zennit hooks
        # them.  Cheap relative to the backward-pass attribution.
        with torch.no_grad():
            out = self.model(x)
        seed = self._target_seed(out)

        with Gradient(model=self.model, composite=self.composite) as attributor:
            _, relevance = attributor(x, seed)

        return relevance
