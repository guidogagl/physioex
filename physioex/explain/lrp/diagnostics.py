"""Conservation diagnostics for LRP.

Relevance conservation ``Σ_i R_i ≈ f_c(x)`` is *the* defining property of LRP.
It holds exactly (up to ε) for bias-free networks; biases (and any block that
is not on an LRP rule) *absorb* part of the relevance, so the ratio ``Σ R / f``
drops below one.  :class:`ConservationReport` makes that visible per sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConservationReport:
    """Per-sample conservation summary.

    Attributes:
        target: the explained logit ``f_c(x)`` per sample, ``(B,)``.
        relevance_sum: ``Σ R`` over all input elements per sample, ``(B,)``.
        ratio: ``Σ R / f`` (1.0 = exact conservation).
        absorbed: ``f − Σ R`` — relevance absorbed by biases / unruled ops.
    """

    target: torch.Tensor
    relevance_sum: torch.Tensor
    ratio: torch.Tensor
    absorbed: torch.Tensor

    @classmethod
    def from_relevance(cls, target: torch.Tensor, relevance: torch.Tensor) -> "ConservationReport":
        target = target.detach().reshape(-1)
        rsum = relevance.detach().reshape(target.shape[0], -1).sum(dim=1)
        safe = torch.where(target.abs() > 1e-12, target, torch.ones_like(target))
        return cls(target=target, relevance_sum=rsum, ratio=rsum / safe, absorbed=target - rsum)

    def is_conserved(self, rtol: float = 1e-3, atol: float = 1e-5) -> bool:
        return bool(torch.allclose(self.relevance_sum, self.target, rtol=rtol, atol=atol))

    def __str__(self) -> str:  # pragma: no cover - formatting
        r = self.ratio.tolist()
        return "ConservationReport(ratio Σ R / f per sample = " + ", ".join(f"{v:.4f}" for v in r) + ")"


def check_conservation(explainer, x: torch.Tensor) -> ConservationReport:
    """Run ``explainer`` (an :class:`~physioex.explain.lrp.model.ModelLRP` or
    :class:`~physioex.explain.lrp.attributor.LRP`) on ``x`` and report."""
    _, report = explainer(x, return_report=True)
    return report
