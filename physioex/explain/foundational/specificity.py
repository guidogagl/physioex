"""Pluggable specificity strategies for CSD.

A specificity strategy determines which embedding dimensions are
class-specific (and should be explained) vs generic (and should be
masked out). The strategy is a callable that produces a mask
m_d ∈ [0, 1] for each dimension d.

Built-in strategies:
    - CohenDSpecificity: static, based on effect size across dataset
    - MarginSpecificity: dynamic, per-input class margin
    - SoftmaxSpecificity: dynamic, softmax-normalized contributions
    - TopKSpecificity: hard selection of top-K dimensions
    - NoFilter: no masking (baseline)

Custom strategies: subclass ``SpecificityStrategy`` and implement
``compute_mask()``.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class SpecificityStrategy:
    """Base class for specificity strategies.

    Subclass and implement ``compute_mask()`` to define custom strategies.
    """

    name: str = "base"

    def compute_mask(
        self,
        W: Tensor,
        embeddings: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
        target_class: Optional[int] = None,
    ) -> Tensor:
        """Compute specificity mask.

        Args:
            W: (n_classes, D) linear probe weights.
            embeddings: (N, D) dataset embeddings (for static strategies).
            labels: (N,) class labels (for static strategies).
            x: (B, D) current input embeddings (for dynamic strategies).
            target_class: class index to explain.

        Returns:
            (D,) for static strategies or (B, D) for dynamic strategies.
            Values in [0, 1].
        """
        raise NotImplementedError


class CohenDSpecificity(SpecificityStrategy):
    """Static specificity via Cohen's d effect size.

    For each dimension d, measures the standardized difference between
    the class-conditional mean and the out-of-class mean::

        d_cohen(d, c) = (μ_d|c - μ_d|¬c) / σ_pooled
        mask_d(c) = sigmoid(d_cohen / tau)

    Requires dataset-level embeddings and labels (computed once, reusable).
    """

    name = "cohen_d"

    def __init__(self, tau: float = 1.0):
        self.tau = tau
        self._cache = {}

    def compute_mask(
        self, W, embeddings=None, labels=None, x=None, target_class=None
    ) -> Tensor:
        if embeddings is None or labels is None:
            raise ValueError("CohenDSpecificity requires embeddings and labels")
        if target_class is None:
            raise ValueError("CohenDSpecificity requires target_class")

        cache_key = target_class
        if cache_key in self._cache:
            return self._cache[cache_key]

        c = target_class
        valid = labels >= 0
        emb, lab = embeddings[valid], labels[valid]

        in_class = emb[lab == c]
        out_class = emb[lab != c]

        mu_in = in_class.mean(dim=0)
        mu_out = out_class.mean(dim=0)

        var_in = in_class.var(dim=0)
        var_out = out_class.var(dim=0)
        n_in = in_class.shape[0]
        n_out = out_class.shape[0]

        # Pooled standard deviation
        sigma_pooled = torch.sqrt(
            ((n_in - 1) * var_in + (n_out - 1) * var_out) / (n_in + n_out - 2)
        ).clamp(min=1e-8)

        cohen_d = (mu_in - mu_out) / sigma_pooled
        mask = torch.sigmoid(cohen_d / self.tau)

        self._cache[cache_key] = mask
        return mask


class MarginSpecificity(SpecificityStrategy):
    """Dynamic per-input specificity via class margin.

    For each dimension d and input x, measures how much the contribution
    to the target class exceeds the best competing class::

        margin_d(x, c) = W[c,d]·ĥ_d(x) - max_{c'≠c} W[c',d]·ĥ_d(x)
        mask_d(x, c) = sigmoid(margin / tau).detach()

    Captures input-specific discriminative dimensions (e.g., K-complex
    present in this particular N2 epoch).
    """

    name = "margin"

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    def compute_mask(
        self, W, embeddings=None, labels=None, x=None, target_class=None
    ) -> Tensor:
        if x is None:
            raise ValueError("MarginSpecificity requires x (current embeddings)")
        if target_class is None:
            raise ValueError("MarginSpecificity requires target_class")

        # x: (B, D), W: (n_classes, D)
        # Per-class contributions: (B, n_classes, D)
        contrib = W.unsqueeze(0) * x.unsqueeze(1)  # (B, C, D)

        c = target_class
        target_contrib = contrib[:, c, :]  # (B, D)

        # Best competing class contribution per dimension
        mask_classes = torch.ones(W.shape[0], dtype=torch.bool, device=W.device)
        mask_classes[c] = False
        runner_up = contrib[:, mask_classes, :].max(dim=1).values  # (B, D)

        margin = target_contrib - runner_up  # (B, D)
        return torch.sigmoid(margin / self.tau).detach()


class SoftmaxSpecificity(SpecificityStrategy):
    """Dynamic per-input specificity via softmax normalization.

    For each dimension d, computes the softmax over class contributions::

        v_d(x) = [W[0,d]·ĥ_d(x), ..., W[C-1,d]·ĥ_d(x)]
        mask_d(x, c) = softmax(v_d / tau)[c].detach()

    mask ≈ 1/C for generic dimensions, ≈ 1 for class-specific ones.
    """

    name = "softmax"

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    def compute_mask(
        self, W, embeddings=None, labels=None, x=None, target_class=None
    ) -> Tensor:
        if x is None:
            raise ValueError("SoftmaxSpecificity requires x (current embeddings)")
        if target_class is None:
            raise ValueError("SoftmaxSpecificity requires target_class")

        # Per-dimension class contributions: (B, n_classes, D)
        contrib = W.unsqueeze(0) * x.unsqueeze(1)

        # Softmax over classes for each dimension
        probs = torch.softmax(contrib / self.tau, dim=1)  # (B, C, D)
        return probs[:, target_class, :].detach()  # (B, D)


class TopKSpecificity(SpecificityStrategy):
    """Hard selection of top-K dimensions by absolute contribution.

    Selects the K dimensions with highest |W[c,d] · ĥ_d(x)| for the
    target class. All other dimensions are zeroed out.
    """

    name = "topk"

    def __init__(self, k: int = 10):
        self.k = k

    def compute_mask(
        self, W, embeddings=None, labels=None, x=None, target_class=None
    ) -> Tensor:
        if x is None:
            raise ValueError("TopKSpecificity requires x (current embeddings)")
        if target_class is None:
            raise ValueError("TopKSpecificity requires target_class")

        contrib = W[target_class].unsqueeze(0) * x  # (B, D)
        _, top_idx = contrib.abs().topk(self.k, dim=-1)  # (B, k)

        mask = torch.zeros_like(contrib)
        mask.scatter_(1, top_idx, 1.0)
        return mask.detach()


class NoFilter(SpecificityStrategy):
    """No filtering — all dimensions contribute equally.

    Baseline strategy: SpectralGradients on the raw class logit
    without any specificity masking.
    """

    name = "none"

    def compute_mask(
        self, W, embeddings=None, labels=None, x=None, target_class=None
    ) -> Tensor:
        D = W.shape[1]
        if x is not None:
            return torch.ones(x.shape[0], D, device=x.device)
        return torch.ones(D, device=W.device)
