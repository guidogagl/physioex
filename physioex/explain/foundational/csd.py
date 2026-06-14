"""Conceptual Spectral Decomposition (CSD).

Explains foundation model embeddings by:
1. Using a pluggable specificity strategy to identify which embedding
   dimensions are class-specific (vs generic/shared across classes)
2. Running SpectralGradients independently on each selected dimension
   to produce per-concept time-frequency attribution maps
3. Optionally aggregating concept maps into a single class-level map

The primary output is a **per-concept decomposition**: one (n_bands, T)
time-frequency map for each class-specific embedding dimension, showing
which spectral patterns in the input signal drive that concept.

The secondary output is a **class-level aggregation**: the weighted sum
of all concept maps, answering "why class c?" in time-frequency.

Usage::

    from physioex.explain.foundational import ConceptualSpectralDecomposition
    from physioex.explain.foundational import MarginSpecificity
    from physioex.models.embed import load_probe

    # Load encoder + trained probe
    model, probe_weights = load_probe("cbramod", "hmc", probe_path="...")

    csd = ConceptualSpectralDecomposition(
        model=model, probe_weights=probe_weights,
        fs=200.0, freq_step=4.0,
        specificity=MarginSpecificity(tau=0.5),
    )

    result = csd.explain(x, target_class=3)

    # Per-concept maps (primary result)
    for d, concept in result.concepts.items():
        print(f"dim {d}: W={concept.weight:.3f}, top freq={concept.top_band_hz:.1f} Hz")
        # concept.attribution: (B, n_bands, T)

    # Aggregated class map (secondary)
    # result.class_attribution: (B, n_bands, T) = Σ_d W[c,d] · map_d
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from physioex.explain.foundational.multichannel_sg import MultiChannelSpectralGradients
from physioex.explain.foundational.sleep_bands import FrequencyBand, SLEEP_BANDS
from physioex.explain.foundational.specificity import (
    SpecificityStrategy,
    MarginSpecificity,
)

logger = logging.getLogger("physioex.explain.csd")


@dataclass
class ConceptAttribution:
    """Attribution map for a single embedding dimension (concept)."""

    dim: int  # embedding dimension index
    attribution: Tensor  # (B, n_bands, C, T) raw SG map OR (B, n_bands) if skip_ig=True
    weight: float  # W[c, d] — probe weight for target class
    mask_value: float  # specificity mask value for this dim
    contribution: Tensor  # (B,) = W[c,d] * emb_normed_d per sample
    skip_ig: bool = False  # Whether fast mode (no temporal resolution) was used

    @property
    def weighted_attribution(self) -> Tensor:
        """(B, n_bands, C, T) attribution weighted by probe weight."""
        return self.attribution * self.weight

    @property
    def channel_aggregated(self) -> Tensor:
        """(B, n_bands, T) attribution summed over channels."""
        if self.skip_ig:
            raise AttributeError("channel_aggregated not available in skip_ig mode (no temporal resolution)")
        return self.attribution.sum(dim=2)

    @property
    def band_energy(self) -> Tensor:
        """(n_bands,) total attribution energy per band (mean over B, sum over C and T)."""
        if self.skip_ig:
            # In skip_ig mode, attribution is (B, n_bands) - mean over B directly
            return self.attribution.abs().mean(dim=0)
        return self.attribution.abs().mean(dim=0).sum(dim=(-2, -1))

    @property
    def channel_energy(self) -> Tensor:
        """(C,) total attribution energy per channel (mean over B, sum over bands and T)."""
        if self.skip_ig:
            raise AttributeError("channel_energy not available in skip_ig mode (no channel resolution)")
        return self.attribution.abs().mean(dim=0).sum(dim=(0, -1))

    @property
    def top_band_idx(self) -> int:
        return self.band_energy.argmax().item()


@dataclass
class CSDResult:
    """Result of a CSD explanation."""

    # Per-concept maps (PRIMARY result)
    concepts: Dict[int, ConceptAttribution]  # {dim: ConceptAttribution}

    # Class-level aggregation (SECONDARY — derived from concepts)
    class_attribution: Tensor  # (B, n_bands, T) = Σ_d W[c,d] · map_d

    # Specificity mask
    mask: Tensor  # (D,) or (B, D)
    selected_dims: List[int]  # dimensions selected by the mask

    # Metadata
    band_frequencies: Tensor  # (n_bands,) in Hz
    target_class: int
    specificity_name: str
    probe_weights: Tensor  # (D,) = W[c, :]
    contributions: Tensor  # (B, D) = W[c,:] * emb_normed

    def top_band_hz(self, dim: int) -> float:
        """Return the center frequency of the top band for a concept."""
        idx = self.concepts[dim].top_band_idx
        return self.band_frequencies[idx].item()


class ConceptualSpectralDecomposition(nn.Module):
    """CSD: per-concept time-frequency explanation of foundation model embeddings.

    For each class-specific embedding dimension (selected by the specificity
    strategy), runs SpectralGradients to produce an independent (n_bands, T)
    attribution map. The class-level map is the weighted sum of all concept maps.

    Args:
        model: FoundationEncoder (pure encoder without head).
        probe_weights: Dictionary containing 'ln' (LayerNorm) and 'W' (Linear weights)
            from the trained linear probe.
        fs: Sampling rate of the preprocessed signal (Hz).
        freq_step: SpectralGradients band width (Hz). Default 4.0.
        bands: Custom frequency bands. If None, uniform bands of width freq_step.
        steps: Integration steps per local IG. Default 10. Ignored if skip_ig=True.
        path: SG ablation direction ('marginal', 'cumulative_add', 'cumulative_remove',
            'both_cumulative', 'shapley'). Default 'marginal'.
        specificity: Strategy for selecting class-specific dimensions.
            Default: MarginSpecificity(tau=0.5).
        mask_threshold: Dimensions with mask > threshold get their own map.
            Default 0.5.
        skip_ig: If True, skip IG computation and return fast band importance only.
            Returns (B, n_bands) instead of (B, n_bands, C, T). Default False.
        n_perms: Number of permutations for Shapley path. Default 20.
        device: Torch device. Default: auto.
    """

    def __init__(
        self,
        model: nn.Module,
        probe_weights: dict,
        fs: float,
        freq_step: float = 4.0,
        bands: Optional[List[FrequencyBand]] = None,
        steps: int = 10,
        path: str = "marginal",
        specificity: Optional[SpecificityStrategy] = None,
        mask_threshold: float = 0.5,
        skip_ig: bool = False,
        n_perms: int = 20,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.fs = fs
        self.freq_step = freq_step
        self.bands = bands
        self.steps = steps
        self.path = path
        self.specificity = specificity or MarginSpecificity(tau=0.5)
        self.mask_threshold = mask_threshold
        self.skip_ig = skip_ig
        self.n_perms = n_perms
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # Extract probe weights from dict
        self._ln = probe_weights["ln"]  # LayerNorm
        self._W = probe_weights["W"]  # Linear weight: (n_classes, D)
        self._n_classes = self._W.shape[0]
        self._D = self._W.shape[1]

    def _prepare_signal(self, x: Tensor) -> Tensor:
        """Preprocess signal, keeping multi-channel structure.

        Returns:
            x_prep: (B, C', T') preprocessed signal (always 3D)
        """
        if x.ndim == 4:
            B, L, C, T = x.shape
            x = x.reshape(B * L, C, T)

        x_prep = self.model._preprocess(x)

        # Ensure 3D: (B, C, T)
        if x_prep.ndim == 2:
            x_prep = x_prep.unsqueeze(1)

        return x_prep

    def _estimate_n_bands(self, T: int) -> int:
        freq_res = self.fs / T
        bin_step = max(1, round(self.freq_step / freq_res))
        n_freqs = T // 2 + 1
        return math.ceil(n_freqs / bin_step)

    def _make_dim_scorer(self, dim: int):
        """Create a scoring function for a single embedding dimension.

        The function maps multi-channel signal (B, C, T) -> (B,) scalar.
        Used as the `f` argument to MultiChannelSpectralGradients.
        """

        def f_dim(x_signal):
            # x_signal: (B, C, T) from MultiChannelSG
            emb = self.model._encode(x_signal)  # (B, D)
            return emb[:, dim]  # (B,) scalar

        return f_dim

    def explain(
        self,
        x: Tensor,
        target_class: int,
        embeddings: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        max_concepts: Optional[int] = None,
    ) -> CSDResult:
        """Compute per-concept CSD explanation.

        Args:
            x: (B, C, T) or (B, L, C, T) raw signal.
            target_class: Class index to explain.
            embeddings: (N, D) dataset embeddings for static strategies.
            labels: (N,) labels for static strategies.
            max_concepts: Limit number of concepts to explain (None = all
                above threshold). Selects top by |mask * W[c,d]|.

        Returns:
            CSDResult with per-concept maps and aggregated class map.
        """
        x = x.to(self.device)
        self.model.to(self.device)
        self.model.eval()

        # 1. Preprocess (keeps multi-channel structure)
        x_prep = self._prepare_signal(x)  # (B, C, T)
        B, C_prep, T_prep = x_prep.shape

        # 2. Get embeddings and compute specificity mask
        with torch.no_grad():
            emb = self.model._encode(x_prep)
            emb_normed = self._ln(emb)

        W_c = self._W[target_class].to(self.device)
        contributions = W_c.unsqueeze(0) * emb_normed  # (B, D)

        # Apply LayerNorm to dataset embeddings too, so that static
        # strategies (CohenDSpecificity) operate on the same quantities
        # that the probe sees.  Without this, Cohen's d would be computed
        # on raw embeddings while the probe uses LayerNorm'd embeddings,
        # potentially giving a different specificity ranking.
        embeddings_normed = None
        if embeddings is not None:
            with torch.no_grad():
                embeddings_normed = self._ln(embeddings.to(self.device))

        mask = self.specificity.compute_mask(
            W=self._W.to(self.device),
            embeddings=embeddings_normed,
            labels=labels,
            x=emb_normed,
            target_class=target_class,
        ).to(self.device)

        # 3. Select dimensions above threshold
        if mask.ndim == 2:
            mask_summary = mask.mean(dim=0)  # (D,) average over batch
        else:
            mask_summary = mask

        above_threshold = (mask_summary > self.mask_threshold).nonzero(as_tuple=True)[0]

        # Rank by |mask * W[c,d]| and optionally limit
        importance = (mask_summary * W_c.abs())[above_threshold]
        order = importance.argsort(descending=True)
        selected = above_threshold[order]

        if max_concepts is not None:
            selected = selected[:max_concepts]

        selected_dims = selected.tolist()
        n_selected = len(selected_dims)

        logger.info(
            f"[CSD] class={target_class}, {n_selected} concepts selected "
            f"(threshold={self.mask_threshold}, strategy={self.specificity.name})"
        )

        # 4. Run MultiChannelSG for each selected dimension independently
        concepts: Dict[int, ConceptAttribution] = {}
        band_freqs = None

        for i, d in enumerate(selected_dims):
            f_d = self._make_dim_scorer(d)

            mcsg = MultiChannelSpectralGradients(
                f=f_d,
                fs=self.fs,
                freq_step=self.freq_step,
                bands=self.bands,
                steps=self.steps,
                path=self.path,
                n_perms=self.n_perms,
            )

            attr_d = mcsg(x_prep, skip_ig=self.skip_ig)  # (B, n_bands) if skip_ig, else (B, n_bands, C, T)

            if band_freqs is None:
                band_freqs = mcsg.band_frequencies()

            mask_val = mask_summary[d].item()
            w_val = W_c[d].item()
            contrib_d = contributions[:, d]  # (B,)

            concepts[d] = ConceptAttribution(
                dim=d,
                attribution=attr_d.detach(),
                weight=w_val,
                mask_value=mask_val,
                contribution=contrib_d.detach(),
                skip_ig=self.skip_ig,
            )

            logger.info(
                f"  concept {i+1}/{n_selected}: dim={d}, W={w_val:+.3f}, "
                f"mask={mask_val:.2f}"
            )

        # 5. Aggregate: class map = Σ_d W[c,d] · map_d
        if concepts:
            if self.skip_ig:
                # In skip_ig mode: attribution is (B, n_bands), just weight and sum
                class_attr = torch.zeros(B, next(iter(concepts.values())).attribution.shape[1], device=self.device)
                for d, concept in concepts.items():
                    class_attr = class_attr + concept.weight * concept.attribution
            else:
                # Full mode: attribution is (B, n_bands, C, T)
                class_attr = torch.zeros_like(next(iter(concepts.values())).attribution)
                for d, concept in concepts.items():
                    class_attr = class_attr + concept.weighted_attribution
        else:
            # No concepts selected — return zeros
            n_bands_val = self._estimate_n_bands(T_prep)
            if self.skip_ig:
                class_attr = torch.zeros(B, n_bands_val, device=self.device)
            else:
                class_attr = torch.zeros(B, n_bands_val, C_prep, T_prep, device=self.device)
            band_freqs = torch.zeros(n_bands_val)

        return CSDResult(
            concepts=concepts,
            class_attribution=class_attr,
            mask=mask,
            selected_dims=selected_dims,
            band_frequencies=band_freqs,
            target_class=target_class,
            specificity_name=self.specificity.name,
            probe_weights=W_c,
            contributions=contributions,
        )
