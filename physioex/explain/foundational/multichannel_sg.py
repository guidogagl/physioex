"""Multi-channel SpectralGradients with sleep-band ablation paths.

Extends SpectralGradients to handle multi-channel input (B, C, T) by
ablating the same frequency band across ALL channels simultaneously.

Supports five ablation path modes:

- **marginal**: Remove one band at a time from the full signal.
  Each band is evaluated independently. No completeness, no order dependence.
- **cumulative_add**: Add bands one at a time from silence to full signal.
  Completeness guaranteed. Order = band list order.
- **cumulative_remove**: Remove bands one at a time from full signal to silence.
  Completeness guaranteed. Order = reverse of band list.
- **both_cumulative**: Average of cumulative_add and cumulative_remove.
  Reduces order dependence.
- **shapley**: Average cumulative_add + cumulative_remove over K random
  permutations of the band order. Completeness + order-independence.
  Most principled, highest compute cost.

Output: (B, n_bands, C, T) — per-band, per-channel, per-sample attribution.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from physioex.explain.foundational.sleep_bands import (
    FrequencyBand,
    bands_to_bin_ranges,
    band_center_frequencies,
)

VALID_PATHS = (
    "marginal",
    "cumulative_add",
    "cumulative_remove",
    "both_cumulative",
    "shapley",
)


class MultiChannelSpectralGradients(nn.Module):
    """SpectralGradients for multi-channel signals with sleep-band ablation.

    Args:
        f: Scoring function ``(B, C, T) -> (B,)``.
        fs: Sampling rate (Hz).
        bands: List of ``FrequencyBand`` objects defining the ablation bands.
        freq_step: Uniform band width (Hz). Used only if ``bands`` is None.
        steps: IG integration steps per local IG.
        path: Ablation path mode (see module docstring).
        n_perms: Number of random permutations for ``"shapley"`` path.
    """

    def __init__(
        self,
        f: callable,
        fs: float = 1.0,
        bands: Optional[List[FrequencyBand]] = None,
        freq_step: float = 1.0,
        steps: int = 10,
        path: str = "marginal",
        n_perms: int = 20,
    ):
        super().__init__()
        if path not in VALID_PATHS:
            raise ValueError(f"path must be one of {VALID_PATHS}, got '{path}'")
        self.f = f
        self.fs = float(fs)
        self.freq_step = float(freq_step)
        self.custom_bands = bands
        self.steps = steps
        self.path = path
        self.n_perms = n_perms
        self._last_signal_length = None

    # ── Band helpers ─────────────────────────────────────────────────

    def _get_bin_ranges(self, T: int) -> List[Tuple[str, int, int]]:
        if self.custom_bands is not None:
            return bands_to_bin_ranges(self.custom_bands, self.fs, T)
        freq_res = self.fs / T
        bin_step = max(1, round(self.freq_step / freq_res))
        n_freqs = T // 2 + 1
        return [
            (
                f"band_{(s + min(s + bin_step, n_freqs)) / 2 * freq_res:.1f}Hz",
                s,
                min(s + bin_step, n_freqs),
            )
            for s in range(0, n_freqs, bin_step)
        ]

    def n_bands(self, T: int = None) -> int:
        if T is None:
            T = self._last_signal_length
        return len(self._get_bin_ranges(T))

    def band_frequencies(self, T: int = None) -> Tensor:
        if self.custom_bands is not None:
            return band_center_frequencies(self.custom_bands)
        if T is None:
            T = self._last_signal_length
        if T is None:
            raise ValueError("Call forward() first or pass T explicitly")
        freq_res = self.fs / T
        ranges = self._get_bin_ranges(T)
        return torch.tensor([(s + e) / 2 * freq_res for _, s, e in ranges])

    def band_names(self) -> List[str]:
        if self._last_signal_length is None and self.custom_bands is None:
            return []
        ranges = self._get_bin_ranges(self._last_signal_length or 1000)
        return [name for name, _, _ in ranges]

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, C, T), got {x.ndim}D")

        B, C, T = x.shape
        self._last_signal_length = T

        xdft = torch.fft.rfft(x, dim=-1)  # (B, C, n_freqs)
        bin_ranges = self._get_bin_ranges(T)

        if self.path == "marginal":
            return self._forward_marginal(xdft, T, bin_ranges)
        elif self.path == "cumulative_add":
            return self._forward_cumulative(xdft, T, bin_ranges, "add")
        elif self.path == "cumulative_remove":
            return self._forward_cumulative(xdft, T, bin_ranges, "remove")
        elif self.path == "both_cumulative":
            a = self._forward_cumulative(xdft, T, bin_ranges, "add")
            r = self._forward_cumulative(xdft, T, bin_ranges, "remove")
            return 0.5 * (a + r)
        elif self.path == "shapley":
            return self._forward_shapley(xdft, T, bin_ranges)

    # ── Marginal: remove one band at a time from full ────────────────

    def _forward_marginal(self, xdft, T, bin_ranges):
        B, C = xdft.shape[0], xdft.shape[1]
        n_bands = len(bin_ranges)
        x_full = torch.fft.irfft(xdft, n=T, dim=-1)  # (B, C, T)

        all_prev = []  # ablated signals (baseline)
        for name, bs, be in bin_ranges:
            xdft_ablated = xdft.clone()
            xdft_ablated[..., bs:be] = 0.0
            all_prev.append(torch.fft.irfft(xdft_ablated, n=T, dim=-1))

        # all_prev: (n_bands, B, C, T)
        all_prev = torch.stack(all_prev)
        # all_curr = x_full repeated
        all_curr = x_full.unsqueeze(0).expand(n_bands, -1, -1, -1)
        delta = all_curr - all_prev  # (n_bands, B, C, T)

        return self._integrate(all_prev, delta, n_bands, B, C, T)

    # ── Cumulative: add or remove bands in order ─────────────────────

    def _forward_cumulative(self, xdft, T, bin_ranges, mode):
        B, C = xdft.shape[0], xdft.shape[1]
        n_bands = len(bin_ranges)

        all_prev = []
        all_curr = []
        band_order = list(range(n_bands))  # natural order

        if mode == "add":
            xdft_cumul = torch.zeros_like(xdft)
            for i in band_order:
                name, bs, be = bin_ranges[i]
                x_prev = torch.fft.irfft(xdft_cumul, n=T, dim=-1)
                xdft_cumul = xdft_cumul.clone()
                xdft_cumul[..., bs:be] = xdft[..., bs:be]
                x_curr = torch.fft.irfft(xdft_cumul, n=T, dim=-1)
                all_prev.append(x_prev)
                all_curr.append(x_curr)
        else:  # remove
            xdft_cumul = xdft.clone()
            for i in reversed(band_order):
                name, bs, be = bin_ranges[i]
                x_curr = torch.fft.irfft(xdft_cumul, n=T, dim=-1)
                xdft_cumul = xdft_cumul.clone()
                xdft_cumul[..., bs:be] = 0.0
                x_prev = torch.fft.irfft(xdft_cumul, n=T, dim=-1)
                all_prev.insert(0, x_prev)  # maintain band order
                all_curr.insert(0, x_curr)

        all_prev = torch.stack(all_prev)
        all_curr = torch.stack(all_curr)
        delta = all_curr - all_prev

        return self._integrate(all_prev, delta, n_bands, B, C, T)

    # ── Shapley: random permutations of cumulative both ──────────────

    def _forward_shapley(self, xdft, T, bin_ranges):
        B, C = xdft.shape[0], xdft.shape[1]
        n_bands = len(bin_ranges)
        device = xdft.device
        dtype = xdft.real.dtype

        attr_acc = torch.zeros(B, n_bands, C, T, device=device, dtype=dtype)

        for perm_idx in range(self.n_perms):
            perm = torch.randperm(n_bands)

            # ── cumulative_add in permuted order ──
            add_prev, add_curr, add_band_idx = [], [], []
            xdft_cumul = torch.zeros_like(xdft)
            for pos in range(n_bands):
                bi = perm[pos].item()
                name, bs, be = bin_ranges[bi]
                x_prev = torch.fft.irfft(xdft_cumul, n=T, dim=-1)
                xdft_cumul = xdft_cumul.clone()
                xdft_cumul[..., bs:be] = xdft[..., bs:be]
                x_curr = torch.fft.irfft(xdft_cumul, n=T, dim=-1)
                add_prev.append(x_prev)
                add_curr.append(x_curr)
                add_band_idx.append(bi)

            # ── cumulative_remove in reverse permuted order ──
            rem_prev, rem_curr, rem_band_idx = [], [], []
            xdft_cumul = xdft.clone()
            for pos in range(n_bands - 1, -1, -1):
                bi = perm[pos].item()
                name, bs, be = bin_ranges[bi]
                x_curr = torch.fft.irfft(xdft_cumul, n=T, dim=-1)
                xdft_cumul = xdft_cumul.clone()
                xdft_cumul[..., bs:be] = 0.0
                x_prev = torch.fft.irfft(xdft_cumul, n=T, dim=-1)
                rem_prev.append(x_prev)
                rem_curr.append(x_curr)
                rem_band_idx.append(bi)

            # Stack both directions: (2*n_bands, B, C, T)
            all_prev = torch.stack(add_prev + rem_prev)
            all_curr = torch.stack(add_curr + rem_curr)
            delta = all_curr - all_prev
            all_band_idx = add_band_idx + rem_band_idx

            # Mega-batch IG
            P = 2 * n_bands
            attr = self._integrate(all_prev, delta, P, B, C, T)
            # attr: (B, P, C, T) — need to accumulate by band index

            # Average add + remove for each band, accumulate
            for j in range(n_bands):
                bi = add_band_idx[j]
                add_attr = attr[:, j]  # (B, C, T)
                rem_attr = attr[
                    :, n_bands + rem_band_idx.index(bi)
                ]  # find matching remove
                attr_acc[:, bi] += 0.5 * (add_attr + rem_attr)

            # Free memory
            del all_prev, all_curr, delta, attr

        return attr_acc / self.n_perms

    # ── Shared IG integration ────────────────────────────────────────

    def _integrate(self, all_prev, delta, P, B, C, T):
        """Trapezoidal IG integration over mega-batch.

        Args:
            all_prev: (P, B, C, T) baselines
            delta: (P, B, C, T) differences
            P, B, C, T: dimensions

        Returns:
            (B, P, C, T) attribution tensor
        """
        device = all_prev.device
        dtype = all_prev.dtype
        mega_B = P * B

        prev_flat = all_prev.reshape(mega_B, C, T)
        delta_flat = delta.reshape(mega_B, C, T)

        alphas = torch.linspace(0.0, 1.0, self.steps, device=device, dtype=dtype)

        step_grads = []
        for s in range(self.steps):
            x_s = (prev_flat + alphas[s] * delta_flat).detach().requires_grad_(True)
            scores = self.f(x_s)
            g = torch.autograd.grad(scores.sum(), x_s)[0]
            step_grads.append(g)

        grads = torch.stack(step_grads)
        avg_grads = torch.trapezoid(grads, dx=1.0 / (self.steps - 1), dim=0)
        attr_flat = delta_flat.detach() * avg_grads

        return attr_flat.reshape(P, B, C, T).permute(1, 0, 2, 3)  # (B, P, C, T)
