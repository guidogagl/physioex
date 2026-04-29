"""Spectral Gradients — time-frequency attribution without Gabor tradeoff.

Produces a ``(batch, n_bands, signal_length)`` attribution map with:
- **Full temporal resolution** (one value per signal sample, from gradient
  back-propagation).
- **Configurable frequency resolution** (from ``freq_step`` in Hz,
  independent of time resolution — no Gabor uncertainty principle).
- **Completeness**: ``sum over bands = IG(x, baseline=silence)``, exact
  by the telescoping property of incremental attributions.

The core idea: instead of computing one global IG from silence to the
full signal, decompose it into **local IG steps** along a frequency
ablation path.  Each step adds one frequency band and the local IG
measures how that band redistributes importance across time samples.
"""

import math

import torch
import torch.nn as nn

from physioex.explain.posthoc.gradients import IntegratedGradients


class SpectralGradients(nn.Module):
    """Time-frequency attribution via local Integrated Gradients along
    a frequency ablation path.

    At each step of the ablation path the method computes::

        attr(band_i) = IG(x_with_band_i, baseline=x_without_band_i)

    where the two signals differ by exactly one frequency band.  Because
    the difference is small, the local IG converges with very few steps
    (typically 5–20), making the method much cheaper than a full IG per
    frequency.

    Summing over all bands recovers standard IG by telescoping::

        sum_i attr(band_i) = IG(x_full, baseline=silence)

    Two ablation directions are supported (and can be averaged to reduce
    path dependence):

    - **low_to_high**: frequencies are added from DC upward.
      ``attr(i)`` = "effect of adding band *i*, given bands 0..i-1 present".
    - **high_to_low**: frequencies are added from Nyquist downward.
      ``attr(i)`` = "effect of adding band *i*, given bands i+1..N present".

    **CUDA parallelisation** — When ``expects_batch=True``, all frequency
    bands (and both path directions when ``path='both'``) are packed into
    a single mega-batch and processed in one GPU kernel per IG step.  The
    number of kernel launches is exactly ``steps``, independent of the
    number of bands, batch size, or path choice.

    Args:
        f: Model or scoring function.

            - If ``expects_batch=False`` (default): ``f(x)`` takes a 1-D
              signal ``(signal_length,)`` and returns a scalar or a 1-D
              class-score vector.
            - If ``expects_batch=True``: ``f(x)`` takes ``(B, signal_length)``
              and returns ``(B, n_classes)`` or ``(B,)``.  This enables
              mega-batched GPU evaluation where all bands are processed
              in parallel.

        fs (float): Sampling frequency in Hz.

        freq_step (float): Frequency band width in Hz.  Default ``1.0``.

        steps (int): Integration points for each local IG.  Default ``10``.

        path (str): ``'both'`` (default), ``'low_to_high'``, or
            ``'high_to_low'``.

        target (int or None): Class index for multi-output models.

        expects_batch (bool): Whether ``f`` accepts ``(B, D)`` input.

        grad_batch_size (int): Memory bound for per-sample gradient path
            (``expects_batch=False`` only).  ``0`` = unlimited.

    Returns:
        ``(batch, n_bands, signal_length)`` attribution tensor.

    Example::

        >>> sg = SpectralGradients(
        ...     f=lambda x: model(x).softmax(-1),
        ...     fs=100.0, freq_step=2.0,
        ...     expects_batch=True, target=2,
        ... )
        >>> attr = sg(x_batch)         # (B, n_bands, signal_length)
        >>> attr.sum(dim=1)            # ≈ IG(x, baseline=0) — completeness
    """

    def __init__(
        self,
        f: callable,
        fs: float = 1.0,
        freq_step: float = 1.0,
        steps: int = 10,
        path: str = "both",
        n_perms: int = 20,
        target: int = None,
        expects_batch: bool = False,
        grad_batch_size: int = 0,
        **kwargs,
    ):
        super().__init__()
        if path not in ("both", "low_to_high", "high_to_low", "shapley"):
            raise ValueError(
                f"path must be 'both', 'low_to_high', 'high_to_low', or 'shapley', got '{path}'"
            )
        if fs <= 0:
            raise ValueError(f"fs must be > 0, got {fs}")
        if freq_step <= 0:
            raise ValueError(f"freq_step must be > 0, got {freq_step}")
        if steps < 2:
            raise ValueError(f"steps must be >= 2, got {steps}")

        self.fs = float(fs)
        self.freq_step = float(freq_step)
        self.steps = steps
        self.path = path
        self.n_perms = n_perms

        # Kept for the per-sample fallback path (expects_batch=False).
        self._ig = IntegratedGradients(
            f=f,
            steps=steps,
            target=target,
            expects_batch=expects_batch,
            grad_batch_size=grad_batch_size,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def f(self):
        return self._ig.f

    @property
    def target(self):
        return self._ig.target

    @target.setter
    def target(self, value):
        self._ig.target = value

    @property
    def expects_batch(self):
        return self._ig.expects_batch

    # ------------------------------------------------------------------
    # Frequency helpers
    # ------------------------------------------------------------------

    def _bin_step(self, signal_length: int) -> int:
        """Convert ``freq_step`` (Hz) to DFT bins for a given signal length."""
        freq_resolution = self.fs / signal_length
        return max(1, round(self.freq_step / freq_resolution))

    def n_bands(self, signal_length: int) -> int:
        """Number of frequency bands for a given signal length."""
        n_freqs = signal_length // 2 + 1
        return math.ceil(n_freqs / self._bin_step(signal_length))

    def band_frequencies(self, signal_length: int = None):
        """Center frequency (Hz) of each band.

        Args:
            signal_length: If ``None``, uses the length from the last
                ``forward`` call.

        Returns:
            1-D tensor of center frequencies in Hz.
        """
        if signal_length is None:
            signal_length = getattr(self, "_last_signal_length", None)
            if signal_length is None:
                raise ValueError(
                    "signal_length required (call forward first or pass explicitly)"
                )
        freqs = torch.fft.rfftfreq(signal_length, d=1.0 / self.fs)
        bin_step = self._bin_step(signal_length)
        centers = []
        for start in range(0, freqs.shape[0], bin_step):
            end = min(start + bin_step, freqs.shape[0])
            centers.append(freqs[start:end].mean())
        return torch.stack(centers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """Compute time-frequency attributions.

        When ``expects_batch=True`` all bands (and both path directions)
        are processed in parallel via a single mega-batch per IG step —
        the number of GPU kernel launches equals ``steps`` regardless of
        the number of bands, batch size, or path.

        When ``expects_batch=False`` bands are processed sequentially,
        each delegated to :class:`IntegratedGradients`.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        self._last_signal_length = x.shape[-1]
        xdft = torch.fft.rfft(x, dim=-1)
        n = x.shape[-1]

        if self.path == "shapley":
            return self._forward_shapley(xdft, n)
        if self.expects_batch:
            return self._forward_parallel(xdft, n)
        return self._forward_sequential(xdft, n)

    # ------------------------------------------------------------------
    # Shapley path: average over random permutations
    # ------------------------------------------------------------------

    def _forward_shapley(self, xdft, n):
        """Approximate Shapley values via random-permutation IG.

        For each permutation all n_bands local-IG pairs are mega-batched
        into a single GPU call per IG step — identical to
        ``_forward_parallel`` but repeated for ``n_perms`` random
        orderings.

        GPU kernel launches = ``n_perms × steps``, independent of
        n_bands or B.  This is **n_bands×** faster than the naive
        per-band loop.

        Memory is explicitly freed between permutations to prevent OOM.

        After accumulating the raw Shapley attributions, each frequency
        band (row) is weighted by its signed importance
        ``w_i = sum_t attr(i, t)`` to suppress noise bands whose
        temporal attributions cancel out (positive ≈ negative).
        """
        B = xdft.shape[0]
        n_freqs = xdft.shape[-1]
        bin_step = self._bin_step(n)
        band_starts = list(range(0, n_freqs, bin_step))
        n_bands = len(band_starts)

        attr_acc = torch.zeros(B, n_bands, n, device=xdft.device, dtype=xdft.real.dtype)

        alphas = torch.linspace(
            0.0,
            1.0,
            self.steps,
            device=xdft.device,
            dtype=xdft.real.dtype,
        )

        for perm_idx in range(self.n_perms):
            perm = torch.randperm(n_bands)

            # Build ALL n_bands (prev, curr) pairs incrementally.
            all_prev = []
            all_curr = []

            xdft_cumul = torch.zeros_like(xdft)
            for pos in range(n_bands):
                band_i = perm[pos].item()
                si = band_starts[band_i]
                ei = min(si + bin_step, n_freqs)

                x_prev = torch.fft.irfft(xdft_cumul, n=n, dim=-1)

                xdft_cumul = xdft_cumul.clone()
                xdft_cumul[..., si:ei] = xdft[..., si:ei]

                x_curr = torch.fft.irfft(xdft_cumul, n=n, dim=-1)

                all_prev.append(x_prev)
                all_curr.append(x_curr)

            # Stack: (n_bands, B, n)
            all_prev = torch.stack(all_prev)
            all_curr = torch.stack(all_curr)
            delta = all_curr - all_prev

            mega_B = n_bands * B
            prev_flat = all_prev.reshape(mega_B, n)
            delta_flat = delta.reshape(mega_B, n)

            # Free intermediates before the heavy IG loop
            del all_prev, all_curr, xdft_cumul

            # IG integration: one GPU call per step for ALL bands
            step_grads = []
            for s in range(self.steps):
                x_s = (prev_flat + alphas[s] * delta_flat).detach().requires_grad_(True)
                scores = self._ig._batched_scores(x_s)
                g = torch.autograd.grad(scores.sum(), x_s)[0]
                step_grads.append(g.detach())

            grads = torch.stack(step_grads)
            avg_grads = torch.trapezoid(grads, dx=1.0 / (self.steps - 1), dim=0)
            attr_flat = delta_flat.detach() * avg_grads

            # Scatter back to band indices
            attr_perm = attr_flat.reshape(n_bands, B, n)
            for pos in range(n_bands):
                band_i = perm[pos].item()
                attr_acc[:, band_i, :] += attr_perm[pos].detach()

            # Explicit cleanup between permutations
            del (
                prev_flat,
                delta_flat,
                delta,
                step_grads,
                grads,
                avg_grads,
                attr_flat,
                attr_perm,
            )

        attr = attr_acc / self.n_perms  # (B, n_bands, n)

        # --- Band importance weighting ---
        # w_i = sum_t attr(i, t): signed importance of band i.
        # Noise bands have w ≈ 0 (positive and negative cancel).
        # Discriminative bands have |w| >> 0.
        # Weighting each row by |w_i| suppresses noise rows while
        # preserving the temporal structure of important bands.
        w = attr.sum(dim=-1, keepdim=True)  # (B, n_bands, 1)
        attr = attr * w.abs()  # row-wise weighting

        return attr

    # ------------------------------------------------------------------
    # Sequential path (expects_batch=False): one IG call per band
    # ------------------------------------------------------------------

    def _forward_sequential(self, xdft, n):
        if self.path == "low_to_high":
            return self._compute_path_seq(xdft, n, "low_to_high")
        elif self.path == "high_to_low":
            return self._compute_path_seq(xdft, n, "high_to_low")
        else:
            fwd = self._compute_path_seq(xdft, n, "low_to_high")
            bwd = self._compute_path_seq(xdft, n, "high_to_low")
            return 0.5 * (fwd + bwd)

    def _compute_path_seq(self, xdft, n, direction):
        n_freqs = xdft.shape[-1]
        bin_step = self._bin_step(n)
        attrs = []
        for start in range(0, n_freqs, bin_step):
            end = min(start + bin_step, n_freqs)
            x_prev, x_curr = self._build_pair(xdft, n, start, end, direction)
            attrs.append(self._ig(x_curr, baseline=x_prev))
        return torch.stack(attrs, dim=1)

    # ------------------------------------------------------------------
    # Parallel path (expects_batch=True): mega-batch all bands at once
    # ------------------------------------------------------------------

    def _forward_parallel(self, xdft, n):
        """Pack all bands (× directions) into one mega-batch per IG step.

        GPU kernel launches = ``self.steps`` regardless of n_bands, B, or
        number of path directions.

        Memory per step ≈ ``n_directions × n_bands × B × signal_length × 4``
        bytes.  For typical EEG (26 bands, B=32, 3000 samples): ~24 MB —
        negligible on any modern GPU.
        """
        B = xdft.shape[0]
        n_freqs = xdft.shape[-1]
        bin_step = self._bin_step(n)
        band_starts = list(range(0, n_freqs, bin_step))
        n_bands = len(band_starts)

        # Decide which directions to compute
        directions = []
        if self.path in ("both", "low_to_high"):
            directions.append("low_to_high")
        if self.path in ("both", "high_to_low"):
            directions.append("high_to_low")

        # Build ALL (prev, curr) pairs for ALL directions at once.
        all_prev, all_curr = [], []
        for direction in directions:
            for start in band_starts:
                end = min(start + bin_step, n_freqs)
                p, c = self._build_pair(xdft, n, start, end, direction)
                all_prev.append(p)
                all_curr.append(c)

        # (P, B, n) where P = n_directions × n_bands
        all_prev = torch.stack(all_prev)
        all_curr = torch.stack(all_curr)
        delta = all_curr - all_prev

        P = all_prev.shape[0]
        mega_B = P * B

        # Flatten (P, B, n) → (P*B, n) for the model forward pass
        prev_flat = all_prev.reshape(mega_B, n)
        delta_flat = delta.reshape(mega_B, n)

        # IG integration: one GPU call per step over the entire mega-batch
        alphas = torch.linspace(
            0.0,
            1.0,
            self.steps,
            device=prev_flat.device,
            dtype=prev_flat.dtype,
        )

        step_grads = []
        for s in range(self.steps):
            x_s = (prev_flat + alphas[s] * delta_flat).detach().requires_grad_(True)
            scores = self._ig._batched_scores(x_s)  # (mega_B,)
            g = torch.autograd.grad(scores.sum(), x_s)[0]  # (mega_B, n)
            step_grads.append(g)

        # (steps, mega_B, n) → trapezoidal → (mega_B, n)
        grads = torch.stack(step_grads)
        avg_grads = torch.trapezoid(grads, dx=1.0 / (self.steps - 1), dim=0)
        attr_flat = delta_flat.detach() * avg_grads

        # Reshape: (P, B, n) → split directions → (B, n_bands, n)
        attr = attr_flat.reshape(P, B, n)

        if len(directions) == 2:
            attr_fwd = attr[:n_bands].permute(1, 0, 2)
            attr_bwd = attr[n_bands:].permute(1, 0, 2)
            return 0.5 * (attr_fwd + attr_bwd)
        return attr.permute(1, 0, 2)

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pair(xdft, n, start, end, direction):
        """Build (x_prev, x_curr) for one frequency band."""
        if direction == "low_to_high":
            xdft_prev = xdft.clone()
            xdft_prev[..., start:] = 0.0
            xdft_curr = xdft.clone()
            xdft_curr[..., end:] = 0.0
        else:
            xdft_curr = xdft.clone()
            xdft_curr[..., :start] = 0.0
            xdft_prev = xdft.clone()
            xdft_prev[..., :end] = 0.0

        return (
            torch.fft.irfft(xdft_prev, n=n, dim=-1),
            torch.fft.irfft(xdft_curr, n=n, dim=-1),
        )


# ======================================================================
# Standalone generator utilities (backward-compatibility)
# ======================================================================


def low_pass(xdft: torch.Tensor, n: int, step: int = 1):
    """Progressively add low frequencies.

    Yields ``(time_domain_signal, magnitude_of_added_bin)`` tuples.
    """
    n_freqs = xdft.shape[-1]
    for i in range(0, n_freqs, step):
        xdft_ablated = xdft.clone()
        ablated = torch.abs(xdft_ablated[..., i])
        xdft_ablated[..., i:] = 0.0
        yield torch.fft.irfft(xdft_ablated, n=n, dim=-1), ablated


def high_pass(xdft: torch.Tensor, n: int, step: int = 1):
    """Progressively remove low frequencies.

    Yields ``(time_domain_signal, magnitude_of_removed_bin)`` tuples.
    """
    n_freqs = xdft.shape[-1]
    for i in range(0, n_freqs, step):
        xdft_ablated = xdft.clone()
        ablated = torch.abs(xdft_ablated[..., i])
        xdft_ablated[..., :i] = 0.0
        yield torch.fft.irfft(xdft_ablated, n=n, dim=-1), ablated


def random_pass(xdft: torch.Tensor, n: int, step: int = 1):
    """Ablate frequencies in random order."""
    indices = torch.randperm(xdft.shape[-1])
    for i in range(0, xdft.shape[-1] + step, step):
        idx = indices[:i]
        xdft_ablated = xdft.clone()
        xdft_ablated[..., idx] = 0.0
        yield torch.fft.irfft(xdft_ablated, n=n, dim=-1), idx


# ======================================================================
# FreqSpectralGradients — kept for reference, not recommended for use.
# ======================================================================

from physioex.explain.posthoc.gradients import Saliency
from physioex.explain.posthoc.vidft import DFTLayer


class FreqSpectralGradients(Saliency):
    """Output-space variant (experimental, known issues — prefer
    :class:`SpectralGradients`)."""

    def __init__(self, f, n=0, dim=-1, use_rfft=True, policy="saliency", **kwargs):
        super().__init__(f, **kwargs)
        self.dim = dim
        self.use_rfft = use_rfft
        self.policy = policy.lower()
        if self.policy not in ("saliency", "input_x_gradient"):
            raise ValueError(
                f"policy must be 'saliency' or 'input_x_gradient', got '{policy}'"
            )
        self.dft_layer = DFTLayer(dim=dim, use_rfft=use_rfft)

    def _compute_scores_along_path(self, xdft, n, func_generator):
        scores = []
        for x_step, _ablated in func_generator(xdft, n=n, step=1):
            out = self.f(x_step)
            if self.policy == "input_x_gradient":
                signal_energy = x_step.abs().sum(dim=-1, keepdim=True)
                out = signal_energy * out
            scores.append(out)
        return torch.stack(scores, dim=0)

    def forward(self, x, baseline=None, steps=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        xdft = self.dft_layer(x)
        n = x.shape[-1]
        scores_low = self._compute_scores_along_path(xdft, n, low_pass)
        scores_high = self._compute_scores_along_path(xdft, n, high_pass)
        importance = scores_low.flip(0) * scores_high
        max_score = importance.abs().max(dim=0, keepdim=True)[0]
        importance = importance / (max_score + 1e-8)
        return importance.movedim(0, 1)
