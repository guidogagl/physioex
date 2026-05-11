"""Attribution quality metrics: complexity, localization, infidelity, tfle.

All metrics are domain-agnostic or parametrised by domain.  Operations
are fully vectorised over the batch dimension.
"""

import math

import torch
import torch.nn.functional as F


# =====================================================================
# Complexity
# =====================================================================


@torch.no_grad()
def complexity(attr: torch.Tensor, batched: bool = True) -> torch.Tensor:
    """Normalised Shannon entropy of attribution magnitude.

    .. math::

        C = \\frac{-\\sum_i p_i \\log p_i}{\\log N}, \\quad
        p_i = |a_i| / \\sum_j |a_j|

    Returns a value in **[0, 1]** where 0 = all mass on one feature
    (maximally concentrated) and 1 = uniform distribution.

    The normalisation by ``log(N)`` makes the metric comparable across
    attributions with different numbers of features (e.g. SG with 26
    bands vs STFT with 129 bins).

    Args:
        attr: ``(B, *feat)`` or ``(*feat,)`` attribution tensor.
        batched: If ``False`` a batch dim is added/removed.

    Returns:
        ``(B,)`` or scalar in [0, 1].
    """
    if not batched:
        attr = attr.unsqueeze(0)

    attr = attr.reshape(attr.size(0), -1)
    N = attr.shape[-1]
    p = attr.abs() + 1e-8
    p = p / p.sum(dim=-1, keepdim=True)
    entropy = -(p * p.log()).sum(dim=-1)
    # Normalise by max possible entropy so result is in [0, 1]
    entropy = entropy / math.log(N)

    return entropy.squeeze(0) if not batched else entropy


# =====================================================================
# Localization
# =====================================================================


@torch.no_grad()
def localization(
    attr: torch.Tensor,
    mask: torch.Tensor,
    sign_weight: torch.Tensor = None,
    batched: bool = True,
) -> torch.Tensor:
    """Weighted localization ratio (Kohlbrenner et al., IJCNN 2020).

    .. math::

        \\mu_w = \\frac{R_{in}}{R_{tot}} \\cdot \\frac{S_{tot}}{S_{in}}

    Fully vectorised (no Python loops).

    Args:
        attr: ``(B, *feat)`` attribution tensor.
        mask: Binary mask, same shape as *attr*.
        sign_weight: Optional tensor (same shape) whose sign is
            multiplied into *attr* before ReLU.
        batched: Auto-unsqueeze when ``False``.

    Returns:
        ``(B,)`` or scalar.
    """
    if not batched:
        attr = attr.unsqueeze(0)
        mask = mask.unsqueeze(0)
        if sign_weight is not None:
            sign_weight = sign_weight.unsqueeze(0)

    attr = attr.reshape(attr.size(0), -1).clone()
    mask = mask.reshape(mask.size(0), -1)

    if sign_weight is not None:
        sign_weight = sign_weight.reshape(sign_weight.size(0), -1)
        attr = attr * sign_weight.sign()

    attr = torch.relu(attr)

    R_tot = attr.sum(dim=-1)
    R_tot = torch.where(R_tot > 0, R_tot, torch.inf)
    R_in = (attr * mask.float()).sum(dim=-1)
    mu = R_in / R_tot

    S_tot = mask.shape[-1]
    S_in = mask.float().sum(dim=-1).clamp(min=1)

    result = mu * (S_tot / S_in)
    return result.squeeze(0) if not batched else result


# =====================================================================
# Infidelity
# =====================================================================


@torch.no_grad()
def infidelity(
    f: callable,
    x: torch.Tensor,
    attr: torch.Tensor,
    domain: str = "time",
    fs: float = None,
    patch_value: float = 0.0,
    patch_size: int = 10,
    batch_size: int = 0,
    batched: bool = True,
) -> torch.Tensor:
    """Progressive top-k ablation (ROAR-style) infidelity.

    **Lower** score = more faithful attribution.

    All ablation steps are built as a batch of perturbed inputs and
    evaluated through ``f`` in chunks of ``batch_size`` (default: all
    at once).  There are **no per-sample Python loops** — the scatter
    is fully vectorised.

    Args:
        f: Model function.  Must accept ``(M, signal_length)`` and
            return ``(M,)`` or ``(M, 1)`` when used with batched data.
        x: Input ``(B, N)`` or ``(N,)``.
        attr: Attribution, same shape as *x*.
        domain: ``'time'`` or ``'frequency'``.
        fs: Sampling frequency (Hz).  Required for ``'frequency'``.
        patch_value: Replacement value for ablated features.
        patch_size: Features removed per ablation step.
        batch_size: Max perturbed inputs per ``f`` call.  ``0`` =
            all steps in one call (fastest but uses most GPU memory).
            Set lower to bound GPU peak memory.
        batched: Auto-unsqueeze when ``False``.

    Returns:
        ``(B,)`` or scalar infidelity score.
    """
    if domain not in ("time", "frequency"):
        raise ValueError(f"domain must be 'time' or 'frequency', got '{domain}'")
    if domain == "frequency" and fs is None:
        raise ValueError("fs is required when domain='frequency'")
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")

    if not batched:
        x = x.unsqueeze(0)
        attr = attr.unsqueeze(0)

    x = x.reshape(x.size(0), -1)
    attr = attr.reshape(attr.size(0), -1)

    if domain == "frequency":
        result = _infidelity_freq(f, x, attr, fs, patch_value, patch_size, batch_size)
    else:
        result = _infidelity_time(f, x, attr, patch_value, patch_size, batch_size)

    return result.squeeze(0) if not batched else result


# ------------------------------------------------------------------
# Core: build all perturbed inputs at once, evaluate in chunks
# ------------------------------------------------------------------


def _build_perturbations_time(x, attr, patch_value, patch_size):
    """Build all ablation steps as a (n_steps, B, N) tensor.

    Vectorised scatter — no per-sample Python loop.
    """
    B, N = x.shape
    _, indices = torch.sort(attr, dim=-1, descending=True)

    steps = list(range(0, N, patch_size))
    n_steps = len(steps)

    # Start from x repeated for each step, then mask cumulatively.
    # Build a (n_steps, B, N) mask of which features to ablate.
    # At step k, features indices[:, :steps[k]+patch_size] are ablated.
    # We build this incrementally using the sorted indices.

    # Expand x: (n_steps, B, N)
    x_all = x.unsqueeze(0).expand(n_steps, -1, -1).clone()

    # Cumulative ablation mask
    for k, j in enumerate(steps):
        end = min(j + patch_size, N)
        # Gather the indices to ablate for ALL samples in batch at once
        ablate_idx = indices[:, :end]  # (B, end)
        # Scatter patch_value at those positions
        x_all[k].scatter_(1, ablate_idx, patch_value)

    return x_all  # (n_steps, B, N)


def _build_perturbations_freq(x, attr, fs, patch_value, patch_size):
    """Build all frequency-domain ablation steps.

    Ablation operates at the **attribution's native resolution**: DFT
    bins are grouped into bands matching the attribution, then entire
    bands are ablated according to the attribution ranking.  This avoids
    interpolating the attribution to rfft resolution, which would
    degrade the ranking for coarse-resolution methods.

    Uses cumulative mask building to avoid O(n_steps^2) inner loops.

    Returns time-domain perturbed signals: (n_steps, B, N).
    """
    B, N = x.shape
    X = torch.fft.rfft(x, dim=-1)  # (B, n_dft)
    n_dft = X.size(-1)
    n_attr = attr.size(-1)

    # Map each DFT bin to its attribution band index
    # band_of_bin[j] = which attr band covers DFT bin j
    band_of_bin = torch.arange(n_dft, device=x.device) * n_attr // n_dft  # (n_dft,)
    band_of_bin = band_of_bin.clamp(max=n_attr - 1)

    # Rank attribution bands (not individual DFT bins)
    _, band_order = torch.sort(attr, dim=-1, descending=True)  # (B, n_attr)

    # Precompute per-band DFT masks: band_masks[i] = which DFT bins belong to band i
    band_masks = torch.zeros(n_attr, n_dft, dtype=torch.bool, device=x.device)
    for i in range(n_attr):
        band_masks[i] = band_of_bin == i

    steps = list(range(0, n_attr, patch_size))
    n_steps = len(steps)

    X_all = X.unsqueeze(0).expand(n_steps, -1, -1).clone()

    # Build cumulative masks incrementally: O(n_steps * patch_size) instead of O(n_steps^2)
    for b in range(B):
        cum_mask = torch.zeros(n_dft, dtype=torch.bool, device=x.device)
        prev_end = 0
        for k, j in enumerate(steps):
            end = min(j + patch_size, n_attr)
            # Add only the NEW bands (prev_end..end) to the cumulative mask
            for bi_idx in range(prev_end, end):
                cum_mask |= band_masks[band_order[b, bi_idx]]
            X_all[k, b, cum_mask] = patch_value
            prev_end = end

    x_all = torch.fft.irfft(X_all.reshape(-1, n_dft), n=N, dim=-1)
    return x_all.reshape(n_steps, B, N)


def _eval_curve(f, x, x_all, patch_value, batch_size):
    """Evaluate f on original, all perturbations, and fully-ablated.

    Args:
        f: model function (M, N) → (M,).
        x: original input (B, N).
        x_all: perturbed inputs (n_steps, B, N).
        patch_value: value for the fully-ablated baseline.
        batch_size: max inputs per f call (0 = all at once).

    Returns:
        (n_steps + 2, B) curve: [f(x), f(step_0), ..., f(step_K), f(baseline)].
    """
    n_steps, B, N = x_all.shape

    # Flatten all inputs into one tensor: original + steps + baseline
    baseline = torch.full((1, B, N), patch_value, device=x.device, dtype=x.dtype)
    # (n_steps + 2, B, N)
    all_inputs = torch.cat([x.unsqueeze(0), x_all, baseline], dim=0)
    total = all_inputs.shape[0]

    # Reshape to (total * B, N) for batched evaluation
    flat = all_inputs.reshape(total * B, N)

    chunk = batch_size if batch_size > 0 else flat.shape[0]
    outputs = []
    for start in range(0, flat.shape[0], chunk):
        end = min(start + chunk, flat.shape[0])
        out = f(flat[start:end])
        if out.dim() > 1:
            out = out.reshape(end - start)
        outputs.append(out)

    curve_flat = torch.cat(outputs, dim=0)  # (total * B,)
    return curve_flat.reshape(total, B)  # (n_steps + 2, B)


def _infidelity_time(f, x, attr, patch_value, patch_size, batch_size):
    x_all = _build_perturbations_time(x, attr, patch_value, patch_size)
    curve = _eval_curve(f, x, x_all, patch_value, batch_size)
    return _normalise_and_integrate(curve)


def _infidelity_freq(f, x, attr, fs, patch_value, patch_size, batch_size):
    x_all = _build_perturbations_freq(x, attr, fs, patch_value, patch_size)
    curve = _eval_curve(f, x, x_all, patch_value, batch_size)
    return _normalise_and_integrate(curve)


def _normalise_and_integrate(curve):
    """Normalise degradation curve and integrate.

    Args:
        curve: ``(K, B)`` tensor where K = n_steps + 2.

    Returns:
        ``(B,)`` scores clamped to [0, 1].

    When ``f(x) ≈ f(baseline)`` the denominator is near zero and the
    normalised curve can explode.  We guard against this by requiring a
    minimum denominator of 1 % of the absolute original value (or 1e-6
    as hard floor), and clamping the final integral to [0, 1].
    """
    curve = curve - curve[-1]  # shift baseline to 0
    curve = curve.clamp(min=0)  # clamp to [baseline, ...]: values below baseline are not informative
    abs_ref = curve[0].abs().clamp(min=1e-6)
    denom = curve[0].sign() * abs_ref  # preserve sign, safe magnitude
    denom = torch.where(denom.abs() < 1e-6, torch.ones_like(denom), denom)
    curve = curve / denom
    result = torch.trapezoid(curve, dx=1.0 / curve.shape[0], dim=0)
    return result.clamp(0.0, 1.0)  # (B,) in [0, 1]


# =====================================================================
# Time-Frequency Localization Error (TFLE)
# =====================================================================


@torch.no_grad()
def tfle(
    attr: torch.Tensor,
    event_freq: float,
    event_time: float,
    fs: float,
    band_frequencies: torch.Tensor,
) -> dict:
    """Time-Frequency Localization Error.

    Measures how accurately a time-frequency attribution map localises a
    known event at ``(event_freq, event_time)``, in **physical units**
    (seconds and Hz) so that methods at different resolutions are
    directly comparable.

    **Theoretical motivation** — An STFT-based attribution has resolution
    cells of constant area ``Δt × Δf = 1`` (Gabor bound), so improving
    one axis degrades the other.  SpectralGradients decouples the two
    axes: ``Δt = 1/fs`` (from gradients) and ``Δf = freq_step`` (from
    DFT ablation), giving a resolution product ``Δt × Δf`` that can be
    orders of magnitude smaller.

    This metric quantifies the *effective* localisation quality
    achievable by each method on a concrete signal.

    Args:
        attr: Time-frequency attribution map ``(B, n_freq, n_time)`` or
            ``(n_freq, n_time)``.
        event_freq: Ground-truth event frequency in Hz.
        event_time: Ground-truth event onset in seconds.
        fs: Sampling frequency in Hz.
        band_frequencies: 1-D tensor of center frequencies (Hz) for each
            band in the attribution map.  Length must equal
            ``attr.shape[-2]``.

    Returns:
        Dictionary with keys:

        - **time_error** ``(B,)`` — absolute temporal error in seconds.
        - **freq_error** ``(B,)`` — absolute frequency error in Hz.
        - **tf_error** ``(B,)`` — product ``time_error × freq_error``
          (Hz·s).  Lower is better.  Gabor-limited methods have a
          floor at their resolution cell area; SG can go below.
        - **time_spread** ``(B,)`` — effective temporal spread
          ``exp(H(T | F=f₀)) / fs`` (seconds) of the attribution
          in the event's frequency band.
        - **freq_spread** ``(B,)`` — effective frequency spread
          ``exp(H(F | T=t₀)) × Δf`` (Hz) at the event's time.
        - **tf_spread** ``(B,)`` — product ``time_spread × freq_spread``
          (Hz·s).  The effective area of the uncertainty region around
          the event.
    """
    batched = True
    if attr.dim() == 2:
        attr = attr.unsqueeze(0)
        batched = False

    B, n_freq, n_time = attr.shape
    attr_abs = attr.abs().float()

    band_frequencies = band_frequencies.float()
    dt = 1.0 / fs  # seconds per time sample
    df = (
        (band_frequencies[1] - band_frequencies[0]).abs().item()
        if n_freq > 1
        else fs / 2
    )

    # --- Identify closest band / sample to the event ---
    freq_idx = (band_frequencies - event_freq).abs().argmin().item()
    time_idx = min(max(round(event_time * fs), 0), n_time - 1)

    # --- Peak-based localisation error ---
    # Time: peak of the conditional P(t | f = freq_idx)
    band_profile = attr_abs[:, freq_idx, :]  # (B, n_time)
    peak_time_idx = band_profile.argmax(dim=-1)  # (B,)
    time_error = (peak_time_idx.float() - time_idx).abs() * dt  # seconds

    # Frequency: peak of P(f | t = time_idx)
    time_profile = attr_abs[:, :, time_idx]  # (B, n_freq)
    peak_freq_idx = time_profile.argmax(dim=-1)  # (B,)
    freq_error = (band_frequencies[peak_freq_idx] - event_freq).abs()  # Hz

    tf_error = time_error * freq_error  # Hz·s

    # --- Spread-based uncertainty ---
    # Time spread: exp(H(T | F=freq_idx)) × dt
    p_t = band_profile + 1e-10
    p_t = p_t / p_t.sum(dim=-1, keepdim=True)
    h_t = -(p_t * p_t.log()).sum(dim=-1)  # (B,)
    time_spread = h_t.exp() * dt  # seconds

    # Freq spread: exp(H(F | T=time_idx)) × df
    p_f = time_profile + 1e-10
    p_f = p_f / p_f.sum(dim=-1, keepdim=True)
    h_f = -(p_f * p_f.log()).sum(dim=-1)  # (B,)
    freq_spread = h_f.exp() * df  # Hz

    tf_spread = time_spread * freq_spread  # Hz·s

    def _maybe_squeeze(t):
        return t.squeeze(0) if not batched else t

    return {
        "time_error": _maybe_squeeze(time_error),
        "freq_error": _maybe_squeeze(freq_error),
        "tf_error": _maybe_squeeze(tf_error),
        "time_spread": _maybe_squeeze(time_spread),
        "freq_spread": _maybe_squeeze(freq_spread),
        "tf_spread": _maybe_squeeze(tf_spread),
    }


@torch.no_grad()
def tf_concentration(
    attr: torch.Tensor,
    fs: float,
    band_frequencies: torch.Tensor,
) -> dict:
    """Unsupervised time-frequency concentration of an attribution map.

    Measures how tightly focused a time-frequency attribution is around
    its own peak — no ground-truth event location required.

    For each sample the peak cell ``(f*, t*)`` is found from the
    attribution itself, then:

    - **time_spread**: effective temporal width ``exp(H(T | F=f*)) × dt``
      in the peak's frequency band.
    - **freq_spread**: effective frequency width ``exp(H(F | T=t*)) × df``
      at the peak's time step.
    - **tf_spread**: product ``time_spread × freq_spread`` (Hz·s).
      Lower = more concentrated attribution.

    Args:
        attr: ``(B, n_freq, n_time)`` or ``(n_freq, n_time)``.
        fs: Sampling frequency in Hz.
        band_frequencies: 1-D tensor of center frequencies (Hz),
            length ``n_freq``.

    Returns:
        Dictionary with ``time_spread``, ``freq_spread``, ``tf_spread``.
    """
    batched = True
    if attr.dim() == 2:
        attr = attr.unsqueeze(0)
        batched = False

    B, n_freq, n_time = attr.shape
    attr_abs = attr.abs().float()

    band_frequencies = band_frequencies.float()
    dt = 1.0 / fs
    df = (
        (band_frequencies[1] - band_frequencies[0]).abs().item()
        if n_freq > 1
        else fs / 2
    )

    # Find peak cell per sample: argmax over the flattened TF map
    flat_peak = attr_abs.reshape(B, -1).argmax(dim=-1)  # (B,)
    peak_freq_idx = flat_peak // n_time  # (B,)
    peak_time_idx = flat_peak % n_time   # (B,)

    # Time spread: exp(H(T | F=f*)) × dt, per sample
    # Gather the frequency band of each sample's peak
    band_profiles = attr_abs[
        torch.arange(B, device=attr.device), peak_freq_idx, :
    ]  # (B, n_time)
    p_t = band_profiles + 1e-10
    p_t = p_t / p_t.sum(dim=-1, keepdim=True)
    h_t = -(p_t * p_t.log()).sum(dim=-1)
    time_spread = h_t.exp() * dt  # seconds

    # Freq spread: exp(H(F | T=t*)) × df, per sample
    time_profiles = attr_abs[
        torch.arange(B, device=attr.device), :, peak_time_idx
    ]  # (B, n_freq)
    p_f = time_profiles + 1e-10
    p_f = p_f / p_f.sum(dim=-1, keepdim=True)
    h_f = -(p_f * p_f.log()).sum(dim=-1)
    freq_spread = h_f.exp() * df  # Hz

    tf_spread = time_spread * freq_spread  # Hz·s

    def _maybe_squeeze(t):
        return t.squeeze(0) if not batched else t

    return {
        "time_spread": _maybe_squeeze(time_spread),
        "freq_spread": _maybe_squeeze(freq_spread),
        "tf_spread": _maybe_squeeze(tf_spread),
    }


@torch.no_grad()
def resolution_product(
    method: str,
    fs: float,
    signal_length: int,
    freq_step: float = None,
    n_fft: int = None,
    hop_length: int = None,
) -> dict:
    """Theoretical resolution cell area for a given method.

    Returns the intrinsic ``Δt × Δf`` product (Hz·s) and the number
    of independent time-frequency cells.  For STFT this product is
    always 1 (Gabor bound); for SpectralGradients it is ``freq_step / fs``.

    Args:
        method: ``'sg'`` or ``'stft'``.
        fs: Sampling frequency (Hz).
        signal_length: Number of samples.
        freq_step: Band width in Hz (for ``method='sg'``).
        n_fft: FFT size (for ``method='stft'``).
        hop_length: Hop in samples (for ``method='stft'``).

    Returns:
        Dict with ``dt``, ``df``, ``dt_x_df``, ``n_cells``.
    """
    if method == "sg":
        if freq_step is None:
            raise ValueError("freq_step required for method='sg'")
        dt = 1.0 / fs
        df = freq_step
        n_time = signal_length
        n_freq = math.ceil(
            (signal_length // 2 + 1) / max(1, round(freq_step / (fs / signal_length)))
        )
    elif method == "stft":
        if n_fft is None or hop_length is None:
            raise ValueError("n_fft and hop_length required for method='stft'")
        dt = n_fft / fs
        df = fs / n_fft
        n_freq = n_fft // 2 + 1
        n_time = (signal_length - n_fft) // hop_length + 1
    else:
        raise ValueError(f"method must be 'sg' or 'stft', got '{method}'")

    return {
        "dt": dt,
        "df": df,
        "dt_x_df": dt * df,
        "n_freq": n_freq,
        "n_time": n_time,
        "n_cells": n_freq * n_time,
    }
