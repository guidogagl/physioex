"""ECG preprocessing for the Jones, Itti & Sheth (2024) sleep-staging network.

Python port of the authors' MATLAB preprocessing
(``ecg-sleep-staging/paper/0_data_processing``: ``2_cut_clip_subtract_resample``,
``3_templates``, ``5_sets``), implemented from the published functional
description. The step takes one raw single-lead ECG channel (whole night,
any sampling rate; for two-lead recordings the channel resolution already
delivers the lead difference) and turns it into the network input of
:class:`physioex.models.jones_ecg.JonesECGNet`: 256 Hz, scaled by the
per-night R-wave normalization factor and hard-clipped to ``[-1, 1]``.

Deviations from the MATLAB reference
------------------------------------
1. ``designfilt(..., 'cheby2', ..., 'MatchExactly', 'passband')`` has no
   closed-form order; the order is found by searching for the minimal N
   whose passband ripple stays <= 0.1 dB up to 0.5 Hz (the stopband spec
   only sets the transition).  Typically < 1e-6 different frequency
   response from MATLAB's solver.
2. ``designfilt('butter', ..., 'MatchExactly', 'passband')`` for the
   local-variation low-pass is handled the same way: minimal Butterworth
   order meeting <= 1 dB ripple at 8 Hz.
3. The mask-smoothing spline call in ``smooth_mask_array.m``
   (``spline([0, 0.5, 1], [0, 0, 0.5, 1, 0])``) is malformed as written
   (3 knots, 5 values); the documented intent is a monotone S-curve with
   zero end derivatives, so we use smoothstep ``g(t) = 3t^2 - 2t^3``.
4. MATLAB ``resample(x, p, q)`` uses a Kaiser-window FIR; we use
   ``scipy.signal.resample_poly(x, p, q, window=('kaiser', 8))`` with the
   same ``10*max(p, q)`` tap count — same architecture, not bit-exact.
5. ``robust_zscore`` is not in the authors' tree; per its signature we
   implement ``z(x) = (x - median(x)) / (1.4826 * med(|x - median(x)|))``
   (assumption to verify against the missing file).
6. Moving-window operators with MATLAB edge truncation (``movvar``,
   ``movmedian``, ``movmax``, ``movmean`` one-sided variants) are
   implemented with explicit sliding windows so edge semantics match
   MATLAB (``scipy.ndimage`` would mirror/reflect instead).
7. ``prctile(x, 50:5:100)(end-2) * 2`` is the 95th percentile (linear
   interpolation, MATLAB default) times 2; the step contract is
   "2 x the 90th percentile of per-beat max |x|", and we implement
   ``2 * np.percentile(beatsmax, 90)`` per the contract.
8. If the archetype download fails, the template falls back to the
   column-median of the detected beats (documented deviation).

Archetypes
----------
``archetypes_path=None`` (the default) downloads
``archetypes.mat`` from the authors' repository on first use through
``physioex.models.foundation_checkpoints`` under the registry name
``"jones-ecg-archetypes"``; an explicit ``archetypes_path`` is used as-is.
A path must not change the cache hash, so ``archetypes_path`` is excluded
from ``spec()``.
"""

from __future__ import annotations

import logging
import threading
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as _sig
from scipy.io import loadmat

from physioex.data.pipeline import CompiledStep, PreprocessingStep

logger = logging.getLogger("physioex.data.steps")

#: Target sampling rate of the Jones ECG network.
JONES_ECG_FS_OUT = 256.0

_ARCHETYPES_NAME = "jones-ecg-archetypes"

# Local-variation / peak-picking constants (get_constants.m, fs=256).
_LV_LOCAL_SAMPLES = 17        # next_odd(0.060 * 256)
_LV_GROUP_SAMPLES = 257       # next_odd(1 * 256)
_LV_MIN_PEAK_HEIGHT = 1.5     # min_peak_height
_LV_MIN_PEAK_SEPARATION = 77  # round(0.3 * 256)
_LV_LP_PASS_HZ = 8.0          # low-pass passband edge
_LV_LP_STOP_HZ = 10.0         # low-pass stopband edge
_LV_LP_STOP_DB = 30.0
_LV_LP_PASS_RIPPLE_DB = 1.0

# find_noise constants (cut_to_resample_process.m / find_noise.m).
_FIND_NOISE_WINDOWS = 10
_FIND_NOISE_AMP_SPAN_S = 20.0
_FIND_NOISE_AMP_FLOOR = 0.1
_FIND_NOISE_SPECTRUM_HZ = 0.5
_FIND_NOISE_MIN_FREQ_HZ = 10.0
_FIND_NOISE_MIN_RATIO = 4.0   # ecg_noise_min_ratio
_FIND_NOISE_MAX_ADD = 10      # max_additional_freqs
_NOTCH_BW_HZ = 1.5            # ecg_filter_bw

_warned_fallback = False
_warned_lock = threading.Lock()
_archetype_download_registered = False


def _register_archetype_source() -> None:
    """Register the raw-URL download source for ``archetypes.mat`` once."""
    global _archetype_download_registered
    if _archetype_download_registered:
        return
    from physioex.models.foundation_checkpoints import (
        CheckpointSource,
        _register_checkpoint_source,
    )

    _register_checkpoint_source(
        _ARCHETYPES_NAME,
        CheckpointSource(
            source_type="raw_url",
            repo_id="adammj/ecg-sleep-staging",
            filenames=["archetypes.mat"],
            raw_urls=[
                "https://raw.githubusercontent.com/adammj/ecg-sleep-staging/"
                "main/paper/0_data_processing/3_templates/archetypes.mat"
            ],
        ),
    )
    _archetype_download_registered = True


def _load_archetypes(archetypes_path: Optional[str]) -> Optional[np.ndarray]:
    """Load the archetype matrix (n_archetypes x 256 in MATLAB); ``None`` if unavailable.

    ``archetypes_path=None`` triggers the one-shot raw-URL download
    (cached under the ``jones-ecg-archetypes`` registry name); any
    failure logs a warning once and returns ``None`` so the caller can
    fall back to the median-of-beats template.
    """
    if archetypes_path is not None:
        path = archetypes_path
    else:
        try:
            _register_archetype_source()
            from physioex.models.foundation_checkpoints import ensure_checkpoint

            resolved = ensure_checkpoint(_ARCHETYPES_NAME)
        except Exception as e:  # download / network failure
            logger.warning(
                "JonesECGPreprocess: could not download archetypes.mat "
                f"({e}); using the median-of-beats template fallback"
            )
            return None
        path = resolved[0] if isinstance(resolved, (list, tuple)) else resolved

    try:
        mat = loadmat(path)
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5; h5py returns the arrays with the
        # dimensions reversed with respect to MATLAB, so transpose back.
        import h5py

        with h5py.File(path, "r") as f:
            mat = {k: np.asarray(f[k]).T for k in f.keys() if isinstance(f[k], h5py.Dataset)}
    if "archetype_templates" not in mat:
        logger.warning(
            "JonesECGPreprocess: %s has no 'archetype_templates' variable; "
            "using the median-of-beats template fallback", path
        )
        return None
    arch = np.asarray(mat["archetype_templates"], dtype=np.float64)
    if arch.ndim != 2 or arch.shape[1] != 256:
        logger.warning(
            "JonesECGPreprocess: archetypes matrix has shape %s, expected "
            "(n, 256); using the median-of-beats template fallback",
            arch.shape,
        )
        return None
    return arch.T  # (256, n_archetypes)


# ── MATLAB primitive equivalents ─────────────────────────────────────


def next_odd(x: float) -> int:
    """Mirror of ``next_odd.m``: smallest odd integer >= x.

    For x = 77 -> 77, for x = 76.8 -> 77, for x = 76 -> 77.
    """
    return int(2 * np.floor(np.ceil(x) / 2) + 1)


def _causal_moving_mean(v: np.ndarray, window: int) -> np.ndarray:
    """``movmean(v, [0, window-1], 'omitnan')``: one-sided causal mean."""
    n = len(v)
    c = np.cumsum(np.insert(v, 0, 0.0))
    idx = np.arange(1, n + 1)
    left = np.maximum(idx - window, 0)
    return (c[idx] - c[left]) / (idx - left)


def _delayed_moving_mean(v: np.ndarray, window: int) -> np.ndarray:
    """``movmean(v, [window-1, 0], 'omitnan')``: mean of the window ending
    ``window - 1`` samples in the past (truncated at the start edge)."""
    n = len(v)
    d = np.zeros(n)
    if window < 1 or n == 0:
        return d
    d[window - 1 :] = _causal_moving_mean(v[: n - window + 1], window)
    return d


def _centered_moving_stat(
    v: np.ndarray, window: int, stat: str = "var"
) -> np.ndarray:
    """Centered moving statistic (``movvar``/``movmedian``/``movmax``) with
    MATLAB's edge truncation.

    ``stat`` is ``"var"`` (sample variance, ddof=1), ``"median"``, or
    ``"max"``.  Window ``i`` covers ``v[max(0, i - window // 2) :
    min(n, i + window - window // 2)]``; edge windows use the available
    samples, exactly like MATLAB with a scalar window.  Vectorised: the
    variance uses windowed sums, median and max use ``scipy.ndimage`` on
    the interior and an explicit loop only on the ``window // 2`` samples
    at each edge.
    """
    from scipy import ndimage

    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    half = window // 2
    if n == 0:
        return np.empty(0)
    idx = np.arange(n)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx + window - half)
    if stat == "var":
        # Windowed sums around the global mean (limits cancellation).
        c = v - v.mean()
        s1 = np.concatenate(([0.0], np.cumsum(c)))
        s2 = np.concatenate(([0.0], np.cumsum(c * c)))
        m = (hi - lo).astype(np.float64)
        w1 = s1[hi] - s1[lo]
        w2 = s2[hi] - s2[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (w2 - w1 * w1 / m) / (m - 1.0)
        out[m < 2] = 0.0
        return np.maximum(out, 0.0)
    if stat == "median":
        filt = ndimage.median_filter
        reduce = np.median
    elif stat == "max":
        filt = ndimage.maximum_filter
        reduce = np.max
    else:  # pragma: no cover
        raise ValueError(f"unknown stat {stat!r}")
    # Interior samples see a full window [i - half, i + window - half),
    # which is ndimage's default placement (origin 0) for odd and even sizes.
    if stat == "median" and window % 2 == 0:
        # np.median of an even window averages the two middle values.
        out = 0.5 * (
            ndimage.rank_filter(v, window // 2 - 1, size=window, mode="nearest")
            + ndimage.rank_filter(v, window // 2, size=window, mode="nearest")
        )
    else:
        out = filt(v, size=window, mode="nearest")
    edge = [i for i in range(min(half, n))] + [
        i for i in range(max(half, n - (window - half) + 1), n)
    ]
    for i in edge:
        out[i] = reduce(v[lo[i]:hi[i]])
    return out


def _fill_mask_gaps(mask: np.ndarray, fill_length: int) -> np.ndarray:
    """Mirror of ``fill_mask_array_gaps.m``: zero runs of 1s of length
    <= ``fill_length``, the trailing run included (vectorised
    run-length encoding)."""
    mask = mask.copy()
    ones = np.concatenate(([0], (np.asarray(mask) == 1).astype(np.int8), [0]))
    d = np.diff(ones)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    for st, en in zip(starts, ends):
        if en - st <= fill_length:
            mask[st:en] = 0
    return mask


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """The intended ``smooth_mask_array`` S-curve: 3t^2 - 2t^3."""
    return t * t * (3.0 - 2.0 * t)


def _smooth_mask(mask: np.ndarray, transition_length: int) -> np.ndarray:
    """Mirror of ``smooth_mask_array.m`` (with the smoothstep profile)."""
    if transition_length < 1:
        return mask.astype(np.float64)
    mask = mask.astype(np.float64).copy()
    n = len(mask)
    # 1 -> 0 boundary at i: ramp mask[max(1, i - L - 1):i]
    transitions = np.diff(mask.astype(np.int64))
    for i in np.where(transitions == -1)[0] + 1:
        a = max(1, i - transition_length - 1)
        span = i - a
        if span < 2:
            continue
        g = _smoothstep(np.linspace(0.0, 1.0, span + 1))
        g = g[1:]
        mask[a:i] *= g
    # 0 -> 1 boundary at i: ramp mask[i : min(n, i + L + 1)]
    for i in np.where(transitions == 1)[0]:
        b = min(n, i + transition_length + 1)
        span = b - i
        if span < 2:
            continue
        g = _smoothstep(np.linspace(0.0, 1.0, span + 1))
        g = g[:-1]
        mask[i:b] *= g
    return mask


def _butter_lowpass_order(
    pw: float, rp_db: float, sw: float, sa_db: float
) -> int:
    """Minimal Butterworth order meeting the passband ripple spec."""
    for N in range(1, 41):
        b, a = _sig.butter(N, pw, btype="low")
        w = np.linspace(0.0, pw, 129)
        h = _sig.freqz(b, a, worN=w)[1]
        if np.max(-20.0 * np.log10(np.maximum(h, 1e-30))) <= rp_db:
            return N
    return 10


# ── Stage 2: clip removal and noise filtering ────────────────────────


def remove_large_clips(
    x: np.ndarray, fs: float, clip_ratio: float = 0.5
) -> np.ndarray:
    """Mirror of ``remove_large_clips.m`` (single-channel case).

    Marks stretches where >= ``clip_ratio`` of a 0.3 s one-sided window
    sits at the global signal max or min, fills short gaps, smooths the
    mask boundaries, and returns ``(x - masked median) * mask``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x
    clip_length = next_odd(0.3 * fs)
    fill_gap_length = int(round(fs))
    smooth_transition_length = int(round(fs))

    mx, mn = x.max(), x.min()
    clip = np.zeros(n)
    if mx != mn:
        clip[x >= mx] = 1.0
        clip[x <= mn] = -1.0
    abs_clip = np.abs(clip)
    if abs_clip.sum() == 0:
        return x  # nothing is pinned at the extremes

    mask = np.ones(n)
    causal = _causal_moving_mean(abs_clip, clip_length)
    delayed = _delayed_moving_mean(abs_clip, clip_length)
    mask[causal >= clip_ratio] = 0
    mask[delayed >= clip_ratio] = 0
    mask = _fill_mask_gaps(mask, fill_gap_length)
    mask = _smooth_mask(mask, smooth_transition_length)

    kept = mask > 0
    med = np.median(x[kept]) if kept.any() else 0.0
    return (x - med) * mask


def highpass_cheby2(
    x: np.ndarray, fs: float, pass_hz: float = 0.5, stop_hz: float = 0.25
) -> np.ndarray:
    """Mirror of ``get_hp_filter.m``: zero-phase Chebyshev-II high-pass.

    0.5 Hz pass edge with 0.1 dB ripple, 0.25 Hz stop edge with 60 dB
    attenuation; the filter order is the minimal one meeting the
    passband spec (``MatchExactly='passband'`` equivalent).
    """
    nyq = 0.5 * fs
    pw = pass_hz / nyq
    sw = stop_hz / nyq
    if pw >= 1.0:
        return np.asarray(x, dtype=np.float64)
    # Minimal order for the spec; scipy's cheby2 takes the *stopband* edge
    # as Wn, which cheb2ord returns. Second-order sections: the (b, a) form
    # of this narrow high-pass is numerically unstable (NaN output).
    N, wn = _sig.cheb2ord(wp=pw, ws=sw, gpass=0.1, gstop=60.0)
    sos = _sig.cheby2(N, 60.0, Wn=wn, btype="high", output="sos")
    return _sig.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def notch_line_noise(x: np.ndarray, fs: float, line_freq: float) -> np.ndarray:
    """Zero-phase notch at the line frequency (or its alias).

    Mirrors ``cut_to_resample_process.m:244-249``: notch at
    ``line_freq`` when ``fs > 2*line_freq``; at ``fs - line_freq`` when
    ``line_freq <= fs <= 2*line_freq`` (the 60 Hz alias); no notch when
    ``fs < line_freq``.
    """
    nyq = 0.5 * fs
    x = np.asarray(x, dtype=np.float64)
    if fs > 2 * line_freq:
        f = line_freq
    elif fs >= line_freq:
        f = fs - line_freq
    else:
        return x
    if not 0 < f < nyq:
        return x
    b, a = _sig.iirnotch(f / nyq, f / _NOTCH_BW_HZ)
    return _sig.filtfilt(b, a, x)


def _scaled_fft_spectrum(x: np.ndarray, fs: float, amp_floor: float) -> np.ndarray:
    """Mirror of ``scaled_fft.m``: one 2 s window, rFFT on the 0.5 Hz grid,
    normalized by its 13-bin moving median (floored at ``amp_floor/2``),
    with inf replaced by 1."""
    n_seg = int(round(fs / _FIND_NOISE_SPECTRUM_HZ))
    N = max(min(n_seg, len(x)), 2)
    seg = x[:N]
    spec = np.abs(np.fft.rfft(seg * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    # interpolate onto the 0.5 Hz grid
    grid = np.arange(0.0, 0.5 * fs + 0.25, _FIND_NOISE_SPECTRUM_HZ)
    mag = np.interp(grid, freqs, spec, right=0.0)
    # moving median over next_odd(6/0.5+1) = 13 bins, edge truncated
    med = _centered_moving_stat(mag, 13, "median")
    med = np.maximum(med, amp_floor / 2.0)
    out = mag / med
    out[~np.isfinite(out)] = 1.0
    return out


def find_and_remove_noise(
    x: np.ndarray,
    fs: float,
    max_add: int = _FIND_NOISE_MAX_ADD,
    min_ratio: float = _FIND_NOISE_MIN_RATIO,
) -> Tuple[np.ndarray, List[float]]:
    """Mirror of ``find_and_remove_noise.m``: iteratively notch the
    strongest non-ECG spectral peaks (>= 10 Hz, ratio >= ``min_ratio``
    over the median-scaled spectrum of 10 equal windows), up to
    ``max_add`` frequencies.  Returns ``(cleaned, freqs)``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    nyq = 0.5 * fs
    win = n // _FIND_NOISE_WINDOWS
    if win < int(_FIND_NOISE_AMP_SPAN_S * fs) or win < 1:
        return x, []

    removed: List[float] = []
    for _ in range(max_add):
        # amplitude floor (find_noise.m:31-44)
        amps = []
        for k in range(_FIND_NOISE_WINDOWS):
            seg = x[k * win : k * win + win]
            sec = seg[: int(_FIND_NOISE_AMP_SPAN_S * fs)]
            if len(sec) < 2:
                continue
            pk = _sig.find_peaks(sec, distance=2 * fs)[0]
            tr = _sig.find_peaks(-sec, distance=2 * fs)[0]
            medp = np.median(sec[pk]) if len(pk) else sec.max()
            medt = np.median(sec[tr]) if len(tr) else sec.min()
            amps.append(medp + medt)
        amp = max(_FIND_NOISE_AMP_FLOOR, float(np.median(amps))) if amps else _FIND_NOISE_AMP_FLOOR

        # per-window scaled spectra, averaged
        grid = np.arange(0.0, nyq + 0.25, _FIND_NOISE_SPECTRUM_HZ)
        mean_fft = np.zeros(len(grid))
        for k in range(_FIND_NOISE_WINDOWS):
            seg = x[k * win : k * win + win]
            mean_fft += _scaled_fft_spectrum(seg, fs, amp)
        mean_fft /= _FIND_NOISE_WINDOWS

        # candidate bins
        cand = np.where(
            (grid >= _FIND_NOISE_MIN_FREQ_HZ)
            & (grid < nyq - 1e-9)
            & (mean_fft >= min_ratio)
            & np.isfinite(mean_fft)
        )[0]
        if len(cand) == 0:
            break
        cand = cand[np.argsort(mean_fft[cand])[::-1]]
        f = float(grid[cand[0]])
        b, a = _sig.iirnotch(f / nyq, f / _NOTCH_BW_HZ)
        x = _sig.filtfilt(b, a, x)
        removed.append(f)
    return x, removed


# ── Stage 3: R-wave detection and normalization ──────────────────────


def local_variation(x: np.ndarray, fs: float) -> np.ndarray:
    """Mirror of ``calculate_local_variation.m``.

    ``lv = movvar(x, 17) / movmedian(movvar(x, 17), 257)``, zero where
    the numerator or denominator is degenerate, then ``log2(lv + 1)``.
    """
    x = np.asarray(x, dtype=np.float64)
    lv = _centered_moving_stat(x, _LV_LOCAL_SAMPLES, "var")
    g = _centered_moving_stat(lv, _LV_GROUP_SAMPLES, "median")
    out = np.where(
        (lv >= 1e-10) & (g >= 1e-10) & np.isfinite(g),
        lv / np.where(g > 0, g, 1.0),
        0.0,
    )
    return np.log2(np.maximum(out, 0.0) + 1.0)


def _lowpass_local_variation(lv: np.ndarray) -> np.ndarray:
    """Mirror of ``filter_local_variation.m``: zero-phase Butterworth LP,
    non-positive pre-filter positions forced back to 0, negatives clipped."""
    zero = lv <= 0
    b, a = _sig.butter(_butter_lowpass_order(8.0 / 128.0, 1.0, 10.0 / 128.0, 30.0),
                       _LV_LP_PASS_HZ / 128.0, btype="low")
    y = _sig.filtfilt(b, a, lv)
    y[zero] = 0.0
    return np.clip(y, 0.0, None)


def _correlate_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of (n, m) ``a`` against (m,) ``b``."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean()
    num = a @ b
    den = np.sqrt((a**2).sum(axis=1)) * np.sqrt((b**2).sum())
    out = np.zeros(a.shape[0])
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _build_template(beats: np.ndarray, archetypes: Optional[np.ndarray]) -> np.ndarray:
    """Mirror of ``use_archetypes_to_find_template.m`` + ``create_template.m``.

    Archetype-matched beats are robust-zscored, column-medians give the
    initial template, then up to 5 rounds drop beats whose |corr| with the
    template is below the rising threshold (k/5 * 0.5).  A second pass
    re-matches against the record-specific template.  With no archetypes
    (download failed), all beats are used directly (deviation 8).
    """
    if len(beats) == 0:
        return np.zeros(256)
    keep = np.ones(len(beats), dtype=bool)
    if archetypes is not None:
        # |corr(beat, archetype)| >= 0.5 for any archetype
        ok = np.zeros(len(beats), dtype=bool)
        for arch in archetypes.T:
            ok |= np.abs(_correlate_rows(beats, arch)) >= 0.5
        keep = keep & ok
    sel = beats[keep]
    if len(sel) == 0:
        sel = beats
    # Robust z-score of each beat (row): (beat - median) / (1.4826 * MAD).
    med = np.median(sel, axis=1, keepdims=True)
    mad = np.median(np.abs(sel - med), axis=1, keepdims=True) * 1.4826
    mad = np.where(mad > 1e-10, mad, 1.0)
    z = (sel - med) / mad
    template = np.median(z, axis=0)
    for k in range(1, 6):
        c = np.abs(_correlate_rows(z, template))
        keepk = c >= (0.1 * k) ** 0.5
        if keepk.all() or not keepk.any():
            break
        z = z[keepk]
        template = np.median(z, axis=0)
    # second pass: match against the record-specific template
    c = np.abs(_correlate_rows(z, template))
    z2 = z[c >= 0.5] if (c >= 0.5).any() else z
    return np.median(z2, axis=0)


def find_r_peaks(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    """R-wave locations via local-variation peaks + template correlation.

    ``findpeaks`` on the filtered local variation (min distance
    ``round(0.3*256)`` samples, min height 1.5), then per candidate the
    256-sample segment starting 100 samples before the peak; candidates
    whose |corr| with the template is >= 0.5 are R-peaks.
    """
    lv = _lowpass_local_variation(local_variation(x, 256.0))
    peaks = _sig.find_peaks(
        lv, distance=_LV_MIN_PEAK_SEPARATION, height=_LV_MIN_PEAK_HEIGHT
    )[0]
    if len(peaks) == 0:
        return np.array([], dtype=int)
    sel = (peaks >= 100) & (peaks < len(x) - 156)
    peaks = peaks[sel]
    if len(peaks) == 0:
        return np.array([], dtype=int)
    beats = np.stack([x[p - 100 : p + 156] for p in peaks])
    c = np.abs(_correlate_rows(beats, template))
    return peaks[c >= 0.5]


def normalization_factor(x: np.ndarray, peaks: np.ndarray) -> float:
    """Normalization factor: 2 x the 90th percentile of per-beat max |x|
    over the template-matched beats (see deviation 7 for the prctile
    rank)."""
    if len(peaks) == 0:
        return 1.0
    beatsmax = np.array([np.max(np.abs(x[p - 100 : p + 156])) for p in peaks])
    beatsmax = beatsmax[beatsmax > 0]
    if len(beatsmax) == 0:
        return 1.0
    return float(2.0 * np.percentile(beatsmax, 90.0))


# ── The step ─────────────────────────────────────────────────────────


class JonesECGPreprocess(PreprocessingStep):
    """Full Jones et al. (2024) ECG preprocessing in a single step.

    Turn one raw single-lead ECG channel (whole night, any fs) into the
    network input of :class:`physioex.models.jones_ecg.JonesECGNet`:
    256 Hz, scaled by the per-night R-wave normalization factor and
    clipped to ``[-1, 1]``.

    Order of operations (MATLAB stage 2 -> stage 3 -> stage 5 scaling):

    1. ``remove_large_clips`` — loose-lead clipping mask removal;
    2. zero-phase Chebyshev-II high-pass (0.5 / 0.25 Hz, 0.1 / 60 dB);
    3. line-noise notch at ``line_freq`` (with the aliasing rules) plus
       up to 10 iterative extra-noise notches (if ``find_noise``);
    4. resample to 256 Hz with ``resample_poly`` (p/q from
       ``Fraction(256/fs).limit_denominator()``);
    5. R-wave detection (local variation + template; archetypes when
       available) and normalization factor = 2 x P90 of per-beat max |x|;
    6. divide by the factor, clip to ``[-1, 1]``.

    Parameters
    ----------
    line_freq : float
        Mains frequency to notch (60.0 in the authors' US data, 50.0 in
        EU data).
    remove_clips : bool
        Apply the loose-lead clip-mask removal.
    find_noise : bool
        Run the iterative extra-noise notch search.
    archetypes_path : str, optional
        Path to ``archetypes.mat``.  ``None`` (default) downloads it from
        the authors' repository on first use.  Deliberately excluded from
        ``spec()`` — a path must not change the cache hash.

    Degenerate inputs (all zeros, fully clipped, no beats found) return
    zeros of the correct length with a warning instead of raising.
    """

    def __init__(
        self,
        line_freq: float = 60.0,
        remove_clips: bool = True,
        find_noise: bool = True,
        archetypes_path: Optional[str] = None,
    ):
        if line_freq <= 0:
            raise ValueError(f"line_freq must be > 0; got {line_freq}")
        self.line_freq = float(line_freq)
        self.remove_clips = bool(remove_clips)
        self.find_noise = bool(find_noise)
        self.archetypes_path = archetypes_path

    def spec(self) -> str:
        # ``impl`` is bumped whenever the numerical output changes for the same
        # parameters, so caches built by an older implementation are not reused
        # (2: SOS high-pass and per-beat template z-score).
        return self._build_spec(
            impl=2,
            line_freq=self.line_freq,
            remove_clips=self.remove_clips,
            find_noise=self.find_noise,
        )

    def compile(self, fs_in: float, header: Optional[dict] = None) -> CompiledStep:
        fs_in = float(fs_in)
        if fs_in <= 0:
            raise ValueError(f"fs_in must be > 0; got {fs_in}")
        line_freq = self.line_freq
        remove_clips = self.remove_clips
        find_noise = self.find_noise
        archetypes_path = self.archetypes_path

        archetypes = _load_archetypes(archetypes_path)

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float64).ravel()
            if len(x) == 0 or not np.any(x):
                logger.warning(
                    "JonesECGPreprocess: input signal is empty/all-zero; "
                    "returning zeros"
                )
                return np.zeros(
                    int(round(len(x) * JONES_ECG_FS_OUT / fs_in)), dtype=np.float32
                )

            if remove_clips:
                x = remove_large_clips(x, fs_in)
            if not np.any(x):
                logger.warning(
                    "JonesECGPreprocess: signal fully clipped out by "
                    "remove_large_clips; returning zeros"
                )
                return np.zeros(
                    int(round(len(x) * JONES_ECG_FS_OUT / fs_in)),
                    dtype=np.float32,
                )

            x = highpass_cheby2(x, fs_in)
            x = notch_line_noise(x, fs_in, line_freq)
            if find_noise:
                x, _ = find_and_remove_noise(x, fs_in)

            # MATLAB: [p, q] = rat(256 / fs); resample(x, p, q).
            ratio = (Fraction(JONES_ECG_FS_OUT) / Fraction(fs_in)).limit_denominator(10000)
            p, q = ratio.numerator, ratio.denominator
            if p != q:
                x = _sig.resample_poly(x, p, q, window=("kaiser", 8.0))

            lv = _lowpass_local_variation(local_variation(x, 256.0))
            cand = _sig.find_peaks(
                lv, distance=_LV_MIN_PEAK_SEPARATION, height=_LV_MIN_PEAK_HEIGHT
            )[0]
            cand = cand[(cand >= 100) & (cand < len(x) - 156)]
            if len(cand) == 0:
                logger.warning(
                    "JonesECGPreprocess: no R-wave candidates detected; "
                    "returning zeros"
                )
                return np.zeros(len(x), dtype=np.float32)
            cand_beats = np.stack([x[c - 100 : c + 156] for c in cand])
            template = _build_template(cand_beats, archetypes)
            peaks = find_r_peaks(x, template)

            if len(peaks) == 0:
                logger.warning(
                    "JonesECGPreprocess: no R-peaks detected; returning zeros"
                )
                return np.zeros(len(x), dtype=np.float32)

            factor = normalization_factor(x, peaks)
            if not np.isfinite(factor) or factor <= 0:
                logger.warning(
                    "JonesECGPreprocess: degenerate normalization factor "
                    f"({factor}); returning zeros"
                )
                return np.zeros(len(x), dtype=np.float32)

            y = np.clip(x / factor, -1.0, 1.0)
            return y.astype(np.float32)

        return CompiledStep(apply=apply, fs_out=JONES_ECG_FS_OUT)
