"""Stationary background of psg_gen (SPEC section 2, ``background.py``).

The background is the stationary floor ``b(t)`` of the generative model, per 4 s window (1 s
hop): the 20th percentile of the per-segment (1 s, hop 0.5 s) log band power of the event-free
signal, taken over the window and a 6 s context, with the event supports masked out and a
closed-form Gamma correction of the quantile bias (R1). Not learned; batched torch on the
model device. The synthesised floor (:func:`synth_background`) is band-shaped noise with those
powers; the residual ``eps`` is measured against it (:func:`floor_excess`).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from scipy.special import gammaincinv, polygamma

from physioex.models.msvq import BANDS as BG_BANDS
from physioex.models.psg_gen.dicts import FS

EPOCH = 3000


def seg_band_logpow(x: torch.Tensor, fs: int = FS, bands=BG_BANDS, seg: float = 1.0,
                    hop: float = 0.5) -> torch.Tensor:
    """(B, T) -> (B, n_seg, n_bands): log10 band power (power per Hz) of Hann segments of
    ``seg`` s with a ``hop`` s step."""
    n, h = int(seg * fs), int(hop * fs)
    w = torch.hann_window(n, device=x.device, dtype=x.dtype)
    segs = x.unfold(-1, n, h) * w                        # (B, S, n)
    p = torch.fft.rfft(segs, dim=-1).abs() ** 2 / (w ** 2).sum()
    f = torch.fft.rfftfreq(n, 1 / fs, device=x.device)
    out = torch.stack([p[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in bands], -1)
    return torch.log10(out + 1e-10)


def q20_gamma_factor(n_bins: int, q: float = 0.2) -> float:
    """Closed-form correction of the q-quantile of a band power: a mean of ``n_bins``
    periodogram bins of a Gaussian background is Gamma(M, P/M), whose q-quantile is
    P * gammaincinv(M, q) / M, so the quantile estimate must be multiplied by M / that."""
    return float(n_bins / gammaincinv(n_bins, q))


def background_estimate(x: torch.Tensor, mask: torch.Tensor | None, fs: int = FS,
                        bands=BG_BANDS, win: float = 4.0, hop: float = 1.0, ctx: float = 6.0,
                        q: float = 0.2, correct: bool = True) -> dict:
    """logP[w, band] = q-quantile of the per-segment log band power of ``x`` over the window and
    the ``ctx`` s context, with segments having >= 50 % of their samples on event supports
    (``mask`` True) dropped; a window with < 2 valid segments falls back to the epoch-level
    quantile. ``correct`` adds the Gamma correction of the quantile bias.
    -> {"logP": (B, n_win, n_bands), "centers": (n_win,) seconds, "n_valid": (B, n_win, n_bands)}
    """
    B, T = x.shape
    n, h = fs, fs // 2                                    # 1 s segments, 0.5 s hop (fixed)
    lp = seg_band_logpow(x, fs, bands)                    # (B, S, nb)
    S, nb = lp.shape[1], lp.shape[2]
    sc = (torch.arange(S, device=x.device) * h + n / 2) / fs
    if mask is not None:
        clean = mask.unfold(-1, n, h).float().mean(-1) < 0.5
    else:
        clean = torch.ones(B, S, dtype=torch.bool, device=x.device)
    wlen, wh = int(win * fs), int(hop * fs)
    n_win = (T - wlen) // wh + 1
    wc = (torch.arange(n_win, device=x.device) * wh + wlen / 2) / fs
    span = sc[None, :] >= wc[:, None] - (win + ctx) / 2
    span &= sc[None, :] <= wc[:, None] + (win + ctx) / 2
    logP = torch.zeros(B, n_win, nb, device=x.device)
    n_valid = torch.zeros(B, n_win, nb, dtype=torch.long, device=x.device)
    allseg = torch.ones(S, dtype=torch.bool, device=x.device)
    for wi in range(n_win):
        for b in range(B):
            v = span[wi] & clean[b]
            n_valid[b, wi] = int(v.sum())
            if int(v.sum()) < 2:
                cb = clean[b]
                v = cb if int(cb.sum()) >= 2 else allseg
            logP[b, wi] = lp[b, v].quantile(q, dim=0)
    if correct:
        f1 = torch.fft.rfftfreq(n, 1 / fs, device=x.device)
        off = [math.log10(q20_gamma_factor(int(((f1 >= lo) & (f1 < hi)).sum()), q))
               for lo, hi in bands]
        logP = logP + torch.tensor(off, device=x.device, dtype=logP.dtype)[None, None]  # off is already log10
    return {"logP": logP, "centers": wc, "n_valid": n_valid}


def whitening_filters(x: torch.Tensor, L: int, fs: int = FS, q: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust whitening of an epoch batch: the q-quantile PSD over 1 s Hann segments (hop 0.5 s),
    smoothed over +-1 Hz and floored at 1e-3 of the max, and the zero-phase FIR of length L with
    frequency response 1/sqrt(PSD). -> (W (B, L), psd (B, F1))."""
    n = fs
    w = torch.hann_window(n, device=x.device, dtype=x.dtype)
    segs = x.unfold(-1, n, n // 2) * w
    psd = torch.fft.rfft(segs, dim=-1).abs() ** 2
    psd = psd.quantile(q, dim=1)
    psd = F.avg_pool1d(psd[:, None], 3, stride=1, padding=1)[:, 0]
    psd = psd.clamp_min(1e-3 * psd.amax(-1, keepdim=True))
    f1 = torch.fft.rfftfreq(n, 1 / fs, device=x.device)

    def interp(fq: torch.Tensor) -> torch.Tensor:
        idx = torch.bucketize(fq, f1).clamp(1, len(f1) - 1)
        t = ((fq - f1[idx - 1]) / (f1[idx] - f1[idx - 1] + 1e-9)).clamp(0, 1)
        return (1 - t)[None] * psd[:, idx - 1] + t[None] * psd[:, idx]

    m = 4 * L                                             # 4x oversampled DFT, centred at DC
    Hp = interp(torch.fft.rfftfreq(m, 1 / fs, device=x.device)) ** -0.5
    Hf = torch.cat([Hp[:, : m // 2], Hp[:, m // 2: m // 2 + 1], Hp.flip(1)[:, 1: m // 2]], dim=-1)
    hh = torch.fft.ifft(Hf).real
    c = (L - 1) // 2
    W = hh[:, (torch.arange(L, device=x.device) - c) % m]
    return 0.5 * (W + W.flip(1)), psd


def whiten(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Zero-phase FFT convolution of x (B, T) with the symmetric filter w (B, L), same length."""
    n, L = x.shape[-1], w.shape[-1]
    m = 1 << (n + L - 2).bit_length()
    c = (L - 1) // 2                                      # centre tap of the symmetric filter
    y = torch.fft.irfft(torch.fft.rfft(x, n=m) * torch.fft.rfft(w, n=m), n=m)
    return y[..., c:c + n]                                # out[i] = sum_j w[j] x[i + c - j]


def gain_g(x_night: torch.Tensor, fs: int = FS, win: float = 3.0, span_min: float = 20.0,
           q: float = 0.2) -> torch.Tensor:
    """R7 gain of a night: (N,) samples -> (n_span,) one scalar per ``span_min``-min span, the
    q-quantile (as an amplitude) of the log-RMS of the ``win``-s windows inside the span, the
    last span padded; the night gain is the median."""
    n = int(win * fs)
    v = x_night.reshape(-1)
    m = v.numel() // n
    rms = torch.sqrt((v[:m * n].reshape(m, n) ** 2).mean(-1))
    lr = torch.log10(rms.clamp_min(1e-9))
    span = max(int(span_min * fs / win), 1)               # windows per span
    step = max(span // 2, 1)
    out = []
    for s in range(0, max(m - span + 1, 1), step):
        blk = lr[s: s + span]
        if blk.numel() < span:
            blk = torch.cat([blk, blk[-1:].expand(span - blk.numel())])
        out.append(10.0 ** blk.quantile(q))
    return torch.tensor(out, dtype=torch.float32)


def change_indicator(logP: torch.Tensor, z: float = 3.0, lag: int = 10) -> torch.Tensor:
    """(B, n_win, n_bands) -> (B, n_win) in [0, 1]: 1 where the standardised jump of the band
    vector between consecutive windows exceeds z sigma (R10). The per-band noise sd of the log
    power of a Gamma(M) band is sqrt(2 * trigamma(M)) for the difference of two windows
    (closed form), M = number of bins of the band. Windows ``lag`` apart are compared (default
    10 = (win + ctx) / hop, the span of one estimate): adjacent windows share most of their
    context, so a floor change spreads over many of them and a lag-1 jump stays under z. The
    jump is assigned to the middle window between the two."""
    f1 = torch.fft.rfftfreq(FS, 1 / FS, device=logP.device)
    M = torch.tensor([float(((f1 >= lo) & (f1 < hi)).sum()) for lo, hi in BG_BANDS],
                     device=logP.device)
    sd = torch.sqrt(torch.as_tensor(2.0 * polygamma(1, M.cpu().numpy()),
                                    device=logP.device))[None, None]
    B, n_win, _ = logP.shape
    lag = max(1, min(lag, n_win - 1))
    j = (torch.abs(logP[:, lag:] - logP[:, :-lag]) / sd).pow(2).mean(-1).sqrt()
    out = torch.zeros(B, n_win, device=logP.device, dtype=logP.dtype)
    out[:, lag // 2: lag // 2 + j.shape[1]] = (j > z).to(logP.dtype)
    return out


def synth_background(logP: torch.Tensor, fs: int = FS, T: int = EPOCH, win: float = 4.0,
                     hop: float = 1.0, bands=BG_BANDS, seed: int | None = None) -> torch.Tensor:
    """Band-shaped noise floor with the powers 10**logP (B, n_win, n_bands) -> (B, T): per
    window a random band-limited Gaussian with the target powers (calibrated on the windowed
    segment), Hann-windowed and overlap-added, normalised by sqrt(sum w^2)."""
    B, n_win, nb = logP.shape
    dev, dt = logP.device, logP.dtype
    gen = torch.Generator(device=dev.type).manual_seed(0 if seed is None else int(seed))
    W, H = int(win * fs), int(hop * fs)
    w = torch.hann_window(W, device=dev, dtype=dt)
    f = torch.fft.rfftfreq(W, 1 / fs, device=dev)
    P = 10.0 ** logP
    idx = (torch.arange(n_win, device=dev) * H)[:, None] + torch.arange(W, device=dev)[None, :]
    rows = torch.arange(n_win, device=dev)[:, None]
    Y = torch.zeros(n_win, T, device=dev, dtype=dt)
    Y[rows, idx] = (w ** 2)[None]
    W2 = Y.sum(0)                                         # (T,) overlap weight, ~1.5 in the interior
    y = torch.zeros(B, T, device=dev, dtype=dt)
    for bnd in range(nb):
        lo, hi = bands[bnd]
        m = (f >= lo) & (f < hi)
        S = torch.randn((B, n_win, int(m.sum())), generator=gen, device=dev) \
            + 1j * torch.randn((B, n_win, int(m.sum())), generator=gen, device=dev)
        Sp = torch.zeros((B, n_win, W // 2 + 1), device=dev, dtype=torch.cfloat)
        Sp[..., m] = S
        z = torch.fft.irfft(Sp, n=W) * w                  # (B, n_win, W) windowed segment
        cur = (torch.fft.rfft(z, n=W).abs() ** 2 / (w ** 2).sum())[..., m].mean(-1)
        z = z * torch.sqrt(P[:, :, bnd:bnd + 1] / cur.unsqueeze(-1).clamp_min(1e-12))
        Z = torch.zeros(B, n_win, T, device=dev, dtype=dt)
        Z[:, rows, idx] = z
        y += Z.sum(1)
    # independent segments s_i windowed once (z_i = w s_i): sum_i w s_i / sqrt(sum_i w^2) has the
    # power of s at every sample, and stays finite at the epoch edges where w -> 0
    return y / W2.clamp_min(1e-12).sqrt()[None]


def floor_excess(resid: torch.Tensor, logP: torch.Tensor, fs: int = FS, bands=BG_BANDS,
                 win: float = 4.0, hop: float = 1.0) -> dict:
    """eps of the residual (x - events) against the floor ``logP``: per window the measured
    band powers of the residual, the floor energy and the excess. -> {"eps": (B, n_win) excess
    energy above the floor / window energy, "floor": (B, n_win) floor energy, "excess":
    (B, n_win, n_bands) log10 ratio}.
    """
    B, T = resid.shape
    W, H = int(win * fs), int(hop * fs)
    n_win = (T - W) // H + 1
    w = torch.hann_window(W, device=resid.device, dtype=resid.dtype)
    f = torch.fft.rfftfreq(W, 1 / fs, device=resid.device)
    X = torch.fft.rfft(resid.unfold(-1, W, H) * w, dim=-1)
    X = X.abs() ** 2 / (w ** 2).sum()                     # (B, n_win, F) power per Hz
    m = torch.stack([(f >= lo) & (f < hi) for lo, hi in bands])    # (nb, F)
    meas = torch.stack([X[..., m[i]].mean(-1) for i in range(m.shape[0])], -1)
    P = 10.0 ** logP
    E = X[..., (f >= 0.5) & (f < 30.0)].sum(-1)
    Fl = (P * m.sum(-1)[None, None]).sum(-1)              # (B, n_win) floor energy
    return {
        "eps": ((E - Fl).clamp_min(0) / E.clamp_min(1e-12)),
        "floor": Fl,
        "excess": torch.log10(meas / P.clamp_min(1e-30)),
    }
