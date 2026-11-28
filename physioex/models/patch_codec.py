"""PatchCodec: tokenizer of clean EEG on 2 s patches (plan of 26.11.2026, xmai-lab/sleep-lens#17, S2 #19).

One residual-VQ code stack per non-overlapping 2 s patch (200 samples at 100 Hz):

* **encoder**, local to the patch: it sees the patch alone (no neighbours, so the same 2 s get the
  same codes wherever they sit), whitened by the fixed filter below with a reflected history inside
  the patch, then SEANet stages (strides 2, 4, 5: 200 samples -> 5 frames), flattened and projected
  to ``dim``;
* **RVQ**: :class:`~physioex.models.msae_vq.ResidualVQ` with ``codes`` per level and up to
  ``levels`` levels, EMA codebooks, dead-code restart, quantizer dropout (with probability
  ``q_dropout`` a random prefix of levels), so one model serves every prefix;
* **decoder**: the latent of each patch is unfolded into 5 frames of 0.4 s, then convolutions over the
  whole sequence of patches upsample to 100 Hz (strides 5, 4, 2) with transposed convolutions
  (``upsample="convT"``). They leave a weak pattern locked to the patch grid on the 12.5 Hz harmonics
  (pilot of 26.11.2026). Repeating every sample ``s`` times and convolving (``"resize"``) was tried in
  job 1669985 and stopped early: after 2,500 steps it had a larger locked pattern (12.5 Hz, 32-35 Hz) and
  a slower warm-up. The receptive field is shorter than
  ±200 samples, so the output inside patch ``i`` depends only on the codes of patches ``i - 1``, ``i``
  and ``i + 1`` (``tests/models/test_patch_codec.py``). An edit of the codes of a span changes the
  signal only within that span ± 1 patch. Adjacent patches are drawn by the same convolutions across
  their boundary; the seam ratio of :mod:`physioex.train.bin.patch_codec` measures what is left.

Whitening (:class:`Whitener`): LPC prediction-error filter ``A(z)`` (order 32, minimum phase by
construction) fitted on the mean PSD of the training pool. The PSD is held constant outside
0.5-35 Hz, so the filter does not amplify the stop bands of the clean channel. The filter feeds the
encoder and weighs the loss: MSE and multi-resolution STFT are computed on ``A * x_hat`` against
``A * x``, so every band weighs about the same (on the raw signal ~78 % of the variance of a second lies
below 4 Hz). The decoder draws ``x_hat`` in the units of the input. Departure from the plan text
("de-whitening after the decoder"): no inverse all-pole filter follows the decoder, because its
infinite tail would carry an edit beyond ± 1 patch; the loss is still computed in the whitened domain.

Revision 2 (27.11.2026, after the negative S2 gate of the first codec, xmai-lab/sleep-lens#19): fidelity
first, bits are free. With one 128-d latent per patch the RVQ gained ~0.5 dB per 4 levels at 28-32 levels, and a
geometric extrapolation of ``1 - R^2`` over the levels stops near whitened R^2 0.87, while every code was in use
(perplexity ~700-740 of 1024). Options of the revision, all off by default so earlier runs load unchanged:

* ``quantizer="fsq"``: :class:`FSQ` instead of the RVQ. The encoder latent (``dim``) is projected to
  ``levels * fsq_group`` dimensions, each rounded to one of ``fsq_levels`` values; ``fsq_group`` dimensions
  make one code, so a patch still has ``levels`` codes (``levels`` counts groups, not residual levels);
* ``whiten="signal"``: the whitening filter runs over the whole signal with its true history instead of a
  history reflected inside every patch, so the first 0.32 s of a patch are whitened correctly. The codes of
  patch ``i`` then also depend on the last 32 samples of patch ``i - 1``;
* ``ctx_dec``: :class:`PatchContext` on the quantised latents before the decoder, output inside patch ``i``
  from the codes of ``i - ctx_dec - 1 .. i + ctx_dec + 1`` (``receptive_patches``); an edit of the codes of a
  span changes the signal within that span ± ``receptive_patches``;
* ``ctx_enc``: the same on the encoder latents before quantisation. The codes of a patch then depend on its
  neighbours (an event leaks into the codes around it), which the plan's surprise and removal steps pay for.

Revision 3 (27.11.2026, after the pilots P0-P4 of xmai-lab/sleep-lens#24). Measured on held-out nights: without
rounding, the trained codecs reach whitened R^2 0.89 (RVQ, 128-d) and 0.83 (FSQ, 216 dims), and the rounding costs
0.04 and 0.015 on top. The cap is the autoencoder, not the bits. A linear KLT of the whitened patches keeps 0.875 of
the variance at 128 components and 0.975 at 160. The FSQ with 108 dimensions saturated its ``tanh`` bound from
the first steps (every value at the bound, no gradient). Options, all off by default:

* ``dec_out="white"``: the decoder draws the whitened patch ``w_hat``; the loss compares it with ``A * x``
  directly, and ``x_hat = gain / A(z) * w_hat`` by a fixed causal FIR of ``INV_TAPS`` samples
  (:meth:`Whitener.inverse`). The inverse of the pool filter keeps 4.5e-7 of its impulse-response energy beyond
  1 s and 1.2e-11 beyond 2 s, so the truncated inverse is exact in practice and an edit still stays local: it
  reaches at most ``INV_TAPS - 1`` samples further (``receptive_patches`` grows by 2);
* ``arch="skip"``: a linear path from the whitened patch to the latent and from the latent to ``w_hat`` (needs
  ``dec_out="white"``), added to the convolutional ones, so the linear (KLT-like) solution is in reach from the
  start; ``arch="linear"``: only the linear paths, the reference transform codec;
* ``fsq_norm``: batch normalisation without affine parameters on the FSQ projection before the bound (running
  statistics in eval). It keeps the inputs of ``tanh`` near unit scale per dimension without removing the
  amplitude of a patch, as a per-patch layer norm would;
* ``fsq_scale``: multiplies the normalised input of ``tanh`` (default 1). At unit scale, a Gaussian latent puts
  30 % of its values in the two outer cells of 9 levels, whose width is unbounded: the rounding error after the
  best decoder (cell centroids) is 0.064 of the latent variance (Monte Carlo, 2 M samples). At 0.5, the
  ``tanh`` is close to the MSE-optimal compander of a Gaussian (Panter-Dite, ``2 Phi(u / sqrt 3) - 1``): 0.029
  with 9 levels, 0.020 with 11, against 0.050 with 11 levels at unit scale.

Revision 4 (28.11.2026, after the pilots Q2 and Q3s of xmai-lab/sleep-lens#24): :class:`TransformCodec`, a
gain-shape transform codec without training (KLT of the pool, the same relative step for every coefficient), as
the reference that a trained codec has to beat at the same bits. On 8 DCSM nights, the linear FSQ codec (Q3s)
saturated on loud patches: relative error 0.21-0.28 on the loudest 1 %, with 5-24 % of the ``tanh`` inputs at the
bound.

Input:``(B, n)`` clean EEG at 100 Hz, ``RobustScale`` units (cache of the clean channels); ``n`` is
cut to a multiple of 200. Sleep stages and concept labels are never used.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physioex.models.msae import MultiScaleStem, _down, _ResUnit, stem_kernels, stft_losses
from physioex.models.msae_vq import ResidualVQ
from physioex.models.msvq import _r2, code_stats
from physioex.train.trainer import Trainer

FS = 100
PATCH = 200  # 2 s
ENC_STRIDES = (2, 4, 5)  # 200 samples -> 5 frames
DEC_STRIDES = (5, 4, 2)  # 5 frames of 40 samples -> 200 samples
FRAMES = PATCH // math.prod(ENC_STRIDES)
assert FRAMES * math.prod(DEC_STRIDES) == PATCH
DEC_DILATIONS = ((1,), (1, 3), (1, 3, 9))  # residual units after each upsampling (receptive field < 200)
STFT_WINDOWS = (25, 50, 100, 200, 400)
BAND = (0.5, 35.0)
ZERO_RUN = 100  # crops with >= 1 s of exact zeros (padding beyond the recording) are dropped
INV_TAPS = 256  # causal FIR of the inverse whitener (dec_out="white")


# ----------------------------------------------------------------------------
# Whitening
# ----------------------------------------------------------------------------


def levinson(r: np.ndarray, order: int):
    """Levinson-Durbin on the autocorrelation ``r[0..order]``: prediction-error filter ``a``
    (``a[0] = 1``, ``e[n] = sum_k a[k] x[n - k]``), final error power and reflection coefficients
    (all ``|k| < 1`` for a positive definite ``r``: ``A(z)`` is minimum phase)."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = float(r[0])
    ks = np.zeros(order)
    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / err
        a[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a[i] = k
        err *= 1.0 - k * k
        ks[i - 1] = k
    return a, err, ks


class Whitener(nn.Module):
    """Fixed causal FIR ``e = A(z) x / gain`` (see module docstring). ``fit_psd`` sets ``A`` from a
    mean PSD; :func:`fit_whitener` also sets ``gain`` so that the whitened training signal has unit
    standard deviation."""

    def __init__(self, order: int = 32, fs: int = FS, band=BAND):
        super().__init__()
        self.order, self.fs, self.band = order, fs, tuple(band)
        a = torch.zeros(order + 1)
        a[0] = 1.0
        self.register_buffer("a", a)
        self.register_buffer("gain", torch.ones(()))

    @torch.no_grad()
    def fit_psd(self, freqs: np.ndarray, psd: np.ndarray) -> dict:
        lo, hi = self.band
        p = np.asarray(psd, np.float64).copy()
        i_lo = int(np.searchsorted(freqs, lo))
        i_hi = int(np.searchsorted(freqs, hi, side="right")) - 1
        p[:i_lo], p[i_hi + 1:] = p[i_lo], p[i_hi]
        r = np.fft.irfft(p)
        a, err, ks = levinson(r[: self.order + 1], self.order)
        self.a.copy_(torch.from_numpy(a).float())
        return {"order": self.order, "max_abs_reflection": float(np.abs(ks).max()),
                "prediction_gain_db": float(10 * np.log10(r[0] / err))}

    def response(self, freqs: np.ndarray) -> np.ndarray:
        """``|A(f)| / gain`` at ``freqs`` (Hz)."""
        a = self.a.double().cpu().numpy()
        z = np.exp(-2j * np.pi * np.outer(freqs, np.arange(len(a))) / self.fs)
        return np.abs(z @ a) / float(self.gain)

    def forward(self, x: torch.Tensor, pad: str = "reflect") -> torch.Tensor:
        """``(..., n) -> (..., n)``. The first ``order`` samples see a reflected history
        (``pad="reflect"``, needs ``n > order``) or zeros (``"zeros"``)."""
        shape = x.shape
        h = x.reshape(-1, 1, shape[-1]).float()
        h = F.pad(h, (self.order, 0), mode="reflect" if pad == "reflect" else "constant")
        e = F.conv1d(h, self.a.flip(0).view(1, 1, -1))  # conv1d correlates: flipped taps convolve
        return (e / self.gain).reshape(shape)

    def inverse(self, e: torch.Tensor, taps: int = INV_TAPS) -> torch.Tensor:
        """``(..., n) -> (..., n)``: ``gain / A(z)`` applied to ``e`` as a causal FIR of ``taps`` samples (zero
        history). ``A`` is minimum phase, so ``1 / A`` is stable and its impulse response decays; ``taps``
        bounds how far an edit of ``e`` moves the output."""
        n_fft = 4096
        h = torch.fft.irfft(float(self.gain) / torch.fft.rfft(self.a.double(), n=n_fft), n=n_fft)[:taps]
        shape = e.shape
        y = F.pad(e.reshape(-1, 1, shape[-1]).float(), (taps - 1, 0))
        return F.conv1d(y, h.float().flip(0).view(1, 1, -1)).reshape(shape)


def has_zero_run(x: torch.Tensor, run: int = ZERO_RUN) -> torch.Tensor:
    """``(B, n) -> (B,)`` True where ``x`` holds ``run`` consecutive exact zeros."""
    z = (x == 0).float()[:, None]
    return (F.avg_pool1d(z, run, stride=1)[:, 0] > 1 - 1e-6).any(1)


@torch.no_grad()
def welch_psd(x: torch.Tensor, fs: int = FS, seg_sec: float = 4.0):
    """Mean Welch PSD (Hann, 50 % overlap; arbitrary units) of ``(B, n)`` -> ``(freqs, psd)``."""
    seg = int(seg_sec * fs)
    w = torch.hann_window(seg, dtype=torch.float64, device=x.device)
    s = x.double().unfold(1, seg, seg // 2) * w
    p = torch.fft.rfft(s, dim=-1).abs().pow(2).mean((0, 1)) / w.pow(2).sum()
    return np.fft.rfftfreq(seg, 1 / fs), p.cpu().numpy()


@torch.no_grad()
def fit_whitener(model: "PatchCodec", loader, max_batches: int = 50) -> dict:
    """Fit ``model.whitener`` on the mean PSD of ``max_batches`` training batches, then its gain."""
    from physioex.data.collate import stack_channels

    acc, n, keep = 0.0, 0, []
    freqs = None
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = stack_channels(batch)[:, :, 0].float()
        x = x.reshape(len(x), -1)
        x = x[~has_zero_run(x)]
        if not len(x):
            continue
        freqs, p = welch_psd(x, model.fs)
        acc, n = acc + p * len(x), n + len(x)
        keep.append(x[:16])
    assert n, "fit_whitener: no window without zero runs"
    info = model.whitener.fit_psd(freqs, acc / n)
    model.whitener.gain.fill_(1.0)
    dev = model.whitener.a.device
    e = model.whitener(torch.cat(keep).to(dev))
    model.whitener.gain.fill_(float(e[:, model.whitener.order:].std()))
    info.update(windows=int(n), gain=float(model.whitener.gain))
    return info


# ----------------------------------------------------------------------------
# Encoder, decoder, codec
# ----------------------------------------------------------------------------


class PatchEncoder(nn.Module):
    """``(M, 200)`` whitened patches -> ``(M, dim)``; nothing crosses the patch boundary."""

    def __init__(self, width: int = 64, dim: int = 128, fs: int = FS):
        super().__init__()
        ch = [width * 2 ** i for i in range(len(ENC_STRIDES) + 1)]
        self.stem = MultiScaleStem(width, stem_kernels(fs))
        self.down = nn.Sequential(*[_down(ch[i], ch[i + 1], s) for i, s in enumerate(ENC_STRIDES)])
        self.out = nn.Sequential(nn.ELU(), nn.Flatten(), nn.Linear(ch[-1] * FRAMES, dim))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.out(self.down(self.stem(p[:, None])))


def _up(c_in: int, c_out: int, s: int, dilations, mode: str = "convT") -> nn.Sequential:
    """Upsampling by ``s``. ``"resize"``: repeat each sample ``s`` times, then a convolution of ``s`` taps
    (``s + 1`` when ``s`` is even): a constant input stays constant and each output sees at most two input
    samples. ``"convT"``: transposed convolution of ``2 s`` taps (pilot of 26.11.2026)."""
    if mode == "resize":
        k = s if s % 2 else s + 1
        up = [nn.Upsample(scale_factor=s, mode="nearest"), nn.Conv1d(c_in, c_out, k, padding=k // 2)]
    elif mode == "convT":
        p = (s + 1) // 2
        up = [nn.ConvTranspose1d(c_in, c_out, 2 * s, stride=s, padding=p, output_padding=2 * p - s)]
    else:
        raise ValueError(f"upsample {mode!r}: 'resize' or 'convT'")
    return nn.Sequential(nn.ELU(), *up, *[_ResUnit(c_out, d) for d in dilations])


class PatchDecoder(nn.Module):
    """``(B, P, dim) -> (B, P * 200)``, receptive field shorter than ±200 samples."""

    def __init__(self, width: int = 64, dim: int = 128, upsample: str = "convT"):
        super().__init__()
        ch = [width * 2 ** i for i in range(len(DEC_STRIDES), -1, -1)]
        self.c0 = ch[0]
        self.unfold = nn.Linear(dim, ch[0] * FRAMES)
        self.frame = nn.Conv1d(ch[0], ch[0], 3, padding=1)
        self.up = nn.Sequential(*[_up(ch[i], ch[i + 1], s, DEC_DILATIONS[i], upsample)
                                  for i, s in enumerate(DEC_STRIDES)])
        self.post = nn.Sequential(nn.ELU(), nn.Conv1d(ch[-1], 1, 7, padding=3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, P, _ = z.shape
        h = self.unfold(z).view(B, P, self.c0, FRAMES).permute(0, 2, 1, 3).reshape(B, self.c0, P * FRAMES)
        return self.post(self.up(self.frame(h)))[:, 0]


class LinearPatchDecoder(nn.Module):
    """``(B, P, dim) -> (B, P * 200)``: one linear map per patch (``arch="linear"`` and ``"skip"``)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, PATCH)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.lin(z).reshape(z.shape[0], -1)


class FSQ(nn.Module):
    """Finite scalar quantisation (Mentzer et al., ICLR 2024). Each of ``groups * group`` dimensions is
    bounded by ``tanh`` and rounded to one of ``levels`` (odd) integers ``-h .. h``, ``h = (levels - 1) / 2``,
    returned as values in [-1, 1]; the gradient passes straight through the rounding. Every level owns a cell
    of the same width in the bounded domain. ``group`` consecutive dimensions make one code of
    ``levels ** group`` values (mixed radix). No codebook: nothing to initialise, no commitment loss, no dead
    codes. A prefix of ``k`` groups leaves the other dimensions at 0, the middle level."""

    def __init__(self, groups: int, group: int = 4, levels: int = 9):
        super().__init__()
        assert levels % 2 == 1 and levels >= 3, "FSQ: odd levels (0 is the middle level)"
        self.groups, self.group, self.levels = groups, group, levels
        self.dims, self.half = groups * group, (levels - 1) // 2
        self.register_buffer("radix", levels ** torch.arange(group), persistent=False)

    @property
    def n_codes(self) -> int:
        return self.levels ** self.group

    def bound(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u.float()) * (self.half + 0.5 - 1e-3)

    def codes_of(self, r: torch.Tensor) -> torch.Tensor:
        """Integer levels ``(..., dims)`` -> codes ``(..., groups)``."""
        d = (r + self.half).long().view(*r.shape[:-1], self.groups, self.group)
        return (d * self.radix).sum(-1)

    def forward(self, u: torch.Tensor, k: Optional[int] = None):
        """``(..., dims)`` -> values ``(..., dims)`` in [-1, 1] and codes ``(..., k)`` of the first ``k`` groups."""
        k = self.groups if k is None else k
        b = self.bound(u)
        r = torch.round(b)
        q = b + (r - b).detach()
        if k < self.groups:
            q = q * (torch.arange(self.dims, device=q.device) < k * self.group)
        return q / self.half, self.codes_of(r.detach())[..., :k]

    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes ``(..., k)`` of the first ``k`` groups -> values ``(..., dims)`` (the other groups at 0)."""
        k = codes.shape[-1]
        d = torch.div(codes.long()[..., None], self.radix, rounding_mode="floor") % self.levels
        v = (d - self.half).float().flatten(-2) / self.half
        return F.pad(v, (0, (self.groups - k) * self.group))


class PatchContext(nn.Module):
    """Residual convolutions along the sequence of patch latents ``(B, P, dim) -> (B, P, dim)``: ``radius / 2``
    blocks of two 3-tap convolutions, so the output at patch ``i`` sees patches ``i - radius .. i + radius``
    (zeros beyond the ends). The second convolution of every block starts at zero: identity at
    initialisation. No normalisation across patches, which would carry information along the whole
    sequence. ``radius = 0``: identity without parameters (state dicts of earlier runs load unchanged)."""

    def __init__(self, dim: int, radius: int = 0):
        super().__init__()
        assert radius >= 0 and radius % 2 == 0, "PatchContext: even radius"
        self.radius = radius
        self.blocks = nn.ModuleList()
        for _ in range(radius // 2):
            c1, c2 = nn.Conv1d(dim, dim, 3, padding=1), nn.Conv1d(dim, dim, 3, padding=1)
            nn.init.zeros_(c2.weight)
            nn.init.zeros_(c2.bias)
            self.blocks.append(nn.Sequential(nn.GELU(), c1, nn.GELU(), c2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not len(self.blocks):
            return z
        h = z.transpose(1, 2)
        for b in self.blocks:
            h = h + b(h)
        return h.transpose(1, 2)


class PatchCodec(nn.Module):
    """Encoder on the 2 s patch (plus ``ctx_enc`` patches), RVQ or FSQ, decoder with ±``receptive_patches``
    patches of receptive field (module docstring)."""

    def __init__(self, width: int = 64, dim: int = 128, codes: int = 1024, levels: int = 8,
                 q_dropout: float = 0.5, order: int = 32, stft_weight: float = 0.1, restart_after: int = 30,
                 crop_sec: float = 16.0, upsample: str = "convT", fs: int = FS, quantizer: str = "rvq",
                 fsq_levels: int = 9, fsq_group: int = 4, whiten: str = "patch", ctx_enc: int = 0,
                 ctx_dec: int = 0, dec_out: str = "raw", arch: str = "conv", fsq_norm: bool = False,
                 fsq_scale: float = 1.0):
        super().__init__()
        assert fs == FS, "PatchCodec: 100 Hz only"
        assert quantizer in ("rvq", "fsq") and whiten in ("patch", "signal")
        assert dec_out in ("raw", "white") and arch in ("conv", "skip", "linear")
        assert arch == "conv" or dec_out == "white", "the linear paths draw the whitened patch"
        assert fsq_scale > 0, "fsq_scale > 0"
        self.fs, self.patch = fs, PATCH
        self.quantizer, self.whiten, self.dec_out, self.arch = quantizer, whiten, dec_out, arch
        self.dim, self.max_levels, self.vq_levels = dim, levels, levels
        self.q_dropout, self.stft_weight = q_dropout, stft_weight
        self.crop = int(round(crop_sec * fs)) // PATCH * PATCH
        self.whitener = Whitener(order, fs)
        self.encoder = nn.Linear(PATCH, dim) if arch == "linear" else PatchEncoder(width, dim, fs)
        self.enc_skip = nn.Linear(PATCH, dim) if arch == "skip" else None
        self.enc_ctx = PatchContext(dim, ctx_enc)
        self.fsq_bn, self.fsq_scale = None, float(fsq_scale)
        if quantizer == "rvq":
            self.rvq = ResidualVQ(levels, codes, dim, restart_after=restart_after)
            self.n_codes = codes
        else:
            self.fsq = FSQ(levels, fsq_group, fsq_levels)
            self.fsq_in, self.fsq_out = nn.Linear(dim, self.fsq.dims), nn.Linear(self.fsq.dims, dim)
            if fsq_norm:
                self.fsq_bn = nn.BatchNorm1d(self.fsq.dims, affine=False)
            self.n_codes = self.fsq.n_codes
        self.dec_ctx = PatchContext(dim, ctx_dec)
        self.upsample = upsample
        self.decoder = LinearPatchDecoder(dim) if arch == "linear" else PatchDecoder(width, dim, upsample)
        self.dec_skip = LinearPatchDecoder(dim) if arch == "skip" else None
        self.bypass_vq = False
        # the codes of patch i see the signal of patches i - encoder_reach .. i + ctx_enc; the output inside
        # patch i depends on the codes of patches i - receptive_patches .. i + receptive_patches (the inverse
        # whitener of dec_out="white" reaches INV_TAPS - 1 samples back, i.e. 2 more patches)
        self.encoder_reach = ctx_enc + (1 if whiten == "signal" else 0)
        self.receptive_patches = ctx_dec + 1 + (math.ceil((INV_TAPS - 1) / PATCH) if dec_out == "white" else 0)

    def _k(self) -> int:
        """Levels used now: a random prefix in training (quantizer dropout), ``vq_levels`` in eval."""
        if self.training and torch.rand(()) < self.q_dropout:
            return int(torch.randint(1, self.max_levels + 1, ()))
        return self.max_levels if self.training else self.vq_levels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, n) -> (B, P, dim)`` continuous latents, ``P = n // 200``."""
        B, n = x.shape
        P = n // PATCH
        x = x[:, : P * PATCH]
        if self.whiten == "signal":  # true history: the first 32 samples of a patch see the previous patch
            w = self.whitener(x).reshape(B * P, PATCH)
        else:  # history reflected inside every patch
            w = self.whitener(x.reshape(B * P, PATCH))
        z = self.encoder(w)
        if self.enc_skip is not None:
            z = z + self.enc_skip(w)
        return self.enc_ctx(z.view(B, P, -1))

    def _fsq_u(self, z: torch.Tensor) -> torch.Tensor:
        """Input of the FSQ bound: projection, batch norm (``fsq_norm``), scale (``fsq_scale``)."""
        u = self.fsq_in(z)
        if self.fsq_bn is not None:
            u = self.fsq_bn(u.reshape(-1, u.shape[-1])).view(u.shape)
        return u if self.fsq_scale == 1.0 else u * self.fsq_scale

    def quantize(self, z: torch.Tensor, k: Optional[int] = None):
        """``(B, P, dim)`` -> quantized latents, codes ``(B, P, k)`` and commitment loss (0 for FSQ)."""
        B, P, D = z.shape
        if self.quantizer == "fsq":
            u = self._fsq_u(z)
            if self.bypass_vq:  # bounded, not rounded
                return self.fsq_out(self.fsq.bound(u) / self.fsq.half), None, z.new_zeros(())
            q, codes = self.fsq(u, self._k() if k is None else k)
            return self.fsq_out(q), codes, z.new_zeros(())
        if self.bypass_vq:
            return z.float(), None, z.new_zeros(())
        zq, codes, loss = self.rvq(z.reshape(B * P, D), self._k() if k is None else k)
        return zq.view(B, P, D), codes.view(B, P, -1), loss

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """``(B, P, k)`` codes (any prefix of levels or groups) -> ``(B, P, dim)``."""
        B, P, k = codes.shape
        if self.quantizer == "fsq":
            return self.fsq_out(self.fsq.lookup(codes))
        return self.rvq.lookup(codes.reshape(B * P, k)).view(B, P, -1)

    def _draw(self, zq: torch.Tensor) -> torch.Tensor:
        """Decoder output in its own domain: the signal, or the whitened signal (``dec_out="white"``)."""
        h = self.dec_ctx(zq)
        y = self.decoder(h)
        return y if self.dec_skip is None else y + self.dec_skip(h)

    def decode(self, zq: torch.Tensor) -> torch.Tensor:
        """``(B, P, dim) -> (B, P * 200)`` signal in the units of the input."""
        y = self._draw(zq)
        return self.whitener.inverse(y) if self.dec_out == "white" else y

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """``(B, P, k)`` -> ``(B, P * 200)`` waveform."""
        return self.decode(self.embed(codes))

    @torch.no_grad()
    def codes(self, x: torch.Tensor, levels: Optional[int] = None) -> torch.Tensor:
        """``(B, n) -> (B, P, levels)`` code indices (default ``vq_levels``)."""
        was = self.training
        self.eval()
        c = self.quantize(self.encode(x), levels or self.vq_levels)[1]
        self.train(was)
        return c

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        zq, codes, commit = self.quantize(z)
        if self.dec_out == "white":
            y = self._draw(zq)
            return {"wave": self.whitener.inverse(y), "white": y, "codes": codes, "commit": commit}
        return {"wave": self.decode(zq), "codes": codes, "commit": commit}

    def loss(self, out: Dict[str, torch.Tensor], x: torch.Tensor):
        """Whitened MSE + ``stft_weight`` x whitened multi-resolution STFT (log magnitude + spectral
        convergence) + RVQ commitment. Returns ``(loss, whitened R^2, extra)``."""
        w = out["wave"].float()
        x = x[:, : w.shape[-1]].float()
        xw = self.whitener(x)  # the same filtering as the encoder input of whiten="signal"
        ww = out["white"].float() if "white" in out else self.whitener(w)
        mse = (ww - xw).pow(2).mean(1)
        mag, _ = stft_losses(ww, xw, windows=STFT_WINDOWS, phase_win=0)
        loss = (mse + self.stft_weight * mag).mean() + out["commit"]
        extra = {"mse_w": mse.mean().item(), "stft_w": mag.mean().item(), "commit": float(out["commit"]),
                 "r2_raw": _r2(w, x).item()}
        if out["codes"] is not None:
            extra["perplexity_1"], extra["used_1"] = code_stats(out["codes"][..., 0], self.n_codes)
        return loss, _r2(ww, xw), extra

    def bits_per_patch(self, levels: Optional[int] = None) -> int:
        per = self.fsq.group * math.log2(self.fsq.levels) if self.quantizer == "fsq" else math.log2(self.n_codes)
        return int(round((levels or self.vq_levels) * per))


class TransformCodec(nn.Module):
    """Gain-shape transform codec of the whitened 2 s patch, without training: the reference of revision 4
    (28.11.2026, xmai-lab/sleep-lens#24). Whitening, patch grid, inverse whitener and code interface are those of
    :class:`PatchCodec` with ``whiten="signal"`` and ``dec_out="white"``, so ``patch_codec eval`` and ``dump`` run
    unchanged.

    * gain: ``log2`` of the RMS of the whitened patch, rounded to ``gain_step`` (code ``0 .. gain_levels - 1``
      from ``gain_min``);
    * shape: the whitened patch over the rounded gain, in the KLT basis ``U`` of the pool (eigenvalues ``lam``
      of the second moment of the shapes, mean 1); coefficient ``j`` in units of ``sqrt(lam_j)``, rounded to a
      step ``delta`` and clipped to ``±half * delta`` (``half = ceil(clip / delta)``): ``2 half + 1`` levels per
      coefficient;
    * decoder: ``w_hat = 2^g U (sqrt(lam) c)``, then the inverse whitener.

    Every coefficient keeps the same relative rounding error, ``delta^2 / 12`` of its variance (up to the
    clipping), so the error of a patch has the second moment of its shape scaled by ``delta^2 / 12`` and the
    whitened R^2 is about ``1 - delta^2 / 12``, at every amplitude. The codes of a patch depend on that patch and
    on the 32-sample history of the whitening; the error of a patch does not depend on its neighbours. Codes per
    patch: the gain, then the ``PATCH`` coefficients (``max_levels = PATCH + 1``; no prefixes).

    ``groups = G > 1`` (revision 4b, 28.11.2026): the clipping of revision 4 hit the KLT directions of smallest
    variance (``lam`` down to 1e-3, 30-50 Hz) in 2.8 % of the patches of DCSM and capped the whitened R^2 near
    0.975. Here the ``PATCH`` directions, in the order of ``lam``, form ``G`` blocks of ``PATCH / G``; each block
    has its own gain (``log2`` RMS of its KLT coefficients, same step and levels) and its coefficients are divided
    by its rounded gain, without ``sqrt(lam)``. The sum of squares of a block is then ``PATCH / G`` times the ratio
    of the true to the rounded gain, so ``|c| <= sqrt(PATCH / G) 2^(gain_step / 2)``: with
    ``half = ceil(that / delta)`` nothing is ever clipped (``clip`` is unused), and every block of every patch
    keeps a rounding error of ``delta^2 / 12`` of its energy. Codes per patch: the ``G`` gains, then the
    coefficients (``max_levels = PATCH + G``)."""

    def __init__(self, delta: float = 0.35, clip: float = 8.0, gain_step: float = 0.05, gain_min: float = -10.0,
                 gain_levels: int = 320, order: int = 32, fs: int = FS, groups: int = 1):
        super().__init__()
        assert fs == FS and delta > 0 and clip > 0 and gain_levels > 1 and groups >= 1 and PATCH % groups == 0
        self.fs, self.patch = fs, PATCH
        self.quantizer, self.whiten, self.dec_out, self.arch = "transform", "signal", "white", "transform"
        self.groups, self.ng = int(groups), PATCH // int(groups)
        self.delta, self.clip = float(delta), float(clip)
        bound = clip if groups == 1 else math.sqrt(self.ng) * 2.0 ** (gain_step / 2)
        self.half = int(math.ceil(bound / delta - 1e-9))
        self.gain_step, self.gain_min, self.gain_levels = float(gain_step), float(gain_min), int(gain_levels)
        self.max_levels = self.vq_levels = PATCH + self.groups
        self.q_dropout = 0.0
        self.n_codes = max(self.gain_levels, 2 * self.half + 1)
        self.whitener = Whitener(order, fs)
        self.register_buffer("U", torch.eye(PATCH))
        self.register_buffer("lam", torch.ones(PATCH))
        self.bypass_vq = False
        self.encoder_reach = 1  # the whitening history reaches the last order samples of the previous patch
        self.receptive_patches = math.ceil((INV_TAPS - 1) / PATCH)  # the inverse FIR reaches 255 samples forward

    @torch.no_grad()
    def fit_basis(self, shapes: torch.Tensor) -> dict:
        """``(M, PATCH)`` whitened patches over their RMS -> ``U`` and ``lam`` (descending, mean 1)."""
        s = shapes.double()
        ev, U = torch.linalg.eigh(s.T @ s / len(s))
        ev, U = ev.flip(0).clamp_min(1e-12), U.flip(1)
        self.U.copy_(U.float())
        self.lam.copy_((ev / ev.mean()).float())
        frac = (ev / ev.sum()).cumsum(0)
        return {"patches": len(s), "lam_max": float(self.lam[0]), "lam_min": float(self.lam[-1]),
                "variance_top": {str(k): float(frac[k - 1]) for k in (64, 128, 160, 180)}}

    def _scale(self) -> torch.Tensor:
        """Unit of each coefficient after the gain: ``sqrt(lam)`` with one gain, 1 with block gains."""
        return self.lam.sqrt() if self.groups == 1 else torch.ones_like(self.lam)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, n) -> (B, P, G + PATCH)``: log2 RMS of the KLT coefficients of each block (with one block, of the
        whitened patch), then the coefficients over the RMS of their block, in units of :meth:`_scale`."""
        B, n = x.shape
        P, G = n // PATCH, self.groups
        w = self.whitener(x[:, : P * PATCH]).reshape(B, P, PATCH)
        p = (w @ self.U).reshape(B, P, G, self.ng)
        rms = p.pow(2).mean(-1).sqrt().clamp_min(2.0 ** self.gain_min)
        c = (p / rms[..., None]).reshape(B, P, PATCH) / self._scale()
        return torch.cat([rms.log2(), c], -1)

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes ``(B, P, G + PATCH)`` -> dequantised values."""
        c, G = codes.float(), self.groups
        return torch.cat([self.gain_min + c[..., :G] * self.gain_step, (c[..., G:] - self.half) * self.delta], -1)

    def quantize(self, z: torch.Tensor, k: Optional[int] = None):
        """``(B, P, G + PATCH)`` -> dequantised values, codes and 0 (``k`` is ignored: no prefixes). Each block is
        rescaled to its rounded gain before it is rounded."""
        if self.bypass_vq:
            return z, None, z.new_zeros(())
        B, P, _ = z.shape
        G = self.groups
        gi = torch.round((z[..., :G] - self.gain_min) / self.gain_step).clamp(0, self.gain_levels - 1)
        r = torch.exp2(z[..., :G] - (self.gain_min + gi * self.gain_step))
        c = (z[..., G:].reshape(B, P, G, self.ng) * r[..., None]).reshape(B, P, PATCH)
        ci = torch.round(c / self.delta).clamp(-self.half, self.half) + self.half
        codes = torch.cat([gi, ci], -1).long()
        return self.embed(codes), codes, z.new_zeros(())

    def _draw(self, zq: torch.Tensor) -> torch.Tensor:
        """Whitened signal ``(B, P * PATCH)`` from dequantised values."""
        B, P, _ = zq.shape
        G = self.groups
        p = (zq[..., G:] * self._scale()).reshape(B, P, G, self.ng) * torch.exp2(zq[..., :G])[..., None]
        return (p.reshape(B, P, PATCH) @ self.U.T).reshape(B, P * PATCH)

    def decode(self, zq: torch.Tensor) -> torch.Tensor:
        return self.whitener.inverse(self._draw(zq))

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decode(self.embed(codes))

    @torch.no_grad()
    def codes(self, x: torch.Tensor, levels: Optional[int] = None) -> torch.Tensor:
        return self.quantize(self.encode(x))[1]

    def bits_per_patch(self, levels: Optional[int] = None) -> int:
        return int(round(self.groups * math.log2(self.gain_levels) + PATCH * math.log2(2 * self.half + 1)))


def crops(x: torch.Tensor, n: int, training: bool) -> torch.Tensor:
    """``(B, L) -> (B', n)``: one crop per window, at a random offset in training (patch grid at a
    random alignment with the epochs) and centred in eval; crops with a zero run are dropped."""
    B, L = x.shape
    if L <= n:
        c = x[:, : L // PATCH * PATCH]
    else:
        o = (torch.randint(0, L - n + 1, (B,), device=x.device) if training
             else torch.full((B,), (L - n) // 2, device=x.device))
        c = x.gather(1, o[:, None] + torch.arange(n, device=x.device))
    keep = ~has_zero_run(c)
    return c[keep] if keep.any() else c


class PatchCodecTrainer(Trainer):
    """Trainer for :class:`PatchCodec`; ``acc`` is the whitened R^2 of the crop, and the validation loss is
    ``1 - acc``, so the best checkpoint and early stopping follow the reconstruction. Call
    :func:`fit_whitener` first; use ``log_confusion_matrix=False``."""

    @staticmethod
    def _step(model, batch, loss_fn, device):
        from physioex.data.collate import stack_channels
        from physioex.models.msvq import _autocast

        x = stack_channels(batch)[:, :, 0].float()
        x = crops(x.reshape(len(x), -1).to(device), model.crop, model.training)
        with _autocast(device):
            out = model(x)
        loss, r2, extra = model.loss(out, x)
        if not model.training:
            # validation: the Trainer keeps the checkpoint and stops early on the lowest loss. The commitment
            # term can grow while R^2 still rises (job 1669985), so select on the whitened R^2 at full depth.
            extra["loss_train"] = float(loss)
            loss = 1 - r2
        return loss, r2, extra


def warmup_continuous(model: PatchCodec, loader, opt, steps: int, device, grad_clip: Optional[float] = 1.0,
                      log_every: int = 500) -> list:
    """Train ``steps`` steps with the RVQ bypassed (continuous latent), before
    :func:`kmeans_init_rvq`. The codebooks are then initialised on the latents of a trained encoder
    rather than a random one. On fresh crops (T4, width 16, 8 levels) the RVQ trained from step 0
    reached whitened R^2 0.14 after 400 steps; the continuous model reached 0.47. Returns
    ``[[step, mean whitened R^2 of the last log_every steps], ...]``."""
    model.train()
    model.bypass_vq = True
    log, acc = [], []

    def endless():
        while True:
            yield from loader

    gen = endless()
    try:
        for s in range(steps):
            opt.zero_grad(set_to_none=True)
            loss, r2, _ = PatchCodecTrainer._step(model, next(gen), None, device)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            acc.append(float(r2))
            if (s + 1) % log_every == 0 or s + 1 == steps:
                log.append([s + 1, float(np.mean(acc))])
                acc = []
                print(f"[patch_codec] warm-up step {s + 1}/{steps}: whitened R^2 {log[-1][1]:.3f}", flush=True)
    finally:
        model.bypass_vq = False
    return log


@torch.no_grad()
def kmeans_init_rvq(model: PatchCodec, loader, device, per_code: int = 20, max_batches: int = 500) -> int:
    """k-means initialisation of every RVQ level (:meth:`ResidualVQ.kmeans_init`, level by level on the
    residuals) on the continuous latents of training crops, ``per_code`` vectors per code. Returns the
    number of vectors."""
    from physioex.data.collate import stack_channels

    if model.quantizer != "rvq":
        return 0
    was = model.training
    model.eval()
    need, acc, n = per_code * model.n_codes, [], 0
    for i, batch in enumerate(loader):
        x = stack_channels(batch)[:, :, 0].float()
        z = model.encode(crops(x.reshape(len(x), -1).to(device), model.crop, True)).reshape(-1, model.dim)
        acc.append(z.float().cpu())
        n += len(z)
        if n >= need or i + 1 >= max_batches:
            break
    z = torch.cat(acc)[:need]
    model.rvq.kmeans_init(z.to(device))
    model.train(was)
    return len(z)


def build_from_args(a: dict):
    """Untrained :class:`PatchCodec` from the ``args`` of a ``run.json`` (``model: patch_codec``), or a
    :class:`TransformCodec` (``arch="transform"``, revision 4) with its basis still at the identity."""
    if a.get("arch") == "transform":
        return TransformCodec(delta=a["tc_delta"], clip=a["tc_clip"], gain_step=a["tc_gain_step"],
                              gain_min=a["tc_gain_min"], gain_levels=a["tc_gain_levels"], order=a["order"],
                              groups=a.get("tc_groups", 1))
    return PatchCodec(width=a["width"], dim=a["dim"], codes=a["codes"], levels=a["levels"], q_dropout=a["q_dropout"],
                      order=a["order"], stft_weight=a["stft_weight"], restart_after=a["restart_after"],
                      crop_sec=a["crop_sec"], upsample=a.get("upsample", "convT"),  # no key: runs before 27.11.2026
                      quantizer=a.get("quantizer", "rvq"), fsq_levels=a.get("fsq_levels", 9),
                      fsq_group=a.get("fsq_group", 4), whiten=a.get("whiten", "patch"), ctx_enc=a.get("ctx_enc", 0),
                      ctx_dec=a.get("ctx_dec", 0),  # no keys: runs before revision 2 (27.11.2026)
                      dec_out=a.get("dec_out", "raw"), arch=a.get("arch", "conv"),
                      fsq_norm=a.get("fsq_norm", False),  # no keys: runs before revision 3
                      fsq_scale=a.get("fsq_scale", 1.0))  # no key: runs before 28.11.2026


# ---------------------------------------------------------------------------- spectral twin (S3, #20)
TWIN_BANDS = {"so": (0.5, 2), "delta": (2, 4), "theta": (4, 7), "alpha": (8, 13), "sigma": (11, 16), "beta": (16, 30)}


def patch_spectra(x: torch.Tensor, patch: int = PATCH, fs: int = FS) -> torch.Tensor:
    """``(n,)`` signal -> ``(n // patch, 37)`` spectral features of every patch, for the spectral twin of the
    codes (plan S3): log power of the 6 bands of the earlier twin (``SPEC_WIN`` of
    :mod:`physioex.train.bin.concept_rationale`) and of 1 Hz bins from 0.5 to 30.5 Hz (``spec_fine``), both
    on the patch itself under a Hann window, and the log line length (mean absolute difference)."""
    P = len(x) // patch
    w = x[: P * patch].reshape(P, patch).float()
    p = torch.fft.rfft((w - w.mean(1, keepdim=True)) * torch.hann_window(patch, periodic=False, device=x.device),
                       dim=1).abs() ** 2
    f = torch.fft.rfftfreq(patch, 1 / fs).to(x.device)
    cols = [p[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in TWIN_BANDS.values()]
    cols += [p[:, (f >= lo) & (f < lo + 1)].sum(1) for lo in np.arange(0.5, 30.5, 1.0)]
    ll = w.diff(dim=1).abs().mean(1)
    return torch.log(torch.cat([torch.stack(cols, 1), ll[:, None]], 1) + 1e-6)


def _nearest(r: torch.Tensor, C: torch.Tensor, chunk: int = 1 << 18) -> torch.Tensor:
    c2 = (C * C).sum(1)
    return torch.cat([(c2[None] - 2 * r[i:i + chunk] @ C.T).argmin(1) for i in range(0, len(r), chunk)])


def _kmeans_torch(r: torch.Tensor, k: int, iters: int, g: torch.Generator) -> torch.Tensor:
    """Lloyd iterations from ``k`` random points; an empty cluster takes a random point again."""
    C = r[torch.randperm(len(r), generator=g, device=r.device)[:k]].clone()
    for _ in range(iters):
        idx = _nearest(r, C)
        cnt = torch.bincount(idx, minlength=k).float()
        s = torch.zeros_like(C).index_add_(0, idx, r)
        empty = cnt == 0
        C = torch.where(empty[:, None], C, s / cnt.clamp_min(1)[:, None])
        if empty.any():
            C[empty] = r[torch.randint(len(r), (int(empty.sum()),), generator=g, device=r.device)]
    return C


class SpectralTwin:
    """Residual k-means of the z-scored :func:`patch_spectra`: ``levels`` codebooks of ``n_codes``, the K
    and L of the codec, so the twin's codes have the same shape and vocabulary size (plan S3)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, codebooks: torch.Tensor):
        self.mean, self.std, self.codebooks = mean, std, codebooks  # (D,), (D,), (L, K, D)

    @classmethod
    def fit(cls, feats: torch.Tensor, n_codes: int, levels: int, iters: int = 25, seed: int = 0) -> "SpectralTwin":
        mean, std = feats.mean(0), feats.std(0).clamp_min(1e-6)
        r = (feats - mean) / std
        g = torch.Generator(device=feats.device).manual_seed(seed)
        cbs = []
        for _ in range(levels):
            C = _kmeans_torch(r, n_codes, iters, g)
            r = r - C[_nearest(r, C)]
            cbs.append(C)
        return cls(mean, std, torch.stack(cbs))

    def codes(self, feats: torch.Tensor, levels: Optional[int] = None) -> torch.Tensor:
        r = (feats - self.mean) / self.std
        out = []
        for C in self.codebooks[: levels or len(self.codebooks)]:
            idx = _nearest(r, C)
            r = r - C[idx]
            out.append(idx)
        return torch.stack(out, 1)

    def residual_r2(self, feats: torch.Tensor, levels: int) -> float:
        """R^2 of the z-scored features reconstructed from the first ``levels`` codebooks."""
        r = (feats - self.mean) / self.std
        q = torch.zeros_like(r)
        for C, idx in zip(self.codebooks[:levels], self.codes(feats, levels).T):
            q = q + C[idx]
        return float(1 - ((r - q) ** 2).sum() / ((r - r.mean(0)) ** 2).sum())

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "codebooks": self.codebooks.cpu()}

    @classmethod
    def from_state(cls, sd: dict, device=None) -> "SpectralTwin":
        return cls(*(sd[k].to(device) for k in ("mean", "std", "codebooks")))
