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

Input: ``(B, n)`` clean EEG at 100 Hz, ``RobustScale`` units (cache of the clean channels); ``n`` is
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


class PatchCodec(nn.Module):
    """Encoder local to the 2 s patch, RVQ, decoder with ±1 patch receptive field (module docstring)."""

    def __init__(self, width: int = 64, dim: int = 128, codes: int = 1024, levels: int = 8,
                 q_dropout: float = 0.5, order: int = 32, stft_weight: float = 0.1, restart_after: int = 30,
                 crop_sec: float = 16.0, upsample: str = "convT", fs: int = FS):
        super().__init__()
        assert fs == FS, "PatchCodec: 100 Hz only"
        self.fs, self.patch = fs, PATCH
        self.dim, self.n_codes, self.max_levels, self.vq_levels = dim, codes, levels, levels
        self.q_dropout, self.stft_weight = q_dropout, stft_weight
        self.crop = int(round(crop_sec * fs)) // PATCH * PATCH
        self.whitener = Whitener(order, fs)
        self.encoder = PatchEncoder(width, dim, fs)
        self.rvq = ResidualVQ(levels, codes, dim, restart_after=restart_after)
        self.upsample = upsample
        self.decoder = PatchDecoder(width, dim, upsample)
        self.bypass_vq = False

    def _k(self) -> int:
        """Levels used now: a random prefix in training (quantizer dropout), ``vq_levels`` in eval."""
        if self.training and torch.rand(()) < self.q_dropout:
            return int(torch.randint(1, self.max_levels + 1, ()))
        return self.max_levels if self.training else self.vq_levels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, n) -> (B, P, dim)`` continuous latents, ``P = n // 200``."""
        B, n = x.shape
        P = n // PATCH
        p = x[:, : P * PATCH].reshape(B * P, PATCH)
        return self.encoder(self.whitener(p)).view(B, P, -1)

    def quantize(self, z: torch.Tensor, k: Optional[int] = None):
        """``(B, P, dim)`` -> quantized latents, codes ``(B, P, k)`` and commitment loss."""
        B, P, D = z.shape
        if self.bypass_vq:
            return z.float(), None, z.new_zeros(())
        zq, codes, loss = self.rvq(z.reshape(B * P, D), self._k() if k is None else k)
        return zq.view(B, P, D), codes.view(B, P, -1), loss

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """``(B, P, k)`` codes (any prefix of levels) -> ``(B, P, dim)``."""
        B, P, k = codes.shape
        return self.rvq.lookup(codes.reshape(B * P, k)).view(B, P, -1)

    def decode(self, zq: torch.Tensor) -> torch.Tensor:
        return self.decoder(zq)

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
        return {"wave": self.decode(zq), "codes": codes, "commit": commit}

    def loss(self, out: Dict[str, torch.Tensor], x: torch.Tensor):
        """Whitened MSE + ``stft_weight`` x whitened multi-resolution STFT (log magnitude + spectral
        convergence) + RVQ commitment. Returns ``(loss, whitened R^2, extra)``."""
        w = out["wave"].float()
        x = x[:, : w.shape[-1]].float()
        ww, xw = self.whitener(w), self.whitener(x)
        mse = (ww - xw).pow(2).mean(1)
        mag, _ = stft_losses(ww, xw, windows=STFT_WINDOWS, phase_win=0)
        loss = (mse + self.stft_weight * mag).mean() + out["commit"]
        extra = {"mse_w": mse.mean().item(), "stft_w": mag.mean().item(), "commit": float(out["commit"]),
                 "r2_raw": _r2(w, x).item()}
        if out["codes"] is not None:
            extra["perplexity_1"], extra["used_1"] = code_stats(out["codes"][..., 0], self.n_codes)
        return loss, _r2(ww, xw), extra

    def bits_per_patch(self, levels: Optional[int] = None) -> int:
        return int(round((levels or self.vq_levels) * math.log2(self.n_codes)))


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


def build_from_args(a: dict) -> PatchCodec:
    """Untrained :class:`PatchCodec` from the ``args`` of a ``run.json`` (``model: patch_codec``)."""
    return PatchCodec(width=a["width"], dim=a["dim"], codes=a["codes"], levels=a["levels"], q_dropout=a["q_dropout"],
                      order=a["order"], stft_weight=a["stft_weight"], restart_after=a["restart_after"],
                      crop_sec=a["crop_sec"], upsample=a.get("upsample", "convT"))  # no key: runs before 27.11.2026


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
