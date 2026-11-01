"""MSAE: hierarchical continuous autoencoder of single-channel EEG (N + E + P).

The continuous model that comes before any quantization (:mod:`physioex.models.msvq`
is the discrete stage that will sit on top of its latents). For the centre epoch
``t`` of a window it returns three latents:

* **N** ``(128,)``: what the 40 neighbouring epochs predict about ``t`` (``t`` masked).
  It is context, not payload: it is computed from the neighbours' encoder features;
* **E** ``(128,)``: what N does not predict about epoch ``t`` (background, spectrum);
  the E encoder is conditioned on N (FiLM);
* **P** ``(T_p, d_p)``: local tokens (1 s or 0.5 s) carrying what N and E do not
  explain (spindles, K-complexes, transients, phase); conditioned on N and E.

The chain N -> E -> P is conditional coding (as in learned video codecs): each
encoder sees the coarser latents and only has to add what they miss. One decoder
is called on the three prefixes (missing latents set to zero): ``x_N = D(N)``,
``x_NE = D(N, E)``, ``x_NEP = D(N, E, P)``; each prefix has its own loss, so the
gain of every scale can be measured.

Compression is enforced, not assumed: E and P pass through ``tanh`` and a
``latent_bits`` channel (uniform noise of one quantization step in training,
rounding in eval, Balle-style), so epoch ``t`` costs ``(128 + T_p * d_p) * latent_bits``
bits given its context, against 3000 * 16 raw bits.

Encoder and decoder follow the EnCodec / SoundStream SEANet design (residual
dilated convs, strided downsampling, mirrored transposed convs) with a
NeuroRVQ-style multi-scale stem. The decoder draws the whole 30 s epoch at once
(no per-second seams). Losses: ``x_NEP`` the full waveform, MSE in time (it carries
the phase) + ``stft_weight`` x multi-resolution STFT 0.25-4 s (log magnitude +
spectral convergence); ``x_N`` and ``x_NE`` only ``stft_weight`` x the spectral
envelope (STFT 2-4 s and 1-4 s: their phase is not predictable); epoch log-PSD
heads from N and N+E. With the STFT terms at full weight, or a log-magnitude phase
term, the shared decoder learns noise with the right spectrum and wave R^2 stays
<= 0 (overfit test of one batch, 29.10.2026); MSE dominant reaches 0.83-0.88.
Sleep stages are never used.

SEANet blocks re-implemented after EnCodec (Defossez et al. 2022, MIT licence,
https://github.com/facebookresearch/encodec).

Input: ``(B, L, 1, 3000)`` clean EEG at 100 Hz, L odd (centre = epoch ``t``).

Other modalities (01.11.2026): ``fs`` (100 or 200 Hz; at 200 Hz one more stride-2
stage keeps the same tokens), ``pow_bands`` (bands of the band-power terms and of the
texture), ``psd_band`` (bins of the epoch log-PSD heads) and ``recon``:
``"wave"`` (default, as above) or ``"texture"`` (EMG: amplitude-modulated noise, the
waveform has no meaning; every prefix is decoded as noise texture only and trained on
the log band power per ``seg_sec`` segment). ``use_n=False`` drops N (ablation: the
FiLM layers get zeros, the decoder only the bias of N). The defaults reproduce the
EEG model exactly.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from physioex.models.msvq import EPOCH, N_PATCH, PATCH, _autocast, _mlp, _r2, _transformer, epoch_logpsd
from physioex.train.trainer import Trainer

STRIDES = {1.0: (2, 5, 10), 0.5: (2, 5, 5)}
STFT_WINDOWS = (25, 50, 100, 200, 400)
# bands (Hz, 1 s rfft bins) of the "band" loss: log10 power per 1 s segment
POW_BANDS = ((1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 31), (31, 51))
LOW_HZ = 4  # "band" loss: phase-exact MSE only up to here
TF_WIN, TF_HOP = 100, 50  # spectrogram of the "istft" decoder: 1 s Hann, 0.5 s hop (COLA), 51 bins x 61 frames
TF_BINS, TF_FRAMES = TF_WIN // 2 + 1, EPOCH // TF_HOP + 1


def stem_kernels(fs: int) -> tuple:
    """Kernels of :class:`MultiScaleStem` (0.5 s ... 50 ms) at ``fs``, odd."""
    return tuple(int(round(k * fs / 100)) | 1 for k in (51, 25, 11, 5))


def psd_bins(fs: int, band=(0.5, 35.0), seg_sec: float = 4.0) -> int:
    """Number of Welch bins of :func:`physioex.models.msvq.epoch_logpsd` in ``band``."""
    seg = int(seg_sec * fs)
    f = torch.fft.rfftfreq(seg, 1 / fs)
    return int(((f >= band[0]) & (f <= band[1])).sum())


def tf_stft(x: torch.Tensor) -> torch.Tensor:
    """``(B, 3000) -> (B, TF_FRAMES, TF_BINS)`` complex spectrogram of the "istft" decoder."""
    w = torch.hann_window(TF_WIN, device=x.device, dtype=x.dtype)
    return torch.stft(x, TF_WIN, TF_HOP, window=w, return_complex=True).transpose(1, 2)


def tf_istft(s: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`tf_stft`: ``(B, TF_FRAMES, TF_BINS)`` complex -> ``(B, 3000)``."""
    w = torch.hann_window(TF_WIN, device=s.device)
    return torch.istft(s.transpose(1, 2), TF_WIN, TF_HOP, window=w, length=EPOCH)


def tf_logmag(s: torch.Tensor) -> torch.Tensor:
    return torch.log(s.abs() + 1e-5)


class _ResUnit(nn.Module):
    def __init__(self, c: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(nn.ELU(), nn.Conv1d(c, c, 7, dilation=dilation, padding=3 * dilation),
                                   nn.ELU(), nn.Conv1d(c, c, 1))

    def forward(self, x):
        return x + self.block(x)


def _down(c_in: int, c_out: int, s: int) -> nn.Sequential:
    # kernel 2s, padding ceil(s/2): length / s exactly for the strides used here
    return nn.Sequential(*[_ResUnit(c_in, d) for d in (1, 3, 9)], nn.ELU(),
                         nn.Conv1d(c_in, c_out, 2 * s, stride=s, padding=(s + 1) // 2))


def _up(c_in: int, c_out: int, s: int) -> nn.Sequential:
    p = (s + 1) // 2
    return nn.Sequential(nn.ELU(), nn.ConvTranspose1d(c_in, c_out, 2 * s, stride=s, padding=p, output_padding=2 * p - s),
                         *[_ResUnit(c_out, d) for d in (1, 3, 9)])


class MultiScaleStem(nn.Module):
    """Parallel convs with long and short kernels (0.5 s ... 50 ms at 100 Hz)."""

    def __init__(self, width: int, kernels: Sequence[int] = (51, 25, 11, 5)):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(1, width // len(kernels), k, padding=k // 2) for k in kernels])

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], 1)


def _bottleneck(z: torch.Tensor, bits: int, training: bool) -> torch.Tensor:
    """``tanh`` then a ``bits``-bit channel: uniform noise of one step in training, rounding in eval."""
    z = torch.tanh(z)
    if bits <= 0:
        return z
    step = 2.0 / (2 ** bits - 1)
    if training:
        return z + (torch.rand_like(z) - 0.5) * step
    return torch.round((z + 1) / step) * step - 1


class MSAE(nn.Module):
    """Hierarchical continuous autoencoder N + E + P (see module docstring)."""

    def __init__(self, sequence_length: int = 41, width: int = 32, dim: int = 128, d_p: int = 8,
                 token_sec: float = 1.0, latent_bits: int = 8, n_psd_bins: int = None,
                 stft_weight: float = 0.1, decoder: str = "time", tf_consistency: float = 0.0,
                 loss: str = "mse", band_weight: float = 1.0, low_hz: float = LOW_HZ,
                 low_weight: float = 1.0, pow_hz: float = None, texture: bool = False, fs: int = 100,
                 pow_bands: Sequence[Sequence[float]] = None, psd_band: Sequence[float] = (0.5, 35.0),
                 recon: str = "wave", seg_sec: float = 1.0, use_n: bool = True):
        super().__init__()
        assert fs in (100, 200), fs
        assert recon in ("wave", "texture"), recon
        self.L, self.c, self.dim, self.d_p = sequence_length, sequence_length // 2, dim, d_p
        self.fs, self.epoch = fs, 30 * fs
        self.pow_bands = tuple(tuple(b) for b in (pow_bands or POW_BANDS))
        self.psd_band = tuple(psd_band)
        self.recon, self.seg_sec, self.use_n = recon, seg_sec, use_n
        if n_psd_bins is None or (fs, tuple(psd_band)) != (100, (0.5, 35.0)):
            n_psd_bins = psd_bins(fs, psd_band)
        # at 200 Hz one more stride-2 stage: the same tokens as at 100 Hz
        self.strides = (2,) * (fs // 100 - 1) + STRIDES[token_sec]
        self.n_tok = self.epoch // math.prod(self.strides)
        self.latent_bits = latent_bits
        self.stft_weight = stft_weight
        assert decoder in ("time", "istft"), decoder
        assert decoder == "time" or fs == 100, "the istft decoder is 100 Hz only"
        self.decoder = decoder
        self.tf_consistency = tf_consistency
        assert loss in ("mse", "band"), loss
        self.loss, self.band_weight, self.low_hz, self.low_weight = loss, band_weight, low_hz, low_weight
        self.pow_hz = low_hz if pow_hz is None else pow_hz  # band power on bands starting above pow_hz
        self.texture = texture

        # encoder: multi-scale stem + SEANet downsampling -> (B, dim, T_p)
        ch = [width * 2 ** min(i, 3) for i in range(len(self.strides) + 1)]  # capped at 8x width (EEG: 3 stages)
        self.stem = MultiScaleStem(width, stem_kernels(fs))
        self.down = nn.Sequential(*[_down(ch[i], ch[i + 1], s) for i, s in enumerate(self.strides)],
                                  nn.ELU(), nn.Conv1d(ch[-1], dim, 3, padding=1))
        self.pos_tok = nn.Parameter(torch.zeros(1, self.n_tok, dim))
        self.enc_tok = _transformer(dim, 2)
        # E: tokens FiLM-conditioned on N, attention pooling
        self.e_film = nn.Linear(dim, 2 * dim)
        self.e_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.e_pool = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.e_out = nn.Linear(dim, dim)
        # P: tokens FiLM-conditioned on [N; E], squeezed to d_p
        self.p_film = nn.Linear(2 * dim, 2 * dim)
        self.p_out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, d_p))
        # N: neighbours' mean encoder features -> transformer with t masked
        self.n_in = nn.Linear(dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_epoch = nn.Parameter(torch.zeros(1, sequence_length, dim))
        self.enc_n = _transformer(dim, 2)
        # decoder: token = W_p P_j + W_e E + W_n N + pos_j -> transformer -> mirrored SEANet
        self.dec_p, self.dec_e, self.dec_n = nn.Linear(d_p, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.pos_dec = nn.Parameter(torch.zeros(1, self.n_tok, dim))
        self.dec_tok = _transformer(dim, 2)
        if recon == "texture":
            pass  # no waveform decoder: every prefix is noise texture (see noise_texture)
        elif decoder == "time":
            self.up = nn.Sequential(nn.Conv1d(dim, ch[-1], 3, padding=1),
                                    *[_up(ch[i + 1], ch[i], s) for i, s in reversed(list(enumerate(self.strides)))],
                                    nn.ELU(), nn.Conv1d(ch[0], 1, 7, padding=3))
        else:
            # spectral decoder: one token per STFT frame; z-scored log|S| for every prefix, phase only with P.
            # The magnitude head starts at zero, i.e. at the dataset mean log-spectrum (lm_mean).
            self.pos_frame = nn.Parameter(torch.zeros(1, TF_FRAMES, dim))
            self.mag_head = nn.Linear(dim, TF_BINS)
            nn.init.zeros_(self.mag_head.weight); nn.init.zeros_(self.mag_head.bias)
            self.phase_head = nn.Linear(dim, 2 * TF_BINS)
            self.register_buffer("lm_mean", torch.zeros(TF_BINS))
            self.register_buffer("lm_std", torch.ones(TF_BINS))
            nn.init.normal_(self.pos_frame, std=0.02)
        if recon == "texture":
            self.texture = texture = True
            self.pow_hz = -1.0  # every band is texture
        if texture:
            # noise-excited texture: band-passed white noise x per-token log10 amplitude per band above pow_hz.
            # Starts nearly silent (gain 10^-3); trained only by the band power loss (see MSAETrainer).
            assert decoder == "time" and loss == "mse", "texture needs the time decoder and loss=\"mse\""
            self.tex_bands = [i for i, (a, _) in enumerate(self.pow_bands) if a > self.pow_hz]
            self.tex_head = nn.Linear(dim, len(self.tex_bands))
            nn.init.zeros_(self.tex_head.weight); nn.init.constant_(self.tex_head.bias, -3.0)
        # epoch spectrum from N (prediction) and from N+E
        self.spec_n, self.spec_ne = _mlp(dim, n_psd_bins), _mlp(2 * dim, n_psd_bins)
        self.register_buffer("psd_mean", torch.zeros(n_psd_bins))
        self.register_buffer("psd_std", torch.ones(n_psd_bins))
        # unused by MSAE, kept so physioex.models.msvq.fit_msvq_norms works unchanged
        self.register_buffer("mag_mean", torch.zeros(PATCH // 2 + 1))
        self.register_buffer("mag_std", torch.ones(PATCH // 2 + 1))
        for p in (self.pos_tok, self.e_query, self.mask_token, self.pos_epoch, self.pos_dec):
            nn.init.normal_(p, std=0.02)

    # --- encoder -----------------------------------------------------------
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, epoch) -> (B, T_p, dim)`` token features (stem, SEANet, transformer)."""
        h = self.down(self.stem(x[:, None])).transpose(1, 2)
        return self.enc_tok(h + self.pos_tok)

    def encode_context(self, x: torch.Tensor) -> torch.Tensor:
        """N from the neighbours' mean token features (no gradient into the encoder); t masked."""
        B, L = x.shape[:2]
        if not self.use_n:  # ablation: no context
            return x.new_zeros(B, self.dim)
        idx = [i for i in range(L) if i != self.c]
        with torch.no_grad():
            f = self.features(x[:, idx].reshape(-1, self.epoch)).mean(1)
        z = self.n_in(f).reshape(B, L - 1, -1)
        z = torch.cat([z[:, :self.c], self.mask_token.expand(B, 1, -1), z[:, self.c:]], 1)
        return self.enc_n(z + self.pos_epoch)[:, self.c]

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``(B, L, 1, 3000) -> {"N": (B,dim), "E": (B,dim), "P": (B,T_p,d_p)}``.

        E is coded given N, P given N and E; both go through the bit channel.
        """
        x = x[:, :, 0].float()
        n = self.encode_context(x)
        h = self.features(x[:, self.c])
        g, b = self.e_film(n).chunk(2, -1)
        he = h * (1 + g[:, None]) + b[:, None]
        e = self.e_out(self.e_pool(self.e_query.expand(len(h), -1, -1), he, he, need_weights=False)[0][:, 0])
        e = _bottleneck(e, self.latent_bits, self.training)
        g, b = self.p_film(torch.cat([n, e], -1)).chunk(2, -1)
        p = _bottleneck(self.p_out(h * (1 + g[:, None]) + b[:, None]), self.latent_bits, self.training)
        return {"N": n, "E": e, "P": p}

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 2 + T_p, dim)`` for a downstream reader (e.g. a Q-Former): ``[N, E, P_1..P_T]``.

        P is lifted to ``dim`` with the decoder's own projection plus its time
        position, so no information is added; token order gives the type.
        """
        z = self.encode(x)
        return torch.cat([z["N"][:, None], z["E"][:, None], self.dec_p(z["P"]) + self.pos_dec], 1)

    # --- decoder -----------------------------------------------------------
    def decode(self, n, e=None, p=None, texture: bool = False):
        """One decoder for every prefix: ``D(N)``, ``D(N, E)``, ``D(N, E, P)`` -> ``(B, 3000)``.
        Missing latents count as zero. ``texture=True`` (model built with ``texture``) returns
        ``(deterministic wave, noise texture)``."""
        z = self.dec_n(n)
        if e is not None:
            z = z + self.dec_e(e)
        z = z[:, None] + self.pos_dec
        if p is not None:
            z = z + self.dec_p(p)
        else:
            z = z + self.dec_p.bias
        if e is None:
            z = z + self.dec_e.bias[None, None]
        h = self.dec_tok(z)
        if self.recon == "texture":
            tex = self.noise_texture(h)
            return (torch.zeros_like(tex), tex) if texture else tex
        wave = self.up(h.transpose(1, 2))[:, 0]
        if not texture:
            return wave
        return wave, self.noise_texture(h)

    def noise_texture(self, h: torch.Tensor) -> torch.Tensor:
        """``(B, T_p, dim)`` decoder tokens -> ``(B, 3000)`` sum over the bands above ``pow_hz`` of
        unit-RMS band-passed white noise x the amplitude predicted per token (linear in time).
        The phase is random: this restores the band power that the MSE path mutes."""
        g = 10 ** self.tex_head(h).float().clamp(-6, 2)  # (B, T_p, n_bands)
        g = F.interpolate(g.transpose(1, 2), size=self.epoch, mode="linear", align_corners=True)
        spec = torch.fft.rfft(torch.randn(len(h), self.epoch, device=h.device), dim=-1)
        f = torch.fft.rfftfreq(self.epoch, d=1 / self.fs, device=h.device)
        bands = []
        for i in self.tex_bands:
            a, b = self.pow_bands[i]
            nb = torch.fft.irfft(spec * ((f >= a) & (f < b)), n=self.epoch, dim=-1)
            bands.append(nb / (nb.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8))
        return (g * torch.stack(bands, 1)).sum(1)

    def decode_tf(self, n, e=None, p=None):
        """Spectral decoder (``decoder="istft"``): z-scored log|S| ``(B, TF_FRAMES, TF_BINS)`` for
        every prefix, plus the unit phase vector ``(B, TF_FRAMES, TF_BINS, 2)`` when P is given.
        Missing latents count as zero; P tokens are interpolated onto the frames."""
        z = self.dec_n(n) + (self.dec_e(e) if e is not None else self.dec_e.bias)
        z = z[:, None] + self.pos_frame
        if p is not None:
            pf = F.interpolate(p.transpose(1, 2), size=TF_FRAMES, mode="linear", align_corners=True).transpose(1, 2)
            z = z + self.dec_p(pf)
        else:
            z = z + self.dec_p.bias
        h = self.dec_tok(z)
        mz = self.mag_head(h)
        if p is None:
            return mz, None
        ph = self.phase_head(h).float().reshape(*h.shape[:2], TF_BINS, 2)
        return mz, ph / (ph.norm(dim=-1, keepdim=True) + 1e-6)

    def tf_to_wave(self, mz: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
        mag = torch.exp(mz.float() * self.lm_std + self.lm_mean)
        return tf_istft(torch.complex(mag * ph[..., 0], mag * ph[..., 1]))

    def logpsd(self, x: torch.Tensor) -> torch.Tensor:
        """Epoch log-PSD of the spectral heads at this model's ``fs`` and ``psd_band``."""
        return epoch_logpsd(x, fs=self.fs, seg=4 * self.fs, fmin=self.psd_band[0], fmax=self.psd_band[1])

    def band_logpow(self, x: torch.Tensor) -> torch.Tensor:
        return band_logpow(x, self.fs, self.pow_bands, self.seg_sec)

    def grid_shift(self, *xs: torch.Tensor):
        return grid_shift(self.training, *xs, seg=int(self.seg_sec * self.fs))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        z["spec_n"] = self.spec_n(z["N"])
        z["spec_ne"] = self.spec_ne(torch.cat([z["N"], z["E"]], -1))
        if self.decoder == "istft":
            z["tf_n"], _ = self.decode_tf(z["N"])
            z["tf_ne"], _ = self.decode_tf(z["N"], z["E"])
            z["tf"], z["phase"] = self.decode_tf(z["N"], z["E"], z["P"])
            z["wave"] = self.tf_to_wave(z["tf"], z["phase"])
            return z
        z["wave_n"] = self.decode(z["N"])
        z["wave_ne"] = self.decode(z["N"], z["E"])
        if self.texture:
            z["wave_det"], z["texture"] = self.decode(z["N"], z["E"], z["P"], texture=True)
            z["wave"] = z["wave_det"] + z["texture"]
        else:
            z["wave"] = self.decode(z["N"], z["E"], z["P"])
        return z

    def bits_per_epoch(self) -> int:
        return (self.dim + self.n_tok * self.d_p) * self.latent_bits


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------


def _stft(x, win):
    return torch.stft(x, win, win // 4, window=torch.hann_window(win, device=x.device), return_complex=True)


def stft_losses(pred: torch.Tensor, target: torch.Tensor, windows=STFT_WINDOWS, phase_win: int = 100):
    """Per-sample multi-resolution STFT losses ``(B,)``: log-magnitude L1 + spectral
    convergence, and a magnitude-weighted phase loss ``1 - cos(dphi)`` at ``phase_win``."""
    mag, phase = 0.0, None
    for w in windows:
        sp, st = _stft(pred, w), _stft(target, w)
        ap, at = sp.abs() + 1e-5, st.abs() + 1e-5
        lm = (ap.log() - at.log()).abs().mean((1, 2))
        sc = (at - ap).pow(2).sum((1, 2)).sqrt() / at.pow(2).sum((1, 2)).sqrt()
        mag = mag + (lm + sc) / len(windows)
        if w == phase_win:
            cos = (sp * st.conj()).real / (ap * at)
            wt = at / at.mean((1, 2), keepdim=True)
            phase = ((1 - cos) * wt).mean((1, 2))
    return mag, phase


def fit_msae_norms(model: MSAE, loader, max_batches: int = 50) -> None:
    """:func:`physioex.models.msvq.fit_msvq_norms` plus, for the "istft" decoder, the per-bin
    mean / std of the log spectrogram (train set): the magnitude head starts at this mean."""
    from physioex.data.collate import stack_channels
    from physioex.models.msvq import fit_msvq_norms

    if model.fs != 100 or model.psd_band != (0.5, 35.0):
        ps = []  # the patch-spectrum norms of MSVQ are unused by MSAE and 100 Hz only
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = stack_channels(batch)[:, :, 0].float()
            ps.append(model.logpsd(x[:, x.shape[1] // 2]))
        ps = torch.cat(ps)
        model.psd_mean.copy_(ps.mean(0)); model.psd_std.copy_(ps.std(0) + 1e-6)
    else:
        fit_msvq_norms(model, loader, max_batches)
    if model.decoder != "istft":
        return
    lm = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = stack_channels(batch)[:, :, 0].float()
        lm.append(tf_logmag(tf_stft(x[:, x.shape[1] // 2])).reshape(-1, TF_BINS))
    lm = torch.cat(lm)
    model.lm_mean.copy_(lm.mean(0)); model.lm_std.copy_(lm.std(0) + 1e-6)


def grid_shift(training: bool, *xs: torch.Tensor, seg: int = 100):
    """In training, crop every ``(B, 3000)`` wave to ``[o, o + 2900)`` with one random ``o`` in
    [0, 100): the 1 s segments of :func:`band_logpow` then move at every step. On a fixed grid
    the band loss reached R^2 1.00 per second on that grid but 0.67-0.86 on the grid shifted
    by 0.5 s (job 1548648, 608 numbers per epoch)."""
    if not training:
        return xs
    o = int(torch.randint(0, seg, ()))
    n = xs[0].shape[-1]
    return tuple(x[:, o:o + n - seg] for x in xs)


def band_logpow(x: torch.Tensor, fs: int = 100, bands=POW_BANDS, seg_sec: float = 1.0) -> torch.Tensor:
    """``(B, n) -> (B, n_seg, len(bands))`` log10 power per ``seg_sec`` segment and band
    (band edges in Hz; rfft bin = Hz x ``seg_sec``). Defaults: EEG, 1 s at 100 Hz."""
    seg = int(seg_sec * fs)
    n = x.shape[-1] // seg * seg
    p = torch.fft.rfft(x[:, :n].reshape(len(x), -1, seg), dim=-1).abs().pow(2)
    return torch.stack([torch.log10(p[..., int(a * seg_sec):int(b * seg_sec)].sum(-1) + 1e-4) for a, b in bands], -1)


def lowpass_mse(pred: torch.Tensor, target: torch.Tensor, hz: float = LOW_HZ, fs: int = 100) -> torch.Tensor:
    """Per-sample MSE of the epochs low-passed at ``hz`` (rfft of the whole 30 s epoch)."""
    n = pred.shape[-1]
    k = int(hz * n / fs) + 1
    d = torch.fft.rfft(pred - target, dim=-1)[:, :k]
    d[:, 1:] *= 2 ** 0.5  # one-sided spectrum: count the negative frequencies too
    return d.abs().pow(2).sum(-1) / n ** 2


class MSAETrainer(Trainer):
    """Trainer for :class:`MSAE`. ``acc`` reports R^2 of the full waveform ``x_NEP``.
    Call :func:`fit_msae_norms` first; use ``log_confusion_matrix=False``.

    ``decoder="time"``: MSE in time on ``x_NEP`` + ``stft_weight`` x STFT on every prefix.
    ``decoder="istft"``: MSE of the z-scored log spectrogram for N, N+E and N+E+P (each band
    weighs the same) + for N+E+P the MSE in time and a magnitude-weighted phase loss ``1 - cos``
    (+ ``tf_consistency`` x the same log-spectrogram MSE re-measured on the iSTFT output).
    """

    @staticmethod
    def _step(model, batch, loss_fn, device):
        from physioex.data.collate import stack_channels

        x = stack_channels(batch).to(device)
        xc = x[:, x.shape[1] // 2, 0].float()
        psd = (model.logpsd(xc) - model.psd_mean) / model.psd_std
        with _autocast(device):
            out = model(x)
        if model.recon == "texture":
            return MSAETrainer._step_texture(model, out, xc, psd)
        if model.decoder == "istft":
            return MSAETrainer._step_tf(model, out, xc, psd)
        if model.loss == "band":
            return MSAETrainer._step_band(model, out, xc, psd)
        w_n, w_ne = out["wave_n"].float(), out["wave_ne"].float()
        w = out["wave_det" if model.texture else "wave"].float()
        sw = model.stft_weight
        l_n, _ = stft_losses(w_n, xc, windows=(200, 400), phase_win=0)
        l_ne, _ = stft_losses(w_ne, xc, windows=(100, 200, 400), phase_win=0)
        mag, phase = stft_losses(w, xc)
        mse = (w - xc).pow(2).mean(1)
        l_sn = F.mse_loss(out["spec_n"].float(), psd)
        l_sne = F.mse_loss(out["spec_ne"].float(), psd)
        loss = (mse + sw * (l_n + l_ne + mag)).mean() + l_sn + l_sne
        tex = {}
        if model.texture:
            # the band power reaches only the texture (and through it the latents), never the MSE wave
            hi = model.tex_bands
            xg, tg = model.grid_shift(xc, w.detach() + out["texture"].float())
            l_tex = (model.band_logpow(tg)[..., hi] - model.band_logpow(xg)[..., hi]).abs().mean()
            loss = loss + model.band_weight * l_tex
            tex = {"pow_tex": l_tex.item(), "r2_wave_tex": _r2(out["wave"].float(), xc).item()}
        r2w = _r2(w, xc)
        extra = {"env_n": l_n.mean().item(), "spec_ne_stft": l_ne.mean().item(), "mse": mse.mean().item(),
                 "stft": mag.mean().item(), "phase": phase.mean().item(),
                 "mse_spec_n": l_sn.item(), "mse_spec_ne": l_sne.item(),
                 "r2_spec_n": _r2(out["spec_n"].float(), psd).item(),
                 "r2_spec_ne": _r2(out["spec_ne"].float(), psd).item(),
                 "r2_wave_n": _r2(w_n, xc).item(), "r2_wave_ne": _r2(w_ne, xc).item(), "r2_wave": r2w.item(), **tex}
        return loss, r2w, extra

    @staticmethod
    def _step_texture(model, out, xc, psd):
        """``recon="texture"`` (EMG): every prefix matches the log band power per ``seg_sec``
        segment (L1, grid shifted in training); spectral heads as usual. ``acc`` = R^2 of the
        band power of N+E+P."""
        w_n, w_ne, w = out["wave_n"].float(), out["wave_ne"].float(), out["texture"].float()
        xg, gn, gne, gw = model.grid_shift(xc, w_n, w_ne, w)
        bt = model.band_logpow(xg)
        bn, bne, bw = (model.band_logpow(v) for v in (gn, gne, gw))
        l_n, l_ne, l_nep = [(v - bt).abs().mean() for v in (bn, bne, bw)]
        l_sn = F.mse_loss(out["spec_n"].float(), psd)
        l_sne = F.mse_loss(out["spec_ne"].float(), psd)
        loss = model.band_weight * (l_n + l_ne + l_nep) + l_sn + l_sne
        r2 = _r2(bw, bt)
        extra = {"pow_n": l_n.item(), "pow_ne": l_ne.item(), "pow_nep": l_nep.item(),
                 "mse_spec_n": l_sn.item(), "mse_spec_ne": l_sne.item(),
                 "r2_spec_n": _r2(out["spec_n"].float(), psd).item(),
                 "r2_spec_ne": _r2(out["spec_ne"].float(), psd).item(),
                 "r2_pow_n": _r2(bn, bt).item(), "r2_pow_ne": _r2(bne, bt).item(), "r2_pow": r2.item()}
        return loss, r2, extra

    @staticmethod
    def _step_band(model, out, xc, psd):
        """Perception-oriented loss: N and N+E match the log band power per second on every
        band (envelope only); N+E+P is phase-exact (MSE) up to ``model.low_hz`` and matches
        the band power of the bands starting above ``model.pow_hz`` (default ``low_hz``: split
        spectrum). ``low_hz=50`` with ``pow_hz=4.5`` is the hybrid: MSE on the whole wave, so the
        phase is kept wherever the budget allows, plus band power against the muting.
        Plain MSE drives the unpredictable bands (> 5 Hz) to zero power: 3-50x too weak
        on DCSM at 368 numbers per epoch, PCA included (job 1548628)."""
        w_n, w_ne, w = out["wave_n"].float(), out["wave_ne"].float(), out["wave"].float()
        xg, gn, gne, gw = model.grid_shift(xc, w_n, w_ne, w)
        bt = model.band_logpow(xg)
        bw = model.band_weight
        l_n, l_ne = [(model.band_logpow(v) - bt).abs().mean() for v in (gn, gne)]
        # N+E+P: the spectrum is split, phase-exact MSE up to low_hz and band power only above it
        # (band power on the low bands too pushes energy in with an uncertain phase: 2-4 Hz phase R^2 0.91 -> 0.7)
        hi = [i for i, (a, _) in enumerate(model.pow_bands) if a > model.pow_hz]
        l_nep = (model.band_logpow(gw)[..., hi] - bt[..., hi]).abs().mean()
        low = lowpass_mse(w, xc, model.low_hz, model.fs).mean()
        l_sn = F.mse_loss(out["spec_n"].float(), psd)
        l_sne = F.mse_loss(out["spec_ne"].float(), psd)
        loss = bw * (l_n + l_ne + l_nep) + model.low_weight * low + l_sn + l_sne
        r2w = _r2(w, xc)
        extra = {"pow_n": l_n.item(), "pow_ne": l_ne.item(), "pow_nep": l_nep.item(), "mse_low": low.item(),
                 "mse_spec_n": l_sn.item(), "mse_spec_ne": l_sne.item(),
                 "r2_spec_n": _r2(out["spec_n"].float(), psd).item(),
                 "r2_spec_ne": _r2(out["spec_ne"].float(), psd).item(),
                 "r2_pow_n": _r2(model.band_logpow(gn), bt).item(), "r2_pow_ne": _r2(model.band_logpow(gne), bt).item(),
                 "r2_pow": _r2(model.band_logpow(gw), bt).item(), "r2_wave": r2w.item()}
        return loss, r2w, extra

    @staticmethod
    def _step_tf(model, out, xc, psd):
        st = tf_stft(xc)
        zt = (tf_logmag(st) - model.lm_mean) / model.lm_std
        tf_n, tf_ne, tf = out["tf_n"].float(), out["tf_ne"].float(), out["tf"].float()
        l_n, l_ne, l_nep = F.mse_loss(tf_n, zt), F.mse_loss(tf_ne, zt), F.mse_loss(tf, zt)
        w = out["wave"].float()
        mse = (w - xc).pow(2).mean()
        ph, at = out["phase"], st.abs()
        cos = (ph[..., 0] * st.real + ph[..., 1] * st.imag) / (at + 1e-6)
        phase = ((1 - cos) * at / at.mean((1, 2), keepdim=True)).mean()
        l_sn = F.mse_loss(out["spec_n"].float(), psd)
        l_sne = F.mse_loss(out["spec_ne"].float(), psd)
        loss = l_n + l_ne + l_nep + mse + phase + l_sn + l_sne
        cons = torch.zeros(())
        if model.tf_consistency:  # the power must survive the iSTFT (overlapping frames with clashing phases cancel)
            cons = F.mse_loss((tf_logmag(tf_stft(w)) - model.lm_mean) / model.lm_std, zt)
            loss = loss + model.tf_consistency * cons
        r2w = _r2(w, xc)
        extra = {"mse_tf_n": l_n.item(), "mse_tf_ne": l_ne.item(), "mse_tf": l_nep.item(), "mse": mse.item(),
                 "phase": phase.item(), "consistency": cons.item(), "mse_spec_n": l_sn.item(), "mse_spec_ne": l_sne.item(),
                 "r2_tf_n": _r2(tf_n, zt).item(), "r2_tf_ne": _r2(tf_ne, zt).item(), "r2_tf": _r2(tf, zt).item(),
                 "r2_spec_n": _r2(out["spec_n"].float(), psd).item(),
                 "r2_spec_ne": _r2(out["spec_ne"].float(), psd).item(), "r2_wave": r2w.item()}
        return loss, r2w, extra
