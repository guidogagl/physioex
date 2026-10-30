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
"""
from __future__ import annotations

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
                 token_sec: float = 1.0, latent_bits: int = 8, n_psd_bins: int = 139,
                 stft_weight: float = 0.1, decoder: str = "time", tf_consistency: float = 0.0,
                 loss: str = "mse", band_weight: float = 1.0):
        super().__init__()
        self.L, self.c, self.dim, self.d_p = sequence_length, sequence_length // 2, dim, d_p
        self.strides = STRIDES[token_sec]
        self.n_tok = EPOCH // (self.strides[0] * self.strides[1] * self.strides[2])
        self.latent_bits = latent_bits
        self.stft_weight = stft_weight
        assert decoder in ("time", "istft"), decoder
        self.decoder = decoder
        self.tf_consistency = tf_consistency
        assert loss in ("mse", "band"), loss
        self.loss, self.band_weight = loss, band_weight

        # encoder: multi-scale stem + SEANet downsampling -> (B, dim, T_p)
        ch = [width * 2 ** i for i in range(len(self.strides) + 1)]
        self.stem = MultiScaleStem(width)
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
        if decoder == "time":
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
        """``(B, 3000) -> (B, T_p, dim)`` token features (stem, SEANet, transformer)."""
        h = self.down(self.stem(x[:, None])).transpose(1, 2)
        return self.enc_tok(h + self.pos_tok)

    def encode_context(self, x: torch.Tensor) -> torch.Tensor:
        """N from the neighbours' mean token features (no gradient into the encoder); t masked."""
        B, L = x.shape[:2]
        idx = [i for i in range(L) if i != self.c]
        with torch.no_grad():
            f = self.features(x[:, idx].reshape(-1, EPOCH)).mean(1)
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
    def decode(self, n, e=None, p=None) -> torch.Tensor:
        """One decoder for every prefix: ``D(N)``, ``D(N, E)``, ``D(N, E, P)`` -> ``(B, 3000)``.
        Missing latents count as zero."""
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
        return self.up(self.dec_tok(z).transpose(1, 2))[:, 0]

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


def band_logpow(x: torch.Tensor) -> torch.Tensor:
    """``(B, 3000) -> (B, 30, len(POW_BANDS))`` log10 power per 1 s segment and band."""
    p = torch.fft.rfft(x.reshape(len(x), -1, 100), dim=-1).abs().pow(2)
    return torch.stack([torch.log10(p[..., a:b].sum(-1) + 1e-4) for a, b in POW_BANDS], -1)


def lowpass_mse(pred: torch.Tensor, target: torch.Tensor, hz: float = LOW_HZ) -> torch.Tensor:
    """Per-sample MSE of the epochs low-passed at ``hz`` (rfft of the whole 30 s epoch)."""
    k = int(hz * EPOCH / 100) + 1
    d = torch.fft.rfft(pred - target, dim=-1)[:, :k]
    d[:, 1:] *= 2 ** 0.5  # one-sided spectrum: count the negative frequencies too
    return d.abs().pow(2).sum(-1) / EPOCH ** 2


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
        psd = (epoch_logpsd(xc) - model.psd_mean) / model.psd_std
        with _autocast(device):
            out = model(x)
        if model.decoder == "istft":
            return MSAETrainer._step_tf(model, out, xc, psd)
        if model.loss == "band":
            return MSAETrainer._step_band(model, out, xc, psd)
        w_n, w_ne, w = out["wave_n"].float(), out["wave_ne"].float(), out["wave"].float()
        sw = model.stft_weight
        l_n, _ = stft_losses(w_n, xc, windows=(200, 400), phase_win=0)
        l_ne, _ = stft_losses(w_ne, xc, windows=(100, 200, 400), phase_win=0)
        mag, phase = stft_losses(w, xc)
        mse = (w - xc).pow(2).mean(1)
        l_sn = F.mse_loss(out["spec_n"].float(), psd)
        l_sne = F.mse_loss(out["spec_ne"].float(), psd)
        loss = (mse + sw * (l_n + l_ne + mag)).mean() + l_sn + l_sne
        r2w = _r2(w, xc)
        extra = {"env_n": l_n.mean().item(), "spec_ne_stft": l_ne.mean().item(), "mse": mse.mean().item(),
                 "stft": mag.mean().item(), "phase": phase.mean().item(),
                 "mse_spec_n": l_sn.item(), "mse_spec_ne": l_sne.item(),
                 "r2_spec_n": _r2(out["spec_n"].float(), psd).item(),
                 "r2_spec_ne": _r2(out["spec_ne"].float(), psd).item(),
                 "r2_wave_n": _r2(w_n, xc).item(), "r2_wave_ne": _r2(w_ne, xc).item(), "r2_wave": r2w.item()}
        return loss, r2w, extra

    @staticmethod
    def _step_band(model, out, xc, psd):
        """Perception-oriented loss: every prefix matches the log band power per second
        (N, N+E: envelope only); N+E+P adds the phase-exact MSE below ``LOW_HZ``.
        Plain MSE drives the unpredictable bands (> 5 Hz) to zero power: 3-50x too weak
        on DCSM at 368 numbers per epoch, PCA included (job 1548628)."""
        bt = band_logpow(xc)
        w_n, w_ne, w = out["wave_n"].float(), out["wave_ne"].float(), out["wave"].float()
        bw = model.band_weight
        l_n, l_ne, l_nep = [(band_logpow(v) - bt).abs().mean() for v in (w_n, w_ne, w)]
        low = lowpass_mse(w, xc).mean()
        l_sn = F.mse_loss(out["spec_n"].float(), psd)
        l_sne = F.mse_loss(out["spec_ne"].float(), psd)
        loss = bw * (l_n + l_ne + l_nep) + low + l_sn + l_sne
        r2w = _r2(w, xc)
        extra = {"pow_n": l_n.item(), "pow_ne": l_ne.item(), "pow_nep": l_nep.item(), "mse_low": low.item(),
                 "mse_spec_n": l_sn.item(), "mse_spec_ne": l_sne.item(),
                 "r2_spec_n": _r2(out["spec_n"].float(), psd).item(),
                 "r2_spec_ne": _r2(out["spec_ne"].float(), psd).item(),
                 "r2_pow_n": _r2(band_logpow(w_n), bt).item(), "r2_pow_ne": _r2(band_logpow(w_ne), bt).item(),
                 "r2_pow": _r2(band_logpow(w), bt).item(), "r2_wave": r2w.item()}
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
