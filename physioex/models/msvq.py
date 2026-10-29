"""MSVQ: multi-scale residual vector quantization of single-channel EEG.

Three dictionaries in cascade, each one reading what the previous ones left:

* **N** (context, ``n_codes_n`` codes): the epochs around epoch ``t`` (t itself
  masked out), i.e. what the neighbourhood predicts about ``t``;
* **E** (epoch, ``n_codes_e`` codes): the 30 s epoch ``t``, residual on N;
* **P** (patch, ``n_codes_p`` codes): each 1 s patch of ``t``, residual on N+E.

The residual is taken in the latent space (VAR-style): each scale quantizes
``h_scale - W * (sum of the coarser codes)``. Encoder, dictionaries and decoder
are trained together: every prefix of the cascade redraws the centre epoch, an
epoch spectrum for N and N+E (log-PSD, every bin z-scored so no band dominates)
and a 1 s waveform for N+E+P (log-magnitude + phase on the unit circle,
NeuroRVQ-style). Label-free cluster targets (:func:`compute_msvq_targets`) can
be added as an auxiliary CE. Sleep stages are never used.

Input: ``(B, L, 1, 3000)`` clean EEG at 100 Hz, L odd (centre = epoch ``t``).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physioex.explain.prototypes.posthoc.vq import VQBottleneck, learn_codebook_kmeans
from physioex.train.trainer import Trainer

FS = 100
EPOCH = 3000
PATCH = 100
N_PATCH = EPOCH // PATCH
# Log-spaced bands (Hz): slow / delta / theta / alpha / sigma / beta
BANDS = ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (11.0, 16.0), (16.0, 30.0))


# ----------------------------------------------------------------------------
# Quantizer
# ----------------------------------------------------------------------------


class EMAVQ(VQBottleneck):
    """:class:`VQBottleneck` with EMA codebook updates and dead-code restart.

    The codebook is not trained by the optimizer: during training it follows
    the exponential moving average of the vectors assigned to each code
    (as in ``physioex.models.sleep_tokenizer``), and a code unused for
    ``restart_after`` steps is moved onto a random vector of the batch. The
    loss is the commitment term only. ``last_idx`` keeps the codes of the last
    call. The first training batch initialises the codebook with k-means.
    """

    def __init__(self, n_codes: int, dim: int, commitment_weight: float = 0.25,
                 decay: float = 0.99, restart_after: int = 200):
        super().__init__(np.zeros((n_codes, dim), np.float32), commitment_weight)
        self.codebook.requires_grad_(False)
        self.decay = decay
        self.restart_after = restart_after
        self.register_buffer("ema_count", torch.ones(n_codes))
        self.register_buffer("ema_sum", torch.zeros(n_codes, dim))
        self.register_buffer("idle", torch.zeros(n_codes, dtype=torch.long))
        self.register_buffer("initialised", torch.tensor(False))
        self.last_idx: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _init(self, z: torch.Tensor) -> None:
        k = self.codebook.shape[0]
        zc = z.detach().float().cpu().numpy()
        if len(zc) < k:
            zc = zc[np.random.randint(0, len(zc), k)] + 1e-3 * np.random.randn(k, zc.shape[1]).astype(np.float32)
        cb = torch.from_numpy(learn_codebook_kmeans(zc, n_prototypes=k, batch_size=max(256, k))).to(z.device)
        self.codebook.copy_(cb)
        self.ema_sum.copy_(cb)
        self.ema_count.fill_(1.0)
        self.initialised.fill_(True)

    def forward(self, z: torch.Tensor):
        z = z.float()
        if self.training and not bool(self.initialised):
            self._init(z)
        dist = (z.pow(2).sum(1, keepdim=True) + self.codebook.pow(2).sum(1)[None]
                - 2 * z @ self.codebook.T)
        idx = dist.argmin(1)
        self.last_idx = idx
        z_q = self.codebook[idx]
        if self.training:
            self._ema_update(z.detach(), idx)
        loss = self.commitment_weight * (z - z_q.detach()).pow(2).mean()
        return z + (z_q - z).detach(), loss

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, idx: torch.Tensor) -> None:
        k = self.codebook.shape[0]
        onehot = F.one_hot(idx, k).type(z.dtype)
        count = onehot.sum(0)
        self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(onehot.T @ z, alpha=1 - self.decay)
        n = self.ema_count.sum()
        smoothed = (self.ema_count + 1e-5) / (n + k * 1e-5) * n
        self.codebook.copy_(self.ema_sum / smoothed[:, None])
        self.idle.add_(1).masked_fill_(count > 0, 0)
        dead = torch.where(self.idle > self.restart_after)[0]
        if len(dead):
            pick = z[torch.randint(0, len(z), (len(dead),), device=z.device)]
            self.codebook[dead] = pick
            self.ema_sum[dead] = pick
            self.ema_count[dead] = 1.0
            self.idle[dead] = 0


def code_stats(idx: torch.Tensor, n_codes: int) -> tuple[float, float]:
    """Perplexity and fraction of codes used in a batch of indices."""
    p = torch.bincount(idx.reshape(-1), minlength=n_codes).float()
    p = p / p.sum().clamp_min(1)
    perplexity = torch.exp(-(p * (p + 1e-10).log()).sum()).item()
    return perplexity, (p > 0).float().mean().item()


# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------


class PatchStem(nn.Module):
    """Multi-scale conv stem on 1 s patches (inception branches as in NeuroRVQ,
    kernels scaled to 100 Hz): ``(..., 100) -> (..., dim)``."""

    def __init__(self, dim: int = 64, filters: int = 8,
                 kernels=((11, 5), (7, 3), (5, 3), (3, 3)), pool: int = 4):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, filters, k1, padding=k1 // 2), nn.GroupNorm(4, filters), nn.GELU(),
                nn.Conv1d(filters, filters, k2, padding=k2 // 2), nn.GroupNorm(4, filters), nn.GELU(),
                nn.AdaptiveAvgPool1d(pool),
            )
            for k1, k2 in kernels
        ])
        self.proj = nn.Linear(len(kernels) * filters * pool, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        x = x.reshape(-1, 1, x.shape[-1])
        h = torch.cat([b(x).flatten(1) for b in self.branches], dim=1)
        return self.proj(h).reshape(*shape, -1)


def _transformer(dim: int, layers: int, heads: int = 4) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.1, batch_first=True,
                                       norm_first=True, activation="gelu")
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class MSVQ(nn.Module):
    """Multi-scale residual VQ encoder (N -> E -> P) with its decoder heads.

    Encoder, dictionaries and decoder are trained together: every prefix of
    the cascade must redraw the centre epoch ``t``.

    * spectral head on N (t never seen: the spectrum the context predicts)
      and on N+E -> log-PSD of ``t``, every bin z-scored;
    * waveform head on N+E+P[j] -> log|X| (z-scored per bin), (sin, cos) phi of
      second j; the waveform is the ``irfft`` of the prediction.

    Args:
        sequence_length: epochs in the input window (odd); the centre is ``t``.
        dim: latent width of every scale.
        n_codes_n / n_codes_e / n_codes_p: dictionary sizes.
        quantize: False bypasses the three dictionaries (continuous latents):
            the upper bound used to size them.
        n_targets_e / n_targets_p: label-free cluster heads (optional
            auxiliary CE, see :func:`compute_msvq_targets`); 0 = no heads.
    """

    def __init__(self, sequence_length: int = 41, dim: int = 64,
                 n_codes_n: int = 32, n_codes_e: int = 128, n_codes_p: int = 256,
                 quantize: bool = True, n_targets_e: int = 0, n_targets_p: int = 0,
                 commitment_weight: float = 0.25, n_psd_bins: int = 139):
        super().__init__()
        assert sequence_length % 2 == 1, "sequence_length must be odd"
        self.L, self.c, self.dim = sequence_length, sequence_length // 2, dim
        self.n_codes = {"N": n_codes_n, "E": n_codes_e, "P": n_codes_p}
        self.quantize_codes = quantize

        self.stem = PatchStem(dim)
        self.pos_epoch = nn.Parameter(torch.zeros(1, sequence_length, dim))
        self.pos_patch = nn.Parameter(torch.zeros(1, N_PATCH, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.pos_epoch, std=0.02)
        nn.init.trunc_normal_(self.pos_patch, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.enc_n = _transformer(dim, 2)
        self.enc_e = _transformer(dim, 2)
        self.enc_p = _transformer(dim, 2)
        self.pre_n, self.pre_e, self.pre_p = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        # what the coarser prefix already explains, subtracted before the next VQ
        self.from_n, self.from_ne = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.vq_n = EMAVQ(n_codes_n, dim, commitment_weight)
        self.vq_e = EMAVQ(n_codes_e, dim, commitment_weight)
        self.vq_p = EMAVQ(n_codes_p, dim, commitment_weight)

        # decoder
        self.n_rfft = PATCH // 2 + 1
        self.scale_emb = nn.Embedding(2, dim)  # 0 = prefix N, 1 = prefix N+E
        self.spec_head = _mlp(dim, n_psd_bins)
        self.wave_head = _mlp(dim, 3 * self.n_rfft)
        self.register_buffer("psd_mean", torch.zeros(n_psd_bins))
        self.register_buffer("psd_std", torch.ones(n_psd_bins))
        self.register_buffer("mag_mean", torch.zeros(self.n_rfft))
        self.register_buffer("mag_std", torch.ones(self.n_rfft))

        self.head_e = nn.Linear(dim, n_targets_e) if n_targets_e else None
        self.head_n = nn.Linear(dim, n_targets_e) if n_targets_e else None
        self.head_p = nn.Linear(dim, n_targets_p) if n_targets_p else None

    def _vq(self, vq: EMAVQ, z: torch.Tensor):
        if not self.quantize_codes:
            return z, z.new_zeros(()), torch.zeros(len(z), dtype=torch.long, device=z.device)
        q, loss = vq(z)
        return q, loss, vq.last_idx

    def quantize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Codes and prefix embeddings of the centre epoch.

        Returns a dict with ``k_n (B,)``, ``k_e (B,)``, ``k_p (B, 30)``, the
        prefix embeddings ``s_n (B, d)``, ``s_ne (B, d)``, ``s_nep (B, 30, d)``
        and ``vq_loss``.
        """
        B, L = x.shape[:2]
        patches = x[:, :, 0].reshape(B, L, N_PATCH, PATCH)
        tok = self.stem(patches)                                    # (B, L, 30, d)

        # N: neighbourhood with t masked out
        ep = tok.mean(2)
        ep = torch.cat([ep[:, :self.c], self.mask_token.expand(B, 1, -1), ep[:, self.c + 1:]], 1)
        h_n = self.enc_n(ep + self.pos_epoch)[:, self.c]
        s_n, l_n, k_n = self._vq(self.vq_n, self.pre_n(h_n))

        # E: epoch t, residual on N
        pt = self.enc_e(tok[:, self.c] + self.pos_patch)            # (B, 30, d)
        h_e = self.pre_e(pt.mean(1))
        q_e, l_e, k_e = self._vq(self.vq_e, h_e - self.from_n(s_n))
        s_ne = s_n + q_e

        # P: each second of t, residual on N+E
        h_p = self.pre_p(self.enc_p(pt))                             # (B, 30, d)
        r_p = h_p - self.from_ne(s_ne)[:, None]
        q_p, l_p, k_p = self._vq(self.vq_p, r_p.reshape(-1, self.dim))
        s_nep = s_ne[:, None] + q_p.reshape(B, N_PATCH, self.dim)

        return {"k_n": k_n, "k_e": k_e, "k_p": k_p.reshape(B, N_PATCH), "s_n": s_n, "s_ne": s_ne,
                "s_nep": s_nep, "vq_loss": l_n + l_e + l_p}

    def draw_spectrum(self, s: torch.Tensor, scale: int) -> torch.Tensor:
        """z-scored log-PSD of an epoch from a prefix embedding (0 = N, 1 = N+E)."""
        return self.spec_head(s + self.scale_emb.weight[scale])

    def draw_patch(self, s: torch.Tensor):
        """``(log|X| z-scored, sin, cos, waveform)`` of a second from an N+E+P embedding."""
        o = self.wave_head(s).float().reshape(*s.shape[:-1], 3, self.n_rfft)
        mag_z, sin, cos = o[..., 0, :], o[..., 1, :], o[..., 2, :]
        mag = torch.exp((mag_z * self.mag_std + self.mag_mean).clamp(max=20))
        wave = torch.fft.irfft(torch.polar(mag, torch.atan2(sin, cos)), n=PATCH, dim=-1)
        return mag_z, sin, cos, wave

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.quantize(x)
        out["spec_n"] = self.draw_spectrum(out["s_n"], 0)
        out["spec_ne"] = self.draw_spectrum(out["s_ne"], 1)
        out["mag_z"], out["sin"], out["cos"], out["wave"] = self.draw_patch(out["s_nep"])
        if self.head_e is not None:
            out["logits_n"] = self.head_n(out["s_n"])
            out["logits_e"] = self.head_e(out["s_ne"])
            out["logits_p"] = self.head_p(out["s_nep"])
        return out

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Codes plus ``emb = [s_n; s_ne; mean_j s_nep]`` (B, 3d) for probing."""
        out = self.quantize(x)
        out["emb"] = torch.cat([out["s_n"], out["s_ne"], out["s_nep"].mean(1)], 1)
        return out


# ----------------------------------------------------------------------------
# Label-free targets
# ----------------------------------------------------------------------------


def _band_logpower(x: torch.Tensor, fs: int = FS) -> torch.Tensor:
    """Log mean power in :data:`BANDS` of the last axis (Hann window)."""
    n = x.shape[-1]
    spec = torch.fft.rfft(x * torch.hann_window(n, device=x.device, dtype=x.dtype), dim=-1).abs().pow(2)
    f = torch.fft.rfftfreq(n, 1 / fs).to(x.device)
    cols = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        if not m.any():  # 1 s patch: bands narrower than 1 Hz take the nearest bin
            m = (f - (lo + hi) / 2).abs() == (f - (lo + hi) / 2).abs().min()
        cols.append(spec[..., m].mean(-1))
    return torch.log(torch.stack(cols, -1) + 1e-6)


def night_descriptors(sig: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Epoch and second descriptors of one night, z-scored per night.

    Args:
        sig: ``(n_epochs, 3000)`` clean EEG.
        lo, hi: main sleep period ``[lo, hi)``; statistics come from it only.
    Returns:
        ``d_e (n, 6)`` band log-powers of the epoch and ``d_p (n, 30, 8)``
        band log-powers of each second plus log peak-to-peak and log line length.
    """
    x = torch.from_numpy(np.asarray(sig, np.float32))
    d_e = _band_logpower(x)
    p = x.reshape(len(x), N_PATCH, PATCH)
    ptp = torch.log(p.amax(-1) - p.amin(-1) + 1e-6)
    ll = torch.log(p.diff(dim=-1).abs().mean(-1) + 1e-6)
    d_p = torch.cat([_band_logpower(p), ptp[..., None], ll[..., None]], -1)

    def z(d):
        ref = d[lo:hi].reshape(-1, d.shape[-1])
        return ((d - ref.mean(0)) / (ref.std(0) + 1e-6)).numpy()

    return z(d_e), z(d_p)


def compute_msvq_targets(dataset, train_ids, n_targets_e: int = 64, n_targets_p: int = 64,
                         max_fit: int = 400_000, seed: int = 0) -> Dict[str, object]:
    """Label-free epoch / second targets for every subject of ``dataset``.

    Descriptors (:func:`night_descriptors`) are clustered with
    :func:`~physioex.explain.prototypes.posthoc.vq.learn_codebook_kmeans` on the
    main sleep period of the *train* subjects, then every epoch of every
    subject is assigned to its nearest centroid. Epochs outside the main
    period get ``-1`` (ignored by the loss). The hypnogram only delimits the
    main period, as ``trim_mode="main_period"`` does; it is never a target.

    Returns:
        ``{"y_e": {sid: (n,)}, "y_p": {sid: (n, 30)}, "centroids_e", "centroids_p"}``.
    """
    rng = np.random.default_rng(seed)
    desc = {}
    for spec in dataset._subjects:
        sid = spec.subject_id
        n = dataset._n_epochs[sid]
        item = dataset._build_item(spec, 0, n)
        sig = next(iter(item["signals"].values())).numpy().reshape(n, -1)
        lo, hi = dataset._main_period.get(sid, (0, n))
        desc[sid] = (*night_descriptors(sig, lo, hi), lo, hi)

    train = [s for s in train_ids if s in desc]
    fit_e = np.concatenate([desc[s][0][desc[s][2]:desc[s][3]] for s in train])
    fit_p = np.concatenate([desc[s][1][desc[s][2]:desc[s][3]].reshape(-1, 8) for s in train])
    if len(fit_p) > max_fit:
        fit_p = fit_p[rng.choice(len(fit_p), max_fit, replace=False)]
    cen_e = learn_codebook_kmeans(fit_e.astype(np.float32), n_targets_e, random_state=seed)
    cen_p = learn_codebook_kmeans(fit_p.astype(np.float32), n_targets_p, random_state=seed)

    def assign(d, cen):
        flat = torch.from_numpy(d.reshape(-1, d.shape[-1]).astype(np.float32))
        return torch.cdist(flat, torch.from_numpy(cen)).argmin(1).numpy().reshape(d.shape[:-1])

    y_e, y_p = {}, {}
    for sid, (d_e, d_p, lo, hi) in desc.items():
        ye, yp = assign(d_e, cen_e).astype(np.int64), assign(d_p, cen_p).astype(np.int64)
        ye[:lo] = ye[hi:] = -1
        yp[:lo] = yp[hi:] = -1
        y_e[sid], y_p[sid] = ye, yp
    return {"y_e": y_e, "y_p": y_p, "centroids_e": cen_e, "centroids_p": cen_p}


def batch_targets(model, batch, device):
    """Targets of the centre epoch of each window: ``(y_e (B,), y_p (B, 30))``."""
    tg = model.targets
    c = batch["epoch_indices"][:, batch["epoch_indices"].shape[1] // 2].tolist()
    sids = [s["id"] for s in batch["subject"]]
    y_e = torch.tensor([int(tg["y_e"][s][i]) for s, i in zip(sids, c)], device=device)
    y_p = torch.from_numpy(np.stack([tg["y_p"][s][i] for s, i in zip(sids, c)])).to(device)
    return y_e, y_p


def _autocast(device):
    # native bf16 only (Ampere and later): on older GPUs (T4) it is emulated and ~50x slower
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


def _acc(logits, y):
    m = y >= 0
    return (logits.argmax(-1)[m] == y[m]).float().mean() if m.any() else torch.tensor(0.0)


def _r2(pred, target):
    return 1 - F.mse_loss(pred, target).detach() / target.var().clamp_min(1e-6)


class MSVQTrainer(Trainer):
    """Trainer for :class:`MSVQ` (encoder, dictionaries and decoder together).

    loss = MSE(spectrum from N) + MSE(spectrum from N+E)   [z-scored log-PSD bins]
    + MSE(z-scored log|X|) + phase loss on the unit circle ``1 - cos(dphi) + 0.4 (|v| - 1)^2``
    + MSE of the waveform in time + VQ commitment
    (+ ``model.ce_weight`` x CE of the cluster heads on ``model.targets``, when present).

    ``acc`` reports R^2 of the N+E spectrum. Call :func:`fit_msvq_norms` first.
    Use ``log_confusion_matrix=False``.
    """

    @staticmethod
    def _step(model, batch, loss_fn, device):
        from physioex.data.collate import stack_channels

        x = stack_channels(batch).to(device)
        xc = x[:, x.shape[1] // 2, 0].float()
        psd = (epoch_logpsd(xc) - model.psd_mean) / model.psd_std
        p = xc.reshape(len(xc), N_PATCH, PATCH)
        mag, sin_t, cos_t = patch_spectrum(p)
        mag = (mag - model.mag_mean) / model.mag_std
        with _autocast(device):
            out = model(x)
        spec_n, spec_ne = out["spec_n"].float(), out["spec_ne"].float()
        l_sn, l_sne = F.mse_loss(spec_n, psd), F.mse_loss(spec_ne, psd)
        l_mag = F.mse_loss(out["mag_z"], mag)
        sin, cos = out["sin"], out["cos"]
        norm = torch.sqrt(sin ** 2 + cos ** 2 + 1e-8)
        l_phase = (1 - (sin * sin_t + cos * cos_t) / norm).mean() + 0.4 * (norm - 1).pow(2).mean()
        l_time = F.mse_loss(out["wave"], p)
        loss = l_sn + l_sne + l_mag + l_phase + l_time + out["vq_loss"]
        extra = {"mse_spec_n": l_sn.item(), "mse_spec_ne": l_sne.item(), "mse_mag": l_mag.item(),
                 "phase": l_phase.item(), "mse_time": l_time.item(), "vq": float(out["vq_loss"]),
                 "r2_spec_n": _r2(spec_n, psd).item(), "r2_mag": _r2(out["mag_z"], mag).item(),
                 "r2_wave": _r2(out["wave"], p).item()}

        ce_w = getattr(model, "ce_weight", 0.0)
        if ce_w and "logits_e" in out and getattr(model, "targets", None) is not None:
            y_e, y_p = batch_targets(model, batch, device)
            ce = lambda lg, y: F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), ignore_index=-1)
            l_ce = ce(out["logits_n"], y_e) + ce(out["logits_e"], y_e) + ce(out["logits_p"], y_p)
            loss = loss + ce_w * l_ce
            extra["ce"] = l_ce.item()
            extra["acc_e"] = _acc(out["logits_e"], y_e).item()

        if model.quantize_codes:
            for s, k in (("n", "k_n"), ("e", "k_e"), ("p", "k_p")):
                perp, used = code_stats(out[k], model.n_codes[s.upper()])
                extra[f"perplexity_{s}"], extra[f"used_{s}"] = perp, used
        return loss, _r2(spec_ne, psd), extra


@torch.no_grad()
def fit_msvq_norms(model: MSVQ, loader, max_batches: int = 50) -> None:
    """Per-bin mean / std of the epoch log-PSD and of the patch log|X| (train set)."""
    from physioex.data.collate import stack_channels

    ps, ms = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = stack_channels(batch)[:, :, 0].float()
        xc = x[:, x.shape[1] // 2]
        ps.append(epoch_logpsd(xc))
        ms.append(patch_spectrum(xc.reshape(len(xc), N_PATCH, PATCH))[0].reshape(-1, PATCH // 2 + 1))
    ps, ms = torch.cat(ps), torch.cat(ms)
    model.psd_mean.copy_(ps.mean(0)); model.psd_std.copy_(ps.std(0) + 1e-6)
    model.mag_mean.copy_(ms.mean(0)); model.mag_std.copy_(ms.std(0) + 1e-6)


# ----------------------------------------------------------------------------
# Reconstruction targets
# ----------------------------------------------------------------------------


def epoch_logpsd(x: torch.Tensor, fs: int = FS, seg: int = 400, fmax: float = 35.0) -> torch.Tensor:
    """Welch log-PSD of 30 s epochs, 4 s Hann segments, 50% overlap, bins 0.5-35 Hz
    (0.25 Hz resolution): ``(..., 3000) -> (..., n_bins)``."""
    w = torch.hann_window(seg, device=x.device, dtype=x.dtype)
    s = x.unfold(-1, seg, seg // 2)
    p = torch.fft.rfft(s * w, dim=-1).abs().pow(2).mean(-2)
    f = torch.fft.rfftfreq(seg, 1 / fs)
    return torch.log(p[..., (f >= 0.5) & (f <= fmax)] + 1e-6)


def patch_spectrum(x: torch.Tensor):
    """rfft of 1 s patches: ``(log|X|, sin phi, cos phi)``, each ``(..., 51)``."""
    X = torch.fft.rfft(x, dim=-1)
    return torch.log(X.abs() + 1e-6), torch.sin(X.angle()), torch.cos(X.angle())


def _mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))
