"""Corrected SeqSleepNet for one clean channel (EEG, EOG or EMG): ChannelNet.

Input chain (designed and range-checked on 24 datasets of the clean cache, 23.11.2026):

    x (B, L, 3000), clean epoch at 100 Hz (RobustScale per night)
    SpectralFrontend   fixed: 2-s Hamming frames, 1-s hop, detrend, |rfft 256|^2 as PSD density -> P (29 x 129)
    Filterbank         trained: convex weights inside fixed triangles, applied to LINEAR power (band power)
    log + fixed scale  x' = clip((10 log10(s + eps) + offset_db + 17) / 53, -1, 1)

A convex mean of powers lies between the smallest and the largest bin of its support, so the band power stays inside
the per-bin range verified on the cache: the window [-70, +36] dB clips nothing by construction.

Encoder: 4 split 1-layer BiLSTMs over the 29 frames, LayerNorm after each, attention pooling, LayerNorm -> e;
4 split 1-layer BiGRUs over the L epochs, LayerNorm after each -> q; linear classifier. Every block output is
recorded in ``out`` (after its LayerNorm); a later multimodal model connects the three channel networks by reading
these outputs through a ``GradHook`` on the connection branch (identity / detach / grl), never on the main path.
There is no LayerNorm after the filterbank: over the filters of a frame it would remove the frame's broadband level,
which carries stage information.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from physioex.models.seqsleepnet import AttentionLayer

FS = 100.0
NPERSEG, HOP, NFFT = 200, 100, 256
FREQS = np.linspace(0, FS / 2, NFFT // 2 + 1)            # 129 bins, 0.390625 Hz apart
EPS = float(np.finfo(np.float32).eps)
SCALE_C, SCALE_K = -17.0, 53.0                           # dB window [-70, +36] -> [-1, 1]

# per modality: frequency bins kept (slice of the 129) and filterbank layout
MODALITY = {
    "EEG": dict(bins=slice(1, 83), nfilt=32, spacing="mel8"),     # 0.39-32.0 Hz
    "EOG": dict(bins=slice(1, 83), nfilt=32, spacing="mel8"),
    "EMG": dict(bins=slice(26, 116), nfilt=12, spacing="linear"),  # 10.2-44.9 Hz
}


# ---------------------------------------------------------------------------------------------------- gradient hooks
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grl(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Gradient reversal: identity forward, gradient multiplied by -lam backward."""
    return _GradReverse.apply(x, lam)


class GradHook(nn.Module):
    """Gradient control on an exposed output: 'identity', 'detach' (no gradient back) or 'grl' (reversed)."""

    def __init__(self, mode: str = "identity", lam: float = 1.0):
        super().__init__()
        assert mode in ("identity", "detach", "grl"), mode
        self.mode, self.lam = mode, lam

    def forward(self, x):
        if self.mode == "detach":
            return x.detach()
        if self.mode == "grl":
            return grl(x, self.lam)
        return x


# --------------------------------------------------------------------------------------------------------- front-end
class SpectralFrontend(nn.Module):
    """Fixed PSD, identical to scipy.signal.spectrogram(fs=100, hamming 200, noverlap 100, nfft 256, density)."""

    def __init__(self):
        super().__init__()
        win = torch.hamming_window(NPERSEG, periodic=True, dtype=torch.float64)
        self.register_buffer("win", win.float())
        scale = 1.0 / (FS * float((win ** 2).sum()))
        dens = torch.full((NFFT // 2 + 1,), 2 * scale)
        dens[0] = dens[-1] = scale                        # onesided: DC and Nyquist not doubled
        self.register_buffer("dens", dens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (..., 3000) -> P (..., 29, 129), linear power density."""
        seg = x.unfold(-1, NPERSEG, HOP)                  # (..., 29, 200)
        seg = seg - seg.mean(-1, keepdim=True)            # scipy detrend='constant'
        X = torch.fft.rfft(seg * self.win, n=NFFT)
        return (X.real ** 2 + X.imag ** 2) * self.dens


def filter_edges(spacing: str, nfilt: int, f_lo: float, f_hi: float) -> np.ndarray:
    """nfilt + 2 edge frequencies: triangle j spans edges[j]..edges[j+2] with its vertex at edges[j+1]."""
    if spacing == "linear":
        return np.linspace(f_lo, f_hi, nfilt + 2)
    fb = float(spacing.removeprefix("mel"))           # m = ln(1 + f / fb): linear below fb, log above
    m = np.linspace(np.log1p(f_lo / fb), np.log1p(f_hi / fb), nfilt + 2)
    return fb * np.expm1(m)


def triangles(freqs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(n_bins, nfilt) continuous triangles; a filter whose support misses every bin gets its nearest bin."""
    nf = len(edges) - 2
    S = np.zeros((len(freqs), nf))
    for j in range(nf):
        lo, c, hi = edges[j], edges[j + 1], edges[j + 2]
        S[:, j] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
        if S[:, j].max() == 0:
            S[np.abs(freqs - c).argmin(), j] = 1.0
    return S


class StaticFilterbank(nn.Module):
    """s_j = sum_b w_jb P_b with w_j = S_j exp(W_j) / sum_b S_j exp(W_j): convex, support fixed, shape learned."""

    def __init__(self, freqs: np.ndarray, edges: np.ndarray):
        super().__init__()
        S = torch.tensor(triangles(freqs, edges), dtype=torch.float32)
        self.register_buffer("logS", torch.log(S))      # -inf outside the support
        self.W = nn.Parameter(torch.zeros_like(S))       # W = 0: normalized triangle
        self.register_buffer("centres", torch.tensor(edges[1:-1], dtype=torch.float32))

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.W + self.logS, dim=0)  # (n_bins, nfilt), columns sum to 1

    def forward(self, P: torch.Tensor, xlog: Optional[torch.Tensor] = None) -> torch.Tensor:
        return P @ self.weights()


class DeformableFilterbank(nn.Module):
    """Gaussian filters, learned centre and width, plus a per-frame centre shift bounded at +-max_shift Hz.

    The shift is predicted from the scaled log spectrum of the frame (zero-initialised: starts static). Weights are
    normalized over the bins, so the output is still a convex mean of powers.
    """

    def __init__(self, freqs: np.ndarray, edges: np.ndarray, max_shift: float = 1.5):
        super().__init__()
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))
        c = torch.tensor(edges[1:-1], dtype=torch.float32)
        width = torch.tensor((edges[2:] - edges[:-2]) / 4, dtype=torch.float32)   # ~ triangle sd
        self.mu = nn.Parameter(c)
        self.log_sigma = nn.Parameter(torch.log(width.clamp_min(0.2)))
        self.shift = nn.Linear(len(freqs), len(c))
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)
        self.max_shift = max_shift
        self.last_shift: Optional[torch.Tensor] = None

    def forward(self, P: torch.Tensor, xlog: torch.Tensor) -> torch.Tensor:
        d = self.max_shift * torch.tanh(self.shift(xlog))                 # (..., T, nfilt)
        self.last_shift = d.detach()
        mu = self.mu + d
        z = (self.freqs[:, None] - mu[..., None, :]) / self.log_sigma.exp()  # (..., T, n_bins, nfilt)
        w = torch.softmax(-0.5 * z ** 2, dim=-2)
        return torch.einsum("...b,...bj->...j", P, w)


def scale_db(power: torch.Tensor, offset_db: Optional[torch.Tensor] = None) -> torch.Tensor:
    db = 10 * torch.log10(power + EPS)
    if offset_db is not None:
        db = db + offset_db
    return ((db - SCALE_C) / SCALE_K).clamp(-1, 1)


# ---------------------------------------------------------------------------------------------------------- encoders
class SplitRNN(nn.Module):
    """Stack of 1-layer bidirectional RNNs (LSTM or GRU), LayerNorm after each; every layer output exposed."""

    def __init__(self, cell: str, d_in: int, hidden: int, layers: int, prefix: str, dropout: float = 0.0):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnns = nn.ModuleList(
            rnn(d_in if i == 0 else 2 * hidden, hidden, batch_first=True, bidirectional=True) for i in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(2 * hidden) for _ in range(layers))
        self.prefix = prefix
        self.drop = nn.Dropout(dropout)                    # on each layer output fed onward (train mode only)

    def forward(self, x: torch.Tensor, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        for i, (rnn, ln) in enumerate(zip(self.rnns, self.norms)):
            h, _ = rnn(x)
            out[f"{self.prefix}{i + 1}_raw"] = h           # before LayerNorm (saturation diagnostics)
            x = ln(h)
            out[f"{self.prefix}{i + 1}"] = x                # exposed before dropout
            x = self.drop(x)
        return x


class ChannelNet(nn.Module):
    """Single-channel corrected SeqSleepNet. forward(x (B, L, 3000), offset_db (B,) or None) -> (logits, out)."""

    def __init__(self, modality: str = "EEG", deformable: bool = False, n_classes: int = 5, hidden1: int = 64,
                 layers1: int = 4, attention: int = 32, hidden2: int = 64, layers2: int = 4, dropout: float = 0.0):
        super().__init__()
        cfg = MODALITY[modality]
        self.modality, self.bins = modality, cfg["bins"]
        freqs = FREQS[self.bins]
        edges = filter_edges(cfg["spacing"], cfg["nfilt"], float(freqs[0]), float(freqs[-1]))
        self.frontend = SpectralFrontend()
        self.filterbank = (DeformableFilterbank if deformable else StaticFilterbank)(freqs, edges)
        self.epoch_rnn = SplitRNN("lstm", cfg["nfilt"], hidden1, layers1, "L", dropout)
        self.attention = AttentionLayer(2 * hidden1, attention)
        self.e_norm = nn.LayerNorm(2 * hidden1)
        self.e_drop = nn.Dropout(dropout)
        self.seq_rnn = SplitRNN("gru", 2 * hidden1, hidden2, layers2, "G", dropout)
        self.clf = nn.Linear(2 * hidden2, n_classes)

    def forward(self, x: torch.Tensor, offset_db: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        out: Dict[str, torch.Tensor] = {}
        P = self.frontend(x)[..., self.bins]                        # (B, L, 29, n_bins)
        off = None if offset_db is None else offset_db.view(B, 1, 1, 1)
        xlog = scale_db(P, off)                                     # scaled bins (deformable shift input)
        s = self.filterbank(P, xlog)                                # band power (B, L, 29, nfilt)
        xs = scale_db(s, off)
        out["F"] = xs
        h = self.epoch_rnn(xs.reshape(B * L, xs.shape[2], -1), out)
        e, alpha = self.attention(h, r_alphas=True)
        out["alpha"] = alpha.view(B, L, -1)
        e = self.e_norm(e).view(B, L, -1)
        out["e"] = e
        q = self.seq_rnn(self.e_drop(e), out)
        out["q"] = q
        logits = self.clf(q)
        out["logits"] = logits
        return logits, out

    # --- block API: the steps of forward, one at a time, for the multimodal lockstep model (channelnet_mm) -------
    # Shared with SleepTransformerChannel (sleeptransformer_clean.py); no parameters, so state_dicts are unchanged.
    @property
    def n_epoch_blocks(self) -> int:
        return len(self.epoch_rnn.rnns)

    @property
    def n_seq_blocks(self) -> int:
        return len(self.seq_rnn.rnns)

    @property
    def dim(self) -> int:
        return self.clf.in_features

    def stem(self, x: torch.Tensor, offset_db: Optional[torch.Tensor] = None):
        """(B, L, 3000) -> (frame features (B*L, 29, nfilt), input record (B, L, 29, nfilt))."""
        B, L, _ = x.shape
        P = self.frontend(x)[..., self.bins]
        off = None if offset_db is None else offset_db.view(B, 1, 1, 1)
        xs = scale_db(self.filterbank(P, scale_db(P, off)), off)
        return xs.reshape(B * L, xs.shape[2], -1), xs

    def epoch_block(self, i: int, h: torch.Tensor):
        """Epoch-encoder block i -> (output after LayerNorm, raw output before it)."""
        raw, _ = self.epoch_rnn.rnns[i](h)
        return self.epoch_rnn.norms[i](raw), raw

    def drop_epoch(self, h):
        return self.epoch_rnn.drop(h)

    def pool(self, h: torch.Tensor, B: int, L: int):
        """(B*L, 29, d) -> (epoch embedding (B, L, d), attention weights (B, L, 29))."""
        em, alpha = self.attention(h, r_alphas=True)
        return self.e_norm(em).view(B, L, -1), alpha.view(B, L, -1)

    def drop_e(self, e):
        return self.e_drop(e)

    def seq_start(self, e):
        return e

    def seq_block(self, i: int, q: torch.Tensor):
        raw, _ = self.seq_rnn.rnns[i](q)
        return self.seq_rnn.norms[i](raw), raw

    def drop_seq(self, q):
        return self.seq_rnn.drop(q)

    def head(self, q):
        return self.clf(q)


@torch.no_grad()
def saturation(model: ChannelNet, x: torch.Tensor, offset_db: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Fraction of |h| > 0.9 per epoch-LSTM layer, and of saturated gates (< 0.02 or > 0.98) in the first layer."""
    _, out = model(x, offset_db)
    res = {f"h_sat_L{i + 1}": float((out[f"L{i + 1}_raw"].abs() > 0.9).float().mean())
           for i in range(len(model.epoch_rnn.rnns))}
    rnn = model.epoch_rnn.rnns[0]
    inp = out["F"].reshape(-1, out["F"].shape[2], out["F"].shape[3])
    hraw = out["L1_raw"]
    H = rnn.hidden_size
    sat = []
    for d, sfx in ((0, ""), (1, "_reverse")):
        Wih, Whh = getattr(rnn, f"weight_ih_l0{sfx}"), getattr(rnn, f"weight_hh_l0{sfx}")
        b = getattr(rnn, f"bias_ih_l0{sfx}") + getattr(rnn, f"bias_hh_l0{sfx}")
        h = hraw[..., d * H:(d + 1) * H]
        hprev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], 1) if d == 0 else \
            torch.cat([h[:, 1:], torch.zeros_like(h[:, :1])], 1)
        z = inp @ Wih.T + hprev @ Whh.T + b
        i, f, _, o = z.chunk(4, -1)
        g = torch.sigmoid(torch.stack([i, f, o]))
        sat.append(((g < 0.02) | (g > 0.98)).float().mean())
    res["gate_sat_L1"] = float(torch.stack(sat).mean())
    return res


if __name__ == "__main__":
    m = ChannelNet("EEG")
    lg, o = m(torch.randn(2, 21, 3000))
    print(lg.shape, {k: tuple(v.shape) for k, v in o.items()}, sum(p.numel() for p in m.parameters()))
    print(math.isfinite(float(lg.sum())))
