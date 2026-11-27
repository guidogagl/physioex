"""Building blocks of the model zoo of the multimodal framework (xmai-lab/sleep-mechinterp#38, #39).

Every block maps (N, T, d) -> (N, T, d) and ends in a LayerNorm, so it can sit at a connection site of
``MultiChannelNet`` (physioex/models/channelnet_mm.py). Bidirectional blocks run one mixer forwards and one on the
time-reversed input (separate parameters) and sum them. All blocks are pure PyTorch (no custom CUDA kernels):

- ``RNNBlock``       1-layer BiLSTM / BiGRU + LayerNorm (the ChannelNet layer)
- ``TransformerBlock`` post-norm ``nn.TransformerEncoderLayer`` (the SleepTransformer layer)
- ``S4DBlock``       diagonal state-space model (Gu et al. 2022, S4D-Lin init, ZOH), kernel by Vandermonde product,
                     convolution by FFT; residual + GLU output, LayerNorm
- ``MambaBlock``     selective SSM (Gu & Dao 2023): in-projection, causal depthwise conv, input-dependent dt/B/C,
                     sequential selective scan, SiLU gate; residual, LayerNorm
- ``MLSTMBlock``     xLSTM mLSTM (Beck et al. 2024): matrix memory, exponential input gate, sigmoid forget gate,
                     stabilised parallel form, multi-head, group norm, learnable skip, SiLU output gate
- ``SLSTMBlock``     xLSTM sLSTM: scalar memory, exponential gating with stabiliser state, block-diagonal recurrent
                     weights per head, then a gated FFN
- ``ConvBlock``      dilated residual Conv1d (TCN / fully convolutional epoch encoder)

Input stems of the raw encoders (``RawMultiScaleStem``, ``PatchStem``) read ``raw_scale(x, offset_db)``: the cache
epoch (RobustScaled per night) times the EMG bandwidth gain 10^(offset_db/20), compressed by a fixed asinh and
mapped to [-1, 1]. RAW_K and RAW_ZMAX are constants calibrated once on the 24 cache datasets
(examples/pretrained/zoo/raw_scale_stats.py, xmai-lab/sleep-mechinterp#40): no dataset statistics enter the model
and the amplitude level (slow-wave amplitude) is kept, unlike instance or per-epoch normalisation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# raw-input scale (xmai-lab/sleep-mechinterp#40): k = median over datasets x modalities of q90(|z|),
# z_max = max of q99.99(|z|); x_raw = clip(asinh(z / k) / asinh(z_max / k), -1, 1)
# Calibrated on 20 nights x 24 datasets (examples/pretrained/zoo/raw_scale.json): q90 EEG 1.9-2.5, EOG 2.2-3.4,
# EMG 2.3-5.1; q99.99 = 20 on EEG/EOG is the +-20 clip of the cache, only EMG times its gain exceeds it (max 42.7,
# MESA). Clipped fraction <= 1e-4 on every dataset x modality (max 9.95e-5, MESA EMG).
RAW_K = 2.595703125
RAW_ZMAX = 42.7265625


def raw_scale(x: torch.Tensor, offset_db: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x (B, L, 3000) RobustScaled epochs, offset_db (B,) -> fixed-scale raw input in [-1, 1]."""
    if offset_db is not None:
        x = x * (10.0 ** (offset_db / 20.0)).view(-1, *([1] * (x.dim() - 1)))
    return (torch.asinh(x / RAW_K) / math.asinh(RAW_ZMAX / RAW_K)).clamp(-1, 1)


def _flip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[1])


# ------------------------------------------------------------------------------------------------ existing layers
class RNNBlock(nn.Module):
    def __init__(self, cell: str, d_in: int, d: int = 128):
        super().__init__()
        self.rnn = (nn.LSTM if cell == "lstm" else nn.GRU)(d_in, d // 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        raw, _ = self.rnn(x)
        return self.norm(raw), raw


class TransformerBlock(nn.Module):
    def __init__(self, d: int = 128, heads: int = 8, d_ff: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d, heads, d_ff, dropout, batch_first=True)    # post-norm

    def forward(self, x):
        return self.layer(x), None


class ConvBlock(nn.Module):
    """y = LN(x + W2 GELU(dilated Conv1d k3 (x))), 'same' padding (non-causal)."""

    def __init__(self, d: int = 128, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.pw = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = self.pw(F.gelu(self.conv(x.transpose(1, 2)))).transpose(1, 2)
        raw = x + self.drop(h)
        return self.norm(raw), raw


# ------------------------------------------------------------------------------------------------------------ S4D
class S4DKernel(nn.Module):
    """Diagonal SSM kernel, S4D-Lin init: A_n = -1/2 + i pi n, B = 1, C ~ N(0, 1/2) complex, log dt ~ U[log 1e-3,
    log 1e-1]. K[l] = 2 Re(sum_n C_n (exp(dt A_n) - 1) / A_n * exp(dt A_n l))  (zero-order hold)."""

    def __init__(self, d: int, n: int = 64):
        super().__init__()
        n2 = n // 2
        self.log_dt = nn.Parameter(torch.rand(d) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.C = nn.Parameter(torch.view_as_real(torch.randn(d, n2, dtype=torch.cfloat) * 0.5 ** 0.5))
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d, n2)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n2).float().repeat(d, 1))

    def discrete(self):
        dt = self.log_dt.exp()
        A = -self.log_A_real.exp() + 1j * self.A_imag                       # (d, n2)
        dtA = dt[:, None] * A
        C = torch.view_as_complex(self.C) * (dtA.exp() - 1.0) / A           # C * B_bar, B = 1
        return dtA, C

    def forward(self, T: int) -> torch.Tensor:
        dtA, C = self.discrete()
        vand = (dtA[..., None] * torch.arange(T, device=dtA.device)).exp()  # (d, n2, T)
        return 2 * torch.einsum("dn,dnl->dl", C, vand).real                 # (d, T)


def fft_conv(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """u (N, d, T), k (d, T): causal convolution y[t] = sum_{s<=t} k[s] u[t-s]."""
    T = u.shape[-1]
    y = torch.fft.irfft(torch.fft.rfft(u, n=2 * T) * torch.fft.rfft(k, n=2 * T), n=2 * T)
    return y[..., :T]


class S4DBlock(nn.Module):
    def __init__(self, d: int = 128, n: int = 64, dropout: float = 0.0):
        super().__init__()
        self.kf, self.kb = S4DKernel(d, n), S4DKernel(d, n)
        self.D = nn.Parameter(torch.randn(d))
        self.out = nn.Linear(d, 2 * d)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def mix(self, x):
        u = x.transpose(1, 2)                                               # (N, d, T)
        T = u.shape[-1]
        y = fft_conv(u, self.kf(T)) + torch.flip(fft_conv(torch.flip(u, [-1]), self.kb(T)), [-1]) + self.D[:, None] * u
        return y.transpose(1, 2)

    def forward(self, x):
        h = F.glu(self.out(F.gelu(self.mix(x))), dim=-1)
        raw = x + self.drop(h)
        return self.norm(raw), raw


# ---------------------------------------------------------------------------------------------------------- Mamba
class MambaMixer(nn.Module):
    """One-direction Mamba mixer (as mamba_ssm.Mamba / mamba.py MambaBlock, sequential scan)."""

    def __init__(self, d: int = 128, d_state: int = 16, expand: int = 2, d_conv: int = 4, dt_rank: Optional[int] = None,
                 dt_min: float = 1e-3, dt_max: float = 0.1):
        super().__init__()
        E = expand * d
        self.E, self.N = E, d_state
        self.dt_rank = dt_rank or math.ceil(d / 16)
        self.in_proj = nn.Linear(d, 2 * E, bias=False)
        self.conv = nn.Conv1d(E, E, d_conv, groups=E, padding=d_conv - 1)
        self.x_proj = nn.Linear(E, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, E)
        std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(torch.rand(E) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))      # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(E, 1)))
        self.D = nn.Parameter(torch.ones(E))
        self.out_proj = nn.Linear(E, d, bias=False)

    def ssm(self, x):
        """x (N, T, E) -> y (N, T, E): h_t = exp(dt A) h_{t-1} + dt B_t x_t, y_t = C_t h_t + D x_t."""
        A = -self.A_log.exp()                                               # (E, N)
        dbc = self.x_proj(x)
        dt, Bm, Cm = dbc.split([self.dt_rank, self.N, self.N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                                   # (N, T, E)
        dA = torch.exp(dt[..., None] * A)                                   # (N, T, E, N)
        dBx = dt[..., None] * Bm[:, :, None, :] * x[..., None]              # (N, T, E, N)
        h = torch.zeros(x.shape[0], self.E, self.N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(x.shape[1]):
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t, None, :]).sum(-1))
        return torch.stack(ys, 1) + x * self.D

    def forward(self, x):
        T = x.shape[1]
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)
        xi = F.silu(self.conv(xi.transpose(1, 2))[..., :T].transpose(1, 2))
        return self.out_proj(self.ssm(xi) * F.silu(z))


class MambaBlock(nn.Module):
    def __init__(self, d: int = 128, dropout: float = 0.0, **kw):
        super().__init__()
        self.f, self.b = MambaMixer(d, **kw), MambaMixer(d, **kw)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        raw = x + self.drop(self.f(x) + _flip(self.b(_flip(x))))
        return self.norm(raw), raw


# ---------------------------------------------------------------------------------------------------------- mLSTM
def mlstm_parallel(q, k, v, i_pre, f_pre, eps: float = 1e-6):
    """Stabilised parallel mLSTM (xLSTM ``parallel_stabilized_simple``). q, k, v (N, H, T, Dh); i_pre, f_pre
    (N, H, T, 1) -> h (N, H, T, Dh)."""
    N, H, T, Dh = q.shape
    logf = F.logsigmoid(f_pre)
    cs = torch.cat([torch.zeros_like(logf[:, :, :1]), torch.cumsum(logf, dim=-2)], dim=-2)     # (N, H, T+1, 1)
    rep = cs.repeat(1, 1, 1, T + 1)
    lf = (rep - rep.transpose(-2, -1))[:, :, 1:, 1:]
    tril = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    logD = torch.where(tril, lf, torch.tensor(float("-inf"), device=q.device)) + i_pre.transpose(-2, -1)
    m = logD.amax(-1, keepdim=True)
    D = torch.exp(logD - m)
    C = (q @ (k / math.sqrt(Dh)).transpose(-2, -1)) * D
    norm = torch.maximum(C.sum(-1, keepdim=True).abs(), torch.exp(-m))
    return (C / (norm + eps)) @ v


def mlstm_recurrent(q, k, v, i_pre, f_pre, eps: float = 1e-6):
    """Same cell, step by step (C_t = f' C + i' v k^T, n_t = f' n + i' k, stabiliser m_t); for tests."""
    N, H, T, Dh = q.shape
    C = q.new_zeros(N, H, Dh, Dh)
    n = q.new_zeros(N, H, Dh, 1)
    m = q.new_full((N, H, 1, 1), float("-inf"))
    hs = []
    for t in range(T):
        lf = F.logsigmoid(f_pre[:, :, t:t + 1])
        it = i_pre[:, :, t:t + 1]
        m_new = torch.maximum(lf + m, it)
        fp, ip = torch.exp(lf + m - m_new), torch.exp(it - m_new)
        kt = k[:, :, t:t + 1].transpose(-2, -1) / math.sqrt(Dh)             # (N, H, Dh, 1)
        vt = v[:, :, t:t + 1].transpose(-2, -1)
        C = fp * C + ip * (vt @ kt.transpose(-2, -1))
        n = fp * n + ip * kt
        qt = q[:, :, t:t + 1].transpose(-2, -1)
        num = (C @ qt).transpose(-2, -1)
        den = torch.maximum((n.transpose(-2, -1) @ qt).abs(), torch.exp(-m_new))
        hs.append(num / (den + eps))
        m = m_new
    return torch.cat(hs, dim=2)


class MLSTMMixer(nn.Module):
    """xLSTM mLSTM layer (pre up-projection block): up-projection x2, causal conv4 + SiLU feeding q/k, v from the
    pre-conv branch, gates from [q, k, v], multi-head cell, group norm, learnable skip, SiLU(z) output gate."""

    def __init__(self, d: int = 128, heads: int = 4, expand: int = 2, d_conv: int = 4):
        super().__init__()
        E = expand * d
        self.H = heads
        self.up = nn.Linear(d, 2 * E)
        self.conv = nn.Conv1d(E, E, d_conv, groups=E, padding=d_conv - 1)
        self.q, self.k, self.v = nn.Linear(E, E), nn.Linear(E, E), nn.Linear(E, E)
        self.igate, self.fgate = nn.Linear(3 * E, heads), nn.Linear(3 * E, heads)
        nn.init.zeros_(self.igate.weight)
        nn.init.normal_(self.igate.bias, std=0.1)
        nn.init.zeros_(self.fgate.weight)
        with torch.no_grad():
            self.fgate.bias.copy_(torch.linspace(3.0, 6.0, heads))          # xLSTM: forget gates start open
        self.gn = nn.GroupNorm(heads, E)
        self.skip = nn.Parameter(torch.ones(E))
        self.down = nn.Linear(E, d)

    def forward(self, x):
        Nb, T, _ = x.shape
        xm, z = self.up(x).chunk(2, dim=-1)
        xc = F.silu(self.conv(xm.transpose(1, 2))[..., :T].transpose(1, 2))
        q, k, v = self.q(xc), self.k(xc), self.v(xm)
        g = torch.cat([q, k, v], -1)
        ip, fp = self.igate(g), self.fgate(g)                               # (N, T, H)

        def heads(t):
            return t.view(Nb, T, self.H, -1).transpose(1, 2)

        h = mlstm_parallel(heads(q), heads(k), heads(v), ip.transpose(1, 2)[..., None], fp.transpose(1, 2)[..., None])
        h = h.transpose(1, 2).reshape(Nb, T, -1)
        h = self.gn(h.transpose(1, 2)).transpose(1, 2) + self.skip * xc
        return self.down(h * F.silu(z))


class MLSTMBlock(nn.Module):
    def __init__(self, d: int = 128, dropout: float = 0.0, **kw):
        super().__init__()
        self.f, self.b = MLSTMMixer(d, **kw), MLSTMMixer(d, **kw)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        raw = x + self.drop(self.f(x) + _flip(self.b(_flip(x))))
        return self.norm(raw), raw


# ---------------------------------------------------------------------------------------------------------- sLSTM
class SLSTMCell(nn.Module):
    """sLSTM, one direction: z = tanh, i = exp, f = sigmoid (log-space), o = sigmoid; recurrent weights
    block-diagonal per head; stabiliser m_t = max(log f_t + m_{t-1}, i~_t); h = o * c / n."""

    def __init__(self, d_in: int, hidden: int = 64, heads: int = 4):
        super().__init__()
        self.Hd, self.nh = hidden, heads
        self.W = nn.Linear(d_in, 4 * hidden)
        self.R = nn.Parameter(torch.randn(4, heads, hidden // heads, hidden // heads) * (hidden // heads) ** -0.5)
        with torch.no_grad():
            self.W.bias[2 * hidden:3 * hidden].fill_(3.0)                   # forget gate bias: start open

    def forward(self, x):
        Nb, T, _ = x.shape
        wx = self.W(x).view(Nb, T, 4, self.nh, -1)                          # gates z, i, f, o
        h = x.new_zeros(Nb, self.nh, self.Hd // self.nh)
        c, n = torch.zeros_like(h), torch.zeros_like(h)
        m = torch.full_like(h, float("-inf"))
        out = []
        for t in range(T):
            r = torch.einsum("bhd,ghde->bghe", h, self.R)                    # (N, 4, H, Dh)
            pre = wx[:, t] + r
            zt, it, ft, ot = torch.tanh(pre[:, 0]), pre[:, 1], F.logsigmoid(pre[:, 2]), torch.sigmoid(pre[:, 3])
            m_new = torch.maximum(ft + m, it)
            ip, fp = torch.exp(it - m_new), torch.exp(ft + m - m_new)
            c = fp * c + ip * zt
            n = fp * n + ip
            h = ot * c / n
            m = m_new
            out.append(h.reshape(Nb, -1))
        return torch.stack(out, 1)


class SLSTMBlock(nn.Module):
    """y1 = x + W [sLSTM_f(x), sLSTM_b(x)]; y = LN(y1 + GLU-FFN(LN0(y1)))  (xLSTM post up-projection block)."""

    def __init__(self, d: int = 128, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.f, self.b = SLSTMCell(d, d // 2, heads), SLSTMCell(d, d // 2, heads)
        self.proj = nn.Linear(d, d)
        self.n0 = nn.LayerNorm(d)
        self.ffn_up = nn.Linear(d, 2 * (4 * d // 3))
        self.ffn_down = nn.Linear(4 * d // 3, d)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        y1 = x + self.drop(self.proj(torch.cat([self.f(x), _flip(self.b(_flip(x)))], -1)))
        a, g = self.ffn_up(self.n0(y1)).chunk(2, -1)
        raw = y1 + self.drop(self.ffn_down(a * F.gelu(g)))
        return self.norm(raw), raw


# ---------------------------------------------------------------------------------------------------------- stems
class RawMultiScaleStem(nn.Module):
    """Raw epoch (N, 3000) -> 29 frames (N, 29, d): three parallel Conv1d (kernels 0.5 / 1 / 4 s, 'same'), GELU,
    then a Conv1d with a 2-s kernel and 1-s hop (the STFT frame grid). No normalisation layer: the level is kept."""

    def __init__(self, d: int = 128, ch: int = 32, kernels=(50, 100, 400)):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(1, ch, k, padding="same") for k in kernels)
        self.frame = nn.Conv1d(ch * len(kernels), d, 200, stride=100)

    def forward(self, x):
        u = x.unsqueeze(1)
        h = F.gelu(torch.cat([c(u) for c in self.convs], 1))
        return self.frame(h).transpose(1, 2)                                 # (N, 29, d)


class PatchStem(nn.Module):
    """Raw epoch (N, 3000) -> 29 patches of 2 s with 1-s hop -> linear embedding (N, 29, d)."""

    def __init__(self, d: int = 128):
        super().__init__()
        self.proj = nn.Linear(200, d)

    def forward(self, x):
        return self.proj(x.unfold(-1, 200, 100))
