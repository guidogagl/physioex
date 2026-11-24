"""SleepTransformer instance of the multimodal framework (xmai-lab/sleep-mechinterp#26).

``SleepTransformerChannel``: one clean channel (EEG, EOG or EMG) -> physioex ``SleepTransformer(in_chan=1)``
(Phan et al. 2022) unchanged. Only the input chain is the framework's, as for ChannelNet:

    x (B, L, 3000), clean epoch at 100 Hz
    SpectralFrontend   fixed PSD (2-s Hamming, 1-s hop, nfft 256) -> 29 x 129
    bins 1..128        as in the paper (0-th bin dropped, F = 128 = d_model)
    scale_db           fixed window [-70, +36] dB -> [-1, 1], plus offset_db (EMG resampling): no dataset statistics

There is no filterbank (a SeqSleepNet layer: SleepTransformer reads the bins directly). The encoder layers are
post-norm ``TransformerEncoderLayer``s, so every block output already ends in a LayerNorm and is exposed as is.
The class follows the block API of ``ChannelNet`` (stem / epoch_block / pool / seq_start / seq_block / head) used by
``MultiChannelNet``; the native dropout (0.1) stays inside the transformer layers and the classifier.

``SleepTransformer3chClean``: the 3-channel baseline, physioex ``SleepTransformer(in_chan=3)`` on the clean EEG, EOG
and EMG with the paper input (bins 1..128 in dB, standardised per channel and frequency with training statistics),
with the interface of ``MultiChannelNet`` (as ``SeqSleepNet3chClean``).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

from physioex.models.seqsleepnet_clean import SpectralFrontend, scale_db
from physioex.models.sleeptransformer import SleepTransformer

MODALITIES = ("EEG", "EOG", "EMG")
BINS = slice(1, 129)                       # paper: 0-th bin excluded, F = 128
EPS = float(np.finfo(np.float32).eps)


class SleepTransformerChannel(nn.Module):
    """forward(x (B, L, 3000), offset_db (B,) or None) -> (logits (B, L, 5), out)."""

    def __init__(self, modality: str = "EEG", n_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.modality = modality
        self.frontend = SpectralFrontend()
        self.net = SleepTransformer(n_classes=n_classes, in_chan=1, dropout=dropout)

    # --- block API ---------------------------------------------------------------------------------------------
    @property
    def n_epoch_blocks(self) -> int:
        return len(self.net.epoch_encoder.encoder.layers)

    @property
    def n_seq_blocks(self) -> int:
        return len(self.net.sequence_encoder.encoder.layers)

    @property
    def dim(self) -> int:
        return self.net.d_model

    def stem(self, x: torch.Tensor, offset_db: Optional[torch.Tensor] = None):
        """(B, L, 3000) -> (frame tokens with positional encoding (B*L, 29, 128), scaled input (B, L, 29, 128))."""
        B, L, _ = x.shape
        P = self.frontend(x)[..., BINS]
        off = None if offset_db is None else offset_db.view(B, 1, 1, 1)
        xs = scale_db(P, off)
        return self.net.epoch_encoder.pe(xs.reshape(B * L, xs.shape[2], -1)), xs

    def epoch_block(self, i: int, h: torch.Tensor):
        return self.net.epoch_encoder.encoder.layers[i](h), None

    def drop_epoch(self, h):
        return h

    def pool(self, h: torch.Tensor, B: int, L: int):
        att = self.net.epoch_encoder.attention_pool.attention
        w = torch.softmax(att(h), dim=1)                                   # (B*L, 29, 1), as AttentionPooling
        return (h * w).sum(1).view(B, L, -1), w.view(B, L, -1)

    def drop_e(self, e):
        return e

    def seq_start(self, e):
        return self.net.sequence_encoder.pe(e)

    def seq_block(self, i: int, q: torch.Tensor):
        return self.net.sequence_encoder.encoder.layers[i](q), None

    def drop_seq(self, q):
        return q

    def head(self, q: torch.Tensor):
        B, L, D = q.shape
        return self.net.classifier(q.reshape(B * L, D)).view(B, L, -1)

    # --- forward -----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, offset_db: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        out: Dict[str, torch.Tensor] = {}
        h, out["F"] = self.stem(x, offset_db)
        for i in range(self.n_epoch_blocks):
            h, _ = self.epoch_block(i, h)
            out[f"L{i + 1}"] = h
        e, out["alpha"] = self.pool(h, B, L)
        out["e"] = e
        q = self.seq_start(e)
        for i in range(self.n_seq_blocks):
            q, _ = self.seq_block(i, q)
            out[f"G{i + 1}"] = q
        out["q"] = q
        logits = self.head(q)
        out["logits"] = logits
        return logits, out


class SleepTransformer3chClean(nn.Module):
    """forward(x: {m: (B, L, 3000)}, offset_db: {m: (B,)} or None, occlude=None) -> (logits (B, L, 5), {}, {})."""

    def __init__(self, mods: Sequence[str] = MODALITIES, dropout: float = 0.1):
        super().__init__()
        self.mods = list(mods)
        self.frontend = SpectralFrontend()
        self.net = SleepTransformer(n_classes=5, in_chan=len(self.mods), dropout=dropout)
        self.register_buffer("mean", torch.zeros(len(self.mods), 128))
        self.register_buffer("std", torch.ones(len(self.mods), 128))
        self.disabled: set = set()          # interface of MultiChannelNet (no connection sites)
        self.sites: list = []

    def spectrogram(self, x: Dict[str, torch.Tensor], offset_db: Optional[Dict[str, torch.Tensor]] = None):
        """(B, L, C, 29, 128) power in dB (bins 1..128, offset_db added), not standardised."""
        specs = []
        for m in self.mods:
            db = 10 * torch.log10(self.frontend(x[m])[..., BINS] + EPS)
            if offset_db is not None and m in offset_db:
                db = db + offset_db[m].view(-1, 1, 1, 1)
            specs.append(db)
        return torch.stack(specs, 2)

    @torch.no_grad()
    def fit_scaling(self, batches):
        """Per channel / frequency mean and std over the frames of some training windows (the paper's z-score)."""
        s = torch.cat([self.spectrogram(x, off).transpose(2, 0).reshape(len(self.mods), -1, 128).transpose(0, 1)
                       for x, off in batches], 0)                              # (N frames, C, 128)
        self.mean.copy_(s.mean(0))
        self.std.copy_(s.std(0).clamp(min=1e-3))

    def forward(self, x, offset_db=None, occlude=None):
        s = (self.spectrogram(x, offset_db) - self.mean[:, None, :]) / self.std[:, None, :]
        return self.net(s), {}, {}

    def gates(self):
        return {}

    def new_parameters(self):
        return list(self.net.parameters())

    def branch_parameters(self):
        return []
