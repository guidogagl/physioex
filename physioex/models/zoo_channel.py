"""Single-channel member of the model zoo of the multimodal framework (xmai-lab/sleep-mechinterp#38, #39).

``ZooChannel(modality, epoch, seq)`` follows the block API of ``ChannelNet`` (stem / epoch_block / pool / seq_start /
seq_block / head), so ``MultiChannelNet``, the trainers, eval_ood and the explanation tools use it unchanged. Every
member has dim 128, 4 epoch blocks over the 29 frames of an epoch (2-s window, 1-s hop: the STFT grid), attention
pooling + LayerNorm (as ChannelNet), 4 sequence blocks, a linear head; every block output ends in a LayerNorm.
Exposed sites: F (input record), L1..L4, e, G1..G4.

Epoch encoders (``epoch``):

    lstm    STFT + filterbank, fixed dB scale  -> BiLSTM x4                (ChannelNet)
    tf      STFT bins 1..128, fixed dB scale   -> PE + Transformer x4      (SleepTransformer epoch encoder)
    s4d / mamba / mlstm / slstm
            STFT + filterbank, fixed dB scale  -> Linear(nfilt, 128) -> block x4
    rawcnn  raw_scale -> multi-scale Conv1d (0.5 / 1 / 4 s) -> 2-s / 1-s framing conv -> dilated ConvBlock x4 (1,2,4,8)
    patch   raw_scale -> 2-s patches, 1-s hop -> Linear(200, 128) + PE -> Transformer x4

Sequence encoders (``seq``): gru (ChannelNet), tf (PE + Transformer), s4d, mamba, mlstm, slstm, tcn (ConvBlock with
dilations 1, 2, 4, 8: receptive field 31 epochs). ``ZooChannel(m, "lstm", "gru")`` has the layer structure of
``ChannelNet(m)``.

One dropout (the protocol's) is used by every block: the framework parameters are the same for all members.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from physioex.models.seqsleepnet import AttentionLayer
from physioex.models.seqsleepnet_clean import (FREQS, MODALITY, SpectralFrontend, StaticFilterbank, filter_edges,
                                               scale_db)
from physioex.models.sleeptransformer import PositionalEncoding
from physioex.models.zoo_blocks import (ConvBlock, MambaBlock, MLSTMBlock, PatchStem, RawMultiScaleStem, RNNBlock,
                                        S4DBlock, SLSTMBlock, TransformerBlock, raw_scale)

EPOCH_ENCODERS = ("lstm", "tf", "s4d", "mamba", "mlstm", "slstm", "rawcnn", "patch")
SEQ_ENCODERS = ("gru", "tf", "s4d", "mamba", "mlstm", "slstm", "tcn")
TF_BINS = slice(1, 129)
DILATIONS = (1, 2, 4, 8)


def make_block(kind: str, i: int, d: int, d_in: int, dropout: float) -> nn.Module:
    if kind in ("lstm", "gru"):
        return RNNBlock(kind, d_in if i == 0 else d, d)
    if kind in ("tf", "patch"):
        return TransformerBlock(d, dropout=dropout)
    if kind == "s4d":
        return S4DBlock(d, dropout=dropout)
    if kind == "mamba":
        return MambaBlock(d, dropout=dropout)
    if kind == "mlstm":
        return MLSTMBlock(d, dropout=dropout)
    if kind == "slstm":
        return SLSTMBlock(d, dropout=dropout)
    if kind in ("rawcnn", "tcn"):
        return ConvBlock(d, DILATIONS[i], dropout=dropout)
    raise ValueError(kind)


class ZooChannel(nn.Module):
    """forward(x (B, L, 3000), offset_db (B,) or None) -> (logits (B, L, 5), out)."""

    def __init__(self, modality: str = "EEG", epoch: str = "lstm", seq: str = "gru", n_classes: int = 5,
                 d: int = 128, n_epoch: int = 4, n_seq: int = 4, attention: int = 32, dropout: float = 0.0):
        super().__init__()
        assert epoch in EPOCH_ENCODERS and seq in SEQ_ENCODERS, (epoch, seq)
        self.modality, self.epoch_kind, self.seq_kind, self.d = modality, epoch, seq, d
        cfg = MODALITY[modality]
        d_in = d
        if epoch in ("lstm", "s4d", "mamba", "mlstm", "slstm"):
            self.bins = cfg["bins"]
            freqs = FREQS[self.bins]
            edges = filter_edges(cfg["spacing"], cfg["nfilt"], float(freqs[0]), float(freqs[-1]))
            self.frontend = SpectralFrontend()
            self.filterbank = StaticFilterbank(freqs, edges)
            if epoch == "lstm":
                d_in = cfg["nfilt"]                       # the first BiLSTM reads the band powers (ChannelNet)
            else:
                self.in_proj = nn.Linear(cfg["nfilt"], d)
        elif epoch == "tf":
            self.frontend = SpectralFrontend()
            self.pe_epoch = PositionalEncoding(d)
        elif epoch == "rawcnn":
            self.raw_stem = RawMultiScaleStem(d)
        else:                                              # patch
            self.raw_stem = PatchStem(d)
            self.pe_epoch = PositionalEncoding(d)
        self.epoch_blocks = nn.ModuleList(make_block(epoch, i, d, d_in, dropout) for i in range(n_epoch))
        self.attention = AttentionLayer(d, attention)
        self.e_norm = nn.LayerNorm(d)
        if seq == "tf":
            self.pe_seq = PositionalEncoding(d)
        self.seq_blocks = nn.ModuleList(make_block(seq, i, d, d, dropout) for i in range(n_seq))
        self.drop = nn.Dropout(dropout)
        self.clf = nn.Linear(d, n_classes)

    # --- block API ---------------------------------------------------------------------------------------------
    @property
    def n_epoch_blocks(self) -> int:
        return len(self.epoch_blocks)

    @property
    def n_seq_blocks(self) -> int:
        return len(self.seq_blocks)

    @property
    def dim(self) -> int:
        return self.d

    def stem(self, x: torch.Tensor, offset_db: Optional[torch.Tensor] = None):
        """(B, L, 3000) -> (frame features (B*L, 29, d_in), input record (B, L, 29, ·))."""
        B, L, _ = x.shape
        k = self.epoch_kind
        if k in ("rawcnn", "patch"):
            xr = raw_scale(x, offset_db)                                   # (B, L, 3000) in [-1, 1]
            h = self.raw_stem(xr.reshape(B * L, -1))                       # (B*L, 29, d)
            if k == "patch":
                h = self.pe_epoch(h)
            return h, xr.unfold(-1, 200, 100)                              # record: the scaled raw frames
        off = None if offset_db is None else offset_db.view(B, 1, 1, 1)
        if k == "tf":
            xs = scale_db(self.frontend(x)[..., TF_BINS], off)
            return self.pe_epoch(xs.reshape(B * L, xs.shape[2], -1)), xs
        P = self.frontend(x)[..., self.bins]
        xs = scale_db(self.filterbank(P), off)
        h = xs.reshape(B * L, xs.shape[2], -1)
        if k != "lstm":
            h = self.in_proj(h)
        return h, xs

    def epoch_block(self, i: int, h: torch.Tensor):
        return self.epoch_blocks[i](h)

    def drop_epoch(self, h):
        return self.drop(h)

    def pool(self, h: torch.Tensor, B: int, L: int):
        em, alpha = self.attention(h, r_alphas=True)
        return self.e_norm(em).view(B, L, -1), alpha.view(B, L, -1)

    def drop_e(self, e):
        return self.drop(e)

    def seq_start(self, e):
        return self.pe_seq(e) if self.seq_kind == "tf" else e

    def seq_block(self, i: int, q: torch.Tensor):
        return self.seq_blocks[i](q)

    def drop_seq(self, q):
        return self.drop(q)

    def head(self, q):
        return self.clf(q)

    # --- forward -----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, offset_db: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        out: Dict[str, torch.Tensor] = {}
        h, out["F"] = self.stem(x, offset_db)
        for i in range(self.n_epoch_blocks):
            h, raw = self.epoch_block(i, h)
            out[f"L{i + 1}"] = h
            if raw is not None:
                out[f"L{i + 1}_raw"] = raw
            h = self.drop_epoch(h)
        e, out["alpha"] = self.pool(h, B, L)
        out["e"] = e
        q = self.seq_start(self.drop_e(e))
        for i in range(self.n_seq_blocks):
            q, raw = self.seq_block(i, q)
            out[f"G{i + 1}"] = q
            if raw is not None:
                out[f"G{i + 1}_raw"] = raw
            q = self.drop_seq(q)
        out["q"] = out[f"G{self.n_seq_blocks}"]
        logits = self.head(q)
        out["logits"] = logits
        return logits, out
