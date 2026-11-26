"""Whole-night sequence level on top of a trained multimodal ChannelNet (xmai-lab/sleep-mechinterp#32).

``WholeNightNet`` wraps a trained ``MultiChannelNet`` (epoch-level connections only, e.g. S3-F). The epoch level
(stem, epoch LSTM blocks, F connections, pooling) is frozen and runs without gradient; only the sequence level is
trained, on sequences of any length (padded batches, ``lengths`` given -> packed GRUs, so the backward direction
starts at the true end of each sequence).

Options (each reduces to the wrapped model at initialisation):
    loops K, tied      the per-branch sequence stack (4 BiGRU + LN) applied K times: q_1 = f(e), q_k = q_{k-1} +
                       a_k * f_k(q_{k-1}), a_k a learnable scalar initialised to 0; f_k = f (tied) or a copy of it
    night_head         new masked BiGRU stack (2 layers, 384 -> 384) on the fused input [q_EEG, q_EOG, q_EMG]:
                       z <- z + c * LN(BiGRU(z)), c initialised to 0, then the fused head
    freeze_seq         per-branch sequence stacks and auxiliary heads frozen (only new modules + fused head train)

forward(x {m: (B, L, E)}, offset_db {m: (B,)}, occlude {m: (B,)}, lengths (B,) or None)
    -> (fused logits of the last loop, aux logits of the last loop, out) with out["loops"] = [(fused, aux), ...] for
    deep supervision. With lengths None and B = 1 the computation is the one of ``MultiChannelNet.forward``.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from physioex.models.channelnet_mm import MultiChannelNet


def run_rnn(rnn: nn.RNNBase, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    """Batch-first RNN; packed when lengths is given (outputs at padded positions are 0)."""
    if lengths is None:
        return rnn(x)[0]
    p = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    return pad_packed_sequence(rnn(p)[0], batch_first=True, total_length=x.shape[1])[0]


class SeqStack(nn.Module):
    """A copy of a branch's sequence stack (SplitRNN rnns + norms), run with lengths."""

    def __init__(self, split_rnn: nn.Module):
        super().__init__()
        self.rnns = copy.deepcopy(split_rnn.rnns)
        self.norms = copy.deepcopy(split_rnn.norms)

    def forward(self, q, lengths=None):
        for rnn, ln in zip(self.rnns, self.norms):
            q = ln(run_rnn(rnn, q, lengths))
        return q


class NightHead(nn.Module):
    def __init__(self, dim: int, layers: int = 2):
        super().__init__()
        self.rnns = nn.ModuleList(nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.c = nn.Parameter(torch.zeros(()))

    def forward(self, z, lengths=None):
        h = z
        for rnn, ln in zip(self.rnns, self.norms):
            h = ln(run_rnn(rnn, h, lengths))
        return z + self.c * h


class WholeNightNet(nn.Module):
    def __init__(self, mm: MultiChannelNet, loops: int = 1, tied: bool = True, night_head: bool = False,
                 freeze_seq: bool = False):
        super().__init__()
        assert not any(s.startswith("Q") or s == "E" for s in mm.sites), "only epoch-level (F) connections"
        self.mm = mm
        self.mods = mm.mods
        self.loops, self.tied = loops, tied
        self.gain = nn.ParameterDict({f"{m}_{k}": nn.Parameter(torch.zeros(())) for m in self.mods
                                      for k in range(1, loops)})
        self.extra = nn.ModuleDict()                       # untied copies f_k, k >= 1
        if loops > 1 and not tied:
            for m in self.mods:
                for k in range(1, loops):
                    self.extra[f"{m}_{k}"] = SeqStack(mm.branches[m].seq_rnn)
        self.night = NightHead(len(self.mods) * mm.branches[self.mods[0]].dim) if night_head else None
        self.freeze_seq = freeze_seq
        # frozen part: everything of the wrapped model except the sequence stacks and the heads
        for p in mm.parameters():
            p.requires_grad_(False)
        for m in self.mods:
            if not freeze_seq:
                for p in list(mm.branches[m].seq_rnn.parameters()) + list(mm.branches[m].clf.parameters()):
                    p.requires_grad_(True)
        for p in mm.fused.parameters():
            p.requires_grad_(True)

    @property
    def disabled(self):
        return self.mm.disabled

    @disabled.setter
    def disabled(self, v):                                  # voting() switches connection sites off on the model
        self.mm.disabled = set(v)

    # --- parameter groups ----------------------------------------------------------------------------------------
    def seq_parameters(self):
        ps = [] if self.freeze_seq else [p for m in self.mods for p in self.mm.branches[m].seq_rnn.parameters()]
        return ps + list(self.extra.parameters())

    def head_parameters(self):
        ps = [] if self.freeze_seq else [p for m in self.mods for p in self.mm.branches[m].clf.parameters()]
        ps += list(self.mm.fused.parameters())
        return ps + ([] if self.night is None else [p for n, p in self.night.named_parameters() if n != "c"])

    def gain_parameters(self):
        """Residual gains of the loops and of the night head (all initialised to 0)."""
        return list(self.gain.parameters()) + ([] if self.night is None else [self.night.c])

    def train(self, mode: bool = True):
        super().train(mode)
        self.mm.eval()                                      # frozen epoch level, no dropout anywhere in the branches
        if mode and not self.freeze_seq:                    # trainable sequence stacks (cuDNN RNN backward needs it)
            for m in self.mods:
                self.mm.branches[m].seq_rnn.train()
        return self

    # --- forward -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def epoch_level(self, x, offset_db=None, occlude=None):
        """Frozen part of MultiChannelNet.forward up to the epoch embeddings e {m: (B, L, 128)}."""
        mm = self.mm
        B, L, _ = x[self.mods[0]].shape
        send_e = None if occlude is None else {m: 1.0 - occlude[m].float() for m in self.mods}
        send_f = None if send_e is None else {m: v.repeat_interleave(L) for m, v in send_e.items()}
        h = {}
        for m in self.mods:
            off = None if offset_db is None or m not in offset_db else offset_db[m]
            h[m], _ = mm.branches[m].stem(x[m], off)
        for i in range(mm.n_lstm):
            for m in self.mods:
                h[m], _ = mm.branches[m].epoch_block(i, h[m])
            h = mm._connect(f"F{i + 1}", h, send_f)
            for m in self.mods:
                h[m] = mm.branches[m].drop_epoch(h[m])
        e = {}
        for m in self.mods:
            e[m], _ = mm.branches[m].pool(h[m], B, L)
            e[m] = mm.branches[m].drop_e(e[m])
        return e, send_e

    def _f(self, m, k, q, lengths):
        if k > 0 and not self.tied:
            return self.extra[f"{m}_{k}"](q, lengths)
        br = self.mm.branches[m]
        if lengths is None:                                 # exactly the MultiChannelNet path
            for i in range(self.mm.n_gru):
                q, _ = br.seq_block(i, q)
                q = br.drop_seq(q)
            return q
        for rnn, ln in zip(br.seq_rnn.rnns, br.seq_rnn.norms):
            q = ln(run_rnn(rnn, q, lengths))
        return q

    def _heads(self, q, send_e, lengths, B):
        mm = self.mm
        aux = {m: mm.branches[m].head(q[m]) for m in self.mods}
        qs = [q[m] for m in self.mods]
        if send_e is not None:
            keep = torch.stack([send_e[m] for m in self.mods], -1)
            scale = len(self.mods) / keep.sum(-1, keepdim=True).clamp(min=1)
            qs = [qm * (keep[:, k:k + 1] * scale).view(B, 1, 1) for k, qm in enumerate(qs)]
        z = torch.cat(qs, -1)
        if self.night is not None:
            z = self.night(z, lengths)
        return mm.fused(z), aux

    def forward(self, x: Dict[str, torch.Tensor], offset_db=None, occlude=None, lengths: Optional[torch.Tensor] = None):
        B = x[self.mods[0]].shape[0]
        e, send_e = self.epoch_level(x, offset_db, occlude)
        q = {m: self.mm.branches[m].seq_start(e[m]) for m in self.mods}
        loops = []
        for k in range(self.loops):
            if k == 0:
                q = {m: self._f(m, 0, q[m], lengths) for m in self.mods}
            else:
                q = {m: q[m] + self.gain[f"{m}_{k}"] * self._f(m, k, q[m], lengths) for m in self.mods}
            loops.append(self._heads(q, send_e, lengths, B))
        logits, aux = loops[-1]
        return logits, aux, {"loops": loops, "logits": logits}
