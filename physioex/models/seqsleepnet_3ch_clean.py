"""3-channel SeqSleepNet baseline on the clean channels, with the interface of ``MultiChannelNet``.

The network is physioex ``SeqSleepNet`` (in_chan = 3) unchanged. Its input is rebuilt from the clean 100 Hz signals as
in Phan et al. (and in the physioex ``seqsleepnet`` preset): STFT (Hamming 2 s, 1 s overlap, nfft 256) power in dB,
10 log10(P + eps), per-epoch frames (29 x 129), then standardised per channel and frequency bin with training-set
statistics (the role of Phan's ``scaling.npz``), kept as buffers so they travel with the checkpoint.

Regularisation (xmai-lab/sleep-mechinterp#21): ``dropout`` is set between the stacked LSTM / GRU layers (nn.RNN
``dropout``) and on the sequence-encoder output before the classifier.

forward(x: {m: (B, L, 3000)}, offset_db: {m: (B,)} or None, occlude=None) -> (logits (B, L, 5), {}, {}).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

from physioex.models.seqsleepnet import SeqSleepNet
from physioex.models.seqsleepnet_clean import SpectralFrontend

MODALITIES = ("EEG", "EOG", "EMG")
EPS = float(np.finfo(np.float32).eps)


class SeqSleepNet3chClean(nn.Module):
    def __init__(self, mods: Sequence[str] = MODALITIES, dropout: float = 0.0):
        super().__init__()
        self.mods = list(mods)
        self.frontend = SpectralFrontend()
        self.net = SeqSleepNet(n_classes=5, in_chan=len(self.mods), F=129, D=32, nfft=256, lowfreq=0, highfreq=50,
                               fs=100, seqnhidden1=64, seqnlayer1=4, attentionsize=32, seqnhidden2=64, seqnlayer2=4)
        for mod in self.net.modules():
            if isinstance(mod, (nn.LSTM, nn.GRU)) and mod.num_layers > 1:
                mod.dropout = float(dropout)
        self.drop = nn.Dropout(dropout)
        self.net.clf.register_forward_pre_hook(lambda _m, a: (self.drop(a[0]),))
        self.register_buffer("mean", torch.zeros(len(self.mods), 129))
        self.register_buffer("std", torch.ones(len(self.mods), 129))
        self.disabled: set = set()          # interface of MultiChannelNet (no connection sites)
        self.sites: list = []

    def spectrogram(self, x: Dict[str, torch.Tensor], offset_db: Optional[Dict[str, torch.Tensor]] = None):
        """(B, L, C, 29, 129) power in dB (offset_db added, e.g. the EMG resampling offset), not standardised."""
        specs = []
        for m in self.mods:
            db = 10 * torch.log10(self.frontend(x[m]) + EPS)
            if offset_db is not None and m in offset_db:
                db = db + offset_db[m].view(-1, 1, 1, 1)
            specs.append(db)
        return torch.stack(specs, 2)

    @torch.no_grad()
    def fit_scaling(self, batches):
        """Per channel / frequency mean and std over the frames of some training windows."""
        s = torch.cat([self.spectrogram(x, off).transpose(2, 0).reshape(len(self.mods), -1, 129).transpose(0, 1)
                       for x, off in batches], 0)                              # (N frames, C, 129)
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
