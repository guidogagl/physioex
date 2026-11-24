"""Multimodal ChannelNet: the three single-channel networks (EEG, EOG, EMG) joined by gated cross-modal connections.

The branches are pretrained single-channel networks with the block API of ``ChannelNet``
(physioex/models/seqsleepnet_clean.py) or ``SleepTransformerChannel`` (physioex/models/sleeptransformer_clean.py),
used unchanged and run in lockstep, block by block. All modalities share the same 29 STFT frames per epoch, so frame-level states are aligned.

Connection points (every one on a LayerNorm output, which also feeds the branch's own next block):
    F  epoch-LSTM layers L1..L4  (B*L, 29, 128), per frame
    E  epoch embedding e_t       (B, L, 128), after attention
    Q  sequence-GRU layers G1..G4 (B, L, 128), per epoch

At a point, for target m:  h_m <- h_m + sum_{n != m} g_mn * W_mn(hook(h_n)),  g_mn a scalar initialised to 0, so with
all gates at 0 the model equals the three pretrained networks. The source goes through a ``GradHook``
(identity / detach / grl).

Heads: fused Linear(3*128 -> 5) on [q_EEG, q_EOG, q_EMG], plus the pretrained per-branch classifiers (auxiliary).

Occlusion (training, per sample): an occluded modality sends nothing (all its outgoing connections are zeroed and its
q is zeroed in the fused head, kept q rescaled as in inverted dropout); its branch still runs and its auxiliary head
still sees its own q.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import nn

from physioex.models.seqsleepnet_clean import ChannelNet, GradHook

MODALITIES = ("EEG", "EOG", "EMG")


def point_names(points: Iterable[str], n_lstm: int = 4, n_gru: int = 4) -> list:
    """Expand {'F','E','Q'} to the concrete connection sites: F1..F4, E, Q1..Q4."""
    sites = []
    if "F" in points:
        sites += [f"F{i + 1}" for i in range(n_lstm)]
    if "E" in points:
        sites += ["E"]
    if "Q" in points:
        sites += [f"Q{i + 1}" for i in range(n_gru)]
    return sites


class CrossConnection(nn.Module):
    """All ordered pairs (n -> m) between modalities at one site."""

    def __init__(self, modalities: Sequence[str], dim: int = 128, hook: str = "identity", lam: float = 1.0):
        super().__init__()
        self.mods = list(modalities)
        self.pairs = [(n, m) for m in self.mods for n in self.mods if n != m]
        self.W = nn.ModuleDict({f"{n}_{m}": nn.Linear(dim, dim) for n, m in self.pairs})
        self.g = nn.ParameterDict({f"{n}_{m}": nn.Parameter(torch.zeros(())) for n, m in self.pairs})
        for lin in self.W.values():
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.hook = GradHook(hook, lam)

    def forward(self, h: Dict[str, torch.Tensor], send: Optional[Dict[str, torch.Tensor]] = None,
                enabled: bool = True) -> Dict[str, torch.Tensor]:
        """h: {m: (N, ..., dim)}; send: {m: (N,) 0/1} per-sample sender mask (occlusion), broadcast over the rest."""
        if not enabled:
            return h
        out = dict(h)
        for n, m in self.pairs:
            src = self.hook(h[n])
            msg = self.g[f"{n}_{m}"] * self.W[f"{n}_{m}"](src)
            if send is not None:
                msg = msg * send[n].view(-1, *([1] * (msg.dim() - 1)))
            out[m] = out[m] + msg
        return out

    def gates(self) -> Dict[str, float]:
        return {k: float(v.detach()) for k, v in self.g.items()}


class MultiChannelNet(nn.Module):
    """forward(x: {m: (B, L, 3000)}, offset_db: {m: (B,)} or None, occlude: {m: (B,) 1 = occluded} or None)
    -> (fused logits (B, L, 5), aux logits {m: (B, L, 5)}, out dict)."""

    def __init__(self, branches: Dict[str, ChannelNet], points: Iterable[str] = (), hook: str = "identity",
                 n_classes: int = 5):
        super().__init__()
        self.mods = [m for m in MODALITIES if m in branches]
        self.branches = nn.ModuleDict({m: branches[m] for m in self.mods})
        b0 = self.branches[self.mods[0]]
        self.n_lstm, self.n_gru = b0.n_epoch_blocks, b0.n_seq_blocks
        dim = b0.dim
        self.sites = point_names(set(points), self.n_lstm, self.n_gru)
        self.conn = nn.ModuleDict({s: CrossConnection(self.mods, dim, hook) for s in self.sites})
        self.fused = nn.Linear(len(self.mods) * dim, n_classes)
        self.disabled: set = set()          # sites switched off (ablation / forgetting check)
        self.frozen_branches = False         # True (stages S1/S2): branches stay in eval() mode while training

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.frozen_branches:
            self.branches.eval()             # frozen features without dropout noise
        return self

    # --- helpers -------------------------------------------------------------------------------------------------
    def _connect(self, site: str, h: Dict[str, torch.Tensor], send) -> Dict[str, torch.Tensor]:
        if site not in self.conn:
            return h
        return self.conn[site](h, send, enabled=site not in self.disabled)

    def gates(self) -> Dict[str, Dict[str, float]]:
        return {s: c.gates() for s, c in self.conn.items()}

    def new_parameters(self):
        return list(self.conn.parameters()) + list(self.fused.parameters())

    def branch_parameters(self):
        return list(self.branches.parameters())

    # --- forward -------------------------------------------------------------------------------------------------
    def forward(self, x: Dict[str, torch.Tensor], offset_db: Optional[Dict[str, torch.Tensor]] = None,
                occlude: Optional[Dict[str, torch.Tensor]] = None):
        m0 = self.mods[0]
        B, L, _ = x[m0].shape
        out: Dict[str, torch.Tensor] = {}
        send_e = None if occlude is None else {m: 1.0 - occlude[m].float() for m in self.mods}       # (B,)
        send_f = None if send_e is None else {m: v.repeat_interleave(L) for m, v in send_e.items()}  # (B*L,)

        # input chain per branch (no connection here: its output is not normalised)
        h = {}
        for m in self.mods:
            off = None if offset_db is None or m not in offset_db else offset_db[m]
            h[m], out[f"{m}.F"] = self.branches[m].stem(x[m], off)

        # epoch encoder, block by block, with frame-level connections
        for i in range(self.n_lstm):
            for m in self.mods:
                h[m], raw = self.branches[m].epoch_block(i, h[m])
                if raw is not None:
                    out[f"{m}.L{i + 1}_raw"] = raw
            h = self._connect(f"F{i + 1}", h, send_f)
            for m in self.mods:
                out[f"{m}.L{i + 1}"] = h[m]
                h[m] = self.branches[m].drop_epoch(h[m])          # branch dropout (identity unless p > 0)

        # pooling, epoch embedding, epoch-level connection
        e = {}
        for m in self.mods:
            e[m], out[f"{m}.alpha"] = self.branches[m].pool(h[m], B, L)
        e = self._connect("E", e, send_e)
        for m in self.mods:
            out[f"{m}.e"] = e[m]
            e[m] = self.branches[m].drop_e(e[m])

        # sequence encoder, block by block, with epoch-level connections
        q = {m: self.branches[m].seq_start(e[m]) for m in self.mods}
        for i in range(self.n_gru):
            for m in self.mods:
                q[m], raw = self.branches[m].seq_block(i, q[m])
                if raw is not None:
                    out[f"{m}.G{i + 1}_raw"] = raw
            q = self._connect(f"Q{i + 1}", q, send_e)
            for m in self.mods:
                out[f"{m}.G{i + 1}"] = q[m]
                q[m] = self.branches[m].drop_seq(q[m])

        aux = {m: self.branches[m].head(q[m]) for m in self.mods}
        qs = [q[m] for m in self.mods]
        if send_e is not None:
            keep = torch.stack([send_e[m] for m in self.mods], -1)                      # (B, M)
            scale = len(self.mods) / keep.sum(-1, keepdim=True).clamp(min=1)            # inverted dropout
            qs = [qm * (keep[:, k:k + 1] * scale).view(B, 1, 1) for k, qm in enumerate(qs)]
        logits = self.fused(torch.cat(qs, -1))
        out["logits"] = logits
        return logits, aux, out


def occlusion_probs(kappa: Dict[str, float], scale: float = 0.5) -> Dict[str, float]:
    """Accuracy-weighted occlusion: the better a channel, the more often it is hidden. p_m = scale * k_m / max k."""
    mx = max(kappa.values())
    return {m: scale * k / mx for m, k in kappa.items()}


def sample_occlusion(B: int, probs: Dict[str, float], mods: Sequence[str], device) -> Dict[str, torch.Tensor]:
    """Independent Bernoulli per modality and sample; samples with every modality occluded are redrawn."""
    p = torch.tensor([probs[m] for m in mods], device=device)
    occ = torch.rand(B, len(mods), device=device) < p
    while True:
        allo = occ.all(-1)
        if not allo.any():
            break
        occ[allo] = torch.rand(int(allo.sum()), len(mods), device=device) < p
    return {m: occ[:, k] for k, m in enumerate(mods)}
