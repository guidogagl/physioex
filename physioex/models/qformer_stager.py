"""Q-Former stager on frozen MSAE / MSAE-VQ latents (downstream check; labels only here).

The encoder is frozen: :func:`extract_latents` stores ``N`` ``(128,)``, ``E`` ``(128,)`` and
``P`` ``(T_p, d_p)`` for every scored epoch of the main sleep period (MSAE-VQ: the quantized
latents of the eval prefix ``vq_levels``). :class:`QFormerStager` reads one epoch:

* tokens: ``[W_n N + type_N, W_e E + type_E, W_p P_j + type_P + pos_j]`` -> ``(2 + T_p, d)``;
* ``k`` learned queries, ``layers`` blocks of self-attention among the queries, cross-attention
  from the queries to the tokens and an FFN (pre-norm, as in the BLIP-2 Q-Former);
* attention pooling of the ``k`` outputs with one learned query, then a linear layer -> 5 stages.

:class:`LinearStager` (``[N; E]`` -> 5) is the reference at the same data. The checkpoint is
chosen on the mean per-night kappa of the validation nights; the test nights are read once.
:func:`attention_by_stage` gives, per stage, the share of the last cross-attention on N, E and
on every P token (which part of the epoch the stager reads).

``python -m physioex.models.qformer_stager extract --run RUN [--vq N8E8P6] --out DIR``
``python -m physioex.models.qformer_stager train --data DIR --arch qformer|linear --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SPLITS = ("train", "valid", "test")
STAGES = ("W", "N1", "N2", "N3", "REM")


# ----------------------------------------------------------------------------
# Latents of every scored epoch
# ----------------------------------------------------------------------------


@torch.no_grad()
def extract_latents(model, ds, subject_ids: Sequence[str], batch: int = 128) -> Dict[str, np.ndarray]:
    """Frozen latents of every scored epoch of the main sleep period whose window fits in the
    night (the first and last ``L // 2`` epochs of a night have no full context and are left out).
    Returns ``N``, ``E`` ``(M, dim)`` and ``P`` ``(M, T_p, d_p)`` in float16, ``y`` ``(M,)``,
    ``night`` (position in ``subject_ids``) and ``epoch`` (index in the night)."""
    device = next(model.parameters()).device
    model.eval()
    have = {s.subject_id: s for s in ds._subjects}
    half = ds.sequence_length // 2
    acc = {k: [] for k in ("N", "E", "P", "y", "night", "epoch")}
    for i, sid in enumerate(subject_ids):
        if sid not in have:
            continue
        n = ds._n_epochs[sid]
        lo, hi = ds._main_period.get(sid, (0, n)) if hasattr(ds, "_main_period") else (0, n)
        lo, hi = max(lo, half), min(hi, n - half)
        if hi <= lo:
            continue
        item = ds._build_item(have[sid], 0, n)
        x = next(iter(item["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        y = item["labels"].numpy().reshape(-1)
        ts = np.arange(lo, hi)
        ts = ts[y[ts] >= 0]
        for j in range(0, len(ts), batch):
            tb = ts[j:j + batch]
            w = np.stack([x[t - half:t + half + 1] for t in tb])
            z = model.encode(torch.from_numpy(w)[:, :, None].to(device))
            for k in ("N", "E", "P"):
                acc[k].append(z[k].float().cpu().numpy().astype(np.float16))
        acc["y"].append(y[ts].astype(np.int8))
        acc["night"].append(np.full(len(ts), i, np.int16))
        acc["epoch"].append(ts.astype(np.int32))
    return {k: np.concatenate(v) for k, v in acc.items()}


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.ln_q, self.ln_c, self.ln_kv, self.ln_f = (nn.LayerNorm(d) for _ in range(4))
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv):
        h = self.ln_q(q)
        q = q + self.drop(self.self_attn(h, h, h, need_weights=False)[0])
        kv = self.ln_kv(kv)
        a, w = self.cross_attn(self.ln_c(q), kv, kv, need_weights=True, average_attn_weights=True)
        q = q + self.drop(a)
        q = q + self.drop(self.ffn(self.ln_f(q)))
        return q, w  # w: (B, k, 2 + T_p)


class QFormerStager(nn.Module):
    """``k`` learned queries read the N, E and P tokens of one epoch; attention pooling -> stage."""

    def __init__(self, dim: int = 128, d_p: int = 16, n_tok: int = 60, d: int = 128, k: int = 8,
                 layers: int = 2, heads: int = 4, n_classes: int = 5, dropout: float = 0.1, **_):
        super().__init__()
        self.proj_n, self.proj_e, self.proj_p = nn.Linear(dim, d), nn.Linear(dim, d), nn.Linear(d_p, d)
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.blocks = nn.ModuleList([_Block(d, heads, dropout) for _ in range(layers)])
        self.pool_query = nn.Parameter(torch.randn(1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        self.last_attn = None

    def tokens(self, n, e, p):
        return torch.cat([(self.proj_n(n) + self.type_emb[0])[:, None], (self.proj_e(e) + self.type_emb[1])[:, None],
                          self.proj_p(p) + self.type_emb[2] + self.pos], 1)

    def embed(self, n, e, p):
        """Pooled vector of one epoch, ``(B, d)``; sets ``last_attn``."""
        kv = self.tokens(n, e, p)
        q = self.queries.expand(len(kv), -1, -1)
        for blk in self.blocks:
            q, w = blk(q, kv)
        self.last_attn = w.detach().mean(1)  # last cross-attention, mean over queries: (B, 2 + T_p)
        return self.pool(self.pool_query.expand(len(q), -1, -1), q, q, need_weights=False)[0][:, 0]

    def forward(self, n, e, p):
        return self.head(self.ln(self.embed(n, e, p)))


class LinearStager(nn.Module):
    """Reference: one linear layer on ``[N; E]`` (P unused), as the logistic probe."""

    def __init__(self, dim: int = 128, n_classes: int = 5, **_):
        super().__init__()
        self.head = nn.Linear(2 * dim, n_classes)
        self.last_attn = None

    def forward(self, n, e, p):
        return self.head(torch.cat([n, e], -1))


class _LinearEmbed(nn.Module):
    """Per-epoch embedding of ``[N; E]`` (P unused) for :class:`SeqStager`."""

    def __init__(self, dim: int = 128, d: int = 128, **_):
        super().__init__()
        self.proj = nn.Linear(2 * dim, d)
        self.last_attn = None

    def embed(self, n, e, p):
        return self.proj(torch.cat([n, e], -1))


class SeqStager(nn.Module):
    """Sequence of ``seq_len`` consecutive epochs, as ProtoSleepNet: a per-epoch embedding
    (``epoch="qformer"``: the Q-Former above without its head; ``"linear"``: ``[N; E]``),
    a temporal transformer over the sequence, a stage for every epoch. Input ``(B, L, ...)``,
    output ``(B, L, n_classes)``."""

    def __init__(self, epoch: str = "qformer", seq_len: int = 21, seq_layers: int = 2, d: int = 128,
                 heads: int = 4, n_classes: int = 5, dropout: float = 0.1, **kw):
        super().__init__()
        self.seq_len = seq_len
        enc = QFormerStager if epoch == "qformer" else _LinearEmbed
        self.enc = enc(d=d, heads=heads, n_classes=n_classes, dropout=dropout, **kw)
        if epoch == "qformer":  # head of the per-epoch model unused
            del self.enc.head, self.enc.ln
        self.pos_t = nn.Parameter(torch.randn(seq_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, seq_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        self.last_attn = None

    def forward(self, n, e, p):
        B, L = n.shape[:2]
        h = self.enc.embed(n.flatten(0, 1), e.flatten(0, 1), p.flatten(0, 1)).view(B, L, -1)
        self.last_attn = self.enc.last_attn  # (B * L, 2 + T_p) or None
        return self.head(self.ln(self.temporal(h + self.pos_t[:L])))


class MultiSeqStager(nn.Module):
    """:class:`SeqStager` over several modalities (EEG, EOG, EMG). Every modality has its own
    per-epoch embedding (``epoch="qformer"`` or ``"linear"``) plus a learned modality embedding;
    the modality embeddings of an epoch are concatenated and projected to ``d``, then the
    temporal transformer and the head as in :class:`SeqStager`.

    Inputs are the packed tensors of :func:`pack_modalities`: ``N``, ``E`` ``(B, L, sum dim_m)``
    and ``P`` ``(B, L, sum T_m, max d_p)``; ``spec`` gives ``(name, dim, d_p, n_tok)`` per modality."""

    def __init__(self, spec, epoch: str = "qformer", seq_len: int = 21, seq_layers: int = 2, d: int = 128,
                 heads: int = 4, n_classes: int = 5, dropout: float = 0.1, k: int = 8, layers: int = 2, **_):
        super().__init__()
        self.spec, self.seq_len = [tuple(s) for s in spec], seq_len
        encs = []
        for _, dim, d_p, n_tok in self.spec:
            if epoch == "qformer":
                enc = QFormerStager(dim=dim, d_p=d_p, n_tok=n_tok, d=d, k=k, layers=layers, heads=heads, dropout=dropout)
                del enc.head, enc.ln
            else:
                enc = _LinearEmbed(dim=dim, d=d)
            encs.append(enc)
        self.encs = nn.ModuleList(encs)
        self.mod_emb = nn.Parameter(torch.randn(len(self.spec), d) * 0.02)
        self.fuse = nn.Linear(len(self.spec) * d, d)
        self.pos_t = nn.Parameter(torch.randn(seq_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, seq_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        self.last_attn = None

    def forward(self, n, e, p):
        B, L = n.shape[:2]
        n, e, p = n.flatten(0, 1), e.flatten(0, 1), p.flatten(0, 1)
        hs, a, t = [], 0, 0
        for i, (enc, (_, dim, d_p, n_tok)) in enumerate(zip(self.encs, self.spec)):
            h = enc.embed(n[:, a:a + dim], e[:, a:a + dim], p[:, t:t + n_tok, :d_p])
            hs.append(h + self.mod_emb[i])
            a, t = a + dim, t + n_tok
        h = self.fuse(torch.cat(hs, -1)).view(B, L, -1)
        return self.head(self.ln(self.temporal(h + self.pos_t[:L])))


def pack_modalities(splits_by_mod: Dict[str, Dict[str, Dict[str, np.ndarray]]], drop: Sequence[str] = ()):
    """Standardize every modality on its own training statistics (:func:`standardize`), align
    them on ``(night, epoch)`` of the first modality (the reference: EEG) and pack them for
    :class:`MultiSeqStager`. An epoch missing in a modality (e.g. a night without EMG) gets zeros,
    i.e. the training mean. ``drop``: levels removed from the input, ``"eog:N"``, ``"emg:P"``...
    (set to zero in every split). Returns ``(data, spec, missing)``; ``missing[split][mod]`` counts
    the zero-filled epochs."""
    mods = list(splits_by_mod)
    std = {m: standardize(splits_by_mod[m]) for m in mods}
    ref = std[mods[0]]
    dpmax = max(std[m]["train"]["P"].shape[-1] for m in mods)
    spec = [(m, std[m]["train"]["N"].shape[-1], std[m]["train"]["P"].shape[-1], std[m]["train"]["P"].shape[1])
            for m in mods]
    gone = {tuple(x.split(":")) for x in drop}
    unknown = gone - {(m, lv) for m in mods for lv in ("N", "E", "P")}
    assert not unknown, f"unknown levels to drop: {sorted(unknown)}"
    data, missing = {}, {}
    for s in ref:
        key = ref[s]["night"] * 10 ** 6 + ref[s]["epoch"]
        M = len(key)
        cols = {"N": [], "E": [], "P": []}
        missing[s] = {}
        for m, dim, d_p, n_tok in spec:
            d = std[m][s]
            row = dict(zip((d["night"] * 10 ** 6 + d["epoch"]).tolist(), range(len(d["night"]))))
            idx = torch.tensor([row.get(k, -1) for k in key.tolist()])
            ok = idx >= 0
            missing[s][m] = int((~ok).sum())
            for lv in ("N", "E", "P"):
                src = d[lv]
                out = torch.zeros((M,) + tuple(src.shape[1:]))
                if (m, lv) not in gone:
                    out[ok] = src[idx[ok]]
                if lv == "P" and d_p < dpmax:
                    out = F.pad(out, (0, dpmax - d_p))
                cols[lv].append(out)
        data[s] = {"N": torch.cat(cols["N"], -1), "E": torch.cat(cols["E"], -1), "P": torch.cat(cols["P"], 1),
                   "y": ref[s]["y"], "night": ref[s]["night"], "epoch": ref[s]["epoch"]}
    return data, spec, missing


def _seq(epoch):
    return lambda **kw: SeqStager(epoch=epoch, **kw)


ARCHS = {"qformer": QFormerStager, "linear": LinearStager, "seqqformer": _seq("qformer"), "seqlinear": _seq("linear")}


# ----------------------------------------------------------------------------
# Data, training, metrics
# ----------------------------------------------------------------------------


def standardize(splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Per-dimension z-score with the training statistics (P: per channel over all tokens)."""
    tr = splits["train"]
    stats = {}
    for k in ("N", "E", "P"):
        a = tr[k].astype(np.float32).reshape(-1, tr[k].shape[-1])
        stats[k] = (a.mean(0), a.std(0) + 1e-6)
    out = {}
    for s, d in splits.items():
        out[s] = {k: torch.from_numpy((d[k].astype(np.float32) - stats[k][0]) / stats[k][1]) for k in ("N", "E", "P")}
        out[s]["y"] = torch.from_numpy(d["y"].astype(np.int64))
        out[s]["night"] = d["night"].astype(np.int64)
        if "epoch" in d:
            out[s]["epoch"] = d["epoch"].astype(np.int64)
    return out


def night_groups(d) -> List[np.ndarray]:
    """Sample indices of every night, in epoch order (the extracted epochs of a night; unscored
    epochs inside the main period are skipped, so a sequence is consecutive *scored* epochs)."""
    ep = d["epoch"] if "epoch" in d else np.arange(len(d["night"]))
    order = np.lexsort((ep, d["night"]))
    return np.split(order, np.flatnonzero(np.diff(d["night"][order])) + 1)


def train_windows(groups, L: int, rng: np.random.Generator) -> np.ndarray:
    """Non-overlapping sequences of ``L`` epochs with a random offset per night: ``(S, L)``."""
    out = []
    for g in groups:
        for s in range(int(rng.integers(0, L)), len(g) - L + 1, L):
            out.append(g[s:s + L])
    return np.stack(out)


def all_windows(groups, L: int) -> np.ndarray:
    """Every sequence of ``L`` epochs with stride 1, ``(S, L)``; every night needs >= ``L`` epochs."""
    assert all(len(g) >= L for g in groups), "a night is shorter than the sequence"
    return np.concatenate([np.lib.stride_tricks.sliding_window_view(g, L) for g in groups])


@torch.no_grad()
def predict(model, d, batch: int = 2048, device="cpu"):
    L = getattr(model, "seq_len", 1)
    if L > 1:
        return _predict_seq(model, d, L, batch, device)
    model.eval()
    logits, attn = [], []
    for i in range(0, len(d["y"]), batch):
        sl = slice(i, i + batch)
        logits.append(model(d["N"][sl].to(device), d["E"][sl].to(device), d["P"][sl].to(device)).float().cpu())
        if model.last_attn is not None:
            attn.append(model.last_attn.float().cpu())
    return torch.cat(logits), (torch.cat(attn) if attn else None)


@torch.no_grad()
def _predict_seq(model, d, L, batch, device):
    """Voting as ProtoSleepNet: every window of ``L`` epochs (stride 1), class probabilities
    averaged over the windows that contain an epoch. Returns log of the mean probabilities."""
    model.eval()
    W = torch.from_numpy(all_windows(night_groups(d), L))
    M = len(d["y"])
    prob, cnt, attn = torch.zeros(M, len(STAGES)), torch.zeros(M), None
    nb = max(1, batch // L)
    for i in range(0, len(W), nb):
        w = W[i:i + nb]
        out = model(d["N"][w].to(device), d["E"][w].to(device), d["P"][w].to(device)).float().softmax(-1).cpu()
        prob.index_add_(0, w.flatten(), out.flatten(0, 1))
        cnt.index_add_(0, w.flatten(), torch.ones(w.numel()))
        if model.last_attn is not None:
            a = model.last_attn.float().cpu()
            attn = torch.zeros(M, a.shape[-1]) if attn is None else attn
            attn.index_add_(0, w.flatten(), a)
    return (prob / cnt[:, None]).clamp_min(1e-8).log(), (attn / cnt[:, None] if attn is not None else None)


def metrics(pred: np.ndarray, y: np.ndarray, night: np.ndarray) -> dict:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

    per = [cohen_kappa_score(y[night == g], pred[night == g]) for g in np.unique(night)]
    return {"kappa_night_mean": float(np.mean(per)), "kappa_night_sd": float(np.std(per)),
            "kappa_pooled": float(cohen_kappa_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "kappa_per_night": [float(v) for v in per], "nights": [int(g) for g in np.unique(night)],
            "confusion": confusion_matrix(y, pred, labels=range(len(STAGES))).tolist()}


def attention_by_stage(attn: torch.Tensor, y: np.ndarray) -> dict:
    """Mean share of the last cross-attention on N, E, all of P and every P token, per true stage."""
    out = {}
    for c, name in enumerate(STAGES):
        a = attn[torch.from_numpy(y == c)]
        if len(a):
            m = a.mean(0).numpy()
            out[name] = {"N": float(m[0]), "E": float(m[1]), "P": float(m[2:].sum()), "P_tokens": m[2:].round(5).tolist()}
    return out


def train(model, data, epochs: int = 30, lr: float = 3e-4, batch: int = 512, weight_decay: float = 0.05,
          patience: int = 6, device="cpu", seed: int = 0, sched: str = "onecycle", evals_per_epoch: int = 1,
          log=print):
    """AdamW; best state on the mean per-night kappa of ``data["valid"]``, evaluated
    ``evals_per_epoch`` times per epoch. ``sched``: ``"onecycle"`` (peak at 5%) or ``"cosine"``
    (2% linear warm-up, then cosine to 0 over all ``epochs``). ``patience`` counts evaluations
    without improvement (0 = no early stopping)."""
    import math

    torch.manual_seed(seed)
    model.to(device)
    tr = data["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # steps per epoch (seq models: sequences of L epochs, ~len/L per epoch with random offsets)
    L0 = getattr(model, "seq_len", 1)
    per_epoch = (len(tr["y"]) + batch - 1) // batch if L0 == 1 else \
        (sum(len(g) // L0 for g in night_groups(tr)) + max(1, batch // L0) - 1) // max(1, batch // L0)
    steps = epochs * per_epoch
    if sched == "onecycle":
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    else:
        warm = max(1, int(0.02 * steps))
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm))))
    eval_at = {round(per_epoch * (q + 1) / evals_per_epoch) for q in range(evals_per_epoch)}
    best, best_state, wait, hist = -1.0, None, 0, []
    g = torch.Generator().manual_seed(seed)
    va = data["valid"]
    L = getattr(model, "seq_len", 1)
    groups, rng = (night_groups(tr), np.random.default_rng(seed)) if L > 1 else (None, None)
    for ep in range(epochs):
        model.train()
        # seq models: ``batch`` counts epochs, i.e. batch // L sequences per step
        units = torch.from_numpy(train_windows(groups, L, rng)) if L > 1 else torch.arange(len(tr["y"]))
        perm, tot, seen = units[torch.randperm(len(units), generator=g)], 0.0, 0
        nb = max(1, batch // L)
        for b, i in enumerate(range(0, len(perm), nb), 1):
            idx = perm[i:i + nb]
            logits = model(tr["N"][idx].to(device), tr["E"][idx].to(device), tr["P"][idx].to(device))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tr["y"][idx].reshape(-1).to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot, seen = tot + loss.item() * idx.numel(), seen + idx.numel()
            if b not in eval_at:
                continue
            k = metrics(predict(model, va, device=device)[0].argmax(1).numpy(), va["y"].numpy(), va["night"])["kappa_night_mean"]
            model.train()
            at = round(ep + b / per_epoch, 3)
            hist.append({"epoch": at, "train_loss": tot / seen, "valid_kappa": k, "lr": opt.param_groups[0]["lr"]})
            log(f"[qformer_stager] epoch {at}/{epochs} loss {tot / seen:.4f} valid kappa {k:.4f}")
            if k > best:
                best, wait = k, 0
                best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
            else:
                wait += 1
        if patience and wait >= patience:
            break
    model.load_state_dict(best_state)
    return hist


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _extract(a):
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run

    run, ds, model = _load_run(a.run, cache_dir=a.cache_dir, root=a.root)
    if a.vq:
        model.vq_levels = parse_vq_key(a.vq)
    elif hasattr(model, "latent_bits"):
        model.latent_bits = a.bits
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    meta = {"run": os.path.abspath(a.run), "vq": a.vq, "bits": None if a.vq else a.bits, "splits": {}}
    for s in SPLITS:
        d = extract_latents(model, ds, run["splits"][s], batch=a.batch)
        np.savez(os.path.join(a.out, f"latents_{s}.npz"), **d)
        meta["splits"][s] = {"nights": len(run["splits"][s]), "epochs": int(len(d["y"])),
                             "stage_counts": np.bincount(d["y"], minlength=len(STAGES)).tolist()}
        print(f"[qformer_stager] {s}: {len(d['y'])} epochs", flush=True)
    json.dump(meta, open(os.path.join(a.out, "latents.json"), "w"), indent=1)


def _load(d):
    return {s: dict(np.load(os.path.join(d, f"latents_{s}.npz"))) for s in SPLITS}


def _train(a):
    extra = {"seq_len": a.seq_len, "seq_layers": a.seq_layers} if a.arch.startswith("seq") else {}
    info = {}
    if a.mods:  # multimodal: one latents directory per modality, the first is the reference
        assert a.arch.startswith("seq"), "multimodal stager: seqqformer or seqlinear"
        dirs = dict(x.split("=", 1) for x in a.mods)
        data, spec, missing = pack_modalities({m: _load(d) for m, d in dirs.items()}, a.drop)
        info = {"spec": spec, "missing": missing, "drop": a.drop}
        print(f"[qformer_stager] modalities {spec}, zero-filled epochs {missing}", flush=True)
        torch.manual_seed(a.seed)  # the seed fixes the initial weights too
        model = MultiSeqStager(spec, epoch=a.arch[3:], d=a.d, k=a.k, layers=a.layers, heads=a.heads,
                               dropout=a.dropout, **extra)
    else:
        raw = _load(a.data)
        data = standardize(raw)
        tr = raw["train"]
        torch.manual_seed(a.seed)  # the seed fixes the initial weights too
        model = ARCHS[a.arch](dim=tr["N"].shape[-1], d_p=tr["P"].shape[-1], n_tok=tr["P"].shape[1], d=a.d, k=a.k,
                              layers=a.layers, heads=a.heads, dropout=a.dropout, **extra)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hist = train(model, data, epochs=a.epochs, lr=a.lr, batch=a.batch, patience=a.patience, device=device, seed=a.seed,
                 sched=a.sched, evals_per_epoch=a.evals_per_epoch)
    res = {"args": vars(a), "params": sum(p.numel() for p in model.parameters()), "history": hist, **info}
    for s in ("valid", "test"):  # test read once, with the checkpoint chosen on valid
        logits, attn = predict(model, data[s], device=device)
        y = data[s]["y"].numpy()
        res[s] = metrics(logits.argmax(1).numpy(), y, data[s]["night"])
        if attn is not None:
            res[s]["attention_by_stage"] = attention_by_stage(attn, y)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "stager.json"), "w"), indent=1)
    torch.save(model.state_dict(), os.path.join(a.out, "stager.pt"))
    print(json.dumps({s: {k: round(res[s][k], 4) for k in ("kappa_night_mean", "kappa_pooled", "macro_f1")}
                      for s in ("valid", "test")}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--run", required=True)
    e.add_argument("--vq", default=None, help="MSAE-VQ prefix, e.g. N8E8P6")
    e.add_argument("--bits", type=int, default=0, help="MSAE: bit depth of E and P (0 = continuous)")
    e.add_argument("--cache-dir", default=None)
    e.add_argument("--root", default=None)
    e.add_argument("--batch", type=int, default=128)
    e.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default=None, help="latents directory (one modality)")
    t.add_argument("--mods", nargs="*", default=None,
                   help="multimodal: name=latents_dir per modality, the first is the reference (eeg=... eog=... emg=...)")
    t.add_argument("--drop", nargs="*", default=(), help="multimodal ablation: levels set to zero, e.g. eog:N emg:P")
    t.add_argument("--arch", choices=sorted(ARCHS), default="qformer")
    t.add_argument("--d", type=int, default=128)
    t.add_argument("--k", type=int, default=8, help="learned queries")
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--seq-len", type=int, default=21, help="seq archs: epochs per sequence (ProtoSleepTransformer: 21)")
    t.add_argument("--seq-layers", type=int, default=2, help="seq archs: temporal transformer layers")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=6, help="evaluations without improvement (0 = off)")
    t.add_argument("--sched", choices=("onecycle", "cosine"), default="onecycle")
    t.add_argument("--evals-per-epoch", type=int, default=1)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "train" and not (a.data or a.mods):
        ap.error("train: --data or --mods")
    (_extract if a.cmd == "extract" else _train)(a)


if __name__ == "__main__":
    main()
