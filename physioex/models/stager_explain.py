"""Why the multimodal stager chose a stage: perturbations of its input latents.

The stager (:class:`physioex.models.qformer_stager.MultiSeqStager`) reads a window of ``L``
epochs of standardized latents per modality (packed as in :func:`pack_modalities`). Zero is the
training mean of every latent, so "removing" a latent means setting it to zero; this is also the
replacement used by the modality dropout (:meth:`MultiSeqStager.zero_embeddings`).

* :func:`token_occlusion`: change of the probability of a stage at the centre epoch when one
  level (N, E) or one 0.5 s P token of one modality is set to zero. Faithful by construction: it
  is the stager's own output under the intervention.
* :func:`modality_ablation`: the same for a whole modality, at the centre epoch or in the whole
  window.
* :func:`deletion_check`: the ``k`` most influential tokens removed together against ``k``
  random tokens (if occlusion explains, the first drop is larger).
* :func:`attention_at`: the last cross-attention of every modality's Q-Former at the centre
  epoch, reported next to the occlusion (attention is not an explanation by itself).
* :func:`set_epoch`: write new latents of one modality into one epoch of the window (for code
  counterfactuals: codes -> :meth:`MSAEVQ.embed` -> standardized latents -> stager).

All inputs are ``n``, ``e`` ``(1, L, sum dim_m)`` and ``p`` ``(1, L, sum T_m, max d_p)``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


def layout(spec) -> Dict[str, dict]:
    """Per modality: ``f`` = feature slice of N and E, ``t`` = token slice of P, ``d_p``."""
    out, a, t = {}, 0, 0
    for name, dim, d_p, n_tok in spec:
        out[name] = {"f": slice(a, a + dim), "t": slice(t, t + n_tok), "d_p": d_p}
        a, t = a + dim, t + n_tok
    return out


@torch.no_grad()
def probs(stager, n, e, p, centre: Optional[int] = None, batch: int = 256) -> torch.Tensor:
    """Stage probabilities at the centre epoch, ``(B, n_classes)``."""
    stager.eval()
    c = n.shape[1] // 2 if centre is None else centre
    out = []
    for i in range(0, len(n), batch):
        out.append(stager(n[i:i + batch], e[i:i + batch], p[i:i + batch])[:, c].float().softmax(-1))
    return torch.cat(out)


def _variants(n, e, p, spec, centre: int, mods: Sequence[str]):
    """Copies of the window, one per removed item: N, E, then every token, per modality."""
    lay = layout(spec)
    names, ns, es, ps = [], [], [], []
    for m in mods:
        L = lay[m]
        for lv in ("N", "E"):
            n2, e2 = n.clone(), e.clone()
            (n2 if lv == "N" else e2)[0, centre, L["f"]] = 0
            names.append((m, lv, -1)); ns.append(n2); es.append(e2); ps.append(p)
        for j in range(L["t"].stop - L["t"].start):
            p2 = p.clone()
            p2[0, centre, L["t"].start + j] = 0
            names.append((m, "P", j)); ns.append(n); es.append(e); ps.append(p2)
    return names, torch.cat(ns), torch.cat(es), torch.cat(ps)


@torch.no_grad()
def token_occlusion(stager, n, e, p, spec, target: Optional[int] = None, centre: Optional[int] = None,
                    mods: Optional[Sequence[str]] = None) -> dict:
    """Change of ``p(target)`` at the centre epoch (default target: the predicted stage) when N, E
    or one P token of a modality is set to zero (training mean). Returns ``{"base": probs,
    "target": int, mod: {"N": float, "E": float, "P": (T_m,) array}}`` (negative = the item
    supported the target)."""
    c = n.shape[1] // 2 if centre is None else centre
    base = probs(stager, n, e, p, c)[0]
    tgt = int(base.argmax()) if target is None else int(target)
    mods = list(mods or [s[0] for s in spec])
    names, N, E, P = _variants(n, e, p, spec, c, mods)
    pr = probs(stager, N, E, P, c)[:, tgt] - base[tgt]
    out = {"base": base.cpu().numpy(), "target": tgt}
    lay = layout(spec)
    for m in mods:
        out[m] = {"P": np.zeros(lay[m]["t"].stop - lay[m]["t"].start)}
    for (m, lv, j), d in zip(names, pr.cpu().numpy()):
        if lv == "P":
            out[m]["P"][j] = d
        else:
            out[m][lv] = float(d)
    return out


@torch.no_grad()
def modality_ablation(stager, n, e, p, spec, target: Optional[int] = None, centre: Optional[int] = None,
                      whole_window: bool = False) -> dict:
    """Stage probabilities with one modality removed (all its levels set to zero) at the centre
    epoch, or in every epoch of the window (``whole_window``). ``{"base", "target", mod: {"probs",
    "delta"}}`` with ``delta`` the change of ``p(target)``."""
    c = n.shape[1] // 2 if centre is None else centre
    base = probs(stager, n, e, p, c)[0]
    tgt = int(base.argmax()) if target is None else int(target)
    ep = slice(None) if whole_window else slice(c, c + 1)
    out = {"base": base.cpu().numpy(), "target": tgt}
    for m, L in layout(spec).items():
        n2, e2, p2 = n.clone(), e.clone(), p.clone()
        n2[0, ep, L["f"]] = 0
        e2[0, ep, L["f"]] = 0
        p2[0, ep, L["t"]] = 0
        pr = probs(stager, n2, e2, p2, c)[0]
        out[m] = {"probs": pr.cpu().numpy(), "delta": float(pr[tgt] - base[tgt])}
    return out


@torch.no_grad()
def deletion_check(stager, n, e, p, spec, occl: dict, k: int = 5, n_random: int = 20, seed: int = 0) -> dict:
    """Remove together the ``k`` P tokens (any modality) whose occlusion lowered ``p(target)`` most,
    and ``n_random`` draws of ``k`` random tokens. Returns both drops of ``p(target)``."""
    c = n.shape[1] // 2
    lay = layout(spec)
    flat = [(lay[m]["t"].start + j, occl[m]["P"][j]) for m in lay if m in occl for j in range(len(occl[m]["P"]))]
    idx = np.array([i for i, _ in flat])
    d = np.array([v for _, v in flat])
    top = idx[np.argsort(d)[:k]]
    rng = np.random.default_rng(seed)
    sets = [top] + [rng.choice(idx, k, replace=False) for _ in range(n_random)]
    P = p.repeat(len(sets), 1, 1, 1)
    for i, s in enumerate(sets):
        P[i, c, torch.as_tensor(s)] = 0
    pr = probs(stager, n.repeat(len(sets), 1, 1), e.repeat(len(sets), 1, 1), P, c)[:, occl["target"]]
    base = float(occl["base"][occl["target"]])
    return {"top_drop": float(base - pr[0]), "random_drop": float((base - pr[1:]).mean()),
            "random_drop_sd": float((base - pr[1:]).std())}


@torch.no_grad()
def attention_at(stager, n, e, p, spec, centre: Optional[int] = None) -> Dict[str, dict]:
    """Last cross-attention of every modality's Q-Former at the centre epoch (mean over queries):
    ``{mod: {"N", "E", "P": (T_m,)}}``; each modality's shares sum to 1."""
    c = n.shape[1] // 2 if centre is None else centre
    stager.eval()
    stager(n, e, p)
    a = stager.last_attn.view(n.shape[0], n.shape[1], -1)[0, c].float().cpu().numpy()
    out, o = {}, 0
    for m, _, _, n_tok in spec:
        out[m] = {"N": float(a[o]), "E": float(a[o + 1]), "P": a[o + 2:o + 2 + n_tok]}
        o += 2 + n_tok
    return out


def set_epoch(n, e, p, spec, mod: str, epoch: int, N=None, E=None, P=None):
    """Copy of the window with new standardized latents of ``mod`` at ``epoch`` (``N``/``E``:
    ``(dim,)``, ``P``: ``(T_m, d_p)``; ``None`` keeps the old ones)."""
    L = layout(spec)[mod]
    n2, e2, p2 = n.clone(), e.clone(), p.clone()
    if N is not None:
        n2[0, epoch, L["f"]] = torch.as_tensor(N, dtype=n.dtype)
    if E is not None:
        e2[0, epoch, L["f"]] = torch.as_tensor(E, dtype=e.dtype)
    if P is not None:
        p2[0, epoch, L["t"], :L["d_p"]] = torch.as_tensor(P, dtype=p.dtype)
    return n2, e2, p2


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation (no scipy needed)."""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 and ra.std() > 0 and rb.std() > 0 else float("nan")
