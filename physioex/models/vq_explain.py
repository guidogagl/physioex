"""Explanations that the discrete MSAE-VQ tokens allow and the continuous latents do not.
Sleep stages are used only to describe codes after the fact, never to build them.

1. **Code atlas** (:func:`atlas_e`, :func:`atlas_p`): the vocabulary can be listed. Per level-1
   E code: count, prototype decoded from the codeword with the typical context of its epochs,
   nearest real epochs, stage mix. Per level-1 P code: count, mean band power of the 0.5 s
   segments it codes against all segments, sigma-burst rate.
2. **Coarse to fine**: the R^2 gained by each RVQ level comes from the prefix grid of
   :mod:`physioex.models.recon_eval` (``vq_N{a}E{b}P{c}`` entries).
3. **A micro-event as a word** (:func:`burst_codes`, :func:`swap_token`): which P codes fire on
   label-free sigma bursts, and what the decoder does when one token's code is replaced.
4. **The night as text** (:func:`sequence_stats`): transitions, entropy, NMI with the hypnogram.
5. **Explanation for a downstream model** (:func:`bag_of_codes`): staging from code counts; the
   weight of each code is a global explanation, the active codes of an epoch a local one.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

FS = 100
STAGES = ("W", "N1", "N2", "N3", "REM")
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (11, 16), "beta": (16, 30)}
EOG_BANDS = ((0.3, 1), (1, 2), (2, 5), (5, 9), (9, 15), (15, 35))


@torch.no_grad()
def encode_all(model, windows: np.ndarray, batch: int = 64) -> Dict[str, np.ndarray]:
    """Codes (eval prefix) and quantized latents of every window: ``codes_{N,E,P}``, ``N``, ``E``, ``P``."""
    device = next(model.parameters()).device
    model.eval()
    acc: Dict[str, list] = {}
    for i in range(0, len(windows), batch):
        z = model.encode(torch.from_numpy(windows[i:i + batch])[:, :, None].to(device))
        for s in ("N", "E", "P"):
            acc.setdefault(s, []).append(z[s].float().cpu().numpy())
            if z["codes"][s] is not None:  # no N codes in a model without N
                acc.setdefault(f"codes_{s}", []).append(z["codes"][s].cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def model_bands(model) -> Dict[str, tuple]:
    """Bands of a modality model: the named EEG bands for an EEG model (100 Hz, wave decoder),
    else the model's ``pow_bands`` named ``"lo-hi"`` (EOG, EMG)."""
    if getattr(model, "fs", FS) == FS and getattr(model, "recon", "wave") != "texture" and \
            tuple(map(tuple, getattr(model, "pow_bands", ()))) != EOG_BANDS:
        return dict(BANDS)
    return {f"{a:g}-{b:g}": (a, b) for a, b in model.pow_bands}


def band_power(seg: np.ndarray, fs: int = FS, bands: Optional[Dict[str, tuple]] = None) -> Dict[str, np.ndarray]:
    """``(M, n)`` segments -> log10 power per band ``{band: (M,)}``."""
    p = np.abs(np.fft.rfft(seg, axis=-1)) ** 2
    f = np.fft.rfftfreq(seg.shape[-1], 1 / fs)
    return {b: np.log10(p[:, (f >= lo) & (f < hi)].sum(-1) + 1e-12) for b, (lo, hi) in (bands or BANDS).items()}


@torch.no_grad()
def decode_full(model, n, e=None, p=None, seed: int = 0) -> torch.Tensor:
    """Decoded epoch with the noise texture when the model has one (``(B, epoch)``). The texture
    noise is drawn with ``seed``, so two decodes with the same seed differ only by the latents."""
    torch.manual_seed(seed)
    out = model.decode(n, e, p, texture=bool(getattr(model, "texture", False)))
    return out[0] + out[1] if isinstance(out, tuple) else out


@torch.no_grad()
def atlas_e(model, z: Dict[str, np.ndarray], centres: np.ndarray, labels: Optional[np.ndarray] = None,
            top: int = 20, min_count: int = 5, fs: int = FS, bands: Optional[Dict[str, tuple]] = None) -> Dict[int, dict]:
    """Level-1 E codes: count, decoded prototype ``D(mean N_q of its epochs, codeword)``, the
    ``top`` epochs nearest to the codeword, mean band power and stage mix (if ``labels``)."""
    device = next(model.parameters()).device
    cb = model.rvq["E"].levels[0].codebook
    c1 = z["codes_E"][:, 0]
    out = {}
    for c in np.unique(c1):
        m = np.where(c1 == c)[0]
        if len(m) < min_count:
            continue
        n = torch.from_numpy(z["N"][m].mean(0, keepdims=True)).to(device)
        proto = decode_full(model, n, cb[int(c)][None])
        d = ((z["E"][m] - cb[int(c)].cpu().numpy()) ** 2).sum(1)
        bp = band_power(centres[m], fs, bands)
        out[int(c)] = {"count": int(len(m)), "prototype": proto[0].float().cpu().numpy(),
                       "nearest": m[np.argsort(d)[:top]].tolist(),
                       "band_power": {b: float(v.mean()) for b, v in bp.items()}}
        if labels is not None:
            lab = labels[m]
            out[int(c)]["stages"] = {s: float((lab == i).mean()) for i, s in enumerate(STAGES)}
    return out


def token_segments(centres: np.ndarray, n_tok: int) -> np.ndarray:
    """``(M, 3000)`` -> ``(M, n_tok, 3000 / n_tok)`` the segment each P token codes."""
    return centres.reshape(len(centres), n_tok, -1)


def atlas_p(z: Dict[str, np.ndarray], centres: np.ndarray, bursts: Optional[np.ndarray] = None,
            min_count: int = 20, fs: int = FS, bands: Optional[Dict[str, tuple]] = None) -> Dict[int, dict]:
    """Level-1 P codes: count, band power of their segments minus the mean of all segments,
    and (with ``bursts`` ``(M, n_tok)`` bool) the fraction of their segments inside an event
    (sigma burst for EEG, rapid eye movement for EOG, burst for EMG)."""
    c1 = z["codes_P"][..., 0]
    seg = token_segments(centres, c1.shape[1]).reshape(-1, centres.shape[1] // c1.shape[1])
    bp = band_power(seg, fs, bands)
    mean = {b: v.mean() for b, v in bp.items()}
    flat = c1.reshape(-1)
    bflat = None if bursts is None else bursts.reshape(-1)
    out = {}
    for c in np.unique(flat):
        m = flat == c
        if m.sum() < min_count:
            continue
        out[int(c)] = {"count": int(m.sum()), "band_power_vs_mean": {b: float(v[m].mean() - mean[b]) for b, v in bp.items()}}
        if bflat is not None:
            out[int(c)]["burst_rate"] = float(bflat[m].mean())
    return out


def bursts_per_token(burst_seconds: np.ndarray, n_tok: int) -> np.ndarray:
    """``(M, 30)`` bursts per second -> ``(M, n_tok)`` per token (a token inherits its second)."""
    rep = n_tok // burst_seconds.shape[1]
    return np.repeat(burst_seconds, rep, axis=1) if rep >= 1 else burst_seconds[:, ::burst_seconds.shape[1] // n_tok]


def burst_codes(codes_p1: np.ndarray, bursts: np.ndarray, top: int = 5) -> dict:
    """Precision and recall of each level-1 P code (and of the ``top`` codes together) as a
    detector of the sigma bursts. ``codes_p1`` and ``bursts``: ``(M, n_tok)``."""
    c, b = codes_p1.reshape(-1), bursts.reshape(-1).astype(bool)
    per = []
    for k in np.unique(c):
        m = c == k
        per.append((int(k), float(b[m].mean()), float((m & b).sum() / max(1, b.sum())), int(m.sum())))
    per.sort(key=lambda t: -t[1] * t[2])
    best = [p[0] for p in per[:top]]
    m = np.isin(c, best)
    return {"base_rate": float(b.mean()), "codes": [{"code": k, "precision": p, "recall": r, "count": n} for k, p, r, n in per[:50]],
            f"top{top}": {"codes": best, "precision": float(b[m].mean()) if m.any() else 0.0,
                          "recall": float((m & b).sum() / max(1, b.sum()))}}


@torch.no_grad()
def swap_token(model, codes: Dict[str, torch.Tensor], token: int, new_code, level: int = 0, band: str = "sigma",
               fs: int = FS, bands: Optional[Dict[str, tuple]] = None, seed: int = 0) -> dict:
    """Intervention: replace the level-``level`` P code of ``token`` with ``new_code`` (an int), or
    the whole code stack of the token with ``new_code`` (a sequence of codes, one per level), and
    decode (same texture noise before and after). Returns both waves and the change of the
    ``band`` power inside the token's segment and outside it."""
    model.eval()
    z0 = model.embed(codes)
    c2 = {k: (v.clone() if v is not None else None) for k, v in codes.items()}
    if np.ndim(new_code):
        nc = torch.as_tensor(np.asarray(new_code), device=c2["P"].device, dtype=c2["P"].dtype)
        c2["P"][:, token, :len(nc)] = nc
    else:
        c2["P"][:, token, level] = new_code
    z1 = model.embed(c2)
    w0 = decode_full(model, z0["N"], z0["E"], z0["P"], seed).float().cpu().numpy()
    w1 = decode_full(model, z1["N"], z1["E"], z1["P"], seed).float().cpu().numpy()
    T = codes["P"].shape[1]
    n = w0.shape[1] // T
    band = band if band in (bands or BANDS) else next(iter(bands or BANDS))  # e.g. EOG/EMG: no sigma
    s0 = band_power(token_segments(w0, T).reshape(-1, n), fs, bands)[band].reshape(len(w0), -1)
    s1 = band_power(token_segments(w1, T).reshape(-1, n), fs, bands)[band].reshape(len(w0), -1)
    d = s1 - s0
    others = np.delete(d, token, axis=1)
    return {"wave_before": w0, "wave_after": w1, "latents_after": z1, "band_change_token": d[:, token],
            "band_change_others": np.abs(others).mean(1),
            # old names (EEG sigma)
            "sigma_change_token": d[:, token], "sigma_change_others": np.abs(others).mean(1)}


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def sequence_stats(seq: np.ndarray, labels: Optional[np.ndarray] = None) -> dict:
    """One night of codes ``(T,)`` (e.g. level-1 E or N): code entropy, transition matrix of the
    used codes, probability of staying, and NMI with the hypnogram (``labels``, -1 = unscored)."""
    codes, inv = np.unique(seq, return_inverse=True)
    k = len(codes)
    tr = np.zeros((k, k))
    np.add.at(tr, (inv[:-1], inv[1:]), 1)
    out = {"codes": codes.tolist(), "entropy_bits": _entropy(np.bincount(inv) / len(inv)),
           "stay": float(np.trace(tr) / max(1, tr.sum())),
           "transitions": (tr / np.maximum(tr.sum(1, keepdims=True), 1)).tolist()}
    if labels is not None:
        from sklearn.metrics import normalized_mutual_info_score

        m = labels >= 0
        out["nmi_stages"] = float(normalized_mutual_info_score(labels[m], seq[m]))
    return out


def bag_of_codes(z: Dict[str, np.ndarray], n_codes: Dict[str, int], sparse: bool = False):
    """Per epoch: one-hot level-1 N and E codes and the histogram of level-1 P codes over the tokens.
    Columns: ``N:0..K_N-1, E:0..K-1, P:0..K-1`` (see :func:`bag_names`); a model without N has an
    all-zero N block. ``sparse=True``: a ``scipy.sparse.csr_matrix`` (large codebooks)."""
    m = len(z["codes_E"])
    if sparse:
        from scipy import sparse as sp

        kn, ke, kp = n_codes["N"], n_codes["E"], n_codes["P"]
        rows, cols, vals = [], [], []
        if z.get("codes_N") is not None:
            rows.append(np.arange(m)); cols.append(z["codes_N"][:, 0]); vals.append(np.ones(m, np.float32))
        rows.append(np.arange(m)); cols.append(kn + z["codes_E"][:, 0]); vals.append(np.ones(m, np.float32))
        c = z["codes_P"][..., 0]
        rows.append(np.repeat(np.arange(m), c.shape[1])); cols.append(kn + ke + c.reshape(-1))
        vals.append(np.full(c.size, 1.0 / c.shape[1], np.float32))
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(m, kn + ke + kp))
    oh_n = np.zeros((m, n_codes["N"]), np.float32)
    if z.get("codes_N") is not None:
        oh_n[np.arange(m), z["codes_N"][:, 0]] = 1
    oh_e = np.zeros((m, n_codes["E"]), np.float32)
    oh_e[np.arange(m), z["codes_E"][:, 0]] = 1
    hist_p = np.zeros((m, n_codes["P"]), np.float32)
    c = z["codes_P"][..., 0]
    np.add.at(hist_p, (np.repeat(np.arange(m), c.shape[1]), c.reshape(-1)), 1.0 / c.shape[1])
    return np.concatenate([oh_n, oh_e, hist_p], 1)


def bag_names(n_codes: Dict[str, int]) -> list:
    return [f"{s}:{i}" for s in ("N", "E", "P") for i in range(n_codes[s])]


def code_evidence(xtr, ytr: np.ndarray, names: Sequence[str], top: int = 10) -> dict:
    """Logistic regression on the bag of codes; per stage the ``top`` codes with the largest weight
    (global explanation). Returns the fitted model too."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=3000, C=0.5).fit(xtr, ytr)
    ev = {}
    for i, cls in enumerate(clf.classes_):
        w = clf.coef_[i]
        idx = np.argsort(-w)[:top]
        ev[STAGES[int(cls)] if 0 <= int(cls) < len(STAGES) else str(cls)] = [(names[j], float(w[j])) for j in idx]
    return {"model": clf, "evidence": ev}
