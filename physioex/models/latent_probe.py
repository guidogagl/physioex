"""Downstream checks of MSAE latents (gate G4). Labels serve only the evaluation.

1. **Staging probe.** Logistic regression (standardised features) trained on epochs of the
   training nights, Cohen's kappa per test night. Features: ``NE`` = [N; E], ``NEP`` = [N; E;
   mean and std of P over the tokens], and two references at no training cost: the epoch
   log-PSD (``logpsd``) and a PCA of the raw epoch with the same numbers per epoch as E + P
   (``pca``).
2. **Spindle proxy (label-free).** Per 1 s segment, sigma (11-16 Hz) power over 4-30 Hz power;
   a segment is a "sigma burst" above the 95th percentile of its night. Logistic regression
   per second from E alone and from [E; P tokens of that second]: test AUC. P should add.

``python -m physioex.models.latent_probe --run <dir>`` writes ``<run>/probe.json``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Sequence

import numpy as np
import torch

FS, EPOCH = 100, 3000


def night_samples(ds, subject_ids: Sequence[str], per_night: int, seed: int = 0):
    """Random windows inside the main sleep period, with the labels of the centre epochs.
    Returns ``(windows (M, L, 3000), labels (M,), night index (M,))``."""
    rng = np.random.default_rng(seed)
    have = {s.subject_id: s for s in ds._subjects}
    half = ds.sequence_length // 2
    W, Y, G = [], [], []
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
        ts = np.sort(rng.choice(np.arange(lo, hi), min(per_night, hi - lo), replace=False))
        W.append(np.stack([x[t - half:t + half + 1] for t in ts]))
        Y.append(y[ts]); G.append(np.full(len(ts), i))
    return np.concatenate(W), np.concatenate(Y), np.concatenate(G)


@torch.no_grad()
def latents(model, windows: np.ndarray, batch: int = 128) -> Dict[str, np.ndarray]:
    """``N``, ``E`` ``(M, dim)`` and ``P`` ``(M, T_p, d_p)`` with the latents rounded as in eval."""
    device = next(model.parameters()).device
    model.eval()
    acc = {"N": [], "E": [], "P": []}
    for i in range(0, len(windows), batch):
        z = model.encode(torch.from_numpy(windows[i:i + batch])[:, :, None].to(device))
        for k in acc:
            acc[k].append(z[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def sigma_bursts(centres: np.ndarray, night: np.ndarray, q: float = 95.0) -> np.ndarray:
    """``(M, 3000)`` -> ``(M, 30)`` bool: sigma / (4-30 Hz) power ratio per second above the
    ``q``-th percentile of its night (label-free spindle proxy)."""
    p = np.abs(np.fft.rfft(centres.reshape(len(centres), 30, FS), axis=-1)) ** 2
    ratio = p[..., 11:17].sum(-1) / (p[..., 4:31].sum(-1) + 1e-12)
    out = np.zeros_like(ratio, dtype=bool)
    for g in np.unique(night):
        m = night == g
        out[m] = ratio[m] > np.percentile(ratio[m], q)
    return out


def _fit_predict(xtr, ytr, xte, proba=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(xtr, ytr)
    return clf.predict_proba(xte)[:, 1] if proba else clf.predict(xte)


def staging_probe(feats_tr: Dict[str, np.ndarray], ytr, feats_te: Dict[str, np.ndarray], yte, gte) -> dict:
    from sklearn.metrics import cohen_kappa_score, f1_score

    out = {}
    for k in feats_tr:
        pred = _fit_predict(feats_tr[k], ytr, feats_te[k])
        per = [cohen_kappa_score(yte[gte == g], pred[gte == g]) for g in np.unique(gte)]
        out[k] = {"kappa_night_mean": float(np.mean(per)), "kappa_night_sd": float(np.std(per)),
                  "kappa_pooled": float(cohen_kappa_score(yte, pred)),
                  "macro_f1": float(f1_score(yte, pred, average="macro")), "kappa_per_night": [float(v) for v in per]}
    return out


def _per_second(z: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per 1 s segment: E repeated, and E + the P tokens covering that second."""
    m, t, d = z["P"].shape
    e = np.repeat(z["E"][:, None], 30, 1).reshape(m * 30, -1)
    p = z["P"].reshape(m, 30, (t // 30) * d).reshape(m * 30, -1)
    return {"E": e, "EP": np.concatenate([e, p], 1)}


def spindle_probe(z_tr, c_tr, g_tr, z_te, c_te, g_te, max_train: int = 200000, seed: int = 0) -> dict:
    from sklearn.metrics import roc_auc_score

    ytr, yte = sigma_bursts(c_tr, g_tr).reshape(-1), sigma_bursts(c_te, g_te).reshape(-1)
    ftr, fte = _per_second(z_tr), _per_second(z_te)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ytr), min(max_train, len(ytr)), replace=False)
    out = {"positive_rate": float(yte.mean())}
    for k in ftr:
        out[f"auc_{k}"] = float(roc_auc_score(yte, _fit_predict(ftr[k][idx], ytr[idx], fte[k], proba=True)))
    return out


def evaluate(model, ds, test_ids, train_ids, per_night: int = 250, max_train_nights: int = 40, seed: int = 0) -> dict:
    from physioex.models.msvq import epoch_logpsd

    wtr, ytr, gtr = night_samples(ds, list(train_ids)[:max_train_nights], per_night, seed + 1)
    wte, yte, gte = night_samples(ds, list(test_ids), per_night, seed)
    c = wtr.shape[1] // 2
    ztr, zte = latents(model, wtr), latents(model, wte)
    ctr, cte = wtr[:, c], wte[:, c]

    def feats(z, cen):
        p = z["P"]
        return {"NE": np.concatenate([z["N"], z["E"]], 1),
                "NEP": np.concatenate([z["N"], z["E"], p.mean(1), p.std(1)], 1),
                "logpsd": epoch_logpsd(torch.from_numpy(cen)).numpy()}

    ftr, fte = feats(ztr, ctr), feats(zte, cte)
    n_num = model.dim + model.n_tok * model.d_p
    mu = ctr.mean(0)
    v = np.linalg.svd(ctr - mu, full_matrices=False)[2][:min(n_num, len(ctr))]
    ftr["pca"], fte["pca"] = (ctr - mu) @ v.T, (cte - mu) @ v.T
    keep = ytr >= 0  # unscored epochs out
    keep_te = yte >= 0
    res = {"n_train_epochs": int(keep.sum()), "n_test_epochs": int(keep_te.sum()), "numbers_per_epoch": n_num,
           "staging": staging_probe({k: f[keep] for k, f in ftr.items()}, ytr[keep],
                                    {k: f[keep_te] for k, f in fte.items()}, yte[keep_te], gte[keep_te]),
           "spindle_proxy": spindle_probe(ztr, ctr, gtr, zte, cte, gte, seed=seed)}
    return res


def main():
    from physioex.models.recon_eval import _load_run

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--per-night", type=int, default=250)
    ap.add_argument("--weights", choices=("best", "final"), default="best")
    a = ap.parse_args()
    run, ds, model = _load_run(a.run, a.cache_dir, a.root, a.weights)
    if torch.cuda.is_available():
        model = model.cuda()
    res = evaluate(model, ds, run["splits"]["test"], run["splits"]["train"], per_night=a.per_night)
    with open(os.path.join(a.run, "probe.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: {m: round(v["kappa_night_mean"], 3) for m, v in res["staging"].items()} if k == "staging" else v
                      for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
