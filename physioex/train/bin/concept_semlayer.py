"""Road A as a baseline (critical evaluation, 09.11.2026): a small semantic layer supervised by the AASM rules.
A two-layer encoder maps every 0.5 s token to ``K`` sigmoid units (the semantic layer) from the spectral twin
(band powers and 1 Hz bins at the token and its +-1 context; ``--input spec``) or from the twin plus the codec
latent ``P`` (``--input spec+P``). The units are trained to predict the token masks of the rule detectors
(K-complex, spindle, slow waves, alpha, vertex, sawtooth, CAP A1 / A3, LAMF: public rules, no expert label),
with a sparsity penalty, a decorrelation penalty between units and a site adversary (gradient reversal).
Sites of the pool are split in three folds (train on six, test on three); DREAMS is always held out and
carries the expert references. Reported per fold and reference: AP of the best single unit (a readable unit),
AP of a logistic probe on the units and on the spectral twin (the same information, undigested), the site
read from the units and from the twin (balanced accuracy on held-out nights of the training sites), unit
sparsity and correlation. This is the number road B must beat: a semantic layer that the spectrum alone gives.
``python -m physioex.train.bin.concept_semlayer --dump DIR --out JSON [--input spec|spec+P --units 32]``
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from physioex.train.bin import concept_rationale as RA

RULES = ["k_complex_rule", "spindle_rule", "delta", "slow_wave", "alpha", "vertex", "sawtooth", "cap_a1", "cap_a3", "lamf"]
EXPERT = {"k_complex_vis1": "dreams", "spindle_vis1": "dreams"}
TWIN = ["spec_tok", "spec_fine", "spec_ctx1", "spec_fine_ctx1"]
LATENT = ["P", "P_ctx±1"]


def site_folds(sites, n_folds=3):
    sites = sorted(sites)
    return [sites[i::n_folds] for i in range(min(n_folds, len(sites)))]


class _Rev:
    """Gradient reversal."""

    @staticmethod
    def apply(x):
        import torch

        class F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a):
                return a.view_as(a)

            @staticmethod
            def backward(ctx, g):
                return -g

        return F.apply(x)


def train_layer(X, Y, H, site, units=32, hidden=128, epochs=3, batch=1024, lr=1e-3, l_sp=0.02, l_dec=0.1,
                l_adv=0.1, seed=0, device="cpu"):
    """``X`` (n, d) standardized inputs, ``Y`` (n, r) rule targets, ``H`` (n, r) availability, ``site`` (n,)
    integer codes. Returns the torch encoder (eval mode) and the training log."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n_site = int(site.max()) + 1
    enc = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Linear(hidden, units), nn.Sigmoid()).to(device)
    head = nn.Linear(units, Y.shape[1]).to(device)
    adv = nn.Linear(units, n_site).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()) + list(adv.parameters()), lr=lr)
    pos = np.clip(Y[H].reshape(-1).mean() if H.any() else 0.1, 1e-3, 0.5)
    pw = torch.full((Y.shape[1],), float((1 - pos) / pos), device=device)
    Xt, Yt, Ht = (torch.from_numpy(np.asarray(a, np.float32)) for a in (X, Y, H))
    St = torch.from_numpy(np.asarray(site, np.int64))
    log = []
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        tot = np.zeros(4)
        nb = 0
        for i in range(0, len(perm), batch):
            ix = perm[i:i + batch]
            x, y, h, s = Xt[ix].to(device), Yt[ix].to(device), Ht[ix].to(device), St[ix].to(device)
            u = enc(x)
            bce = nn.functional.binary_cross_entropy_with_logits(head(u), y, pos_weight=pw, reduction="none")
            l_rule = (bce * h).sum() / h.sum().clamp_min(1)
            uc = u - u.mean(0, keepdim=True)
            c = (uc.T @ uc) / (len(u) - 1)
            d = torch.sqrt(torch.diag(c)).clamp_min(1e-6)
            corr = c / (d[:, None] * d[None])
            l_dc = (corr - torch.eye(units, device=device)).pow(2).sum() / (units * (units - 1))
            l_s = u.mean()
            l_a = nn.functional.cross_entropy(adv(_Rev.apply(u)), s) if n_site > 1 else torch.zeros((), device=device)
            loss = l_rule + l_sp * l_s + l_dec * l_dc + l_adv * l_a
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += np.array([l_rule.item(), l_s.item(), l_dc.item(), l_a.item()])
            nb += 1
        log.append({"epoch": ep, "rule": tot[0] / nb, "sparsity": tot[1] / nb, "decorr": tot[2] / nb, "adv": tot[3] / nb})
    return enc.eval(), log


def units_of(enc, X, batch=65536, device="cpu"):
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(enc(torch.from_numpy(np.asarray(X[i:i + batch], np.float32)).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def best_unit_ap(U, r):
    aps = [max(RA.ap(U[:, k], r), RA.ap(-U[:, k], r)) for k in range(U.shape[1])]
    k = int(np.argmax(aps))
    return float(aps[k]), k


def _site_acc(Z, gname, night, rng, max_fit=60_000):
    """Balanced accuracy of a logistic site reader: half of the nights of every site train, the rest test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    tr_n, te_n = set(), set()
    for g in np.unique(gname):  # half of the nights of every site train, the rest test; single nights train
        ng = rng.permutation(np.unique(night[gname == g])).tolist()
        k = (len(ng) + 1) // 2
        tr_n.update(ng[:k])
        te_n.update(ng[k:])
    tr = np.flatnonzero([n in tr_n for n in night])
    te = np.flatnonzero([n in te_n for n in night])
    if len(np.unique(gname[tr])) < 2 or not len(te):
        return None
    sub = rng.choice(tr, min(max_fit, len(tr)), replace=False)
    mu, sd = Z[sub].mean(0), Z[sub].std(0) + 1e-6
    m = LogisticRegression(max_iter=300, C=0.5).fit((Z[sub] - mu) / sd, gname[sub])
    return float(balanced_accuracy_score(gname[te], m.predict((Z[te] - mu) / sd)))


def run(D, input_mode="spec", units=32, epochs=3, n_train=400_000, max_fit=30_000, seed=0, n_folds=3, hidden=128):
    rng = np.random.default_rng(seed)
    n = len(D["y"])
    names = TWIN + (LATENT if input_mode == "spec+P" else [])
    X, _, bad = RA.build(D, names, np.arange(n))
    Xtwin, _, _ = RA.build(D, TWIN, np.arange(n))
    rules = [r for r in RULES if r in D["refs"]]
    Y = np.stack([D["refs"][r][0] for r in rules], 1).astype(np.float32)
    H = np.stack([D["refs"][r][1] for r in rules], 1)
    fin = np.isfinite(X).all(1) & np.isfinite(Xtwin).all(1)
    pool = (D["set"] == "pool") & fin
    dreams = (D["set"] == "dreams") & fin
    sites = sorted(set(D["gname"][pool].tolist()))
    folds = site_folds(sites, n_folds)
    refs_eval = {r: "pool" for r in rules}
    refs_eval.update({r: s for r, s in EXPERT.items() if r in D["refs"]})
    res = {"input": input_mode, "dim": int(X.shape[1]), "bad": bad, "rules": rules, "folds": folds, "per_fold": []}
    for fi, held in enumerate(folds):
        t0 = time.time()
        tr_mask = pool & ~np.isin(D["gname"], held)
        te_mask = pool & np.isin(D["gname"], held)
        tr = np.flatnonzero(tr_mask)
        tr = rng.choice(tr, min(n_train, len(tr)), replace=False)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = (X - mu) / sd
        site_codes = {s: i for i, s in enumerate(sorted(set(D["gname"][tr].tolist())))}
        enc, log = train_layer(Xs[tr], Y[tr], H[tr], np.array([site_codes[g] for g in D["gname"][tr]]), units=units,
                               hidden=hidden, epochs=epochs, seed=seed + fi)
        U = units_of(enc, Xs)
        fold = {"held_out": held, "n_train": int(len(tr)), "log": log, "refs": {}, "train_sec": round(time.time() - t0, 1)}
        act = U[te_mask].mean(0) if te_mask.any() else U[tr].mean(0)
        c = np.corrcoef(U[tr][:20000].T)
        fold["units"] = {"mean_act": float(act.mean()), "n_alive": int(((act > 0.02) & (act < 0.98)).sum()),
                         "mean_abs_corr": float(np.abs(c - np.eye(units)).sum() / (units * (units - 1)))}
        for ref, where in refs_eval.items():
            r, h = D["refs"][ref]
            te = np.flatnonzero((te_mask if where == "pool" else dreams) & h)
            if len(te) < 2 * RA.T or not r[te].any() or r[te].all():
                continue
            row = {"set": where, "n": int(len(te)), "pos": int(r[te].sum())}
            row["best_unit_ap"], row["best_unit"] = best_unit_ap(U[te], r[te])
            trn = np.flatnonzero(tr_mask & h)
            trn = rng.choice(trn, min(max_fit, len(trn)), replace=False) if len(trn) else trn
            for name, Z in (("probe_units_ap", U), ("probe_twin_ap", Xtwin)):
                if len(trn) and r[trn].any() and not r[trn].all():
                    idx = np.r_[trn, te]
                    train = np.r_[np.ones(len(trn), bool), np.zeros(len(te), bool)]
                    s = RA.fold_probe(Z[idx], [], r[idx], None, None, max_fit=max_fit, seed=seed, train=train)
                    row[name] = RA.ap(s[~train], r[te])
            if "probe_units_ap" in row and "probe_twin_ap" in row:
                row["units_minus_twin"] = row["probe_units_ap"] - row["probe_twin_ap"]
            fold["refs"][ref] = row
        trn_rows = np.flatnonzero(tr_mask)
        sub = rng.choice(trn_rows, min(200_000, len(trn_rows)), replace=False)
        fold["site_from_units"] = _site_acc(U[sub], D["gname"][sub], D["night"][sub], np.random.default_rng(seed))
        fold["site_from_twin"] = _site_acc(Xtwin[sub], D["gname"][sub], D["night"][sub], np.random.default_rng(seed))
        fold["site_chance"] = 1 / len(site_codes)
        res["per_fold"].append(fold)
        print(f"[semlayer] fold {fi} held {held}: {fold['units']} site units {fold['site_from_units']} twin {fold['site_from_twin']} "
              + " ".join(f"{k}:{v.get('best_unit_ap', 0):.2f}/{v.get('probe_units_ap', 0):.2f}/{v.get('probe_twin_ap', 0):.2f}" for k, v in fold["refs"].items())
              + f" ({time.time() - t0:.0f} s)", flush=True)
    # mean over folds
    agg = {}
    for ref in refs_eval:
        rows = [f["refs"][ref] for f in res["per_fold"] if ref in f["refs"]]
        if rows:
            agg[ref] = {k: float(np.mean([r[k] for r in rows if k in r])) for k in ("best_unit_ap", "probe_units_ap", "probe_twin_ap", "units_minus_twin")
                        if any(k in r for r in rows)}
            agg[ref]["n_folds"] = len(rows)
    res["mean"] = agg
    res["site"] = {k: float(np.mean([f[k] for f in res["per_fold"] if f.get(k) is not None])) for k in ("site_from_units", "site_from_twin", "site_chance")}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input", default="spec", choices=["spec", "spec+P"])
    ap.add_argument("--units", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=400_000)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--n-ss01", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    RA.SPEC_FINE = True
    D, keys, sets = RA.load(a.dump, "eeg", RULES + list(EXPERT), frac=a.frac, n_ss01=a.n_ss01, seed=a.seed)
    print(f"[semlayer] loaded {len(keys)} nights, {len(D['y'])} tokens, refs {sorted(D['refs'])}, {time.time() - t0:.0f} s", flush=True)
    res = run(D, a.input, units=a.units, epochs=a.epochs, n_train=a.n_train, seed=a.seed)
    res["nights"] = {s: sum(v == s for v in sets.values()) for s in sorted(set(sets.values()))}
    res["minutes"] = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"[semlayer] wrote {a.out}: {res['mean']} site {res['site']} ({res['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
