"""Iteration S4b of the explainability workflow: an event-level gate and a quadratic lexicon on P.

After S4a (``WORKFLOW_SPIEGABILITA.md``) the events are in the codes but entangled: an MLP on
``pq`` x3 reads the spindles (AP 0.22-0.25) where every linear reader stays at 0.05, and the
decoder puts the expert spindles back. The S4 panel found three facts this script turns into
measurements, without any retraining and without the GPU (it reads the arrays saved by
``vq_event_readout`` / ``vq_event_lexicon`` / ``vq_event_codes`` for each run):

1. **Ceilings of the gate.** The token gate (purity >= 0.5, coverage >= 0.5) is out of reach by
   construction: an oracle firing on the tokens entirely inside the expert-1 events has purity 1
   and coverage ~0.43 (K-complexes) / ~0.20 (spindles). The same oracle, the existing detectors
   and every new lexicon are scored at the EVENT level (one-to-one matching, +-1 token tolerance)
   against expert 1, expert 2, their union and intersection, and the DREAMS automatic detector,
   overall and in N2 only.
2. **A quadratic label-free lexicon.** A spindle is an envelope, a power: in every directional
   lexicon its words are both tails of the same direction, so a signed reading cancels it. The
   bases (ICA / PCA of ``pq`` and of ``pq`` x3) are fitted on DCSM, never on DREAMS; the words are
   label-free too: signed tails, folded tails |z|, and energies (sum of z^2 over direction groups
   chosen on DCSM by kurtosis and by co-activation of z^2, raw and smoothed over +-1 token), plus
   conjunctions P-word AND E-word (k-means on DCSM ``zE``). The DREAMS labels only select and score
   the words, leave-one-excerpt-out: a lexicon is at most three words chosen on the other excerpts.
3. **Nulls.** A random rotation of the whitened latent in place of ICA (10 seeds) and a circular
   shift of the expert events inside each excerpt (20 shifts), through the same selection; paired
   per-excerpt differences between the runs (d_p 16 / 32 / 64).

``python -m physioex.train.bin.vq_event_lexicon2 --runs RUN16 RUN32 RUN64 --arrays DIR16 DIR32 DIR64 --out DIR``

``--arrays DIR`` is the output root of the battery of one run, with ``readout/dreams_readout.npz``,
``lexicon/dcsm_readout.npz`` and ``codes/scores_*.npz``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from typing import Dict, List

import numpy as np

from physioex.train.bin.domain_shift_report import _jsonable
from physioex.train.bin.vq_event_codes import KINDS, TOK_S, evaluate_scores, expert_refs, token_labels
from physioex.train.bin.vq_event_lexicon import NBIN, bin_edges, fit_basis, nb_scores_loo, words_of
from physioex.train.bin.vq_event_readout import ctx3

N2 = 2
EPOCH_S = 30.0
TAILS_LO, TAILS_HI = (0.025, 0.05, 0.1), (0.9, 0.95, 0.975)
FOLD_Q = (0.8, 0.9, 0.95, 0.975)
MAX_WORDS = 3
COVERAGE_MIN = {"k_complex": 0.5, "spindle": 0.4}
DCSM_SUB = 4  # DCSM tokens kept for the stage mix of a word (every 4th)


# ---------------------------------------------------------------------------- events
def mask_to_events(mask: np.ndarray, tol: float) -> List[tuple]:
    """Runs of positive tokens (gaps of one token merged) -> events (s), widened by ``tol`` s."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    starts = np.r_[idx[0], idx[1:][np.diff(idx) > 2]]
    ends = np.r_[idx[:-1][np.diff(idx) > 2], idx[-1]]
    return [(s * TOK_S - tol, (e + 1) * TOK_S + tol) for s, e in zip(starts, ends)]


def match_count(pred: List[tuple], ref: List[tuple]) -> int:
    """One-to-one greedy matching by overlap (same rule as ``vq_event_codes.match_one_to_one``),
    vectorised: the masks of the lexicon search produce thousands of candidates."""
    if not pred or not ref:
        return 0
    P, R = np.asarray(pred), np.asarray(ref)
    ov = np.minimum(P[:, None, 1], R[None, :, 1]) - np.maximum(P[:, None, 0], R[None, :, 0])
    i, j = np.nonzero(ov > 0)
    if not len(i):
        return 0
    order = np.argsort(-ov[i, j], kind="stable")
    used_p, used_r, tp = set(), set(), 0
    for k in order:
        if i[k] in used_p or j[k] in used_r:
            continue
        used_p.add(i[k]); used_r.add(j[k]); tp += 1
    return tp


def counts(mask: np.ndarray, ref: List[tuple], tol: float, keep=None) -> np.ndarray:
    """(tp, n_pred, n_ref) of one excerpt; ``keep(t_mid)`` restricts both sides (N2 stratum)."""
    pe = mask_to_events(mask, tol)
    if keep is not None:
        pe = [e for e in pe if keep(0.5 * (e[0] + e[1]))]
        ref = [e for e in ref if keep(0.5 * (e[0] + e[1]))]
    return np.array([match_count(pe, ref), len(pe), len(ref)], np.int64)


def prf(c: np.ndarray) -> dict:
    tp, n_p, n_r = (int(v) for v in c)
    p = tp / n_p if n_p else 0.0
    r = tp / n_r if n_r else 0.0
    return {"purity": p, "coverage": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "n_pred": n_p, "n_ref": n_r}


def n2_keeper(stages: np.ndarray):
    return lambda t: 0 <= int(t // EPOCH_S) < len(stages) and stages[int(t // EPOCH_S)] == N2


def merge_refs(a: List[tuple], b: List[tuple], mode: str) -> List[tuple]:
    """Union (overlapping events merged) or intersection (events of ``a`` overlapping one of ``b``)."""
    if mode == "inter":
        return [e for e in a if any(min(e[1], f[1]) > max(e[0], f[0]) for f in b)]
    ev = sorted(a + b)
    out = []
    for s, e in ev:
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def reference_sets(R: List[dict], kind: str) -> Dict[str, Dict[str, list]]:
    v1, v2, au = expert_refs(R, kind, "vis1"), expert_refs(R, kind, "vis2"), expert_refs(R, kind, "auto")
    both = [r["id"] for r in R if v2[r["id"]]]
    out = {"vis1": v1, "vis1_on_vis2_excerpts": {i: v1[i] for i in both}, "vis2": {i: v2[i] for i in both},
           "union": {i: merge_refs(v1[i], v2[i], "union") for i in both},
           "inter": {i: merge_refs(v1[i], v2[i], "inter") for i in both}}
    if any(au.values()):
        out["auto"] = au
    return out


def interior_tokens(ref: List[tuple], n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in ref:
        j0, j1 = int(np.ceil(a / TOK_S)), int(np.floor(b / TOK_S))
        m[max(0, j0):min(n, j1)] = True
    return m


# ---------------------------------------------------------------------------- words
def basis_words(Zd: np.ndarray, Zr: Dict[str, np.ndarray], tok_per_rec: Dict[str, int], name: str) -> tuple:
    """Label-free word masks on DREAMS from a basis fitted on DCSM. Returns (names, masks per
    excerpt (W, n_tok) bool, DCSM masks (W, n_dcsm) bool for the stage mix)."""
    names, rmask, dmask = [], {r: [] for r in Zr}, []

    def add(label, dvals, rvals, thr, side):
        names.append(label)
        dv = dvals[::DCSM_SUB]
        dmask.append(dv >= thr if side > 0 else dv <= thr)
        for r in Zr:
            rmask[r].append(rvals[r] >= thr if side > 0 else rvals[r] <= thr)

    D = Zd.shape[1]
    for d in range(D):
        for q in TAILS_LO:
            add(f"{name}:d{d}<=q{q}", Zd[:, d], {r: Zr[r][:, d] for r in Zr}, np.quantile(Zd[:, d], q), -1)
        for q in TAILS_HI:
            add(f"{name}:d{d}>=q{q}", Zd[:, d], {r: Zr[r][:, d] for r in Zr}, np.quantile(Zd[:, d], q), +1)
        ad = np.abs(Zd[:, d])
        for q in FOLD_Q:
            add(f"{name}:|d{d}|>=q{q}", ad, {r: np.abs(Zr[r][:, d]) for r in Zr}, np.quantile(ad, q), +1)
    for gname, g in energy_groups(Zd).items():
        for smooth in (False, True):
            ed = energy(Zd, g, smooth, None)
            er = {r: energy(Zr[r], g, smooth, tok_per_rec[r]) for r in Zr}
            for q in FOLD_Q:
                add(f"{name}:E[{gname}]{'~' if smooth else ''}>=q{q}", ed, er, np.quantile(ed, q), +1)
    return names, {r: np.stack(v) for r, v in rmask.items()}, np.stack(dmask)


def energy_groups(Zd: np.ndarray, n_kurt: int = 6, n_pairs: int = 10) -> Dict[str, list]:
    """Direction groups chosen on DCSM only: the most kurtotic directions (singletons and the
    top-4 group), the pairs with the most correlated z^2 (co-active energy), and all directions."""
    from scipy.stats import kurtosis

    k = kurtosis(Zd, axis=0)
    top = list(np.argsort(-k)[:n_kurt])
    groups = {f"d{d}": [int(d)] for d in top}
    groups["kurt4"] = [int(d) for d in top[:4]]
    sq = Zd ** 2
    C = np.corrcoef(sq[:: max(1, len(sq) // 50000)].T)
    iu = np.triu_indices(len(C), 1)
    for a in np.argsort(-C[iu])[:n_pairs]:
        i, j = int(iu[0][a]), int(iu[1][a])
        groups[f"d{i}+d{j}"] = [i, j]
    groups["all"] = list(range(Zd.shape[1]))
    return groups


def energy(Z: np.ndarray, g: list, smooth: bool, n_tok) -> np.ndarray:
    e = (Z[:, g] ** 2).sum(1)
    if not smooth:
        return e
    # +-1 token mean; DCSM rows are 60-token epochs, DREAMS rows are one excerpt in time order
    T = 60 if n_tok is None else len(e)
    e2 = e.reshape(-1, T)
    pad = np.concatenate([e2[:, :1], e2, e2[:, -1:]], 1)
    return ((pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3).reshape(-1)


# ---------------------------------------------------------------------------- selection
def word_counts(masks: Dict[str, np.ndarray], refs: Dict[str, list], tol: float) -> Dict[str, np.ndarray]:
    """(W, 3) event counts of every word in every excerpt."""
    return {r: np.stack([counts(m, refs[r], tol) for m in masks[r]]) for r in refs}


def select_lexicon(masks, refs, wc, train: List[str], cov_min: float, tol: float, n_cand: int = 30) -> List[int]:
    """Greedy lexicon of <= MAX_WORDS words on the training excerpts: start from the purest word
    reaching ``cov_min`` (else the best-F1 word), add the word that most raises the F1 of the union."""
    c = sum(wc[r] for r in train)
    tp, n_p, n_r = c[:, 0], np.maximum(c[:, 1], 1), np.maximum(c[:, 2], 1)
    pur, cov = tp / n_p, tp / n_r
    f1 = np.where(pur + cov > 0, 2 * pur * cov / np.maximum(pur + cov, 1e-9), 0)
    ok = np.flatnonzero(cov >= cov_min)
    first = int(ok[np.argmax(pur[ok])]) if len(ok) else int(np.argmax(f1))
    chosen = [first]
    cand = [int(w) for w in np.argsort(-f1)[:n_cand] if w != first]

    def union_f1(ws):
        cc = sum(counts(np.any(masks[r][ws], 0), refs[r], tol) for r in train)
        return prf(cc)["f1"]

    best = union_f1(chosen)
    while len(chosen) < MAX_WORDS and cand:
        scores = [(union_f1(chosen + [w]), w) for w in cand]
        f, w = max(scores)
        if f <= best + 1e-4:
            break
        chosen.append(w); cand.remove(w); best = f
    return chosen


def loo_lexicon(masks, refs, cov_min, tol, keep=None, names=None) -> dict:
    """Leave-one-excerpt-out: words chosen on the others, scored on the held-out excerpt; also the
    best single word (purest at ``cov_min`` on the others)."""
    ids = list(refs)
    wc = word_counts(masks, refs, tol)
    lex, single, per_ex, picks = np.zeros(3, np.int64), np.zeros(3, np.int64), {}, []
    for te in ids:
        tr = [r for r in ids if r != te]
        ws = select_lexicon(masks, refs, wc, tr, cov_min, tol)
        picks.append(ws)
        kp = keep[te] if keep else None
        c_lex = counts(np.any(masks[te][ws], 0), refs[te], tol, kp)
        c_one = counts(masks[te][ws[0]], refs[te], tol, kp)
        lex += c_lex; single += c_one
        per_ex[te] = prf(c_lex)
    freq = {}
    for ws in picks:
        for w in ws:
            freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:6]
    return {"lexicon": prf(lex), "single_word": prf(single), "per_excerpt": per_ex,
            "words_picked": [{"word": names[w] if names else int(w), "folds": n} for w, n in top],
            "_top_index": [w for w, _ in top]}


def event_pr_scores(scores: Dict[str, np.ndarray], refs: Dict[str, list], tol: float, keep=None,
                    recall_target: float = 0.5) -> dict:
    """Event purity at event coverage >= ``recall_target`` for a continuous token score: the
    threshold of each excerpt is the highest one reaching the target on the other excerpts."""
    ids = [r for r in refs if r in scores]
    if not ids:
        return None
    qs = np.unique(np.quantile(np.concatenate([scores[r] for r in ids]), np.linspace(0.5, 0.999, 80)))
    C = {r: np.stack([counts(scores[r] >= q, refs[r], tol) for q in qs]) for r in ids}
    tot, best_f1 = np.zeros(3, np.int64), np.zeros(3, np.int64)
    for te in ids:
        c = sum(C[r] for r in ids if r != te)
        rec = c[:, 0] / np.maximum(c[:, 2], 1)
        pur = c[:, 0] / np.maximum(c[:, 1], 1)
        ok = np.flatnonzero(rec >= recall_target)
        qi = int(ok[np.argmax(pur[ok])]) if len(ok) else 0
        f1 = 2 * pur * rec / np.maximum(pur + rec, 1e-9)
        tot += counts(scores[te] >= qs[qi], refs[te], tol, keep[te] if keep else None)
        best_f1 += counts(scores[te] >= qs[int(np.argmax(f1))], refs[te], tol, keep[te] if keep else None)
    return {"at_coverage_target": prf(tot), "best_f1_threshold": prf(best_f1)}


def stage_mix(dmask: np.ndarray, dstage_tok: np.ndarray) -> list:
    if not dmask.any():
        return [None] * 5
    s = dstage_tok[dmask]
    return [float((s == i).mean()) for i in range(5)]


# ---------------------------------------------------------------------------- one run
def load_arrays(root: str, recs_meta: List[dict]) -> tuple:
    rd = np.load(os.path.join(root, "readout", "dreams_readout.npz"))
    dc = np.load(os.path.join(root, "lexicon", "dcsm_readout.npz"))
    rep = {m["id"]: {k: rd[f"{m['id']}_{k}"] for k in ("pq", "zE")} for m in recs_meta}
    dcsm = {"pq": dc["pq"], "zE": dc["zE"], "stage": dc["stage"]}
    sc = {}
    for kind in KINDS.values():
        f = os.path.join(root, "codes", f"scores_{kind}.npz")
        if os.path.exists(f):
            z = np.load(f)
            sc[kind] = {}
            for k in z.files:
                det, rid = k.split("__", 1)
                if det != "label":
                    sc[kind].setdefault(det, {})[rid] = z[k]
    return rep, dcsm, sc


def fit_bases(dcsm: dict, seed: int, fit_rows: int = 120000) -> dict:
    """Label-free bases on DCSM: ICA on pq (full rank), ICA on pq x3 (S2 setting, 16 comps, for the
    consistency check) and ICA on pq x3 at a larger rank."""
    rng = np.random.default_rng(seed)
    X1 = dcsm["pq"].reshape(-1, dcsm["pq"].shape[-1]).astype(np.float32)
    X3 = ctx3(dcsm["pq"])
    sub = rng.choice(len(X1), min(fit_rows, len(X1)), replace=False)
    d = X1.shape[1]
    out = {"pq1_ica": ("pq1", fit_basis(X1[sub], "ica", d, seed)),
           "pq3_ica16": ("pq3", fit_basis(X3[sub], "ica", 16, seed)),
           "pq3_ica": ("pq3", fit_basis(X3[sub], "ica", min(3 * d, 48), seed))}
    return out, {"pq1": X1, "pq3": X3}


class RandomRotation:
    """Null basis: PCA whitening followed by a random orthogonal rotation (same rank)."""

    def __init__(self, X: np.ndarray, n_comp: int, seed: int):
        from scipy.stats import ortho_group
        from sklearn.decomposition import PCA

        self.pca = PCA(n_comp, whiten=True, random_state=seed).fit(X)
        self.Q = ortho_group.rvs(n_comp, random_state=seed)

    def transform(self, X):
        return self.pca.transform(X) @ self.Q


def rep_of(name: str, pq: np.ndarray) -> np.ndarray:
    return pq.reshape(-1, pq.shape[-1]).astype(np.float32) if name == "pq1" else ctx3(pq)


def run_battery(tag: str, root: str, meta: List[dict], tol: float, n_null_rot: int, n_null_shift: int, log) -> dict:
    T = 60
    rep, dcsm, saved_scores = load_arrays(root, meta)
    dstage_tok = np.repeat(dcsm["stage"], T)[::DCSM_SUB]
    bases, Xd = fit_bases(dcsm, 0)
    # E words: k-means on DCSM zE (label-free), one word per epoch cluster
    from sklearn.cluster import KMeans

    kmE = KMeans(12, n_init=4, random_state=0).fit(dcsm["zE"].astype(np.float32))
    out = {"tag": tag, "root": root, "d_p": int(dcsm["pq"].shape[-1]), "tol_s": tol, "events": {}}
    for pref, kind in KINDS.items():
        R = [m for m in meta if m["id"].startswith(pref)]
        ids = [m["id"] for m in R]
        ntok = {m["id"]: len(m["y"]) * T for m in R}
        refsets = reference_sets(R, kind)
        keep = {m["id"]: n2_keeper(m["y"]) for m in R}
        y = {m["id"]: token_labels(m["events"], kind, "vis1", len(m["y"]), T) for m in R}
        res = {"n_events": {k: int(sum(len(v) for v in d.values())) for k, d in refsets.items()},
               "excerpts": {k: list(d) for k, d in refsets.items()}}
        # 1. oracle and existing detectors
        orc = {r: interior_tokens(refsets["vis1"][r], ntok[r]) for r in ids}
        ylab = np.concatenate([y[r] for r in ids]); om = np.concatenate([orc[r] for r in ids])
        res["oracle_vis1_interior"] = {
            "token_purity": float(ylab[om].mean()) if om.any() else None, "token_coverage": float(om[ylab].mean()),
            "event": prf(sum(counts(orc[r], refsets["vis1"][r], tol) for r in ids)),
            "event_n2": prf(sum(counts(orc[r], refsets["vis1"][r], tol, keep[r]) for r in ids))}
        res["saved_detectors"] = {}
        for det, sc in saved_scores.get(kind, {}).items():
            res["saved_detectors"][det] = {ref: event_pr_scores(sc, refsets[ref], tol) for ref in ("vis1", "vis2", "union") if refsets[ref]}
        log(tag, kind, "oracle", res["oracle_vis1_interior"]["token_coverage"], res["oracle_vis1_interior"]["event"])
        # 2. words of every basis
        all_names, all_masks, all_dmasks = [], {r: [] for r in ids}, []
        res["nb_lexicon"] = {}
        for bname, (rname, B) in bases.items():
            Zd = B.transform(Xd[rname])
            Zr = {r: B.transform(rep_of(rname, rep[r]["pq"])) for r in ids}
            names, masks, dmask = basis_words(Zd, Zr, ntok, bname)
            all_names += names; all_dmasks.append(dmask)
            for r in ids:
                all_masks[r].append(masks[r])
            # naive-Bayes token scores: signed octiles (= S2), folded octiles
            edges = bin_edges(Zd)
            Ws = {r: words_of(Zr[r], edges) for r in ids}
            fedges = bin_edges(np.abs(Zd))
            Wf = {r: words_of(np.abs(Zr[r]), fedges) for r in ids}
            for lname, W in (("signed", Ws), ("folded", Wf), ("signed+folded", {r: np.concatenate([Ws[r], Wf[r]], 1) for r in ids})):
                s = nb_scores_loo(W, y)
                ev = evaluate_scores(s, y, refsets["vis1"], n_boot=300)
                ev["event_vis1"] = event_pr_scores(s, refsets["vis1"], tol)
                ev["event_union"] = event_pr_scores(s, refsets["union"], tol) if refsets["union"] else None
                ev["per_excerpt_ap"] = per_excerpt_ap(s, y)
                res["nb_lexicon"][f"{bname}_{lname}"] = ev
                log(tag, kind, "NB", bname, lname, round(ev["token_ap"], 3), ev["token_ap_ci"])
        masks = {r: np.concatenate(all_masks[r]) for r in ids}
        dmask = np.concatenate(all_dmasks)
        # conjunctions P-word AND E-word on the words most often picked against expert 1
        cov_min = COVERAGE_MIN[kind]
        res["lexicon"] = {}
        for ref in refsets:
            if ref == "vis1_on_vis2_excerpts" or not refsets[ref]:
                continue
            rr = refsets[ref]
            mk = {r: masks[r] for r in rr}
            L = loo_lexicon(mk, rr, cov_min, tol, None, all_names)
            L["n2_only"] = loo_lexicon(mk, rr, cov_min, tol, {r: keep[r] for r in rr}, all_names)["lexicon"]
            L["words_picked"] = [{**w, "dcsm_firing": float(dmask[i].mean()), "dcsm_stage_mix": stage_mix(dmask[i], dstage_tok)}
                                 for w, i in zip(L["words_picked"], L.pop("_top_index"))]
            res["lexicon"][ref] = L
            log(tag, kind, "lexicon", ref, {k: round(v, 3) for k, v in L["lexicon"].items() if k in ("purity", "coverage", "f1")},
                L["words_picked"][:2])
        # conjunctions with E (expert 1 and union)
        top_words = [all_names.index(w["word"]) for w in res["lexicon"]["vis1"]["words_picked"][:5]]
        Ew = {r: np.repeat(kmE.predict(rep[r]["zE"].astype(np.float32)), T) for r in ids}
        dE = np.repeat(kmE.labels_, T)[::DCSM_SUB]
        cj_names, cj_masks, cj_d = [], {r: [] for r in ids}, []
        for w in top_words:
            for c in range(12):
                cj_names.append(f"{all_names[w]} & E{c}"); cj_d.append(dmask[w] & (dE == c))
                for r in ids:
                    cj_masks[r].append(masks[r][w] & (Ew[r] == c))
        cjm = {r: np.concatenate([masks[r], np.stack(cj_masks[r])]) for r in ids}
        cj_all = all_names + cj_names
        cj_dm = np.concatenate([dmask, np.stack(cj_d)])
        res["lexicon_with_E"] = {}
        for ref in ("vis1", "union"):
            if not refsets.get(ref):
                continue
            rr = refsets[ref]
            L = loo_lexicon({r: cjm[r] for r in rr}, rr, cov_min, tol, None, cj_all)
            L["words_picked"] = [{**w, "dcsm_firing": float(cj_dm[i].mean()), "dcsm_stage_mix": stage_mix(cj_dm[i], dstage_tok)}
                                 for w, i in zip(L["words_picked"], L.pop("_top_index"))]
            res["lexicon_with_E"][ref] = L
            log(tag, kind, "lexicon+E", ref, {k: round(v, 3) for k, v in L["lexicon"].items() if k in ("purity", "coverage", "f1")})
        # 3. nulls on the primary reference
        prim = "vis1" if kind == "k_complex" else "union"
        rr = refsets[prim]
        res["null"] = {"reference": prim, "random_rotation": [], "event_shift": []}
        for s in range(n_null_rot):
            Bn = RandomRotation(Xd["pq3"][:: 3], min(3 * out["d_p"], 48), seed=100 + s)
            nm, nmask, _ = basis_words(Bn.transform(Xd["pq3"]), {r: Bn.transform(ctx3(rep[r]["pq"])) for r in rr}, ntok, "rot")
            res["null"]["random_rotation"].append(loo_lexicon(nmask, rr, cov_min, tol)["lexicon"])
        rng = np.random.default_rng(7)
        mk = {r: masks[r] for r in rr}
        for s in range(n_null_shift):
            shifted = {}
            for r in rr:
                dur = ntok[r] * TOK_S
                off = rng.uniform(0.2, 0.8) * dur
                shifted[r] = sorted(((a + off) % dur, (a + off) % dur + (b - a)) for a, b in rr[r])
            res["null"]["event_shift"].append(loo_lexicon(mk, shifted, cov_min, tol)["lexicon"])
        for k in ("random_rotation", "event_shift"):
            v = [x["f1"] for x in res["null"][k]]
            p = [x["purity"] for x in res["null"][k]]
            if v:
                res["null"][f"{k}_summary"] = {"f1_median": float(np.median(v)), "f1_q95": float(np.quantile(v, 0.95)),
                                               "purity_median": float(np.median(p)), "purity_q95": float(np.quantile(p, 0.95))}
        log(tag, kind, "null", {k: v for k, v in res["null"].items() if k.endswith("summary")})
        out["events"][kind] = res
    return out


def per_excerpt_ap(scores: Dict[str, np.ndarray], y: Dict[str, np.ndarray]) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score

    return {r: float(average_precision_score(y[r], scores[r])) if y[r].any() else None for r in scores}


def paired(reps: List[dict], n_boot: int = 2000) -> dict:
    """Per-excerpt differences between runs: NB lexicon AP (vis1) and lexicon F1 (primary reference)."""
    out = {}
    rng = np.random.default_rng(0)
    for kind in KINDS.values():
        prim = "vis1" if kind == "k_complex" else "union"
        out[kind] = {}
        for a, b in itertools.combinations(range(len(reps)), 2):
            ra, rb = reps[a]["events"][kind], reps[b]["events"][kind]
            key = f"{reps[b]['tag']}-{reps[a]['tag']}"
            d = {}
            for metric, fa, fb in (
                    ("nb_ap_pq3_ica_signed+folded", ra["nb_lexicon"]["pq3_ica_signed+folded"]["per_excerpt_ap"],
                     rb["nb_lexicon"]["pq3_ica_signed+folded"]["per_excerpt_ap"]),
                    ("lexicon_f1", {k: v["f1"] for k, v in ra["lexicon"][prim]["per_excerpt"].items()},
                     {k: v["f1"] for k, v in rb["lexicon"][prim]["per_excerpt"].items()})):
                ks = [k for k in fa if fa[k] is not None and fb.get(k) is not None]
                diff = np.array([fb[k] - fa[k] for k in ks])
                boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
                d[metric] = {"mean": float(diff.mean()), "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "n": len(ks)}
            out[kind][key] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, help="VQ runs (the first one gives the dataset recipe)")
    ap.add_argument("--arrays", nargs="+", required=True, help="battery output root of each run")
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--target", default="dreams")
    ap.add_argument("--tol", type=float, default=TOK_S, help="event tolerance (s) on each side of a predicted event")
    ap.add_argument("--null-rot", type=int, default=10)
    ap.add_argument("--null-shift", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from physioex.data.datasets import get_dataset
    from physioex.models.recon_eval import _load_run, channel_pipelines

    os.makedirs(a.out, exist_ok=True)
    log = lambda *s: print("[vq_event_lexicon2]", *s, flush=True)  # noqa: E731
    run, _, _ = _load_run(a.runs[0])
    ra = run["args"]
    tds = get_dataset(a.target)(channels=[ra.get("channel", "EEG_clean")], pipelines=channel_pipelines(ra),
                                sequence_length=1, cache_dir=ra.get("cache_dir"))
    meta = []
    for spec in tds._subjects:
        sid = spec.subject_id
        n = tds._n_epochs[sid]
        meta.append({"id": sid, "y": np.asarray(tds._get_labels(spec))[:n], "events": tds._load_or_compute_events(spec)})
    tags = a.tags or [os.path.basename(p.rstrip("/")) for p in a.arrays]
    t0 = time.time()
    reps = []
    for tag, root in zip(tags, a.arrays):
        reps.append(run_battery(tag, root, meta, a.tol, a.null_rot, a.null_shift, log))
        json.dump(_jsonable(reps[-1]), open(os.path.join(a.out, f"lexicon2_{tag}.json"), "w"), indent=1)
        log(tag, "done", f"{time.time() - t0:.0f}s")
    json.dump(_jsonable({"runs": a.runs, "tags": tags, "paired": paired(reps)}), open(os.path.join(a.out, "lexicon2_paired.json"), "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
