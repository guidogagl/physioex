"""L3 of the explainability workflow: does the stager use the event words of the S4b lexicon?

For the words that passed the event-level gate in :mod:`vq_event_lexicon2` (K-complex vs
expert 1, spindles vs the union of the experts), every DREAMS centre epoch with full N context in
which the word fires is edited in the latent ("project and requantize"): the word's coordinates
in its DCSM basis are brought back to the DCSM median (energy words: the group is rescaled to the
median energy; signed / folded tails: the coordinate is set to the DCSM median), the edited P
tokens are requantized with the model's RVQ (same prefix), and the EEG sequence stager
(``qformer_stager``) reads the window again. Reported: the change of p(N2) of the centre epoch
against two controls with the same edit size, (a) the same edit at random tokens of the epoch
where the word is silent and (b) a random direction group of the same size at the word's tokens;
and, in signal space, the sigma-envelope RMS (spindles) or slow-wave peak-to-peak (K-complexes)
of the decoded epoch inside the edited tokens, before and after the edit.

``python -m physioex.train.bin.vq_event_fidelity --run VQ_RUN --stager DIR --latents DIR --arrays DIR --lexicon2 JSON --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np

from physioex.train.bin.domain_shift_report import _jsonable
from physioex.train.bin.vq_event_codes import KINDS, TOK_S, reflect_index
from physioex.train.bin.vq_event_lexicon2 import basis_words, fit_bases, reference_sets
from physioex.train.bin.vq_event_readout import ctx3
from physioex.train.bin.vq_explain_n2 import _bp

N2 = 2
WORD_RE = re.compile(r"^(?P<basis>[^:]+):(?:(?P<sd>d\d+)(?P<op><=|>=)q(?P<q>[\d.]+)|\|(?P<fd>d\d+)\|>=q(?P<fq>[\d.]+)"
                     r"|E\[(?P<grp>[^\]]+)\](?P<sm>~?)>=q(?P<eq>[\d.]+))$")


def parse_word(name: str, groups: dict) -> dict:
    m = WORD_RE.match(name)
    if m is None:
        raise ValueError(name)
    if m["sd"]:
        return {"basis": m["basis"], "kind": "signed", "dirs": [int(m["sd"][1:])]}
    if m["fd"]:
        return {"basis": m["basis"], "kind": "folded", "dirs": [int(m["fd"][1:])]}
    return {"basis": m["basis"], "kind": "energy", "dirs": groups[m["grp"]]}


def edit_epoch(P: np.ndarray, prev_last, next_first, toks, B, rname: str, w: dict, ref: dict, dirs=None) -> np.ndarray:
    """Edit the (T, d) P tokens ``toks`` of one epoch in the basis ``B``: coordinates ``dirs``
    (default: the word's) are brought to the DCSM reference. pq x3 words edit the 3-token window
    around each token, sequentially, writing back only the blocks inside the epoch."""
    wd = w["dirs"]
    dirs = wd if dirs is None else dirs
    P = P.astype(np.float64).copy()
    T, d = P.shape
    for t in toks:
        if rname == "pq1":
            x = P[t][None]
        else:
            a = P[t - 1] if t > 0 else prev_last
            c = P[t + 1] if t < T - 1 else next_first
            x = np.concatenate([a, P[t], c])[None]
        z = B.transform(x)
        if w["kind"] == "energy":
            # factor from the word's own energy, applied to ``dirs`` (= the word, or the control group)
            e = float((z[0, wd] ** 2).sum())
            if e > ref["energy_median"]:
                z[0, dirs] *= np.sqrt(ref["energy_median"] / e)
        else:
            z[0, dirs] = ref["median"][dirs] if w["kind"] == "signed" else np.sign(z[0, dirs]) * ref["abs_median"][dirs]
        x2 = B.inverse_transform(z)[0]
        if rname == "pq1":
            P[t] = x2
        else:
            for k, tt in ((0, t - 1), (1, t), (2, t + 1)):
                if 0 <= tt < T:
                    P[tt] = x2[k * d:(k + 1) * d]
    return P.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--vq", default="N8E8P6")
    ap.add_argument("--stager", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--arrays", required=True, help="battery root of the run (lexicon/dcsm_readout.npz)")
    ap.add_argument("--lexicon2", required=True, help="lexicon2_<tag>.json of vq_event_lexicon2")
    ap.add_argument("--n-words", type=int, default=2)
    ap.add_argument("--n-control", type=int, default=20)
    ap.add_argument("--target", default="dreams")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    from physioex.data.datasets import get_dataset
    from physioex.models import vq_explain as vx
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.qformer_stager import ARCHS, latent_stats
    from physioex.models.recon_eval import _load_run, channel_pipelines

    os.makedirs(a.out, exist_ok=True)
    log = lambda *s: print("[vq_event_fidelity]", *s, flush=True)  # noqa: E731
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    run, _, model = _load_run(a.run)
    ra = run["args"]
    model.vq_levels = parse_vq_key(a.vq)
    model.to(dev).eval()
    seq, T = ra["seq"], model.n_tok
    hseq = seq // 2
    lz = np.load(os.path.join(a.latents, "latents_train.npz"))
    lat = {k: lz[k] for k in ("N", "E", "P")}
    st = latent_stats(lat)
    sj = json.load(open(os.path.join(a.stager, "stager.json")))
    sa = sj["args"]
    stager = ARCHS[sa["arch"]](dim=lat["N"].shape[-1], d_p=lat["P"].shape[-1], n_tok=lat["P"].shape[1], d=sa["d"], k=sa["k"],
                               layers=sa["layers"], heads=sa["heads"], dropout=sa["dropout"], seq_len=sa["seq_len"],
                               seq_layers=sa["seq_layers"])
    stager.load_state_dict(torch.load(os.path.join(a.stager, "stager.pt"), map_location="cpu"))
    stager.to(dev).eval()
    L, half = sa["seq_len"], sa["seq_len"] // 2
    stdz = lambda k, v: (v - st[k][0]) / st[k][1]  # noqa: E731

    @torch.no_grad()
    def p_n2(N, E, P):
        return stager(torch.from_numpy(N[None]).float().to(dev), torch.from_numpy(E[None]).float().to(dev),
                      torch.from_numpy(P[None]).float().to(dev))[0, half].float().softmax(-1).cpu().numpy()

    @torch.no_grad()
    def requant(P):
        return model._q("P", torch.from_numpy(P).float().to(dev))[0].cpu().numpy()

    # DREAMS: latents of every epoch (reflected windows), signal for the decoded check
    tds = get_dataset(a.target)(channels=[ra.get("channel", "EEG_clean")], pipelines=channel_pipelines(ra),
                                sequence_length=1, cache_dir=ra.get("cache_dir"))
    recs = {}
    for spec in tds._subjects:
        sid = spec.subject_id
        n = tds._n_epochs[sid]
        x = next(iter(tds._build_item(spec, 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        idx = reflect_index(n, hseq)
        Z = {"N": [], "E": [], "P": []}
        with torch.no_grad():
            for i in range(0, n, 32):
                w = np.stack([x[idx[s:s + seq]] for s in range(i, min(n, i + 32))])
                z = model.encode(torch.from_numpy(w)[:, :, None].to(dev))
                for k in Z:
                    Z[k].append(z[k].float().cpu().numpy())
        recs[sid] = {"id": sid, "x": x, "y": np.asarray(tds._get_labels(spec))[:n], "events": tds._load_or_compute_events(spec),
                     **{k: np.concatenate(v) for k, v in Z.items()}}
    log("encoded", len(recs), "excerpts")
    # lexicon bases (same fit as vq_event_lexicon2: DCSM, seed 0)
    dc = np.load(os.path.join(a.arrays, "lexicon", "dcsm_readout.npz"))
    dcsm = {"pq": dc["pq"], "zE": dc["zE"], "stage": dc["stage"]}
    bases, Xd = fit_bases(dcsm, 0)
    lex = json.load(open(a.lexicon2))
    from physioex.train.bin.vq_event_lexicon2 import energy_groups

    rep = {"run": a.run, "stager": a.stager, "lexicon2": a.lexicon2, "events": {}}
    for pref, kind in KINDS.items():
        prim = "vis1" if kind == "k_complex" else "union"
        R = [recs[s] for s in recs if s.startswith(pref)]
        refsets = reference_sets(R, kind)
        refs = {**refsets["vis1"], **refsets.get(prim, {})} if prim != "vis1" else refsets["vis1"]
        words = [w["word"] for w in lex["events"][kind]["lexicon"][prim]["words_picked"][:a.n_words]]
        res = {"reference": prim, "words": {}}
        for wname in words:
            bname = wname.split(":")[0]
            rname, B = bases[bname]
            Zd = B.transform(Xd[rname])
            groups = energy_groups(Zd)
            w = parse_word(wname, groups)
            ref = {"median": np.median(Zd, 0), "abs_median": np.median(np.abs(Zd), 0),
                   "energy_median": float(np.median((Zd[:, w["dirs"]] ** 2).sum(1)))}
            Zr = {r["id"]: B.transform(r["P"].reshape(-1, r["P"].shape[-1]) if rname == "pq1" else ctx3(r["P"])) for r in R}
            names, masks, _ = basis_words(Zd, Zr, {r["id"]: len(r["y"]) * T for r in R}, bname)
            wi = names.index(wname)
            rows = []
            for r in R:
                n = len(r["y"])
                m = masks[r["id"]][wi].reshape(n, T)
                Ns, Es, Ps = stdz("N", r["N"]), stdz("E", r["E"]), stdz("P", r["P"])
                for t in range(hseq, n - hseq):  # full N context
                    if t - half < 0 or t + half >= n:
                        continue
                    toks = np.flatnonzero(m[t])
                    silent = np.flatnonzero(~m[t])
                    if not len(toks) or len(silent) < len(toks):
                        continue
                    win = slice(t - half, t + half + 1)
                    base = p_n2(Ns[win], Es[win], Ps[win])
                    prev_last = r["P"][t - 1, -1]; next_first = r["P"][t + 1, 0]

                    def delta(tk, dirs=None):
                        P2 = r["P"][t].copy()
                        Pe = edit_epoch(r["P"][t], prev_last, next_first, tk, B, rname, w, ref, dirs)
                        ch = np.unique(np.clip(np.concatenate([tk - 1, tk, tk + 1]) if rname != "pq1" else tk, 0, T - 1))
                        P2[ch] = requant(Pe[ch])
                        Pw = Ps[win].copy(); Pw[half] = stdz("P", P2)
                        return float(p_n2(Ns[win], Es[win], Pw)[N2] - base[N2]), P2, ch

                    d_word, P_ed, ch = delta(toks)
                    d_loc = [delta(np.sort(rng.choice(silent, len(toks), replace=False)))[0] for _ in range(a.n_control)]
                    D = B.transform(np.zeros((1, Xd[rname].shape[1]))).shape[1]
                    d_dir = [delta(toks, list(rng.choice(D, len(w["dirs"]), replace=False)))[0] for _ in range(a.n_control)]
                    # decoded check inside the edited tokens
                    with torch.no_grad():
                        tt = lambda v: torch.from_numpy(v[None]).float().to(dev)  # noqa: E731
                        dec0 = vx.decode_full(model, tt(r["N"][t]), tt(r["E"][t]), tt(r["P"][t])).float().cpu().numpy()[0]
                        dec1 = vx.decode_full(model, tt(r["N"][t]), tt(r["E"][t]), tt(P_ed)).float().cpu().numpy()[0]
                    seg = 3000 // T
                    sel = np.concatenate([np.arange(j * seg, (j + 1) * seg) for j in toks])
                    if kind == "spindle":
                        f0, f1 = np.sqrt((_bp(dec0, 11, 16)[sel] ** 2).mean()), np.sqrt((_bp(dec1, 11, 16)[sel] ** 2).mean())
                    else:
                        f0, f1 = np.ptp(_bp(dec0, 0.3, 4.0)[sel]), np.ptp(_bp(dec1, 0.3, 4.0)[sel])
                    has_ev = any(a_ < (t + 1) * 30 and b_ > t * 30 for a_, b_ in refs.get(r["id"], []))
                    rows.append({"excerpt": r["id"], "epoch": t, "stage": int(r["y"][t]), "expert_event_in_epoch": bool(has_ev),
                                 "n_tokens": int(len(toks)), "p_n2": float(base[N2]), "pred": int(base.argmax()),
                                 "delta_word": d_word, "delta_loc_mean": float(np.mean(d_loc)), "delta_dir_mean": float(np.mean(d_dir)),
                                 "rank_vs_loc": float(np.mean(np.array(d_loc) <= d_word)), "rank_vs_dir": float(np.mean(np.array(d_dir) <= d_word)),
                                 "decoded_feature_before": float(f0), "decoded_feature_after": float(f1)})
            res["words"][wname] = {"parsed": {k: v for k, v in w.items()}, "rows": rows, "summary": summarize(rows)}
            log(kind, wname, res["words"][wname]["summary"]["all"])
        rep["events"][kind] = res
    json.dump(_jsonable(rep), open(os.path.join(a.out, "vq_event_fidelity.json"), "w"), indent=1)
    log("DONE")


def summarize(rows: list) -> dict:
    from scipy.stats import wilcoxon

    def s(sel):
        if len(sel) < 3:
            return {"n": len(sel)}
        dw = np.array([r["delta_word"] for r in sel]); dl = np.array([r["delta_loc_mean"] for r in sel])
        dd = np.array([r["delta_dir_mean"] for r in sel])
        fb = np.array([r["decoded_feature_before"] for r in sel]); fa = np.array([r["decoded_feature_after"] for r in sel])
        out = {"n": len(sel), "delta_word_median": float(np.median(dw)), "delta_word_mean": float(dw.mean()),
               "delta_loc_median": float(np.median(dl)), "delta_dir_median": float(np.median(dd)),
               "frac_word_below_loc": float(np.mean(dw < dl)), "frac_word_below_dir": float(np.mean(dw < dd)),
               "decoded_ratio_median": float(np.median(fa / np.maximum(fb, 1e-9)))}
        for nm, c in (("p_wilcoxon_vs_loc", dl), ("p_wilcoxon_vs_dir", dd)):
            try:
                out[nm] = float(wilcoxon(dw, c).pvalue)
            except ValueError:
                out[nm] = None
        return out

    return {"all": s(rows), "scored_n2": s([r for r in rows if r["stage"] == N2]),
            "expert_event": s([r for r in rows if r["expert_event_in_epoch"]]),
            "no_expert_event": s([r for r in rows if not r["expert_event_in_epoch"]])}


if __name__ == "__main__":
    main()
