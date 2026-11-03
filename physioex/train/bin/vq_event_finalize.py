"""Freeze the event lexicon of an MSAE-VQ EEG run and run the last checks of the explainability
workflow (L2 stability, L4 examples).

1. **Selection.** The bases of :mod:`vq_event_lexicon2` (DCSM, seed 0) and its label-free words;
   the final lexicon of each event is chosen on ALL DREAMS excerpts with the same greedy rule
   (K-complex vs expert 1, spindles vs the union of the experts); the leave-one-excerpt-out
   numbers of S4b remain the performance estimate. The lexicon is saved as a
   :class:`physioex.models.vq_lexicon.Lexicon` (``lexicon.json`` + ``lexicon.npz``) in ``--out``
   and in ``RUN/lexicon``; the frozen reader must reproduce the masks of the selection exactly.
2. **L2 stability.**
   (a) domain: firing rate of every word per stage in DCSM and in DREAMS;
   (b) context: the centre epochs re-encoded as if the recording started 5 or 10 epochs before
       (reflected context, as ``vq_event_codes.stability``): token agreement of every word and
       agreement of the events;
   (c) basis: the bases refitted with 4 other seeds / DCSM subsamples, leave-one-excerpt-out
       lexicon again, and the best-matching word of the new basis for every final word
       (Jaccard of the DREAMS masks).
3. **L4 examples.** Two epochs per event (expert event present, word firing): input, decoded,
   decoded after the word edit of ``vq_event_fidelity``, word masks and expert events, saved for
   the page.

``python -m physioex.train.bin.vq_event_finalize --run VQ_RUN --arrays DIR --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from physioex.train.bin.domain_shift_report import _jsonable
from physioex.train.bin.vq_event_codes import KINDS, reflect_index
from physioex.train.bin.vq_event_fidelity import edit_epoch, parse_word
from physioex.train.bin.vq_event_lexicon2 import (COVERAGE_MIN, basis_words, counts, energy, energy_groups, fit_bases,
                                                  loo_lexicon, prf, reference_sets, select_lexicon, word_counts)
from physioex.train.bin.vq_event_readout import ctx3, readout_batch, readout_recording

TOL = 0.5


def word_spec(name: str, Zd: np.ndarray, groups: dict, rep: str, B, event: str) -> dict:
    w = parse_word(name, groups)
    q = float(name.rsplit("q", 1)[1])
    if w["kind"] == "signed":
        vals, side = Zd[:, w["dirs"][0]], (-1 if "<=" in name else 1)
    elif w["kind"] == "folded":
        vals, side = np.abs(Zd[:, w["dirs"][0]]), 1
    else:
        vals, side = energy(Zd, w["dirs"], "~" in name, None), 1
    return {"name": name, "event": event, "basis": name.split(":")[0], "kind": w["kind"], "dirs": [int(d) for d in w["dirs"]],
            "threshold": float(np.quantile(vals, q)), "side": side, "smooth": "~" in name, "dcsm_quantile": q}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--vq", default="N8E8P6")
    ap.add_argument("--arrays", required=True)
    ap.add_argument("--target", default="dreams")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--no-save-run", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    from physioex.data.datasets import get_dataset
    from physioex.models import vq_explain as vx
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run, channel_pipelines
    from physioex.models.vq_lexicon import Lexicon

    os.makedirs(a.out, exist_ok=True)
    log = lambda *s: print("[vq_event_finalize]", *s, flush=True)  # noqa: E731
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run, _, model = _load_run(a.run)
    ra = run["args"]
    model.vq_levels = parse_vq_key(a.vq)
    model.to(dev).eval()
    seq, T = ra["seq"], model.n_tok
    half = seq // 2
    tds = get_dataset(a.target)(channels=[ra.get("channel", "EEG_clean")], pipelines=channel_pipelines(ra),
                                sequence_length=1, cache_dir=ra.get("cache_dir"))
    t0 = time.time()
    recs = []
    for spec in tds._subjects:
        sid = spec.subject_id
        n = tds._n_epochs[sid]
        x = next(iter(tds._build_item(spec, 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        ro = readout_recording(model, x, seq)
        recs.append({"id": sid, "x": x, "y": np.asarray(tds._get_labels(spec))[:n], "events": tds._load_or_compute_events(spec),
                     "pq": ro["pq"], "zE": ro["zE"]})
    rec_by = {r["id"]: r for r in recs}
    log("read", len(recs), f"{time.time() - t0:.0f}s")
    dc = np.load(os.path.join(a.arrays, "lexicon", "dcsm_readout.npz"))
    dcsm = {"pq": dc["pq"], "zE": dc["zE"], "stage": dc["stage"]}
    dstage = np.repeat(dcsm["stage"], T)

    def all_words(seed):
        bases, Xd = fit_bases(dcsm, seed)
        names, masks, Zds = [], {r["id"]: [] for r in recs}, {}
        for bname, (rname, B) in bases.items():
            Zd = B.transform(Xd[rname])
            Zds[bname] = Zd
            Zr = {r["id"]: B.transform(r["pq"].reshape(-1, r["pq"].shape[-1]).astype(np.float32) if rname == "pq1"
                                       else ctx3(r["pq"]))
                  for r in recs}
            nm, mk, _ = basis_words(Zd, Zr, {r["id"]: len(r["y"]) * T for r in recs}, bname)
            names += nm
            for r in recs:
                masks[r["id"]].append(mk[r["id"]])
        return bases, Xd, Zds, names, {r: np.concatenate(v) for r, v in masks.items()}

    bases, Xd, Zds, names, masks = all_words(0)
    rep = {"run": a.run, "vq": a.vq, "selection": {}, "L2": {"domain": {}, "context": {}, "basis": {}}, "consistency": {}}
    specs = []
    sel_idx = {}
    for pref, kind in KINDS.items():
        R = [r for r in recs if r["id"].startswith(pref)]
        refsets = reference_sets(R, kind)
        prim = "vis1" if kind == "k_complex" else "union"
        rr = refsets[prim]
        mk = {r: masks[r] for r in rr}
        wc = word_counts(mk, rr, TOL)
        ws = select_lexicon(mk, rr, wc, list(rr), COVERAGE_MIN[kind], TOL)
        sel_idx[kind] = ws
        c_all = sum(counts(np.any(mk[r][ws], 0), rr[r], TOL) for r in rr)
        rep["selection"][kind] = {"reference": prim, "words": [names[w] for w in ws], "in_sample": prf(c_all),
                                  "per_word_in_sample": {names[w]: prf(sum(counts(mk[r][w], rr[r], TOL) for r in rr)) for w in ws}}
        for w in ws:
            bname = names[w].split(":")[0]
            rname, B = bases[bname]
            specs.append(word_spec(names[w], Zds[bname], energy_groups(Zds[bname]), rname, B, kind))
        log(kind, "final words", rep["selection"][kind]["words"], rep["selection"][kind]["in_sample"])
    lb = {}
    for bname in {s["basis"] for s in specs}:
        rname, B = bases[bname]
        lb[bname] = {"rep": rname, "mean": B.mean_.astype(np.float32), "unmix": B.components_.astype(np.float32),
                     "mix": B.mixing_.astype(np.float32)}
    lex = Lexicon(lb, specs, {"run": a.run, "vq": a.vq, "fit": "DCSM 40 train nights x 150 epochs, FastICA seed 0",
                              "named_with": "DREAMS experts (K-complex: expert 1; spindles: union of experts 1 and 2)",
                              "tolerance_s": TOL})
    lex.save(a.out)
    if not a.no_save_run:
        lex.save(os.path.join(a.run, "lexicon"))
    lex2 = Lexicon.load(a.out)
    # consistency: the frozen reader reproduces the selection masks
    for s in specs:
        wi = names.index(s["name"])
        agree = [float(np.mean(lex2.fire(r["pq"])[s["name"]] == masks[r["id"]][wi])) for r in recs]
        rep["consistency"][s["name"]] = float(np.min(agree))
    log("consistency (min token agreement frozen reader vs selection)", rep["consistency"])

    # L2a domain: firing per stage
    for s in specs:
        wi = names.index(s["name"])
        fd = lex2.fire(dcsm["pq"], continuous=False)[s["name"]]
        dre = np.concatenate([lex2.fire(r["pq"])[s["name"]] for r in recs])
        dst = np.concatenate([np.repeat(r["y"], T) for r in recs])
        rep["L2"]["domain"][s["name"]] = {
            "dcsm_rate_by_stage": [float(fd[dstage == k].mean()) if (dstage == k).any() else None for k in range(5)],
            "dreams_rate_by_stage": [float(dre[dst == k].mean()) if (dst == k).any() else None for k in range(5)],
            "dcsm_rate": float(fd.mean()), "dreams_rate": float(dre.mean())}
    log("L2 domain", {k: (round(v["dcsm_rate_by_stage"][2], 4), round(v["dreams_rate_by_stage"][2] or 0, 4)) for k, v in rep["L2"]["domain"].items()})

    # L2b context: recordings starting d epochs before the centre epoch
    for d in (5, 10):
        agree = {s["name"]: [] for s in specs}
        ev_agree = {k: [] for k in KINDS.values()}
        for r in recs:
            n = len(r["y"])
            full = lex2.fire(r["pq"])
            for t in range(half, n - half):
                sub = r["x"][t - d:]
                idx = reflect_index(len(sub), half)
                win = np.stack([sub[idx[d:d + seq]]])
                pq_t = readout_batch(model, win)["pq"]  # (1, T, d_p)
                pq_mix = r["pq"].copy()
                pq_mix[t] = pq_t[0]
                alt = lex2.fire(pq_mix)
                sl = slice(t * T, (t + 1) * T)
                for s in specs:
                    a_, b_ = full[s["name"]][sl], alt[s["name"]][sl]
                    if a_.any() or b_.any():
                        agree[s["name"]].append(float((a_ & b_).sum() / (a_ | b_).sum()))
                for kind in KINDS.values():
                    ns = [s["name"] for s in specs if s["event"] == kind]
                    ea, eb = np.any([full[k][sl] for k in ns], 0), np.any([alt[k][sl] for k in ns], 0)
                    if ea.any() or eb.any():
                        ev_agree[kind].append(float((ea & eb).sum() / (ea | eb).sum()))
        rep["L2"]["context"][str(d)] = {"word_jaccard_median": {k: float(np.median(v)) if v else None for k, v in agree.items()},
                                        "word_jaccard_mean": {k: float(np.mean(v)) if v else None for k, v in agree.items()},
                                        "event_jaccard_mean": {k: float(np.mean(v)) if v else None for k, v in ev_agree.items()},
                                        "n": {k: len(v) for k, v in agree.items()}}
        log("L2 context", d, rep["L2"]["context"][str(d)]["event_jaccard_mean"])

    # L2c basis: other seeds / subsamples
    for seed in a.seeds:
        _, _, _, nm2, mk2 = all_words(seed)
        out = {}
        for pref, kind in KINDS.items():
            R = [r for r in recs if r["id"].startswith(pref)]
            refsets = reference_sets(R, kind)
            prim = "vis1" if kind == "k_complex" else "union"
            rr = refsets[prim]
            L = loo_lexicon({r: mk2[r] for r in rr}, rr, COVERAGE_MIN[kind], TOL, None, nm2)
            match = {}
            for s in specs:
                if s["event"] != kind:
                    continue
                m0 = np.concatenate([lex2.fire(rec_by[r]["pq"])[s["name"]] for r in rr])
                M = np.concatenate([mk2[r] for r in rr], 1)
                jac = (M & m0).sum(1) / np.maximum((M | m0).sum(1), 1)
                j = int(np.argmax(jac))
                match[s["name"]] = {"best_word": nm2[j], "jaccard": float(jac[j])}
            out[kind] = {"loo_lexicon": L["lexicon"], "words_picked": L["words_picked"][:3], "match": match}
        rep["L2"]["basis"][str(seed)] = out
        log("L2 basis seed", seed, {k: (round(v["loo_lexicon"]["purity"], 3), round(v["loo_lexicon"]["coverage"], 3),
                                         [round(m["jaccard"], 2) for m in v["match"].values()]) for k, v in out.items()})

    # L4 examples
    waves = {}
    rep["examples"] = {}
    for pref, kind in KINDS.items():
        R = [r for r in recs if r["id"].startswith(pref)]
        refsets = reference_sets(R, kind)
        prim = "vis1" if kind == "k_complex" else "union"
        rr = refsets[prim]
        ks = [s for s in specs if s["event"] == kind]
        cand = []
        for r in R:
            if r["id"] not in rr:
                continue
            fm = lex2.fire(r["pq"])
            for t in range(half, len(r["y"]) - half):
                sl = slice(t * T, (t + 1) * T)
                ev = [(a_ - t * 30, b_ - t * 30) for a_, b_ in rr[r["id"]] if a_ < (t + 1) * 30 and b_ > t * 30]
                fired = np.any([fm[s["name"]][sl] for s in ks], 0)
                if not ev or not fired.any():
                    continue
                exp_tok = np.zeros(T, bool)
                for a_, b_ in ev:
                    exp_tok[max(0, int(a_ // 0.5)):min(T, int(np.ceil(b_ / 0.5)))] = True
                cand.append(((fired & exp_tok).sum() / max((fired | exp_tok).sum(), 1), r["id"], t, ev))
        cand.sort(key=lambda c: -c[0])
        picked, seen = [], set()
        for c in cand:
            if c[1] not in seen and int(rec_by[c[1]]["y"][c[2]]) == 2:
                picked.append(c); seen.add(c[1])
            if len(picked) == 2:
                break
        rep["examples"][kind] = []
        for j, (score, rid, t, ev) in enumerate(picked):
            r = rec_by[rid]
            fm = lex2.fire(r["pq"])
            s0 = ks[0]
            wi_tok = np.flatnonzero(fm[s0["name"]][t * T:(t + 1) * T])
            rname = lex2.bases[s0["basis"]]["rep"]
            Bn = bases[s0["basis"]][1]
            Zd = Zds[s0["basis"]]
            ref = {"median": np.median(Zd, 0), "abs_median": np.median(np.abs(Zd), 0),
                   "energy_median": float(np.median((Zd[:, s0["dirs"]] ** 2).sum(1)))}
            w = {"kind": s0["kind"], "dirs": s0["dirs"]}
            P0 = r["pq"][t].astype(np.float32)
            Pe = edit_epoch(P0, r["pq"][t - 1, -1].astype(np.float32), r["pq"][t + 1, 0].astype(np.float32), wi_tok, Bn, rname, w, ref)
            with torch.no_grad():
                P2 = P0.copy()
                if len(wi_tok):
                    ch = np.unique(np.clip(np.concatenate([wi_tok - 1, wi_tok, wi_tok + 1]) if rname != "pq1" else wi_tok, 0, T - 1))
                    P2[ch] = model._q("P", torch.from_numpy(Pe[ch]).float().to(dev))[0].cpu().numpy()
                idx = reflect_index(len(r["x"]), half)
                z = model.encode(torch.from_numpy(r["x"][idx[t:t + seq]][None, :, None]).to(dev))
                dec0 = vx.decode_full(model, z["N"], z["E"], z["P"]).float().cpu().numpy()[0]
                dec1 = vx.decode_full(model, z["N"], z["E"], torch.from_numpy(P2[None]).float().to(dev)).float().cpu().numpy()[0]
            key = f"{kind}_{j}"
            waves[f"{key}_input"], waves[f"{key}_decoded"], waves[f"{key}_edited"] = r["x"][t], dec0, dec1
            rep["examples"][kind].append({"excerpt": rid, "epoch": int(t), "jaccard": float(score), "expert_events_s": ev,
                                          "words": {s["name"]: np.flatnonzero(fm[s["name"]][t * T:(t + 1) * T]).tolist() for s in ks},
                                          "edited_word": s0["name"], "key": key})
    np.savez_compressed(os.path.join(a.out, "examples_waves.npz"), **waves)
    json.dump(_jsonable(rep), open(os.path.join(a.out, "vq_event_finalize.json"), "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
