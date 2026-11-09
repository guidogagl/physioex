"""Blind review of the lexicon concepts by critics who hold the sleep manual (plan of 09.11.2026, part 3).

For every concept, 20 examples drawn with a fixed seed and stratified by agreement between the **rule**
(the reference detector of :mod:`physioex.explain.concept_refs`) and the **model** (the lexicon rule of the
kept battery iteration, masks saved by ``concept_compose battery --export``):

- 8 ``both``: a run of the model overlapping a run of the rule with IoU >= 0.2 (:func:`concept_examples.events`);
- 6 ``rule_only``: a run of the rule that no model token touches;
- 6 ``model_only``: a run of the model that no rule token touches.

A short stratum is filled from the others and the shortfall is written in ``selection.json``. The model never
saw these tokens while it was fitted: pool nights carry leave-one-site-out scores, the other sets the
pool fit. Nights listed in ``--prefer`` (the nights of the vaglini mini-cache) are drawn first; at most one
example per night and stratum until the candidates run out.

Sub-commands:

- ``pick`` (CPU, where the dump is): writes ``<out>/<concept>/ex_XX.npz`` (3 epochs of uV around the event,
  rule and model token masks, hypnogram) and ``<out>/selection.json``.
- ``png`` (anywhere): one image per example, the same signal twice, panels A and B, each with its own
  highlights; which panel is the rule is drawn at random per image and kept in ``key.json`` only, never in
  the image. ``<out>/<concept>/ex_XX.png``.
- ``aggregate``: reads the critics' verdicts (``verdicts/<concept>_<lens>.json``), opens the key and writes
  ``review.json``: share of "yes" for rule and model per stratum and lens, Fleiss kappa among the critics.

Signals are NSRR / MASS data under DUA: outputs stay on Sofia, in ``~/work/outputs`` and in private scratch.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from physioex.train.bin.concept_examples import SPECS, Masks, Night, events, lex_mask

T = 60
EP_S = 30.0
STRATA = {"both": ("tp", 8), "rule_only": ("fn", 6), "model_only": ("fp", 6)}
STAGES = ["W", "N1", "N2", "N3", "R"]
#: channels shown per concept (the concept's own modality first)
SHOW = {"kc": ["eeg"], "n3": ["eeg"], "spindle": ["eeg"], "alpha_W": ["eeg", "eog"], "alpha_REM": ["eeg", "emg"],
        "rem_eye": ["eog", "emg"], "blink": ["eog"], "cap_a1": ["eeg"], "arousal": ["eeg", "emg"]}
#: seconds shown around the event (30 s = the centre epoch; longer windows are centred on the event)
WIN_S = {"cap_a1": 60.0, "arousal": 60.0, "spindle": 10.0, "kc": 15.0}  # 10 s to count spindle cycles [F1 p1]


# ---------------------------------------------------------------------------- pick
def preferred(key, prefer):
    """``key`` = ``<unit>/<sid>`` of the dump; ``prefer`` holds ``<cache folder>/<sid>`` (``hpap/x`` matches
    ``hpap_lab_full/x``) or full dump keys."""
    if key in prefer:
        return True
    unit, _, sid = key.rpartition("/")
    return any(p.endswith("/" + sid) and unit.startswith(p.rpartition("/")[0]) for p in prefer)


def _draw(cands, n, prefer, rng):
    """Up to ``n`` candidates: preferred nights first, one per night until the candidates run out."""
    order = [cands[i] for i in rng.permutation(len(cands))]
    pref = {c[0]: preferred(c[0], prefer) for c in order}
    order.sort(key=lambda c: not pref[c[0]])  # stable: random within preferred / others
    out, seen = [], set()
    for c in order:
        if len(out) < n and c[0] not in seen:
            out.append(c)
            seen.add(c[0])
    for c in order:
        if len(out) < n and c not in out:
            out.append(c)
    return out


def select(cands, prefer, rng):
    """Stratified 8 / 6 / 6; a short stratum is filled from the others (largest first)."""
    want = {s: n for s, (_, n) in STRATA.items()}
    picks = {s: _draw(cands.get(s, []), want[s], prefer, rng) for s in STRATA}
    short = {s: want[s] - len(picks[s]) for s in STRATA if len(picks[s]) < want[s]}
    missing = sum(short.values())
    for s in sorted(STRATA, key=lambda s: -len(cands.get(s, []))):
        if not missing:
            break
        rest = [c for c in cands.get(s, []) if c not in picks[s]]
        extra = _draw(rest, missing, prefer, rng)
        picks[s] += extra
        missing -= len(extra)
    return picks, short


def pick(a):
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    sc = json.load(open(os.path.join(a.dump, "scale.json")))
    fs = {m: float(v) for m, v in meta["fs"].items()}
    masks = {g: Masks(p) for g, p in (s.split("=", 1) for s in a.masks)}
    prefer = set(open(a.prefer).read().split()) if a.prefer else set()
    from physioex.train.bin.concept_scan import load_refs

    rng = np.random.default_rng(a.seed)
    sel = {"seed": a.seed, "strata": {s: n for s, (_, n) in STRATA.items()}, "prefer_n": len(prefer), "concepts": {}}
    for cname in a.concepts:
        spec = SPECS[cname]
        M = masks.get(spec["masks"])
        if M is None or not M.has(spec["lex"]):
            print(f"[review] {cname}: masks missing, skipped", flush=True)
            continue
        cands = {}
        for key in M.keys:
            ref = spec["ref"].get(M.set_n.get(key))
            if not ref:
                continue
            rf = load_refs(os.path.join(a.dump, "refs", f"{key}.npz"))
            if ref not in rf:
                continue
            lx = lex_mask(M, spec, key)
            r = np.asarray(rf[ref], bool)[:len(lx)]
            for s, (kind, _) in STRATA.items():
                cands.setdefault(s, []).extend((key, *ev) for ev in events(lx, r, len(lx) // T, kind))
        picks, short = select(cands, prefer, rng)
        rows = []
        os.makedirs(os.path.join(a.out, cname), exist_ok=True)
        i = 0
        for s in STRATA:
            for key, t0, t1, ep in picks[s]:
                nt = Night(a.dump, key, sc, fs)
                ctx = [ep - 1, ep, ep + 1]
                k = M.keys.index(key)
                sl = slice((ep - 1) * T, (ep + 2) * T)
                ref = spec["ref"][M.set_n[key]]
                arr = {"y_ep": M.z["y_ep"][M.ep0[k] + ep - 1:M.ep0[k] + ep + 2],
                       "rule": np.asarray(load_refs(os.path.join(a.dump, "refs", f"{key}.npz"))[ref], bool)[sl],
                       "model": lex_mask(M, spec, key)[sl]}
                for m in ("eeg", "eog", "emg"):
                    if nt.has(m):
                        arr[f"x_{m}"] = nt.uv(m, nt.x(m)[ctx]).astype(np.float32)
                        arr[f"fs_{m}"] = np.array(fs[m])
                row = {"concept": cname, "i": i, "stratum": s, "key": key, "unit": meta["nights"][key]["unit"],
                       "set": M.set_n[key], "epoch": int(ep), "a": int(t0 - (ep - 1) * T),
                       "b": int(t1 - (ep - 1) * T), "rule_ref": ref, "preferred": preferred(key, prefer)}
                np.savez_compressed(os.path.join(a.out, cname, f"ex_{i:02d}.npz"), meta=json.dumps(row), **arr)
                rows.append(row)
                i += 1
        sel["concepts"][cname] = {"candidates": {s: len(c) for s, c in cands.items()}, "short": short,
                                  "examples": rows}
        print(f"[review] {cname}: {i} examples, candidates {sel['concepts'][cname]['candidates']}, short {short}",
              flush=True)
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)


# ---------------------------------------------------------------------------- png
def _window(row, cname, n_samp, fs):
    """Sample slice of the 3-epoch window shown for the example."""
    w = WIN_S.get(cname, EP_S)
    if w == EP_S:
        lo = int(EP_S * fs)
    else:
        mid = (row["a"] + row["b"]) / 2 * 0.5  # seconds from the window start
        lo = int(np.clip(mid - w / 2, 0, 3 * EP_S - w) * fs)
    return slice(lo, lo + int(w * fs))


def render(path_npz, path_png, cname, rule_panel, dpi=110):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    z = np.load(path_npz)
    row = json.loads(str(z["meta"]))
    mods = [m for m in SHOW[cname] if f"x_{m}" in z.files]
    fs0 = float(z[f"fs_{mods[0]}"])
    sl0 = _window(row, cname, len(z[f"x_{mods[0]}"].ravel()), fs0)
    t_lo, t_hi = sl0.start / fs0, sl0.stop / fs0
    panels = {"A": "rule" if rule_panel == "A" else "model", "B": "model" if rule_panel == "A" else "rule"}
    n_rows = len(mods)
    fig, axes = plt.subplots(2 * n_rows, 1, figsize=(16, 2.1 * 2 * n_rows + 0.8), dpi=dpi, sharex=True,
                             gridspec_kw={"hspace": 0.15})
    axes = np.atleast_1d(axes)
    span = {}
    for m in mods:
        x = z[f"x_{m}"].ravel()
        fs = float(z[f"fs_{m}"])
        seg = x[int(t_lo * fs):int(t_hi * fs)]
        span[m] = max(np.abs(seg - np.median(seg)).max() * 1.08, 10.0)  # never clip: amplitude is a criterion
    for p_i, (pname, src) in enumerate(panels.items()):
        mask = z[src]
        for r_i, m in enumerate(mods):
            ax = axes[p_i * n_rows + r_i]
            x = z[f"x_{m}"].ravel()
            fs = float(z[f"fs_{m}"])
            t = np.arange(len(x)) / fs
            k = (t >= t_lo) & (t < t_hi)
            ax.plot(t[k], x[k] - np.median(x[k]), lw=0.6, color="#1f2328")
            for c, d in _runs(mask):
                s0, s1 = max(c * 0.5, t_lo), min(d * 0.5, t_hi)
                if s1 > s0:
                    ax.axvspan(s0, s1, color="#e8590c", alpha=0.28, lw=0)
            ax.set_ylim(-span[m], span[m])
            ax.set_xlim(t_lo, t_hi)
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))
            ax.grid(True, which="major", color="#c9ced3", lw=0.6)
            ax.grid(True, which="minor", color="#eceff1", lw=0.4)
            ax.set_yticks([])
            ax.set_ylabel(f"{pname}\n{m.upper()}", rotation=0, ha="right", va="center", fontsize=11,
                          fontweight="bold" if r_i == 0 else "normal")
            bar = _bar(span[m])
            xb = t_hi + (t_hi - t_lo) * 0.008  # scale bar outside the trace, right margin
            ax.plot([xb, xb], [-bar / 2, bar / 2], color="#1f2328", lw=2, clip_on=False)
            ax.text(xb + (t_hi - t_lo) * 0.004, 0, f"{bar:g} µV", ha="left", va="center", fontsize=9)
            for s in (EP_S, 2 * EP_S):
                if t_lo < s < t_hi:
                    ax.axvline(s, color="#57606a", lw=1.0, ls="--")
            ax.tick_params(axis="x", which="both", labelbottom=False, length=0)
    ax = axes[-1]
    ax.tick_params(axis="x", which="major", labelbottom=True, length=3)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v - t_lo:.0f}" if abs((v - t_lo) % 5) < 1e-6 else "")
    ax.set_xlabel("time (s); grid: 1 s major, 0.2 s minor; dashed line = epoch boundary; positive voltage UP")
    y = z["y_ep"]
    hyp = " | ".join(f"{'▶ ' if j == 1 else ''}epoch {j - 1:+d}: {STAGES[int(v)] if 0 <= int(v) < 5 else '?'}"
                     for j, v in enumerate(y))
    fs_txt = ", ".join(f"{m.upper()} {float(z[f'fs_{m}']):g} Hz" for m in mods)
    fig.suptitle(f"{cname} · example {row['i']:02d} · hypnogram: {hyp} · {fs_txt} · shaded = marked by the "
                 f"method of that panel", fontsize=10, x=0.01, ha="left")
    fig.savefig(path_png, bbox_inches="tight")
    plt.close(fig)


def _runs(m):
    m = np.asarray(m, bool)
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _bar(span):
    for b in (10, 20, 50, 100, 200, 500):
        if b >= span * 0.35:
            return b
    return 1000


def png(a):
    rng = np.random.default_rng(a.seed + 1)
    key = {}
    for d in sorted(glob.glob(os.path.join(a.out, "*", ""))):
        cname = os.path.basename(os.path.dirname(d))
        if cname not in SHOW:
            continue
        for f in sorted(glob.glob(os.path.join(d, "ex_*.npz"))):
            rp = "A" if rng.random() < 0.5 else "B"
            render(f, f[:-4] + ".png", cname, rp)
            key[f"{cname}/{os.path.basename(f)[:-4]}"] = {"rule_panel": rp}
    json.dump(key, open(os.path.join(a.out, "key.json"), "w"), indent=1)
    n_a = sum(v["rule_panel"] == "A" for v in key.values())
    print(f"[review] {len(key)} images, rule in panel A {n_a}/{len(key)}", flush=True)


# ---------------------------------------------------------------------------- aggregate
YES = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def fleiss(table):
    """Fleiss kappa of an (items, categories) count table with the same number of raters per item."""
    t = np.asarray(table, float)
    t = t[t.sum(1) == t.sum(1).max()]
    if len(t) < 2:
        return None
    n = t.sum(1)[0]
    p = t.sum(0) / t.sum()
    P = ((t ** 2).sum(1) - n) / (n * (n - 1))
    pe = (p ** 2).sum()
    return float((P.mean() - pe) / (1 - pe)) if pe < 1 else None


def aggregate(a):
    key = json.load(open(os.path.join(a.out, "key.json")))
    sel = json.load(open(os.path.join(a.out, "selection.json")))
    out = {}
    for cname, cs in sel["concepts"].items():
        strat = {f"ex_{r['i']:02d}": r["stratum"] for r in cs["examples"]}
        votes = {}  # (example, method) -> [verdict per lens]
        lenses = []
        for f in sorted(glob.glob(os.path.join(a.out, "verdicts", f"{cname}_*.json"))):
            v = json.load(open(f))
            lens = os.path.basename(f)[len(cname) + 1:-5]
            lenses.append(lens)
            for e in v["examples"]:
                ex = f"ex_{int(e['example']):02d}"
                rp = key[f"{cname}/{ex}"]["rule_panel"]
                for panel in ("A", "B"):
                    meth = "rule" if panel == rp else "model"
                    votes.setdefault((ex, meth), {})[lens] = e[f"panel_{panel}"]["verdict"]
        res = {"lenses": lenses, "n_examples": len(strat)}
        for meth in ("rule", "model"):
            for s in list(STRATA) + ["all"]:
                exs = [ex for ex in strat if s in ("all", strat[ex])]
                vals = [YES[v] for ex in exs for v in votes.get((ex, meth), {}).values()]
                res.setdefault(meth, {})[s] = {"n_votes": len(vals),
                                               "support": float(np.mean(vals)) if vals else None}
            per_lens = {ln: [YES[votes[(ex, meth)][ln]] for ex in strat if ln in votes.get((ex, meth), {})]
                        for ln in lenses}
            res[meth]["by_lens"] = {ln: float(np.mean(v)) if v else None for ln, v in per_lens.items()}
            tab = [[sum(v == c for v in votes.get((ex, meth), {}).values()) for c in YES] for ex in strat]
            res[meth]["fleiss_kappa"] = fleiss(tab)
        out[cname] = res
    json.dump(out, open(os.path.join(a.out, "review.json"), "w"), indent=1)
    for c, r in out.items():
        print(f"[review] {c}: rule {r['rule']['all']['support']} model {r['model']['all']['support']} "
              f"kappa {r['rule']['fleiss_kappa']}/{r['model']['fleiss_kappa']}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--dump", required=True)
    p.add_argument("--masks", nargs="+", required=True, help="group=path (kc, rem, alpha, cap, arousal)")
    p.add_argument("--out", required=True)
    p.add_argument("--concepts", nargs="*", default=list(SPECS))
    p.add_argument("--prefer", default="", help="file of night keys (unit/sid) drawn first")
    p.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("png")
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("aggregate")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"pick": pick, "png": png, "aggregate": aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
