"""Examples and counterfactuals of the lexicon concepts for the private explanation page, plus V4 on every
editable concept (the C8 protocol of :mod:`concept_counterfactual` extended beyond the K-complex).

Inputs: the dump (signals, P / E / N are re-encoded here), the reference masks and the per-token masks of
the kept rules saved by ``concept_compose battery --export`` (C1e K-complex / N3 and the eye words, C2b REM
eye movements, C3b alpha in context, C4b CAP, C6b arousal).

**Example choice** (fixed before looking): for each concept and set, the true-positive events (a run of the
lexicon rule overlapping a reference run with IoU >= 0.2, centre epoch at least 2 epochs from the night's
edges) are listed and one is drawn at random with the seed; the second example comes from another set or
site. For the arousal, which fails, one hit, one miss and one false detection in hypnogram wake.

**Edits** (V4, same controls as C8): the P tokens of the event (+-1 token) are replaced by the median P of the
epoch's tokens without the concept (the neighbouring epochs' when too few), requantized and decoded; controls
= the same replacement at random tokens without the concept, and a random direction of the same norm at the
event tokens. ``remove_epoch`` (N3, alpha of wake) edits every token of the epoch where the detector fires
and asks the epoch predicate to cross its threshold. ``tone_swap`` (REM eye movements) puts the EMG epoch
latent E of a lexicon-wake epoch of the same night (atonia < 0.5, alpha > 0.5) into a lexicon-REM epoch;
control = E of another REM epoch; efficacy = the decoded chin EMG leaves atonia (<= 0.5 of the tokens at the
night's lowest level). ``E_swap`` (N3) puts E of an epoch with few slow waves (the predicate axis, negative
in C8). Detectors run on the decoded signals; "before" is the unedited decode, as in C8.

Signals are NSRR / DREAMS / MASS data under DUA: outputs stay on Sofia and in private scratch.

``python -m physioex.train.bin.concept_examples --combo-dir DIR --dump DIR --masks kc=NPZ rem=NPZ ... --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from physioex.explain import concept_refs as CRF
from physioex.explain import concept_rules as CR
from physioex.train.bin.concept_counterfactual import Codec, _tok, detect, reflect_index

T = 60
EP_S = 30.0
IOU_MIN = 0.2
ATONIA_EPOCH = 0.5

#: concept -> mask file group, lexicon key (optional gate), modality edited, reference per set, detector, edits
SPECS = {
    "kc": dict(masks="kc", lex="lex_kc", mod="eeg", ref={"dreams": "k_complex_vis1", "pool": "k_complex_rule"},
               det="kc", edits=("remove", "add")),
    "n3": dict(masks="kc", lex="lex_n3", mod="eeg", ref={"pool": "slow_wave"}, det="delta",
               edits=("remove_epoch", "E_swap")),
    "spindle": dict(masks="kc", lex="word_spindle", mod="eeg",
                    ref={"dreams": "spindle_vis1", "pool": "spindle_rule"}, det="spindle", edits=("remove",)),
    "alpha_W": dict(masks="alpha", lex="lex_alpha_W", mod="eeg", ref={"pool": "alpha"}, det="alpha",
                    edits=("remove_epoch",)),
    "alpha_REM": dict(masks="alpha", lex="lex_alpha_REM", mod="eeg", ref={"pool": "alpha_rem"}, det="alpha",
                      edits=("remove",)),
    "rem_eye": dict(masks="rem", lex="lex_rem", mod="eog", ref={"pool": "rem_eye"}, det="rem_eye",
                    edits=("remove", "tone_swap")),
    "blink": dict(masks="kc", lex="word_blink", gate="wake_ep", mod="eog", ref={"pool": "blink"}, det="blink",
                  edits=("remove",)),
    "cap_a1": dict(masks="cap", lex="lex_a1", mod="eeg", ref={"pool": "cap_a1"}, det="cap_a1", edits=("remove",)),
    "arousal": dict(masks="arousal", lex="lex_arousal", mod="eeg",
                    ref={"mass_ss01": "arousal_ann", "pool": "arousal_ann"}, det=None, edits=()),
}
EPOCH_THR = {"delta": (CR.N3_SWA, "below"), "alpha": (CR.W_ALPHA, "below")}  # remove_epoch: predicate must cross


# ---------------------------------------------------------------------------- data
class Night:
    """One dumped night: signals (uV), references, scale; encodings cached per modality."""

    def __init__(self, dump, key, sc, fs):
        self.key = key
        self.z = np.load(os.path.join(dump, f"{key}.npz"))
        from physioex.train.bin.concept_scan import load_refs

        self.refs = load_refs(os.path.join(dump, "refs", f"{key}.npz"))
        self.sc, self.fs = sc.get(key, {}), fs
        self.n = len(self.z["x_eeg"])
        self._enc = {}

    def has(self, mod):
        return f"x_{mod}" in self.z.files and mod in self.sc

    def x(self, mod):
        return self.z[f"x_{mod}"].astype(np.float32)

    def uv(self, mod, a):
        s = self.sc[mod]
        return a * float(s["scale"]) + float(s.get("median", 0.0))

    def enc(self, codec, mod, epochs):
        """N, E, P of ``epochs`` (cached)."""
        need = [e for e in epochs if (mod, e) not in self._enc]
        if need:
            x = self.x(mod)
            xp = x[reflect_index(len(x), codec.half)]
            z = codec.encode(xp, np.asarray(need) + codec.half)
            for i, e in enumerate(need):
                self._enc[(mod, e)] = {k: z[k][i:i + 1] for k in "NEP"}
        import torch

        return {k: torch.cat([self._enc[(mod, e)][k] for e in epochs]) for k in "NEP"}


def det_tokens(kind, u, fs, eeg=None, p5=None):
    """Token mask (T * epochs,) of detector ``kind`` on uV ``u`` (one or more epochs concatenated)."""
    n_tok = int(round(len(u) / fs / 0.5))
    if kind in ("kc", "delta", "spindle", "alpha"):
        if len(u) == int(EP_S * fs):
            return detect(u, fs)[kind]
        return np.concatenate([detect(v, fs)[kind] for v in u.reshape(-1, int(EP_S * fs))])
    if kind in ("rem_eye", "blink"):
        return _tok(CRF.detect_eye_movements(u, fs, ref=eeg)[kind], len(u), fs)[:n_tok]
    if kind == "cap_a1":
        return _tok(CRF.detect_cap_a(u, fs)["a1"], len(u), fs)[:n_tok]
    if kind == "atonia":
        x = CRF.bandpass(u, 10, None, fs)
        tok = int(0.5 * fs)
        trms = np.sqrt((x[:n_tok * tok].reshape(n_tok, tok) ** 2).mean(1))
        return trms <= 1.5 * p5
    raise ValueError(kind)


def night_p5(night):
    """The night's 5th percentile of the chin token RMS (the atonia level of :func:`concept_refs.emg_masks`)."""
    x = CRF.bandpass(night.uv("emg", night.x("emg")).ravel(), 10, None, night.fs["emg"])
    tok = int(0.5 * night.fs["emg"])
    n = len(x) // tok
    return float(np.percentile(np.sqrt((x[:n * tok].reshape(n, tok) ** 2).mean(1)), 5))


# ---------------------------------------------------------------------------- masks and events
class Masks:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.z = z
        self.keys = [str(k) for k in z["keys"]]
        ne = np.bincount(z["night_ep"], minlength=len(self.keys))
        self.ep0 = np.r_[0, np.cumsum(ne)]
        self.set_n = {k: str(z["set_ep"][self.ep0[i]]) for i, k in enumerate(self.keys) if ne[i]}

    def tok(self, name, key):
        i = self.keys.index(key)
        return self.z[name][self.ep0[i] * T:self.ep0[i + 1] * T]

    def ep(self, name, key):
        i = self.keys.index(key)
        return self.z[name][self.ep0[i]:self.ep0[i + 1]]

    def has(self, name):
        return name in self.z.files


def lex_mask(M, spec, key):
    m = np.asarray(M.tok(spec["lex"], key), bool)
    if spec.get("gate"):
        m = m & np.repeat(np.asarray(M.ep(spec["gate"], key), bool), T)
    return m


def events(lex, ref, n_ep, kind="tp"):
    """(start, end, centre epoch) of the runs of ``lex`` (tp / fp) or of ``ref`` (fn), centre >= 2 epochs inside."""
    out = []
    src, other = (ref, lex) if kind == "fn" else (lex, ref)
    for a, b in CRF.runs(src):
        e = (a + b - 1) // 2 // T
        if not 2 <= e < n_ep - 2:
            continue
        hit = other[a:b].any()
        if kind == "tp":
            if not hit:
                continue
            ra = [(c, d) for c, d in CRF.runs(ref[max(0, a - 120):b + 120]) if c + max(0, a - 120) < b and d + max(0, a - 120) > a]
            lo = min([a] + [c + max(0, a - 120) for c, _ in ra])
            hi = max([b] + [d + max(0, a - 120) for _, d in ra])
            inter = (src[lo:hi] & other[lo:hi]).sum()
            union = (src[lo:hi] | other[lo:hi]).sum()
            if union == 0 or inter / union < IOU_MIN:
                continue
        elif hit:
            continue
        out.append((int(a), int(b), int(e)))
    return out


# ---------------------------------------------------------------------------- edits
def _silent_median(P_ctx, ev_ctx, silent_ctx):
    import torch

    s = torch.from_numpy(silent_ctx & ~ev_ctx)
    return P_ctx.reshape(-1, P_ctx.shape[-1])[s].median(0).values if int(s.sum()) >= 3 else None


def edit_set(P_c, ev_c, silent_c, rng):
    """P of the ctx epochs (E, T, d) and token masks (E * T,): remove / ctrl_silent / ctrl_random tensors."""
    import torch

    flat = P_c.reshape(-1, P_c.shape[-1])
    med = _silent_median(P_c, ev_c, silent_c)
    if med is None:
        return None
    out = {}
    p1 = flat.clone()
    p1[torch.from_numpy(ev_c)] = med
    out["remove"] = p1
    cand = np.flatnonzero(silent_c & ~ev_c)
    if len(cand) >= ev_c.sum():
        cs = np.zeros_like(ev_c)
        cs[rng.choice(cand, int(ev_c.sum()), replace=False)] = True
        p2 = flat.clone()
        p2[torch.from_numpy(cs)] = med
        out["ctrl_silent"] = p2
    d = med - flat[torch.from_numpy(ev_c)]
    r = torch.randn_like(d)
    r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-9) * d.norm(dim=1, keepdim=True)
    p3 = flat.clone()
    p3[torch.from_numpy(ev_c)] = flat[torch.from_numpy(ev_c)] + r
    out["ctrl_random"] = p3
    return {k: v.reshape(P_c.shape) for k, v in out.items()}


def decode_ctx(codec, z, P=None, E=None, requant=False):
    p = z["P"] if P is None else P
    if requant:
        p = codec.requant(p)
    return codec.decode(z["N"], z["E"] if E is None else E, p)  # (E, samples)


def run_edit(codec, night, spec, a, b, e, rng, eeg_codec=None, keep_signals=False):
    """One event: base decode, remove (+ controls) and the concept's extra edits. Returns (row, signals)."""
    mod, kind = spec["mod"], spec["det"]
    fs = night.fs[mod]
    ctx = list(range(e - 2, e + 3))
    z = night.enc(codec, mod, ctx)  # the 5 context epochs; CAP phase A may span them all (60 s background)
    tok0 = (e - 2) * T
    base = decode_ctx(codec, z)  # (5, spe)
    uv = lambda w: night.uv(mod, w)
    eeg_uv = night.uv("eeg", night.x("eeg")[ctx]) if mod == "eog" else None
    detf = lambda w: (det_tokens(kind, uv(w).ravel(), fs, eeg=None if eeg_uv is None else eeg_uv.ravel())
                      if kind in ("rem_eye", "blink", "cap_a1") else det_tokens(kind, uv(w).ravel(), fs))
    d0 = detf(base)
    ev = np.zeros(5 * T, bool)
    lo, hi = max(a - tok0 - 1, 0), min(b - tok0 + 1, 5 * T)
    ev[lo:hi] = True
    if "remove_epoch" in spec["edits"]:  # every token of the centre epoch where the detector fires
        ev = np.zeros(5 * T, bool)
        ev[2 * T:3 * T] = d0[2 * T:3 * T]
    if kind != "cap_a1":  # other concepts: the edit stays in the centre epoch
        ev[:2 * T] = ev[3 * T:] = False
    if not ev.any():
        return None, None
    silent = ~d0
    row = {"key": night.key, "epoch": int(e), "a": int(a - tok0), "b": int(b - tok0), "n_tok": int(ev.sum())}
    row["det_before"] = float(d0[ev].mean())
    if "remove_epoch" in spec["edits"]:
        row["frac_before"] = float(d0[2 * T:3 * T].mean())
    sig = {"ctx_epochs": np.array(ctx), "orig": uv(night.x(mod)[ctx]).astype(np.float32),
           "base": uv(base).astype(np.float32), "det_base": d0, "ev": ev} if keep_signals else None
    eds = edit_set(z["P"], ev, silent, rng)  # median of the 5 epochs' tokens without the concept
    if eds is None:
        return None, None
    for k, pe in eds.items():
        w = decode_ctx(codec, z, P=pe, requant=True)
        d1 = detf(w)
        row[f"{k}_det_after"] = float(d1[ev].mean())
        row[f"{k}_gone"] = bool(d0[ev].any() and not d1[ev].any())
        if "remove_epoch" in spec["edits"]:
            row[f"{k}_frac_after"] = float(d1[2 * T:3 * T].mean())
        diff = (w - base).reshape(-1)
        tk = diff.reshape(5 * T, -1)
        row[f"{k}_rms_in"] = float(np.sqrt((tk[ev] ** 2).mean()))
        row[f"{k}_rms_out"] = float(np.sqrt((tk[~ev] ** 2).mean()))
        if keep_signals:
            sig[f"ed_{k}"] = uv(w).astype(np.float32)
            sig[f"det_{k}"] = d1
    if "E_swap" in spec["edits"]:  # predicate axis: E of the ctx epoch with the fewest slow waves
        fr = [d0[i * T:(i + 1) * T].mean() for i in range(5)]
        j = int(np.argmin(fr))
        if fr[j] < CR.N3_SWA and j != 2:
            E = z["E"].clone()
            E[2] = z["E"][j]
            w = decode_ctx(codec, z, E=E)
            d1 = detf(w)
            row["E_swap_frac_after"] = float(d1[2 * T:3 * T].mean())
            row["E_swap_donor_frac"] = float(fr[j])
            if keep_signals:
                sig["ed_E_swap"] = uv(w).astype(np.float32)
                sig["det_E_swap"] = d1
    return row, sig


def tone_swap(codec_emg, night, M, e, rng, keep_signals=False):
    """REM -> wake on the predicate axis: the EMG epoch latent E of a lexicon-wake epoch of the night."""
    aton = np.asarray(M.ep("aton_ep", night.key), float)
    alpha = np.asarray(M.ep("alpha_ep", night.key), float)
    wake = np.flatnonzero((aton < ATONIA_EPOCH) & (alpha > CR.W_ALPHA))
    rem = np.flatnonzero((aton > ATONIA_EPOCH) & (alpha <= CR.W_ALPHA))
    rem = rem[rem != e]
    if not len(wake) or not len(rem) or not night.has("emg"):
        return None, None
    fs = night.fs["emg"]
    if not hasattr(night, "_p5"):
        night._p5 = night_p5(night)
    jw, jr = int(rng.choice(wake)), int(rng.choice(rem))
    z = night.enc(codec_emg, "emg", [e, jw, jr])
    uv = lambda w: night.uv("emg", w)
    base = codec_emg.decode(z["N"][:1], z["E"][:1], z["P"][:1])
    a0 = det_tokens("atonia", uv(base[0]), fs, p5=night._p5).mean()
    row = {"key": night.key, "epoch": int(e), "donor_wake": jw, "donor_rem": jr, "atonia_before": float(a0)}
    sig = {"orig_emg": uv(night.x("emg")[e]).astype(np.float32), "base_emg": uv(base[0]).astype(np.float32)} if keep_signals else None
    for k, j in (("tone_swap", 1), ("ctrl_tone", 2)):
        w = codec_emg.decode(z["N"][:1], z["E"][j:j + 1], z["P"][:1])[0]
        row[f"{k}_atonia_after"] = float(det_tokens("atonia", uv(w), fs, p5=night._p5).mean())
        row[f"{k}_left_atonia"] = bool(a0 > ATONIA_EPOCH and row[f"{k}_atonia_after"] <= ATONIA_EPOCH)
        if keep_signals:
            sig[f"ed_{k}_emg"] = uv(w).astype(np.float32)
    return row, sig


# ---------------------------------------------------------------------------- summary
def summarize(rows, epoch_kind=None):
    from scipy.stats import wilcoxon

    rows = [r for r in rows if r and r.get("det_before", 0) > 0]
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    if epoch_kind:  # remove_epoch: the epoch predicate must cross its threshold
        thr = EPOCH_THR[epoch_kind][0]
        ok = [r for r in rows if r["frac_before"] >= thr]
        out["n_epoch_pred"] = len(ok)
        for k in ("remove", "ctrl_silent", "ctrl_random"):
            v = [r[f"{k}_frac_after"] < thr for r in ok if f"{k}_frac_after" in r]
            out[f"{k}_crosses"] = float(np.mean(v)) if v else None
    for k in ("remove", "ctrl_silent", "ctrl_random"):
        v = [r[f"{k}_gone"] for r in rows if f"{k}_gone" in r]
        out[f"{k}_gone"] = float(np.mean(v)) if v else None
    a = np.array([r["det_before"] - r["remove_det_after"] for r in rows])
    c = np.array([r["det_before"] - r["ctrl_random_det_after"] for r in rows])
    if len(rows) > 5 and np.any(a != c):
        out["wilcoxon_p_remove_vs_random"] = float(wilcoxon(a, c).pvalue)
    out["locality_rms_out_over_in"] = float(np.median([r["remove_rms_out"] / (r["remove_rms_in"] + 1e-9) for r in rows]))
    if any("E_swap_frac_after" in r for r in rows):
        v = [r["E_swap_frac_after"] for r in rows if "E_swap_frac_after" in r]
        out["E_swap"] = {"n": len(v), "still_n3": float(np.mean([x >= CR.N3_SWA for x in v]))}
    return out


def summarize_tone(rows):
    rows = [r for r in rows if r and r["atonia_before"] > ATONIA_EPOCH]
    if not rows:
        return {"n": 0}
    return {"n": len(rows), "efficacy": float(np.mean([r["tone_swap_left_atonia"] for r in rows])),
            "ctrl": float(np.mean([r["ctrl_tone_left_atonia"] for r in rows])),
            "atonia_after_median": float(np.median([r["tone_swap_atonia_after"] for r in rows])),
            "ctrl_atonia_after_median": float(np.median([r["ctrl_tone_atonia_after"] for r in rows]))}


# ---------------------------------------------------------------------------- command
def _site(meta, key):
    return meta["nights"][key]["unit"]


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--combo-dir", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--masks", nargs="+", required=True, help="group=path (kc, rem, alpha, cap, arousal)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concepts", nargs="*", default=list(SPECS))
    ap.add_argument("--n-v4", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    sc = json.load(open(os.path.join(a.dump, "scale.json")))
    fs = {m: float(v) for m, v in meta["fs"].items()}
    masks = {g: Masks(p) for g, p in (s.split("=", 1) for s in a.masks)}
    codecs = {}
    nights = {}

    def codec(mod):
        if mod not in codecs:
            codecs[mod] = Codec(a.combo_dir, dev, mod)
        return codecs[mod]

    def night(key):
        if key not in nights:
            if len(nights) > 6:
                nights.pop(next(iter(nights)))
            nights[key] = Night(a.dump, key, sc, fs)
        return nights[key]

    refcache = {}

    def ref_of(key, name):
        """Reference token mask of a night (None if absent); only the masks in use are kept."""
        if (key, name) not in refcache:
            from physioex.train.bin.concept_scan import load_refs

            rf = load_refs(os.path.join(a.dump, "refs", f"{key}.npz"))
            refcache[(key, name)] = np.asarray(rf[name], bool) if name in rf else None
        return refcache[(key, name)]

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    v4, index = {}, []
    for cname in a.concepts:
        spec = SPECS[cname]
        M = masks.get(spec["masks"])
        if M is None or not M.has(spec["lex"]):
            print(f"[ex] {cname}: masks missing, skipped", flush=True)
            continue
        cands = {}
        for key in M.keys:
            st = M.set_n.get(key)
            ref = spec["ref"].get(st)
            r = ref_of(key, ref) if ref else None
            if r is None:
                continue
            lx = lex_mask(M, spec, key)
            r = r[:len(lx)]
            n_ep = len(lx) // T
            for kind in (("tp", "fn", "fp") if cname == "arousal" else ("tp",)):
                for ev in events(lx, r, n_ep, kind):
                    cands.setdefault((st, kind), []).append((key, *ev))
        if cname == "arousal":  # the false detection is shown in hypnogram wake
            ycache = {}
            fp = []
            for key, s_, e_, ep in cands.get(("mass_ss01", "fp"), []) + cands.get(("pool", "fp"), []):
                y = ycache.setdefault(key, M.z["y_ep"][M.ep0[M.keys.index(key)]:M.ep0[M.keys.index(key) + 1]])
                if y[ep] == 0:
                    fp.append((key, s_, e_, ep))
            cands[("any", "fp_wake")] = fp
        print(f"[ex] {cname}: " + ", ".join(f"{k[0]}/{k[1]} {len(v)}" for k, v in cands.items()), flush=True)

        # examples
        picks = []
        if cname == "arousal":
            for k in (("mass_ss01", "tp"), ("mass_ss01", "fn"), ("any", "fp_wake")):
                if cands.get(k):
                    picks.append((k, cands[k][int(rng.integers(len(cands[k])))]))
        else:
            sets = sorted({k[0] for k in cands}, key=lambda s: (s != "dreams", s))
            used_sites = set()
            for s_ in sets:
                c = cands[(s_, "tp")]
                order = rng.permutation(len(c))
                for i in order:
                    site = _site(meta, c[i][0])
                    if site not in used_sites:
                        picks.append(((s_, "tp"), c[i]))
                        used_sites.add(site)
                        break
                if len(picks) >= 2:
                    break
            if len(picks) < 2 and sets:  # one set only: second example from another site of it
                c = cands[(sets[0], "tp")]
                for i in rng.permutation(len(c)):
                    if _site(meta, c[i][0]) not in used_sites:
                        picks.append(((sets[0], "tp"), c[i]))
                        break
        for n_ex, ((s_, kind), (key, s0, s1, ep)) in enumerate(picks):
            nt = night(key)
            ctx = list(range(ep - 2, ep + 3))
            ex = {"concept": cname, "set": s_, "kind": kind, "key": key, "site": _site(meta, key), "epoch": ep,
                  "a": s0 - (ep - 2) * T, "b": s1 - (ep - 2) * T}
            arrays = {"ctx_epochs": np.array(ctx), "y_ep": M.z["y_ep"][M.ep0[M.keys.index(key)] + ep - 2:M.ep0[M.keys.index(key)] + ep + 3]}
            for m in ("eeg", "eog", "emg"):
                if nt.has(m):
                    arrays[f"x_{m}"] = nt.uv(m, nt.x(m)[ctx]).astype(np.float32)
            sl = slice((ep - 2) * T, (ep + 3) * T)
            arrays["lex"] = lex_mask(M, spec, key)[sl]
            arrays["ref"] = ref_of(key, spec["ref"][M.set_n[key]])[sl]
            for nm in M.z.files:
                if nm.startswith(("word_", "lex_", "ora_")) and M.z[nm].shape[0] == M.z["night_ep"].shape[0] * T:
                    arrays[f"m_{nm}"] = M.tok(nm, key)[sl]
                elif nm.endswith("_ep") and nm not in ("night_ep", "y_ep", "set_ep"):
                    arrays[f"e_{nm}"] = M.ep(nm, key)[ep - 2:ep + 3]
                elif nm in ("fast_lex", "fast_true", "lo_lex", "lo_true", "hi_lex", "hi_true"):
                    arrays[f"f_{nm}"] = M.tok(nm, key)[sl]
            if spec["det"]:
                row, sig = run_edit(codec(spec["mod"]), nt, spec, s0, s1, ep, rng, keep_signals=True)
                if row:
                    ex["edit"] = row
                    arrays.update({f"cf_{k}": v for k, v in sig.items()})
                if "tone_swap" in spec["edits"]:
                    row, sig = tone_swap(codec("emg"), nt, masks["rem"], ep, rng, keep_signals=True)
                    if row:
                        ex["tone_swap"] = row
                        arrays.update({f"ts_{k}": v for k, v in sig.items()})
            p = os.path.join(a.out, f"{cname}_{n_ex}.npz")
            np.savez_compressed(p, meta=json.dumps(ex), **arrays)
            index.append(ex)
            print(f"[ex] {cname} example {n_ex}: {s_} {kind} {key} ep {ep} {time.time() - t0:.0f}s", flush=True)

        # V4 on up to n_v4 true positives per set
        if spec["det"]:
            v4[cname] = {}
            for (s_, kind), c in cands.items():
                if kind != "tp" or not c:
                    continue
                rows, trows = [], []
                for i in rng.permutation(len(c))[:a.n_v4]:
                    key, s0, s1, ep = c[i]
                    row, _ = run_edit(codec(spec["mod"]), night(key), spec, s0, s1, ep, rng)
                    rows.append(row)
                    if "tone_swap" in spec["edits"]:
                        trows.append(tone_swap(codec("emg"), night(key), masks["rem"], ep, rng)[0])
                ek = spec["det"] if "remove_epoch" in spec["edits"] else None
                v4[cname][s_] = {"remove": summarize(rows, ek), "n_candidates": len(c)}
                if trows:
                    v4[cname][s_]["tone_swap"] = summarize_tone(trows)
                print(f"[ex] {cname} V4 {s_}: {v4[cname][s_]} {time.time() - t0:.0f}s", flush=True)
    from physioex.train.bin.concept_compose import _jsonable

    json.dump(_jsonable({"v4": v4, "examples": index, "seed": a.seed, "n_v4": a.n_v4,
                         "minutes": (time.time() - t0) / 60}), open(os.path.join(a.out, "v4_extra.json"), "w"), indent=1)
    print(f"[ex] wrote {a.out} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
