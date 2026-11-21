"""Manual-guided visual annotator (M2, xmai-lab/sleep-lens#14): Qwen vision reads rendered PSG epochs.

Each selected 30 s epoch of a dump (``concept_scan`` layout) is drawn like a textbook PSG page (central EEG,
EOG, chin EMG in uV, 1 s grid, six 5 s segments labelled A-F, scored stage in the title). The vLLM endpoint of
the Qwen job on Sofia gets the image and the manual definitions of the pilot concepts and returns, per segment,
the concepts it sees from a closed vocabulary. Segments, not instants: Qwen places events badly in time.
The rendering and the calls run on Sofia only (the signals never leave it); the API key comes from the
environment (``QWEN_KEY``), never from a file.

Epoch selection (``--pos``, ``--neg`` per concept and night): epochs where the M1 mask of the concept fires
(``DUMP/m1``) and random epochs of the stages where the concept is defined, so agreement is measured on an
enriched sample; the stratum of every epoch is stored.

    QWEN_KEY=... python -m physioex.train.bin.manual_visual_annot annotate --dump DUMP --endpoint http://NODE:8000
        [--groups mass_ss1,dreams] [--pos 2 --neg 2] [--workers 8]          # -> DUMP/m2/annot.jsonl (resumable)
    python -m physioex.train.bin.manual_visual_annot agree --dump DUMP      # -> DUMP/m2/agree.json
    python -m physioex.train.bin.manual_visual_annot render --dump DUMP --ids ids.txt --out DIR   # review pages
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import urllib.request

import numpy as np

from physioex.explain.concept_refs import T, bandpass

SEG_S = 5.0
SEGS = "ABCDEF"
TOK_PER_SEG = int(SEG_S / 0.5)
STAGE = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
STAGE_ID = {v: k for k, v in STAGE.items()}
#: vocabulary -> (M1 mask, stages where random negatives are drawn, manual definition with source)
VOCAB = {
    "arousal": ("arousal_m1", (1, 2, 3, 4),
                "abrupt shift of EEG frequency including alpha, theta and/or frequencies greater than 16 Hz (but "
                "not spindles) that lasts at least 3 s, with at least 10 s of stable sleep preceding the change; "
                "in stage R a concurrent increase in chin EMG lasting at least 1 s is also required "
                "[Fundamentals ch3; AASM]"),
    "k_complex": ("kc_m1", (2, 3),
                  "well-delineated negative sharp wave (upward deflection) immediately followed by a positive "
                  "component (downward deflection), standing out from the background EEG, total duration >= 0.5 s "
                  "[Kryger atlas 2024; AASM]"),
    "vertex_sharp_wave": ("vertex_m1", (1, 2),
                          "sharply contoured wave with duration < 0.5 s, maximal over the central region and "
                          "distinguishable from the background activity [Fundamentals ch4; AASM]"),
    "sawtooth_wave": ("sawtooth_m1", (4,),
                      "trains of triangular, often serrated waves of 2-6 Hz, highest in central derivations, "
                      "typical of stage R [Fundamentals ch1]"),
    "alpha_intrusion": ("alpha_burst_m1", (2, 3),
                        "prominent alpha activity (8-13 Hz) during NREM sleep, as bursts shorter than 3 s that "
                        "are not an arousal [Fundamentals ch4]"),
    "slow_eye_movement": ("sem_m1", (0, 1),
                          "conjugate, reasonably regular, sinusoidal eye movements with an initial deflection "
                          "lasting more than 500 ms, in drowsiness and N1 [Fundamentals ch1; AASM]"),
    "artifact": (None, (), "movement, electrode or other non-physiological signal that hides the EEG"),
}
LANES = (("eeg", "EEG (central)", 0.3, 35.0, 75.0), ("eog", "EOG", 0.3, 35.0, 150.0),
         ("emg", "Chin EMG", 10.0, 45.0, None))  # modality, label, band (Hz), half lane height in uV (None: per night)

SCHEMA = {"type": "object", "properties": {
    "segments": {"type": "array", "minItems": 6, "maxItems": 6, "items": {"type": "object", "properties": {
        "segment": {"enum": list(SEGS)},
        "concepts": {"type": "array", "items": {"enum": list(VOCAB)}},
        "confidence": {"enum": ["low", "medium", "high"]}}, "required": ["segment", "concepts", "confidence"]}},
    "note": {"type": "string"}}, "required": ["segments", "note"]}

PROMPT = """You are an expert polysomnography scorer. The image is one 30 s epoch of a sleep recording scored
as stage {stage}: central EEG, EOG and chin EMG, in microvolts (scale bars on the left), thin grid lines every
1 s, thick lines every 5 s; the six 5 s segments are labelled A to F at the top.

For each segment list the events below that you can see in it, using only these definitions:
{defs}

Rules: list an event only in the segments where you see its waveform in the traces; the stage alone is never
a reason to list an event; a K complex is a single wave standing out of the background, not one wave of a
train of slow waves; an empty list when none is visible; confidence = how sure you are of that segment.
note = one short sentence on the epoch.
Return JSON only."""


def _defs():
    return "\n".join(f"- {k}: {v[2]}" for k, v in VOCAB.items())


def _signals(z, sc, key, mods=("eeg", "eog", "emg")):
    out = {}
    for m in mods:
        if f"x_{m}" in z.files and m in sc.get(key, {}):
            o = sc[key][m]
            out[m] = z[f"x_{m}"].astype(np.float32) * np.float32(o["scale"]) + np.float32(o["median"])
    return out


def render(sig, fs, i, stage, emg_half=None, hl=None):
    """PNG bytes of epoch ``i`` (the neighbours only feed the filters); ``hl``: segment index to shade."""
    from matplotlib.figure import Figure  # no pyplot: renders run in threads

    fig = Figure(figsize=(16, 6.5), dpi=100)
    axes = fig.subplots(len(LANES), 1, sharex=True, gridspec_kw={"hspace": 0.05})
    n_ep = int(30 * fs)
    t = np.arange(n_ep) / fs
    for ax, (m, label, lo, hi, half) in zip(axes, LANES):
        ax.set_yticks([])
        if hl is not None:
            ax.axvspan(hl * SEG_S, (hl + 1) * SEG_S, color="gold", alpha=0.25, zorder=0)
        for s in np.arange(0, 31, 1.0):
            ax.axvline(s, color="0.85" if s % 5 else "0.45", lw=0.5 if s % 5 else 1.2, zorder=0)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        if m not in sig:
            ax.set_ylabel(label + "\n(n/a)", rotation=0, ha="right", va="center", fontsize=11)
            continue
        x = sig[m]
        lo_i, hi_i = max(0, i - 1), min(len(x), i + 2)
        seg = bandpass(x[lo_i:hi_i].reshape(-1).astype(np.float64), lo, min(hi, fs / 2 - 1), fs)
        seg = seg[(i - lo_i) * n_ep:(i - lo_i + 1) * n_ep]
        h = half if half is not None else (emg_half or 50.0)
        ax.plot(t, np.clip(seg, -1.2 * h, 1.2 * h), color="k", lw=0.6)
        ax.set_ylim(-1.2 * h, 1.2 * h)
        bar = 50.0 if h <= 100 else 100.0
        ax.plot([-0.4, -0.4], [-bar / 2, bar / 2], color="tab:red", lw=2, clip_on=False)
        ax.text(-0.5, 0, f"{bar:.0f} uV", ha="right", va="center", fontsize=9, color="tab:red")
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11, labelpad=55)
    for k, c in enumerate(SEGS):
        axes[0].text(k * SEG_S + SEG_S / 2, axes[0].get_ylim()[1], c, ha="center", va="bottom", fontsize=14,
                     fontweight="bold")
    axes[-1].set_xlim(0, 30)
    axes[-1].set_xticks(np.arange(0, 31, 5))
    axes[-1].set_xlabel("s")
    fig.suptitle(f"30 s epoch, stage {stage}", fontsize=13, y=0.99)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()


def call(endpoint, key, png, stage, model="qwen3.8-27b", tries=3):
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
               {"type": "text", "text": PROMPT.format(stage=stage, defs=_defs())}]
    body = {"model": model, "temperature": 0.0, "max_tokens": 1500, "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "epoch", "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    err = ""
    for a in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(json.loads(r.read())["choices"][0]["message"]["content"]), ""
        except Exception as e:  # noqa: BLE001 - retried, then reported in the output row
            err = repr(e)[:200]
            time.sleep(5 * (a + 1))
    return None, err


def _load_mask(path):
    from physioex.train.bin.concept_scan import load_refs

    return load_refs(path) if os.path.exists(path) else {}


def select(dump, meta, groups, pos, neg, seed):
    """``[(key, epoch, stratum)]``: per night and concept, ``pos`` epochs where M1 fires and ``neg`` random
    epochs of the concept's stages (strata ``pos:<concept>`` / ``rnd:<concept>``; an epoch picked twice keeps
    its first stratum)."""
    from physioex.train.bin.concept_scan import _group

    rng = np.random.default_rng(seed)
    out, seen = [], set()
    for k, v in sorted(meta["nights"].items()):
        if groups and _group(v["unit"]) not in groups:
            continue
        y = np.load(os.path.join(dump, f"{k}.npz"))["y"].astype(np.int8)
        m1 = _load_mask(os.path.join(dump, "m1", f"{k}.npz"))
        for c, (mask, stages, _) in VOCAB.items():
            if mask is None:
                continue
            cand_pos = np.flatnonzero(m1[mask].reshape(-1, T).any(1)) if mask in m1 else np.array([], int)
            cand_rnd = np.flatnonzero(np.isin(y, stages))
            for tag, cand, n in (("pos", cand_pos, pos), ("rnd", cand_rnd, neg)):
                for i in rng.permutation(cand)[:n]:
                    if (k, int(i)) not in seen:
                        seen.add((k, int(i)))
                        out.append((k, int(i), f"{tag}:{c}"))
    return out


def annotate(a):
    import concurrent.futures as cf

    from physioex.train.bin.concept_scan import _group

    key = os.environ.get("QWEN_KEY", "")
    if not key:
        raise SystemExit("QWEN_KEY not set")
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    sc = json.load(open(os.path.join(a.dump, "scale.json")))
    groups = set(filter(None, a.groups.split(",")))
    items = select(a.dump, meta, groups, a.pos, a.neg, a.seed)
    out_path = os.path.join(a.dump, "m2", "annot.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        done = {(r["key"], r["epoch"]) for r in map(json.loads, open(out_path)) if r.get("ann")}
    items = [it for it in items if (it[0], it[1]) not in done][: a.limit or None]
    print(f"[m2] {len(items)} epochs to annotate ({len(done)} done)", flush=True)
    fs = int(meta["fs"]["eeg"])
    by_night = {}
    for it in items:
        by_night.setdefault(it[0], []).append(it)

    def job(k):  # one night per task: the signals are loaded once, epochs are annotated in order
        z = np.load(os.path.join(a.dump, f"{k}.npz"))
        sig, y = _signals(z, sc, k), z["y"].astype(np.int8)
        emg_half = float(np.percentile(np.abs(sig["emg"]), 95) * 2) if "emg" in sig else None
        res = []
        for _, i, stratum in by_night[k]:
            stage = STAGE[int(y[i])]
            t0 = time.time()
            ann, err = call(a.endpoint, key, render(sig, fs, i, stage, emg_half), stage)
            res.append({"key": k, "group": _group(meta["nights"][k]["unit"]), "epoch": i, "stage": stage,
                        "stratum": stratum, "ann": ann, "error": err, "s": round(time.time() - t0, 1)})
        return res

    t0 = time.time()
    n_err = n = 0
    with open(out_path, "a") as fo, cf.ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(job, by_night):
            for r in res:
                fo.write(json.dumps(r) + "\n")
                n_err += r["ann"] is None
            fo.flush()
            n += len(res)
            print(f"[m2] {n}/{len(items)} errors {n_err}, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[m2] done, errors {n_err}, {(time.time() - t0) / 60:.1f} min -> {out_path}\nDONE", flush=True)


def _kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not len(a):
        return float("nan")
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def _seg_pos(mask, i):
    """Per-segment positives of a token mask on epoch ``i``."""
    return mask[i * T:(i + 1) * T].reshape(len(SEGS), TOK_PER_SEG).any(1)


def agree(a):
    """Segment-level agreement (Cohen kappa, M2 precision/recall taking M1 or the human as reference) of M2
    vs M1 per concept, and of M1 / M2 vs the human references where they exist, with the inter-scorer kappa
    of the DREAMS K complexes on the same segments as ceiling."""
    rows = [r for r in map(json.loads, open(os.path.join(a.dump, "m2", "annot.jsonl"))) if r.get("ann")]
    human = {"arousal": ("arousal_ann",), "k_complex": ("k_complex_vis1", "k_complex_vis2")}
    acc = {}  # (concept, pair, group) -> (list a, list b)
    m1c, rfc = {}, {}
    for r in rows:
        k, i, g = r["key"], r["epoch"], r["group"]
        if k not in m1c:
            m1c = {k: _load_mask(os.path.join(a.dump, "m1", f"{k}.npz"))}
            rfc = {k: _load_mask(os.path.join(a.dump, "refs", f"{k}.npz"))}
        m1, rf = m1c[k], rfc[k]
        seen = {s["segment"]: set(s["concepts"]) for s in r["ann"]["segments"]}
        for c, (mask, _, _) in VOCAB.items():
            if mask is None:
                continue
            m2 = np.array([c in seen.get(s, set()) for s in SEGS])
            if mask in m1:
                p1 = _seg_pos(m1[mask], i)
                for gg in (g, "all"):
                    x = acc.setdefault((c, "m1_vs_m2", gg), ([], []))
                    x[0].extend(p1)
                    x[1].extend(m2)
            for h in human.get(c, ()):
                if h not in rf:
                    continue
                ph = _seg_pos(rf[h], i)
                for name, pr in (("m2_vs_" + h, m2), ("m1_vs_" + h, _seg_pos(m1[mask], i) if mask in m1 else None)):
                    if pr is None:
                        continue
                    x = acc.setdefault((c, name, g), ([], []))
                    x[0].extend(ph)
                    x[1].extend(pr)
            if c == "k_complex" and "k_complex_vis1" in rf and "k_complex_vis2" in rf:
                x = acc.setdefault((c, "vis1_vs_vis2", g), ([], []))
                x[0].extend(_seg_pos(rf["k_complex_vis1"], i))
                x[1].extend(_seg_pos(rf["k_complex_vis2"], i))
    res = {}
    for (c, pair, g), (ref, pr) in sorted(acc.items()):
        ref, pr = np.array(ref, bool), np.array(pr, bool)
        tp = int((ref & pr).sum())
        v = {"kappa": _kappa(ref, pr), "precision": tp / max(1, pr.sum()), "recall": tp / max(1, ref.sum()),
             "n_seg": int(len(ref)), "n_ref": int(ref.sum()), "n_pred": int(pr.sum())}
        res.setdefault(c, {})[f"{pair}@{g}"] = v
        print(f"[agree] {c:18s} {pair:24s} {g:10s} kappa {v['kappa']:.2f} P {v['precision']:.2f} "
              f"R {v['recall']:.2f} ({v['n_pred']}/{v['n_ref']} of {v['n_seg']} segments)", flush=True)
    strata = {}
    for r in rows:
        strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1
    out = os.path.join(a.dump, "m2", "agree.json")
    json.dump({"epochs": len(rows), "strata": strata, "agree": res}, open(out, "w"), indent=1, default=float)
    print(f"[agree] -> {out}\nDONE", flush=True)


def render_ids(a):
    """PNG files of the epochs listed in ``--ids`` (lines ``key epoch``), for the blind review pages."""
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    sc = json.load(open(os.path.join(a.dump, "scale.json")))
    fs = int(meta["fs"]["eeg"])
    os.makedirs(a.out, exist_ok=True)
    for line in open(a.ids):
        k, i = line.split()[:2]
        i = int(i)
        z = np.load(os.path.join(a.dump, f"{k}.npz"))
        sig = _signals(z, sc, k)
        emg_half = float(np.percentile(np.abs(sig["emg"]), 95) * 2) if "emg" in sig else None
        stage = STAGE[int(z["y"][i])]
        open(os.path.join(a.out, f"{k}_{i:05d}.png"), "wb").write(render(sig, fs, i, stage, emg_half))
    print("DONE", flush=True)


def review(a):
    """Blind review set: per concept ``--n`` segments from the annotated epochs, in four strata of equal size
    (M1+M2+, M1 only, M2 only, neither), shuffled across concepts. Writes ``OUT/<id>.png`` (segment shaded),
    ``OUT/items.json`` (id, concept, question: no stratum) and ``OUT/key.json`` (stratum per id: keep it
    away from the reviewer). Ids are random hex."""
    rng = np.random.default_rng(a.seed)
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    sc = json.load(open(os.path.join(a.dump, "scale.json")))
    fs = int(meta["fs"]["eeg"])
    rows = [r for r in map(json.loads, open(os.path.join(a.dump, "m2", "annot.jsonl"))) if r.get("ann")]
    m1c = {}
    pool = {}  # (concept, stratum) -> [(key, epoch, seg, stage)]
    for r in rows:
        k, i = r["key"], r["epoch"]
        if k not in m1c:
            m1c = {k: _load_mask(os.path.join(a.dump, "m1", f"{k}.npz"))}
        m1 = m1c[k]
        seen = {s["segment"]: set(s["concepts"]) for s in r["ann"]["segments"]}
        for c, (mask, stages, _) in VOCAB.items():
            if mask is None or mask not in m1 or STAGE_ID[r["stage"]] not in stages:
                continue
            p1 = _seg_pos(m1[mask], i)
            for j, sname in enumerate(SEGS):
                st = {(1, 1): "both", (1, 0): "m1", (0, 1): "m2", (0, 0): "none"}[(int(p1[j]), int(c in seen[sname]))]
                pool.setdefault((c, st), []).append((k, i, j, r["stage"]))
    per = a.n // 4
    items, key = [], {}
    for c in VOCAB:
        for st in ("both", "m1", "m2", "none"):
            cand = pool.get((c, st), [])
            for idx in rng.permutation(len(cand))[:per]:
                uid = "%08x" % rng.integers(0, 2 ** 32)
                items.append({"id": uid, "concept": c, "src": cand[idx]})
                key[uid] = {"concept": c, "stratum": st, "pool": len(cand), "key": cand[idx][0],
                            "epoch": int(cand[idx][1]), "segment": SEGS[cand[idx][2]]}
    rng.shuffle(items)
    os.makedirs(a.out, exist_ok=True)
    sig_k = None
    for it in sorted(items, key=lambda t: t["src"][0]):
        k, i, j, stage = it["src"]
        if sig_k != k:
            z = np.load(os.path.join(a.dump, f"{k}.npz"))
            sig, sig_k = _signals(z, sc, k), k
            emg_half = float(np.percentile(np.abs(sig["emg"]), 95) * 2) if "emg" in sig else None
        open(os.path.join(a.out, it["id"] + ".png"), "wb").write(render(sig, fs, i, stage, emg_half, hl=j))
    json.dump([{"id": t["id"], "concept": t["concept"], "stage": t["src"][3]} for t in items],
              open(os.path.join(a.out, "items.json"), "w"), indent=0)
    json.dump(key, open(a.key, "w"), indent=0)
    print(f"[review] {len(items)} segments -> {a.out}; key -> {a.key}\n"
          + " ".join(f"{c}:{st}={len(v)}" for (c, st), v in sorted(pool.items())) + "\nDONE", flush=True)


def page(a):
    """``DIR/index.html`` for the blind review of ``DIR/items.json``: one shaded segment per card, the manual
    definition of the concept, yes / no / unsure; a button copies the answers as JSON (paste them back)."""
    import html

    items = json.load(open(os.path.join(a.dir, "items.json")))
    cards = []
    for n, it in enumerate(items):
        c = it["concept"]
        cards.append(f'''<div class="card"><div class="h">{n + 1}/{len(items)} - <b>{html.escape(c)}</b>
(stage {it["stage"]})</div><div class="d">{html.escape(VOCAB[c][2])}</div>
<img loading="lazy" src="{it["id"]}.png"><div class="q">Is <b>{html.escape(c)}</b> present in the shaded 5 s?
{"".join(f'<label><input type="radio" name="{it["id"]}" value="{v}"> {v}</label>' for v in ("yes", "no", "unsure"))}
</div></div>''')
    doc = f'''<!doctype html><html><head><meta charset="utf-8"><title>M1/M2 blind review</title><style>
body{{font-family:sans-serif;max-width:1500px;margin:auto;padding:12px}}.card{{border-bottom:1px solid #ccc;
padding:12px 0}}img{{width:100%}}.d{{color:#555;font-size:14px}}.q label{{margin-right:18px}}
textarea{{width:100%;height:120px}}button{{padding:8px 16px;margin:12px 0}}</style></head><body>
<h2>Blind review: {len(items)} segments</h2><p>Judge only the shaded 5 s, by the definition shown. The stage is the
scored one. Answers stay in this page until you copy them.</p>{"".join(cards)}
<button onclick="go()">Copy answers</button><span id="n"></span><textarea id="out"></textarea>
<script>function go(){{const r={{}};document.querySelectorAll('input[type=radio]:checked').forEach(e=>r[e.name]=e.value);
const t=JSON.stringify(r);document.getElementById('out').value=t;document.getElementById('n').textContent=
' '+Object.keys(r).length+' answered';navigator.clipboard&&navigator.clipboard.writeText(t)}}</script></body></html>'''
    out = os.path.join(a.dir, "index.html")
    open(out, "w").write(doc)
    os.chmod(out, 0o600)
    print(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("annotate")
    p.add_argument("--dump", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--groups", default="")
    p.add_argument("--pos", type=int, default=2)
    p.add_argument("--neg", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("agree")
    p.add_argument("--dump", required=True)
    p = sub.add_parser("render")
    p.add_argument("--dump", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("review")
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True, help="folder of the PNGs and items.json (goes to the review page)")
    p.add_argument("--key", required=True, help="answer key (strata), stays on Sofia")
    p.add_argument("--n", type=int, default=48, help="segments per concept (4 equal strata)")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("page")
    p.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    {"annotate": annotate, "agree": agree, "render": render_ids, "review": review, "page": page}[a.cmd](a)


if __name__ == "__main__":
    main()
