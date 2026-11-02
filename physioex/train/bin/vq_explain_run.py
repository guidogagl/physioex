"""Explainability examples of the multimodal MSAE-VQ system on the test nights.

Inputs: the latents directories the multimodal stager was trained on (``--mods eeg=DIR ...``, each
with ``latents.json`` naming its MSAE-VQ run and prefix) and the stager run (``--stager DIR`` with
``stager.json`` + ``stager.pt``). Every test night is re-encoded to get the codes and the centre
epochs. Sleep stages are used only to describe codes and to pick cases, never to build codes.

Families (``<out>/*.json`` + ``<out>/waves.npz``):

* A ``atlas.json``: per modality, the most used level-1 E codes (decoded prototype, band power,
  stage mix, nearest real epochs) and the level-1 P codes (band power vs mean, event rate, stage mix).
* B ``events.json``: which P codes fire on the label-free events (EEG sigma bursts, EOG rapid eye
  movements, EMG bursts) and two counterfactuals: an event token replaced by the code stack of the
  nearest non-event token of the same epoch (erase), and an event code stack written into an
  epoch without events (insert); decoded event score and stager probabilities before and after.
* C ``cases.json``: six epochs (three correct, three typical errors) explained by token occlusion,
  modality ablation and attention; ``global.json``: modality influence per stage, occlusion vs
  attention rank correlation, deletion check.
* D ``text.json``: the night as a sequence of code words (entropy, staying, NMI with the
  hypnogram, most typical words per stage) and staging from the bag of codes (linear, global
  evidence per stage) against the stager.

``python -m physioex.train.bin.vq_explain_run --mods eeg=DIR eog=DIR emg=DIR --stager DIR --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from physioex.models import stager_explain as sx
from physioex.models import vq_explain as vx
from physioex.models.qformer_stager import (STAGES, MultiSeqStager, latent_stats, metrics, night_groups,
                                            pack_modalities)

EVENT = {"eeg": "sigma burst", "eog": "rapid eye movement", "emg": "EMG burst"}


# ----------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------


@torch.no_grad()
def encode_epochs(model, ds, ids, nights: np.ndarray, epochs: np.ndarray, centres: bool = True,
                  batch: int = 128) -> Dict[str, np.ndarray]:
    """Codes, quantized latents and (``centres``) the centre epochs of the given ``(night, epoch)``
    pairs (``nights`` index into ``ids``)."""
    device = next(model.parameters()).device
    model.eval()
    have = {s.subject_id: s for s in ds._subjects}
    half = ds.sequence_length // 2
    acc: Dict[str, list] = {}
    for g in np.unique(nights):
        sid = ids[int(g)]
        n = ds._n_epochs[sid]
        item = ds._build_item(have[sid], 0, n)
        x = next(iter(item["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        ts = epochs[nights == g]
        for j in range(0, len(ts), batch):
            tb = ts[j:j + batch]
            w = np.stack([x[t - half:t + half + 1] for t in tb])
            z = model.encode(torch.from_numpy(w)[:, :, None].to(device))
            for s in ("N", "E", "P"):
                acc.setdefault(s, []).append(z[s].float().cpu().numpy())
                if z["codes"][s] is not None:
                    acc.setdefault(f"codes_{s}", []).append(z["codes"][s].cpu().numpy().astype(np.int16))
            if centres:
                acc.setdefault("centres", []).append(x[tb])
        acc.setdefault("night", []).append(np.full(len(ts), g))
        acc.setdefault("epoch", []).append(ts)
    return {k: np.concatenate(v) for k, v in acc.items()}


def events_per_token(mod: str, model, centres: np.ndarray, night: np.ndarray, q: float = 95.0) -> np.ndarray:
    """``(M, T_p)`` bool label-free events: EEG sigma bursts (``latent_probe.sigma_bursts``), EOG
    max slope of 0.3-5 Hz and EMG RMS above the ``q``-th percentile of the night."""
    from physioex.models.latent_probe import sigma_bursts
    from physioex.models.recon_eval_modality import event_feature

    T = model.n_tok
    if mod == "eeg":
        return vx.bursts_per_token(sigma_bursts(centres, night, q), T)
    f = event_feature(centres, model.fs, "eye" if mod == "eog" else "burst")[:, :T]
    out = np.zeros_like(f, dtype=bool)
    for g in np.unique(night):
        m = night == g
        out[m] = f[m] > np.percentile(f[m], q)
    return out


def event_score(mod: str, model, waves: np.ndarray) -> np.ndarray:
    """``(B, T_p)`` per-token event strength of decoded waves: EEG log sigma power, EOG max slope
    of 0.3-5 Hz, EMG log RMS."""
    from physioex.models.recon_eval_modality import event_feature

    T = model.n_tok
    if mod == "eeg":
        seg = vx.token_segments(waves, T).reshape(-1, waves.shape[1] // T)
        return vx.band_power(seg)["sigma"].reshape(len(waves), T)
    f = event_feature(waves, model.fs, "eye" if mod == "eog" else "burst")[:, :T]
    return f if mod == "eog" else np.log10(f + 1e-6)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _codes_of(z, i, device):
    return {s: (torch.from_numpy(z[f"codes_{s}"][i:i + 1].astype(np.int64)).to(device)
                if f"codes_{s}" in z else None) for s in ("N", "E", "P")}


def _std(lat: Dict[str, torch.Tensor], st) -> Dict[str, np.ndarray]:
    return {k: (lat[k][0].float().cpu().numpy() - st[k][0]) / st[k][1] for k in ("N", "E", "P")}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    return o


def stage_mix(lab: np.ndarray) -> Dict[str, float]:
    return {s: float((lab == i).mean()) for i, s in enumerate(STAGES)} if len(lab) else {}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mods", nargs="+", required=True, help="name=latents_dir, as the stager was trained")
    ap.add_argument("--stager", required=True, help="stager run dir (stager.json, stager.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases-global", type=int, default=1500, help="random test epochs for the modality summary")
    ap.add_argument("--occl-global", type=int, default=300, help="of which with token occlusion")
    ap.add_argument("--cf", type=int, default=300, help="counterfactuals per modality and kind")
    ap.add_argument("--bag-train-nights", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    dirs = dict(x.split("=", 1) for x in a.mods)
    mods = list(dirs)
    log = lambda *s: print("[vq_explain_run]", *s, flush=True)  # noqa: E731

    # --- stager and packed test latents
    sj = json.load(open(os.path.join(a.stager, "stager.json")))
    sa = sj["args"]
    lat = {m: {s: dict(np.load(os.path.join(d, f"latents_{s}.npz"))) for s in ("train", "test")} for m, d in dirs.items()}
    data, spec, missing = pack_modalities(lat, sa.get("drop") or ())
    st = {m: latent_stats(lat[m]["train"]) for m in mods}
    stager = MultiSeqStager(spec, epoch=sa["arch"][3:], seq_len=sa["seq_len"], seq_layers=sa["seq_layers"], d=sa["d"],
                            heads=sa["heads"], dropout=sa["dropout"], k=sa["k"], layers=sa["layers"],
                            fusion=sa.get("fusion", "late"))
    stager.load_state_dict(torch.load(os.path.join(a.stager, "stager.pt"), map_location="cpu"))
    stager.to(dev).eval()
    te = data["test"]
    y = te["y"].numpy()
    L, half = sa["seq_len"], sa["seq_len"] // 2
    log("spec", spec, "missing", missing["test"])

    # windows centred on every test epoch that has L//2 scored neighbours on each side
    centre_win = {}
    for g in night_groups(te):
        for k in range(half, len(g) - half):
            centre_win[int(g[k])] = g[k - half:k + half + 1]
    idx_all = np.array(sorted(centre_win))

    def window(i):
        w = torch.from_numpy(centre_win[int(i)])
        return te["N"][w][None].to(dev), te["E"][w][None].to(dev), te["P"][w][None].to(dev)

    # centre-only probabilities of every windowed test epoch (what the explanations perturb)
    P_all = np.zeros((len(y), len(STAGES)), np.float32)
    for b in range(0, len(idx_all), 256):
        ws = torch.from_numpy(np.stack([centre_win[int(i)] for i in idx_all[b:b + 256]]))
        P_all[idx_all[b:b + 256]] = sx.probs(stager, te["N"][ws].to(dev), te["E"][ws].to(dev), te["P"][ws].to(dev)).cpu().numpy()
    pred = P_all.argmax(1)
    m_ = metrics(pred[idx_all], y[idx_all], te["night"][idx_all])
    log("centre-window kappa per night", round(m_["kappa_night_mean"], 4), "stager.json test", round(sj["test"]["kappa_night_mean"], 4))
    summary = {"stager": a.stager, "spec": spec, "kappa_centre_window": m_["kappa_night_mean"],
               "kappa_voting_stager_json": sj["test"]["kappa_night_mean"], "n_test_windows": int(len(idx_all))}

    # --- re-encode the test nights, per modality
    vq, Z, ev, bands, key2row, test_ids = {}, {}, {}, {}, {}, None
    for m in mods:
        meta = json.load(open(os.path.join(dirs[m], "latents.json")))
        run, ds, model = _load_run(meta["run"])
        model.vq_levels = parse_vq_key(meta["vq"])
        model.to(dev).eval()
        tl = lat[m]["test"]
        test_ids = test_ids if m != mods[0] else list(run["splits"]["test"])
        assert list(run["splits"]["test"]) == test_ids, f"{m}: test nights differ from {mods[0]}"
        Z[m] = encode_epochs(model, ds, run["splits"]["test"], tl["night"].astype(int), tl["epoch"].astype(int))
        # the codes must give the latents the stager was trained on
        diff = float(np.abs(Z[m]["E"] - tl["E"].astype(np.float32)).max())
        log(m, "re-encoded", len(Z[m]["E"]), "epochs, max |E - saved E|", round(diff, 4))
        summary.setdefault("reencode_max_abs_diff_E", {})[m] = diff
        vq[m] = (model, ds, run, meta)
        bands[m] = dict(vx.BANDS) if m == "eeg" else {f"{lo:g}-{hi:g}": (lo, hi) for lo, hi in model.pow_bands}
        ev[m] = events_per_token(m, model, Z[m]["centres"], Z[m]["night"])
        key2row[m] = dict(zip((Z[m]["night"].astype(np.int64) * 10 ** 6 + Z[m]["epoch"]).tolist(), range(len(Z[m]["E"]))))
    tkeys = (te["night"].astype(np.int64) * 10 ** 6 + te["epoch"]).tolist()
    row = {m: np.array([key2row[m].get(k, -1) for k in tkeys]) for m in mods}  # packed test index -> Z row
    ylab = {}
    for m in mods:  # stage of every Z row (descriptive only)
        lab = np.full(len(Z[m]["E"]), -1)
        ok = row[m] >= 0
        lab[row[m][ok]] = y[ok]
        ylab[m] = lab
    waves = {}

    # --- A: atlas
    atlas = {}
    for m in mods:
        model = vq[m][0]
        z = Z[m]
        ea = vx.atlas_e(model, z, z["centres"], ylab[m], top=3, min_count=30, fs=model.fs, bands=bands[m])
        top_e = sorted(ea, key=lambda c: -ea[c]["count"])[:15]
        for c in top_e:
            waves[f"atlas_{m}_E{c}_proto"] = ea[c].pop("prototype")
            near = list(ea[c]["nearest"])
            waves[f"atlas_{m}_E{c}_nearest"] = z["centres"][near]
            ea[c]["nearest"] = [(int(z["night"][i]), int(z["epoch"][i])) for i in near]
        pa = vx.atlas_p(z, z["centres"], ev[m], min_count=200, fs=model.fs, bands=bands[m])
        tok_lab = np.repeat(ylab[m], z["codes_P"].shape[1])
        c1 = z["codes_P"][..., 0].reshape(-1)
        for c in pa:
            pa[c]["stages"] = stage_mix(tok_lab[(c1 == c) & (tok_lab >= 0)])
        base = float(ev[m].mean())
        for c in pa:
            pa[c]["lift"] = pa[c]["burst_rate"] / max(base, 1e-9)
        atlas[m] = {"event": EVENT[m], "event_base_rate": base,
                    "E_used_level1": int(len(np.unique(z["codes_E"][:, 0]))),
                    "P_used_level1": int(len(np.unique(c1))),
                    "E": {c: ea[c] for c in top_e},
                    "P_top_count": {c: pa[c] for c in sorted(pa, key=lambda c: -pa[c]["count"])[:15]},
                    "P_top_event": {c: pa[c] for c in sorted(pa, key=lambda c: -pa[c]["lift"])[:15]}}
        # stage purity of E codes (descriptive): code entropy given stage
        log(m, "atlas: E codes used", atlas[m]["E_used_level1"], "P codes used", atlas[m]["P_used_level1"])
    json.dump(_jsonable(atlas), open(os.path.join(a.out, "atlas.json"), "w"), indent=1)

    # --- B: event words and counterfactuals
    events = {}
    for m in mods:
        model = vq[m][0]
        z = Z[m]
        T = z["codes_P"].shape[1]
        bc = vx.burst_codes(z["codes_P"][..., 0], ev[m], top=5)
        top5 = set(bc["top5"]["codes"])
        inv = np.full(len(z["E"]), -1)
        ok = row[m] >= 0
        inv[row[m][ok]] = np.arange(len(y))[ok]  # Z row -> packed test index
        has_win = np.isin(inv, idx_all)
        sc = event_score(m, model, z["centres"])
        res = {"event": EVENT[m], "detector": bc, "erase": [], "insert": []}
        # erase: event token with a top-5 event code -> code stack of the nearest non-event token
        cand = [(r, j) for r, j in zip(*np.nonzero(ev[m])) if has_win[r] and z["codes_P"][r, j, 0] in top5
                and (~ev[m][r]).any()]
        pick = [cand[i] for i in rng.permutation(len(cand))[:a.cf]]
        # insert: the code stack of the strongest event token of the top code, into epochs without events
        flat_top = [(r, j) for r, j in zip(*np.nonzero(ev[m])) if z["codes_P"][r, j, 0] == bc["top5"]["codes"][0]]
        src = max(flat_top, key=lambda t: sc[t]) if flat_top else None
        quiet = [r for r in np.nonzero(has_win & ~ev[m].any(1))[0]]
        ins = [quiet[i] for i in rng.permutation(len(quiet))[:a.cf]] if src is not None else []
        for kind, items in (("erase", pick), ("insert", [(r, T // 2) for r in ins])):
            for r, j in items:
                codes = _codes_of(z, r, dev)
                if kind == "erase":
                    non = np.nonzero(~ev[m][r])[0]
                    j2 = int(non[np.argmin(np.abs(non - j))])
                    new = z["codes_P"][r, j2]
                else:
                    new = z["codes_P"][src[0], src[1]]
                out = vx.swap_token(model, codes, int(j), new, fs=model.fs, bands=bands[m])
                s0, s1 = event_score(m, model, out["wave_before"])[0], event_score(m, model, out["wave_after"])[0]
                i = int(inv[r])
                n, e, p = window(i)
                zs = _std(out["latents_after"], st[m])
                n2, e2, p2 = sx.set_epoch(n, e, p, spec, m, half, N=zs["N"] if "codes_N" in z else None, E=zs["E"], P=zs["P"])
                pb, pa_ = sx.probs(stager, n, e, p)[0].cpu().numpy(), sx.probs(stager, n2, e2, p2)[0].cpu().numpy()
                res[kind].append({"row": int(r), "night": int(z["night"][r]), "epoch": int(z["epoch"][r]), "token": int(j),
                                  "stage": STAGES[y[i]], "code_before": int(z["codes_P"][r, j, 0]), "code_after": int(new[0]),
                                  "score_change_token": float(s1[j] - s0[j]),
                                  "score_change_others": float(np.abs(np.delete(s1 - s0, j)).mean()),
                                  "probs_before": pb, "probs_after": pa_})
                if len(res[kind]) <= 3:
                    waves[f"cf_{m}_{kind}{len(res[kind])}_before"] = out["wave_before"][0]
                    waves[f"cf_{m}_{kind}{len(res[kind])}_after"] = out["wave_after"][0]
                    waves[f"cf_{m}_{kind}{len(res[kind])}_input"] = z["centres"][r]
        for kind in ("erase", "insert"):
            it = res[kind]
            if not it:
                continue
            dp = np.array([x["probs_after"] - x["probs_before"] for x in it])
            flip = np.mean([np.argmax(x["probs_after"]) != np.argmax(x["probs_before"]) for x in it])
            res[f"{kind}_summary"] = {
                "n": len(it), "score_change_token_mean": float(np.mean([x["score_change_token"] for x in it])),
                "score_change_others_mean": float(np.mean([x["score_change_others"] for x in it])),
                "delta_probs_mean": dict(zip(STAGES, dp.mean(0))), "delta_probs_abs_mean": float(np.abs(dp).sum(1).mean()),
                "pred_flip_rate": float(flip),
                "delta_probs_by_stage": {s: dict(zip(STAGES, dp[[x["stage"] == s for x in it]].mean(0)))
                                         for s in STAGES if any(x["stage"] == s for x in it)}}
            res[kind] = it[:12]  # examples kept in the json; the summary covers all
        if src is not None:
            res["insert_source"] = {"night": int(z["night"][src[0]]), "epoch": int(z["epoch"][src[0]]), "token": int(src[1]),
                                    "stack": z["codes_P"][src[0], src[1]]}
        events[m] = res
        log(m, "events: top5 precision", round(bc["top5"]["precision"], 3), "recall", round(bc["top5"]["recall"], 3),
            "base", round(bc["base_rate"], 3), {k: res.get(f"{k}_summary", {}).get("score_change_token_mean") for k in ("erase", "insert")})
    json.dump(_jsonable(events), open(os.path.join(a.out, "events.json"), "w"), indent=1)

    # --- C: cases and global summary
    def explain(i, occl=True):
        n, e, p = window(i)
        tgt = int(pred[i])
        r = {"index": int(i), "night": int(te["night"][i]), "epoch": int(te["epoch"][i]), "true": STAGES[y[i]],
             "pred": STAGES[tgt], "probs": P_all[i]}
        r["ablation_centre"] = sx.modality_ablation(stager, n, e, p, spec, tgt)
        r["ablation_window"] = sx.modality_ablation(stager, n, e, p, spec, tgt, whole_window=True)
        if occl:
            oc = sx.token_occlusion(stager, n, e, p, spec, tgt)
            r["occlusion"] = oc
            r["attention"] = sx.attention_at(stager, n, e, p, spec)
            r["deletion"] = sx.deletion_check(stager, n, e, p, spec, oc)
            r["rank_corr_occl_attn"] = {m: sx.rank_corr(-oc[m]["P"], r["attention"][m]["P"]) for m in mods}
        return r

    def has_event(i, m):
        return row[m][i] >= 0 and ev[m][row[m][i]].any()

    win = set(idx_all.tolist())
    rules = [("N2 corretta con fuso", lambda i: y[i] == 2 and pred[i] == 2 and has_event(i, "eeg")),
             ("REM corretta con movimenti oculari", lambda i: y[i] == 4 and pred[i] == 4 and has_event(i, "eog")),
             ("Wake corretta", lambda i: y[i] == 0 and pred[i] == 0),
             ("errore N1 -> Wake", lambda i: y[i] == 1 and pred[i] == 0),
             ("errore N1 -> REM", lambda i: y[i] == 1 and pred[i] == 4),
             ("errore REM -> N2", lambda i: y[i] == 4 and pred[i] == 2)]
    cases = []
    for name, rule in rules:
        cand = [i for i in idx_all if rule(i) and all(row[m][i] >= 0 for m in mods)]
        if not cand:
            continue
        i = max(cand, key=lambda i: P_all[i].max())
        r = explain(i)
        r["name"], r["n_candidates"] = name, len(cand)
        # the five most influential P tokens, with their codes and what the atlas says of the code
        flat = sorted(((m, j, r["occlusion"][m]["P"][j]) for m in mods for j in range(len(r["occlusion"][m]["P"]))),
                      key=lambda t: t[2])[:5]
        r["top_tokens"] = []
        for m, j, d in flat:
            zr = row[m][i]
            c = int(Z[m]["codes_P"][zr, j, 0])
            info = atlas[m]["P_top_count"].get(c) or atlas[m]["P_top_event"].get(c) or {}
            r["top_tokens"].append({"mod": m, "token": int(j), "t_sec": j * 30 / Z[m]["codes_P"].shape[1], "delta": float(d),
                                    "code": c, "event_here": bool(ev[m][zr, j]),
                                    "code_event_rate": info.get("burst_rate"), "code_stages": info.get("stages")})
        r["codes_E"] = {m: int(Z[m]["codes_E"][row[m][i], 0]) for m in mods}
        k = len(cases) + 1
        for m in mods:
            zr = row[m][i]
            waves[f"case{k}_{m}_input"] = Z[m]["centres"][zr]
            zq = vq[m][0].embed(_codes_of(Z[m], zr, dev))
            waves[f"case{k}_{m}_decoded"] = vx.decode_full(vq[m][0], zq["N"], zq["E"], zq["P"]).float().cpu().numpy()[0]
        cases.append(r)
        log("case", name, "pred p", round(float(P_all[i].max()), 3), "deletion", r["deletion"])
    json.dump(_jsonable(cases), open(os.path.join(a.out, "cases.json"), "w"), indent=1)

    sel = rng.choice(idx_all, min(a.cases_global, len(idx_all)), replace=False)
    occl_set = set(sel[:a.occl_global].tolist())
    glob = {"by_stage": {}, "rank_corr": {m: [] for m in mods}, "deletion": []}
    infl = {s: {m: {"centre": [], "window": []} for m in mods} for s in STAGES}
    share = {s: {m: [] for m in mods} for s in STAGES}
    for i in sel:
        if not all(row[m][i] >= 0 for m in mods):
            continue
        r = explain(i, occl=int(i) in occl_set)
        s = STAGES[y[i]]
        for m in mods:
            infl[s][m]["centre"].append(r["ablation_centre"][m]["delta"])
            infl[s][m]["window"].append(r["ablation_window"][m]["delta"])
        if "occlusion" in r:
            tot = {m: -np.minimum(r["occlusion"][m]["P"], 0).sum() - min(r["occlusion"][m]["N"], 0) - min(r["occlusion"][m]["E"], 0)
                   for m in mods}
            z_ = sum(tot.values()) or 1.0
            for m in mods:
                share[s][m].append(tot[m] / z_)
                glob["rank_corr"][m].append(r["rank_corr_occl_attn"][m])
            glob["deletion"].append(r["deletion"])
    for s in STAGES:
        glob["by_stage"][s] = {m: {"delta_p_centre": float(np.mean(infl[s][m]["centre"])) if infl[s][m]["centre"] else None,
                                   "delta_p_window": float(np.mean(infl[s][m]["window"])) if infl[s][m]["window"] else None,
                                   "occlusion_share": float(np.mean(share[s][m])) if share[s][m] else None,
                                   "n": len(infl[s][m]["centre"])} for m in mods}
    glob["rank_corr"] = {m: {"mean": float(np.nanmean(v)) if v else None, "n": len(v)} for m, v in glob["rank_corr"].items()}
    dl = glob.pop("deletion")
    glob["deletion"] = {"top_drop_mean": float(np.mean([d["top_drop"] for d in dl])),
                        "random_drop_mean": float(np.mean([d["random_drop"] for d in dl])),
                        "top_beats_random": float(np.mean([d["top_drop"] > d["random_drop"] for d in dl])), "n": len(dl)}
    glob["summary"] = summary
    json.dump(_jsonable(glob), open(os.path.join(a.out, "global.json"), "w"), indent=1)
    log("global deletion", glob["deletion"], "rank corr", glob["rank_corr"])

    # --- D: the night as text, bag of codes
    text = {"per_modality": {}}
    for m in mods:
        z = Z[m]
        st_ = []
        for g in np.unique(z["night"]):
            r_ = np.nonzero(z["night"] == g)[0]
            r_ = r_[np.argsort(z["epoch"][r_])]
            st_.append(vx.sequence_stats(z["codes_E"][r_, 0], ylab[m][r_]))
        e1 = z["codes_E"][:, 0]
        words = {}
        for k, s in enumerate(STAGES):
            msk = ylab[m] == k
            u, c = np.unique(e1[msk], return_counts=True)
            top = u[np.argsort(-c)[:3]]
            words[s] = [{"code": int(w), "share_of_stage": float((e1[msk] == w).mean()),
                         "p_stage_given_code": float((ylab[m][e1 == w] == k).mean())} for w in top]
        text["per_modality"][m] = {"entropy_bits_mean": float(np.mean([x["entropy_bits"] for x in st_])),
                                   "stay_mean": float(np.mean([x["stay"] for x in st_])),
                                   "codes_per_night_mean": float(np.mean([len(x["codes"]) for x in st_])),
                                   "nmi_stages_mean": float(np.mean([x["nmi_stages"] for x in st_])),
                                   "typical_words": words}
    # one stretch of a night around a transition into REM, written as words
    g0 = night_groups(te)[0]
    tr = [k for k in range(1, len(g0)) if y[g0[k]] == 4 and y[g0[k - 1]] != 4]
    k0 = max(0, (tr[0] if tr else len(g0) // 2) - 8)
    lines = []
    for i in g0[k0:k0 + 16]:
        lines.append({"epoch": int(te["epoch"][i]), "stage": STAGES[y[i]], "pred": STAGES[pred[i]] if i in win else None,
                      "words": {m: (f"E{int(Z[m]['codes_E'][row[m][i], 0])}" if row[m][i] >= 0 else None) for m in mods},
                      "events": {m: (int(ev[m][row[m][i]].sum()) if row[m][i] >= 0 else None) for m in mods}})
    text["night_excerpt"] = lines

    # bag of codes: train on a subset of the training nights (re-encoded), test on the test nights
    from scipy import sparse as sp

    bags_tr, bags_te, names, ytr_ = [], [], [], None
    for m in mods:
        model, ds, run, meta = vq[m]
        tl = lat[m]["train"]
        nights = np.unique(tl["night"])
        keep = np.sort(rng.choice(nights, min(a.bag_train_nights, len(nights)), replace=False))
        msk = np.isin(tl["night"], keep)
        zt = encode_epochs(model, ds, run["splits"]["train"], tl["night"][msk].astype(int), tl["epoch"][msk].astype(int),
                           centres=False)
        k_tr = (zt["night"].astype(np.int64) * 10 ** 6 + zt["epoch"]).tolist()
        if ytr_ is None:  # reference rows: the first modality's
            ref_keys = k_tr
            ytr_ = tl["y"][msk].astype(int)
        pos = dict(zip(k_tr, range(len(k_tr))))
        ridx = np.array([pos.get(k, -1) for k in ref_keys])
        nc = {"N": model.n_codes["N"] if model.use_n else 0, "E": model.n_codes["E"], "P": model.n_codes["P"]}
        b = vx.bag_of_codes(zt, nc, sparse=True)
        b_al = b[np.maximum(ridx, 0)].multiply((ridx >= 0)[:, None]).tocsr()
        bt = vx.bag_of_codes(Z[m], nc, sparse=True)
        bt_al = bt[np.maximum(row[m], 0)].multiply((row[m] >= 0)[:, None]).tocsr()
        bags_tr.append(b_al)
        bags_te.append(bt_al)
        names += [f"{m}:{x}" for x in [f"{s}:{i}" for s in ("N", "E", "P") for i in range(nc[s])]]
    xtr, xte = sp.hstack(bags_tr).tocsr(), sp.hstack(bags_te).tocsr()
    evd = vx.code_evidence(xtr, ytr_, names, top=8)
    pb = evd["model"].predict(xte)
    kb = metrics(pb, y, te["night"])
    text["bag_of_codes"] = {"train_nights": int(a.bag_train_nights), "kappa_night_mean": kb["kappa_night_mean"],
                            "kappa_stager_voting": sj["test"]["kappa_night_mean"], "evidence": evd["evidence"],
                            "features": int(xtr.shape[1])}
    # bag per single modality (which vocabulary carries the stage)
    off = 0
    for m, b in zip(mods, bags_tr):
        w = b.shape[1]
        clf = vx.code_evidence(xtr[:, off:off + w], ytr_, names[off:off + w], top=1)["model"]
        text["bag_of_codes"][f"kappa_only_{m}"] = metrics(clf.predict(xte[:, off:off + w]), y, te["night"])["kappa_night_mean"]
        off += w
    log("bag of codes kappa", round(kb["kappa_night_mean"], 4), {m: round(text["bag_of_codes"][f"kappa_only_{m}"], 4) for m in mods})
    json.dump(_jsonable(text), open(os.path.join(a.out, "text.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "waves.npz"), **{k: np.asarray(v, np.float32) for k, v in waves.items()})
    log("DONE")


if __name__ == "__main__":
    main()
