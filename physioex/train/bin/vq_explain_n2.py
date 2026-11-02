"""N2 micro-events in the EEG codes: find epochs with a spindle or a K-complex coded by P tokens
whose removal lowers the stager's N2 confidence.

EEG only: MSAE-VQ EEG (r3a) and the EEG-only sequence stager. Events are found label-free in the
raw signal (stages only select N2 epochs):

* spindles: YASA-like, two of three criteria (relative sigma power 11-16 / 1-30 Hz >= 0.2, sigma
  RMS above mean + 1.5 sd of the night, sigma-broadband correlation >= 0.65), 0.5-2 s;
* K-complex candidates: in 0.3-4 Hz, a negative trough followed within 0.2-1 s by a positive
  peak, peak-to-peak >= ``kc_k`` x the median 0.3-4 Hz peak-to-peak of the night's N2 epochs.

Two removals per event (centre epoch of a 21-epoch window, neighbours untouched):

* **code**: the code stacks of the event's P tokens are replaced by those of the nearest
  non-event tokens of the same epoch; control = the same number of random non-event tokens
  replaced the same way (``n_control`` draws);
* **signal**: the event is removed from the EEG (spindle: 11-16 Hz removed on its span; KC:
  0.3-4 Hz component removed on its span, Hann taper) and the window re-encoded; we report which
  P tokens changed code and the new N2 probability.

``python -m physioex.train.bin.vq_explain_n2 --run VQ_RUN --vq N8E8P6 --latents DIR --stager DIR --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from physioex.models import vq_explain as vx
from physioex.models.qformer_stager import ARCHS, STAGES, latent_stats, night_groups, standardize

FS = 100
N2 = 2


def _bp(x, lo, hi, fs=FS):
    f = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * ((f >= lo) & (f <= hi)), n=x.shape[-1], axis=-1)


def _runs(mask):
    """Start/stop indices of True runs in a 1-d bool array."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def spindles(x: np.ndarray, fs: int = FS, win: float = 0.3) -> list:
    """YASA-like spindle detection on a 1-d signal; returns ``[(start, stop)]`` in samples."""
    w = int(win * fs)
    k = np.ones(w) / w
    sig = _bp(x, 11, 16, fs)
    broad = _bp(x, 1, 30, fs)
    p_sig = np.convolve(sig ** 2, k, "same")
    p_br = np.convolve(broad ** 2, k, "same") + 1e-12
    rel = p_sig / p_br
    rms = np.sqrt(p_sig)
    thr_rms = rms.mean() + 1.5 * rms.std()
    # moving correlation sigma vs broadband
    ms, mb = np.convolve(sig, k, "same"), np.convolve(broad, k, "same")
    cov = np.convolve(sig * broad, k, "same") - ms * mb
    vs = np.convolve(sig ** 2, k, "same") - ms ** 2
    vb = np.convolve(broad ** 2, k, "same") - mb ** 2
    corr = cov / np.sqrt(np.maximum(vs * vb, 1e-12))
    votes = (rel >= 0.2).astype(int) + (rms >= thr_rms).astype(int) + (corr >= 0.65).astype(int)
    out = []
    for a, b in _runs(votes >= 2):
        if 0.5 * fs <= b - a <= 2.0 * fs:
            out.append((int(a), int(b)))
    return out


def kcomplexes(x_epoch: np.ndarray, ref_ptp: float, k: float = 4.0, fs: int = FS) -> list:
    """K-complex candidates in one epoch: trough then peak within 0.2-1 s on 0.3-4 Hz, with
    peak-to-peak >= k x ref_ptp; returns ``[(start, stop)]`` in samples (trough - 0.3 s .. peak + 0.3 s)."""
    lo = _bp(x_epoch, 0.3, 4.0, fs)
    out, i = [], 0
    n = len(lo)
    while i < n:
        t = i + int(np.argmin(lo[i:min(n, i + fs)]))
        seg = lo[t + int(0.2 * fs):t + fs]
        if len(seg):
            p = t + int(0.2 * fs) + int(np.argmax(seg))
            if lo[p] - lo[t] >= k * ref_ptp and lo[t] < 0 < lo[p]:
                out.append((max(0, t - int(0.3 * fs)), min(n, p + int(0.3 * fs))))
                i = p + int(0.3 * fs)
                continue
        i += fs // 2
    return out


def remove_event(x_epoch, a, b, kind, fs=FS):
    """Event removed on [a, b) with a Hann taper: spindle -> 11-16 Hz, KC -> 0.3-4 Hz component."""
    comp = _bp(x_epoch, 11, 16, fs) if kind == "spindle" else _bp(x_epoch, 0.3, 4.0, fs)
    taper = np.zeros_like(x_epoch)
    pad = int(0.1 * fs)
    a0, b0 = max(0, a - pad), min(len(x_epoch), b + pad)
    taper[a0:b0] = np.hanning(b0 - a0)
    taper[a:b] = np.maximum(taper[a:b], 1.0)
    return x_epoch - taper * comp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="MSAE-VQ EEG run")
    ap.add_argument("--vq", default="N8E8P6")
    ap.add_argument("--latents", required=True, help="latents dir of the EEG-only stager")
    ap.add_argument("--stager", required=True)
    ap.add_argument("--split", default="test", choices=("test", "train", "valid"))
    ap.add_argument("--max-nights", type=int, default=0)
    ap.add_argument("--kc-k", type=float, default=4.0)
    ap.add_argument("--n-control", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    os.makedirs(a.out, exist_ok=True)
    log = lambda *s: print("[vq_explain_n2]", *s, flush=True)  # noqa: E731

    run, ds, model = _load_run(a.run)
    model.vq_levels = parse_vq_key(a.vq)
    model.to(dev).eval()
    lat = {s: dict(np.load(os.path.join(a.latents, f"latents_{s}.npz"))) for s in ("train", a.split)}
    st = latent_stats(lat["train"])
    data = standardize({"train": lat["train"], a.split: lat[a.split]})[a.split]
    sj = json.load(open(os.path.join(a.stager, "stager.json")))
    sa = sj["args"]
    stager = ARCHS[sa["arch"]](dim=data["N"].shape[-1], d_p=data["P"].shape[-1], n_tok=data["P"].shape[1], d=sa["d"],
                               k=sa["k"], layers=sa["layers"], heads=sa["heads"], dropout=sa["dropout"],
                               seq_len=sa["seq_len"], seq_layers=sa["seq_layers"])
    stager.load_state_dict(torch.load(os.path.join(a.stager, "stager.pt"), map_location="cpu"))
    stager.to(dev).eval()
    L, half = sa["seq_len"], sa["seq_len"] // 2
    y = data["y"].numpy()

    @torch.no_grad()
    def p_centre(N, E, P):
        return stager(N.to(dev), E.to(dev), P.to(dev))[:, half].float().softmax(-1).cpu().numpy()

    def std_(z):
        return {k: torch.from_numpy((z[k].float().cpu().numpy() - st[k][0]) / st[k][1]).float() for k in ("N", "E", "P")}

    ids = run["splits"][a.split]
    have = {s.subject_id: s for s in ds._subjects}
    hseq = ds.sequence_length // 2
    T = model.n_tok
    seg = 3000 // T
    groups = night_groups(data)
    if a.max_nights:
        groups = groups[:a.max_nights]
    found = []
    for g in groups:
        night = int(data["night"][g[0]])
        sid = ids[night]
        n = ds._n_epochs[sid]
        item = ds._build_item(have[sid], 0, n)
        x = next(iter(item["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        eps = data["epoch"][g]
        # centre-window N2 probability of every scored N2 epoch with a full window
        cand = [k for k in range(half, len(g) - half) if y[g[k]] == N2]
        if not cand:
            continue
        W = torch.from_numpy(np.stack([g[k - half:k + half + 1] for k in cand]))
        pr = np.concatenate([p_centre(data["N"][W[i:i + 256]], data["E"][W[i:i + 256]], data["P"][W[i:i + 256]])
                             for i in range(0, len(W), 256)])
        # night-level event detection on the N2 epochs' signal
        n2_eps = eps[[k for k in range(len(g)) if y[g[k]] == N2]]
        ref_ptp = float(np.median(np.ptp(_bp(x[n2_eps], 0.3, 4.0), axis=1))) / 2
        sp_night = spindles(x[n2_eps].reshape(-1))
        sp_by_ep = {}
        for s0, s1 in sp_night:
            e_ = s0 // 3000
            if s1 // 3000 == e_:  # spindle inside one epoch
                sp_by_ep.setdefault(int(n2_eps[e_]), []).append((s0 % 3000, s1 % 3000))
        for ci, k in enumerate(cand):
            p_n2 = float(pr[ci, N2])
            if not 0.4 <= p_n2 <= 0.97 or pr[ci].argmax() != N2:
                continue
            t = int(eps[k])
            if t - hseq < 0 or t + hseq + 1 > n:
                continue
            evs = [("spindle", s, e) for s, e in sp_by_ep.get(t, [])]
            evs += [("kcomplex", s, e) for s, e in kcomplexes(x[t], ref_ptp, a.kc_k)]
            if not evs:
                continue
            win = g[k - half:k + half + 1]
            Nw, Ew, Pw = data["N"][win][None], data["E"][win][None], data["P"][win][None]
            # codes of the centre epoch (re-encode the original window)
            xw = torch.from_numpy(x[t - hseq:t + hseq + 1][None, :, None]).to(dev)
            with torch.no_grad():
                z0 = model.encode(xw)
            codes0 = {s: (c.clone() if c is not None else None) for s, c in z0["codes"].items()}
            ev_tok = sorted({int(j) for _, s, e in evs for j in range(s // seg, min(T, (e - 1) // seg + 1))})
            non = np.array([j for j in range(T) if j not in ev_tok])
            if len(non) < len(ev_tok) + 2:
                continue

            def code_swap(toks, full=False):
                c2 = {kk: (v.clone() if v is not None else None) for kk, v in codes0.items()}
                swap = {}
                for j in toks:
                    pool = non[~np.isin(non, toks)]
                    j2 = int(pool[np.argmin(np.abs(pool - j))])
                    c2["P"][:, j] = codes0["P"][:, j2]
                    swap[int(j)] = j2
                zq = model.embed(c2)
                zs = std_(zq)
                P2 = Pw.clone()
                P2[0, half] = zs["P"][0]
                pr2 = p_centre(Nw, Ew, P2)[0]
                return (pr2, swap, zq) if full else float(pr2[N2])

            pr_code, swap_map, zq_swap = code_swap(ev_tok, full=True)
            p_code = float(pr_code[N2])
            ctrl = [code_swap(sorted(rng.choice(non, len(ev_tok), replace=False).tolist())) for _ in range(a.n_control)]
            # signal edit: remove every event, re-encode, replace the centre epoch's latents
            xe = x[t].copy()
            for kind, s, e in evs:
                xe = remove_event(xe, s, e, kind)
            xw2 = x[t - hseq:t + hseq + 1].copy()
            xw2[hseq] = xe
            with torch.no_grad():
                z1 = model.encode(torch.from_numpy(xw2[None, :, None]).to(dev))
            zs1 = std_(z1)
            N2w, E2w, P2w = Nw.clone(), Ew.clone(), Pw.clone()
            N2w[0, half], E2w[0, half], P2w[0, half] = zs1["N"][0], zs1["E"][0], zs1["P"][0]
            p_sig = float(p_centre(N2w, E2w, P2w)[0, N2])
            changed = (z1["codes"]["P"][0, :, 0] != codes0["P"][0, :, 0]).cpu().numpy()
            rec = {"night": night, "subject": sid, "epoch": t, "index": int(g[k]), "p_n2": p_n2, "probs": pr[ci].tolist(),
                   "events": [{"kind": kind, "start_s": float(s / FS), "stop_s": float(e / FS)} for kind, s, e in evs],
                   "event_tokens": ev_tok, "p_n2_code_removed": p_code, "p_n2_code_control_mean": float(np.mean(ctrl)),
                   "p_n2_code_control_min": float(np.min(ctrl)),
                   "p_n2_signal_removed": p_sig, "tokens_changed_by_edit": np.flatnonzero(changed).tolist(),
                   "E_code_changed_by_edit": bool((z1["codes"]["E"][0, 0] != codes0["E"][0, 0]).item()),
                   "codes_P_level1_event_tokens": codes0["P"][0, ev_tok, 0].cpu().tolist()}
            rec["delta_code"] = p_code - p_n2
            rec["delta_code_vs_control"] = p_code - rec["p_n2_code_control_mean"]
            rec["delta_signal"] = p_sig - p_n2
            found.append(rec)
            # waves for the figure (kept for every candidate, small)
            dec0 = vx.decode_full(model, z0["N"], z0["E"], z0["P"]).float().cpu().numpy()[0]
            dec1 = vx.decode_full(model, z1["N"], z1["E"], z1["P"]).float().cpu().numpy()[0]
            dec2 = vx.decode_full(model, zq_swap["N"], zq_swap["E"], zq_swap["P"]).float().cpu().numpy()[0]
            rec["_waves"] = {"input": x[t], "edited": xe, "decoded": dec0, "decoded_edited": dec1, "decoded_code_swapped": dec2}
            rec["probs_code_removed"] = pr_code.tolist()
            rec["probs_signal_removed"] = p_centre(N2w, E2w, P2w)[0].tolist()
            rec["swap_map"] = {str(k_): v_ for k_, v_ in swap_map.items()}
        log(f"night {night}: {len(cand)} N2 epochs, {len(sp_night)} spindles, {sum(1 for r in found if r['night'] == night)} candidates")
    found.sort(key=lambda r: (r["delta_code_vs_control"] + r["delta_signal"]))
    waves = {}
    for i, r in enumerate(found[:12]):
        for k_, v in r["_waves"].items():
            waves[f"c{i}_{k_}"] = np.asarray(v, np.float32)
    for r in found:
        r.pop("_waves", None)
    summ = {"n_candidates": len(found),
            "n_spindle": sum(any(e["kind"] == "spindle" for e in r["events"]) for r in found),
            "n_kc": sum(any(e["kind"] == "kcomplex" for e in r["events"]) for r in found),
            "mean_delta_code": float(np.mean([r["delta_code"] for r in found])) if found else None,
            "mean_delta_code_vs_control": float(np.mean([r["delta_code_vs_control"] for r in found])) if found else None,
            "mean_delta_signal": float(np.mean([r["delta_signal"] for r in found])) if found else None,
            "frac_code_below_all_controls": float(np.mean([r["p_n2_code_removed"] < r["p_n2_code_control_min"] for r in found])) if found else None}
    json.dump({"summary": summ, "top": found[:12], "all": [{k: r[k] for k in ("night", "epoch", "p_n2", "delta_code", "delta_code_vs_control", "delta_signal")} for r in found]},
              open(os.path.join(a.out, f"n2_events_{a.split}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, f"n2_events_{a.split}_waves.npz"), **waves)
    log("summary", summ)
    for r in found[:6]:
        log({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("night", "epoch", "p_n2", "delta_code", "delta_code_vs_control", "delta_signal", "events", "tokens_changed_by_edit")})
    log("DONE")


if __name__ == "__main__":
    main()
