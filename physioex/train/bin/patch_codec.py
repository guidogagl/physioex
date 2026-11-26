"""Train and audit the 2 s patch tokenizer (:mod:`physioex.models.patch_codec`, xmai-lab/sleep-lens#19).

    python -m physioex.train.bin.patch_codec fit --items shhs,mesa,... --splits S.json --cache-dir C \\
        [--local-cache] --out RUN [--levels 8 --steps-per-epoch 5000 --epochs 30 --batch 128]
    python -m physioex.train.bin.patch_codec eval --run RUN [--cache-dir C] [--per-unit 10] [--prefixes 1 2 4 8]

``fit``: whitening filter on the training pool (``whitener.json``), then encoder, RVQ and decoder
together on 16 s crops at a random alignment (site-balanced windows of the train split; validation
windows of the val split). Writes ``ckpt/``, ``model.pt`` (best validation checkpoint), ``eval.json``
(validation crops, per prefix of levels) and ``run.json`` (``model: patch_codec``).

``eval``: the S2 gates on held-out nights (test split of the pool units, ``--per-unit`` nights each).
The main sleep period of every night is coded once at full depth (patch grid on the first epoch of
the period) and decoded for every prefix of levels. Metrics on the scored epochs:

* gate 0 (R12): token F1 of the rule masks of ``x`` and ``x_hat`` with the thresholds frozen on ``x``
  (:func:`physioex.models.msae_factor.rule_masks`, 0.5 s tokens), pooled over the nights: KC, SWA,
  spindle and alpha >= 0.9;
* band power: R^2 of the log10 power per 1 s segment (:func:`physioex.models.recon_eval.band_r2`,
  which also gives the R^2 with phase, of the amplitude, on the grid shifted by 0.5 s and per 5 s)
  >= 0.9 on the bands 1 ... 17-30 Hz, on ``--per-night`` random scored epochs of every night;
* seams: whitened squared error within ±0.1 s of the patch boundaries over the error inside the
  patches <= 1.1;
* comb: whitened residual PSD at the multiples of 0.5 Hz (0.5-35 Hz) over the median of the
  neighbouring bins (±0.075-0.2 Hz), median over the lines <= 1.1 (the same ratio of ``x`` is the control);
* phase test (reported, no gate): whitened R^2 of 30 s epochs coded alone, real epochs against
  phase-randomised surrogates with the same amplitude spectrum.

``L*`` is the shortest prefix that passes every gate. Writes ``gates.json``: aggregate numbers only.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from physioex.models.recon_eval import load_best
from physioex.train.bin.msvq import _merged_data, _merged_loaders

GATE = {"f1": 0.9, "logpow": 0.9, "seam": 1.1, "comb": 1.1}
GATE_BANDS = ("1", "2-4", "5-8", "9-12", "13-16", "17-30")
#: pool units with a test split (the healthy units of concept_scan without the out-of-pool controls)
EVAL_UNITS = ("ccshs", "dcsm", "hmc", "hpap_lab_full", "mass_ss02", "mass_ss03", "mass_ss04", "mass_ss05", "mesa",
              "mros", "sleepedf", "wsc_visit1")
SEAM = 10  # samples on each side of a patch boundary (0.1 s)
COMB_SEG = 4000  # Welch segment of the comb test: 40 s, 0.025 Hz bins


def _git(*cmd):
    try:
        return subprocess.check_output(["git", *cmd], cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:
        return None


def default_prefixes(levels: int):
    return sorted({k for k in (1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32) if k <= levels} | {levels})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--items", required=True, help="comma-separated pool items (physioex.data.merged)")
    f.add_argument("--splits", required=True)
    f.add_argument("--cache-dir", default=None)
    f.add_argument("--local-cache", action="store_true", help="copy the train/val cache files to node-local disk")
    f.add_argument("--channel", default="EEG_clean", choices=("EEG_clean",))
    f.add_argument("--seq", type=int, default=1, help="epochs per loaded window; one crop per window")
    f.add_argument("--fold", type=int, default=0)
    f.add_argument("--steps-per-epoch", type=int, default=5000)
    f.add_argument("--valid-per-item", type=int, default=500)
    f.add_argument("--batch", type=int, default=128)
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--out", required=True)
    f.add_argument("--epochs", type=int, default=30)
    f.add_argument("--patience", type=int, default=5)
    f.add_argument("--valid-ratio", type=float, default=0.5)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--grad-clip", type=float, default=1.0)
    f.add_argument("--width", type=int, default=64)
    f.add_argument("--dim", type=int, default=128)
    f.add_argument("--codes", type=int, default=1024)
    f.add_argument("--levels", type=int, default=8)
    f.add_argument("--q-dropout", type=float, default=0.5)
    f.add_argument("--order", type=int, default=32, help="order of the LPC whitening filter")
    f.add_argument("--stft-weight", type=float, default=0.1)
    f.add_argument("--restart-after", type=int, default=30)
    f.add_argument("--crop-sec", type=float, default=16.0)
    f.add_argument("--whiten-batches", type=int, default=50)
    f.add_argument("--vq-warmup", type=int, default=2000,
                   help="steps with the RVQ bypassed before the k-means initialisation of the codebooks (0: none)")
    f.add_argument("--kmeans-per-code", type=int, default=20)
    f.add_argument("--eval-batches", type=int, default=40, help="validation batches of eval.json")
    e = sub.add_parser("eval")
    e.add_argument("--run", required=True)
    e.add_argument("--splits", default=None, help="default: the run's")
    e.add_argument("--cache-dir", default=None, help="default: the run's source cache")
    e.add_argument("--units", default=",".join(EVAL_UNITS))
    e.add_argument("--per-unit", type=int, default=10)
    e.add_argument("--per-night", type=int, default=60, help="scored epochs per night for the band R^2")
    e.add_argument("--phase-per-night", type=int, default=20, help="epochs per night for the phase test")
    e.add_argument("--prefixes", type=int, nargs="*", default=None)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None, help="default: <run>/gates.json")
    a = ap.parse_args(argv)
    if a.cmd == "eval":
        return _eval(a)
    a.model, a.dataset, a.load_fs, a.fs, a.root, a.resample = "patch_codec", None, None, None, None, None
    a.cache_dir_src = a.cache_dir or os.environ.get("PHYSIOEX_CACHE_DIR")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.fold)
    t0 = time.time()
    local = None
    if a.local_cache:
        from physioex.data.merged import stage_local

        local = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                             f"physioex_cache_{os.environ.get('SLURM_JOB_ID', os.getpid())}_patch_codec")
        stage_local(a.items.split(","), a.splits, a.cache_dir_src, local, [a.channel])
        a.cache_dir = local
        import signal
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        _fit(a, t0)
    finally:
        if local:
            shutil.rmtree(local, ignore_errors=True)


@torch.no_grad()
def _valid_prefixes(model, loader, n_batches, device, prefixes):
    """Whitened and raw R^2 of the validation crops per prefix, code usage per level at full depth."""
    from physioex.data.collate import stack_channels
    from physioex.models.msvq import code_stats
    from physioex.models.patch_codec import PatchCodecTrainer, crops

    model.eval()
    batches = [b for _, b in zip(range(n_batches), loader)]
    out = {}
    for k in prefixes:
        model.vq_levels = k
        acc = {"r2_w": [], "r2_raw": []}
        for b in batches:
            _, r2, extra = PatchCodecTrainer._step(model, b, None, device)
            acc["r2_w"].append(float(r2))
            acc["r2_raw"].append(extra["r2_raw"])
        out[str(k)] = {m: float(np.mean(v)) for m, v in acc.items()}
        out[str(k)]["bits_per_patch"] = model.bits_per_patch(k)
    model.vq_levels = model.max_levels
    use = {}
    for b in batches:
        x = stack_channels(b)[:, :, 0].float()
        x = crops(x.reshape(len(x), -1).to(device), model.crop, False)
        c = model.codes(x, model.max_levels)
        for l in range(c.shape[-1]):
            use.setdefault(l + 1, []).append(c[..., l].reshape(-1).cpu())
    out["codes"] = {str(l): dict(zip(("perplexity", "used"), code_stats(torch.cat(v), model.n_codes)))
                    for l, v in use.items()}
    return out


def _fit(a, t0):
    from physioex.models.patch_codec import (PatchCodec, PatchCodecTrainer, fit_whitener, kmeans_init_rvq,
                                            warmup_continuous)

    md, ds, train_ids, valid_ids, _ = _merged_data(a)
    tr, n_tr, va, n_va = _merged_loaders(a, md, ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCodec(width=a.width, dim=a.dim, codes=a.codes, levels=a.levels, q_dropout=a.q_dropout, order=a.order,
                       stft_weight=a.stft_weight, restart_after=a.restart_after, crop_sec=a.crop_sec).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[patch_codec] windows: train {n_tr}, valid {n_va}; {n_par / 1e6:.2f} M parameters, "
          f"{model.bits_per_patch()} bits per 2 s patch at {a.levels} levels", flush=True)
    winfo = fit_whitener(model, tr, a.whiten_batches)
    json.dump(winfo, open(os.path.join(a.out, "whitener.json"), "w"), indent=1)
    print(f"[patch_codec] whitener {winfo}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    if a.vq_warmup > 0:
        wlog = warmup_continuous(model, tr, opt, a.vq_warmup, device, a.grad_clip or None)
        t1 = time.time()
        nk = kmeans_init_rvq(model, tr, device, a.kmeans_per_code)
        print(f"[patch_codec] k-means of {a.levels} levels on {nk} latents ({time.time() - t1:.0f} s)", flush=True)
        json.dump({"warmup": wlog, "kmeans_vectors": nk}, open(os.path.join(a.out, "warmup.json"), "w"), indent=1)
    n_val = a.epochs * max(1, round(1 / a.valid_ratio))  # the Trainer steps the scheduler at each validation
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_val, eta_min=a.lr / 20)
    model = PatchCodecTrainer.train(model, (tr, va), checkpoint_path=os.path.join(a.out, "ckpt"), max_epochs=a.epochs,
                                    lr=a.lr, weight_decay=1e-4, valid_interval_ratio=a.valid_ratio,
                                    log_confusion_matrix=False, early_stopping_patience=a.patience,
                                    logger="tensorboard", run_name="patch_codec", seed=a.fold,
                                    gpu_id=0 if torch.cuda.is_available() else None, grad_clip=a.grad_clip or None,
                                    optimizer=opt, scheduler=sched)
    best = load_best(model, a.out)
    print(f"[patch_codec] best checkpoint {best}", flush=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    device = next(model.parameters()).device
    ev = _valid_prefixes(model, va, a.eval_batches, device, default_prefixes(a.levels))
    ev["n_params"] = n_par
    json.dump(ev, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    print("[patch_codec] valid " + " ".join(f"L{k}: r2_w {v['r2_w']:.3f} r2_raw {v['r2_raw']:.3f}"
                                           for k, v in ev.items() if k.isdigit()), flush=True)

    import physioex
    run = {"args": vars(a), "git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
           "physioex_version": getattr(physioex, "__version__", None), "host": socket.gethostname(),
           "job_id": os.environ.get("SLURM_JOB_ID"), "merged": md.report, "best_checkpoint": best,
           "splits": {"train": train_ids, "valid": valid_ids, "test": []},
           "windows": {"train": n_tr, "valid": n_va}, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(run, open(os.path.join(a.out, "run.json"), "w"), indent=1, default=str)
    print("[patch_codec] DONE", flush=True)


# ----------------------------------------------------------------------------
# eval: S2 gates on held-out nights
# ----------------------------------------------------------------------------


def load_codec(run_dir: str, device):
    from physioex.models.patch_codec import build_from_args

    run = json.load(open(os.path.join(run_dir, "run.json")))
    model = build_from_args(run["args"])
    sd = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(sd)
    return run, model.to(device).eval()


@torch.no_grad()
def encode_signal(model, sig: torch.Tensor, levels: int, chunk: int = 2048) -> torch.Tensor:
    """``(n,)`` contiguous signal -> ``(n // 200, levels)`` codes; the encoder is local, so chunks are exact."""
    step = chunk * model.patch
    return torch.cat([model.codes(sig[i:i + step][None], levels)[0] for i in range(0, len(sig), step)])


@torch.no_grad()
def decode_signal(model, codes: torch.Tensor, chunk: int = 2048, margin: int = 2) -> torch.Tensor:
    """``(P, k)`` codes -> ``(P * 200,)``. Chunks overlap by ``margin`` patches on each side: with the
    ±1 patch receptive field the result equals one decode of the whole sequence."""
    P, out = len(codes), []
    for s in range(0, P, chunk):
        lo, hi = max(0, s - margin), min(P, s + chunk + margin)
        y = model.decode_codes(codes[None, lo:hi])[0]
        n = min(chunk, P - s)
        out.append(y[(s - lo) * model.patch:(s - lo + n) * model.patch])
    return torch.cat(out)


def _line_ratios(psd: np.ndarray, freqs: np.ndarray, base: float = 0.5, f_max: float = 35.0):
    """PSD at ``k * base`` Hz over the median of the bins at ±0.075-0.2 Hz, for ``k * base <= f_max``."""
    df = freqs[1] - freqs[0]
    near = [j for j in range(int(round(0.075 / df)), int(round(0.2 / df)) + 1)]
    out = []
    for k in range(1, int(f_max / base) + 1):
        i = int(round(k * base / df))
        nb = [psd[i + s * j] for j in near for s in (-1, 1) if 0 <= i + s * j < len(psd)]
        out.append(float(psd[i] / np.median(nb)))
    return np.array(out)


def _welch_sum(x: torch.Tensor, seg: int = COMB_SEG):
    """Sum over segments of the Hann-windowed periodogram of ``(n,)`` (50 % overlap) and the count."""
    if len(x) < seg:
        return 0.0, 0
    w = torch.hann_window(seg, dtype=torch.float64, device=x.device)
    s = x.double().unfold(0, seg, seg // 2) * w
    return torch.fft.rfft(s, dim=-1).abs().pow(2).sum(0).cpu().numpy(), s.shape[0]


def _pick_nights(test, unit_names, units, per_unit, seed):
    rng = np.random.default_rng(seed)
    out = []
    for u in units:
        have = sorted((s, d) for d, s, _, _ in test if unit_names[d] == u)
        if not have:
            continue
        pick = np.sort(rng.choice(len(have), min(per_unit, len(have)), replace=False))
        out.extend((u, have[i][1], have[i][0]) for i in pick)
    return out


def _eval(a):
    from physioex.data.merged import build_test
    from physioex.models.msae_factor import RULES, rule_masks
    from physioex.models.psg_gen.p1 import phase_surrogate
    from physioex.models.recon_eval import band_r2, channel_pipelines
    from physioex.train.bin.pretrain_combo import Nights

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run, model = load_codec(a.run, device)
    ra = run["args"]
    prefixes = a.prefixes or default_prefixes(model.max_levels)
    cache = a.cache_dir or ra.get("cache_dir_src") or ra.get("cache_dir")
    md = build_test(a.splits or ra["splits"], [ra["channel"]], channel_pipelines(ra), 1, cache,
                    units=a.units.split(","), memmap_cache_size=4, trim_mode="main_period")
    nights = _pick_nights(md.test, md.unit_names, a.units.split(","), a.per_unit, a.seed)
    print(f"[patch_codec eval] {len(nights)} nights, prefixes {prefixes}", flush=True)
    rng = np.random.default_rng(a.seed)
    W = model.whitener
    R = len(RULES)
    cnt = {k: {"all": np.zeros((R, 3))} for k in prefixes}  # tp, fp, fn per rule
    seam = {k: np.zeros(4) for k in prefixes}  # seam sum, seam n, interior sum, interior n
    comb = {k: [0.0, 0] for k in prefixes}
    comb_x = [0.0, 0]
    phase = {k: {"real": np.zeros(2), "surr": np.zeros(2)} for k in prefixes}  # SSE, SST
    sub_x, sub_h = [], {k: [] for k in prefixes}
    used = []
    for unit, d, sid in nights:
        x, y, ts = Nights(md.multi._datasets[d], {"seq": 1}).get(sid)
        if len(ts) < 20:
            continue
        e0, e1 = int(ts.min()), int(ts.max()) + 1
        sig = torch.from_numpy(x[e0:e1].reshape(-1)).to(device)
        sc = torch.from_numpy(ts - e0).to(device)
        X = sig.view(-1, 3000)[sc]
        codes = encode_signal(model, sig, model.max_levels)
        m0 = rule_masks(X, model.fs, 60) > 0
        pos = torch.arange(len(sig), device=device)
        in_scored = torch.zeros(len(sig) // 3000, dtype=torch.bool, device=device)
        in_scored[sc] = True
        in_scored = in_scored.repeat_interleave(3000)
        near = ((pos % model.patch) < SEAM) | ((pos % model.patch) >= model.patch - SEAM)
        xw = W(sig[None])[0]
        p_, n_ = _welch_sum(xw)
        comb_x[0], comb_x[1] = comb_x[0] + p_, comb_x[1] + n_
        sub = np.sort(rng.choice(len(ts), min(a.per_night, len(ts)), replace=False))
        sub_x.append(X[sub].cpu().numpy())
        ph = torch.from_numpy(np.sort(rng.choice(len(ts), min(a.phase_per_night, len(ts)), replace=False))).to(device)
        Xp = X[ph]
        Sp = phase_surrogate(Xp, seed=a.seed)
        cp, cs = model.codes(Xp, model.max_levels), model.codes(Sp, model.max_levels)
        for k in prefixes:
            h = decode_signal(model, codes[:, :k])
            H = h.view(-1, 3000)[sc]
            m1 = rule_masks(H, model.fs, 60, ref=X) > 0
            c = torch.stack([(m0 & m1).sum((0, 1)), (~m0 & m1).sum((0, 1)), (m0 & ~m1).sum((0, 1))], -1).cpu().numpy()
            cnt[k]["all"] += c
            cnt[k].setdefault(unit, np.zeros((R, 3)))
            cnt[k][unit] += c
            ew = (W(h[None])[0] - xw).pow(2)
            seam[k] += [float(ew[near & in_scored].sum()), int((near & in_scored).sum()),
                        float(ew[~near & in_scored].sum()), int((~near & in_scored).sum())]
            p_, n_ = _welch_sum(W(h[None])[0] - xw)
            comb[k][0], comb[k][1] = comb[k][0] + p_, comb[k][1] + n_
            sub_h[k].append(H[sub].cpu().numpy())
            for name, src, cc in (("real", Xp, cp), ("surr", Sp, cs)):
                hw, tw = W(model.decode_codes(cc[:, :, :k])), W(src)
                inner = slice(model.patch, -model.patch)  # the first and last patch of an epoch coded alone
                hw, tw = hw[:, inner], tw[:, inner]
                phase[k][name] += [float((hw - tw).pow(2).sum()), float((tw - tw.mean(1, keepdim=True)).pow(2).sum())]
        used.append(f"{unit}/{sid}")
        print(f"[patch_codec eval] {unit}/{sid}: {len(ts)} scored epochs ({time.time() - t0:.0f}s)", flush=True)

    freqs = np.fft.rfftfreq(COMB_SEG, 1 / model.fs)
    xs = np.concatenate(sub_x)
    ctrl = _line_ratios(comb_x[0] / max(comb_x[1], 1), freqs)
    res = {"run": os.path.abspath(a.run), "n_nights": len(used),
           "units": {u: sum(1 for n in used if n.startswith(u + "/")) for u in a.units.split(",")},
           "n_band_epochs": int(len(xs)), "gate": {**GATE, "bands": GATE_BANDS},
           "comb_control_x": {"median": float(np.median(ctrl)), "max": float(ctrl.max())}, "prefix": {}}

    def f1(c):
        tp, fp, fn = c
        return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 1.0

    for k in prefixes:
        br = band_r2(np.concatenate(sub_h[k]), xs)
        g0 = {r: {"f1": f1(cnt[k]["all"][i]), "n_ref": int(cnt[k]["all"][i][0] + cnt[k]["all"][i][2]),
                  "n_hat": int(cnt[k]["all"][i][0] + cnt[k]["all"][i][1])} for i, r in enumerate(RULES)}
        per_unit = {u: {r: f1(v[i]) for i, r in enumerate(RULES)} for u, v in cnt[k].items() if u != "all"}
        s = seam[k]
        seam_ratio = float((s[0] / max(s[1], 1)) / max(s[2] / max(s[3], 1), 1e-12))
        lines = _line_ratios(comb[k][0] / max(comb[k][1], 1), freqs)
        ph = {n: float(1 - v[0] / max(v[1], 1e-12)) for n, v in phase[k].items()}
        checks = {"f1": all(g0[r]["f1"] >= GATE["f1"] for r in RULES),
                  "logpow": all(br["logpow"][b] >= GATE["logpow"] for b in GATE_BANDS),
                  "seam": seam_ratio <= GATE["seam"], "comb": float(np.median(lines)) <= GATE["comb"]}
        res["prefix"][str(k)] = {"bits_per_patch": model.bits_per_patch(k), "gate0": g0, "gate0_per_unit": per_unit,
                                 "band_r2": br, "seam_ratio": seam_ratio,
                                 "comb": {"median": float(np.median(lines)), "max": float(lines.max()),
                                          "worst_hz": float(0.5 * (1 + int(lines.argmax())))},
                                 "phase_test": {"r2_w_real": ph["real"], "r2_w_surrogate": ph["surr"],
                                                "gap": ph["real"] - ph["surr"]},
                                 "checks": checks, "pass": all(checks.values())}
        print(f"[patch_codec eval] L{k}: " + " ".join(f"{r} {g0[r]['f1']:.3f}" for r in RULES)
              + f" | logpow min {min(br['logpow'][b] for b in GATE_BANDS):.3f} | seam {seam_ratio:.3f}"
              + f" | comb {np.median(lines):.3f} | phase gap {ph['real'] - ph['surr']:+.3f} | pass {all(checks.values())}",
              flush=True)
    passing = [k for k in prefixes if res["prefix"][str(k)]["pass"]]
    res["L_star"] = min(passing) if passing else None
    res["minutes"] = round((time.time() - t0) / 60, 1)
    res["git_sha"] = _git("rev-parse", "HEAD")
    out = a.out or os.path.join(a.run, "gates.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[patch_codec eval] L* = {res['L_star']} -> {out}", flush=True)
    print("[patch_codec eval] DONE", flush=True)


if __name__ == "__main__":
    main()
