"""Train and audit the 2 s patch tokenizer (:mod:`physioex.models.patch_codec`, xmai-lab/sleep-lens#19).

    python -m physioex.train.bin.patch_codec fit --items shhs,mesa,... --splits S.json --cache-dir C \\
        [--local-cache] --out RUN [--levels 8 --steps-per-epoch 5000 --epochs 30 --batch 128]
    python -m physioex.train.bin.patch_codec eval --run RUN [--cache-dir C] [--per-unit 10] [--prefixes 1 2 4 8]
    python -m physioex.train.bin.patch_codec dump --run RUN [--train] [--nights-from DUMP/meta.json] [--out RUN/dump]
    python -m physioex.train.bin.patch_codec transform --from-run RUN --out DIR [--deltas 0.35,0.25] [--groups 8]

``transform`` (revision 4, 28.11.2026): the reference transform codec without training
(:class:`~physioex.models.patch_codec.TransformCodec`), with the whitener, splits and pipeline of ``RUN``; one run
directory ``DIR/d<delta>`` per step (``DIR/d<delta>_g<G>`` with ``--groups G``, revision 4b: one gain per block of
KLT directions, nothing clipped), read by ``eval`` and ``dump`` like a trained run.

``fit``: whitening filter on the training pool (``whitener.json``), then encoder, quantiser (RVQ, or FSQ
with ``--quantizer fsq``; revision 2 also adds ``--whiten signal``, ``--ctx-enc`` and ``--ctx-dec``, see
:mod:`physioex.models.patch_codec`) and decoder together on 16 s crops at a random alignment (site-balanced windows of the train split; validation
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
  patches <= 1.1. Control added after the pilot of 26.11.2026 (``seam_vs_centre``): the same error over
  the error within ±0.1 s of the patch centres, a window of the same width;
* comb: whitened residual PSD at the multiples of 0.5 Hz (0.5-35 Hz) over the median of the
  neighbouring bins (±0.075-0.2 Hz), median over the lines <= 1.1 (the same ratio of ``x`` is the control).
  Since 26.11.2026 (decision after the pilot, xmai-lab/sleep-lens#19) the gate is ``on_over_off`` <= 1.1:
  that median over the median of the same ratio off the grid (0.25 Hz + multiples of 0.5 Hz), because
  ``x`` itself gave 1.066 on the grid in the pilot. The registered statistic stays in the report;
* fidelity (revision 2, 27.11.2026): whitened R^2 of the scored samples of the nights >= 0.95
  (``r2_w_nights``; the raw R^2 ``r2_raw_nights`` is reported). FSQ models without nested dropout are
  evaluated at full depth only (:func:`model_prefixes`);
* phase test (reported, no gate): whitened R^2 of 30 s epochs coded alone, real epochs against
  phase-randomised surrogates with the same amplitude spectrum.

``L*`` is the shortest prefix that passes every gate (``L_star``; ``L_star_registered`` with the comb
statistic of the plan, ``L_star_controlled`` with both controls). Writes ``gates.json``: aggregate numbers only.

``dump``: the codes of whole main sleep periods for the autoregressive model (S3) and the evaluation
(S5): ``--train`` takes the train nights of the run's splits (``splits_train_subset.json``), and
``--nights-from`` takes the nights of an existing dump (``concept_scan``: pool test, MASS SS01,
DREAMS). Every night is coded on 4 patch grids shifted by 0, 0.5, 1 and 1.5 s (``OFFSETS``), which
together cover the 0.5 s token grid. One ``<out>/<unit>/<sid>.npz`` per night: ``c0``, ``c50``,
``c100``, ``c150`` ``(P, levels)`` int16 codes (patch ``p`` of grid ``o`` starts at sample ``o + 200 p``
of the period), ``e0`` (first epoch of the period in the recording), ``n`` (samples of the period)
and ``y`` (stage of every epoch of the period, -1 when not scored); ``<out>/meta.json``.
``--twin-out DIR`` also writes the spectral twin of the codes in the same layout (S3, xmai-lab/sleep-lens#20):
:func:`~physioex.models.patch_codec.patch_spectra` of every patch, coded by a residual k-means with the
codec's K and levels (:class:`~physioex.models.patch_codec.SpectralTwin`, fitted on ``--twin-fit-nights``
train nights of the same call, ``twin.pt`` and ``twin_fit.json``).
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import torch

from physioex.models.recon_eval import load_best
from physioex.train.bin.msvq import _merged_data, _merged_loaders

GATE = {"f1": 0.9, "logpow": 0.9, "seam": 1.1, "comb": 1.1, "r2": 0.95}  # r2: revision 2, 27.11.2026
GATE_BANDS = ("1", "2-4", "5-8", "9-12", "13-16", "17-30")
#: pool units with a test split (the healthy units of concept_scan without the out-of-pool controls)
EVAL_UNITS = ("ccshs", "dcsm", "hmc", "hpap_lab_full", "mass_ss02", "mass_ss03", "mass_ss04", "mass_ss05", "mesa",
              "mros", "sleepedf", "wsc_visit1")
SEAM = 10  # samples on each side of a patch boundary (0.1 s)
COMB_SEG = 4000  # Welch segment of the comb test: 40 s, 0.025 Hz bins
#: a 1 s segment with a smaller sd (RobustScale units, ~1e-3 of a typical second) is flat: a gap of the clean channel,
#: filled with zeros by ``NanToZero`` and constant after the scale. Amendment of 28.11.2026 (xmai-lab/sleep-lens#24):
#: 0.3 % of the seconds of the eval sample are flat, with a true band power ~1e-37 of the median; against the 1e-6
#: floor of the log they carried 85-91 % of the squared log error of the 2-30 Hz bands of pilot Q2.
FLAT_SD = 1e-3


def _git(*cmd):
    try:
        return subprocess.check_output(["git", *cmd], cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:
        return None


def default_prefixes(levels: int):
    return sorted({k for k in (1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32) if k <= levels} | {levels})


def model_prefixes(model):
    """Prefixes worth evaluating: every default prefix when the model was trained on prefixes (RVQ, or FSQ with
    nested dropout ``q_dropout > 0``), else the full depth only (also the transform codec of revision 4)."""
    if model.quantizer == "transform" or (model.quantizer == "fsq" and not model.q_dropout):
        return [model.max_levels]
    return default_prefixes(model.max_levels)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--items", required=True, help="comma-separated pool items (physioex.data.merged)")
    f.add_argument("--splits", required=True)
    f.add_argument("--train-subjects", type=int, default=0,
                   help="keep this many train nights over the items, the same number per item "
                        "(<out>/splits_train_subset.json, reused by the dump); 0: all")
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
    f.add_argument("--upsample", choices=("convT", "resize"), default="convT",
                   help="decoder upsampling: transposed convolution, or repeat + convolution (stopped early, job 1669985)")
    f.add_argument("--restart-after", type=int, default=30)
    f.add_argument("--quantizer", choices=("rvq", "fsq"), default="rvq",
                   help="fsq: finite scalar quantisation; --levels then counts groups of --fsq-group dimensions")
    f.add_argument("--fsq-levels", type=int, default=9, help="values per FSQ dimension (odd)")
    f.add_argument("--fsq-group", type=int, default=4, help="FSQ dimensions per code (fsq_levels ** group < 32768)")
    f.add_argument("--whiten", choices=("patch", "signal"), default="patch",
                   help="signal: whitening filter over the whole crop with its true history (revision 2)")
    f.add_argument("--ctx-enc", type=int, default=0, help="patches of context of the encoder on each side (even)")
    f.add_argument("--ctx-dec", type=int, default=0, help="patches of context of the decoder on each side (even)")
    f.add_argument("--dec-out", choices=("raw", "white"), default="raw",
                   help="white: the decoder draws the whitened patch, a fixed inverse FIR gives the signal (revision 3)")
    f.add_argument("--arch", choices=("conv", "skip", "linear"), default="conv",
                   help="skip: linear paths added to the convolutional ones; linear: linear paths only (dec-out white)")
    f.add_argument("--fsq-norm", action="store_true", help="batch norm (no affine) before the FSQ bound")
    f.add_argument("--fsq-scale", type=float, default=1.0,
                   help="scale of the (normalised) FSQ input before tanh; 0.5 ~ MSE-optimal for a Gaussian latent")
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
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True)
    d.add_argument("--train", action="store_true", help="train nights of the run's splits (pool items)")
    d.add_argument("--nights-from", default=None, help="meta.json of a dump whose nights to code (unit/sid keys)")
    d.add_argument("--splits", default=None, help="default: the run's")
    d.add_argument("--cache-dir", default=None, help="default: the run's source cache")
    d.add_argument("--levels", type=int, default=None, help="default: all the levels of the model")
    d.add_argument("--max-nights", type=int, default=0, help="per group (smoke tests); 0: all")
    d.add_argument("--out", default=None, help="default: <run>/dump")
    d.add_argument("--twin-out", default=None,
                   help="also write the spectral twin of the codes here (S3): residual k-means of patch spectra")
    d.add_argument("--twin-fit-nights", type=int, default=300, help="train nights for the twin's codebooks")
    d.add_argument("--twin-fit-vectors", type=int, default=1_000_000)
    t = sub.add_parser("transform", help="reference transform codec of revision 4 (no training), from a pilot run")
    t.add_argument("--from-run", required=True, help="run whose whitener, splits, items and pipeline are reused")
    t.add_argument("--out", required=True, help="directory: klt.pt (the basis), then one run directory per delta")
    t.add_argument("--deltas", default="0.35", help="comma-separated relative steps; whitened R^2 ~ 1 - delta^2 / 12")
    t.add_argument("--clip", type=float, default=8.0, help="coefficients clipped to ± clip standard deviations")
    t.add_argument("--gain-step", type=float, default=0.05, help="step of log2 RMS")
    t.add_argument("--gain-min", type=float, default=-10.0)
    t.add_argument("--gain-levels", type=int, default=320)
    t.add_argument("--groups", type=int, default=1,
                   help="blocks of KLT directions with their own gain (revision 4b; 1 = revision 4, with --clip)")
    t.add_argument("--nights", type=int, default=300, help="train nights for the basis")
    t.add_argument("--patches-per-night", type=int, default=500)
    t.add_argument("--cache-dir", default=None, help="default: the run's source cache")
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "eval":
        return _eval(a)
    if a.cmd == "dump":
        return _dump(a)
    if a.cmd == "transform":
        return _transform(a)
    a.model, a.dataset, a.load_fs, a.fs, a.root, a.resample = "patch_codec", None, None, None, None, None
    a.cache_dir_src = a.cache_dir or os.environ.get("PHYSIOEX_CACHE_DIR")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.fold)
    t0 = time.time()
    if a.train_subjects:
        sub, quota = subset_splits(json.load(open(a.splits)), a.items.split(","), a.train_subjects, a.fold)
        a.splits_src, a.splits = a.splits, os.path.join(a.out, "splits_train_subset.json")
        json.dump(sub, open(a.splits, "w"), indent=1)
        print(f"[patch_codec] train nights per item {quota} -> {a.splits}", flush=True)
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


def subset_splits(splits: dict, items, total: int, seed: int = 0):
    """Copy of ``splits`` whose train lists keep ``total`` nights over the pool ``items``, the same
    number per item (an item with fewer ids gives all of them and the rest is shared by the others),
    drawn uniformly within the item, hence in proportion to its sites. val and test are unchanged.
    Returns ``(splits, ids per item)``."""
    from physioex.data.merged import UNITS

    rng = np.random.default_rng(seed)
    sites = {it: sorted({u["site"] for u in UNITS if u.get("item") == it and u["site"] in splits}) for it in items}
    avail = {it: sum(len(splits[s]["train"]) for s in sites[it]) for it in items}
    quota, left, todo = {}, total, sorted(items, key=lambda it: avail[it])
    while todo:
        it = todo.pop(0)
        quota[it] = min(avail[it], left // (len(todo) + 1))
        left -= quota[it]
    out = json.loads(json.dumps(splits))
    for it in items:
        ids = [tuple(x) for s in sites[it] for x in splits[s]["train"]]  # [unit, night id]
        keep = {ids[i] for i in rng.choice(len(ids), quota[it], replace=False)}
        for s in sites[it]:
            out[s]["train"] = [x for x in splits[s]["train"] if tuple(x) in keep]
    return out, quota


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
                       stft_weight=a.stft_weight, restart_after=a.restart_after, crop_sec=a.crop_sec,
                       upsample=a.upsample, quantizer=a.quantizer, fsq_levels=a.fsq_levels, fsq_group=a.fsq_group,
                       whiten=a.whiten, ctx_enc=a.ctx_enc, ctx_dec=a.ctx_dec, dec_out=a.dec_out, arch=a.arch,
                       fsq_norm=a.fsq_norm, fsq_scale=a.fsq_scale).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[patch_codec] windows: train {n_tr}, valid {n_va}; {n_par / 1e6:.2f} M parameters, "
          f"{model.bits_per_patch()} bits per 2 s patch at {a.levels} levels", flush=True)
    winfo = fit_whitener(model, tr, a.whiten_batches)
    json.dump(winfo, open(os.path.join(a.out, "whitener.json"), "w"), indent=1)
    print(f"[patch_codec] whitener {winfo}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    if a.vq_warmup > 0 and a.quantizer == "rvq":  # FSQ has no codebook to initialise
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
    ev = _valid_prefixes(model, va, a.eval_batches, device, model_prefixes(model))
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
    """``(n,)`` contiguous signal -> ``(n // 200, levels)`` codes, equal to one coding of the whole signal.
    Chunks overlap by the encoder's reach (``model.encoder_reach`` patches, plus one) on each side; a local
    encoder needs none."""
    reach = getattr(model, "encoder_reach", 0)
    P = len(sig) // model.patch
    if not reach:
        step = chunk * model.patch
        return torch.cat([model.codes(sig[i:i + step][None], levels)[0] for i in range(0, P * model.patch, step)])
    m, out = reach + 1, []
    for s in range(0, P, chunk):
        lo, hi = max(0, s - m), min(P, s + chunk + m)
        c = model.codes(sig[lo * model.patch:hi * model.patch][None], levels)[0]
        out.append(c[s - lo:s - lo + min(chunk, P - s)])
    return torch.cat(out)


@torch.no_grad()
def decode_signal(model, codes: torch.Tensor, chunk: int = 2048, margin: Optional[int] = None) -> torch.Tensor:
    """``(P, k)`` codes -> ``(P * 200,)``. Chunks overlap by ``margin`` patches on each side (default: the
    decoder's receptive field plus one), so the result equals one decode of the whole sequence."""
    margin = margin or getattr(model, "receptive_patches", 1) + 1
    P, out = len(codes), []
    for s in range(0, P, chunk):
        lo, hi = max(0, s - margin), min(P, s + chunk + margin)
        y = model.decode_codes(codes[None, lo:hi])[0]
        n = min(chunk, P - s)
        out.append(y[(s - lo) * model.patch:(s - lo + n) * model.patch])
    return torch.cat(out)


OFFSETS = (0, 50, 100, 150)  # samples: 4 shifts of the 2 s patch grid, one per 0.5 s


def _dump(a):
    from physioex.data.datasets import get_dataset
    from physioex.data.merged import BY_NAME, UNITS, cached_subjects
    from physioex.models.recon_eval import channel_pipelines
    from physioex.train.bin.pretrain_combo import Nights, open_unit

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run, model = load_codec(a.run, device)
    ra = run["args"]
    levels = a.levels or model.max_levels
    assert model.n_codes <= 32767, "dump: codes are stored as int16"
    cache = a.cache_dir or ra.get("cache_dir_src") or ra.get("cache_dir")
    out = a.out or os.path.join(a.run, "dump")
    os.makedirs(out, exist_ok=True)
    jobs = []  # (group, unit, sid)
    if a.train:
        sp = json.load(open(a.splits or ra["splits"]))
        sites = sorted({u["site"] for it in ra["items"].split(",") for u in UNITS
                        if u.get("item") == it and u["site"] in sp})
        tr = [("train", u, s) for site in sites for u, s in sp[site]["train"]]
        jobs += tr[: a.max_nights] if a.max_nights else tr
    if a.nights_from:
        ev = [("eval",) + tuple(k.split("/", 1)) for k in sorted(json.load(open(a.nights_from))["nights"])]
        jobs += ev[: a.max_nights] if a.max_nights else ev
    mp = os.path.join(out, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {"nights": {}}
    meta.update({"run": os.path.abspath(a.run), "git_sha": _git("rev-parse", "HEAD"), "levels": levels,
                 "n_codes": model.n_codes, "offsets": list(OFFSETS), "patch": model.patch, "fs": model.fs})
    print(f"[patch_codec dump] {len(jobs)} nights ({sum(g == 'train' for g, _, _ in jobs)} train), "
          f"{levels} levels -> {out}", flush=True)
    readers, skipped = {}, []

    def period(unit, sid):
        """Main sleep period of a night as one contiguous signal on the device, or None (skipped)."""
        if unit not in readers:
            if unit == "dreams":
                ds = get_dataset("dreams")(channels=[ra["channel"]], pipelines=channel_pipelines(ra),
                                           sequence_length=1, cache_dir=cache)
            else:
                ds = open_unit(unit, ra, cache)
            readers[unit] = (Nights(ds, ra), cached_subjects(ds))
        night, cached = readers[unit]
        if sid not in cached:  # a night outside the cache would start the cleaner
            return None
        x, y, ts = night.get(sid)
        if unit == "dreams":
            ts = np.flatnonzero(y >= 0)  # excerpts: every scored epoch
        if len(ts) < 20:
            return None
        e0, e1 = int(ts.min()), int(ts.max()) + 1
        return torch.from_numpy(np.ascontiguousarray(x[e0:e1].reshape(-1))).to(device), y, ts, e0, e1

    twin, tmeta, tmp = None, None, None
    if a.twin_out:  # spectral twin of the codes (S3, #20): fitted once on train nights, then coded alongside
        from physioex.models.patch_codec import SpectralTwin, patch_spectra

        os.makedirs(a.twin_out, exist_ok=True)
        tp = os.path.join(a.twin_out, "twin.pt")
        if os.path.exists(tp):
            twin = SpectralTwin.from_state(torch.load(tp), device)
        else:
            tr = [(u, s) for g, u, s in jobs if g == "train"] or [(u, s) for _, u, s in jobs]  # smoke tests: any
            pick = np.random.default_rng(0).permutation(len(tr))[: a.twin_fit_nights]
            feats = [patch_spectra(got[0]) for got in (period(*tr[j]) for j in pick) if got is not None]
            feats = torch.cat(feats)
            feats = feats[torch.randperm(len(feats), generator=torch.Generator().manual_seed(0))[: a.twin_fit_vectors]
                          .to(feats.device)]
            twin = SpectralTwin.fit(feats, model.n_codes, levels)
            r2 = {l: twin.residual_r2(feats, l) for l in default_prefixes(levels)}
            torch.save(twin.state_dict(), tp)
            json.dump({"nights": len(pick), "vectors": len(feats), "levels": levels, "n_codes": model.n_codes,
                       "r2_per_prefix": r2}, open(os.path.join(a.twin_out, "twin_fit.json"), "w"), indent=1)
            print(f"[patch_codec dump] spectral twin fitted on {len(feats)} patches of {len(pick)} nights, "
                  f"R^2 per prefix {r2} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        tmp = os.path.join(a.twin_out, "meta.json")
        tmeta = json.load(open(tmp)) if os.path.exists(tmp) else {"nights": {}}
        tmeta.update({k: meta[k] for k in ("git_sha", "levels", "n_codes", "offsets", "patch", "fs")})
        tmeta.update({"run": None, "twin_of": os.path.abspath(out), "features": "patch_spectra"})
    for i, (group, unit, sid) in enumerate(jobs):
        key = f"{unit}/{sid}"
        path = os.path.join(out, unit, f"{sid}.npz")
        need = not (key in meta["nights"] and os.path.exists(path))
        tpath = os.path.join(a.twin_out, unit, f"{sid}.npz") if twin is not None else None
        need_twin = twin is not None and not (key in tmeta["nights"] and os.path.exists(tpath))
        if not (need or need_twin):
            continue
        got = period(unit, sid)
        if got is None:
            skipped.append(key)
            continue
        sig, y, ts, e0, e1 = got
        info = {"group": group, "unit": unit, "site": BY_NAME.get(unit, {}).get("site", unit),
                "e0": e0, "n_epochs": e1 - e0, "n_scored": int(len(ts)), "n_patches": len(sig) // model.patch}
        if need:
            arr = {f"c{o}": encode_signal(model, sig[o:], levels).cpu().numpy().astype(np.int16) for o in OFFSETS}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, e0=e0, n=len(sig), y=y[e0:e1].astype(np.int8), **arr)
            meta["nights"][key] = info
        if need_twin:
            arr = {f"c{o}": twin.codes(patch_spectra(sig[o:]), levels).cpu().numpy().astype(np.int16) for o in OFFSETS}
            os.makedirs(os.path.dirname(tpath), exist_ok=True)
            np.savez(tpath, e0=e0, n=len(sig), y=y[e0:e1].astype(np.int8), **arr)
            tmeta["nights"][key] = info
        if (i + 1) % 50 == 0 or i + 1 == len(jobs):
            json.dump(meta, open(mp, "w"))
            if tmeta is not None:
                json.dump(tmeta, open(tmp, "w"))
            print(f"[patch_codec dump] {i + 1}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    meta["skipped"] = sorted(set(meta.get("skipped", [])) | set(skipped))
    json.dump(meta, open(mp, "w"))
    if tmeta is not None:
        tmeta["skipped"] = meta["skipped"]
        json.dump(tmeta, open(tmp, "w"))
    print(f"[patch_codec dump] {len(meta['nights'])} nights, {len(meta['skipped'])} skipped "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("[patch_codec dump] DONE", flush=True)


def _line_ratios(psd: np.ndarray, freqs: np.ndarray, base: float = 0.5, f_max: float = 35.0, offset: float = 0.0):
    """PSD at ``k * base - offset`` Hz over the median of the bins at ±0.075-0.2 Hz, for ``k * base <= f_max``
    (``offset=0.25``: the same statistic off the grid, as a control)."""
    df = freqs[1] - freqs[0]
    near = [j for j in range(int(round(0.075 / df)), int(round(0.2 / df)) + 1)]
    out = []
    for k in range(1, int(f_max / base) + 1):
        i = int(round((k * base - offset) / df))
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


def flat_seconds(x: torch.Tensor, fs: int = 100) -> torch.Tensor:
    """``(..., n)`` -> ``(..., n // fs)`` bool: the 1 s segments whose sd is below :data:`FLAT_SD`."""
    n = x.shape[-1] // fs * fs
    return x[..., :n].reshape(*x.shape[:-1], -1, fs).float().std(-1) < FLAT_SD


def logpow_kept(pred: np.ndarray, target: np.ndarray, keep: np.ndarray, fs: int = 100) -> dict:
    """R^2 of the log band power per 1 s segment, as ``logpow`` of :func:`~physioex.models.recon_eval.band_r2`,
    over the segments in ``keep`` only (``(M * n // fs,)`` bool, segments in the order of ``reshape(-1, fs)``)."""
    from physioex.models.recon_eval import BANDS, r2

    fp = np.abs(np.fft.rfft(pred.reshape(-1, fs)[keep], axis=1)) ** 2
    ft = np.abs(np.fft.rfft(target.reshape(-1, fs)[keep], axis=1)) ** 2
    return {name: r2(np.log10(fp[:, a:b].sum(1) + 1e-6), np.log10(ft[:, a:b].sum(1) + 1e-6)) for a, b, name in BANDS}


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


def _transform(a):
    """Reference transform codec of revision 4 (:class:`~physioex.models.patch_codec.TransformCodec`): the KLT of
    the shapes (whitened patch over its RMS) of ``--patches-per-night`` random patches of ``--nights`` random train
    nights of ``--from-run`` (its whitener, splits, items and pipeline), saved in ``<out>/klt.pt`` and reused when
    present; then one run directory ``<out>/d<delta>`` per step, which ``eval`` and ``dump`` read like any run."""
    import physioex
    from physioex.data.merged import UNITS, cached_subjects
    from physioex.models.patch_codec import PATCH, TransformCodec
    from physioex.train.bin.pretrain_combo import Nights, open_unit

    t0 = time.time()
    ra = json.load(open(os.path.join(a.from_run, "run.json")))["args"]
    cache = a.cache_dir or ra.get("cache_dir_src") or ra.get("cache_dir")
    sd = torch.load(os.path.join(a.from_run, "model.pt"), map_location="cpu")
    base = TransformCodec(order=ra["order"])
    base.whitener.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("whitener.")})
    os.makedirs(a.out, exist_ok=True)
    kp = os.path.join(a.out, "klt.pt")
    if os.path.exists(kp):
        st = torch.load(kp)
        base.U.copy_(st["U"])
        base.lam.copy_(st["lam"])
        fit = st["fit"]
    else:
        sp = json.load(open(ra["splits"]))
        sites = sorted({u["site"] for it in ra["items"].split(",") for u in UNITS
                        if u.get("item") == it and u["site"] in sp})
        train = [(u, s) for site in sites for u, s in sp[site]["train"]]
        rng = np.random.default_rng(a.seed)
        pick = [train[i] for i in np.sort(rng.choice(len(train), min(a.nights, len(train)), replace=False))]
        readers, shapes, used = {}, [], []
        with torch.no_grad():
            for unit, sid in pick:
                if unit not in readers:
                    ds = open_unit(unit, ra, cache)
                    readers[unit] = (Nights(ds, ra), cached_subjects(ds))
                night, cached = readers[unit]
                if sid not in cached:  # a night outside the cache would start the cleaner
                    continue
                x, y, ts = night.get(sid)
                if len(ts) < 20:
                    continue
                e0, e1 = int(ts.min()), int(ts.max()) + 1
                w = base.whitener(torch.from_numpy(np.ascontiguousarray(x[e0:e1].reshape(-1))).float()[None])[0]
                p = w[: len(w) // PATCH * PATCH].view(-1, PATCH)
                rms = p.pow(2).mean(1).sqrt()
                ok = torch.nonzero(rms > FLAT_SD).squeeze(1)  # patches inside a gap of the clean channel have no shape
                if len(ok) == 0:
                    continue
                sel = ok[torch.from_numpy(rng.choice(len(ok), min(a.patches_per_night, len(ok)), replace=False))]
                shapes.append(p[sel] / rms[sel, None])
                used.append(f"{unit}/{sid}")
        fit = base.fit_basis(torch.cat(shapes))
        fit.update(nights=len(used), units=sorted({u.split("/")[0] for u in used}))
        torch.save({"U": base.U, "lam": base.lam, "fit": fit, "nights": used}, kp)
    print(f"[patch_codec transform] basis {fit}", flush=True)
    for d in (float(v) for v in a.deltas.split(",")):
        m = TransformCodec(delta=d, clip=a.clip, gain_step=a.gain_step, gain_min=a.gain_min,
                           gain_levels=a.gain_levels, order=ra["order"], groups=a.groups)
        m.load_state_dict(base.state_dict())
        out = os.path.join(a.out, f"d{d:g}" + (f"_g{a.groups}" if a.groups > 1 else ""))
        os.makedirs(out, exist_ok=True)
        torch.save(m.state_dict(), os.path.join(out, "model.pt"))
        args = {**ra, "arch": "transform", "quantizer": "transform", "tc_delta": d, "tc_clip": a.clip,
                "tc_gain_step": a.gain_step, "tc_gain_min": a.gain_min, "tc_gain_levels": a.gain_levels,
                "tc_groups": a.groups, "from_run": os.path.abspath(a.from_run), "out": out}
        run = {"args": args, "git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
               "physioex_version": getattr(physioex, "__version__", None), "host": socket.gethostname(),
               "job_id": os.environ.get("SLURM_JOB_ID"), "basis": fit, "bits_per_patch": m.bits_per_patch(),
               "expected_r2_w": 1 - d * d / 12, "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(run, open(os.path.join(out, "run.json"), "w"), indent=1, default=str)
        print(f"[patch_codec transform] delta {d:g}: {m.bits_per_patch()} bits per patch, whitened R^2 about "
              f"{1 - d * d / 12:.4f} -> {out}", flush=True)
    print("[patch_codec transform] DONE", flush=True)


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
    prefixes = a.prefixes or model_prefixes(model)
    cache = a.cache_dir or ra.get("cache_dir_src") or ra.get("cache_dir")
    md = build_test(a.splits or ra["splits"], [ra["channel"]], channel_pipelines(ra), 1, cache,
                    units=a.units.split(","), memmap_cache_size=4, trim_mode="main_period")
    nights = _pick_nights(md.test, md.unit_names, a.units.split(","), a.per_unit, a.seed)
    print(f"[patch_codec eval] {len(nights)} nights, prefixes {prefixes}", flush=True)
    rng = np.random.default_rng(a.seed)
    W = model.whitener
    R = len(RULES)
    cnt = {k: {"all": np.zeros((R, 3))} for k in prefixes}  # tp, fp, fn per rule
    cnt_v = {k: {"all": np.zeros((R, 3))} for k in prefixes}  # the same over the epochs without a flat second
    n_ep = np.zeros(2)  # scored epochs: all, with a flat second
    seam = {k: np.zeros(6) for k in prefixes}  # seam sum, n; interior sum, n; patch-centre sum, n (control)
    comb = {k: [0.0, 0] for k in prefixes}
    comb_x = [0.0, 0]
    fid = {k: np.zeros(2) for k in prefixes}  # whitened and raw squared error over the scored samples
    tot = np.zeros(5)  # sum and sum of squares of the whitened and raw signal over the scored samples, count
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
        ok = ~flat_seconds(X, model.fs).any(-1)  # epochs without a gap of the clean channel
        n_ep += [len(X), int((~ok).sum())]
        codes = encode_signal(model, sig, model.max_levels)
        m0 = rule_masks(X, model.fs, 60) > 0
        pos = torch.arange(len(sig), device=device)
        in_scored = torch.zeros(len(sig) // 3000, dtype=torch.bool, device=device)
        in_scored[sc] = True
        in_scored = in_scored.repeat_interleave(3000)
        near = ((pos % model.patch) < SEAM) | ((pos % model.patch) >= model.patch - SEAM)
        centre = ((pos % model.patch) >= model.patch // 2 - SEAM) & ((pos % model.patch) < model.patch // 2 + SEAM)
        xw = W(sig[None])[0]
        xs_w, xs_r = xw[in_scored].double(), sig[in_scored].double()
        tot += [float(xs_w.sum()), float(xs_w.pow(2).sum()), float(xs_r.sum()), float(xs_r.pow(2).sum()), len(xs_w)]
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
            cv = torch.stack([(m0 & m1)[ok].sum((0, 1)), (~m0 & m1)[ok].sum((0, 1)), (m0 & ~m1)[ok].sum((0, 1))],
                             -1).cpu().numpy()
            cnt_v[k]["all"] += cv
            cnt_v[k].setdefault(unit, np.zeros((R, 3)))
            cnt_v[k][unit] += cv
            ew = (W(h[None])[0] - xw).pow(2)
            fid[k] += [float(ew[in_scored].sum()), float((h - sig)[in_scored].pow(2).sum())]
            seam[k] += [float(ew[near & in_scored].sum()), int((near & in_scored).sum()),
                        float(ew[~near & in_scored].sum()), int((~near & in_scored).sum()),
                        float(ew[centre & in_scored].sum()), int((centre & in_scored).sum())]
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
    keep = ~flat_seconds(torch.from_numpy(xs), model.fs).reshape(-1).numpy()  # band sample: seconds that are not flat
    px = comb_x[0] / max(comb_x[1], 1)
    ctrl, ctrl_off = _line_ratios(px, freqs), _line_ratios(px, freqs, offset=0.25)
    res = {"run": os.path.abspath(a.run), "n_nights": len(used),
           "units": {u: sum(1 for n in used if n.startswith(u + "/")) for u in a.units.split(",")},
           "n_band_epochs": int(len(xs)), "gate": {**GATE, "bands": GATE_BANDS},
           "comb_control_x": {"median": float(np.median(ctrl)), "max": float(ctrl.max()),
                              "median_offgrid": float(np.median(ctrl_off))}, "prefix": {},
           "flat": {"sd_below": FLAT_SD, "band_seconds": int(len(keep)), "band_seconds_flat": int((~keep).sum()),
                    "scored_epochs": int(n_ep[0]), "scored_epochs_with_flat_second": int(n_ep[1])}}

    sst_w, sst_r = tot[1] - tot[0] ** 2 / max(tot[4], 1), tot[3] - tot[2] ** 2 / max(tot[4], 1)

    def f1(c):
        tp, fp, fn = c
        return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 1.0

    def gate0(cn):
        return ({r: {"f1": f1(cn["all"][i]), "n_ref": int(cn["all"][i][0] + cn["all"][i][2]),
                     "n_hat": int(cn["all"][i][0] + cn["all"][i][1])} for i, r in enumerate(RULES)},
                {u: {r: f1(v[i]) for i, r in enumerate(RULES)} for u, v in cn.items() if u != "all"})

    for k in prefixes:
        hs = np.concatenate(sub_h[k])
        br = band_r2(hs, xs)
        lp_v = logpow_kept(hs, xs, keep, model.fs)
        g0, per_unit = gate0(cnt[k])
        g0v, per_unit_v = gate0(cnt_v[k])
        s = seam[k]
        seam_ratio = float((s[0] / max(s[1], 1)) / max(s[2] / max(s[3], 1), 1e-12))
        seam_centre = float((s[0] / max(s[1], 1)) / max(s[4] / max(s[5], 1), 1e-12))
        pk = comb[k][0] / max(comb[k][1], 1)
        lines, off = _line_ratios(pk, freqs), _line_ratios(pk, freqs, offset=0.25)
        comb_rel = float(np.median(lines) / np.median(off))
        ph = {n: float(1 - v[0] / max(v[1], 1e-12)) for n, v in phase[k].items()}
        r2_w, r2_r = float(1 - fid[k][0] / max(sst_w, 1e-12)), float(1 - fid[k][1] / max(sst_r, 1e-12))
        registered = {"f1": all(g0[r]["f1"] >= GATE["f1"] for r in RULES),
                      "logpow": all(br["logpow"][b] >= GATE["logpow"] for b in GATE_BANDS),
                      "seam": seam_ratio <= GATE["seam"], "comb": float(np.median(lines)) <= GATE["comb"],
                      "r2": r2_w >= GATE["r2"]}  # whitened R^2 of the scored nights (revision 2)
        checks = {**registered, "comb": comb_rel <= GATE["comb"]}  # binding since 26.11.2026
        controlled = {**checks, "seam": seam_centre <= GATE["seam"]}
        # amendment of 28.11.2026: F1 and log power without the gaps of the clean channel (FLAT_SD); binding from then
        valid = {**checks, "f1": all(g0v[r]["f1"] >= GATE["f1"] for r in RULES),
                 "logpow": all(lp_v[b] >= GATE["logpow"] for b in GATE_BANDS)}
        res["prefix"][str(k)] = {"bits_per_patch": model.bits_per_patch(k), "r2_w_nights": r2_w, "r2_raw_nights": r2_r,
                                 "gate0": g0, "gate0_per_unit": per_unit,
                                 "band_r2": br, "seam_ratio": seam_ratio, "seam_vs_centre": seam_centre,
                                 "comb": {"median": float(np.median(lines)), "max": float(lines.max()),
                                          "worst_hz": float(0.5 * (1 + int(lines.argmax()))),
                                          "median_offgrid": float(np.median(off)), "on_over_off": comb_rel},
                                 "phase_test": {"r2_w_real": ph["real"], "r2_w_surrogate": ph["surr"],
                                                "gap": ph["real"] - ph["surr"]},
                                 "checks": checks, "pass": all(checks.values()),
                                 "checks_registered": registered, "pass_registered": all(registered.values()),
                                 "checks_controlled": controlled, "pass_controlled": all(controlled.values()),
                                 "gate0_valid": g0v, "gate0_valid_per_unit": per_unit_v, "logpow_valid": lp_v,
                                 "checks_valid": valid, "pass_valid": all(valid.values())}
        print(f"[patch_codec eval] L{k}: R2 w {r2_w:.4f} raw {r2_r:.4f} | " + " ".join(f"{r} {g0[r]['f1']:.3f}" for r in RULES)
              + f" | logpow min {min(br['logpow'][b] for b in GATE_BANDS):.3f} | seam {seam_ratio:.3f}"
              + f" (vs centre {seam_centre:.3f}) | comb {np.median(lines):.3f} (on/off {comb_rel:.3f})"
              + f" | phase gap {ph['real'] - ph['surr']:+.3f} | pass {all(checks.values())}"
              + f" (registered {all(registered.values())}, controlled {all(controlled.values())})"
              + " | without gaps: " + " ".join(f"{r} {g0v[r]['f1']:.3f}" for r in RULES)
              + f" logpow min {min(lp_v[b] for b in GATE_BANDS):.3f} pass {all(valid.values())}",
              flush=True)
    passing = [k for k in prefixes if res["prefix"][str(k)]["pass"]]
    res["L_star"] = min(passing) if passing else None
    passing_r = [k for k in prefixes if res["prefix"][str(k)]["pass_registered"]]
    res["L_star_registered"] = min(passing_r) if passing_r else None
    passing_c = [k for k in prefixes if res["prefix"][str(k)]["pass_controlled"]]
    res["L_star_controlled"] = min(passing_c) if passing_c else None
    passing_v = [k for k in prefixes if res["prefix"][str(k)]["pass_valid"]]
    res["L_star_valid"] = min(passing_v) if passing_v else None
    res["minutes"] = round((time.time() - t0) / 60, 1)
    res["git_sha"] = _git("rev-parse", "HEAD")
    out = a.out or os.path.join(a.run, "gates.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[patch_codec eval] L* = {res['L_star']} (controlled {res['L_star_controlled']}, without gaps "
          f"{res['L_star_valid']}) -> {out}", flush=True)
    print("[patch_codec eval] DONE", flush=True)


if __name__ == "__main__":
    main()
