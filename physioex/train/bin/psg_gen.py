"""Fit and dump the generative PSG models P1 / P2 / P3 (:mod:`physioex.models.psg_gen`).

    python -m physioex.train.bin.psg_gen fit --model p1|p2|p3 [--variant full|fixed|random|noadv|events_only]
        --items mass --splits S.json --cache-dir C --out RUN [--steps-per-epoch 2000 --epochs 5 --batch 64]
    python -m physioex.train.bin.psg_gen dump --run RUN --out DIR [--batches 50]

``fit``: P1 has no trainable parameters; its fit is :meth:`PSGGenP1.fit_transients` on
``--fit-batches`` training batches. P2 and P3 train with :class:`PSGGenTrainer`. The null
thresholds (P1, P2) are estimated on phase-randomised surrogates of the first training batches; P3
uses its closed-form gate. Writes ``model.pt``, ``ckpt/`` (P2, P3), ``eval.json`` (validation
terms), ``gate0.json`` (R12 fidelity gate on validation windows), ``run.json`` (``model:
psg_gen`` so :func:`physioex.models.recon_eval.model_from_args` rebuilds it for the dump tools,
including ``concept_scan dump``).

``dump``: per validation window the events (``Events.to_dict``) of the centre epoch with the
night id and epoch index (``events.json``), the background (``bg.npz``: logP, change, eps, gain)
and ``N``/``E``/``P``; aggregate numbers only are meant to leave the cluster.
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

from physioex.models.psg_gen.inference import null_thresholds
from physioex.models.psg_gen.model import build_from_args, fidelity_gate
from physioex.models.psg_gen.p1 import phase_surrogate
from physioex.models.psg_gen.trainer import PSGGenTrainer
from physioex.models.recon_eval import load_best
from physioex.train.bin.msvq import _merged_data, _merged_loaders

BUDGET = {0: 14, 1: 4, 2: 20}


def _centre(batch):
    from physioex.data.collate import stack_channels

    x = stack_channels(batch)
    return x, x[:, x.shape[1] // 2, 0].float()


def _git(*cmd):
    try:
        return subprocess.check_output(["git", *cmd], cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:
        return None


def _data_args(ap):
    ap.add_argument("--items", required=True, help="comma-separated pool items (physioex.data.merged)")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--local-cache", action="store_true")
    ap.add_argument("--channel", default="EEG_clean")
    ap.add_argument("--seq", type=int, default=41)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--valid-per-item", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    _data_args(f)
    f.add_argument("--model", dest="gen_model", choices=("p1", "p2", "p3"), required=True)
    f.add_argument("--variant", default="full", choices=("full", "fixed", "random", "noadv", "events_only"))
    f.add_argument("--out", required=True)
    f.add_argument("--epochs", type=int, default=5)
    f.add_argument("--patience", type=int, default=3)
    f.add_argument("--valid-ratio", type=float, default=0.5)
    f.add_argument("--lr", type=float, default=3e-3)
    f.add_argument("--grad-clip", type=float, default=1.0)
    f.add_argument("--null-batches", type=int, default=4, help="training batches for the null thresholds")
    f.add_argument("--fit-batches", type=int, default=20, help="P1: training batches for fit_transients")
    f.add_argument("--fit-iters", type=int, default=2)
    f.add_argument("--eval-batches", type=int, default=50)
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--batches", type=int, default=50)
    d.add_argument("--cache-dir", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "dump":
        return _dump(a)
    a.model, a.dataset, a.load_fs, a.fs, a.root, a.resample = "psg_gen", None, None, None, None, None
    a.pow_bands = a.psd_band = a.recon = a.seg_sec = None
    a.budget = BUDGET
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.fold)
    t0 = time.time()
    local = None
    if a.local_cache:
        from physioex.data.merged import stage_local

        src = a.cache_dir or os.environ.get("PHYSIOEX_CACHE_DIR")
        local = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                             f"physioex_cache_{os.environ.get('SLURM_JOB_ID', os.getpid())}_psggen")
        stage_local(a.items.split(","), a.splits, src, local, [a.channel])
        a.cache_dir = local
        import signal
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        _fit(a, t0)
    finally:
        if local:
            shutil.rmtree(local, ignore_errors=True)


def _evaluate(model, loader, n_batches, device):
    acc, gates = {}, []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            x, xc = _centre(batch)
            out = model(x.to(device))
            out["night"] = None
            _, a_, extra = model.loss(out, xc.to(device), None)
            extra["acc"] = float(a_)
            for k, v in extra.items():
                if isinstance(v, (int, float)):
                    acc.setdefault(k, []).append(float(v))
            gates.append(fidelity_gate(xc.to(device), out["wave"].float()))
    ev = {k: float(np.mean(v)) for k, v in acc.items()}
    gate = {}
    for r in gates[0] if gates else {}:
        if isinstance(gates[0][r], dict):
            n = {k: sum(g[r][k] for g in gates) for k in ("n_ref", "n_hat")}
            gate[r] = {"f1_mean": float(np.mean([g[r]["f1"] for g in gates])), **n}
        else:
            gate[r] = float(np.mean([g[r] for g in gates]))
    return ev, gate


def _fit(a, t0):
    md, ds, train_ids, valid_ids, _ = _merged_data(a)
    tr, n_tr, va, n_va = _merged_loaders(a, md, ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[psg_gen] windows: train {n_tr}, valid {n_va}; model {a.gen_model}/{a.variant}", flush=True)
    model = build_from_args(vars(a)).to(device)
    # null thresholds on phase-randomised surrogates of training epochs (P3: closed form)
    if a.gen_model != "p3":
        xs = []
        for i, batch in enumerate(tr):
            if i >= a.null_batches:
                break
            xs.append(_centre(batch)[1])
        x0 = torch.cat(xs).to(device)
        with torch.no_grad():   # P2: model.d holds the learnable shapes; the buffer must stay a leaf
            model.thresholds.copy_(null_thresholds(model.d, phase_surrogate(x0)))
    if a.gen_model == "p1":
        batches = []
        for i, batch in enumerate(tr):
            if i >= a.fit_batches:
                break
            batches.append(_centre(batch)[1])
        log = model.fit_transients(batches, n_iter=a.fit_iters)
        json.dump(log, open(os.path.join(a.out, "fit_log.json"), "w"), indent=1, default=str)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
        n_val = a.epochs * max(1, round(1 / a.valid_ratio))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_val, eta_min=a.lr / 20)
        model = PSGGenTrainer.train(model, (tr, va), checkpoint_path=os.path.join(a.out, "ckpt"), max_epochs=a.epochs,
                                    lr=a.lr, weight_decay=1e-4, valid_interval_ratio=a.valid_ratio,
                                    log_confusion_matrix=False, early_stopping_patience=a.patience,
                                    logger="tensorboard", run_name=f"psg_gen_{a.gen_model}", seed=a.fold,
                                    gpu_id=0 if torch.cuda.is_available() else None, grad_clip=a.grad_clip or None,
                                    optimizer=opt, scheduler=sched)
        load_best(model, a.out)
        device = next(model.parameters()).device
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    ev, gate = _evaluate(model, va, a.eval_batches, device)
    ev["n_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ev["bits_per_epoch"] = model.bits_per_epoch()
    json.dump(ev, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    json.dump(gate, open(os.path.join(a.out, "gate0.json"), "w"), indent=1)
    print("[psg_gen] valid " + " ".join(f"{k}={v:.3f}" for k, v in ev.items()), flush=True)
    print("[psg_gen] gate0 " + json.dumps(gate), flush=True)

    import physioex
    run = {"args": vars(a), "git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
           "physioex_version": getattr(physioex, "__version__", None), "host": socket.gethostname(),
           "job_id": os.environ.get("SLURM_JOB_ID"), "merged": md.report,
           "splits": {"train": train_ids, "valid": valid_ids, "test": []},
           "windows": {"train": n_tr, "valid": n_va}, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(run, open(os.path.join(a.out, "run.json"), "w"), indent=1, default=str)
    print("[psg_gen] DONE", flush=True)


def _dump(a):
    run = json.load(open(os.path.join(a.run, "run.json")))
    ra = argparse.Namespace(**run["args"])
    if a.cache_dir:
        ra.cache_dir = a.cache_dir
    model = build_from_args(run["args"])
    model.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location="cpu"), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    md, ds, _, _, _ = _merged_data(ra)
    _, _, va, _ = _merged_loaders(ra, md, ds)
    os.makedirs(a.out, exist_ok=True)
    evs, bg, nep = [], {k: [] for k in ("logP", "change", "eps", "N", "E", "P")}, []
    with torch.no_grad():
        for i, batch in enumerate(va):
            if i >= a.batches:
                break
            x, _ = _centre(batch)
            dec = model.encode(x.to(device))
            e = dec["events"].to("cpu").to_dict()
            subs = batch.get("subject") or [{}] * x.shape[0]
            e["night"] = [str(subs[j].get("id", "")) for j in e["epoch"]]
            e["epoch_index"] = [int(batch["epoch_indices"][j, x.shape[1] // 2]) for j in e["epoch"]]
            e["batch"] = i
            evs.append(e)
            for k in bg:
                bg[k].append(dec[k].float().cpu().numpy().astype(np.float16 if k in "NEP" else np.float32))
            nep.extend((str(s.get("id", "")), int(batch["epoch_indices"][j, x.shape[1] // 2]))
                       for j, s in enumerate(subs))
    json.dump(evs, open(os.path.join(a.out, "events.json"), "w"))
    np.savez_compressed(os.path.join(a.out, "bg.npz"), **{k: np.concatenate(v) for k, v in bg.items()},
                        night=np.array([n for n, _ in nep]), epoch_index=np.array([t for _, t in nep]))
    print(f"[psg_gen] dump {len(nep)} windows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
