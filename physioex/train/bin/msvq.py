"""Train MSVQ (multi-scale residual VQ) or MSAE (continuous hierarchical autoencoder)
and its decoder on one dataset, clean EEG only.

    python -m physioex.train.bin.msvq --dataset dcsm --out <run dir> [--cache-dir ...] [--root ...]
    python -m physioex.train.bin.msvq --model msae --d-p 8 --token-sec 1.0 --out <run dir> ...

Encoder, dictionaries and decoder are trained together (``MSVQTrainer``); ``--no-vq``
trains the same network with continuous latents (the upper bound used to size the
dictionaries), ``--ce-weight`` adds the label-free cluster targets as auxiliary CE.
Windows are restricted to the main sleep period (``trim_mode="main_period"``);
validation uses windows of the validation subjects (not whole recordings).
Writes the best checkpoint in ``ckpt/``, the final weights ``model.pt``, the
validation metrics ``eval.json`` (R^2 per prefix, code usage), the reconstruction
audit of the test nights ``recon_eval.json`` (:mod:`physioex.models.recon_eval`)
and ``run.json``. MSAE runs use a cosine learning rate over the whole training.
"""
import argparse
import json
import os
import socket
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from physioex.data.collate import dict_collate_fn
from physioex.data.datasets import get_dataset
from physioex.data.steps import clean_pipelines
from physioex.models.msae import MSAE, MSAETrainer
from physioex.models.msvq import (
    MSVQ, MSVQTrainer, compute_msvq_targets, fit_msvq_norms,
)
from physioex.models.recon_eval import evaluate as recon_evaluate


def _loader(ds, ids, n_max, batch, workers, shuffle, seed):
    idx = np.asarray(ds._subject_ids_to_flat_indices(ids))
    if n_max and len(idx) > n_max:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n_max, replace=False))
    kw = dict(batch_size=batch, shuffle=shuffle, num_workers=workers, collate_fn=dict_collate_fn,
              pin_memory=True, persistent_workers=workers > 0)
    if workers > 0:
        kw["prefetch_factor"] = 4
    return DataLoader(Subset(ds, idx.tolist()), **kw), len(idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dcsm")
    ap.add_argument("--root", default=None, help="raw root (default: dataset default under PHYSIOEX_DATA)")
    ap.add_argument("--cache-dir", default=None, help="cache root (the folder that contains v1/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seq", type=int, default=41)
    ap.add_argument("--model", choices=("msvq", "msae"), default="msvq")
    ap.add_argument("--dim", type=int, default=None, help="latent width (default: 64 msvq, 128 msae)")
    ap.add_argument("--width", type=int, default=32, help="msae: channels of the first SEANet stage")
    ap.add_argument("--d-p", type=int, default=8, help="msae: numbers per P token")
    ap.add_argument("--token-sec", type=float, default=1.0, choices=(1.0, 0.5), help="msae: P token length (s)")
    ap.add_argument("--latent-bits", type=int, default=8, help="msae: bit channel of E and P (0 = tanh only)")
    ap.add_argument("--codes", type=int, nargs=3, default=(32, 128, 256), metavar=("N", "E", "P"))
    ap.add_argument("--targets", type=int, nargs=2, default=(64, 64), metavar=("E", "P"))
    ap.add_argument("--no-vq", action="store_true", help="continuous latents (upper bound)")
    ap.add_argument("--restart-after", type=int, default=30, help="steps a code may stay unused before restart")
    ap.add_argument("--ce-weight", type=float, default=0.0, help="weight of the auxiliary cluster CE (0 = off)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=0.0, help="max gradient norm (0 = no clipping)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=0, help="cap on training windows per epoch (0 = all; smoke tests)")
    ap.add_argument("--max-valid", type=int, default=20000)
    ap.add_argument("--valid-ratio", type=float, default=0.25, help="validations per training epoch = 1/ratio")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-subjects", type=int, default=0, help="use only the first N subjects (smoke tests)")
    ap.add_argument("--recon-per-night", type=int, default=250, help="test epochs per night in recon_eval (0 = skip)")
    a = ap.parse_args()
    a.dim = a.dim or (128 if a.model == "msae" else 64)

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.fold)
    cls = get_dataset(a.dataset)
    kw = dict(channels=["EEG_clean"], pipelines=clean_pipelines(), sequence_length=a.seq,
              cache_dir=a.cache_dir, trim_mode="main_period")
    if a.root:
        kw["root"] = a.root
    ds = cls(**kw)
    if a.max_subjects:
        keep = {s.subject_id for s in ds._subjects[: a.max_subjects]}
        ds._subjects = [s for s in ds._subjects if s.subject_id in keep]
        ds._build_index()
    train_ids, valid_ids, test_ids = ds.get_splits(fold=a.fold)
    print(f"[msvq] {a.dataset}: {len(ds._subjects)} subjects, {len(ds)} windows "
          f"(train/valid/test subjects {len(train_ids)}/{len(valid_ids)}/{len(test_ids)})", flush=True)

    t0 = time.time()
    targets = None
    if a.ce_weight > 0:
        tpath = os.path.join(a.out, "targets.pt")
        if os.path.exists(tpath):
            targets = torch.load(tpath, weights_only=False)
        else:
            targets = compute_msvq_targets(ds, train_ids, *a.targets, seed=a.fold)
            torch.save(targets, tpath)
        print(f"[msvq] targets ready in {time.time() - t0:.0f}s", flush=True)

    tr, n_tr = _loader(ds, train_ids, a.max_train, a.batch, a.workers, True, a.fold)
    va, n_va = _loader(ds, valid_ids, a.max_valid, a.batch, a.workers, False, a.fold)
    print(f"[msvq] windows: train {n_tr}, valid {n_va}", flush=True)

    opt_kw = {}
    if a.model == "msae":
        trainer = MSAETrainer
        model = MSAE(a.seq, width=a.width, dim=a.dim, d_p=a.d_p, token_sec=a.token_sec, latent_bits=a.latent_bits)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        n_val = a.epochs * max(1, round(1 / a.valid_ratio))  # the Trainer steps the scheduler at each validation
        opt_kw = dict(optimizer=opt, scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_val, eta_min=a.lr / 20))
        print(f"[msae] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, "
              f"{model.bits_per_epoch()} bits per epoch", flush=True)
    else:
        trainer = MSVQTrainer
        n_te, n_tp = (a.targets if targets is not None else (0, 0))
        model = MSVQ(a.seq, a.dim, *a.codes, quantize=not a.no_vq, n_targets_e=n_te, n_targets_p=n_tp,
                     restart_after=a.restart_after)
        model.targets, model.ce_weight = targets, a.ce_weight
    fit_msvq_norms(model, tr)
    model = trainer.train(model, (tr, va), checkpoint_path=os.path.join(a.out, "ckpt"), max_epochs=a.epochs,
                          lr=a.lr, weight_decay=1e-4, valid_interval_ratio=a.valid_ratio,
                          log_confusion_matrix=False, early_stopping_patience=a.patience,
                          logger="tensorboard", run_name=a.model, seed=a.fold,
                          gpu_id=0 if torch.cuda.is_available() else None, grad_clip=a.grad_clip or None,
                          **opt_kw)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))

    # validation metrics of the final model, for the K sweep
    device = next(model.parameters()).device
    model.eval()
    acc, codes = {}, {"N": [], "E": [], "P": []}
    with torch.no_grad():
        for batch in va:
            _, _, extra = trainer._step(model, batch, None, device)
            for k, v in extra.items():
                acc.setdefault(k, []).append(v)
            if getattr(model, "quantize_codes", False):
                from physioex.data.collate import stack_channels
                q = model.quantize(stack_channels(batch).to(device))
                for s_, k_ in (("N", "k_n"), ("E", "k_e"), ("P", "k_p")):
                    codes[s_].append(q[k_].reshape(-1).cpu())
    ev = {k: float(np.mean(v)) for k, v in acc.items() if not k.startswith(("perplexity", "used"))}
    if getattr(model, "quantize_codes", False):
        from physioex.models.msvq import code_stats
        for s_ in codes:
            perp, used = code_stats(torch.cat(codes[s_]), model.n_codes[s_])
            ev[f"perplexity_{s_}"], ev[f"used_{s_}"] = perp, used
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump(ev, f, indent=1)
    print("[msvq] valid " + " ".join(f"{k}={v:.3f}" for k, v in ev.items()), flush=True)

    if a.recon_per_night:
        rec = recon_evaluate(model, ds, test_ids, train_ids, per_night=a.recon_per_night, seed=a.fold)
        with open(os.path.join(a.out, "recon_eval.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[msvq] recon_eval: {rec['n_test_epochs']} test epochs written", flush=True)

    def _git(*cmd):
        try:
            return subprocess.check_output(["git", *cmd], cwd=os.path.dirname(__file__), text=True).strip()
        except Exception:
            return None

    import physioex
    run = {"args": vars(a), "git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
           "physioex_version": getattr(physioex, "__version__", None), "host": socket.gethostname(),
           "job_id": os.environ.get("SLURM_JOB_ID"), "n_subjects": len(ds._subjects),
           "splits": {"train": train_ids, "valid": valid_ids, "test": test_ids},
           "windows": {"train": n_tr, "valid": n_va}, "minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(a.out, "run.json"), "w") as f:
        json.dump(run, f, indent=1)
    print("[msvq] DONE", flush=True)


if __name__ == "__main__":
    main()
