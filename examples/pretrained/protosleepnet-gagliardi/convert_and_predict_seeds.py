"""Convert seed checkpoint to model.pt and generate clean predictions.

For each seed directory:
1. Extracts model_state_dict from best checkpoint → model.pt
2. Creates config.json metadata
3. Runs clean sliding-window evaluation on in-domain test subjects
4. Saves predictions (per-subject proba + labels) and metrics

Usage:
    python convert_and_predict_seeds.py \
        --seed_dir /results/seed_123 --backbone seq --dataset mass \
        --seq_len 20 --gpu_id 0
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from build_protosleepnet import build_model
from train import build_dataset, BACKBONE_CONFIGS, MIXER_KWARGS

from physioex.data.collate import stack_channels
from physioex.train.trainer import Trainer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


@torch.no_grad()
def evaluate_subject(model, inputs, L, device):
    """Sliding-window voting for one subject (same as test_prototypes.py)."""
    inputs = inputs.to(device)
    night_length = inputs.shape[1]

    if night_length < L:
        y = model(inputs)
        return F.softmax(y.squeeze(0), dim=-1).cpu()

    probe = model(inputs[:, :L])
    n_classes = probe.shape[-1]
    votes = torch.zeros(1, night_length, n_classes, device=device, dtype=probe.dtype)
    counts = torch.zeros(1, night_length, device=device, dtype=torch.float32)

    for offset in range(L):
        x = inputs[:, offset:]
        usable = x.shape[1] - (x.shape[1] % L)
        if usable == 0:
            continue
        x = x[:, :usable]
        num_windows = usable // L
        rest_dims = x.shape[2:]
        x = x.reshape(num_windows, L, *rest_dims)
        y = model(x)
        y = y.reshape(1, num_windows * L, n_classes)
        votes[:, offset:offset + usable] += y
        counts[:, offset:offset + usable] += 1

    safe_counts = counts.clamp(min=1).unsqueeze(-1)
    logits = votes / safe_counts
    return F.softmax(logits.squeeze(0), dim=-1).cpu()


def compute_metrics(all_proba, all_labels):
    """Compute per-subject metrics, ignoring label=-1."""
    kappas, accs, f1s = [], [], []
    for proba, labels in zip(all_proba, all_labels):
        y_true = np.array(labels)
        y_proba = np.array(proba)
        mask = y_true >= 0
        y_true = y_true[mask]
        y_pred = np.argmax(y_proba[mask], axis=1)
        if len(y_true) == 0:
            continue
        kappas.append(cohen_kappa_score(y_true, y_pred))
        accs.append(accuracy_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "cohen_kappa": float(np.mean(kappas)),
        "cohen_kappa_std": float(np.std(kappas)),
        "f1_macro": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "n_subjects": len(kappas),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint + predict")
    parser.add_argument("--seed_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True, choices=["seq", "st"])
    parser.add_argument("--dataset", type=str, default="mass")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    seed_dir = args.seed_dir
    seed_name = os.path.basename(seed_dir)
    model_path = os.path.join(seed_dir, "model.pt")
    config_path = os.path.join(seed_dir, "config.json")
    pred_path = os.path.join(seed_dir, "predictions_clean.json")
    metrics_path = os.path.join(seed_dir, "metrics_clean.json")

    # Skip if already done
    if os.path.exists(pred_path) and os.path.exists(metrics_path):
        print(f"SKIP {seed_name}: predictions already exist")
        return

    # ── Step 1: Convert checkpoint → model.pt ───────────────────
    if not os.path.exists(model_path):
        ckpts = sorted(glob.glob(os.path.join(seed_dir, "checkpoints", "epoch=*.pt")))
        if not ckpts:
            print(f"ERROR {seed_name}: no checkpoint found")
            return
        ckpt_path = ckpts[-1]
        print(f"[1/3] Converting {os.path.basename(ckpt_path)} → model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        torch.save(ckpt["model_state_dict"], model_path)

        # Save config
        cfg = BACKBONE_CONFIGS[args.backbone]
        config = {
            "model_class": "physioex.models.protosleepnet:ProtoSleepNet",
            "factory": cfg["factory"],
            "factory_kwargs": {"n_channels": 3, **MIXER_KWARGS},
            "training": {
                "backbone": args.backbone,
                "dataset": args.dataset,
                "seed": int(seed_name.replace("seed_", "")),
                "checkpoint": os.path.basename(ckpt_path),
                "epoch": ckpt.get("epoch", "unknown"),
            },
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved model.pt ({os.path.getsize(model_path) / 1e6:.1f} MB) + config.json")
    else:
        print(f"[1/3] SKIP: model.pt exists")

    # ── Step 2: Load model ──────────────────────────────────────
    print(f"[2/3] Loading model on {device}")
    model = build_model(backbone=args.backbone)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # ── Step 3: Run clean predictions ───────────────────────────
    print(f"[3/3] Running clean evaluation on {args.dataset} test split")
    t0 = time.time()

    cfg = BACKBONE_CONFIGS[args.backbone]
    ds_extra = cfg["dataset_kwargs"].get(args.dataset, {})
    dataset = build_dataset(args.dataset, cfg["channels_3ch"], "seqsleepnet",
                            args.seq_len, ds_extra)
    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, fold=args.fold,
    )

    all_predictions = []
    all_proba = []
    all_labels = []

    for subj_idx, batch in enumerate(test_loader):
        inputs = stack_channels(batch)
        targets = batch["labels"].reshape(-1).tolist()

        proba = evaluate_subject(model, inputs, args.seq_len, device)
        proba_list = proba.numpy().tolist()

        all_predictions.append({
            "subject_idx": subj_idx,
            "proba": proba_list,
            "labels": targets,
        })
        all_proba.append(proba_list)
        all_labels.append(targets)

    elapsed = time.time() - t0
    metrics = compute_metrics(all_proba, all_labels)
    metrics["elapsed_s"] = elapsed

    with open(pred_path, "w") as f:
        json.dump(all_predictions, f)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  {metrics['n_subjects']} subjects, "
          f"kappa={metrics['cohen_kappa']:.4f}±{metrics['cohen_kappa_std']:.4f}, "
          f"acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f} "
          f"({elapsed:.0f}s)")
    print(f"  Saved → {pred_path}")


if __name__ == "__main__":
    main()
