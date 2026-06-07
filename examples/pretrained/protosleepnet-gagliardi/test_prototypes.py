"""Test ProtoSleepNet with VQ prototypes using model.set_codebook + forward(quantize=True).

Usage:
    python examples/pretrained/protosleepnet-gagliardi/test_prototypes.py \
        --backbone seq --checkpoint /path/to/model.pt \
        --codebook_path /path/to/codebook.npy \
        --dataset mass --seq_len 20 --gpu_id 0 --output_dir /results
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from physioex.data.datasets import get_dataset
from physioex.data.collate import stack_channels
from physioex.train.trainer import Trainer

sys.path.insert(0, os.path.dirname(__file__))
from build_protosleepnet import build_model

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def load_model(backbone, checkpoint_path, device):
    model = build_model(backbone=backbone)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()


@torch.no_grad()
def evaluate_subject(model, inputs, L, device):
    inputs = inputs.to(device)
    night_length = inputs.shape[1]

    if night_length < L:
        y = model(inputs, quantize=True)
        return F.softmax(y.squeeze(0), dim=-1).cpu()

    probe = model(inputs[:, :L], quantize=True)
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
        y = model(x, quantize=True)
        y = y.reshape(1, num_windows * L, n_classes)
        votes[:, offset : offset + usable] += y
        counts[:, offset : offset + usable] += 1

    safe_counts = counts.clamp(min=1).unsqueeze(-1)
    logits = votes / safe_counts
    return F.softmax(logits.squeeze(0), dim=-1).cpu()


def compute_metrics(all_proba, all_targets, ignore_index=-1):
    preds = torch.cat(all_proba, dim=0)
    targets = torch.cat(all_targets, dim=0)
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]
    pred_labels = preds.argmax(dim=1)
    n_classes = preds.shape[1]

    acc = (pred_labels == targets).float().mean().item()
    per_class_f1 = []
    for c in range(n_classes):
        tp = ((pred_labels == c) & (targets == c)).sum().float()
        fp = ((pred_labels == c) & (targets != c)).sum().float()
        fn = ((pred_labels != c) & (targets == c)).sum().float()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        per_class_f1.append((2 * prec * rec / (prec + rec + 1e-8)).item())
    f1_macro = sum(per_class_f1) / n_classes

    pe = 0.0
    for c in range(n_classes):
        pe += (pred_labels == c).float().mean().item() * (targets == c).float().mean().item()
    kappa = (acc - pe) / (1 - pe + 1e-8)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "cohen_kappa": kappa,
        "f1_per_class": {CLASS_NAMES[c]: per_class_f1[c] for c in range(n_classes)},
    }


def build_dataset(dataset_name, channels, pipeline, seq_len):
    if dataset_name == "mass":
        from physioex.data.multi import MultiDataset
        MASS = get_dataset("mass")
        cohorts = []
        for c in [1, 2, 3, 4, 5]:
            ds = MASS(cohort=c, channels=channels, pipelines=pipeline, sequence_length=seq_len)
            if ds.get_n_subjects() > 0:
                cohorts.append(ds)
        return MultiDataset(cohorts)
    else:
        DatasetClass = get_dataset(dataset_name)
        ds_kwargs = {"visit": 1} if dataset_name == "shhs" else {}
        return DatasetClass(channels=channels, pipelines=pipeline, sequence_length=seq_len, **ds_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Test ProtoSleepNet with VQ prototypes")
    parser.add_argument("--backbone", type=str, required=True, choices=["seq", "st"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--codebook_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="mass")
    parser.add_argument("--channels", nargs="+", default=["EEG", "EOG", "EMG"])
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = load_model(args.backbone, args.checkpoint, device)
    codebook = np.load(args.codebook_path)
    model.set_codebook(codebook)
    M = codebook.shape[0]
    print(f"Model loaded, codebook M={M}, d={codebook.shape[1]}")

    dataset = build_dataset(args.dataset, args.channels, "seqsleepnet", args.seq_len)
    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, fold=args.fold,
    )
    print(f"Test subjects: {len(test_loader)}")

    subject_predictions = []
    all_proba = []
    all_targets = []

    for subj_idx, batch in enumerate(tqdm(test_loader, desc=f"VQ M={M}")):
        if isinstance(batch, dict) and "signals" in batch:
            inputs = stack_channels(batch)
            targets = batch["labels"]
        else:
            inputs, targets = batch

        proba = evaluate_subject(model, inputs, args.seq_len, device)
        targets_flat = targets.reshape(-1)
        subject_predictions.append({
            "subject_idx": subj_idx,
            "proba": proba.tolist(),
            "labels": targets_flat.tolist(),
        })
        all_proba.append(proba)
        all_targets.append(targets_flat)

    metrics = compute_metrics(all_proba, all_targets)
    print(f"\nM={M}: acc={metrics['accuracy']:.4f}  f1={metrics['f1_macro']:.4f}  kappa={metrics['cohen_kappa']:.4f}")

    pred_path = os.path.join(args.output_dir, f"predictions_vq_m{M}.json")
    with open(pred_path, "w") as f:
        json.dump(subject_predictions, f)

    metrics_path = os.path.join(args.output_dir, f"metrics_vq_m{M}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
