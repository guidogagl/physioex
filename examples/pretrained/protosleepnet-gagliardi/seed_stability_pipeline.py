"""Seed stability pipeline: train → extract embeddings → K-Means → evaluate.

Runs the full pipeline for a single seed. Designed to be launched in parallel
(6 seeds on 2 GPUs) from a sbatch wrapper.

Usage:
    python seed_stability_pipeline.py \
        --backbone seq --dataset mass --seed 123 \
        --output_dir /results/seed_123 --gpu_id 0 --n_prototypes 12
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from build_protosleepnet import build_model
from train import build_dataset, BACKBONE_CONFIGS, MIXER_KWARGS

from physioex.data.collate import stack_channels
from physioex.data.datasets import get_dataset
from physioex.models.protosleepnet import ProtoSleepNet, ProtoSleepNetTrainer
from physioex.train.trainer import Trainer
from physioex.explain.prototypes.posthoc import learn_codebook_kmeans


# Reuse evaluation functions from test_prototypes.py
from test_prototypes import (
    evaluate_subject,
    compute_metrics,
)


def extract_embeddings(model, loader, split_dir, device, batch_size=256):
    """Extract epoch embeddings for all subjects in a split."""
    os.makedirs(split_dir, exist_ok=True)
    n_extracted = 0

    for batch in loader:
        subject_id = batch["subject"][0]["id"]
        emb_path = os.path.join(split_dir, f"{subject_id}_embeddings.npy")
        lbl_path = os.path.join(split_dir, f"{subject_id}_labels.npy")

        if os.path.exists(emb_path) and os.path.exists(lbl_path):
            continue

        inputs = stack_channels(batch)
        x = inputs.squeeze(0).to(device)
        y = batch["labels"].squeeze(0).numpy()

        N = x.shape[0]
        embs = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                chunk = x[i:i + batch_size].unsqueeze(0)
                h = model.epoch_encode(chunk)
                embs.append(h.squeeze(0).cpu().numpy())

        embs = np.concatenate(embs, axis=0).astype(np.float32)
        np.save(emb_path, embs)
        np.save(lbl_path, y.astype(np.int64))
        n_extracted += 1

    return n_extracted


def load_training_embeddings(emb_dir):
    """Load all training-split embeddings."""
    train_dir = os.path.join(emb_dir, "train")
    files = sorted(glob.glob(os.path.join(train_dir, "*_embeddings.npy")))
    if not files:
        raise FileNotFoundError(f"No embedding files in {train_dir}")
    arrays = [np.load(f) for f in files]
    Z = np.concatenate(arrays, axis=0).astype(np.float32)
    print(f"  Training embeddings: {Z.shape} ({Z.nbytes / 1e6:.1f} MB)")
    return Z


def evaluate_with_codebook(model, test_loader, codebook, seq_len, device):
    """Evaluate staging with a VQ codebook."""
    model.set_codebook(codebook)
    all_proba = []
    all_targets = []

    for batch in test_loader:
        inputs = stack_channels(batch)
        targets = batch["labels"]
        proba = evaluate_subject(model, inputs, seq_len, device)
        all_proba.append(proba)
        all_targets.append(targets.reshape(-1))

    return compute_metrics(all_proba, all_targets)


def main():
    parser = argparse.ArgumentParser(description="Seed stability: train + embed + VQ + eval")
    parser.add_argument("--backbone", type=str, required=True, choices=["seq", "st"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_prototypes", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = BACKBONE_CONFIGS[args.backbone]
    dataset_name = args.dataset or cfg["default_dataset"]
    seq_len = cfg["seq_len"]
    max_epochs = cfg["max_epochs"]
    channels = cfg["channels_3ch"]
    valid_every = 100 if args.backbone == "seq" else 1000

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"=== Seed stability pipeline: seed={args.seed}, backbone={args.backbone}, "
          f"dataset={dataset_name}, M={args.n_prototypes} ===")

    # ── Step 1: Train ───────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "model.pt")
    if os.path.exists(model_path):
        print(f"SKIP training: {model_path} exists")
        model = build_model(backbone=args.backbone)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
        model = model.to(device).eval()
    else:
        print(f"\n[1/4] Training (seed={args.seed})...")
        t0 = time.time()

        ds_extra = cfg["dataset_kwargs"].get(dataset_name, {})
        train_dataset = build_dataset(dataset_name, channels, "seqsleepnet", seq_len, ds_extra)

        factory = getattr(ProtoSleepNet, cfg["factory"])
        mixer_kwargs = {**MIXER_KWARGS, "cdropout": 0.5}
        model = factory(n_channels=3, **mixer_kwargs)

        n_train = len(train_dataset.split(fold=0)[0])
        steps_per_epoch = max(1, n_train // args.batch_size)
        valid_interval_ratio = valid_every / steps_per_epoch

        model = ProtoSleepNetTrainer.train(
            model=model,
            dataset=train_dataset,
            max_epochs=max_epochs,
            lr=1e-4,
            weight_decay=0,
            train_batch_size=args.batch_size,
            fold=0,
            gpu_id=args.gpu_id,
            checkpoint_path=os.path.join(args.output_dir, "checkpoints"),
            early_stopping_patience=10,
            valid_interval_ratio=valid_interval_ratio,
            num_workers=args.num_workers,
            pin_memory=args.num_workers > 0,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2,
            seed=args.seed,
        )

        torch.save(model.cpu().state_dict(), model_path)
        elapsed = time.time() - t0
        print(f"  Training done in {elapsed / 60:.1f} min → {model_path}")
        model = model.to(device).eval()

    # ── Step 2: Extract embeddings ──────────────────────────────
    emb_dir = os.path.join(args.output_dir, "embeddings")
    emb_done_marker = os.path.join(emb_dir, ".done")
    if os.path.exists(emb_done_marker):
        print(f"SKIP embeddings: {emb_done_marker} exists")
    else:
        print(f"\n[2/4] Extracting embeddings...")
        t0 = time.time()

        ds_extra = cfg["dataset_kwargs"].get(dataset_name, {})
        emb_dataset_name = dataset_name
        if emb_dataset_name == "mass":
            from physioex.data.multi import MultiDataset
            MASS = get_dataset("mass")
            cohorts = []
            for c in [1, 2, 3, 4, 5]:
                ds = MASS(cohort=c, channels=channels, pipelines="seqsleepnet", sequence_length=0)
                if ds.get_n_subjects() > 0:
                    cohorts.append(ds)
            emb_dataset = MultiDataset(cohorts)
        else:
            DatasetClass = get_dataset(emb_dataset_name)
            ds_kwargs = {"visit": 1} if emb_dataset_name == "shhs" else {}
            emb_dataset = DatasetClass(
                channels=channels, pipelines="seqsleepnet", sequence_length=0, **ds_kwargs
            )

        train_loader, valid_loader, test_loader = Trainer.build_dataloaders(
            dataset=emb_dataset, train_batch_size=1, eval_batch_size=1,
            num_workers=0, fold=0,
        )

        for split_name, loader in [("train", train_loader), ("valid", valid_loader), ("test", test_loader)]:
            split_dir = os.path.join(emb_dir, split_name)
            n = extract_embeddings(model, loader, split_dir, device)
            print(f"  {split_name}: {n} subjects extracted")

        open(emb_done_marker, "w").close()
        elapsed = time.time() - t0
        print(f"  Embeddings done in {elapsed / 60:.1f} min")

    # ── Step 3: K-Means codebook ────────────────────────────────
    codebook_path = os.path.join(args.output_dir, f"codebook_m{args.n_prototypes}.npy")
    if os.path.exists(codebook_path):
        print(f"SKIP K-Means: {codebook_path} exists")
        codebook = np.load(codebook_path)
    else:
        print(f"\n[3/4] K-Means codebook (M={args.n_prototypes})...")
        t0 = time.time()
        Z_train = load_training_embeddings(emb_dir)
        codebook = learn_codebook_kmeans(Z_train, n_prototypes=args.n_prototypes)
        np.save(codebook_path, codebook)
        elapsed = time.time() - t0
        print(f"  K-Means done in {elapsed:.1f}s → {codebook_path}")

    # ── Step 4: Evaluate staging with VQ ────────────────────────
    metrics_path = os.path.join(args.output_dir, "staging_metrics.json")
    if os.path.exists(metrics_path):
        print(f"SKIP eval: {metrics_path} exists")
    else:
        print(f"\n[4/4] Evaluating staging with VQ (M={args.n_prototypes})...")
        t0 = time.time()

        # Build test dataloader with sequence_length for sliding-window eval
        ds_extra = cfg["dataset_kwargs"].get(dataset_name, {})
        eval_dataset = build_dataset(dataset_name, channels, "seqsleepnet", seq_len, ds_extra)
        _, _, test_loader = Trainer.build_dataloaders(
            dataset=eval_dataset, train_batch_size=1, eval_batch_size=1,
            num_workers=0, fold=0,
        )

        metrics = evaluate_with_codebook(model, test_loader, codebook, seq_len, device)
        elapsed = time.time() - t0

        metrics["seed"] = args.seed
        metrics["n_prototypes"] = args.n_prototypes
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  kappa={metrics['cohen_kappa']:.4f}  acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1_macro']:.4f}  ({elapsed / 60:.1f} min)")

    print(f"\n{'='*60}")
    print(f"Seed {args.seed} pipeline complete → {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
