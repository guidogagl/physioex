"""Learn VQ codebook for ProtoSleepNet (K-Means init + supervised refinement).

The downstream function uses ProtoSleepNet's residual: z_q + seq(z_q) → clf.

Usage:
    # K-Means init + supervised refinement
    python learn_prototypes_vq.py --backbone seq --checkpoint /path/to/model.pt \
        --emb_dir /path/to/embeddings --n_prototypes 48 --gpu_id 0

    # Random init + supervised (learned)
    python learn_prototypes_vq.py --backbone seq --checkpoint /path/to/model.pt \
        --emb_dir /path/to/embeddings --n_prototypes 48 --gpu_id 0 --no_kmeans_init
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from build_protosleepnet import build_model

from physioex.explain.prototypes.posthoc import (
    learn_codebook_kmeans,
    quantize_embeddings,
    train_codebook,
    load_epoch_embeddings,
    load_epoch_embeddings_per_subject,
)


def load_model(backbone, checkpoint_path, device):
    model = build_model(backbone=backbone)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()


def build_downstream_fn(model, device):
    """Build frozen downstream: z_q + seq(z_q) → classifier (residual)."""
    for p in model.parameters():
        p.requires_grad_(False)

    def downstream_fn(z_seq):
        B, L, D = z_seq.shape
        if isinstance(model.sequence_encoder, nn.GRU):
            seq_out, _ = model.sequence_encoder(z_seq)
        else:
            seq_out = model.sequence_encoder(z_seq)
        z = z_seq + seq_out  # residual
        z = z.reshape(B * L, -1)
        return model.classifier(z).reshape(B, L, -1)

    return downstream_fn


def main():
    parser = argparse.ArgumentParser(description="Learn VQ codebook for ProtoSleepNet")
    parser.add_argument("--backbone", type=str, required=True, choices=["seq", "st"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--emb_dir", type=str, required=True)
    parser.add_argument("--n_prototypes", type=int, default=48)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_iter_kmeans", type=int, default=300)
    parser.add_argument("--no_kmeans_init", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seq_len", type=int, default=20)
    args = parser.parse_args()

    output_dir = args.output_dir or args.emb_dir
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"vq_learned_m{args.n_prototypes}" if args.no_kmeans_init else f"vq_m{args.n_prototypes}"
    codebook_path = os.path.join(output_dir, f"codebook_{suffix}.npy")
    meta_path = os.path.join(output_dir, f"codebook_{suffix}_meta.json")

    if not args.force and os.path.exists(codebook_path) and os.path.exists(meta_path):
        print(f"SKIP: {codebook_path} already exists (use --force)")
        return

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Load embeddings
    Z_train, Y_train = load_epoch_embeddings(args.emb_dir, split="train")
    valid = Y_train >= 0
    print(f"Train: {valid.sum()} valid epochs, d_model={Z_train.shape[1]}")

    # Stage 1: Init
    if args.no_kmeans_init:
        print(f"Stage 1: Kaiming init M={args.n_prototypes}")
        d_model = Z_train.shape[1]
        cb = torch.empty(args.n_prototypes, d_model)
        nn.init.kaiming_uniform_(cb)
        codebook = cb.numpy().astype(np.float32)
    else:
        print(f"Stage 1: K-Means M={args.n_prototypes}")
        codebook = learn_codebook_kmeans(Z_train[valid], n_prototypes=args.n_prototypes, max_iter=args.max_iter_kmeans)
        Z_q, assignments = quantize_embeddings(Z_train[valid], codebook)
        recon = np.sqrt(((Z_train[valid] - Z_q) ** 2).sum(axis=1).mean())
        print(f"  Recon error: {recon:.4f}, Active: {len(np.unique(assignments))}/{args.n_prototypes}")

    # Stage 2: Supervised refinement
    if args.n_epochs > 0:
        print(f"Stage 2: Supervised refinement ({args.n_epochs} epochs)")
        model = load_model(args.backbone, args.checkpoint, device)
        downstream_fn = build_downstream_fn(model, device)

        train_subjects = load_epoch_embeddings_per_subject(args.emb_dir, split="train")
        val_subjects = None
        try:
            val_subjects = load_epoch_embeddings_per_subject(args.emb_dir, split="valid")
            print(f"  Valid: {len(val_subjects)} subjects")
        except FileNotFoundError:
            pass

        codebook = train_codebook(
            train_subjects=train_subjects,
            downstream_fn=downstream_fn,
            codebook_init=codebook,
            val_subjects=val_subjects,
            n_epochs=args.n_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            commitment_weight=args.commitment_weight,
            device=str(device),
            sequence_length=args.seq_len,
            save_path=codebook_path,
        )

    np.save(codebook_path, codebook)
    meta = {
        "method": "vq_supervised" if args.n_epochs > 0 else "vq_kmeans",
        "init": "kaiming" if args.no_kmeans_init else "kmeans",
        "n_prototypes": args.n_prototypes,
        "d_model": int(codebook.shape[1]),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {codebook_path}")


if __name__ == "__main__":
    main()
