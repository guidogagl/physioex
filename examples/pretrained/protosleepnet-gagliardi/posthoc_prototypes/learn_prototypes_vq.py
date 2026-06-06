"""Learn post-hoc codebook via VQ on pre-extracted epoch embeddings.

Two stages:
  1. K-Means initialization on training embeddings (CPU)
  2. Supervised refinement with stride-1 sliding windows through the
     frozen sequence_encoder + classifier (GPU)

Reference: Rymarczyk et al., "ProtoQuant", 2025 (arXiv:2602.06592)

Usage:
    python examples/pretrained/protosleepnet-gagliardi/learn_prototypes_vq.py \
        --model_dir /path/to/pretrained/st-baseline \
        --emb_dir /path/to/embeddings/st-baseline \
        --n_prototypes 48 \
        --gpu_id 0
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch

from physioex.explain.prototypes.posthoc import (
    learn_codebook_kmeans,
    quantize_embeddings,
    train_codebook,
    load_epoch_embeddings,
    load_epoch_embeddings_per_subject,
)

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
SEQ_LEN = 21


def load_model(model_dir, device):
    """Load model from config.json + model.pt."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    module_path, class_name = config["model_class"].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    ModelClass = getattr(mod, class_name)

    model = ModelClass(**config["model_kwargs"])
    weights_path = os.path.join(model_dir, "model.pt")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, config


def build_downstream_fn(model, device):
    """Build a frozen downstream function: seq_encoder -> classifier."""
    for p in model.parameters():
        p.requires_grad_(False)

    def downstream_fn(z_seq):
        B, L, D = z_seq.shape

        if hasattr(model, "sequence_encoder") and hasattr(model, "classifier"):
            z = model.sequence_encoder(z_seq)
            z = z.reshape(B * L, -1)
            return model.classifier(z).reshape(B, L, -1)

        if hasattr(model, "seqn2") and hasattr(model, "classifier"):
            z, _ = model.seqn2(z_seq)
            z = z.reshape(B * L, -1)
            return model.classifier(z).reshape(B, L, -1)

        if hasattr(model, "seqn2") and hasattr(model, "clf"):
            z, _ = model.seqn2(z_seq)
            z = z.reshape(B * L, -1)
            return model.clf(z).reshape(B, L, -1)

        raise ValueError(f"Unknown model type: {type(model).__name__}")

    return downstream_fn


def main():
    parser = argparse.ArgumentParser(
        description="Learn VQ codebook (K-Means init + supervised refinement)"
    )
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--emb_dir", type=str, required=True)
    parser.add_argument("--n_prototypes", type=int, default=48)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of L-length windows per step")
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_iter_kmeans", type=int, default=300)
    parser.add_argument("--no_kmeans_init", action="store_true",
                        help="Skip K-Means init, use Kaiming uniform instead")
    parser.add_argument("--force", action="store_true",
                        help="Force re-training even if codebook exists")
    parser.add_argument("--model_name", type=str, default=None,
                        help="HF model name (alternative to --model_dir)")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HF repo ID")
    args = parser.parse_args()

    output_dir = args.output_dir or args.emb_dir
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"vq_learned_m{args.n_prototypes}" if args.no_kmeans_init else f"vq_m{args.n_prototypes}"
    codebook_path = os.path.join(output_dir, f"codebook_{suffix}.npy")
    meta_path = os.path.join(output_dir, f"codebook_{suffix}_meta.json")

    # Skip if already trained
    if not args.force and os.path.exists(codebook_path) and os.path.exists(meta_path):
        print(f"SKIP: {codebook_path} already exists (use --force to retrain)")
        return

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Load training embeddings (flat for K-Means)
    print(f"Loading embeddings from {args.emb_dir}")
    Z_train_flat, Y_train_flat = load_epoch_embeddings(args.emb_dir, split="train")
    valid = Y_train_flat >= 0
    print(f"  Train: {valid.sum()} valid epochs, d_model={Z_train_flat.shape[1]}")

    # Stage 1: Codebook initialization
    if args.no_kmeans_init:
        print(f"\nStage 1: Kaiming uniform init with M={args.n_prototypes}...")
        d_model = Z_train_flat.shape[1]
        import torch.nn as nn
        codebook_param = torch.empty(args.n_prototypes, d_model)
        nn.init.kaiming_uniform_(codebook_param)
        codebook = codebook_param.numpy().astype(np.float32)
        print(f"  Codebook: {codebook.shape}")
    else:
        print(f"\nStage 1: K-Means with M={args.n_prototypes} clusters...")
        codebook = learn_codebook_kmeans(
            Z_train_flat[valid], n_prototypes=args.n_prototypes,
            max_iter=args.max_iter_kmeans,
        )
        print(f"  Codebook: {codebook.shape}")

        Z_q, assignments = quantize_embeddings(Z_train_flat[valid], codebook)
        recon_error = np.sqrt(((Z_train_flat[valid] - Z_q) ** 2).sum(axis=1).mean())
        print(f"  Reconstruction error (L2): {recon_error:.4f}")
        print(f"  Active clusters: {len(np.unique(assignments))} / {args.n_prototypes}")

    # Stage 2: Supervised refinement (per-subject sliding windows)
    if args.n_epochs > 0:
        print(f"\nStage 2: Supervised refinement ({args.n_epochs} epochs, patience={args.patience})...")
        if args.model_name:
            from physioex.models import load_from_pretrained
            kwargs = {"device": str(device)}
            if args.repo_id:
                kwargs["repo_id"] = args.repo_id
            model = load_from_pretrained(args.model_name, **kwargs)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            print(f"  Loading model from HF: {args.model_name}")
        elif args.model_dir:
            print(f"  Loading model from {args.model_dir}")
            model, config = load_model(args.model_dir, device)
        else:
            raise ValueError("--model_dir or --model_name required for supervised refinement")
        downstream_fn = build_downstream_fn(model, device)

        train_subjects = load_epoch_embeddings_per_subject(args.emb_dir, split="train")

        val_subjects = None
        try:
            val_subjects = load_epoch_embeddings_per_subject(args.emb_dir, split="valid")
            print(f"  Valid: {len(val_subjects)} subjects")
        except FileNotFoundError:
            print("  Valid split not found — no early stopping")

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
            sequence_length=SEQ_LEN,
            save_path=codebook_path,
        )

        # Post-training stats
        Z_q2, assignments2 = quantize_embeddings(Z_train_flat[valid], codebook)
        recon_error2 = np.sqrt(((Z_train_flat[valid] - Z_q2) ** 2).sum(axis=1).mean())
        print(f"  Post-training recon error: {recon_error2:.4f}")
        print(f"  Active clusters: {len(np.unique(assignments2))} / {args.n_prototypes}")

    # Save final codebook + metadata
    np.save(codebook_path, codebook)

    meta = {
        "method": "vq_supervised" if args.n_epochs > 0 else "vq_kmeans",
        "n_prototypes": args.n_prototypes,
        "d_model": int(codebook.shape[1]),
        "n_refinement_epochs": args.n_epochs,
        "lr": args.lr,
        "commitment_weight": args.commitment_weight,
        "batch_size": args.batch_size,
        "sequence_length": SEQ_LEN,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {codebook_path}")


if __name__ == "__main__":
    main()
