"""Learn post-hoc codebook via VQ on pre-extracted epoch embeddings.

Two stages:
  1. K-Means initialization on training embeddings (CPU)
  2. Supervised refinement: optimize codebook with classification loss
     through the frozen sequence_encoder + classifier (GPU)

Reference: Rymarczyk et al., "ProtoQuant", 2025 (arXiv:2602.06592)

Usage:
    python examples/pretrained/protosleepnet-gagliardi/learn_prototypes_vq.py \
        --model_dir /path/to/pretrained/st-baseline \
        --emb_dir /path/to/embeddings/st-baseline \
        --n_prototypes 50 \
        --n_epochs 20 \
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
    """Build a frozen downstream function: seq_encoder -> classifier.

    Takes (B, L, d_model) quantized embeddings, returns (B, L, n_classes) logits.
    """
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)

    def downstream_fn(z_seq):
        # z_seq: (B, L, d_model)
        B, L, D = z_seq.shape

        # ProtoSleepTransformer / SleepTransformer
        if hasattr(model, "sequence_encoder") and hasattr(model, "classifier"):
            z = model.sequence_encoder(z_seq)
            z = z.reshape(B * L, -1)
            return model.classifier(z).reshape(B, L, -1)

        # ProtoSeqSleepNet
        if hasattr(model, "seqn2") and hasattr(model, "classifier"):
            z, _ = model.seqn2(z_seq)
            z = z.reshape(B * L, -1)
            return model.classifier(z).reshape(B, L, -1)

        # Plain SeqSleepNet
        if hasattr(model, "seqn2") and hasattr(model, "clf"):
            z, _ = model.seqn2(z_seq)
            z = z.reshape(B * L, -1)
            return model.clf(z).reshape(B, L, -1)

        raise ValueError(f"Unknown model type: {type(model).__name__}")

    return downstream_fn


def main():
    parser = argparse.ArgumentParser(
        description="Learn VQ codebook from epoch embeddings (K-Means init + supervised refinement)"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with config.json + model.pt (for downstream)")
    parser.add_argument("--emb_dir", type=str, required=True,
                        help="Directory with train/ embeddings")
    parser.add_argument("--n_prototypes", type=int, default=50)
    parser.add_argument("--n_epochs", type=int, default=50,
                        help="Supervised refinement epochs (0 = K-Means only)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience on val loss")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_iter_kmeans", type=int, default=300)
    args = parser.parse_args()

    output_dir = args.output_dir or args.emb_dir
    os.makedirs(output_dir, exist_ok=True)

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Load training + validation embeddings
    print(f"Loading embeddings from {args.emb_dir}")
    Z_train, Y_train = load_epoch_embeddings(args.emb_dir, split="train")
    valid = Y_train >= 0
    Z_train, Y_train = Z_train[valid], Y_train[valid]
    print(f"  Train: {Z_train.shape[0]} valid epochs, d_model={Z_train.shape[1]}")

    Z_val, Y_val = None, None
    try:
        Z_val, Y_val = load_epoch_embeddings(args.emb_dir, split="valid")
        val_mask = Y_val >= 0
        Z_val, Y_val = Z_val[val_mask], Y_val[val_mask]
        print(f"  Valid: {Z_val.shape[0]} valid epochs")
    except FileNotFoundError:
        print("  Valid split not found — no early stopping")

    # Stage 1: K-Means initialization
    print(f"\nStage 1: K-Means with M={args.n_prototypes} clusters...")
    codebook = learn_codebook_kmeans(
        Z_train, n_prototypes=args.n_prototypes, max_iter=args.max_iter_kmeans,
    )
    print(f"  Codebook: {codebook.shape}")

    # Stats
    Z_q, assignments = quantize_embeddings(Z_train, codebook)
    recon_error = np.sqrt(((Z_train - Z_q) ** 2).sum(axis=1).mean())
    unique = np.unique(assignments)
    print(f"  Reconstruction error (L2): {recon_error:.4f}")
    print(f"  Active clusters: {len(unique)} / {args.n_prototypes}")

    # Stage 2: Supervised refinement
    if args.n_epochs > 0:
        print(f"\nStage 2: Supervised refinement ({args.n_epochs} epochs)...")
        print(f"  Loading model from {args.model_dir}")

        model, config = load_model(args.model_dir, device)
        downstream_fn = build_downstream_fn(model, device)

        codebook = train_codebook(
            Z_train=Z_train,
            Y_train=Y_train,
            downstream_fn=downstream_fn,
            codebook_init=codebook,
            Z_val=Z_val,
            Y_val=Y_val,
            n_epochs=args.n_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            commitment_weight=args.commitment_weight,
            device=str(device),
            sequence_length=SEQ_LEN,
        )

        # Post-training stats
        Z_q2, assignments2 = quantize_embeddings(Z_train, codebook)
        recon_error2 = np.sqrt(((Z_train - Z_q2) ** 2).sum(axis=1).mean())
        unique2 = np.unique(assignments2)
        print(f"  Post-training recon error: {recon_error2:.4f}")
        print(f"  Active clusters: {len(unique2)} / {args.n_prototypes}")

    # Save
    suffix = f"vq_m{args.n_prototypes}"
    np.save(os.path.join(output_dir, f"codebook_{suffix}.npy"), codebook)

    meta = {
        "method": "vq_supervised" if args.n_epochs > 0 else "vq_kmeans",
        "n_prototypes": args.n_prototypes,
        "d_model": int(codebook.shape[1]),
        "n_train_epochs": int(len(Z_train)),
        "n_refinement_epochs": args.n_epochs,
        "lr": args.lr,
        "commitment_weight": args.commitment_weight,
    }
    with open(os.path.join(output_dir, f"codebook_{suffix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {output_dir}/codebook_{suffix}.npy")


if __name__ == "__main__":
    main()
