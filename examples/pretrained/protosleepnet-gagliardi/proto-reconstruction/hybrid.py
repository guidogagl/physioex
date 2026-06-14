"""Hybrid prototype reconstruction.

Initializes from data-driven results (nearest real epochs) then
optimizes to further minimize L2 distance to each prototype. All M
prototypes are optimized in parallel on GPU, same as model_driven.

Usage:
    python hybrid.py --backbone seq --data_driven_dir /path/to/data_driven/ --n_steps 500
    python hybrid.py --backbone st  --data_driven_dir /path/to/data_driven/ --n_steps 500
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    CONFIGS, STAGE_NAMES,
    add_common_args, resolve_output_dir, get_device,
    load_frozen_model, load_codebook,
    save_prototype_results, save_summary,
)
from model_driven import optimize_all_prototypes, C, T, F


def load_data_driven_init(data_driven_dir, M, N):
    """Load data-driven epochs and build initialization tensor.

    For prototypes with fewer than N epochs, pads with Gaussian noise.

    Args:
        data_driven_dir: Path to data_driven output.
        M: Number of prototypes.
        N: Target samples per prototype.

    Returns:
        init: (M*N, 1, C, T, F) tensor
        n_real: (M,) number of real (non-padded) samples per prototype
    """
    data_dir = Path(data_driven_dir)
    all_epochs = []
    n_real = np.zeros(M, dtype=int)

    for k in range(M):
        proto_dir = data_dir / f"proto_{k:03d}"
        epochs_path = proto_dir / "epochs.npy"

        if epochs_path.exists():
            epochs = np.load(epochs_path).astype(np.float32)  # (N_k, C, T, F)
            n_k = min(len(epochs), N)
            n_real[k] = n_k
            epochs = epochs[:n_k]

            if n_k < N:
                # Pad with Gaussian noise
                pad = np.random.randn(N - n_k, C, T, F).astype(np.float32)
                epochs = np.concatenate([epochs, pad], axis=0)
        else:
            # No data-driven result — all noise
            print(f"  Warning: no data_driven epochs for proto_{k:03d}, "
                  f"using random init")
            epochs = np.random.randn(N, C, T, F).astype(np.float32)
            n_real[k] = 0

        all_epochs.append(epochs)  # (N, C, T, F) each

    # Stack: (M*N, C, T, F) → add L=1 dim → (M*N, 1, C, T, F)
    init = np.concatenate(all_epochs, axis=0)  # (M*N, C, T, F)
    init = init[:, np.newaxis, :, :, :]         # (M*N, 1, C, T, F)
    return torch.from_numpy(init), n_real


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid prototype reconstruction"
    )
    add_common_args(parser)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--chunk_size", type=int, default=None,
        help="Forward pass chunk size (default: no chunking)",
    )
    parser.add_argument(
        "--data_driven_dir", type=str, required=True,
        help="Path to data_driven output directory",
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args, "hybrid")
    if output_dir.exists() and not args.force:
        print(f"Output exists: {output_dir}  (use --force to overwrite)")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args)
    print(f"Device: {device}")

    codebook = load_codebook(args.backbone, m=args.m, codebook_path=args.codebook_path)
    M = codebook.shape[0]
    N = args.top_k
    print(f"Codebook: M={M}, d={codebook.shape[1]}")

    # Load data-driven initialization
    print(f"Loading data-driven init from {args.data_driven_dir}...")
    np.random.seed(42)
    init_tensor, n_real = load_data_driven_init(args.data_driven_dir, M, N)
    print(f"  Initialization: {init_tensor.shape}, "
          f"real epochs per proto: min={n_real.min()}, max={n_real.max()}, "
          f"mean={n_real.mean():.0f}")

    model = load_frozen_model(args.backbone, device, checkpoint_path=args.checkpoint_path)
    print(f"Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Optimizing {M}×{N} = {M*N} samples, {args.n_steps} steps")

    torch.manual_seed(42)
    x_np, embeddings, distances, loss_curve = optimize_all_prototypes(
        model=model,
        codebook=codebook,
        device=device,
        n_per_proto=N,
        n_steps=args.n_steps,
        lr=args.lr,
        chunk_size=args.chunk_size,
        init=init_tensor,
    )

    # Save per-prototype
    print("Saving results...")
    summary_per_proto = []
    for k in range(M):
        metadata = {
            "prototype_idx": k,
            "n_samples": N,
            "n_real_init": int(n_real[k]),
            "n_padded_init": int(N - n_real[k]),
            "method": "hybrid",
            "backbone": args.backbone,
            "model_name": CONFIGS[args.backbone]["model_name"],
            "codebook_size": int(M),
            "n_steps": args.n_steps,
            "lr": args.lr,
            "mean_distance": float(distances[k].mean()),
            "std_distance": float(distances[k].std()),
            "final_loss": float(loss_curve[-1]),
            "data_driven_dir": str(args.data_driven_dir),
        }
        save_prototype_results(
            output_dir, k,
            epochs=x_np[k],             # (N, C, T, F)
            embeddings=embeddings[k],    # (N, d_model)
            distances=distances[k],      # (N,)
            loss_curve=loss_curve,
            metadata=metadata,
        )
        summary_per_proto.append({
            "prototype": k,
            "n_samples": N,
            "n_real_init": int(n_real[k]),
            "mean_distance": float(distances[k].mean()),
        })
        print(f"  proto_{k:03d}: dist={distances[k].mean():.4f}±{distances[k].std():.4f} "
              f"(init: {n_real[k]} real + {N - n_real[k]} noise)")

    save_summary(output_dir, {
        "method": "hybrid",
        "backbone": args.backbone,
        "model_name": CONFIGS[args.backbone]["model_name"],
        "n_prototypes": int(M),
        "n_per_proto": N,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "final_loss": float(loss_curve[-1]),
        "data_driven_dir": str(args.data_driven_dir),
        "per_prototype": summary_per_proto,
    })

    print(f"\nDone. Output at {output_dir}/")


if __name__ == "__main__":
    main()
