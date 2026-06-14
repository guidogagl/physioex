"""Model-driven prototype reconstruction.

For each prototype, optimizes 256 Gaussian-noise inputs so their epoch
embeddings match the prototype in L2 distance. All M prototypes are
optimized in parallel in a single batch on GPU.

Usage:
    python model_driven.py --backbone seq --n_steps 1000
    python model_driven.py --backbone st  --n_steps 1000
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    CONFIGS, STAGE_NAMES,
    add_common_args, resolve_output_dir, get_device,
    load_frozen_model, load_codebook,
    save_prototype_results, save_summary,
)

# Input spectrogram shape (seqsleepnet pipeline)
C, T, F = 3, 29, 129


def optimize_all_prototypes(
    model, codebook, device,
    n_per_proto=256, n_steps=1000, lr=0.01,
    chunk_size=None, init=None,
):
    """Optimize inputs for all prototypes simultaneously.

    Args:
        model: Frozen ProtoSleepNet in eval mode.
        codebook: (M, d_model) numpy array.
        device: torch device.
        n_per_proto: Number of samples per prototype.
        n_steps: Optimization steps.
        lr: Learning rate.
        chunk_size: If set, chunk the forward pass for memory.
        init: Optional (M*N, 1, C, T, F) initial tensor. If None, use randn.

    Returns:
        x: (M, N, C, T, F) optimized inputs (numpy).
        embeddings: (M, N, d_model) final embeddings (numpy).
        distances: (M, N) final L2 distances (numpy).
        loss_curve: list of floats, one per step.
    """
    M = codebook.shape[0]
    N = n_per_proto
    total = M * N
    codebook_t = torch.from_numpy(codebook).float().to(device)

    # Initialize
    if init is not None:
        x = init.clone().detach().to(device).requires_grad_(True)
    else:
        x = torch.randn(total, 1, C, T, F, device=device).requires_grad_(True)

    # Target for each sample: sample i belongs to prototype i // N
    targets = codebook_t.repeat_interleave(N, dim=0)  # (M*N, d_model)

    optimizer = torch.optim.Adam([x], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps
    )

    if chunk_size is None or chunk_size >= total:
        chunk_size = total
    n_chunks = (total + chunk_size - 1) // chunk_size

    loss_curve = []
    progress = tqdm(range(n_steps), desc="Optimizing")

    for step in progress:
        optimizer.zero_grad()
        total_loss = 0.0

        for i in range(0, total, chunk_size):
            j = min(i + chunk_size, total)
            h = model.epoch_encode(x[i:j], quantize=False)  # (chunk, 1, d)
            h = h.squeeze(1)  # (chunk, d)
            dist_sq = ((h - targets[i:j]) ** 2).sum(dim=1)  # (chunk,)
            chunk_loss = dist_sq.mean()
            (chunk_loss / n_chunks).backward()
            total_loss += chunk_loss.item()

        total_loss /= n_chunks
        torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_curve.append(total_loss)

        if step % 100 == 0 or step == n_steps - 1:
            progress.set_postfix(loss=f"{total_loss:.6f}")

    # Compute final embeddings and distances
    with torch.no_grad():
        all_h = []
        for i in range(0, total, chunk_size):
            j = min(i + chunk_size, total)
            h = model.epoch_encode(x[i:j], quantize=False).squeeze(1)
            all_h.append(h.cpu())
        all_h = torch.cat(all_h, dim=0).numpy()  # (M*N, d_model)

    x_np = x.detach().cpu().squeeze(1).numpy()  # (M*N, C, T, F)

    # Reshape to per-prototype
    x_np = x_np.reshape(M, N, C, T, F)
    all_h = all_h.reshape(M, N, -1)

    # Per-prototype L2 distances
    distances = np.sqrt(
        np.maximum(0, ((all_h - codebook[:, None, :]) ** 2).sum(axis=2))
    )  # (M, N)

    return x_np, all_h, distances, loss_curve


def main():
    parser = argparse.ArgumentParser(
        description="Model-driven prototype reconstruction"
    )
    add_common_args(parser)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--chunk_size", type=int, default=None,
        help="Forward pass chunk size (default: no chunking)",
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args, "model_driven")
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
    print(f"Optimizing {M}×{N} = {M*N} samples, {args.n_steps} steps")

    model = load_frozen_model(args.backbone, device, checkpoint_path=args.checkpoint_path)
    print(f"Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

    torch.manual_seed(42)
    x_np, embeddings, distances, loss_curve = optimize_all_prototypes(
        model=model,
        codebook=codebook,
        device=device,
        n_per_proto=N,
        n_steps=args.n_steps,
        lr=args.lr,
        chunk_size=args.chunk_size,
    )

    # Save per-prototype
    print("Saving results...")
    summary_per_proto = []
    for k in range(M):
        metadata = {
            "prototype_idx": k,
            "n_samples": N,
            "method": "model_driven",
            "backbone": args.backbone,
            "model_name": CONFIGS[args.backbone]["model_name"],
            "codebook_size": int(M),
            "n_steps": args.n_steps,
            "lr": args.lr,
            "mean_distance": float(distances[k].mean()),
            "std_distance": float(distances[k].std()),
            "final_loss": float(loss_curve[-1]),
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
            "mean_distance": float(distances[k].mean()),
        })
        print(f"  proto_{k:03d}: dist={distances[k].mean():.4f}±{distances[k].std():.4f}")

    save_summary(output_dir, {
        "method": "model_driven",
        "backbone": args.backbone,
        "model_name": CONFIGS[args.backbone]["model_name"],
        "n_prototypes": int(M),
        "n_per_proto": N,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "final_loss": float(loss_curve[-1]),
        "per_prototype": summary_per_proto,
    })

    print(f"\nDone. Output at {output_dir}/")


if __name__ == "__main__":
    main()
