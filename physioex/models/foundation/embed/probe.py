"""Linear probe training on cached foundation model embeddings.

Extracts embeddings if not already cached, then trains a lightweight
linear head (LayerNorm + Linear) using the library's Trainer.

Usage as function::

    from physioex.models.foundation.embed.probe import train_probe
    results = train_probe("cbramod", "hmc", fold=0, max_epochs=50)
    print(results["accuracy"], results["f1_score"])

Usage as CLI::

    python -m physioex.models.foundation.embed.probe \\
        --model cbramod --dataset hmc --fold 0 --max-epochs 50
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("physioex.foundation.embed")


class LinearProbe(nn.Module):
    """Linear probe head for foundation model embeddings.

    Input:  (B, L, D) — sequence of pre-extracted embeddings
    Output: (B, L, n_classes) — per-epoch logits
    """

    def __init__(self, embedding_dim: int, n_classes: int = 5):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return self.head(x.reshape(B * L, D)).reshape(B, L, -1)


def _parse_gpu_id(device: Optional[str]) -> Optional[int]:
    if device is None:
        return 0 if torch.cuda.is_available() else None
    if device == "cpu":
        return None
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    return None


def train_probe(
    model_name: str,
    dataset_name: str,
    checkpoint_path: Optional[str] = None,
    n_classes: int = 5,
    fold: int = 0,
    max_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 64,
    device: Optional[str] = None,
    max_batch_extract: int = 256,
    overwrite_embeddings: bool = False,
    overwrite_probe: bool = False,
) -> dict:
    """Extract embeddings (if needed) and train a linear probe.

    Uses the library's ``Trainer.train()`` and ``Trainer.evaluate()``
    for training, checkpointing, progress display, and evaluation.

    Embeddings and probes are cached under:
        ``~/.cache/physioex/v1/foundation_models/{model}/embeddings/{dataset}/``
        ``~/.cache/physioex/v1/foundation_models/{model}/probes/{dataset}/fold_{fold}/``

    Args:
        model_name: foundation model slug
        dataset_name: dataset slug
        checkpoint_path: model checkpoint (if not set, auto-downloaded)
        n_classes: number of sleep stage classes (default 5)
        fold: cross-validation fold (default 0)
        max_epochs: training epochs for the linear probe
        lr: learning rate
        weight_decay: weight decay
        batch_size: training batch size
        device: torch device string
        max_batch_extract: max epochs per GPU batch during extraction
        overwrite_embeddings: re-extract embeddings even if cached
        overwrite_probe: re-train probe even if cached

    Returns:
        dict with evaluation metrics (accuracy, f1_score, cohen_kappa, etc.)
    """
    from physioex.models.foundation.embed.extract import extract_embeddings
    from physioex.models.foundation.embed.dataset import (
        EmbeddingDataset,
        embedding_collate_fn,
    )
    from physioex.models.foundation._checkpoints import get_probes_dir

    # Step 0: Check if probe is already cached
    probe_dir = get_probes_dir(model_name, dataset_name, fold)
    probe_ckpt = probe_dir / "best.pt"
    metrics_path = probe_dir / "metrics.json"

    if probe_ckpt.exists() and metrics_path.exists() and not overwrite_probe:
        import json

        logger.info(f"Probe cached at {probe_dir}, loading metrics...")
        metrics = json.loads(metrics_path.read_text())
        logger.info(
            f"Cached results: acc={metrics.get('accuracy', 0):.4f}, "
            f"f1={metrics.get('f1_score', 0):.4f}"
        )
        return metrics

    # Step 1: Extract embeddings if not cached
    logger.info(f"Ensuring embeddings for {model_name}/{dataset_name}...")
    extract_embeddings(
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        max_batch=max_batch_extract,
        device=device,
        overwrite=overwrite_embeddings,
    )

    # Step 2: Create embedding dataset
    ds = EmbeddingDataset(
        model_name=model_name,
        dataset_name=dataset_name,
        sequence_length=21,
    )
    embedding_dim = ds.embedding_dim
    logger.info(f"EmbeddingDataset: {len(ds)} items, dim={embedding_dim}")

    # Step 3: Build dataloaders using Trainer.build_dataloaders pattern
    from torch.utils.data import DataLoader, Subset

    train_indices, valid_subjects, test_subjects = ds.split(fold=fold)

    train_dataset = Subset(ds, train_indices.tolist())
    valid_ids = [sid for _, sid in valid_subjects]
    test_ids = [sid for _, sid in test_subjects]
    valid_indices = ds._subject_ids_to_flat_indices(valid_ids)
    test_indices = ds._subject_ids_to_flat_indices(test_ids)
    valid_dataset = Subset(ds, valid_indices)
    test_dataset = Subset(ds, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=embedding_collate_fn,
        num_workers=4,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=embedding_collate_fn,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=embedding_collate_fn,
        num_workers=4,
        persistent_workers=True,
    )

    logger.info(
        f"Splits: train={len(train_dataset)}, val={len(valid_dataset)}, "
        f"test={len(test_dataset)}"
    )

    # Step 4: Train using the library's Trainer
    from physioex.train.trainer import Trainer

    probe = LinearProbe(embedding_dim, n_classes)

    probe = Trainer.train(
        model=probe,
        dataset=(train_loader, valid_loader),
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
        loss=nn.CrossEntropyLoss(ignore_index=-1),
        gpu_id=_parse_gpu_id(device),
        log_device=False,
    )

    # Step 5: Evaluate using the library's Trainer
    results = Trainer.evaluate(
        model=probe,
        dataset=test_loader,
        gpu_id=_parse_gpu_id(device),
    )

    logger.info(
        f"Test results: acc={results.get('accuracy', 0):.4f}, "
        f"f1={results.get('f1_score', 0):.4f}, "
        f"kappa={results.get('cohen_kappa', 0):.4f}"
    )

    # Step 6: Cache probe checkpoint and metrics
    import json

    probe_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), probe_ckpt)
    # Serialize metrics (convert numpy types to Python)
    serializable = {}
    for k, v in results.items():
        if hasattr(v, "tolist"):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    metrics_path.write_text(json.dumps(serializable, indent=2))
    logger.info(f"Probe cached at {probe_dir}")

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train a linear probe on cached foundation model embeddings."
    )
    parser.add_argument("--model", required=True, help="Foundation model slug")
    parser.add_argument("--dataset", required=True, help="Dataset slug")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--n-classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--fold", type=int, default=0, help="CV fold")
    parser.add_argument("--max-epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", default=None, help="Device")
    parser.add_argument(
        "--max-batch-extract",
        type=int,
        default=256,
        help="Max epochs per GPU batch during extraction",
    )
    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Re-extract embeddings even if cached",
    )
    parser.add_argument(
        "--overwrite-probe", action="store_true", help="Re-train probe even if cached"
    )
    args = parser.parse_args()

    results = train_probe(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        n_classes=args.n_classes,
        fold=args.fold,
        max_epochs=args.max_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        max_batch_extract=args.max_batch_extract,
        overwrite_embeddings=args.overwrite_embeddings,
        overwrite_probe=args.overwrite_probe,
    )

    print("\n=== Linear Probe Results ===")
    for k, v in results.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
