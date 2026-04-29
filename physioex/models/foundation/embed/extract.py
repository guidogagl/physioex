"""Extract per-subject embeddings from a foundation model on a physioex dataset.

Saves embeddings as bfloat16/float32 numpy memmaps via ChannelCache, identical
to how BasePhysioDataset caches preprocessed signals. Cached subjects are
skipped on re-run.

Usage as function::

    from physioex.models.foundation.embed.extract import extract_embeddings
    cache_dir = extract_embeddings("cbramod", "hmc")

Usage as CLI::

    python -m physioex.models.foundation.embed.extract \\
        --model cbramod --dataset hmc --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from physioex.data.cache import ChannelCache, recommended_dtype, cast_to_cache_dtype
from physioex.models.foundation._checkpoints import get_embeddings_dir

logger = logging.getLogger("physioex.foundation.embed")


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@torch.no_grad()
def _extract_subject(
    model_wrapper,
    eeg: torch.Tensor,
    device: torch.device,
    max_batch: int = 256,
) -> np.ndarray:
    """Run frozen encoder on a full-night recording in batches.

    Args:
        model_wrapper: FoundationModelWrapper instance (uses ._preprocess + ._encode)
        eeg: (n_epochs, C, T) signal tensor
        device: torch device
        max_batch: max epochs per GPU forward pass

    Returns:
        (n_epochs, D) numpy array of embeddings
    """
    model_wrapper.encoder.to(device)
    model_wrapper.encoder.eval()
    n = eeg.shape[0]
    parts = []
    for start in range(0, n, max_batch):
        end = min(start + max_batch, n)
        chunk = eeg[start:end].to(device, dtype=torch.float32)
        chunk = model_wrapper._preprocess(chunk)
        emb = model_wrapper._encode(chunk)
        parts.append(emb.cpu().float().numpy())
    return np.concatenate(parts, axis=0)


def extract_embeddings(
    model_name: str,
    dataset_name: str,
    checkpoint_path: Optional[str] = None,
    max_batch: int = 256,
    device: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Extract per-subject embeddings and save as memmapped numpy arrays.

    Embeddings are stored in bfloat16 (if CUDA supports it) via ChannelCache,
    with sidecar metadata JSON. Subjects already cached are skipped unless
    ``overwrite=True``.

    Cache location: ``~/.cache/physioex/v1/foundation_models/{model}/embeddings/{dataset}/``

    Args:
        model_name: foundation model slug (biot, bendr, cbramod, labram, sleepfm, tfc)
        dataset_name: dataset slug from the registry (hmc, sleepedf, mass, ...)
        checkpoint_path: path to model checkpoint (if not set, auto-downloaded)
        max_batch: max epochs per GPU forward pass
        device: "cuda:0", "cpu", or None for auto
        overwrite: if True, re-extract even if cached embeddings exist

    Returns:
        Path to the embeddings directory.
    """
    from physioex.models.foundation import get_foundation_model
    from physioex.models.foundation._datasets import get_dataset_config
    from physioex.models.foundation._checkpoints import _cache_root

    dev = torch.device(device) if device else _auto_device()
    logger.info(f"Device: {dev}")

    model_cls = get_foundation_model(model_name)
    config = get_dataset_config(dataset_name)

    # Create dataset in RECORDING MODE (sequence_length=0 → one item per subject)
    dataset = model_cls.get_dataset(dataset_name, sequence_length=0)
    n_subjects = len(dataset)
    logger.info(f"Dataset {dataset_name!r}: {n_subjects} subjects")

    # Create model wrapper (for _preprocess + _encode)
    model_kwargs = dict(
        n_classes=5,
        in_chan=len(config.channels),
        sequence_length=1,
    )
    if checkpoint_path is not None:
        model_kwargs["checkpoint_path"] = checkpoint_path
    # channel_names + channel_map for models that need them
    model_kwargs["channel_names"] = list(dataset.channels)
    model_kwargs["channel_map"] = dict(config.channel_map) if config.channel_map else {}

    model = model_cls(**model_kwargs)
    model.to(dev)
    model.eval()

    embedding_dim = model.embedding_dim
    logger.info(f"Model {model_name!r}: embedding_dim={embedding_dim}")

    # Output directory
    out_dir = get_embeddings_dir(model_name, dataset_name)
    cache = ChannelCache(str(_cache_root()))
    dtype_name = recommended_dtype()

    # Global metadata
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        meta = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "embedding_dim": embedding_dim,
            "n_subjects": n_subjects,
            "pipeline_preset": model_cls.PIPELINE_PRESET,
            "channels": list(config.channels),
            "cache_dtype": dtype_name,
        }
        cache.save_json(meta_path, meta)

    for subj_idx in range(n_subjects):
        subject_id = dataset._subjects[subj_idx].subject_id
        subj_dir = out_dir / subject_id
        emb_path = subj_dir / "embeddings.npy"
        lab_path = subj_dir / "labels.npy"

        if emb_path.exists() and lab_path.exists() and not overwrite:
            logger.info(f"[{subj_idx+1}/{n_subjects}] {subject_id}: cached, skipping")
            continue

        t0 = time.time()
        logger.info(f"[{subj_idx+1}/{n_subjects}] {subject_id}: loading recording...")
        item = dataset[subj_idx]

        # Stack channels: (n_epochs, C, T)
        channel_order = item["channel_order"]
        ch_tensors = [item["signals"][ch] for ch in channel_order]
        eeg = torch.stack(ch_tensors, dim=1).float()  # (n_epochs, C, T)
        labels = item["labels"].numpy().astype(np.int64)

        logger.info(f"  {subject_id}: {eeg.shape[0]} epochs, shape {tuple(eeg.shape)}")

        # Extract embeddings
        embeddings = _extract_subject(model, eeg, dev, max_batch)
        logger.info(f"  embeddings: {embeddings.shape}")

        # Save as memmap via ChannelCache (bfloat16 + sidecar)
        emb_casted = cast_to_cache_dtype(embeddings.astype(np.float32), dtype_name)
        emb_meta = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "subject_id": subject_id,
            "embedding_dim": embedding_dim,
            "n_epochs": int(embeddings.shape[0]),
        }
        cache.atomic_save_array(emb_path, emb_casted, emb_meta)

        lab_meta = {
            "dataset_name": dataset_name,
            "subject_id": subject_id,
            "n_epochs": int(labels.shape[0]),
        }
        cache.atomic_save_array(lab_path, labels, lab_meta)

        logger.info(f"  saved ({time.time()-t0:.1f}s)")

    logger.info(f"Done. Embeddings in {out_dir}")
    return out_dir


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Extract foundation model embeddings for a physioex dataset."
    )
    parser.add_argument("--model", required=True, help="Foundation model slug")
    parser.add_argument("--dataset", required=True, help="Dataset slug")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument(
        "--max-batch", type=int, default=256, help="Max epochs per GPU batch"
    )
    parser.add_argument("--device", default=None, help="Device (cuda:0, cpu, auto)")
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-extract even if cached"
    )
    args = parser.parse_args()

    extract_embeddings(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        max_batch=args.max_batch,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
