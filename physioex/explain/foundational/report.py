"""Aggregation and reporting utilities for CSD results.

Provides functions to compute class-level spectral profiles and
concept atlases by running CSD over multiple samples.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger("physioex.explain.csd")


def spectral_class_profile(
    csd,
    dataloader: DataLoader,
    target_class: int,
    n_samples: int = 100,
    embeddings: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
) -> dict:
    """Compute the average CSD spectral profile for a class.

    Iterates over the dataloader, selects epochs of the target class,
    runs CSD on each, and averages the time-frequency maps.

    Args:
        csd: ConceptualSpectralDecomposition instance.
        dataloader: yields batches with "signals"/"embeddings" + "labels" keys,
            or (input, label) tuples.
        target_class: class index to profile.
        n_samples: max number of epochs to use.
        embeddings: (N, D) for static specificity strategies.
        labels: (N,) for static specificity strategies.

    Returns:
        dict with:
            band_profile: (n_bands,) mean absolute attribution per band
            temporal_profile: (T,) mean absolute attribution per sample
            full_map: (n_bands, T) mean attribution map
            band_frequencies: (n_bands,) Hz
            n_samples_used: int
            mask_mean: (D,) average specificity mask
    """
    maps = []
    masks = []
    n_used = 0

    for batch in dataloader:
        if n_used >= n_samples:
            break

        # Unpack batch
        if isinstance(batch, dict) and "embeddings" in batch:
            # Can't run CSD on pre-extracted embeddings (need raw signal)
            raise ValueError(
                "spectral_class_profile requires raw signal batches, "
                "not pre-extracted embeddings."
            )
        elif isinstance(batch, dict) and "signals" in batch:
            from physioex.data.collate import stack_channels

            x = stack_channels(batch)
            lab = batch["labels"]
        elif isinstance(batch, (tuple, list)):
            x, lab = batch[0], batch[1]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        # Select epochs of target class
        # x: (B, L, C, T), lab: (B, L) or (B,)
        if lab.ndim == 2:
            # Sequence mode: flatten
            B, L = lab.shape
            x_flat = x.reshape(B * L, x.shape[2], x.shape[3])
            lab_flat = lab.reshape(B * L)
        else:
            x_flat = x
            lab_flat = lab

        class_mask = lab_flat == target_class
        if not class_mask.any():
            continue

        x_class = x_flat[class_mask]
        remaining = n_samples - n_used
        x_class = x_class[:remaining]

        # Run CSD
        result = csd.explain(
            x_class,
            target_class=target_class,
            embeddings=embeddings,
            labels=labels,
        )

        maps.append(result.attribution.detach().cpu())
        masks.append(result.mask.detach().cpu())
        n_used += x_class.shape[0]

    if not maps:
        raise RuntimeError(f"No samples found for class {target_class}")

    all_maps = torch.cat(maps, dim=0)  # (N, n_bands, T)
    full_map = all_maps.mean(dim=0)  # (n_bands, T)
    band_profile = full_map.abs().sum(dim=-1)  # (n_bands,)
    temporal_profile = full_map.abs().sum(dim=0)  # (T,)

    all_masks = torch.cat(masks, dim=0)  # (N, D) or stack of (D,)
    mask_mean = all_masks.float().mean(dim=0) if all_masks.ndim > 1 else all_masks

    return {
        "band_profile": band_profile,
        "temporal_profile": temporal_profile,
        "full_map": full_map,
        "band_frequencies": result.band_frequencies.cpu(),
        "n_samples_used": n_used,
        "mask_mean": mask_mean,
    }


def concept_atlas(
    csd,
    dataloader: DataLoader,
    class_names: List[str] = None,
    n_samples_per_class: int = 50,
    embeddings: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
) -> Dict[str, dict]:
    """Build a concept atlas: spectral profile for every class.

    Args:
        csd: ConceptualSpectralDecomposition instance.
        dataloader: yields raw signal batches.
        class_names: list of class names (default: ["W","N1","N2","N3","REM"]).
        n_samples_per_class: samples per class.
        embeddings: for static specificity.
        labels: for static specificity.

    Returns:
        dict mapping class_name -> spectral_class_profile result.
    """
    if class_names is None:
        class_names = ["W", "N1", "N2", "N3", "REM"]

    atlas = {}
    for c, name in enumerate(class_names):
        logger.info(f"Computing profile for class {name} ({c})...")
        try:
            profile = spectral_class_profile(
                csd,
                dataloader,
                target_class=c,
                n_samples=n_samples_per_class,
                embeddings=embeddings,
                labels=labels,
            )
            atlas[name] = profile
        except RuntimeError as e:
            logger.warning(f"Skipping class {name}: {e}")
            atlas[name] = None

    return atlas
