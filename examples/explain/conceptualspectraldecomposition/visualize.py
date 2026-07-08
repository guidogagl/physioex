"""Matplotlib visualizations for CSD results.

Provides functions to plot:
- Per-concept attribution heatmaps
- Aggregated class-level attribution
- Top concepts by weight/contribution
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from physioex.explain.foundational import CSDResult, ConceptAttribution

# Sleep stage names
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {
    "W": "#1f77b4",  # blue
    "N1": "#ff7f0e",  # orange
    "N2": "#2ca02c",  # green
    "N3": "#9467bd",  # purple
    "REM": "#d62728",  # red
}


def plot_concept_attribution(
    concept: ConceptAttribution,
    band_frequencies: torch.Tensor,
    stage_name: str,
    ax=None,
    figsize=(10, 4),
):
    """Plot attribution heatmap for a single concept.

    Args:
        concept: ConceptAttribution for one dimension.
        band_frequencies: (n_bands,) center frequencies in Hz.
        stage_name: Sleep stage name for title.
        ax: Optional matplotlib axis.
        figsize: Figure size if creating new figure.

    Returns:
        matplotlib Figure and Axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Aggregate over channels: (B, n_bands, T) -> mean over B, sum over C
    attr = concept.attribution  # (B, n_bands, C, T)
    attr_agg = attr.mean(dim=0).sum(dim=1).cpu().numpy()  # (n_bands, T)

    # Plot heatmap
    im = ax.imshow(
        attr_agg,
        aspect="auto",
        cmap="RdBu_r",
        origin="lower",
        interpolation="bilinear",
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attribution", rotation=270, labelpad=15)

    # Labels
    n_bands, T = attr_agg.shape
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Freq (Hz)")
    ax.set_title(f"Dim {concept.dim}: W={concept.weight:+.3f}, mask={concept.mask_value:.2f}")

    # Y-axis: frequency bands (use center frequencies)
    freq_ticks = np.linspace(0, n_bands - 1, min(5, n_bands)).astype(int)
    freq_labels = [f"{band_frequencies[i].item():.0f}" for i in freq_ticks]
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels(freq_labels)

    return fig, ax


def plot_class_attribution(
    class_attr: torch.Tensor,
    band_frequencies: torch.Tensor,
    stage_name: str,
    ax=None,
    figsize=(10, 4),
):
    """Plot aggregated class-level attribution heatmap.

    Args:
        class_attr: (B, n_bands, T) or (B, n_bands, C, T) class attribution.
        band_frequencies: (n_bands,) center frequencies in Hz.
        stage_name: Sleep stage name for title.
        ax: Optional matplotlib axis.
        figsize: Figure size if creating new figure.

    Returns:
        matplotlib Figure and Axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Handle both (B, n_bands, T) and (B, n_bands, C, T) shapes
    attr = class_attr.mean(dim=0).cpu().numpy()  # (n_bands, T) or (n_bands, C, T)
    if attr.ndim == 3:
        attr = attr.sum(axis=1)  # Sum over channels: (n_bands, C, T) -> (n_bands, T)

    # Plot heatmap
    im = ax.imshow(
        attr,
        aspect="auto",
        cmap="RdBu_r",
        origin="lower",
        interpolation="bilinear",
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attribution", rotation=270, labelpad=15)

    # Labels
    n_bands, T = attr.shape
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Freq (Hz)")
    ax.set_title(f"Class {stage_name} Attribution")

    # Y-axis: frequency bands
    freq_ticks = np.linspace(0, n_bands - 1, min(5, n_bands)).astype(int)
    freq_labels = [f"{band_frequencies[i].item():.0f}" for i in freq_ticks]
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels(freq_labels)

    return fig, ax


def plot_top_concepts(
    result: CSDResult,
    top_k: int = 10,
    ax=None,
    figsize=(8, 5),
):
    """Plot bar chart of top concepts by weighted importance.

    Args:
        result: CSDResult from explanation.
        top_k: Number of top concepts to show.
        ax: Optional matplotlib axis.
        figsize: Figure size if creating new figure.

    Returns:
        matplotlib Figure and Axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Sort concepts by |mask * weight|
    concepts_data = []
    for dim, concept in result.concepts.items():
        importance = abs(concept.mask_value * concept.weight)
        concepts_data.append((dim, concept.weight, concept.mask_value, importance))

    concepts_data.sort(key=lambda x: x[3], reverse=True)
    concepts_data = concepts_data[:top_k]

    # Plot
    dims = [f"d{d}" for d, _, _, _ in concepts_data]
    weights = [w for _, w, _, _ in concepts_data]

    colors = [STAGE_COLORS.get(STAGE_NAMES[result.target_class], "gray")] * len(dims)
    bars = ax.barh(dims, weights, color=colors)

    # Add mask value as text
    for i, (_, _, mask, _) in enumerate(concepts_data):
        ax.text(
            weights[i],
            i,
            f" m={mask:.2f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel(f"Probe Weight W[{STAGE_NAMES[result.target_class]}, d]")
    ax.set_title(f"Top {top_k} Concepts for {STAGE_NAMES[result.target_class]}")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5)

    return fig, ax


def plot_per_channel_energy(
    result: CSDResult,
    top_n: int = 15,
    ax=None,
    figsize=(10, 5),
):
    """Plot per-channel attribution energy for top concepts.

    Shows which EEG channels contribute most to each concept.

    Args:
        result: CSDResult from explanation.
        top_n: Number of top concepts to show.
        ax: Optional matplotlib axis.
        figsize: Figure size if creating new figure.

    Returns:
        matplotlib Figure and Axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Get top concepts
    concepts_list = sorted(
        result.concepts.items(),
        key=lambda x: abs(x[1].mask_value * x[1].weight),
        reverse=True,
    )[:top_n]

    # Stack channel energies: (top_n, C)
    channel_energies = []
    labels = []

    for dim, concept in concepts_list:
        energy = concept.channel_energy.cpu().numpy()  # (C,)
        channel_energies.append(energy)
        labels.append(f"d{dim}")

    channel_energies = np.array(channel_energies)  # (top_n, C)

    # Plot heatmap
    im = ax.imshow(
        channel_energies,
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Energy", rotation=270, labelpad=15)

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Channel Index")
    ax.set_title(f"Per-Channel Energy (Top {top_n} Concepts)")

    return fig, ax


def save_all_plots(
    result: CSDResult,
    output_dir: str | Path,
    top_k: int = 10,
):
    """Generate and save all CSD visualization plots.

    Creates:
    - class_attribution.png: Aggregated class heatmap
    - top_concepts.png: Bar chart of top concepts
    - top_concept_*.png: Individual concept heatmaps

    Args:
        result: CSDResult from explanation.
        output_dir: Directory to save plots.
        top_k: Number of top concepts to visualize individually.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_name = STAGE_NAMES[result.target_class]

    # 1. Class attribution
    fig, ax = plot_class_attribution(
        result.class_attribution,
        result.band_frequencies,
        stage_name,
    )
    fig.savefig(output_dir / "class_attribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: class_attribution.png")

    # 2. Top concepts bar chart
    fig, ax = plot_top_concepts(result, top_k=top_k)
    fig.savefig(output_dir / "top_concepts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: top_concepts.png")

    # 3. Per-channel energy
    fig, ax = plot_per_channel_energy(result, top_n=top_k)
    fig.savefig(output_dir / "per_channel_energy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: per_channel_energy.png")

    # 4. Individual concept heatmaps
    top_concepts = sorted(
        result.concepts.items(),
        key=lambda x: abs(x[1].mask_value * x[1].weight),
        reverse=True,
    )[:top_k]

    for dim, concept in top_concepts:
        fig, ax = plot_concept_attribution(
            concept,
            result.band_frequencies,
            stage_name,
        )
        fig.savefig(
            output_dir / f"top_concept_{dim:03d}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"  Saved: {len(top_concepts)} individual concept heatmaps")


def plot_summary_figure(
    result: CSDResult,
    output_path: str | Path,
    top_k: int = 6,
):
    """Create a summary figure with multiple subplots.

    Layout:
    - Top: Class attribution heatmap (wide)
    - Bottom left: Top concepts bar chart
    - Bottom right: Per-channel energy

    Args:
        result: CSDResult from explanation.
        output_path: Path to save figure.
        top_k: Number of top concepts for subplots.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    stage_name = STAGE_NAMES[result.target_class]

    # Top: Class attribution (spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    plot_class_attribution(
        result.class_attribution,
        result.band_frequencies,
        stage_name,
        ax=ax1,
    )

    # Bottom left: Top concepts
    ax2 = fig.add_subplot(gs[1, 0])
    plot_top_concepts(result, top_k=top_k, ax=ax2)

    # Bottom right: Per-channel energy
    ax3 = fig.add_subplot(gs[1, 1])
    plot_per_channel_energy(result, top_n=top_k, ax=ax3)

    plt.suptitle(
        f"CSD Explanation: {stage_name} (subject N={result.class_attribution.shape[0]} epochs)",
        fontsize=14,
    )
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved summary: {output_path}")
