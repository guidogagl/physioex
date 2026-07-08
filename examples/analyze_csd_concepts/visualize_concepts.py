"""Visualize CSD concepts with per-concept maps and spectral fingerprints.

This script:
1. Loads concepts extracted by extract_all_concepts.py
2. Generates per-concept visualizations:
   - Concept atlas (grid of time-frequency heatmaps)
   - Spectral fingerprints (band energy profiles)
   - Top concepts summary

Usage:
    python examples/analyze_csd_concepts/visualize_concepts.py \\
        --input_dir ./csd_concepts_analysis \\
        --class_id 3 \\
        --output_dir ./csd_viz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.analyze_csd_concepts.utils import (
    load_all_strategies_for_class,
    get_top_concepts,
    STAGE_NAMES,
)


def plot_concept_attribution(
    attribution: np.ndarray,
    band_frequencies: np.ndarray,
    title: str,
    ax=None,
    figsize=(10, 4),
):
    """Plot attribution heatmap for a single concept.

    Args:
        attribution: (n_bands, C, T) or (B, n_bands, C, T) attribution map.
        band_frequencies: (n_bands,) center frequencies in Hz.
        title: Plot title.
        ax: Optional matplotlib axis.
        figsize: Figure size if creating new figure.

    Returns:
        matplotlib Figure and Axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Aggregate over batch and channels if needed
    if attribution.ndim == 4:
        # (B, n_bands, C, T) -> mean over B, sum over C
        attr = attribution.mean(axis=0).sum(axis=1)
    else:
        # (n_bands, C, T) -> sum over C
        attr = attribution.sum(axis=1)

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
    ax.set_title(title)

    # Y-axis: frequency bands
    freq_ticks = np.linspace(0, n_bands - 1, min(5, n_bands)).astype(int)
    freq_labels = [f"{band_frequencies[i]:.0f}" for i in freq_ticks]
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels(freq_labels)

    return fig, ax


def plot_concept_atlas(
    strategies_data: dict,
    class_id: int,
    output_path: Path,
    strategy_name: str = "margin",
    top_k: int = 12,
):
    """Create a grid atlas of top concepts for a class and strategy.

    Args:
        strategies_data: Dict {strategy: concepts_data}.
        class_id: Class index for title.
        output_path: Path to save figure.
        strategy_name: Which strategy to visualize.
        top_k: Number of top concepts to show.
    """
    if strategy_name not in strategies_data:
        print(f"  [SKIP] Strategy {strategy_name} not found")
        return

    data = strategies_data[strategy_name]
    band_frequencies = data["band_frequencies"]
    top_concepts = get_top_concepts(data, top_k=top_k)

    # Determine grid layout
    n_concepts = len(top_concepts)
    n_cols = min(4, n_concepts)
    n_rows = (n_concepts + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_concepts == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    for i, (dim, concept) in enumerate(top_concepts):
        ax = axes[i]

        # Plot attribution
        attr = concept["attribution"]
        if attr.ndim == 4:
            attr_agg = attr.mean(axis=0).sum(axis=1)  # (n_bands, T)
        else:
            attr_agg = attr.sum(axis=1)

        im = ax.imshow(
            attr_agg,
            aspect="auto",
            cmap="RdBu_r",
            origin="lower",
            interpolation="bilinear",
        )

        # Title with weight and top band
        top_band_hz = band_frequencies[concept["top_band_idx"]]
        ax.set_title(
            f"d{dim}: W={concept['weight']:+.2f}, m={concept['mask_value']:.2f}\n"
            f"top={top_band_hz:.0f}Hz",
            fontsize=9
        )

        # Y-axis labels only for left column
        if i % n_cols == 0:
            n_bands = attr_agg.shape[0]
            freq_ticks = np.linspace(0, n_bands - 1, 3).astype(int)
            freq_labels = [f"{band_frequencies[idx]:.0f}" for idx in freq_ticks]
            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(freq_labels)
        else:
            ax.set_yticks([])

        ax.set_xlabel("Epoch" if i >= (n_rows - 1) * n_cols else "")

    # Hide unused subplots
    for i in range(n_concepts, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"Concept Atlas: {STAGE_NAMES[class_id]} ({strategy_name}, top {top_k})",
        fontsize=12
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_spectral_fingerprints(
    strategies_data: dict,
    class_id: int,
    output_path: Path,
    strategy_name: str = "margin",
    top_k: int = 10,
):
    """Plot spectral fingerprints for top concepts.

    Args:
        strategies_data: Dict {strategy: concepts_data}.
        class_id: Class index for title.
        output_path: Path to save figure.
        strategy_name: Which strategy to visualize.
        top_k: Number of top concepts to show.
    """
    if strategy_name not in strategies_data:
        print(f"  [SKIP] Strategy {strategy_name} not found")
        return

    data = strategies_data[strategy_name]
    band_frequencies = data["band_frequencies"]
    top_concepts = get_top_concepts(data, top_k=top_k)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each concept's band energy
    for i, (dim, concept) in enumerate(top_concepts):
        energy = concept["band_energy"]
        label = f"d{dim} (W={concept['weight']:+.2f})"

        ax.plot(band_frequencies, energy, marker="o", markersize=3,
                label=label, alpha=0.8)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Band Energy")
    ax.set_title(f"Spectral Fingerprints: {STAGE_NAMES[class_id]} ({strategy_name})")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_top_concepts_summary(
    strategies_data: dict,
    class_id: int,
    output_path: Path,
    top_k: int = 15,
):
    """Plot bar chart summary of top concepts across all strategies.

    Args:
        strategies_data: Dict {strategy: concepts_data}.
        class_id: Class index for title.
        output_path: Path to save figure.
        top_k: Number of top concepts to show.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all concepts with their strategy
    all_concepts = []
    for strategy_name, data in strategies_data.items():
        for dim, concept in data["concepts"].items():
            importance = abs(concept["weight"] * concept["mask_value"])
            all_concepts.append({
                "dim": dim,
                "weight": concept["weight"],
                "mask_value": concept["mask_value"],
                "importance": importance,
                "strategy": strategy_name,
                "top_band_idx": concept["top_band_idx"],
                "band_frequencies": data["band_frequencies"],
            })

    # Sort by importance
    all_concepts.sort(key=lambda x: x["importance"], reverse=True)

    # Take top K
    top_concepts = all_concepts[:top_k]

    # Plot
    dims = [f"d{c['dim']}" for c in top_concepts]
    weights = [c["weight"] for c in top_concepts]
    colors = []
    for c in top_concepts:
        if c["strategy"] == "margin":
            colors.append("#1f77b4")
        elif c["strategy"] == "softmax":
            colors.append("#ff7f0e")
        elif c["strategy"] == "topk":
            colors.append("#2ca02c")
        else:
            colors.append("#9467bd")

    bars = ax.barh(dims, weights, color=colors)

    # Add strategy labels
    for i, bar in enumerate(bars):
        strategy = top_concepts[i]["strategy"]
        ax.text(weights[i], i, f" {strategy}", va="center", fontsize=8)

    ax.set_xlabel(f"Probe Weight W[{STAGE_NAMES[class_id]}, dim]")
    ax.set_title(f"Top {top_k} Concepts Across All Strategies - {STAGE_NAMES[class_id]}")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="margin"),
        Patch(facecolor="#ff7f0e", label="softmax"),
        Patch(facecolor="#2ca02c", label="topk"),
        Patch(facecolor="#9467bd", label="nofilter"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_strategy_comparison(
    strategies_data: dict,
    class_id: int,
    output_path: Path,
):
    """Plot side-by-side comparison of top concepts per strategy.

    Args:
        strategies_data: Dict {strategy: concepts_data}.
        class_id: Class index for title.
        output_path: Path to save figure.
    """
    n_strategies = len(strategies_data)
    if n_strategies == 0:
        return

    fig, axes = plt.subplots(1, n_strategies, figsize=(4 * n_strategies, 5),
                             sharey=True)
    if n_strategies == 1:
        axes = [axes]

    for ax, (strategy_name, data) in zip(axes, strategies_data.items()):
        top_concepts = get_top_concepts(data, top_k=10)

        dims = [f"d{dim}" for dim, _ in top_concepts]
        weights = [concept["weight"] for _, concept in top_concepts]

        ax.barh(dims, weights, color="#1f77b4")
        ax.set_title(f"{strategy_name.capitalize()}")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.5)

    axes[0].set_xlabel(f"Probe Weight W[{STAGE_NAMES[class_id]}, dim]")
    axes[0].invert_yaxis()
    fig.suptitle(f"Top Concepts per Strategy - {STAGE_NAMES[class_id]}")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CSD concepts"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory with class_X subdirs from extract_all_concepts.py")
    parser.add_argument("--output_dir", type=str, default="./csd_viz")
    parser.add_argument("--class_id", type=int, default=None,
                       help="Single class to visualize (0-4)")
    parser.add_argument("--all_classes", action="store_true",
                       help="Visualize all 5 classes")
    parser.add_argument("--strategy", type=str, default="margin",
                       choices=["margin", "softmax", "topk", "nofilter"],
                       help="Strategy for atlas/fingerprints")
    parser.add_argument("--top_k", type=int, default=12,
                       help="Number of top concepts to visualize")
    args = parser.parse_args()

    # Determine classes to process
    if args.all_classes:
        class_ids = list(range(5))
    elif args.class_id is not None:
        class_ids = [args.class_id]
    else:
        parser.error("Must specify either --class_id or --all_classes")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CSD Concept Visualization")
    print("=" * 60)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Classes: {[STAGE_NAMES[c] for c in class_ids]}")

    for class_id in class_ids:
        class_name = STAGE_NAMES[class_id]
        print(f"\n{'=' * 40}")
        print(f"Class {class_id} ({class_name})")
        print(f"{'=' * 40}")

        # Load all strategies for this class
        print(f"\n[1] Loading concepts...")
        strategies_data = load_all_strategies_for_class(class_id, input_dir)

        if not strategies_data:
            print(f"  WARNING: No strategies found for class {class_id}")
            continue

        print(f"  Loaded {len(strategies_data)} strategies: {list(strategies_data.keys())}")

        # Create class output directory
        class_output_dir = output_dir / f"class_{class_id}_{class_name}"
        class_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate visualizations
        print(f"\n[2] Generating visualizations...")

        # Concept atlas
        plot_concept_atlas(
            strategies_data, class_id,
            class_output_dir / f"atlas_{args.strategy}.png",
            strategy_name=args.strategy,
            top_k=args.top_k
        )

        # Spectral fingerprints
        plot_spectral_fingerprints(
            strategies_data, class_id,
            class_output_dir / f"fingerprint_{args.strategy}.png",
            strategy_name=args.strategy,
            top_k=args.top_k
        )

        # Top concepts summary (all strategies)
        plot_top_concepts_summary(
            strategies_data, class_id,
            class_output_dir / "top_concepts_summary.png",
            top_k=args.top_k
        )

        # Strategy comparison
        plot_strategy_comparison(
            strategies_data, class_id,
            class_output_dir / "strategy_comparison.png"
        )

    # Summary
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"\nOutput: {output_dir}")
    for class_id in class_ids:
        class_name = STAGE_NAMES[class_id]
        class_dir = output_dir / f"class_{class_id}_{class_name}"
        if class_dir.exists():
            png_files = list(class_dir.glob("*.png"))
            print(f"  {class_name}: {len(png_files)} plots")

    return 0


if __name__ == "__main__":
    exit(main())
