"""Compare CSD concepts across different specificity strategies.

This script:
1. Loads concepts extracted by extract_all_concepts.py
2. Generates comparative visualizations:
   - Concept overlap heatmap
   - Strategy-specific vs robust concepts
   - Spectral profile comparison

Usage:
    python examples/analyze_csd_concepts/compare_strategies.py \\
        --input_dir ./csd_concepts_analysis \\
        --output_dir ./csd_comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.analyze_csd_concepts.utils import (
    load_all_strategies_for_class,
    compute_concept_overlap,
    save_overlap_report,
    get_top_concepts,
    STAGE_NAMES,
)


def plot_overlap_heatmap(
    overlap_data: dict,
    class_id: int,
    output_path: Path,
):
    """Plot strategy overlap heatmap (Jaccard index).

    Args:
        overlap_data: Result from compute_concept_overlap.
        class_id: Class index for title.
        output_path: Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    matrix = overlap_data["overlap_matrix"]
    names = overlap_data["strategy_names"]

    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title(f"Concept Overlap (Jaccard) - {STAGE_NAMES[class_id]}")
    fig.colorbar(im, ax=ax, label="Jaccard Index")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_dim_frequency_bar(
    overlap_data: dict,
    class_id: int,
    output_path: Path,
    top_n: int = 20,
):
    """Plot bar chart of dimension selection frequency.

    Args:
        overlap_data: Result from compute_concept_overlap.
        class_id: Class index for title.
        output_path: Path to save figure.
        top_n: Number of top dimensions to show.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Sort by frequency
    dim_freq = overlap_data["dim_frequency"]
    sorted_dims = sorted(dim_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

    dims = [f"d{d}" for d, _ in sorted_dims]
    freqs = [f for _, f in sorted_dims]
    n_strategies = len(overlap_data["strategy_names"])

    colors = [plt.cm.viridis(f / n_strategies) for f in freqs]
    bars = ax.barh(dims, freqs, color=colors)

    ax.set_xlabel("Selection Frequency (# strategies)")
    ax.set_ylabel("Embedding Dimension")
    ax.set_title(f"Concept Frequency Across Strategies - {STAGE_NAMES[class_id]}")
    ax.set_xlim(0, n_strategies + 0.5)

    # Add count labels
    for i, (bar, freq) in enumerate(zip(bars, freqs)):
        ax.text(freq + 0.1, i, str(freq), va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_spectral_profiles_comparison(
    strategies_data: dict,
    class_id: int,
    output_path: Path,
    top_k: int = 5,
):
    """Compare spectral profiles of top concepts across strategies.

    Args:
        strategies_data: Dict {strategy: concepts_data}.
        class_id: Class index for title.
        output_path: Path to save figure.
        top_k: Number of top concepts per strategy to compare.
    """
    n_strategies = len(strategies_data)
    if n_strategies == 0:
        return

    fig, axes = plt.subplots(1, n_strategies, figsize=(4 * n_strategies, 4),
                             sharey=True)
    if n_strategies == 1:
        axes = [axes]

    band_frequencies = None

    for ax, (strategy_name, data) in zip(axes, strategies_data.items()):
        if band_frequencies is None:
            band_frequencies = data["band_frequencies"]

        # Get top K concepts
        top_concepts = get_top_concepts(data, top_k=top_k)

        # Aggregate band energies (mean over top concepts)
        energies = []
        for dim, concept in top_concepts:
            energies.append(concept["band_energy"])

        mean_energy = np.mean(energies, axis=0)
        std_energy = np.std(energies, axis=0) if len(energies) > 1 else np.zeros_like(mean_energy)

        # Plot
        ax.plot(band_frequencies, mean_energy, marker="o", label=strategy_name)
        ax.fill_between(band_frequencies,
                        mean_energy - std_energy,
                        mean_energy + std_energy,
                        alpha=0.3)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{strategy_name.capitalize()} (top {top_k})")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Band Energy")
    fig.suptitle(f"Spectral Profile Comparison - {STAGE_NAMES[class_id]}")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_robust_vs_specific(
    overlap_data: dict,
    strategies_data: dict,
    class_id: int,
    output_path: Path,
):
    """Plot comparison of robust vs strategy-specific concepts.

    Args:
        overlap_data: Result from compute_concept_overlap.
        strategies_data: Dict {strategy: concepts_data}.
        class_id: Class index for title.
        output_path: Path to save figure.
    """
    robust_dims = set(overlap_data["robust_dims"])
    strategy_specific = overlap_data["strategy_specific"]

    # Count robust vs specific per strategy
    strategy_names = overlap_data["strategy_names"]
    n_robust = []
    n_specific = []

    for name in strategy_names:
        specific_dims = set(strategy_specific.get(name, []))
        n_robust.append(len(robust_dims))
        n_specific.append(len(specific_dims))

    x = np.arange(len(strategy_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width/2, n_robust, width, label="Robust (≥2 strategies)",
                   color="#2ca02c")
    bars2 = ax.bar(x + width/2, n_specific, width, label="Strategy-specific",
                   color="#ff7f0e")

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Number of Concepts")
    ax.set_title(f"Robust vs Strategy-Specific Concepts - {STAGE_NAMES[class_id]}")
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names)
    ax.legend()

    # Add count labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f"{int(height)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare CSD concepts across strategies"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory with class_X subdirs from extract_all_concepts.py")
    parser.add_argument("--output_dir", type=str, default="./csd_comparison")
    parser.add_argument("--class_id", type=int, default=None,
                       help="Single class to analyze (0-4)")
    parser.add_argument("--all_classes", action="store_true",
                       help="Analyze all 5 classes")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Top K concepts for frequency plot")
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
    print("CSD Strategy Comparison")
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

        # Compute overlap
        print(f"\n[2] Computing overlap...")
        overlap_data = compute_concept_overlap(strategies_data)

        print(f"  Robust concepts (≥2 strategies): {len(overlap_data['robust_dims'])}")
        for name, dims in overlap_data["strategy_specific"].items():
            print(f"    {name}: {len(dims)} specific concepts")

        # Save overlap report
        report_path = class_output_dir / "overlap_report.json"
        save_overlap_report(overlap_data, report_path, class_id)

        # Generate plots
        print(f"\n[3] Generating plots...")

        # Overlap heatmap
        plot_overlap_heatmap(
            overlap_data, class_id,
            class_output_dir / "overlap_heatmap.png"
        )

        # Dimension frequency
        plot_dim_frequency_bar(
            overlap_data, class_id,
            class_output_dir / "dim_frequency.png",
            top_n=args.top_k
        )

        # Spectral profiles comparison
        plot_spectral_profiles_comparison(
            strategies_data, class_id,
            class_output_dir / "spectral_profiles.png",
            top_k=5
        )

        # Robust vs specific
        plot_robust_vs_specific(
            overlap_data, strategies_data, class_id,
            class_output_dir / "robust_vs_specific.png"
        )

    # Summary
    print("\n" + "=" * 60)
    print("Comparison Complete!")
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
