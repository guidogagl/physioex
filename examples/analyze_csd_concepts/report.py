"""Generate comprehensive CSD concept analysis report.

This script:
1. Loads concepts extracted by extract_all_concepts.py
2. Generates JSON and markdown reports with:
   - Per-class, per-strategy concept summaries
   - Robust vs strategy-specific concept analysis
   - Spectral characteristic summaries

Usage:
    python examples/analyze_csd_concepts/report.py \\
        --input_dir ./csd_concepts_analysis \\
        --output_dir ./csd_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.analyze_csd_concepts.utils import (
    load_all_strategies_for_class,
    compute_concept_overlap,
    get_top_concepts,
    STAGE_NAMES,
)


def generate_class_report(
    class_id: int,
    strategies_data: dict,
    overlap_data: dict,
) -> dict:
    """Generate report data for a single class.

    Args:
        class_id: Class index (0-4).
        strategies_data: Dict {strategy: concepts_data}.
        overlap_data: Result from compute_concept_overlap.

    Returns:
        Dict with class report data.
    """
    class_name = STAGE_NAMES[class_id]

    # Per-strategy summaries
    strategies_summary = {}
    all_band_frequencies = None

    for strategy_name, data in strategies_data.items():
        if all_band_frequencies is None:
            all_band_frequencies = data["band_frequencies"]

        top_concepts = get_top_concepts(data, top_k=20)

        # Compute statistics
        concepts_list = list(data["concepts"].values())
        weights = [c["weight"] for c in concepts_list]
        mask_values = [c["mask_value"] for c in concepts_list]

        # Compute top band Hz for top 5 concepts
        top_5 = top_concepts[:5]
        top_bands = []
        for dim, concept in top_5:
            band_idx = concept["top_band_idx"]
            top_bands.append(float(all_band_frequencies[band_idx]))

        strategies_summary[strategy_name] = {
            "n_concepts": len(concepts_list),
            "top_dims": [int(dim) for dim, _ in top_concepts[:10]],
            "weight_stats": {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
            },
            "mask_stats": {
                "mean": float(np.mean(mask_values)),
                "std": float(np.std(mask_values)),
                "min": float(np.min(mask_values)),
                "max": float(np.max(mask_values)),
            },
            "avg_top_band_hz": float(np.mean(top_bands)) if top_bands else 0.0,
        }

    # Robust concepts
    robust_dims = overlap_data["robust_dims"]

    # Strategy-specific concepts
    strategy_specific = overlap_data["strategy_specific"]

    # Spectral characteristics of robust concepts
    robust_spectral = {}
    if robust_dims and all_band_frequencies is not None:
        # Find which strategy has these concepts
        for strategy_name, data in strategies_data.items():
            for dim in robust_dims:
                if dim in data["concepts"]:
                    concept = data["concepts"][dim]
                    band_idx = concept["top_band_idx"]
                    robust_spectral[dim] = {
                        "weight": concept["weight"],
                        "mask_value": concept["mask_value"],
                        "top_band_hz": float(all_band_frequencies[band_idx]),
                        "band_energy": concept["band_energy"].tolist(),
                    }
                    break  # Only need first occurrence

    return {
        "class_id": class_id,
        "class_name": class_name,
        "n_strategies": len(strategies_data),
        "strategies": strategies_summary,
        "n_robust_concepts": len(robust_dims),
        "robust_concepts": robust_dims,
        "robust_spectral": robust_spectral,
        "strategy_specific": {
            name: dims for name, dims in strategy_specific.items()
        },
        "overlap_matrix": overlap_data["overlap_matrix"].tolist(),
    }


def save_json_report(
    report_data: dict,
    output_path: Path,
):
    """Save complete report as JSON.

    Args:
        report_data: Full report dict.
        output_path: Path to save JSON.
    """
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Saved: {output_path}")


def save_markdown_report(
    report_data: dict,
    output_path: Path,
):
    """Save human-readable report as Markdown.

    Args:
        report_data: Full report dict.
        output_path: Path to save markdown.
    """
    lines = [
        "# CSD Concept Analysis Report",
        "",
        f"**Analysis of CBRAMod + linear probe concepts across specificity strategies**",
        "",
        f"- **Model**: CBRAMod (D=200)",
        f"- **Dataset**: MASS SS03",
        f"- **Classes analyzed**: {len(report_data['classes'])}",
        f"- **Strategies**: {', '.join(report_data.get('strategies', []))}",
        "",
        "---",
        "",
    ]

    # Per-class summaries
    for class_report in report_data["classes"]:
        class_id = class_report["class_id"]
        class_name = class_report["class_name"]

        lines.append(f"## Class {class_id}: {class_name}")
        lines.append("")

        # Strategy summary table
        lines.append("### Strategy Summary")
        lines.append("")
        lines.append("| Strategy | Concepts | Avg Weight | Avg Mask | Avg Top Band |")
        lines.append("|----------|----------|------------|----------|--------------|")

        for strategy_name, summary in class_report["strategies"].items():
            n_concepts = summary["n_concepts"]
            avg_weight = summary["weight_stats"]["mean"]
            avg_mask = summary["mask_stats"]["mean"]
            avg_band = summary["avg_top_band_hz"]

            lines.append(
                f"| {strategy_name} | {n_concepts} | {avg_weight:+.3f} | "
                f"{avg_mask:.3f} | {avg_band:.1f} Hz |"
            )

        lines.append("")

        # Robust concepts
        lines.append(f"### Robust Concepts ({class_report['n_robust_concept']})")
        lines.append("")
        lines.append("Concepts selected by ≥2 strategies:")

        if class_report["robust_concepts"]:
            lines.append("")
            lines.append("| Dimension | Weight | Top Band |")
            lines.append("|-----------|--------|----------|")

            for dim in class_report["robust_concepts"][:20]:  # Limit to top 20
                spectral = class_report["robust_spectral"].get(dim, {})
                weight = spectral.get("weight", 0)
                top_band = spectral.get("top_band_hz", 0)
                lines.append(f"| d{dim} | {weight:+.3f} | {top_band:.1f} Hz |")
        else:
            lines.append("*No robust concepts found*")

        lines.append("")

        # Strategy-specific concepts
        lines.append("### Strategy-Specific Concepts")
        lines.append("")

        for strategy_name, dims in class_report["strategy_specific"].items():
            if dims:
                lines.append(f"- **{strategy_name}**: {len(dims)} unique concepts")
                if len(dims) <= 10:
                    lines.append(f"  Dimensions: {', '.join(f'd{d}' for d in dims)}")
            else:
                lines.append(f"- **{strategy_name}**: No unique concepts")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Cross-class comparison
    if len(report_data["classes"]) > 1:
        lines.append("## Cross-Class Comparison")
        lines.append("")

        # Find concepts shared across classes
        all_class_concepts = {}
        for class_report in report_data["classes"]:
            class_name = class_report["class_name"]
            all_class_concepts[class_name] = set(class_report["robust_concepts"])

        lines.append("### Overlap Between Classes")
        lines.append("")
        lines.append("| Class A | Class B | Shared Concepts |")
        lines.append("|---------|---------|-----------------|")

        class_names = list(all_class_concepts.keys())
        for i in range(len(class_names)):
            for j in range(i + 1, len(class_names)):
                set_a = all_class_concepts[class_names[i]]
                set_b = all_class_concepts[class_names[j]]
                shared = set_a & set_b
                lines.append(
                    f"| {class_names[i]} | {class_names[j]} | "
                    f"{len(shared)} |"
                )

        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSD concept analysis report"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory with class_X subdirs from extract_all_concepts.py")
    parser.add_argument("--output_dir", type=str, default="./csd_report")
    parser.add_argument("--all_classes", action="store_true",
                       help="Generate report for all 5 classes")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine classes to process
    if args.all_classes:
        class_ids = list(range(5))
    else:
        # Default to all classes if nothing specified
        class_ids = list(range(5))

    print("=" * 60)
    print("CSD Concept Report Generation")
    print("=" * 60)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Classes: {[STAGE_NAMES[c] for c in class_ids]}")

    # Collect all strategies seen
    all_strategies = set()

    # Generate per-class reports
    classes_data = []

    for class_id in class_ids:
        class_name = STAGE_NAMES[class_id]
        print(f"\n{'=' * 40}")
        print(f"Class {class_id} ({class_name})")
        print(f"{'=' * 40}")

        # Load all strategies for this class
        print(f"[1] Loading concepts...")
        strategies_data = load_all_strategies_for_class(class_id, input_dir)

        if not strategies_data:
            print(f"  WARNING: No strategies found for class {class_id}")
            continue

        all_strategies.update(strategies_data.keys())
        print(f"  Loaded {len(strategies_data)} strategies: {list(strategies_data.keys())}")

        # Compute overlap
        print(f"[2] Computing overlap...")
        overlap_data = compute_concept_overlap(strategies_data)

        # Generate class report
        print(f"[3] Generating class report...")
        class_report = generate_class_report(
            class_id, strategies_data, overlap_data
        )
        classes_data.append(class_report)

        print(f"  {len(class_report['robust_concepts'])} robust concepts")
        for name, summary in class_report["strategies"].items():
            print(f"    {name}: {summary['n_concepts']} concepts")

    if not classes_data:
        print("\nERROR: No class data found!")
        return 1

    # Assemble full report
    print(f"\n{'=' * 60}")
    print("Assembling Full Report")
    print(f"{'=' * 60}")

    full_report = {
        "model": "CBRAMod",
        "dataset": "MASS SS03",
        "n_classes": len(classes_data),
        "strategies": sorted(list(all_strategies)),
        "classes": classes_data,
    }

    # Save JSON
    json_path = output_dir / "csd_report.json"
    save_json_report(full_report, json_path)

    # Save Markdown
    md_path = output_dir / "csd_report.md"
    save_markdown_report(full_report, md_path)

    print("\n" + "=" * 60)
    print("Report Generation Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  {json_path}")
    print(f"  {md_path}")

    return 0


if __name__ == "__main__":
    exit(main())
