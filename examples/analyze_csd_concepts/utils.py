"""Utility functions for CSD concept analysis.

Provides functions to:
- Extract concepts for a single class and strategy
- Save/load concept data from .npz files
- Compute concept overlap across strategies
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from physioex.explain.foundational import (
    ConceptualSpectralDecomposition,
    CSDResult,
    MarginSpecificity,
    SoftmaxSpecificity,
    TopKSpecificity,
    NoFilter,
)
from physioex.explain.foundational.specificity import SpecificityStrategy

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

# Strategy registry
STRATEGIES: Dict[str, type[SpecificityStrategy]] = {
    "margin": MarginSpecificity,
    "softmax": SoftmaxSpecificity,
    "topk": TopKSpecificity,
    "nofilter": NoFilter,
}


def get_strategy(
    strategy_name: str, tau: float = 0.5, k: int = 10
) -> SpecificityStrategy:
    """Get a specificity strategy instance by name.

    Args:
        strategy_name: One of "margin", "softmax", "topk", "nofilter".
        tau: Tau parameter for Margin/Softmax strategies.
        k: K parameter for TopK strategy.

    Returns:
        SpecificityStrategy instance.
    """
    strategy_cls = STRATEGIES.get(strategy_name.lower())
    if strategy_cls is None:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(STRATEGIES.keys())}"
        )

    if strategy_name == "topk":
        return strategy_cls(k=k)
    elif strategy_name == "nofilter":
        return strategy_cls()
    else:
        return strategy_cls(tau=tau)


def extract_concepts_for_class(
    model: torch.nn.Module,
    probe_weights: Dict[str, torch.nn.Module],
    signals: torch.Tensor,
    target_class: int,
    strategy: SpecificityStrategy,
    fs: float = 200.0,
    freq_step: float = 4.0,
    mask_threshold: float = 0.5,
    device: str = "cpu",
) -> CSDResult:
    """Extract CSD concepts for a single class using a given strategy.

    Args:
        model: Foundation encoder (CBraModEncoder).
        probe_weights: Dict with "ln" and "W" from trained probe.
        signals: (B, C, T) input signal tensor.
        target_class: Class index to explain (0-4).
        strategy: SpecificityStrategy instance.
        fs: Sampling rate (Hz).
        freq_step: Frequency step for SpectralGradients (Hz).
        mask_threshold: Mask threshold for concept selection.
        device: Torch device.

    Returns:
        CSDResult with concepts dict, band_frequencies, etc.
    """
    csd = ConceptualSpectralDecomposition(
        model=model,
        probe_weights=probe_weights,
        fs=fs,
        freq_step=freq_step,
        specificity=strategy,
        mask_threshold=mask_threshold,
        device=device,
    )

    result = csd.explain(
        signals.to(device),
        target_class=target_class,
        max_concepts=None,  # All concepts above threshold
    )

    return result


def save_concepts_npz(
    result: CSDResult,
    output_path: Union[str, Path],
    subject_id: str,
    strategy_name: str,
):
    """Save CSD result concepts to .npz file.

    Saves per-concept attribution maps and metadata.

    Args:
        result: CSDResult from CSD explanation.
        output_path: Path to save .npz file.
        subject_id: Subject ID for metadata.
        strategy_name: Strategy name for metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data dict
    save_dict = {}

    # Metadata
    save_dict["subject_id"] = subject_id
    save_dict["target_class"] = result.target_class
    save_dict["target_class_name"] = STAGE_NAMES[result.target_class]
    save_dict["specificity"] = result.specificity_name
    save_dict["strategy_name"] = strategy_name
    save_dict["n_concepts"] = len(result.concepts)
    save_dict["selected_dims"] = np.array(result.selected_dims, dtype=np.int32)
    save_dict["band_frequencies"] = result.band_frequencies.cpu().numpy()

    # Probe weights (detach if requires grad)
    probe_weights = result.probe_weights
    if probe_weights.requires_grad:
        probe_weights = probe_weights.detach()
    save_dict["probe_weights"] = probe_weights.cpu().numpy()

    # Per-concept data
    for dim, concept in result.concepts.items():
        prefix = f"dim_{dim:03d}"

        # Attribution map: (B, n_bands, C, T)
        save_dict[f"{prefix}_attribution"] = concept.attribution.cpu().numpy()

        # Metadata
        save_dict[f"{prefix}_weight"] = concept.weight
        save_dict[f"{prefix}_mask_value"] = concept.mask_value
        save_dict[f"{prefix}_contribution"] = concept.contribution.cpu().numpy()
        save_dict[f"{prefix}_band_energy"] = concept.band_energy.cpu().numpy()
        save_dict[f"{prefix}_channel_energy"] = concept.channel_energy.cpu().numpy()
        save_dict[f"{prefix}_top_band_idx"] = concept.top_band_idx

    # Class-level attribution (for reference)
    save_dict["class_attribution"] = result.class_attribution.cpu().numpy()

    # Save
    np.savez_compressed(output_path, **save_dict)

    print(f"  Saved: {output_path} ({len(result.concepts)} concepts)")


def load_concepts_npz(
    input_path: Union[str, Path],
) -> Dict:
    """Load CSD concepts from .npz file.

    Args:
        input_path: Path to .npz file.

    Returns:
        Dict with metadata and per-concept data.
    """
    data = np.load(input_path, allow_pickle=True)

    result = {
        "metadata": {
            "subject_id": str(data["subject_id"]),
            "target_class": int(data["target_class"]),
            "target_class_name": str(data["target_class_name"]),
            "specificity": str(data["specificity"]),
            "strategy_name": str(data["strategy_name"]),
            "n_concepts": int(data["n_concepts"]),
        },
        "band_frequencies": data["band_frequencies"],
        "probe_weights": data["probe_weights"],
        "selected_dims": data["selected_dims"].tolist(),
        "class_attribution": data["class_attribution"],
        "concepts": {},
    }

    # Load per-concept data
    for dim in result["selected_dims"]:
        prefix = f"dim_{dim:03d}"
        result["concepts"][dim] = {
            "attribution": data[f"{prefix}_attribution"],
            "weight": float(data[f"{prefix}_weight"]),
            "mask_value": float(data[f"{prefix}_mask_value"]),
            "contribution": data[f"{prefix}_contribution"],
            "band_energy": data[f"{prefix}_band_energy"],
            "channel_energy": data[f"{prefix}_channel_energy"],
            "top_band_idx": int(data[f"{prefix}_top_band_idx"]),
        }

    return result


def load_all_strategies_for_class(
    class_id: int,
    input_dir: Union[str, Path],
) -> Dict[str, Dict]:
    """Load concepts for all strategies for a given class.

    Args:
        class_id: Class index (0-4).
        input_dir: Root directory with class_X subdirs.

    Returns:
        Dict mapping strategy_name to concepts data.
    """
    input_dir = Path(input_dir)
    class_dir = input_dir / f"class_{class_id}_{STAGE_NAMES[class_id]}"

    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory not found: {class_dir}")

    results = {}
    for strategy_name in STRATEGIES.keys():
        npz_path = class_dir / f"{strategy_name}_concepts.npz"
        if npz_path.exists():
            results[strategy_name] = load_concepts_npz(npz_path)
        else:
            print(f"  [WARNING] Missing: {npz_path}")

    return results


def compute_concept_overlap(
    strategies_data: Dict[str, Dict],
    top_k: Optional[int] = None,
) -> Dict:
    """Compute concept overlap across strategies.

    Args:
        strategies_data: Dict {strategy_name: concepts_data}.
        top_k: Optional limit to top K concepts per strategy.

    Returns:
        Dict with overlap metrics:
            - "overlap_matrix": (n_strategies, n_strategies) Jaccard index
            - "dim_frequency": {dim: count of strategies selecting it}
            - "robust_dims": dimensions selected by >= 2 strategies
            - "strategy_specific": {strategy: [dims unique to this strategy]}
    """
    strategy_names = list(strategies_data.keys())
    n_strategies = len(strategy_names)

    # Get selected dims per strategy
    strategy_dims = {}
    for name, data in strategies_data.items():
        dims = set(data["selected_dims"])
        if top_k is not None:
            # Sort by |weight * mask| and take top K
            concepts_list = []
            for dim in dims:
                concept = data["concepts"][dim]
                importance = abs(concept["weight"] * concept["mask_value"])
                concepts_list.append((dim, importance))
            concepts_list.sort(key=lambda x: x[1], reverse=True)
            dims = set(d for d, _ in concepts_list[:top_k])
        strategy_dims[name] = dims

    # Compute overlap matrix (Jaccard index)
    overlap_matrix = np.zeros((n_strategies, n_strategies))
    for i, s1 in enumerate(strategy_names):
        for j, s2 in enumerate(strategy_names):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                intersection = len(strategy_dims[s1] & strategy_dims[s2])
                union = len(strategy_dims[s1] | strategy_dims[s2])
                overlap_matrix[i, j] = intersection / union if union > 0 else 0.0

    # Count frequency of each dimension across strategies
    dim_frequency = {}
    for name in strategy_names:
        for dim in strategy_dims[name]:
            dim_frequency[dim] = dim_frequency.get(dim, 0) + 1

    # Robust dimensions (selected by >= 2 strategies)
    robust_dims = [d for d, count in dim_frequency.items() if count >= 2]

    # Strategy-specific dimensions
    strategy_specific = {}
    for name in strategy_names:
        specific_dims = []
        for dim in strategy_dims[name]:
            if dim_frequency[dim] == 1:  # Only this strategy
                specific_dims.append(dim)
        strategy_specific[name] = specific_dims

    return {
        "strategy_names": strategy_names,
        "overlap_matrix": overlap_matrix,
        "dim_frequency": dim_frequency,
        "robust_dims": robust_dims,
        "strategy_specific": strategy_specific,
    }


def get_top_concepts(
    concepts_data: Dict,
    top_k: int = 10,
) -> List[Tuple[int, Dict]]:
    """Get top K concepts by importance |weight * mask|.

    Args:
        concepts_data: Concepts dict from load_concepts_npz.
        top_k: Number of top concepts to return.

    Returns:
        List of (dim, concept_dict) tuples sorted by importance.
    """
    concepts_list = []
    for dim, concept in concepts_data["concepts"].items():
        importance = abs(concept["weight"] * concept["mask_value"])
        concepts_list.append((dim, concept, importance))

    concepts_list.sort(key=lambda x: x[2], reverse=True)
    return [(d, c) for d, c, _ in concepts_list[:top_k]]


def save_overlap_report(
    overlap_data: Dict,
    output_path: Union[str, Path],
    class_id: int,
):
    """Save concept overlap analysis to JSON.

    Args:
        overlap_data: Result from compute_concept_overlap.
        output_path: Path to save JSON.
        class_id: Class index for metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare JSON-serializable dict
    report = {
        "class_id": class_id,
        "class_name": STAGE_NAMES[class_id],
        "n_strategies": len(overlap_data["strategy_names"]),
        "strategies": overlap_data["strategy_names"],
        "overlap_matrix": overlap_data["overlap_matrix"].tolist(),
        "n_robust_dims": len(overlap_data["robust_dims"]),
        "robust_dims": overlap_data["robust_dims"],
        "strategy_specific": overlap_data["strategy_specific"],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Saved: {output_path}")
