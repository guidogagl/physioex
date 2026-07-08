"""CSD explanation script for trained linear probe.

This script:
1. Loads a trained linear probe (from train_probe.py)
2. Loads CBRAMod encoder
3. Loads a subject's raw signal from MASS SS03
4. Runs CSD (Conceptual Spectral Decomposition) explanation
5. Generates and saves visualizations

Usage:
    python examples/explain/conceptualspectraldecomposition/explain_csd.py \\
        --gpu_id 0 \\
        --probe_path ./csd_checkpoints/probe.pt \\
        --subject_id 01-01-0001 \\
        --target_class 3 \\
        --output_dir ./csd_results
"""
import argparse
from pathlib import Path

import torch

from physioex.data.datasets import MASSDataset
from physioex.models import CBraModEncoder
from physioex.explain.foundational import (
    ConceptualSpectralDecomposition,
    MarginSpecificity,
)
from examples.explain.conceptualspectraldecomposition.utils import load_probe
from examples.explain.conceptualspectraldecomposition.visualize import (
    save_all_plots,
    plot_summary_figure,
)

MODEL_NAME = "cbramod"
DATASET_NAME = "mass_ss03"
COHORT = 3
EMBEDDING_DIM = 200
N_CLASSES = 5

# Sleep stage names
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser(
        description=f"CSD explanation for {MODEL_NAME} on {DATASET_NAME}"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--subject_id", type=str, default=None)
    parser.add_argument("--target_class", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./csd_results")
    parser.add_argument("--max_concepts", type=int, default=None)
    parser.add_argument("--specificity_tau", type=float, default=0.5)
    parser.add_argument("--freq_step", type=float, default=4.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--top_k_plots", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=None,
                       help="Limit number of epochs to process (for GPU memory)")
    args = parser.parse_args()

    if args.gpu_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu_id}"

    print("=" * 60)
    print(f"CSD Explanation: {MODEL_NAME} on {DATASET_NAME}")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1] Loading dataset...")
    dataset = MASSDataset(
        cohort=COHORT,
        channels=["EEG", "EOG", "EMG"],
        pipelines="cbramod",
        sequence_length=1,
    )

    subject_ids = dataset.get_subjects()

    if args.subject_id is None:
        args.subject_id = subject_ids[0]
        print(f"  No subject_id provided, using: {args.subject_id}")
    elif args.subject_id not in subject_ids:
        print(f"  WARNING: Subject {args.subject_id} not found!")
        print(f"  Available subjects: {', '.join(subject_ids[:5])}...")
        args.subject_id = subject_ids[0]
        print(f"  Using: {args.subject_id}")

    print(f"  Subject: {args.subject_id}")
    print(f"  Target class: {args.target_class} ({STAGE_NAMES[args.target_class]})")

    # 2. Load model
    print(f"\n[2] Loading {MODEL_NAME} encoder...")
    model = CBraModEncoder(in_chan=len(dataset.channels))
    model = model.to(device).eval()
    print(f"  Embedding dim: {EMBEDDING_DIM}")

    # 3. Load probe
    print(f"\n[3] Loading probe from {args.probe_path}...")
    probe_weights = load_probe(
        probe_path=args.probe_path,
        embedding_dim=EMBEDDING_DIM,
        n_classes=N_CLASSES,
        device=device,
    )
    print(f"  Probe loaded: W shape = {probe_weights['W'].shape}")

    # 4. Load subject signal
    print(f"\n[4] Loading subject signal...")
    try:
        spec = next(s for s in dataset._subjects if s.subject_id == args.subject_id)
        n_epochs = dataset._n_epochs[args.subject_id]

        # Load full recording
        item = dataset._build_item(spec, 0, n_epochs)
        ch_tensors = [item["signals"][ch] for ch in item["channel_order"]]
        signals = torch.stack(ch_tensors, dim=1)  # (n_epochs, C, T)
        labels = item["labels"]

        # Keep only scored epochs (label >= 0)
        valid_mask = labels >= 0
        signals = signals[valid_mask]  # (N, C, T)
        labels = labels[valid_mask]

        # Limit epochs for GPU memory
        if args.max_epochs is not None and len(signals) > args.max_epochs:
            signals = signals[:args.max_epochs]
            labels = labels[:args.max_epochs]
            print(f"  Limited to {args.max_epochs} epochs for GPU memory")

        print(f"  Signal shape: {signals.shape} (N={len(signals)} scored epochs)")
        print(f"  Channels: {item['channel_order']}")

    except Exception as e:
        print(f"  ERROR loading subject: {e}")
        raise

    # 5. Get sampling rate
    # CBRAMod uses 200 Hz as standard sampling rate
    fs = 200.0
    print(f"  Sampling rate: {fs} Hz")

    # 6. Create CSD explainer
    print(f"\n[5] Creating CSD explainer...")
    csd = ConceptualSpectralDecomposition(
        model=model,
        probe_weights=probe_weights,
        fs=fs,
        freq_step=args.freq_step,
        specificity=MarginSpecificity(tau=args.specificity_tau),
        mask_threshold=args.mask_threshold,
        device=device,
    )
    print(f"  Freq step: {args.freq_step} Hz")
    print(f"  Specificity: Margin (tau={args.specificity_tau})")
    print(f"  Mask threshold: {args.mask_threshold}")

    # 7. Run CSD explanation
    print(f"\n[6] Running CSD explanation...")
    result = csd.explain(
        signals.to(device),
        target_class=args.target_class,
        max_concepts=args.max_concepts,
    )

    print(f"\n  Results:")
    print(f"    Selected concepts: {len(result.concepts)}")
    print(f"    Class attribution shape: {result.class_attribution.shape}")
    print(f"    Band frequencies: {len(result.band_frequencies)} bands")

    # Print top concepts
    print(f"\n  Top {min(5, len(result.concepts))} concepts:")
    sorted_concepts = sorted(
        result.concepts.items(),
        key=lambda x: abs(x[1].mask_value * x[1].weight),
        reverse=True,
    )[:5]
    for dim, concept in sorted_concepts:
        top_freq = result.top_band_hz(dim)
        print(
            f"    Dim {dim}: W={concept.weight:+.3f}, "
            f"mask={concept.mask_value:.2f}, top_freq={top_freq:.1f} Hz"
        )

    # 8. Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[7] Saving results to {output_dir}...")

    # Save concepts data
    concepts_data = {}
    for dim, concept in result.concepts.items():
        concepts_data[str(dim)] = {
            "weight": float(concept.weight),
            "mask_value": float(concept.mask_value),
            "top_band_hz": float(result.top_band_hz(dim)),
        }

    import json

    metadata = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "subject_id": args.subject_id,
        "target_class": int(args.target_class),
        "target_class_name": STAGE_NAMES[args.target_class],
        "n_epochs": int(signals.shape[0]),
        "n_concepts": len(result.concepts),
        "selected_dims": result.selected_dims,
        "specificity": result.specificity_name,
        "concepts": concepts_data,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: metadata.json")

    # Save class attribution as numpy
    class_attr_np = result.class_attribution.mean(dim=0).cpu().numpy()
    import numpy as np

    np.save(output_dir / "class_attribution.npy", class_attr_np)
    print(f"  Saved: class_attribution.npy (shape={class_attr_np.shape})")

    # 9. Generate visualizations
    print(f"\n[8] Generating visualizations...")
    save_all_plots(
        result,
        output_dir=output_dir,
        top_k=args.top_k_plots,
    )

    # Summary figure
    summary_path = output_dir / f"csd_summary_{STAGE_NAMES[args.target_class]}.png"
    plot_summary_figure(result, summary_path, top_k=min(6, len(result.concepts)))

    print("\n" + "=" * 60)
    print("CSD explanation complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - metadata.json")
    print(f"  - class_attribution.npy")
    print(f"  - class_attribution.png")
    print(f"  - top_concepts.png")
    print(f"  - per_channel_energy.png")
    print(f"  - csd_summary_*.png")
    print(f"  - top_concept_*.png ({len(result.concepts)} concepts)")


if __name__ == "__main__":
    main()
