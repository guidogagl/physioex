"""Extract CSD concepts for all classes and strategies.

This script:
1. Loads a trained linear probe (from train_probe.py)
2. Loads CBRAMod encoder
3. Loads a subject's raw signal from MASS SS03
4. Runs CSD explanation for each class (W, N1, N2, N3, REM) and each strategy
5. Saves per-concept attributions to .npz files

Usage:
    # Single class, single strategy
    python examples/analyze_csd_concepts/extract_all_concepts.py \\
        --subject_id 01-03-0002 \\
        --class_id 3 \\
        --strategies margin \\
        --output_dir ./csd_test

    # All classes, all strategies
    python examples/analyze_csd_concepts/extract_all_concepts.py \\
        --subject_id 01-03-0002 \\
        --all_classes \\
        --all_strategies \\
        --output_dir ./csd_concepts_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
# Disable cuDNN temporarily if not available (for CUDA environments without cuDNN)
try:
    import torch.backends.cudnn as cudnn
    cudnn.enabled = False
except Exception:
    pass

from physioex.data.datasets import MASSDataset
from physioex.models import CBraModEncoder
from examples.explain.conceptualspectraldecomposition.utils import load_probe
from examples.analyze_csd_concepts.utils import (
    get_strategy,
    extract_concepts_for_class,
    save_concepts_npz,
    STAGE_NAMES,
)

MODEL_NAME = "cbramod"
DATASET_NAME = "mass_ss03"
EMBEDDING_DIM = 200
N_CLASSES = 5

# Available strategies (dynamic only, no CohenD)
AVAILABLE_STRATEGIES = ["margin", "softmax", "topk", "nofilter"]

DEFAULT_PROBE_PATH = "/tmp/csd_test_checkpoints/probe.pt"


def main():
    parser = argparse.ArgumentParser(
        description=f"CSD Concept Extraction: {MODEL_NAME} on {DATASET_NAME}"
    )
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--probe_path", type=str, default=DEFAULT_PROBE_PATH)
    parser.add_argument("--subject_id", type=str, default="01-03-0002")
    parser.add_argument("--class_id", type=int, default=None,
                       help="Single class to explain (0-4)")
    parser.add_argument("--all_classes", action="store_true",
                       help="Extract concepts for all 5 classes")
    parser.add_argument("--strategies", type=str, nargs="+", default=None,
                       choices=AVAILABLE_STRATEGIES,
                       help="Strategies to use")
    parser.add_argument("--all_strategies", action="store_true",
                       help="Use all available strategies")
    parser.add_argument("--output_dir", type=str, default="./csd_concepts_analysis")
    parser.add_argument("--tau", type=float, default=0.5,
                       help="Tau parameter for Margin/Softmax strategies")
    parser.add_argument("--k", type=int, default=10,
                       help="K parameter for TopK strategy")
    parser.add_argument("--freq_step", type=float, default=4.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--max_epochs", type=int, default=None,
                       help="Limit number of epochs to process")
    args = parser.parse_args()

    if args.gpu_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu_id}"

    print("=" * 60)
    print(f"CSD Concept Extraction: {MODEL_NAME} on {DATASET_NAME}")
    print("=" * 60)

    # Determine classes to process
    if args.all_classes:
        class_ids = list(range(N_CLASSES))
    elif args.class_id is not None:
        class_ids = [args.class_id]
    else:
        parser.error("Must specify either --class_id or --all_classes")

    # Determine strategies to use
    if args.all_strategies:
        strategies = AVAILABLE_STRATEGIES
    elif args.strategies:
        strategies = args.strategies
    else:
        parser.error("Must specify either --strategies or --all_strategies")

    print(f"\nConfiguration:")
    print(f"  Subject: {args.subject_id}")
    print(f"  Classes: {[STAGE_NAMES[c] for c in class_ids]}")
    print(f"  Strategies: {strategies}")
    print(f"  Device: {device}")

    # 1. Load dataset
    print(f"\n[1] Loading dataset...")
    dataset = MASSDataset(
        cohort=3,  # SS03
        root=None,  # Let the system auto-detect
        channels=["EEG", "EOG", "EMG"],
        pipelines="cbramod",
        sequence_length=1,
    )

    subject_ids = dataset.get_subjects()
    if args.subject_id not in subject_ids:
        print(f"  ERROR: Subject {args.subject_id} not found!")
        print(f"  Available: {', '.join(subject_ids[:5])}...")
        return 1

    print(f"  Subject: {args.subject_id}")

    # 2. Load model
    print(f"\n[2] Loading {MODEL_NAME} encoder...")
    model = CBraModEncoder(in_chan=len(dataset.channels))
    model = model.to(device).eval()
    print(f"  Embedding dim: {EMBEDDING_DIM}")

    # 3. Load probe
    print(f"\n[3] Loading probe from {args.probe_path}...")
    probe_path = Path(args.probe_path)
    if not probe_path.exists():
        print(f"  ERROR: Probe not found at {probe_path}")
        print(f"  Please train probe first:")
        print(f"    python examples/explain/conceptualspectraldecomposition/train_probe.py")
        return 1

    probe_weights = load_probe(
        probe_path=probe_path,
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

        # Limit epochs if requested
        if args.max_epochs is not None and len(signals) > args.max_epochs:
            signals = signals[:args.max_epochs]
            labels = labels[:args.max_epochs]
            print(f"  Limited to {args.max_epochs} epochs")

        print(f"  Signal shape: {signals.shape} (N={len(signals)} scored epochs)")
        print(f"  Channels: {item['channel_order']}")

    except Exception as e:
        print(f"  ERROR loading subject: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 5. Get sampling rate
    fs = 200.0
    print(f"  Sampling rate: {fs} Hz")

    # 6. Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 7. Run CSD for each (class, strategy) combination
    print(f"\n[5] Running CSD explanations...")
    print(f"  Total: {len(class_ids)} classes × {len(strategies)} strategies = "
          f"{len(class_ids) * len(strategies)} explanations")

    for class_id in class_ids:
        class_name = STAGE_NAMES[class_id]
        print(f"\n  Class {class_id} ({class_name}):")

        # Create class directory
        class_dir = output_dir / f"class_{class_id}_{class_name}"
        class_dir.mkdir(parents=True, exist_ok=True)

        for strategy_name in strategies:
            print(f"    Strategy: {strategy_name}...", end=" ", flush=True)

            # Get strategy instance
            strategy = get_strategy(strategy_name, tau=args.tau, k=args.k)

            # Run CSD
            try:
                result = extract_concepts_for_class(
                    model=model,
                    probe_weights=probe_weights,
                    signals=signals,
                    target_class=class_id,
                    strategy=strategy,
                    fs=fs,
                    freq_step=args.freq_step,
                    mask_threshold=args.mask_threshold,
                    device=device,
                )

                # Save to .npz
                output_path = class_dir / f"{strategy_name}_concepts.npz"
                save_concepts_npz(
                    result=result,
                    output_path=output_path,
                    subject_id=args.subject_id,
                    strategy_name=strategy_name,
                )

                print(f"✓ ({len(result.concepts)} concepts)")

            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 8. Summary
    print("\n" + "=" * 60)
    print("CSD Concept Extraction Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    for class_id in class_ids:
        class_name = STAGE_NAMES[class_id]
        class_dir = output_dir / f"class_{class_id}_{class_name}"
        if class_dir.exists():
            npz_files = list(class_dir.glob("*.npz"))
            print(f"  {class_name}: {len(npz_files)} strategy files")

    print(f"\nNext steps:")
    print(f"  1. Compare strategies:")
    print(f"     python examples/analyze_csd_concepts/compare_strategies.py \\")
    print(f"         --input_dir {output_dir} --all_classes")
    print(f"  2. Visualize concepts:")
    print(f"     python examples/analyze_csd_concepts/visualize_concepts.py \\")
    print(f"         --input_dir {output_dir} --class_id 3")
    print(f"  3. Generate report:")
    print(f"     python examples/analyze_csd_concepts/report.py \\")
    print(f"         --input_dir {output_dir} --all_classes")

    return 0


if __name__ == "__main__":
    exit(main())
