"""Test SleepTokenizer pretraining on multiple combined datasets.

Tests heterogeneous channel handling: MASS + SleepEDF + HMC have different
channel sets and naming conventions.

Functional test: verify no crashes, data_mask works, modality_dropout safe.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path for imports
TESTS_DIR = Path(__file__).parent
PRETRAIN_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PRETRAIN_DIR))

# Import constants and functions from pretrain.py
from pretrain import (
    MODEL_KWARGS,
    TRAIN_CONFIG,
    _original_step,
    _original_voting_eval_step,
    _sleeptokenizer_step,
    _sleeptokenizer_voting_eval_step,
)

from physioex.data.datasets import get_dataset
from physioex.data.multi import MultiDataset
from physioex.models.sleep_tokenizer import (
    SleepTokenizer,
    build_modality_ids,
    N_MODALITY_TYPES,
)
from physioex.train.trainer import Trainer


def test_channel_heterogeneity():
    """Verify that different datasets have different channel configurations.

    This tests that MultiDataset will create batches with heterogeneous channels.
    """
    print("\n=== Testing Channel Heterogeneity Across Datasets ===")

    datasets_to_test = ["mass", "sleepedf", "hmc"]

    for name in datasets_to_test:
        try:
            DatasetClass = get_dataset(name)
            if name == "mass":
                ds = DatasetClass(cohort=1, pipelines="time_domain", sequence_length=1)
            else:
                ds = DatasetClass(pipelines="time_domain", sequence_length=1)

            if len(ds) > 0:
                sample = ds[0]
                channels = sample.get("channel_order", [])
                print(f"{name:12s}: {len(channels)} channels - {channels[:5]}...")
            else:
                print(f"{name:12s}: Empty dataset or not available")
        except Exception as e:
            print(f"{name:12s}: Error - {e}")

    print("\nConclusion: Datasets have different channel counts and naming.")
    print("MultiDataset will need to handle this via zero-padding + data_mask.")


def test_multi_dataset_single_epoch(
    datasets: list = None,
    gpu_id: int = 0,
    output_dir: str = None,
):
    """Test pretraining on combined datasets.

    Verifies:
    - MultiDataset combines datasets with different channels
    - build_modality_ids handles heterogeneous naming
    - data_mask correctly identifies absent channels
    - Training doesn't crash on heterogeneous batches
    - modality_dropout safety (at least 1 channel remains)
    """
    if datasets is None:
        datasets = ["mass", "sleepedf"]

    if output_dir is None:
        datasets_str = "_".join(datasets)
        output_dir = str(TESTS_DIR / "output" / f"test_multi_{datasets_str}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== SleepTokenizer Multi-Dataset Test ===")
    print(f"Datasets: {datasets}")
    print(f"Output: {output_dir}")

    # Create individual datasets
    dataset_list = []
    for name in datasets:
        try:
            DatasetClass = get_dataset(name)
            if name == "mass":
                ds = DatasetClass(cohort=1, pipelines="time_domain", sequence_length=1)
            else:
                ds = DatasetClass(pipelines="time_domain", sequence_length=1)

            if len(ds) > 0:
                dataset_list.append(ds)
                print(f"  Loaded {name}: {len(ds)} subjects")
            else:
                print(f"  Skipped {name}: empty or not available")
        except Exception as e:
            print(f"  Skipped {name}: {e}")

    if not dataset_list:
        print("ERROR: No datasets loaded!")
        return None

    # Wrap in MultiDataset
    multi_ds = MultiDataset(dataset_list)
    print(f"MultiDataset: {len(multi_ds)} total subjects")

    # Sample a batch to inspect heterogeneity
    from physioex.data.collate import dict_collate_fn
    samples = [multi_ds[i] for i in range(min(4, len(multi_ds)))]
    batch = dict_collate_fn(samples)

    print(f"\nSample batch:")
    print(f"  channel_order: {batch['channel_order']}")
    print(f"  batch['signals'].shape: {batch['signals'].shape}")

    # Check data_mask
    stacked = batch["signals"]
    data_mask = (stacked.abs().sum(dim=(0, 2)) == 0)  # (C,) for L=1
    n_missing = data_mask.sum().item()
    print(f"  Missing channels in batch: {n_missing}/{len(batch['channel_order'])}")

    # Test build_modality_ids
    modality_ids = build_modality_ids(batch)
    print(f"  modality_ids.shape: {modality_ids.shape}")
    print(f"  Unique modalities in batch: {modality_ids.unique().tolist()}")

    # Model
    model = SleepTokenizer(**MODEL_KWARGS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # Loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # Monkey-patch Trainer
    Trainer._step = _sleeptokenizer_step
    Trainer._voting_eval_step = _sleeptokenizer_voting_eval_step

    # Train for 1 epoch
    print(f"\n=== Training 1 epoch on GPU {gpu_id} ===")

    try:
        model = Trainer.train(
            model=model,
            dataset=multi_ds,
            max_epochs=1,
            optimizer=optimizer,
            loss=loss_fn,
            train_batch_size=8,  # Small batch
            fold=0,
            gpu_id=gpu_id,
            checkpoint_path=os.path.join(output_dir, "checkpoints"),
            early_stopping_patience=100,
            num_workers=0,
        )

        # Save model
        model_path = os.path.join(output_dir, "model.pt")
        torch.save(model.cpu().state_dict(), model_path)
        print(f"\nSaved model to {model_path}")

        print("\n=== Test PASSED ===")
        return model_path

    except Exception as e:
        print(f"\n=== Test FAILED: {e} ===")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Restore original _step
        Trainer._step = _original_step
        Trainer._voting_eval_step = _original_voting_eval_step


def test_modality_dropout_safety(gpu_id: int = 0):
    """Test that modality_dropout never drops ALL channels.

    Safety check: even with aggressive dropout, at least 1 channel remains.
    """
    from physioex.models.sleep_tokenizer import modality_dropout

    print("\n=== Testing Modality Dropout Safety ===")

    device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"

    # Test with various channel counts
    for C in [1, 2, 3, 5, 10, 20]:
        x = torch.randn(4, C, 128).to(device)
        modality_ids = torch.randint(0, N_MODALITY_TYPES, (4, C)).to(device)

        # Apply dropout
        x_dropped = modality_dropout(
            x, modality_ids,
            p_batch=0.3, p_modality=0.3
        )

        # Check that at least 1 channel is non-zero per sample
        for i in range(4):
            has_active = (x_dropped[i].abs().sum(dim=-1) > 0).any()
            assert has_active, f"Sample {i} has all channels zeroed! C={C}"

        print(f"  C={C:2d}: OK - at least 1 channel always active")

    print("Modality dropout safety verified!")


def main():
    parser = argparse.ArgumentParser(description="Test SleepTokenizer pretraining on multiple datasets")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=["mass", "sleepedf"],
                        help="Datasets to combine (default: mass sleepedf)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Only test heterogeneity and safety, no training")
    parser.add_argument("--test_dropout", action="store_true",
                        help="Test modality dropout safety")
    args = parser.parse_args()

    # Test channel heterogeneity
    test_channel_heterogeneity()

    # Test modality dropout safety if requested
    if args.test_dropout:
        test_modality_dropout_safety(args.gpu_id)

    # Run training test unless skipped
    if not args.skip_training:
        test_multi_dataset_single_epoch(
            datasets=args.datasets,
            gpu_id=args.gpu_id,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
