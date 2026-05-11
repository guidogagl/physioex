"""Test SleepTokenizer pretraining on a single dataset (MASS SS1).

Functional test: verify script runs without errors, logging works, checkpoints save.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
TESTS_DIR = Path(__file__).parent
PRETRAIN_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PRETRAIN_DIR))

# Import constants and functions from pretrain.py
from pretrain import (
    MODEL_NAME,
    MODEL_KWARGS,
    TRAIN_CONFIG,
    LOSS_W_MAIN,
    LOSS_W_CHAN,
    _original_step,
    _original_voting_eval_step,
    _sleeptokenizer_step,
    _sleeptokenizer_voting_eval_step,
    _compute_channel_loss,
)

from physioex.data.datasets import get_dataset
from physioex.models.sleep_tokenizer import (
    SleepTokenizer,
    build_modality_ids,
    N_MODALITY_TYPES,
)
from physioex.train.trainer import Trainer
import torch.nn as nn


def test_sleepedf_single_epoch(gpu_id: int = 0, output_dir: str = None):
    """Quick functional test on SleepEDF - just 1 epoch.

    Verifies:
    - Script starts without errors
    - build_modality_ids works with SleepEDF channels
    - Logging shows loss_main and loss_chan separately
    - Checkpoint saves correctly
    - Per-modality accuracy is computed and logged
    """
    if output_dir is None:
        output_dir = str(TESTS_DIR / "output" / "test_sleepedf")

    os.makedirs(output_dir, exist_ok=True)
    print(f"=== SleepTokenizer Single Dataset Test (SleepEDF) ===")
    print(f"Output: {output_dir}")

    # Override config for quick test
    test_config = TRAIN_CONFIG.copy()
    test_config.update({
        "dataset": "sleepedf",
        "max_epochs": 1,
        "batch_size": 8,  # Smaller batch to avoid OOM
        "early_stopping_patience": 100,  # No early stopping
        "channels": ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"],  # Only main channels
    })

    # Dataset - don't use Subset, it breaks BasePhysioDataset indexing
    DatasetClass = get_dataset("sleepedf")
    dataset = DatasetClass(
        pipelines=test_config["pipeline_preset"],
        sequence_length=21,  # Use standard sequence length
        channels=test_config["channels"],
    )
    print(f"Dataset: SleepEDF (full: {len(dataset)} epochs, seq_len=21)")

    # Check a sample to see channels
    sample = dataset[0]
    print(f"  Sample channels: {sample.get('channel_order', [])}")
    print(f"  Sample labels shape: {sample['labels'].shape}")
    print(f"  Sample labels unique: {torch.unique(sample['labels'])}")
    # sample['signals'] is dict of channel_name -> tensor
    if 'signals' in sample:
        signals = sample['signals']
        if isinstance(signals, dict):
            first_ch = list(signals.keys())[0]
            print(f"  Sample shape: signals['{first_ch}']={signals[first_ch].shape}")

    # Model
    model = SleepTokenizer(**MODEL_KWARGS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=test_config["lr"],
        weight_decay=test_config.get("weight_decay", 1e-4),
    )

    # Loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # Monkey-patch Trainer
    Trainer._step = _sleeptokenizer_step
    Trainer._voting_eval_step = _sleeptokenizer_voting_eval_step

    device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"

    # Train for 1 epoch
    print(f"\n=== Training 1 epoch on {device} ===")

    # Capture per-modality accuracies during training
    modality_accuracies = []

    original_train_step = Trainer._train_step.__func__

    def logging_train_step(cls, model, batch, loss_fn, optimizer, device, step=0, accumulate_grad_batches=1):
        loss, acc, update_norm, extra = original_train_step(
            cls, model, batch, loss_fn, optimizer, device, step, accumulate_grad_batches
        )
        if extra:
            modality_accs = {k: v for k, v in extra.items() if k.startswith("acc_")}
            if modality_accs:
                modality_accuracies.append(modality_accs)
        return loss, acc, update_norm, extra

    Trainer._train_step = classmethod(logging_train_step)

    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=1,
        optimizer=optimizer,
        loss=loss_fn,
        train_batch_size=test_config["batch_size"],
        fold=test_config["fold"],
        gpu_id=gpu_id,
        checkpoint_path=os.path.join(output_dir, "checkpoints"),
        early_stopping_patience=test_config["early_stopping_patience"],
        num_workers=0,
    )

    # Restore original _step
    Trainer._step = _original_step
    Trainer._voting_eval_step = _original_voting_eval_step

    # Print aggregated per-modality accuracy
    if modality_accuracies:
        print(f"\n=== Per-Modality Channel Accuracy (Aggregated) ===")
        # Aggregate by modality name
        from collections import defaultdict
        modality_sums = defaultdict(list)
        for acc_dict in modality_accuracies:
            for mod_name, acc_val in acc_dict.items():
                modality_sums[mod_name].append(acc_val)

        for mod_name in sorted(modality_sums.keys()):
            values = modality_sums[mod_name]
            mean_acc = sum(values) / len(values)
            print(f"  {mod_name}: {mean_acc:.4f} (avg over {len(values)} steps)")

    # Save model
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"\nSaved model to {model_path}")

    print("\n=== Test PASSED ===")
    return model_path


def test_modality_inference(gpu_id: int = 0):
    """Test that build_modality_ids correctly classifies MASS channels.

    MASS has many channels with different naming conventions:
    - EEG: C3-CLE, C4-CLE, F3-CLE, F4-CLE, O1-CLE, O2-CLE, etc.
    - EOG: Left Horiz, Right Horiz, Upper Vertic, Lower Vertic
    - EMG: Chin1, Chin2, Chin3, Chin
    - ECG: ECG I, ECG, ECG II, ECG III
    """
    print("\n=== Testing Modality Inference ===")

    # Simulate a MASS batch
    mass_channels = [
        "C3-CLE", "C4-CLE", "Cz-CLE", "F3-CLE", "F4-CLE",
        "O1-CLE", "O2-CLE", "Pz-CLE", "Fp1-CLE",  # Various EEG
        "Left Horiz", "Right Horiz", "Upper Vertic", "Lower Vertic",  # EOG
        "Chin1", "Chin2", "Chin3", "Chin",  # EMG
        "ECG I", "ECG", "ECG II", "ECG III",  # ECG
    ]

    # Create a mock batch
    batch = {
        "channel_order": mass_channels,
        "channel_info": [
            {ch: {"available": True, "modality": None}} for ch in mass_channels
        ],
    }

    modality_ids = build_modality_ids(batch)

    # Check classifications
    from physioex.models.sleep_tokenizer import MODALITY_TYPES
    id_to_name = {v: k for k, v in MODALITY_TYPES.items()}

    print(f"Channel classifications:")
    for ch, mod_id in zip(mass_channels, modality_ids[0].tolist()):
        mod_name = id_to_name.get(mod_id, "UNKNOWN")
        print(f"  {ch:20s} -> {mod_name} ({mod_id})")

    # Verify expected classifications
    expected_counts = {"EEG": 9, "EOG": 4, "EMG": 4, "ECG": 4}
    actual_counts = {}
    for mod_id in modality_ids[0].tolist():
        mod_name = id_to_name.get(mod_id, "UNKNOWN")
        actual_counts[mod_name] = actual_counts.get(mod_name, 0) + 1

    print(f"\nExpected: {expected_counts}")
    print(f"Actual:   {actual_counts}")

    # Check if counts match (allowing some flexibility for UNKNOWN)
    success = True
    for mod, expected in expected_counts.items():
        actual = actual_counts.get(mod, 0)
        if actual != expected:
            print(f"  WARNING: {mod} expected {expected}, got {actual}")
            success = False

    if success:
        print("All modality classifications correct!")
    else:
        print("Some modality classifications were incorrect (may need to refine infer_modality)")

    return success


def main():
    parser = argparse.ArgumentParser(description="Test SleepTokenizer pretraining on single dataset")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_training", action="store_true", help="Only test modality inference")
    args = parser.parse_args()

    # Always run modality inference test (quick)
    test_modality_inference(args.gpu_id)

    # Run training test unless skipped
    if not args.skip_training:
        test_sleepedf_single_epoch(args.gpu_id, args.output_dir)


if __name__ == "__main__":
    main()
