"""Quick synthetic test for SleepTokenizer with per-modality accuracy logging."""
import sys
from pathlib import Path

# Add parent directory to path
TESTS_DIR = Path(__file__).parent
PRETRAIN_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PRETRAIN_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pretrain import (
    MODEL_KWARGS,
    _original_step,
    _original_voting_eval_step,
    _sleeptokenizer_step,
)
from physioex.models.sleep_tokenizer import (
    SleepTokenizer,
    MODALITY_TYPES,
    N_MODALITY_TYPES,
)
from physioex.train.trainer import Trainer

def test_synthetic_training():
    """Test with synthetic data to verify per-modality accuracy works."""
    print("=== Synthetic SleepTokenizer Test ===")

    # Create synthetic data
    B, L, C, T = 16, 1, 4, 3000  # batch, seq_len, channels, time
    n_classes = 5

    # Simulate batch with different modalities
    # Channel 0, 1: EEG
    # Channel 2: EOG
    # Channel 3: EMG
    modality_ids = torch.tensor([[0, 0, 1, 2]])  # EEG, EEG, EOG, EMG
    modality_ids = modality_ids.expand(B, C)

    # Create random signals
    x = torch.randn(B, L, C, T)

    # Create random labels (no -1)
    labels = torch.randint(0, n_classes, (B, L))

    # Build mock batch dict
    channel_names = [f"Channel_{i}" for i in range(C)]
    batch = {
        "channel_order": channel_names,
        "channel_info": [{ch: {"available": True}} for ch in channel_names],
    }

    # Create model
    model = SleepTokenizer(**MODEL_KWARGS)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Create loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # Create a minimal optimizer step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Mock training step
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = x.to(device)
    modality_ids = modality_ids.to(device)
    labels = labels.to(device)

    # Build a proper inputs tensor format (stack_channels output)
    from physioex.data.collate import stack_channels

    # For this test, create a simple dict format that stack_channels would produce
    # Actually, let's just use the model directly with the format it expects
    model.train()

    # Forward pass
    outputs = model(x, modality_ids=modality_ids)

    print(f"\nForward pass outputs:")
    print(f"  logits: {outputs['logits'].shape}")
    print(f"  channel_logits: {outputs['channel_logits'].shape}")
    print(f"  embedding: {outputs['embedding'].shape}")
    print(f"  data_mask: {outputs['data_mask'].shape}")

    # Compute loss manually
    logits_flat = outputs["logits"].reshape(-1, n_classes)
    labels_flat = labels.reshape(-1)
    loss = loss_fn(logits_flat, labels_flat)

    print(f"\nLoss: {loss.item():.4f}")

    # Compute accuracy
    preds = logits_flat.argmax(dim=-1)
    acc = (preds == labels_flat).float().mean()
    print(f"Accuracy: {acc.item():.4f}")

    # Compute per-modality accuracy (same logic as in pretrain.py)
    with torch.no_grad():
        channel_logits = outputs["channel_logits"]  # (B, L, C, n_classes)
        data_mask = outputs["data_mask"]  # (B, C)
        B_, L_, C_, _ = channel_logits.shape

        # Expand data_mask to (B, L, C)
        mask_expanded = data_mask.unsqueeze(1).expand(B_, L_, C_)

        # Get predictions per channel
        chan_preds = channel_logits.argmax(dim=-1)  # (B, L, C)
        targets_expanded = labels.unsqueeze(-1).expand(B_, L_, C_)  # (B, L, C)

        # Compute accuracy per channel - ONLY count valid targets (not -1)
        valid_mask = (targets_expanded != -1) & ~mask_expanded
        chan_correct = (chan_preds == targets_expanded) & valid_mask

        # Group by modality type
        modality_expanded = modality_ids.unsqueeze(1).expand(B_, L_, C_)

        modality_stats = {}
        for mod_id in range(N_MODALITY_TYPES):
            mod_mask = (modality_expanded == mod_id) & valid_mask
            if not mod_mask.any():
                continue
            mod_correct = chan_correct[mod_mask].sum().item()
            mod_total = mod_mask.sum().item()
            if mod_total > 0:
                modality_stats[mod_id] = (mod_correct, mod_total)

        print(f"\nPer-Modality Accuracy:")
        modality_names = {v: k for k, v in MODALITY_TYPES.items()}
        for mod_id, (correct, total) in modality_stats.items():
            mod_name = modality_names.get(mod_id, f"MOD_{mod_id}")
            print(f"  {mod_name}: {correct / total:.4f} ({correct}/{total})")

    # Test with multiple steps to see if centroids get updated
    print(f"\n=== Training 10 steps ===")
    for step in range(10):
        optimizer.zero_grad()

        outputs = model(x, modality_ids=modality_ids)
        logits_flat = outputs["logits"].reshape(-1, n_classes)
        labels_flat = labels.reshape(-1)
        loss = loss_fn(logits_flat, labels_flat)

        # Update centroids
        with torch.no_grad():
            emb_flat = outputs["embedding"].reshape(-1, outputs["embedding"].shape[-1])
            model.clf.update_centroids(emb_flat, labels_flat)

        loss.backward()
        optimizer.step()

        preds = logits_flat.argmax(dim=-1)
        acc = (preds == labels_flat).float().mean()

        if step % 5 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, acc={acc.item():.4f}")

    print("\n=== Test PASSED ===")

if __name__ == "__main__":
    test_synthetic_training()
