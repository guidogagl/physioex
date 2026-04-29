"""DataLoader collate function and channel-stacking helper for dict-returning datasets."""
from typing import Dict, List, Any
import torch


def dict_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of dict-returning dataset items.

    Stacks tensor fields across the batch dimension. Preserves dict/list metadata
    as lists-of-dicts (not dicts-of-lists) so individual sample metadata stays
    accessible.

    Expects each item to have at least:
      - 'signals': dict[str, Tensor]
      - 'channel_order': list[str]
      - 'labels': Tensor
    Optional fields: 'epoch_indices', 'channel_info', 'subject', 'recording_length'.
    """
    if not batch:
        raise ValueError("Empty batch")

    out: Dict[str, Any] = {}

    # signals: stack per channel key
    signal_keys = list(batch[0]["signals"].keys())
    out["signals"] = {
        k: torch.stack([b["signals"][k] for b in batch]) for k in signal_keys
    }

    # channel_order: must be identical across batch (assert)
    ref_order = batch[0]["channel_order"]
    for b in batch[1:]:
        if b["channel_order"] != ref_order:
            raise ValueError(
                f"channel_order mismatch in batch: {b['channel_order']} vs {ref_order}. "
                "All samples in a batch must share the same channel_order."
            )
    out["channel_order"] = list(ref_order)

    # labels
    out["labels"] = torch.stack([b["labels"] for b in batch])

    # optional stacked fields
    if "epoch_indices" in batch[0]:
        out["epoch_indices"] = torch.stack([b["epoch_indices"] for b in batch])

    # metadata kept as list-of-dicts
    for k in ("channel_info", "subject"):
        if k in batch[0]:
            out[k] = [b[k] for b in batch]

    if "recording_length" in batch[0]:
        out["recording_length"] = torch.tensor(
            [b["recording_length"] for b in batch], dtype=torch.long
        )

    if "events" in batch[0]:
        out["events"] = [b["events"] for b in batch]

    return out


def stack_channels(batch: Dict[str, Any]) -> torch.Tensor:
    """Convert a collated dict batch to a (B, L, C, ...) tensor for model input.

    Uses the user-specified channel_order to stack signals consistently.
    """
    order = batch["channel_order"]
    if not order:
        raise ValueError("channel_order is empty")
    # signals[name] after collate has shape (B, L, ...); stack adds a new dim 2 -> (B, L, C, ...)
    return torch.stack([batch["signals"][name] for name in order], dim=2)


def is_dict_batch(batch: Any) -> bool:
    """Return True if `batch` looks like a dict-style PhysioEx batch."""
    return isinstance(batch, dict) and "signals" in batch and "channel_order" in batch
