"""DataLoader collate function and channel-stacking helper for dict-returning datasets."""
from typing import Dict, List, Any
import torch


def dict_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of dict-returning dataset items.

    Stacks tensor fields across the batch dimension. Preserves dict/list metadata
    as lists-of-dicts (not dicts-of-lists) so individual sample metadata stays
    accessible.

    Expects each item to have at least:

    - ``signals``: dict[str, Tensor]
    - ``channel_order``: list[str]
    - ``labels``: Tensor

    Optional fields: ``epoch_indices``, ``channel_info``, ``subject``, ``recording_length``.

    Note:
        Channel names are expected to be in MODALITY_INDEX format (e.g., "EEG_0", "EOG_1").
        The collate sorts channels by (modality, index) for deterministic ordering.
    """
    if not batch:
        raise ValueError("Empty batch")

    out: Dict[str, Any] = {}

    # signals: collect ALL unique channel keys across the entire batch
    all_channels = set()
    for b in batch:
        all_channels.update(b["signals"].keys())

    # Sort channels by (modality_type, index) for deterministic ordering
    # Keys are in format "MODALITY_INDEX" (e.g., "EEG_0", "EOG_1")
    # Sort by modality type index (EEG=0, EOG=1, EMG=2, ...), then by numeric index
    from physioex.data.modality import MODALITY_TYPES

    def sort_key(ch: str) -> tuple:
        parts = ch.rsplit("_", 1)  # Split only on last underscore
        if len(parts) == 2:
            modality_name = parts[0]
            # Get modality type index, default to OTHER (13) if not found
            modality_idx = MODALITY_TYPES.get(modality_name, 13)
            try:
                return (modality_idx, int(parts[1]))
            except ValueError:
                return (modality_idx, 0)
        return (13, 0)  # OTHER for malformed names

    ref_order = sorted(all_channels, key=sort_key)
    out["channel_order"] = ref_order

    # zero-fill missing keys in each batch element
    for b in batch:
        for k in ref_order:
            if k not in b["signals"]:
                # determine the shape of the missing signal (use the shape of an existing signal)
                example_signal = next(iter(b["signals"].values()))
                b["signals"][k] = torch.zeros_like(example_signal)
    
    # Stack signals according to the sorted channel_order
    out["signals"] = {
        k: torch.stack([b["signals"][k] for b in batch]) for k in ref_order
    }
    
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
    
    try:
        # signals[name] after collate has shape (B, L, ...); stack adds a new dim 2 -> (B, L, C, ...)
        return torch.stack([batch["signals"][name] for name in order], dim=2)
    except KeyError as e:
        print( "Error stacking channels: missing channel in signals:", e )
        print("Available channels in signals:", list(batch["signals"].keys()))
        print("Requested channel order:", order)
        exit(1)


def is_dict_batch(batch: Any) -> bool:
    """Return True if `batch` looks like a dict-style PhysioEx batch."""
    return isinstance(batch, dict) and "signals" in batch and "channel_order" in batch
