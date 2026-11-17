"""Training step of the psg_gen models on the physioex :class:`Trainer` (SPEC section 2).

The step stacks the channels, takes the centre epoch of channel 0, runs the model and calls
``model.loss(out, xc, site)``. The night of each window (``batch["subject"][i]["id"]``, plus the
dataset index when present) is put in ``out["night"]`` as an integer id for the night adversary
of P2 and P3.
"""
from __future__ import annotations

import zlib

import torch

from physioex.models.msae_factor import sites_of
from physioex.models.msvq import _autocast
from physioex.train.trainer import Trainer


def nights_of(batch, n_slots: int = 4096):
    """(B,) long night ids hashed from the subject records, or None."""
    subs = batch.get("subject")
    if not subs:
        return None
    ids = []
    for s in subs:
        key = f"{s.get('dataset_idx', '')}/{s.get('id', '')}" if isinstance(s, dict) else str(s)
        ids.append(zlib.crc32(key.encode()) % n_slots)
    return torch.tensor(ids, dtype=torch.long)


class PSGGenTrainer(Trainer):
    """``acc`` = what ``model.loss`` returns (R^2 of the band log power for the base model)."""

    @staticmethod
    def _step(model, batch, loss_fn, device):
        from physioex.data.collate import stack_channels

        x = stack_channels(batch).to(device)
        xc = x[:, x.shape[1] // 2, 0].float()
        site = sites_of(batch, model)
        site = site.to(device) if site is not None else None
        with _autocast(device):
            out = model(x)
        night = nights_of(batch, getattr(model, "n_night_slots", 4096))
        out["night"] = night.to(device) if night is not None else None
        return model.loss(out, xc, site)
