"""Residual sequence encoder wrapper with deep supervision.

Adds a skip connection around the sequence encoder and supervises
the epoch encoder output with the same shared classifier.

Architecture:
    h = epoch_encoder(x)
    epoch_logits = classifier(h)          ← auxiliary CE loss
    z = h + sequence_encoder(h)           ← residual skip
    logits = classifier(z)                ← main CE loss

The sequence encoder is zero-initialized so that z ≈ h at training start,
letting the classifier train on epoch embeddings first.
"""
import torch
import torch.nn as nn

from physioex.train.trainer import Trainer
from physioex.train.metrics import accuracy_score


# ── Zero initialization ─────────────────────────────────────────────


def zero_init_transformer(encoder: nn.TransformerEncoder):
    """Zero-initialize TransformerEncoder so output ≈ 0 at start."""
    for layer in encoder.layers:
        nn.init.constant_(layer.linear1.weight, 0.0)
        nn.init.constant_(layer.linear1.bias, 0.0)
        nn.init.constant_(layer.linear2.weight, 0.0)
        nn.init.constant_(layer.linear2.bias, 0.0)
        nn.init.constant_(layer.self_attn.out_proj.weight, 0.0)
        nn.init.constant_(layer.self_attn.out_proj.bias, 0.0)
        nn.init.normal_(layer.self_attn.in_proj_weight, mean=0.0, std=0.01)
        if layer.self_attn.in_proj_bias is not None:
            nn.init.constant_(layer.self_attn.in_proj_bias, 0.0)
    return encoder


def zero_init_gru(gru: nn.GRU):
    """Zero-initialize GRU so output ≈ 0 at start."""
    for name, param in gru.named_parameters():
        nn.init.constant_(param, 0.0)
    return gru


# ── ResidualSequenceWrapper ──────────────────────────────────────────


class ResidualSequenceWrapper(nn.Module):
    """Wraps epoch_encoder + sequence_encoder + classifier with residual skip.

    The sequence encoder is zero-initialized. The classifier is shared
    between epoch-level and sequence-level predictions.

    Args:
        epoch_encoder: maps (N, C, T, F) → (N, d_model)
        sequence_encoder: maps (B, L, d_model) → (B, L, d_model)
        classifier: maps (B*L, d_model) → (B*L, n_classes) — final output
        epoch_classifier: maps (B*L, d_model) → (B*L, n_classes) — epoch-level auxiliary
    """

    def __init__(self, epoch_encoder, sequence_encoder, classifier, epoch_classifier):
        super().__init__()
        self.epoch_encoder = epoch_encoder
        self.sequence_encoder = sequence_encoder
        self.classifier = classifier
        self.epoch_classifier = epoch_classifier

        self._epoch_logits = None

    def get_metrics(self):
        return {"epoch_logits": self._epoch_logits}

    def forward(self, x):
        """
        Args:
            x: (B, L, C, T, F)
        Returns:
            (B, L, n_classes) — final logits
        """
        B, L, C, T, F_dim = x.shape

        # Epoch encoding
        h = self.epoch_encoder(x.reshape(B * L, C, T, F_dim))
        h = h.reshape(B, L, -1)  # (B, L, d_model)

        # Epoch-level classification (auxiliary, separate classifier)
        d = h.shape[-1]
        self._epoch_logits = self.epoch_classifier(h.reshape(B * L, d)).reshape(B, L, -1)

        # Residual sequence encoding
        if isinstance(self.sequence_encoder, nn.GRU):
            seq_out, _ = self.sequence_encoder(h)
        else:
            seq_out = self.sequence_encoder(h)
        z = h + seq_out  # residual skip

        # Final classification (same classifier)
        logits = self.classifier(z.reshape(B * L, d)).reshape(B, L, -1)

        return logits


# ── ResidualTrainer ──────────────────────────────────────────────────


def _residual_step(model, batch, loss_fn, device, double_loss=True):
    """Shared step logic for residual trainers."""
    if isinstance(batch, dict) and "signals" in batch:
        from physioex.data.collate import stack_channels
        inputs = stack_channels(batch).to(device)
        targets = batch["labels"].to(device)
    elif isinstance(batch, dict) and "embeddings" in batch:
        inputs = batch["embeddings"].to(device)
        targets = batch["labels"].to(device)
    else:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

    with torch.autocast(device.type if "cuda" in device.type else "cpu"):
        outputs = model(inputs)

    B, L, n_classes = outputs.shape
    outputs_flat = outputs.reshape(-1, n_classes)
    targets_flat = targets.reshape(-1)

    loss = loss_fn(outputs_flat, targets_flat)

    if double_loss:
        metrics = model.get_metrics()
        epoch_logits = metrics["epoch_logits"]
        epoch_flat = epoch_logits.reshape(-1, n_classes)
        loss = loss + loss_fn(epoch_flat, targets_flat)

    acc = accuracy_score(
        outputs_flat, targets_flat,
        ignore_index=getattr(loss_fn, "ignore_index", None),
    )

    return loss, acc


class ResidualTrainer(Trainer):
    """Trainer with epoch-level auxiliary loss (double loss)."""

    @staticmethod
    def _step(model, batch, loss_fn, device):
        return _residual_step(model, batch, loss_fn, device, double_loss=True)


class ResidualTrainerSingleLoss(Trainer):
    """Trainer with only final output loss (single loss)."""

    @staticmethod
    def _step(model, batch, loss_fn, device):
        return _residual_step(model, batch, loss_fn, device, double_loss=False)
