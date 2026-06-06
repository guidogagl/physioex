"""Channel dropout wrapper for training with random channel zeroing.

Wraps any sleep staging model to randomly zero input channels during training,
forcing the model to be robust to missing channels at inference time.
"""
import torch
import torch.nn as nn


class ChannelDropoutWrapper(nn.Module):
    """Wraps a model to randomly zero input channels during training.

    Each channel is independently zeroed with probability ``p`` at every epoch
    in the sequence. At least one channel is always kept (rescue mechanism).

    Only active during training (``self.training=True``); at eval time the
    input passes through unchanged.

    Args:
        model: The inner model (e.g. SleepTransformer, SeqSleepNet).
        p: Per-channel dropout probability. Default 0.5.
    """

    def __init__(self, model, p=0.5):
        super().__init__()
        self.model = model
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            B, L, C, T, F = x.shape
            keep = torch.rand(B, L, C, 1, 1, device=x.device) >= self.p
            # Ensure at least 1 channel kept per (batch, epoch)
            all_dropped = ~keep.any(dim=2, keepdim=True)
            if all_dropped.any():
                rescue = torch.zeros_like(keep)
                rescue[:, :, torch.randint(C, (1,)).item()] = True
                keep = keep | (all_dropped & rescue)
            x = x * keep.float()
        return self.model(x)
