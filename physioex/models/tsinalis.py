"""Tsinalis et al. (2016) CNN for sleep staging from raw EEG.

Reference: "Automatic Sleep Stage Scoring with Single-Channel EEG
Using Convolutional Neural Networks" (arXiv:1610.01683).

Architecture (Table 3 of the paper):
  Input: 5 concatenated 30s epochs (15000 samples at 100Hz)
  C1: 20 filters, kernel=200, stride=1, ReLU
  P1: max-pool kernel=20, stride=10
  S1: stack (reshape 20 channels into 2D image)
  C2: 400 filters, kernel=(20,30), stride=1, ReLU
  P2: max-pool kernel=(1,10), stride=(1,2)
  F1: 500 units, ReLU, dropout
  F2: 500 units, ReLU, dropout
  Output: 5-class softmax

PhysioEx integration:
  Input from dataset: (B, 5, 1, 3000) — 5 epochs × 1 channel × 3000 samples
  Internally reshaped to: (B, 1, 15000) — concatenated signal
  Output: (B, 1, n_classes) — prediction for the CENTRAL epoch only

This is compatible with the standard Trainer when paired with a
target_transform that extracts the central epoch label.
"""

import torch
from torch import nn


class TsinalisCNN(nn.Module):
    """CNN for single-channel sleep staging (Tsinalis et al. 2016).

    Accepts the standard PhysioEx sequence format ``(B, L, C, T)`` with
    ``L=5, C=1, T=3000``, concatenates the 5 epochs into a single
    15000-sample signal, and predicts the sleep stage of the **central**
    (3rd) epoch.

    Args:
        n_classes: Number of output classes (default 5: W, N1, N2, N3, REM).
        sfreq:     Sampling frequency in Hz (default 100).
        n_filters_c1: Number of filters in first conv layer (default 20).
        n_filters_c2: Number of filters in second conv layer (default 400).
        fc_size:   Hidden units in FC layers (default 500).
        dropout:   Dropout probability (default 0.5).
    """

    def __init__(
        self,
        n_classes: int = 5,
        sfreq: int = 100,
        n_filters_c1: int = 20,
        n_filters_c2: int = 400,
        fc_size: int = 500,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_classes = n_classes

        # Input length: 5 epochs × 30s × sfreq
        n_times = 5 * 30 * sfreq  # 15000

        # C1: Long temporal filters (2 seconds = sfreq*2 samples)
        c1_kernel = sfreq * 2  # 200 for 100Hz
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, n_filters_c1, kernel_size=c1_kernel, stride=1),
            nn.ReLU(inplace=True),
        )

        # P1: Max-pool
        self.pool1 = nn.MaxPool1d(kernel_size=20, stride=10)

        # Compute size after C1 + P1
        c1_out = n_times - c1_kernel + 1  # 15000 - 200 + 1 = 14801
        p1_out = (c1_out - 20) // 10 + 1  # (14801 - 20) // 10 + 1 = 1479

        # C2: Cross-filter convolution (2D)
        c2_kernel = (n_filters_c1, 30)
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filters_c2, kernel_size=c2_kernel, stride=1),
            nn.ReLU(inplace=True),
        )

        # P2: Temporal pooling
        c2_time = p1_out - c2_kernel[1] + 1  # 1479 - 30 + 1 = 1450
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 10), stride=(1, 2))
        p2_time = (c2_time - 10) // 2 + 1  # (1450 - 10) // 2 + 1 = 721

        flat_size = n_filters_c2 * 1 * p2_time

        # FC layers (paper: F1=500, F2=500)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flat_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc_size, n_classes),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 5, 1, 3000) from PhysioEx dataset (5-epoch sequence),
               or (B, 1, 15000) pre-concatenated signal.

        Returns:
            (B, 1, n_classes) logits for the central epoch.
        """
        if x.dim() == 4:
            # (B, L=5, C=1, T=3000) -> (B, 1, 15000)
            B = x.shape[0]
            x = x.reshape(B, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # (B, T) -> (B, 1, T)

        # C1 + P1
        x = self.conv1(x)
        x = self.pool1(x)

        # Stack: treat as 2D image (filter × time)
        x = x.unsqueeze(1)

        # C2 + P2
        x = self.conv2(x)
        x = self.pool2(x)

        # Flatten + classify
        x = x.flatten(1)
        x = self.classifier(x)  # (B, n_classes)

        # Return (B, 1, n_classes) for Trainer compatibility
        return x.unsqueeze(1)


if __name__ == "__main__":
    model = TsinalisCNN(n_classes=5, sfreq=100)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TsinalisCNN: {n_params:,} parameters")

    # Test with PhysioEx format: (B, 5, 1, 3000)
    x = torch.randn(4, 5, 1, 3000)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (4, 1, 5)

    # Verify gradient flow
    y.sum().backward()
    print("Gradient flow: OK")
