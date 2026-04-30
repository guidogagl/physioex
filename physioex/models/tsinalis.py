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

The input is the current epoch flanked by 2 preceding and 2 succeeding
epochs, concatenated into a single 15000-sample signal.  The prediction
is for the **central** epoch only.
"""

import torch
from torch import nn


class TsinalisCNN(nn.Module):
    """CNN for single-channel sleep staging (Tsinalis et al. 2016).

    Input:  (B, 1, n_times) — single EEG channel, raw time domain
            n_times = 15000 (5 epochs × 3000 samples at 100Hz)
    Output: (B, n_classes)

    Args:
        n_classes: Number of output classes (default 5: W, N1, N2, N3, REM).
        n_times:   Number of time samples (default 15000 = 5 × 30s @ 100Hz).
        sfreq:     Sampling frequency in Hz (default 100).
        n_filters_c1: Number of filters in first conv layer (default 20).
        n_filters_c2: Number of filters in second conv layer (default 400).
        fc_size:   Hidden units in FC layers (default 500).
        dropout:   Dropout probability (default 0.5).
    """

    def __init__(
        self,
        n_classes: int = 5,
        n_times: int = 15000,
        sfreq: int = 100,
        n_filters_c1: int = 20,
        n_filters_c2: int = 400,
        fc_size: int = 500,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_times = n_times

        # C1: Long temporal filters (2 seconds = sfreq*2 samples)
        c1_kernel = sfreq * 2  # 200 for 100Hz
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, n_filters_c1, kernel_size=c1_kernel, stride=1, bias=False),
            nn.BatchNorm1d(n_filters_c1),
            nn.ReLU(inplace=True),
        )

        # P1: Max-pool with large kernel to reduce temporal resolution
        self.pool1 = nn.MaxPool1d(kernel_size=20, stride=10)

        # Compute size after C1 + P1
        c1_out = n_times - c1_kernel + 1  # 15000 - 200 + 1 = 14801
        p1_out = (c1_out - 20) // 10 + 1  # (14801 - 20) // 10 + 1 = 1479

        # S1: Stack — reshape from (B, n_filters_c1, T) to (B, 1, n_filters_c1, T)
        # Treats C1 filter outputs as a "spectral" dimension for 2D conv

        # C2: Cross-filter convolution (2D)
        # kernel spans ALL C1 filters × 30 time steps
        c2_kernel = (n_filters_c1, 30)
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filters_c2, kernel_size=c2_kernel, stride=1, bias=False),
            nn.BatchNorm2d(n_filters_c2),
            nn.ReLU(inplace=True),
        )

        # P2: Temporal pooling on the remaining time dimension
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
            x: (B, 1, n_times) or (B, n_times) raw EEG signal.

        Returns:
            (B, n_classes) logits.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)

        # C1 + P1: temporal filter bank + pooling
        x = self.conv1(x)
        x = self.pool1(x)

        # Stack: treat as 2D image (filter × time)
        x = x.unsqueeze(1)

        # C2 + P2: cross-filter combination + pooling
        x = self.conv2(x)
        x = self.pool2(x)

        # Flatten + classify
        x = x.flatten(1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = TsinalisCNN(n_classes=5, n_times=15000, sfreq=100)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TsinalisCNN: {n_params:,} parameters")

    # Input: 5 concatenated 30s epochs at 100Hz
    x = torch.randn(4, 1, 15000)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    # Verify gradient flow
    y.sum().backward()
    print("Gradient flow: OK")
