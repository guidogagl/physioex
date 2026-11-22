"""SleepPPG-Net (replica) quickstart: load the weights and run one forward pass.

Usage:
    python quickstart.py
"""
import torch

from physioex.models import load_from_pretrained

model = load_from_pretrained("sleepppgnet-replica", verbose=True)

# Whole night: 1200 x 30-s epochs of PPG, 1024 samples per epoch (34.13 Hz).
x = torch.randn(1, 1200, 1024)
logits = model(x)  # (1, 1200, 4)
print("logits shape:", tuple(logits.shape), "-> classes [W, Light, Deep, REM]")
