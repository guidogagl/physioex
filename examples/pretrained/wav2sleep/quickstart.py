"""Wav2Sleep quickstart: load the pretrained model and run one forward pass.

Usage:
    python quickstart.py
"""
import torch

from physioex.models import load_from_pretrained

model = load_from_pretrained("wav2sleep", verbose=True)

# Whole night, one signal each: ECG/PPG at 1024 samples per 30-s epoch,
# ABD/THX at 256; 1200 epochs = 10 h.  An absent signal is all -inf.
signals = {
    "ECG": torch.randn(1, 1200, 1024),
    "PPG": torch.full((1, 1200, 1024), float("-inf")),  # absent
}
logits = model(signals)  # (1, 1200, 4)
print("logits shape:", tuple(logits.shape), "-> classes [W, Light, Deep, REM]")
