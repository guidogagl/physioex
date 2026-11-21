"""Jones ECG quickstart: load a released checkpoint and run one forward pass.

Usage:
    python quickstart.py
"""
import torch

from physioex.models import load_from_pretrained

model = load_from_pretrained("jones-ecg", verbose=True)  # = jones-ecg-primary

# Whole night: 1200 x 30-s epochs of ECG, 7680 samples per epoch (256 Hz),
# values in [-1, 1] (per-recording R-peak scaling of the authors' pipeline).
x = torch.randn(1, 1200, 7680).tanh()
pred = model.predict_night({"ECG": x}, {"id": "quickstart"})
print("prediction shape:", tuple(pred.shape), "-> classes [W, N1, N2, N3, REM]")
