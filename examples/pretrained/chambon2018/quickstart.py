"""Chambon2018 quickstart: load a pretrained model in 3 lines.

Install:
    pip install physioex

Usage:
    python quickstart.py
"""
from physioex.models import load_from_pretrained

# Load the pretrained Chambon2018Net (Chambon et al. 2018) from HuggingFace
model = load_from_pretrained("chambon2018", verbose=True)

# The model is a standard PyTorch nn.Module in eval mode.
# Input:  (batch, L, channels, 3000) — L epochs of raw EEG at 100 Hz
# Output: (batch, 1, 5)              — AASM sleep stage logits for CENTRAL epoch
#         W=0, N1=1, N2=2, N3=3, REM=4
