"""Tsinalis-2016 quickstart: load a pretrained model in 3 lines.

Install:
    pip install physioex

Usage:
    python quickstart.py
"""
from physioex.models import load_from_pretrained

# Load the pretrained Tsinalis CNN (Tsinalis et al. 2016) from HuggingFace
model = load_from_pretrained("tsinalis-2016", verbose=True)

# The model is a standard PyTorch nn.Module in eval mode.
# Input:  (batch, 1, 15000) — 5 concatenated 30s epochs at 100 Hz
# Output: (batch, 5)        — AASM sleep stage logits for the CENTRAL epoch
#         W=0, N1=1, N2=2, N3=3, REM=4
