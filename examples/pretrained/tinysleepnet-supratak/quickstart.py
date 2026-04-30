"""TinySleepNet-Supratak quickstart: load a pretrained model in 3 lines.

Install:
    pip install physioex

Usage:
    python quickstart.py
"""
from physioex.models import load_from_pretrained

# Load the pretrained TinySleepNet (Supratak & Guo, EMBC 2020) from HuggingFace
model = load_from_pretrained("tinysleepnet-supratak", verbose=True)

# The model is a standard PyTorch nn.Module in eval mode.
# Input:  (batch, sequence_length, channels, time_samples) — raw EEG at 100 Hz
# Output: (batch, sequence_length, 5)                      — AASM sleep stage logits
#         W=0, N1=1, N2=2, N3=3, REM=4
