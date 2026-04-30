"""SleepTransformer-Phan quickstart: load a pretrained model in 3 lines.

Install:
    pip install physioex

Usage:
    python quickstart.py
"""
from physioex.models import load_from_pretrained

# Load the pretrained SleepTransformer (Phan et al. 2022) from HuggingFace
model = load_from_pretrained("sleeptransformer-phan", verbose=True)

# The model is a standard PyTorch nn.Module in eval mode.
# Input:  (batch, L, channels, T, F) — STFT spectrograms (T=29, F=129)
# Output: (batch, L, 5)              — AASM sleep stage logits per epoch
#         W=0, N1=1, N2=2, N3=3, REM=4
