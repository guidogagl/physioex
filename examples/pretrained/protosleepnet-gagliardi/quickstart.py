"""ProtoSleepTransformer-Gagliardi quickstart: load a pretrained model in 3 lines.

Install:
    pip install physioex

Usage:
    python quickstart.py
"""
from physioex.models import load_from_pretrained

# Load the pretrained ProtoSleepTransformer from HuggingFace
model = load_from_pretrained("prosleepnet-gagliardi", verbose=True)

# The model is a standard PyTorch nn.Module in eval mode.
# Input:  (batch, L, channels, T, F) — STFT spectrograms (T=29, F=129)
#         channels: EEG, EOG, EMG (3 channels)
# Output: (batch, L, 5)              — AASM sleep stage logits per epoch
#         W=0, N1=1, N2=2, N3=3, REM=4
