"""ProtoSleepNet quickstart: load a pretrained interpretable sleep stager.

    pip install physioex
    python quickstart.py

Full experiment/reproduction code for the paper lives in the dedicated repo:
    https://github.com/guidogagl/protosleepnet
Weights are on the Hugging Face Hub in 4rooms/physioex.
"""
import torch
from physioex.models import load_from_pretrained

# ProtoSleepTransformer (PST, L=21). Also: "protosleepnet-gagliardi" (PSN, L=20).
model = load_from_pretrained("protosleeptransformer-gagliardi", verbose=True)

# Input:  (batch, L, channels, T, F) STFT log-power spectrograms (T=29, F=129),
#         channels = EEG, EOG, EMG (3). Output: (batch, L, 5) AASM stage logits
#         (W=0, N1=1, N2=2, N3=3, REM=4).
x = torch.randn(2, 21, 3, 29, 129)
with torch.no_grad():
    logits = model(x)
print("output shape:", tuple(logits.shape))
