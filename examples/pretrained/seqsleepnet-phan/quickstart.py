"""SeqSleepNet-Huy quickstart: load a pretrained model in 3 lines.

Install:
    pip install physioex

Usage:
    python quickstart.py
"""
from physioex.models import load_from_pretrained

# Load the pretrained SeqSleepNet (Phan et al. 2019) from HuggingFace
# Weights + config are downloaded automatically and cached locally.
model = load_from_pretrained("seqsleepnet-phan", verbose=True)

# The model is a standard PyTorch nn.Module in eval mode.
# Input:  (batch, sequence_length, channels, time, freq) — spectrograms
# Output: (batch, sequence_length, 5)                    — AASM sleep stage logits
#         W=0, N1=1, N2=2, N3=3, REM=4

# Example: evaluate on your own Sleep-EDF data
# from physioex.data.datasets import get_dataset
# from physioex.train.trainer import Trainer
#
# dataset = get_dataset("sleepedf")(
#     root="/path/to/physionet-sleep-data",
#     channels=["EEG"],
#     pipelines="seqsleepnet",
#     sequence_length=20,
# )
# results = Trainer.voting_evaluate(model=model, dataset=dataset, L=20, fold=0)
# print(results)
