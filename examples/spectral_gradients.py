import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import IntegratedGradients
from loguru import logger
from pytorch_lightning import seed_everything
from scipy import signal
from tqdm import tqdm

seed_everything(42)

import os

from captum.attr import IntegratedGradients

from physioex.data import Shhs, SleepEDF, TimeDistributedModule
from physioex.explain.spectralgradients import (SpectralGradients,
                                                generate_frequency_bands, plot)
from physioex.explain.spectralgradients.viz import plot_class_spectrum
from physioex.models import load_pretrained_model
from physioex.train.networks import config as networks
from physioex.train.networks.utils.loss import config as losses
from physioex.train.networks.utils.target_transform import get_mid_label


def smooth(x, kernel_size=3):
    return torch.nn.AvgPool1d(kernel_size=kernel_size, stride=int(kernel_size / 2))(x)


def process_explanations(explanations, kernel_size=300):
    explanations = explanations.squeeze()

    batch_size, seq_len, num_samples, n_bands = explanations.size()
    explanations = explanations[..., : int(n_bands / 2) + 1]
    # consider only the mid epoch of the sequence (the one that is more relevant)
    explanations = explanations[:, int((seq_len - 1) / 2)]

    explanations = torch.permute(explanations, [0, 2, 1])

    # smooth the num_samples dimension
    explanations = smooth(explanations, kernel_size) * kernel_size

    return explanations


# model parameters
model_name = "chambon2018"
sequence_length = 21

# dataset
picks = ["EEG"]
fold = 0

# dataloader
batch_size = 6
num_workers = os.cpu_count()
n_batches = 2

# num of splitting bands
n_bands = 20

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset and model
model = networks[model_name]

dataset = Shhs(
    picks=picks,
    sequence_length=sequence_length,
    # target_transform=model["target_transform"],
    target_transform=get_mid_label,
    preprocessing=model["input_transform"],
)

dataset.split(fold=fold)

dataset = TimeDistributedModule(
    dataset=dataset, batch_size=batch_size, fold=fold, num_workers=num_workers
)

model = load_pretrained_model(
    name=model_name,
    in_channels=len(picks),
    sequence_length=sequence_length,
    softmax=True,
).eval()


class MidModel(torch.nn.Module):
    def __init__(self, model):
        super(MidModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[:, int((x.shape[1] - 1) / 2)]


# model = MidModel(model)


# setup the explanations algorithms
sg = SpectralGradients(model, n_bands=n_bands)
ig = IntegratedGradients(model)

### compute the spectral gradients over the batches
datalaoder = dataset.train_dataloader()

explanations = []
explanations_ig = []
for i, (x, y) in enumerate(tqdm(datalaoder, total=n_batches)):
    if i == n_batches:
        break

    explanations.extend(
        sg.attribute(x.to(device), target=y.to(device), n_steps=5).detach().cpu()
    )
    explanations_ig.extend(
        ig.attribute(x.to(device), target=y.to(device), n_steps=5).detach().cpu()
    )

explanations = torch.stack(explanations)
explanations_ig = torch.stack(explanations_ig)

approximation_error = torch.abs(explanations.sum(dim=-1) - explanations_ig)
micro_approximation_error = approximation_error.reshape(-1).mean()
macro_approximation_error = approximation_error.mean(dim=-1).reshape(-1).mean()

explanations = process_explanations(explanations)


# batch, bands, samples

# compute the variance over the bands dimension

variances = explanations.var(dim=1).reshape(-1).mean()  # batch, samples

# write the variances to a file

with open(f"results/{model_name}/explanations_report.txt", "w") as f:
    f.write(f"Mean variance: {variances} \n")


with open(f"results/{model_name}/explanations_report.txt", "a") as f:
    f.write(f"Macro approximation error: {macro_approximation_error}\n")
    f.write(f"Micro approximation error: {micro_approximation_error}\n")

# compute the frequency of elements of different sign over the bands dimension

explanations_sign = (
    explanations.sign()
)  # compute the frequency of discording signs in the same bands

num_discording_macro = 0
num_discording_micro = 0
total_elements_macro = 0
total_elements_micro = 0

print(explanations_sign.size())

for sample in explanations_sign:
    num_pos = (sample == 1).sum(dim=0)
    num_neg = (sample == -1).sum(dim=0)

    discording = (num_pos >= 1) & (num_neg >= 1)

    num_discording_macro += discording.sum().item()
    num_discording_micro += discording.sum(dim=0).item()

    total_elements_macro += sample.size(0)
    total_elements_micro += sample.numel()

frequency_discording_macro = num_discording_macro / total_elements_macro
frequency_discording_micro = num_discording_micro / total_elements_micro

with open(f"results/{model_name}/explanations_report.txt", "a") as f:
    f.write(f"Macro frequency of discording signs: {frequency_discording_macro}\n")
    f.write(f"Micro frequency of discording signs: {frequency_discording_micro}\n")


"""
plot_class_spectrum(
    dataloader=dataset.train_dataloader(),
    model=model,
    sg=sg,
    n_classes=5,
    n_batches=n_batches,
    classes={0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"},
    relevant_bands={
        "Wake": ["Beta", "Gamma"],
        "N1": ["Alpha", "Theta"],
        "N2": ["Theta", "Delta"],
        "N3": ["Delta"],
        "REM": ["Alpha", "Beta"],
    },
    band_colors={
        "Delta": "blue",
        "Theta": "green",
        "Alpha": "red",
        "Sigma": "purple",
        "Beta": "orange",
        "Gamma": "brown",
    },
    eeg_bands={
        "Delta": (0, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Sigma": (12, 16),
        "Beta": (16, 30),
        "Gamma": (30, 50),
    },
    filename=f"results/{model_name}/class_spectrum.png",
)
"""
