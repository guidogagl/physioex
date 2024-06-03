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

from physioex.data import Shhs, SleepEDF, TimeDistributedModule
from physioex.explain.spectralgradients import (
    SpectralGradients,
    generate_frequency_bands,
    plot,
)
from physioex.explain.spectralgradients.viz import plot_class_spectrum
from physioex.models import load_pretrained_model
from physioex.train.networks import config as networks
from physioex.train.networks.utils.loss import config as losses
from physioex.train.networks.utils.target_transform import get_mid_label

# model parameters
model_name = "chambon2018"
sequence_length = 21

# dataset
picks = ["EEG"]
fold = 0

# dataloader
batch_size = 64
num_workers = os.cpu_count()
n_batches = 1000

# num of splitting bands
n_bands = 40

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
sg = SpectralGradients(model, n_bands=n_bands, mode="log")

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
