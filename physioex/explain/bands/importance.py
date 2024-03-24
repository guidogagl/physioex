import os
import uuid
from itertools import combinations
from typing import List

import numpy as np
import torch
from captum.attr import IntegratedGradients
from loguru import logger
from npy_append_array import NpyAppendArray
from scipy import signal
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


def calculate_combinations(elements):
    all_combinations = []

    for r in range(1, len(elements) + 1):
        for combination in combinations(elements, r):
            matrix = np.zeros(len(elements), dtype=int)

            for element in combination:
                index = elements.index(element)
                matrix[index] = 1

            all_combinations.append(matrix)

    return np.array(all_combinations, dtype=int)


def filtered_band_importance(
    bands: List[List[float]],
    model: torch.nn.Module,
    inputs: torch.Tensor,
    sampling_rate: int = 100,
):
    batch_size, seq_len, n_channels, n_samples = inputs.shape
    baseline = torch.from_numpy(inputs.copy())
    inputs = np.squeeze(inputs).reshape(batch_size, seq_len * n_samples)

    for band in bands:
        # filter bandstop - reject the frequencies specified in freq_band
        lowcut = band[0]
        highcut = band[1]
        order = 5
        nyq = 0.5 * sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high], btype="bandstop", output="sos")

        for index in range(batch_size):
            inputs[index] = signal.sosfilt(sos, inputs[index])

    inputs = inputs.reshape(batch_size, seq_len, n_channels, n_samples)

    inputs = torch.from_numpy(inputs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    band_score = model(inputs.to(device)).cpu().detach().numpy()

    return band_score


def band_importance(
    bands: List[List[float]],
    model: torch.nn.Module,
    dataloader: DataLoader,
    sampling_rate: int = 100,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SoftModel(model).to(device)

    combinations = calculate_combinations(list(range(len(bands))))

    exp = eXpDataset()

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            inputs, y_true = batch

            # compute the probas of the input data from the model
            probas = model(inputs.to(device)).cpu().detach().numpy()

            # debug: check if probas contains negative elements and log their index
            if (probas < 0).any():
                logger.debug(f"Negative probas found in batch {_}")

            inputs = inputs.cpu().detach().numpy()
            batch_size, seq_len, n_channels, n_samples = inputs.shape

            bands_importance = np.zeros((batch_size, len(bands), probas.shape[1]))

            D = len(np.where(combinations[:, 0] == 1)[0])

            for combination in combinations:
                b_indx = np.where(combination == 1)[0].astype(int)
                band_score = filtered_band_importance(
                    [bands[i] for i in b_indx],
                    model,
                    inputs.copy(),
                    sampling_rate,
                )

                for indx in b_indx:
                    bands_importance[:, indx, :] += band_score / D

            # debug log the min and max value of bands_importance
            # logger.debug(f"Min value of bands_importance: {bands_importance.min()}")
            # logger.debug(f"Max value of bands_importance: {bands_importance.max()}")
            # debug log the max value of probas
            # logger.debug(f"Max value of probas: {probas.max()}")
            # logger.debug(f"Min value of probas: {probas.min()}")

            for i in range(len(bands)):
                bands_importance[:, i, :] = probas - bands_importance[:, i, :]

            exp.add(
                y_true.numpy().astype(int),
                probas.astype(float),
                bands_importance.astype(float),
            )

    return exp


class SoftModel(torch.nn.Module):
    def __init__(self, model):
        super(SoftModel, self).__init__()
        self.model = model.eval()

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)


class eXpDataset(Dataset):
    def __init__(self):
        self.targets = []
        self.probas = []
        self.bands = []

    def add(
        self,
        targets: np.ndarray,
        probas: np.ndarray,
        bands: np.ndarray,
    ):
        self.targets.extend(targets)
        self.probas.extend(probas)
        self.bands.extend(bands)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (
            self.targets[index],
            self.probas[index],
            np.argmax(self.probas[index]),
            self.bands[index],
        )

    def get_class_importance(self, c):
        preds = np.argmax(self.probas, axis=1)

        c_indx = np.where(np.array(preds) == c)[0]

        ret = np.zeros((len(c_indx), len(self.bands[0])))

        for i, index in enumerate(c_indx):
            ret[i] = self.bands[index][:, c]

        return ret


class MeanImportance:
    def __init__(self, n_class, n_bands, input_shape):
        batch_size, seq_len, n_channels, n_samples = input_shape

        self.band_importance = np.zeros((batch_size, n_bands, n_class))
        self.time_importance = np.zeros(
            (batch_size, seq_len, n_channels, n_samples, n_bands, n_class)
        )

        self.W = np.sum([1 / i for i in range(1, n_bands + 1)])

    def update(self, bands, band_importance, time_importance):
        # compute the incremental mean of the band and time importance
        selected_bands = np.where(bands == 1)[0]
        weight = (1 / len(selected_bands)) / self.W

        for band in selected_bands:
            self.band_importance[:, band, :] += (
                band_importance - self.band_importance[:, band, :]
            ) * weight
            self.time_importance[:, :, :, :, band, :] += (
                time_importance - self.time_importance[:, :, :, :, band, :]
            ) * weight

    def get(self):
        return self.band_importance, self.time_importance
