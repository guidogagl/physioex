from itertools import combinations
from typing import List

import numpy as np
import torch
from captum.attr import IntegratedGradients
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
    compute_time: bool = True,
):
    batch_size, seq_len, n_channels, n_samples = inputs.shape
    baseline = torch.from_numpy(inputs.copy())

    inputs = inputs.reshape(batch_size, seq_len * n_samples)

    for band in bands:
        # filter bandstop - reject the frequencies specified in freq_band
        lowcut = band[0]
        highcut = band[1]
        order = 4
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

    batch_size, n_class = band_score.shape

    time_importance = np.zeros((batch_size, seq_len, n_channels, n_samples, n_class))

    if compute_time:
        ig = IntegratedGradients(model)

        for c in range(n_class):
            time_importance[:, :, :, :, c] = (
                ig.attribute(inputs.to(device), baseline.to(device), target=c)
                .cpu()
                .numpy()
            )

    return band_score, time_importance


def band_importance(
    bands: List[List[float]],
    model: torch.nn.Module,
    dataloader: DataLoader,
    sampling_rate: int = 100,
    compute_time: bool = True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SoftModel(model).to(device)

    combinations = calculate_combinations(list(range(len(bands))))

    exp = eXpDataset()

    for _, batch in enumerate(tqdm(dataloader)):
        inputs, y_true = batch

        # compute the probas of the input data from the model

        probas = model(inputs.to(device)).cpu().detach().numpy()
        inputs = inputs.cpu().detach().numpy()
        batch_size, n_class = probas.shape

        mean = MeanImportance(n_class, len(bands), inputs.shape)

        for combination in combinations:
            band_score, time_importance = filtered_band_importance(
                [bands[i] for i in range(len(bands)) if combination[i] == 1],
                model,
                inputs.copy(),
                sampling_rate,
                compute_time,
            )

            # band_importance = probas - band_score
            # mean.update(combination, band_importance, time_importance)

            mean.update(combination, band_score, time_importance)

        mean_band_score, times_importance = mean.get()

        bands_importance = mean_band_score

        for band in range(len(bands)):
            bands_importance[:, band, :] = probas - bands_importance[:, band, :]

        exp.add(
            inputs.astype(float),
            y_true.numpy().astype(int),
            probas.astype(float),
            np.argmax(probas, axis=1).astype(int),
            bands_importance.astype(float),
            times_importance.astype(float),
        )

        if _ >= 2:
            break
    return exp


class SoftModel(torch.nn.Module):
    def __init__(self, model):
        super(SoftModel, self).__init__()
        self.model = model.eval()

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)


class eXpDataset(Dataset):
    def __init__(self):
        self.inputs = []
        self.targets = []
        self.probas = []
        self.preds = []
        self.bands = []
        self.time = []

    def add(self, inputs, targets, probas, preds, bands, time):
        self.inputs.extend(inputs)
        self.targets.extend(targets)
        self.probas.extend(probas)
        self.preds.extend(preds)
        self.bands.extend(bands)
        self.time.extend(time)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return (
            self.inputs[index],
            self.targets[index],
            self.probas[index],
            self.preds[index],
            self.bands[index],
            self.time[index],
        )

    def get_class_importance(self, c):
        c_indx = np.where(np.array(self.preds) == c)[0]

        ret = np.zeros((len(c_indx), len(self.bands[0])))

        for i, index in enumerate(c_indx):
            ret[i] = self.bands[index][:, c]

        return ret

    def get_bands_importance(self, b):
        ret = np.zeros((len(self.inputs), len(self.probas[0])))

        for index in range(len(self.inputs)):
            ret[index] = self.bands[index][b, :]

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
