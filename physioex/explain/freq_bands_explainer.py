import csv
import itertools as it
import os
import re
from os.path import exists
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
from captum.attr import IntegratedGradients
from joblib import Parallel, delayed
from loguru import logger
from pytorch_lightning import LightningModule
from scipy import signal
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from physioex.data import TimeDistributedModule, datasets
from physioex.explain.base import PhysioExplainer
from physioex.train.networks import config
from physioex.train.networks.utils.loss import config as loss_config


def _compute_cross_band_importance(
    bands: List[List[float]],
    model: torch.nn.Module,
    dataloader: DataLoader,
    model_device: torch.device,
    sampling_rate: int = 100,
):

    for i in range(len(bands)):
        assert len(bands[i]) == 2

    y_pred = []
    y_true = []
    band_importance = []
    time_importance = []

    for batch in dataloader:
        inputs, y_true_batch = batch

        # store the true label of the input element
        y_true.append(y_true_batch.numpy())

        # compute the prediction of the model
        pred_proba = F.softmax(model(inputs.to(model_device)).cpu()).detach().numpy()
        y_pred.append(np.argmax(pred_proba, axis=-1))
        n_class = pred_proba.shape[-1]

        # port the input to numpy
        inputs = inputs.cpu().detach().numpy()
        batch_size, seq_len, n_channels, n_samples = inputs.shape

        # reshape the input to consider only the input signal
        filtered_inputs = inputs.copy()
        filtered_inputs = filtered_inputs.reshape(-1, seq_len * n_samples)

        # remove the frequency band from the input using scipy
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
                filtered_inputs[index] = signal.sosfilt(sos, filtered_inputs[index])

        # reshape the input signal to the original size and port it to tensor
        filtered_inputs = filtered_inputs.reshape(
            batch_size, seq_len, n_channels, n_samples
        )
        filtered_inputs = torch.from_numpy(filtered_inputs)
        inputs = torch.from_numpy(inputs)

        # compute the prediction of the model with the filtered input, the prediction is a tensor of size batch_size * seq_len, n_classes
        batch_importance = (
            F.softmax(model(filtered_inputs.to(model_device)).cpu()).detach().numpy()
        )

        # the importance is the difference between the prediction with the original input and the prediction with the filtered input
        batch_importance = pred_proba - batch_importance
        band_importance.append(batch_importance)

        ig = IntegratedGradients(model)

        partial_time_importance = []
        for c in range(n_class):
            partial_time_importance.append(
                ig.attribute(
                    inputs.to(model_device), filtered_inputs.to(model_device), target=c
                )
                .cpu()
                .numpy()
            )

        time_importance.append(partial_time_importance)

    # reshape the lists to ignore the batch_size dimension
    y_pred = np.concatenate(y_pred).reshape(-1)
    y_true = np.concatenate(y_true).reshape(-1)
    band_importance = np.concatenate(band_importance).reshape(-1, n_class)

    return time_importance, band_importance, y_pred, y_true


# RICORDA DI LEVARE I PRIMI DUE PARAMETRI
def compute_band_importance(
    bands: List[List[float]],
    band_names: List[str],
    model: torch.nn.Module,
    dataloader: DataLoader,
    model_device: torch.device,
    sampling_rate: int = 100,
    class_names: list[str] = ["Wake", "N1", "N2", "DS", "REM"],
    average_type: int = 0,
):

    for i in range(len(bands)):
        assert len(bands[i]) == 2
    assert len(band_names) == 6
    assert len(class_names) == 5
    assert average_type == 0 or average_type == 1

    # the dataloader is recreated here with the parameter shuffle = False in order to have consistency with the order of data
    # this allows us to be precise in calculating different importances referring to the same samples

    dataloader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
    )

    band_freq_combinations = []
    band_combinations_dict = {}
    band_time_combinations_dict = {}
    permutations_array = []

    for i in range(len(bands)):
        combination_list = it.combinations(bands, i + 1)
        for elem in combination_list:
            band_freq_combinations.append(elem)

    for cross_band in band_freq_combinations:
        permuted_bands = np.zeros(len(bands))

        for i, band in enumerate(bands):
            if band in cross_band:
                permuted_bands[i] = 1

        print(permuted_bands)
        permutations_array.append(permuted_bands)
        time_importance, band_importance, y_pred, y_true = (
            _compute_cross_band_importance(
                cross_band, model, dataloader, model_device, sampling_rate
            )
        )

        band_combinations_dict[str(permuted_bands)] = band_importance
        band_time_combinations_dict[str(permuted_bands)] = time_importance

    permuted_bands_importance = []
    permuted_bands_time_importance = []

    for i in range(len(permutations_array)):
        key = permutations_array[i]
        permuted_bands_importance.append(band_combinations_dict[str(key)])
        permuted_bands_time_importance.append(band_time_combinations_dict[str(key)])

    importances_matrix = []
    time_importances_matrix = []

    for i in range(len(bands)):

        # simple_average
        if average_type == 0:
            band_importance = get_simple_importance(
                permuted_bands_importance, permutations_array, i
            )
            band_time_importance = get_simple_importance(
                permuted_bands_time_importance, permutations_array, i
            )
        # weighted_average
        elif average_type == 1:
            band_importance = get_weighted_importance(
                permuted_bands_importance, permutations_array, i
            )
            band_time_importance = get_weighted_importance(
                permuted_bands_time_importance, permutations_array, i
            )

        importances_matrix.append(band_importance)
        time_importances_matrix.append(band_time_importance)

    return time_importances_matrix, importances_matrix, y_pred, y_true


def sum_nested_lists(list1, list2):
    if not isinstance(list1[0], list):
        return [a + b for a, b in zip(list1, list2)]

    return [
        sum_nested_lists(sublist1, sublist2) for sublist1, sublist2 in zip(list1, list2)
    ]


def divide_nested_list(list, factor):
    if not isinstance(list[0], list):
        return [element / factor for element in list]

    return [divide_nested_list(sublist, factor) for sublist in list]


def multiply_nested_list(list, factor):
    if not isinstance(list[0], list):
        return [element * factor for element in list]

    return [multiply_nested_list(sublist, factor) for sublist in list]


def get_simple_importance(
    permuted_bands_importance: List[List],
    permutations_array: List[List[int]],
    band: int = 0,
):
    importance = []
    counter = 0

    for i in range(len(permutations_array)):
        if permutations_array[i][band] == 1:
            if len(importance) == 0:
                importance = permuted_bands_importance[i].copy()
            else:
                importance = sum_nested_lists(importance, permuted_bands_importance[i])
            counter += 1

    importance = divide_nested_list(importance, counter)
    return importance


def get_weighted_importance(
    permuted_bands_importance: List[List],
    permutations_array: List[List[int]],
    band: int = 0,
):
    importance = []
    weights_sum = 0

    for i in range(len(permutations_array)):
        if permutations_array[i][band] == 1:
            weight = 1 / (np.sum(permutations_array[i] == 1))
            arr = multiply_nested_list(permuted_bands_importance[i], weight)

            if len(importance) == 0:
                importance = arr.copy()
            else:
                importance = sum_nested_lists(importance, arr)

            weights_sum += weight

    importance = divide_nested_list(importance, weights_sum)

    return importance


class FreqBandsExplainer(PhysioExplainer):
    def __init__(
        self,
        model_name: str = "chambon2018",
        dataset_name: str = "sleep_physioex",
        loss_name: str = "cel",
        ckp_path: str = None,
        version: str = "2018",
        use_cache: bool = True,
        sequence_lenght: int = 3,
        batch_size: int = 32,
        sampling_rate: int = 100,
        class_name: list = ["Wake", "NREM1", "NREM2", "DeepSleep", "REM"],
    ):
        super().__init__(
            model_name,
            dataset_name,
            loss_name,
            ckp_path,
            version,
            use_cache,
            sequence_lenght,
            batch_size,
        )
        self.sampling_rate = sampling_rate
        self.class_name = class_name

    def compute_band_importance(
        self,
        bands: List[List[float]],
        band_names: List[str],
        fold: int = 0,
        plot_pred: bool = False,
        plot_true: bool = False,
        save_csv: bool = False,
        pred_or_true: int = 0,
    ):
        logger.info(
            "JOB:%d-Loading model %s from checkpoint %s"
            % (fold, str(self.model_call), self.checkpoints[fold])
        )
        model = self.model_call.load_from_checkpoint(
            self.checkpoints[fold], module_config=self.module_config
        ).eval()

        model_device = next(model.parameters()).device

        logger.info(
            "JOB:%d-Splitting dataset into train, validation and test sets" % fold
        )
        self.dataset.split(fold)

        datamodule = TimeDistributedModule(
            dataset=self.dataset,
            sequence_lenght=self.module_config["seq_len"],
            batch_size=self.batch_size,
            transform=self.input_transform,
            target_transform=self.target_transform,
        )

        self.module_config["loss_params"]["class_weights"] = datamodule.class_weights()

        dataloader = datamodule.train_dataloader()

        ##############################################################################################
        # In order to elaborate our bands' time importance, we cut the dataset and only retrieve the #
        # first batch. This choice has been made for limitations with computing power, resulting in  #
        # a very large use of RAM, which prevented the function to elaborate everything until its    #
        # conclusion. If you think you have enough computing power and you want to use the original  #
        # dataset, please remove or comment the following line, and remember to swap "new_dataloader"#
        # with "dataloader" in the function parameters.                                              #
        ##############################################################################################

        # Comment should begin here
        d_iter = iter(dataloader)
        first_input, first_target = next(d_iter)

        new_dataloader = DataLoader(
            TensorDataset(first_input, first_target),
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
        )
        # Comment should end here

        for i in range(2):
            # Swap the parameter here
            time_importances_matrix, importances_matrix, y_pred, y_true = (
                compute_band_importance(
                    bands,
                    band_names,
                    model,
                    new_dataloader,
                    model_device,
                    self.sampling_rate,
                    self.class_name,
                    i,
                )
            )

            if i == 0:
                word = "simple"
            elif i == 1:
                word = "weighted"

            for j, band in enumerate(band_names):

                if plot_true:
                    ########## plot of simple importance ###########

                    # boxplot of the band simple importance of the true label
                    logger.info(
                        "JOB:%d-Plotting band %s %s importance for true label"
                        % (fold, band, word)
                    )
                    true_importance = []

                    for k in range(len(y_true)):
                        true_importance.append(importances_matrix[j][k][y_true[k]])

                    true_importance = np.array(true_importance)

                    df = pd.DataFrame(
                        {
                            "Band "
                            + band
                            + " "
                            + word
                            + " Importance": true_importance,
                            "Class": y_true,
                        }
                    )

                    # boxplot of the true importance of the band with seaborn
                    plt.figure(figsize=(10, 10))
                    ax = sns.boxplot(
                        x="Class",
                        y="Band " + band + " " + word + " Importance",
                        data=df,
                    )
                    ax.set_xticklabels(self.class_name)
                    plt.title(
                        "Band " + band + " " + word + " Importance for True Label"
                    )
                    plt.xlabel("Class")
                    plt.ylabel("Importance")
                    plt.savefig(
                        self.ckpt_path
                        + ("fold=%d_true_band=" + band + "_" + word + "_importance.png")
                        % fold
                    )
                    plt.close()

                if plot_pred:
                    ########## plot of simple importance ###########

                    logger.info(
                        "JOB:%d-Plotting band %s %s importance for predicted label"
                        % (fold, band, word)
                    )
                    pred_importance = []

                    for k in range(len(y_true)):
                        pred_importance.append(importances_matrix[j][k][y_pred[k]])

                    pred_importance = np.array(pred_importance)

                    df = pd.DataFrame(
                        {
                            "Band "
                            + band
                            + " "
                            + word
                            + " Importance": pred_importance,
                            "Class": y_true,
                        }
                    )

                    # boxplot of the true importance of the band with seaborn
                    plt.figure(figsize=(10, 10))
                    ax = sns.boxplot(
                        x="Class",
                        y="Band " + band + " " + word + " Importance",
                        data=df,
                    )
                    ax.set_xticklabels(self.class_name)
                    plt.title(
                        "Band " + band + " " + word + " Importance for Predicted Label"
                    )
                    plt.xlabel("Class")
                    plt.ylabel("Importance")
                    plt.savefig(
                        self.ckpt_path
                        + ("fold=%d_pred_band=" + band + "_" + word + "_importance.png")
                        % fold
                    )
                    plt.close()

            for k, class_name in enumerate(self.class_name):

                for l in range(len(y_true)):
                    if pred_or_true == 0 and y_pred[l] == k:
                        index = l
                        target = self.class_name[y_pred[l]]
                        break
                    elif pred_or_true == 1 and y_true[l] == k:
                        index = l
                        target = self.class_name[y_true[l]]
                        break

                internal_index = l % 32
                batch_index = int(l / 32)
                data_iter = iter(dataloader)
                current_batch = next(data_iter)
                for z in range(batch_index):
                    current_batch = next(data_iter)

                inputs, _ = current_batch
                inputs = inputs.cpu().detach().numpy()
                batch_size, seq_len, n_channels, n_samples = inputs.shape
                inputs = inputs[internal_index].reshape(seq_len, n_samples)

                heatmap_rows = []
                heatmap_input = []

                for j, band in enumerate(band_names):
                    logger.info(
                        "JOB:%d-Plotting time importance of target band %s for target class %s"
                        % (fold, band, class_name)
                    )
                    plot_matrix = time_importances_matrix[j][batch_index][k][
                        internal_index
                    ]
                    heatmap_rows.append([])

                    fig, axs = plt.subplots(2, 3, figsize=(30, 5))

                    for a in range(seq_len):
                        y = []
                        for b in range(n_samples):
                            y.append(plot_matrix[a][0][b])

                        heatmap_rows[j].append(plot_matrix[a][0])

                        x = np.arange(n_samples)

                        plt.subplots_adjust(hspace=0.5)

                        plt.subplot(2, 3, a + 1)
                        plt.plot(x, y)

                        plt.ylabel("Time Importance")

                        x = np.arange(n_samples)
                        y = inputs[a]
                        if len(heatmap_input) < seq_len:
                            heatmap_input.append(y)

                        plt.subplot(2, 3, a + 4)
                        plt.plot(x, y)

                        plt.ylabel("Wave value")
                        plt.xlabel("Samples")

                    axs[0, 1].set_title(
                        "Band "
                        + band
                        + ": predicted "
                        + self.class_name[y_pred[index]]
                        + ", true "
                        + self.class_name[y_true[index]]
                        + ", "
                        + word
                        + " importance for "
                        + target
                    )
                    axs[1, 1].set_title("Original corresponding input wave")
                    plt.savefig(
                        self.ckpt_path
                        + "band="
                        + band
                        + "_class="
                        + class_name
                        + "_"
                        + word
                        + ".png"
                    )
                    plt.close(fig)

                heatmap_rows1 = np.array(heatmap_rows).copy()
                heatmap_rows1 = zscore(heatmap_rows1, axis=None)
                heatmap_rows1 = (
                    2
                    * (heatmap_rows1 - np.min(heatmap_rows1))
                    / (np.max(heatmap_rows1) - np.min(heatmap_rows1))
                    - 1
                )

                df_heatmap_input = []
                heatmap_dataframe = []

                fig, axs = plt.subplots(2, 3, figsize=(30, 6))
                personalized_colors = sns.color_palette("coolwarm", as_cmap=True)
                for p in range(seq_len):
                    df_heatmap_input.append(pd.DataFrame({"input": heatmap_input[p]}))
                    df_heatmap_input[p].to_csv(
                        self.ckpt_path
                        + "heatmap_input_dataframe_class_"
                        + class_name
                        + "_sequence="
                        + str(p)
                        + ".csv",
                        index=False,
                    )
                    seq_row = []
                    for o in range(len(bands)):
                        seq_row.append(heatmap_rows1[o][p])
                    heatmap_dataframe.append(pd.DataFrame(seq_row))
                    heatmap_dataframe[p].to_csv(
                        self.ckpt_path
                        + "heatmap_dataframe_class_"
                        + class_name
                        + "_sequence="
                        + str(p)
                        + ".csv",
                        index=False,
                    )
                    sns.heatmap(
                        heatmap_dataframe[p],
                        xticklabels=False,
                        yticklabels=band_names,
                        ax=axs[0, p],
                        cmap=personalized_colors,
                        vmin=-1,
                        vmax=1,
                        cbar=False,
                    )
                    x = np.arange(n_samples)
                    y = heatmap_input[p]
                    plt.subplot(2, 3, p + 4)
                    plt.plot(x, y)
                    plt.xlim([0, max(x)])
                    plt.ylabel("Wave value")
                    plt.xlabel("Samples")

                axs[0, 1].set_title(
                    "Time Importance: predicted "
                    + self.class_name[y_pred[index]]
                    + ", true "
                    + self.class_name[y_true[index]]
                    + ", "
                    + word
                    + " importance for "
                    + target
                    + "(normalized between -1 and 1)"
                )
                plt.tight_layout()
                plt.savefig(
                    self.ckpt_path
                    + "bands_heatmap_for_class="
                    + class_name
                    + "_(predicted_"
                    + self.class_name[y_pred[index]]
                    + "_true_"
                    + self.class_name[y_true[index]]
                    + ")_"
                    + word
                    + "_norm1.png"
                )
                plt.close(fig)

                heatmap_rows2 = np.array(heatmap_rows).copy()
                heatmap_rows2 = np.abs(heatmap_rows2)
                heatmap_rows2 = zscore(heatmap_rows2, axis=None)
                heatmap_rows2 = (heatmap_rows2 - np.min(heatmap_rows2)) / (
                    np.max(heatmap_rows2) - np.min(heatmap_rows2)
                )

                df_heatmap_input = []
                heatmap_dataframe = []

                fig, axs = plt.subplots(2, 3, figsize=(30, 6))
                personalized_colors = sns.light_palette("red", as_cmap=True)
                for p in range(seq_len):
                    seq_row = []
                    for o in range(len(bands)):
                        seq_row.append(heatmap_rows2[o][p])
                    heatmap_dataframe.append(pd.DataFrame(seq_row))
                    sns.heatmap(
                        heatmap_dataframe[p],
                        xticklabels=False,
                        yticklabels=band_names,
                        ax=axs[0, p],
                        cmap=personalized_colors,
                        vmin=0,
                        vmax=1,
                        cbar=False,
                    )
                    x = np.arange(n_samples)
                    y = heatmap_input[p]
                    plt.subplot(2, 3, p + 4)
                    plt.plot(x, y)
                    plt.xlim([0, max(x)])
                    plt.ylabel("Wave value")
                    plt.xlabel("Samples")

                axs[0, 1].set_title(
                    "Time Importance: predicted "
                    + self.class_name[y_pred[index]]
                    + ", true "
                    + self.class_name[y_true[index]]
                    + ", "
                    + word
                    + " importance for "
                    + target
                    + "(normalized between 0 and 1)"
                )
                plt.tight_layout()
                plt.savefig(
                    self.ckpt_path
                    + "bands_heatmap_for_class="
                    + class_name
                    + "_(predicted_"
                    + self.class_name[y_pred[index]]
                    + "_true_"
                    + self.class_name[y_true[index]]
                    + ")_"
                    + word
                    + "_norm2.png"
                )
                plt.close(fig)

                df_current_average = pd.DataFrame(
                    importances_matrix[j], columns=self.class_name
                )
                df_current_average["Predicted Label"] = y_pred
                df_current_average["True Label"] = y_true
                df_current_average["Fold"] = fold

                if save_csv:
                    df_current_average.to_csv(
                        self.ckpt_path
                        + "band="
                        + band
                        + "_"
                        + word
                        + "_importance.csv",
                        mode="a",
                        index=False,
                    )

        return importances_matrix

    def plot(self, band_names: List[str]):

        for k, class_name in enumerate(self.class_name):

            heatmap_input = []
            heatmap_dataframe = []

            for p in range(3):

                if len(heatmap_input) < 3:
                    df_heatmap_input = pd.read_csv(
                        self.ckpt_path
                        + "heatmap_input_dataframe_class_"
                        + class_name
                        + "_sequence="
                        + str(p)
                        + ".csv"
                    )
                    heatmap_input.append(df_heatmap_input["input"].tolist())

                df_heatmap_rows = pd.read_csv(
                    self.ckpt_path
                    + "heatmap_dataframe_class_"
                    + class_name
                    + "_sequence="
                    + str(p)
                    + ".csv"
                )
                heatmap_dataframe.append(df_heatmap_rows.values.tolist())

            heatmap_dataframe = np.array(heatmap_dataframe)
            heatmap_zscore1 = zscore(heatmap_dataframe, axis=None)
            heatmap_zscore1_normalized = (
                2
                * (heatmap_zscore1 - np.min(heatmap_zscore1))
                / (np.max(heatmap_zscore1) - np.min(heatmap_zscore1))
                - 1
            )

            fig, axs = plt.subplots(2, 3, figsize=(30, 6))
            personalized_colors = sns.color_palette("coolwarm", as_cmap=True)

            for p in range(3):
                sns.heatmap(
                    heatmap_zscore1_normalized[p],
                    xticklabels=False,
                    yticklabels=band_names,
                    ax=axs[0, p],
                    cmap=personalized_colors,
                    vmin=-1,
                    vmax=1,
                    cbar=False,
                )
                x = np.arange(3000)
                y = heatmap_input[p]
                plt.subplot(2, 3, p + 4)
                plt.plot(x, y)
                plt.xlim([0, max(x)])
                plt.ylabel("Wave value")
                plt.xlabel("Samples")

            axs[0, 1].set_title("Title")
            plt.tight_layout()
            plt.savefig(
                self.ckpt_path + "bands_heatmap_for_class=" + class_name + "_norm1.png"
            )
            plt.close(fig)

            heatmap_dataframe_abs = np.abs(heatmap_dataframe)
            heatmap_zscore2 = zscore(heatmap_dataframe_abs, axis=None)
            heatmap_zscore2 = (heatmap_zscore2 - np.min(heatmap_zscore2)) / (
                np.max(heatmap_zscore2) - np.min(heatmap_zscore2)
            )

            fig, axs = plt.subplots(2, 3, figsize=(30, 6))
            personalized_colors = sns.light_palette("red", as_cmap=True)

            for p in range(3):
                sns.heatmap(
                    heatmap_zscore2[p],
                    xticklabels=False,
                    yticklabels=band_names,
                    ax=axs[0, p],
                    cmap=personalized_colors,
                    vmin=0,
                    vmax=1,
                    cbar=False,
                )
                x = np.arange(3000)
                y = heatmap_input[p]
                plt.subplot(2, 3, p + 4)
                plt.plot(x, y)
                plt.xlim([0, max(x)])
                plt.ylabel("Wave value")
                plt.xlabel("Samples")

            axs[0, 1].set_title("Title")
            plt.tight_layout()
            plt.savefig(
                self.ckpt_path + "bands_heatmap_for_class=" + class_name + "_norm2.png"
            )
            plt.close(fig)

    def explain(
        self,
        bands: list[list[float]],
        band_names: list[str],
        save_csv: bool = False,
        plot_pred: bool = False,
        plot_true: bool = False,
        n_jobs: int = 10,
    ):

        # Only execute compute_band_importance for the fold 0
        result = self.compute_band_importance(
            bands, band_names, 0, plot_pred, plot_true, save_csv
        )
        return result

        # Execute compute_band_importance in parallel for every checkpoint
        # result = Parallel(n_jobs=n_jobs)(delayed(self.compute_band_importance)(bands, band_names, int(fold), plot_pred, plot_true, save_csv) for fold in self.checkpoints.keys())

        return result
