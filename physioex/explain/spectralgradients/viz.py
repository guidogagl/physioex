from typing import Callable, List, Tuple

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

sns.set_theme(style="whitegrid")

from typing import Dict

import pandas as pd
from loguru import logger
from tqdm import tqdm

from physioex.explain.spectralgradients.spectral_gradients import \
    SpectralGradients
from physioex.explain.spectralgradients.utils import generate_frequency_bands


def plot_class_spectrum(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    sg: SpectralGradients,
    n_classes: int = 5,
    n_batches: int = torch.inf,
    classes: Dict = None,
    relevant_bands: Dict = None,
    band_colors: Dict = None,
    eeg_bands: Dict = None,
    filename: str = None,
    figsize: Tuple[float, float] = (10, 20),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):

    n_bands = len(sg.bands)
    spectral_importances = [
        pd.DataFrame([], columns=[str(band) for band in sg.bands])
        for _ in range(n_classes)
    ]

    logger.info(f"Computing spectral importances for {n_batches} batches")
    error = False

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader)):

            baselines = sg._construct_baseline(inputs)
            importances = torch.zeros(len(inputs), n_bands)

            temp_probas = model(inputs.to(device)).detach().cpu()
            y_preds = temp_probas.argmax(dim=-1)

            if not error:
                try:
                    # if we got engough gpu memory we can compute the spectral importances in one go
                    # baselines is a tensor of shape
                    batch_size, seq_len, n_channels, n_samples, n_frequencies = (
                        baselines.shape
                    )

                    b = torch.permute(baselines.clone(), (0, 4, 1, 2, 3)).reshape(
                        batch_size * n_frequencies, seq_len, n_channels, n_samples
                    )
                    y_probas = model(b.to(device)).detach().cpu()
                    y_probas = y_probas.reshape(batch_size, n_frequencies, n_classes)
                    y_probas = torch.permute(y_probas, (0, 2, 1))

                    for i in range(1, n_frequencies + 1):
                        for j in range(len(inputs)):
                            importances[j, -i] = (
                                temp_probas[j, y_preds[j]] - y_probas[j, y_preds[j], -i]
                            )

                        temp_probas = y_probas[..., -i]

                except RuntimeError:
                    error = True
                    logger.exception("Error computing spectral importances in one go")
                    # log the stack trace

            else:
                for i in range(1, baselines.shape[-1] + 1):
                    y_probas = model(baselines[..., -i].to(device)).detach().cpu()

                    for j in range(len(inputs)):
                        importances[j, -i] = torch.norm(
                            y_probas[j, y_preds[j]] - temp_probas[j, y_preds[j]]
                        )

                    temp_probas = y_probas

            for i in range(len(inputs)):
                new_row = pd.DataFrame(
                    importances[i]
                    .numpy()
                    .reshape(-1, len(spectral_importances[y_preds[i]].columns)),
                    columns=spectral_importances[y_preds[i]].columns,
                )
                spectral_importances[y_preds[i]] = spectral_importances[
                    y_preds[i]
                ].append(new_row, ignore_index=True)

            if batch_idx >= n_batches:
                break

    fig, ax = plt.subplots(n_classes, 1, figsize=figsize, sharex=True, sharey=True)

    logger.info("Plotting spectral importances")

    for i in range(n_classes):
        class_label = classes[i] if classes is not None else f"Class {i}"
        ax[i].set_title(
            f"Predicted: {class_label}", fontsize=12, y=1.05
        )  # Usa il nome della fase del sonno come titolo

        df_new = pd.DataFrame(columns=["frequency", "value"])
        for column in spectral_importances[i].columns:
            start, end = map(float, column.strip("[]").split(", "))
            df_new = pd.concat(
                [
                    df_new,
                    pd.DataFrame(
                        {
                            "frequency": [start],
                            "value": [spectral_importances[i][column].mean()],
                        }
                    ),
                ]
            )
            df_new = pd.concat(
                [
                    df_new,
                    pd.DataFrame(
                        {
                            "frequency": [end],
                            "value": [spectral_importances[i][column].mean()],
                        }
                    ),
                ]
            )

        if (
            relevant_bands is not None
            and band_colors is not None
            and eeg_bands is not None
        ):
            for band in relevant_bands[
                classes[i]
            ]:  # Usa solo le bande rilevanti per la fase del sonno corrente
                ax[i].fill_betweenx(
                    [-1, 1],
                    *eeg_bands[band],
                    color=band_colors[band],
                    alpha=0.2,
                    label=band,
                )  # Regola l'opacità con alpha

        # Crea il grafico a linee con l'asse x rappresentante le frequenze e l'asse y rappresentante i valori
        sns.lineplot(
            x="frequency", y="value", data=df_new, drawstyle="steps-pre", ax=ax[i]
        )

        ax[i].set_ylim([-0.5, 0.5])
        ax[i].legend()  # Mostra la legenda

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)

    return


def plot(
    input: torch.Tensor,
    attr: torch.Tensor,
    bands: List[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (25, 5),
):

    assert input.dim() == 3, "Input must be a 3D tensor with dimensions: S x C x NS"
    S, C, NS = input.size()

    assert attr.dim() == 4, "Attr must be a 4D tensor with dimensions: S x C x NS x F"
    F = attr.size(-1)
    # check input and attr dimensions match
    assert (
        S == attr.size(0) and C == attr.size(1) and NS == attr.size(2)
    ), "Input and Attr dimensions do not match"

    if bands is not None:
        assert F == len(
            bands
        ), "Number of bands must match the number of frequency bands"
        assert all(
            len(band) == 2 for band in bands
        ), "Each band must be a tuple with 2 elements"

    ##### table plot #####
    ### table            S1,        S2,         S3 ...
    ###              AttrC1S1,  AttrC1S2,  AttrC1S3 ...
    ###              InputC1S1, InputC1S2, InputC1S3 ...
    ###
    ###              AttrC2S1,  AttrC2S2,  AttrC2S3 ...
    ###              InputC2S1, InputC2S2, InputC2S3 ...
    ###

    max_time = input.max().item()
    min_time = input.min().item()

    attr_vmin = attr.min().item()
    attr_vmax = attr.max().item()

    # plot one subplot for each element of the sequence and one row for each channel
    fig, axs = plt.subplots(C * 2, S, figsize=figsize, sharex="col", sharey="row")

    for c in range(C):
        # time on the 2 line
        # attr on the first line

        time_row = (c * 2) + 1
        attr_row = c * 2

        for s in range(S):

            # plot the time
            sns.lineplot(x=range(NS), y=input[s, c].numpy(), ax=axs[time_row, s])

            axs[time_row, s].set_ylim(min_time, max_time)
            axs[time_row, s].set_xlim(0, NS)
            axs[time_row, s].set_xticklabels([])  # Remove x labels
            axs[time_row, s].set_yticklabels([])  # Remove y labels

            # plot the attr
            sns.heatmap(
                np.transpose(attr[s, c].numpy(), (1, 0)),
                ax=axs[attr_row, s],
                cmap="RdBu",
                cbar=False,
                vmin=attr_vmin,
                vmax=attr_vmax,
            )
            axs[attr_row, s].set_xticklabels([])  # Remove x labels
            axs[attr_row, s].set_yticklabels([])

        axs[attr_row, 0].set_yticklabels([band[0] for band in bands])

    plt.tight_layout()
    return
