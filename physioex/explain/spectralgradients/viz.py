from typing import Callable, List, Tuple

import numpy as np

import torch

import seaborn as sns
from matplotlib import pyplot as plt


sns.set_theme(style="whitegrid")

from physioex.explain.spectralgradients.utils import generate_frequency_bands


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
