from typing import Callable, List, Tuple

import torch
from captum.attr import IntegratedGradients
from scipy import signal

import seaborn as sns
from matplotlib import pyplot as plt

sleep_bands = [[0.5, 4], [4, 8], [8, 11.5], [11.5, 15.5], [15.5, 30], [30, 49.5]]


def plot(
    input: torch.Tensor,
    attr: torch.Tensor,
    output: torch.Tensor = None,
    target: int = None,
    band_labels: List[str] = None,
    target_labels: List[str] = None,
):

    assert input.dim() == 3, "Input must be a 3D tensor with dimensions: S x C x NS"
    S, C, NS = input.size()

    assert attr.dim() == 4, "Attr must be a 4D tensor with dimensions: S x C x NS x F"
    F = attr.size(-1)
    # check input and attr dimensions match
    assert (
        S == attr.size(0) and C == attr.size(1) and NS == attr.size(2)
    ), "Input and Attr dimensions do not match"

    input = input.permute(1, 0, 2).reshape(C, S * NS)
    attr = attr.permute(
        1,
        3,
        0,
        2,
    ).reshape(C, F, S * NS)
    time_attr = attr.sum(dim=1)

    # plot one subplot for each element of the sequence and one row for each channel
    fig, axs = plt.subplots(C * 3, figsize=(25, 5))

    for c in range(C):
        # timeseries on the first line y_limit [-1, 1] and no x_labels
        # set xlabels at the end of each sequence
        sns.lineplot(x=range(S * NS), y=input[c].numpy(), ax=axs[c * 3])
        axs[c * 3].set_ylim(-3, 3)
        axs[c * 3].set_xlim(0, S * NS)
        axs[c * 3].set_xticks([i * NS for i in range(S)])

        # time-frequency heatmap on the second line
        # display as y_labels the bands
        # display no xlabels
        sns.heatmap(
            attr[c].numpy(),
            ax=axs[c * 3 + 1],
            cmap="RdBu",
            yticklabels=band_labels,
            cbar=False,
        )
        axs[c * 3 + 1].set_xticks([])

        # time_importance lineplot on the third line
        # display no xlabels
        sns.lineplot(x=range(S * NS), y=time_attr[c].numpy(), ax=axs[c * 3 + 2])
        axs[c * 3 + 2].set_xticks([])
        axs[c * 3 + 2].set_xlim(0, S * NS)
        axs[c * 3 + 2].set_xticks([i * NS for i in range(S)])

    plt.tight_layout()
    return


class SpectralGradients(IntegratedGradients):
    def __init__(
        self,
        forward_func: Callable,
        multiply_by_inputs: bool = True,
        band_stop: List[Tuple[float, float]] = sleep_bands,
        fs: int = 100,
        filter_order=5,
    ):
        super().__init__(forward_func, multiply_by_inputs)

        self.filters = [
            signal.butter(
                filter_order,
                [band[0] / (0.5 * fs), band[1] / (0.5 * fs)],
                btype="bandstop",
                output="sos",
            )
            for band in band_stop
        ]

    def attribute(
        self,
        inputs,
        target=None,
        additional_forward_args=None,
        n_steps=50,
        method="gausslegendre",
        internal_batch_size=None,
    ):
        assert inputs.dim() in {
            3,
            4,
        }, "Input must be a 4D tensor with dimensions: N x S x C x NS"

        F = len(self.filters)

        baselines = self._construct_baseline(inputs)
        attr = torch.zeros_like(baselines).to(inputs.device)

        for f in range(F):
            attr[..., f] = super().attribute(
                inputs,
                baselines=baselines[..., f],
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=False,
            )

            inputs = baselines[..., f]
        return attr

    def _construct_baseline(self, in_signals: torch.Tensor) -> torch.Tensor:
        assert (
            in_signals.dim() == 4
        ), "Input must be a 4D tensor with dimensions: N x S x C x NS"

        N, S, C, NS = in_signals.size()
        F = len(self.filters)

        device = in_signals.device
        in_signal = in_signals.clone().detach().cpu()
        in_signal = in_signal.permute(0, 2, 1, 3).reshape(N, C, S * NS)

        baselines = torch.zeros(F, N, C, S * NS)

        in_signal_np = in_signal.numpy()

        for i in range(N):
            for c in range(C):
                for f, filt in enumerate(self.filters):
                    baselines[f, i, c] = torch.tensor(
                        signal.sosfilt(filt, in_signal_np[i, c])
                    )

        baselines = baselines.reshape(F, N, C, S, NS).permute(1, 3, 2, 4, 0)

        return baselines.to(device)
