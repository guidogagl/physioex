from typing import Callable

import torch
from captum.attr import IntegratedGradients
from scipy import signal

from physioex.explain.spectralgradients.utils import generate_frequency_bands


class SpectralGradients(IntegratedGradients):
    def __init__(
        self,
        forward_func: Callable,
        multiply_by_inputs: bool = True,
        mode: str = "linear",
        n_bands: int = 10,
        start_freq: float = 0.1,
        fs: int = 100,
        filter_order=5,
    ):
        super().__init__(forward_func, multiply_by_inputs)

        assert mode in {"linear", "log"}, "band_stop must be either 'linear' or 'log'"

        nyquist = fs / 2
        bands = generate_frequency_bands(nyquist, n_bands, start_freq, mode)
        self.bands = bands

        end_band = bands[-1][1]

        print(f"Frequency bands: {bands}")
        self.filters = [
            signal.butter(
                filter_order,
                [band[0] / nyquist, end_band / nyquist],
                btype="bandstop",
                output="sos",
            )
            for band in bands
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
        attr = torch.zeros_like(baselines)

        for f in reversed(range(F)):
            attr[..., f] = (
                super()
                .attribute(
                    inputs,
                    baselines=baselines[..., f].to(inputs.device),
                    target=(
                        target.to(inputs.device)
                        if isinstance(target, torch.Tensor)
                        else target
                    ),  # if tensor put it on the same device as inputs
                    additional_forward_args=additional_forward_args,
                    n_steps=n_steps // F,
                    method=method,
                    internal_batch_size=internal_batch_size,
                    return_convergence_delta=False,
                )
                .cpu()
            )

            inputs = baselines[..., f].to(inputs.device)
        return attr

    def _construct_baseline(self, in_signals: torch.Tensor) -> torch.Tensor:
        assert (
            in_signals.dim() == 4
        ), "Input must be a 4D tensor with dimensions: N x S x C x NS"

        N, S, C, NS = in_signals.size()
        F = len(self.filters)

        in_signal = in_signals.clone().detach().cpu()
        in_signal = in_signal.permute(0, 2, 1, 3).reshape(N, C, S * NS)

        baselines = torch.zeros(F, N, C, S * NS)

        in_signal_np = in_signal.numpy()

        for n in range(N):
            for c in range(C):
                x = in_signal_np[n, c]

                for f, filt in reversed(list(enumerate(self.filters))):

                    baselines[f, n, c] = torch.tensor(signal.sosfilt(filt, x))

                    # x = baselines[f, n, c].numpy()

        baselines = baselines.reshape(F, N, C, S, NS).permute(1, 3, 2, 4, 0)

        return baselines.cpu()
