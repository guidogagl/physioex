from typing import Callable, Tuple, List

import torch
from captum.attr import IntegratedGradients

from scipy import signal

sleep_bands = [[0.5, 4], [4, 8], [8, 11.5], [11.5, 15.5], [15.5, 30], [30, 49.5]]


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

        self.filters = []
        for band in band_stop:
            lowcut = band[0]
            highcut = band[1]
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            self.filters.append(
                signal.butter(filter_order, [low, high], btype="bandstop", output="sos")
            )

    def attribute(
        self,
        inputs,
        target=None,
        additional_forward_args=None,
        n_steps=50,
        method="gausslegendre",
        internal_batch_size=None,
    ):

        baselines = self._construct_baseline(inputs)
        attr = torch.zeros_like(baselines).to(inputs.device)

        for i in len(self.filters):
            attr[:, i] = super().attribute(
                inputs,
                baselines=baselines[:, i],
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=False,
            )
        return attr

    def _construct_baseline(self, signal: torch.Tensor) -> torch.Tensor:

        baselines = torch.zeros(len(self.filters), **signal.size())
        axes = list(range(len(baselines.size())))
        axes[0] = 1
        axes[1] = 0
        baselines = torch.permute(baselines, axes)

        for i in range(baselines.size(0)):
            for f, filter in enumerate(self.filters):
                baselines[i, f] = torch.tensor(signal.sosfilt(filter, signal[i]))

        return baselines.to(signal.device)
