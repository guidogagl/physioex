import torch
import torch.nn as nn

from physioex.explain.posthoc import gradients as gradients_module
from physioex.models.prosleepnet import ProtoSleepTransformer


def proj_fn(model: ProtoSleepTransformer, x: torch.Tensor) -> torch.Tensor:

    # x shape : ( batch_size, in_chans, T, F )
    batch_size, in_chans, T, F = x.shape

    # Per-channel epoch encoding
    x = x.reshape(batch_size * in_chans, 1, T, F)
    x = model.epoch_encoder(x)  # (B*C, d_model)

    x = x.reshape(batch_size, in_chans, -1)  # (B, C, d_model)

    # Channel mixer (residual)
    x = x + model.channel_mixer(x)

    return x.mean(dim=1)  # (batch_size, d_model)


def get_prototypes(model: torch.nn.Module, index: int = None) -> torch.Tensor:
    codebook = model.prototype.codebook.detach().cpu().clone()

    if index is None:
        return codebook

    assert (
        index >= 0 and index < codebook.shape[0]
    ), f"Index {index} out of bounds for codebook with shape {codebook.shape}"

    return codebook[index]


class PrototypeRelevance(gradients_module.IntegratedGradients):
    def __init__(
        self,
        model: ProtoSleepTransformer,
        index: int = 0,
        steps: int = 64,
        create_graph: bool = False,
        expects_batch: bool = False,
        **kwargs,
    ):

        super(PrototypeRelevance, self).__init__(
            None,
            steps=steps,
            create_graph=create_graph,
            expects_batch=expects_batch,
            **kwargs,
        )

        self.model = model.eval()  # set the model to evaluation mode

        f = lambda x: torch.nn.functional.cosine_similarity(
            proj_fn(self.model.to(x.device), x),
            get_prototypes(self.model.to(x.device), index=index)
            .unsqueeze(0)
            .to(x.device),
        )
        self.f = f

    def forward(
        self, x, index: int = None, baseline: torch.Tensor = None, steps: int = None
    ):
        if index is not None:
            self.f = lambda x: torch.nn.functional.cosine_similarity(
                proj_fn(self.model.to(x.device), x),
                get_prototypes(self.model.to(x.device), index=index)
                .unsqueeze(0)
                .to(x.device),
            )

        if baseline is None:
            baseline = torch.zeros_like(x)

        return super().forward(x, baseline=baseline, steps=steps)
