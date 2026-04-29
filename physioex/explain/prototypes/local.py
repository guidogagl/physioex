import torch
import torch.nn as nn

from physioex.explain.posthoc import gradients as gradients_module
from physioex.models.protosleepnet import ProtoSleepNet


def proj_fn(model: ProtoSleepNet, x: torch.Tensor) -> torch.Tensor:

    # x shape : ( batch_size, in_chans, ... )
    batch_size, in_chans, T, F = x.shape
    x = x.reshape(
        batch_size * in_chans, 1, T, F
    )  # shape ( batch_size*in_chans, 1, T, F )
    x = model.filterbank(x)

    x = x.permute(0, 2, 1, 3)  # shape ( batch_size*L*in_chan, T, 1, D )

    x = x.reshape(batch_size * in_chans, T, -1)

    x, _ = model.seqn1(x)
    x, _ = model.time_masking(x)

    x = x.reshape(batch_size, in_chans, -1)

    x = x + model.channel_mixer(x)

    return x.mean(dim=1)  # shape ( batch_size, tmhidden )


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
        model: ProtoSleepNet,
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

        # Keep model frozen; disable cuDNN to allow RNN backward in eval mode
        with torch.backends.cudnn.flags(enabled=False):
            return super().forward(x, baseline=baseline, steps=steps)
