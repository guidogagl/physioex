"""Chambon et al. (2018) deep learning architecture for sleep staging.

Reference: "A deep learning architecture for temporal sleep stage
classification using multivariate and multimodal time series"
(IEEE TNSRE 2018, arXiv:1707.03321).

Architecture (from the paper):
  Per-epoch: spatial_conv -> 2x(conv + ReLU + maxpool) -> flatten
  Sequence:  concatenate all L epoch features -> dropout -> linear -> softmax

The model classifies the **central epoch** of a sequence of L epochs,
using temporal context from surrounding epochs.  This matches the
paper's approach where features from multiple epochs are concatenated
before the final classification layer.

Uses braindecode's SleepStagerChambon2018 as the epoch encoder (official
implementation matching the paper's architecture).
"""

import torch
from braindecode.models import SleepStagerChambon2018
from torch import nn
from torchinfo import summary

from physioex.train.trainer import Trainer
from physioex.data.dataset import PhysioExDataset


class Chambon2018Net(nn.Module):
    """Chambon et al. (2018) sleep staging network.

    Encodes each epoch independently via braindecode's SleepStagerChambon2018,
    then concatenates all L epoch features and classifies the **central epoch**.

    Input:  (B, L, C, T) — batch, sequence_length, channels, time_samples
    Output: (B, 1, n_classes) — logit for the central epoch only

    This matches the paper: "the temporal context of each 30s window of data"
    is exploited by concatenating features from surrounding epochs.

    Args:
        n_classes: Number of sleep stages (default 5).
        in_channels: Number of input channels (default 1).
        sf: Sampling frequency in Hz (default 100).
        n_times: Samples per epoch (default 3000 = 30s @ 100Hz).
        dropout: Dropout before the classifier (default 0.25, as in paper).
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_channels: int = 1,
        sf: int = 100,
        n_times: int = 3000,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.epoch_encoder = SleepStagerChambon2018(
            n_chans=in_channels,
            sfreq=sf,
            n_outputs=n_classes,
            n_times=n_times,
            return_feats=True,
        )

        # The paper concatenates features from ALL L epochs and classifies
        # with a single linear layer.  We don't know L at init time, so
        # we use a lazy linear that infers input size on first forward.
        self.drop = nn.Dropout(dropout)
        self.clf = nn.LazyLinear(n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode each epoch independently into a feature vector.

        Args:
            x: (B, L, C, T) input tensor.

        Returns:
            (B, L, D) epoch-level feature embeddings.
        """
        batch_size, seqlen, nchan, nsamp = x.size()
        x = x.reshape(-1, nchan, nsamp)
        x = self.epoch_encoder(x)
        x = x.reshape(batch_size, seqlen, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode all epochs, concatenate, classify central.

        Args:
            x: (B, L, C, T) input tensor.

        Returns:
            (B, 1, n_classes) logits for the central epoch.
        """
        # Encode each epoch independently
        feats = self.encode(x)  # (B, L, D)

        # Concatenate all L epoch features into one vector
        batch_size = feats.size(0)
        concat = feats.reshape(batch_size, -1)  # (B, L*D)

        # Classify the central epoch
        out = self.drop(concat)
        out = self.clf(out)  # (B, n_classes)

        # Return as (B, 1, n_classes) for consistency with other models
        return out.unsqueeze(1)


if __name__ == "__main__":

    dataset = PhysioExDataset(datasets=["sleepedf"])

    model = Chambon2018Net(
        in_channels=dataset.get_num_channels(),
    )

    print("Chambon2018Net summary:")
    summary(model, (32, 21, dataset.get_num_channels(), 3000))

    print("\nTraining Chambon2018Net...")
    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=20,
        lr=1e-3,
    )

    print("\nEvaluating Chambon2018Net...")
    results = Trainer.evaluate(
        model=model,
        dataset=dataset,
    )
    print(results)
