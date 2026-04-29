import torch
from braindecode.models import SleepStagerChambon2018
from torch import nn
from torchinfo import summary

from physioex.train.trainer import Trainer
from physioex.data.dataset import PhysioExDataset


class Chambon2018Net(nn.Module):
    """
    Chambon2018Net wraps braindecode's SleepStagerChambon2018 epoch encoder
    in the PhysioEx nn.Module pattern.

    Input:  (B, L, C, T) — batch, sequence_length, channels, time_samples
    Output: (B, L, n_classes) — per-epoch logits

    The epoch encoder extracts features independently per epoch, and then a
    linear classifier maps each epoch's feature vector to class logits.
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_channels: int = 1,
        sf: int = 100,
        n_times: int = 3000,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.epoch_encoder = SleepStagerChambon2018(
            n_chans=in_channels,
            sfreq=sf,
            n_outputs=n_classes,
            n_times=n_times,
            return_feats=True,
        )

        self.drop = nn.Dropout(dropout)
        self.clf = nn.Linear(self.epoch_encoder.len_last_layer, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode each epoch independently into a feature vector.

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
        """
        Forward pass: encode epochs then classify each one.

        Args:
            x: (B, L, C, T) input tensor.

        Returns:
            (B, L, n_classes) logits.
        """
        x = self.encode(x)

        batch_size, seqlen, feat_dim = x.size()
        x = x.reshape(batch_size * seqlen, feat_dim)

        x = self.drop(x)
        x = self.clf(x)

        x = x.reshape(batch_size, seqlen, -1)

        return x


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
