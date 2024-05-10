import torch
import torch.optim as optim
from braindecode.models import SleepStagerChambon2018
from torch import nn

from physioex.train.networks.base import SeqtoSeq

module_config = dict()


class SequenceEncoder(nn.Module):
    """
    The sequence encoder neural network module used in Chambon 2018.

    This module encodes the input sequences by concatenating them and using a linear layer to perform classification.

    Args:
        input_dim (int): The dimension of the input.
        n_classes (int): The number of classes for classification.
        latent_dim (int): The dimension of the latent space i.e. intermediate layer between encodings and classification.

    Returns:
        torch.Tensor: The output tensor after classification.
    """
    def __init__(self, input_dim: int, n_classes: int, latent_dim: int):
        super(SequenceEncoder, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.proj = nn.Linear(input_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.clf = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        """
        Forward pass of the sequence encoder.

        Args:
            x: torch.Tensor - The input tensor.

        Returns:
            torch.Tensor: The output tensor after classification.
        """
        x = self.encode(x)
        x = self.clf(x)
        return x

    def encode(self, x):
        """
        Encodes the input tensor.

        Args:
            x: torch.Tensor - The input tensor.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        batch_size, sequence_lenght, input_dim = x.size()
        x = x.reshape(batch_size, sequence_lenght * input_dim)
        x = self.drop(x)
        x = nn.ReLU()(self.proj(x))
        x = self.norm(x)
        return x


class Chambon2018Net(SeqtoSeq):
    """
    A neural network model based on Chambon et al. (2018).

    This model utilizes an epoch encoder and a sequence encoder for classification.

    Args:
        module_config: The configuration for the model.

    Returns:
        torch.Tensor: The computed loss value.
    """
    def __init__(self, module_config=module_config):
        super(Chambon2018Net, self).__init__(None, None, module_config)

        epoch_encoder = SleepStagerChambon2018(
            n_chans=module_config["n_channels"],
            sfreq=module_config["sfreq"],
            n_outputs=module_config["n_classes"],
            n_times=module_config["n_times"],
            return_feats=True,
        )

        sequence_encoder = SequenceEncoder(
            epoch_encoder.len_last_layer * module_config["seq_len"],
            module_config["n_classes"],
            module_config["latent_space_dim"],
        )

        super(Chambon2018Net, self).__init__(
            epoch_encoder, sequence_encoder, module_config
        )

    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):
        """
        Computes the loss for the Chambon2018Net model.

        Args:
            embeddings: The embeddings.
            outputs: The model outputs.
            targets: The target values.
            log (str): The logging information.
            log_metrics (bool): Whether to log metrics.

        Returns:
            torch.Tensor: The computed loss value.
        """
        batch_size, n_class = outputs.size()
        outputs = outputs.reshape(batch_size, 1, n_class)
        embeddings = embeddings.reshape(batch_size, 1, -1)

        return super().compute_loss(embeddings, outputs, targets, log, log_metrics)
