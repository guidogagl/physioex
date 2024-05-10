import torch
import torch.optim as optim
from braindecode.models import SleepStagerChambon2018
from torch import nn

from physioex.train.networks.base import SeqtoSeq


class EpochEncoder(nn.Module):
    def __init__(self, module_config) -> None:
        super(EpochEncoder, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=module_config["in_channels"],
            out_channels=32,
            kernel_size=10,
            stride=5,
        )
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=16, kernel_size=10, stride=5
        )
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=10, stride=5)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = x.reshape(x.size(0), -1)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, module_config):
        super(SequenceEncoder, self).__init__()

        self.LSTM = nn.LSTM(
            input_size=module_config["epoch_out"],
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.proj = nn.Linear(2 * 64, module_config["latent_space_dim"])
        self.norm = nn.LayerNorm(module_config["latent_space_dim"])
        self.lin = nn.Linear(
            module_config["latent_space_dim"], module_config["n_classes"]
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.lin(x)
        return x

    def encode(self, x):
        x, _ = self.LSTM(x)
        x = nn.ReLU()(self.proj(x))
        x = self.norm(x)
        return x


class SeqECGnet(SeqtoSeq):
    def __init__(self, module_config: dict = None):
        super(SeqECGnet, self).__init__(
            epoch_encoder=EpochEncoder(module_config=module_config),
            sequence_encoder=SequenceEncoder(module_config=module_config),
            config=module_config,
        )
