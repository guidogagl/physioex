from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from physioex.train.networks.base import SleepModule

module_config = dict()


class FeatureExtractor(nn.Module):
    def __init__(self, config=module_config):
        super(FeatureExtractor, self).__init__()
        self.config = config
        self.padding_edf = {
            "conv1": (22, 22),
            "max_pool1": (2, 2),
            "conv2": (3, 4),
            "max_pool2": (0, 1),
        }
        first_filter_size = int(self.config["sf"] / 2.0)
        first_filter_stride = int(self.config["sf"] / 16.0)

        self.cnn = nn.Sequential(
            self._conv_block(
                self.config["in_channels"],
                128,
                first_filter_size,
                first_filter_stride,
                self.padding_edf["conv1"],
            ),
            nn.ConstantPad1d(self.padding_edf["max_pool1"], 0),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            self._conv_block(128, 128, 8, 1, self.padding_edf["conv2"]),
            self._conv_block(128, 128, 8, 1, self.padding_edf["conv2"]),
            self._conv_block(128, 128, 8, 1, self.padding_edf["conv2"]),
            nn.ConstantPad1d(self.padding_edf["max_pool2"], 0),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConstantPad1d(padding, 0),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False),
            nn.BatchNorm1d(num_features=out_channels, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.cnn(x)


class Classifier(nn.Module):
    def __init__(self, config=module_config):
        super(Classifier, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(
            input_size=2048,
            hidden_size=self.config["n_rnn_units"],
            num_layers=1,
            batch_first=True,
        )
        self.rnn_dropout = nn.Dropout(p=0.5)
        self.clf = nn.Linear(self.config["n_rnn_units"], self.config["n_classes"])

    def forward(self, x):
        x = self.encode(x)

        batch_size, sequence_length, latent_dim = x.size()
        x = x.reshape(batch_size * sequence_length, latent_dim)

        x = self.clf(x)
        return x.reshape(batch_size, sequence_length, -1)

    def encode(self, x):
        batch_size, sequence_length, feature_size = x.size()
        x, _ = self.rnn(x)
        x = x.reshape(-1, self.config["n_rnn_units"])

        x = self.rnn_dropout(x)
        return x.reshape(batch_size, sequence_length, -1)


class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()

        self.feature_extractor = FeatureExtractor(module_config)
        self.clf = Classifier(module_config)

    def encode(self, x):
        batch_size, seqlen, inchan, insamp = x.size()

        x = x.reshape(-1, inchan, insamp)

        x = self.feature_extractor(x)

        x = x.reshape(batch_size, seqlen, -1)

        x = self.clf.encode(x)

        batch_size, sequence_length, rnn_units = x.size()
        y = x.reshape(batch_size * sequence_length, rnn_units)

        y = self.clf.clf(y)
        y = y.reshape(batch_size, sequence_length, -1)
        return x, y

    def forward(self, x):
        x, y = self.encode(x)

        return y


class TinySleepNet(SleepModule):
    def __init__(self, module_config=module_config):

        module_config["n_rnn_units"] = 128
        module_config["n_rnn_layers"] = 1

        super(TinySleepNet, self).__init__(
            Net(module_config=module_config),
            module_config,
        )
