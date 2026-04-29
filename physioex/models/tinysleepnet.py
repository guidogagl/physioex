import torch
from torch import nn
from torchinfo import summary

from physioex.train.trainer import Trainer
from physioex.data.dataset import PhysioExDataset


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_chan: int = 3,
        conv1=(22, 22),
        max_pool1=(2, 2),
        conv2=(3, 4),
        max_pool2=(0, 1),
        sf: int = 100,
    ):
        super(FeatureExtractor, self).__init__()
        self.padding_edf = {
            "conv1": conv1,
            "max_pool1": max_pool1,
            "conv2": conv2,
            "max_pool2": max_pool2,
        }

        first_filter_size = int(sf / 2.0)
        first_filter_stride = int(sf / 16.0)

        self.cnn = nn.Sequential(
            self._conv_block(
                in_chan,
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
    def __init__(
        self, n_classes: int = 5, n_rnn_units: int = 128, n_rnn_layers: int = 1
    ):
        super(Classifier, self).__init__()

        self.n_rnn_units = n_rnn_units

        self.rnn = nn.LSTM(
            input_size=2048,
            hidden_size=n_rnn_units,
            num_layers=n_rnn_layers,
            batch_first=True,
        )
        self.rnn_dropout = nn.Dropout(p=0.5)
        self.clf = nn.Linear(n_rnn_units, n_classes)

    def forward(self, x):
        x = self.encode(x)

        batch_size, sequence_length, latent_dim = x.size()
        x = x.reshape(batch_size * sequence_length, latent_dim)

        x = self.clf(x)
        return x.reshape(batch_size, sequence_length, -1)

    def encode(self, x):
        batch_size, sequence_length, feature_size = x.size()
        x, _ = self.rnn(x)
        x = x.reshape(-1, self.n_rnn_units)

        x = self.rnn_dropout(x)
        return x.reshape(batch_size, sequence_length, -1)


class TinySleepNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 5,
        in_chan: int = 3,
        conv1=(22, 22),
        max_pool1=(2, 2),
        conv2=(3, 4),
        max_pool2=(0, 1),
        sf: int = 100,
        n_rnn_units: int = 128,
        n_rnn_layers: int = 1,
    ):
        super().__init__()

        self.feature_extractor = FeatureExtractor(
            in_chan=in_chan,
            conv1=conv1,
            max_pool1=max_pool1,
            conv2=conv2,
            max_pool2=max_pool2,
            sf=sf,
        )
        self.clf = Classifier(
            n_classes=n_classes, n_rnn_units=n_rnn_units, n_rnn_layers=n_rnn_layers
        )

    def forward(self, x):
        batch_size, seqlen, inchan, insamp = x.size()

        x = x.reshape(-1, inchan, insamp)

        x = self.feature_extractor(x)

        x = x.reshape(batch_size, seqlen, -1)

        x = self.clf.encode(x)

        batch_size, sequence_length, rnn_units = x.size()
        y = x.reshape(batch_size * sequence_length, rnn_units)

        y = self.clf.clf(y)
        y = y.reshape(batch_size, sequence_length, -1)

        return y


if __name__ == "__main__":

    dataset = PhysioExDataset(datasets=["sleepedf"])

    model = TinySleepNet(
        in_chan=dataset.get_num_channels(),
    )

    print("TinySleepNet summary:")

    summary(model, (32, 21, dataset.get_num_channels(), 3000))

    print("\nTraining TinySleepNet...")

    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=20,
        lr=1e-3,
    )

    print("\nEvaluating TinySleepNet...")

    results = Trainer.evaluate(
        model=model,
        dataset=dataset,
    )

    print(results)
