from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
from physioex.train.networks.base import SeqtoSeq

module_config = dict()


class FeatureExtractor( nn.Module ):
    def __init__(self, config = module_config):
        super(FeatureExtractor, self).__init__()
        self.config = config
        self.padding_edf = {  # same padding in tensorflow
            "conv1": (22, 22),
            "max_pool1": (2, 2),
            "conv2": (3, 4),
            "max_pool2": (0, 1),
        }
        first_filter_size = int(
            self.config["sampling_rate"] / 2.0
        )  # 100/2 = 50, 与以往使用的Resnet相比，这里的卷积核更大
        first_filter_stride = int(
            self.config["sampling_rate"] / 16.0
        )  # todo 与论文不同，论文给出的stride是100/4=25
        self.cnn = nn.Sequential(
            nn.ConstantPad1d(self.padding_edf["conv1"], 0),  # conv1
            nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv1",
                            nn.Conv1d(
                                in_channels= self.config["n_channels"],
                                out_channels=128,
                                kernel_size=first_filter_size,
                                stride=first_filter_stride,
                                bias=False,
                            ),
                        )
                    ]
                )
            ),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf["max_pool1"], 0),  # max p 1
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            nn.ConstantPad1d(self.padding_edf["conv2"], 0),  # conv2
            nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv2",
                            nn.Conv1d(
                                in_channels=128,
                                out_channels=128,
                                kernel_size=8,
                                stride=1,
                                bias=False,
                            ),
                        )
                    ]
                )
            ),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf["conv2"], 0),  # conv3
            nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv3",
                            nn.Conv1d(
                                in_channels=128,
                                out_channels=128,
                                kernel_size=8,
                                stride=1,
                                bias=False,
                            ),
                        )
                    ]
                )
            ),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf["conv2"], 0),  # conv4
            nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv4",
                            nn.Conv1d(
                                in_channels=128,
                                out_channels=128,
                                kernel_size=8,
                                stride=1,
                                bias=False,
                            ),
                        )
                    ]
                )
            ),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf["max_pool2"], 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        return self.cnn(x)

class Classifier(nn.Module):
    def __init__(self, config = module_config):
        super(Classifier, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(
            input_size=2048,
            hidden_size=self.config["n_rnn_units"],
            num_layers=1,
            batch_first=True,
        )
        self.rnn_dropout = nn.Dropout(p=0.5)  # todo 是否需要这个dropout?

        self.proj = nn.Linear( self.config["n_rnn_units"], self.config["latent_space_dim"])
        self.norm = nn.LayerNorm( self.config["latent_space_dim"] )
        self.clf = nn.Linear( self.config["latent_space_dim"], self.config["n_classes"] )

    def forward(self, x):
        x = self.encode(x)
        
        batch_size, seq_len, latent_dim = x.size()
        x = x.reshape(batch_size * seq_len, latent_dim)

        x = self.clf(x)
        return x.reshape(batch_size, seq_len, -1)
    
    def encode(self, x):
        batch_size, seq_len, feature_size = x.size()
        x, _ = self.rnn(x)
        x = x.reshape(-1, self.config["n_rnn_units"])

        x = self.rnn_dropout(x)
        x = nn.ReLU()(self.proj(x))
        x = self.norm(x)

        return x.reshape(batch_size, seq_len, -1)
        
class TinySleepNet( SeqtoSeq ):
    def __init__(self, module_config = module_config):
        super(TinySleepNet, self).__init__(FeatureExtractor(config=module_config), Classifier(config=module_config), module_config)

    
