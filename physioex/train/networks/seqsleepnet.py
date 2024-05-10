from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from physioex.train.networks.base import SleepModule


module_config = dict()


import torch
import torch.nn as nn

from sklearn.metrics import cohen_kappa_score
from physioex.train.networks.utils.filterbank_shape import FilterbankShape

module_config = {
    "L": 21,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "dropOutProb": 0.2,
    "nChannels": 1,
    "use_scheduler": "ReduceLROnPlateau",  # ["ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR"],
}



class SeqSleepNet(SleepModule):
    def __init__(self, module_config):
        super(SeqSleepNet, self).__init__(Net(module_config), module_config)

    
    
class Net(nn.Module):
    def __init__(self, config = module_config):
        super().__init__()

        # settings:
        self.L = config["seq_len"]  # sequence length
        self.nChan = config["in_channels"]

        self.nHidden = 64
        self.nFilter = 32
        self.attentionSize = 64
        self.dropOutProb = config["dropOutProb"]
        self.timeBins = 29

        # ---------------------------filterbank:--------------------------------
        filtershape = FilterbankShape()

        # triangular filterbank shape
        shape = torch.tensor(
            filtershape.lin_tri_filter_shape(
                nfilt=self.nFilter, nfft=256, samplerate=100, lowfreq=0, highfreq=50
            ),
            dtype=torch.float,
        )
        self.Wbl = nn.Parameter(shape, requires_grad=False)
        # filter weights:
        self.Weeg = nn.Parameter(torch.randn(self.nFilter, self.nChan))
        # ----------------------------------------------------------------------

        self.epochrnn = nn.GRU(
            self.nFilter, self.nHidden, 1, bidirectional=True, batch_first=True
        )

        # attention-layer:
        self.attweight_w = nn.Parameter(
            torch.randn(2 * self.nHidden, self.attentionSize)
        )
        self.attweight_b = nn.Parameter(torch.randn(self.attentionSize))
        self.attweight_u = nn.Parameter(torch.randn(self.attentionSize))

        # epoch sequence block:
        self.seqDropout = torch.nn.Dropout(self.dropOutProb, inplace=False)
        self.seqRnn = nn.GRU(
            self.nHidden * 2, self.nHidden, 1, bidirectional=True, batch_first=True
        )

        # output:
        self.fc = nn.Linear(2 * self.nHidden, 5)

    def epoch_encoder(self, x):
        assert (
            x.shape[1] == self.timeBins
        ), f"Shape: {x.shape}, expected (B, {self.timeBins}, 129, {self.nChan})"
        assert (
            x.shape[2] == 129
        ), f"Shape: {x.shape}, expected (B, {self.timeBins}, 129, {self.nChan})"
        assert (
            x.shape[3] == self.nChan
        ), f"Shape: {x.shape}, expected (B, {self.timeBins}, 129, {self.nChan})"

        x = x.permute([0, 3, 1, 2])

        Wfb = torch.multiply(torch.sigmoid(self.Weeg[:, 0]), self.Wbl)
        x = torch.matmul(x, Wfb)  # filtering
        x = torch.reshape(x, [-1, self.timeBins, self.nFilter])

        # biGRU:
        x, hn = self.epochrnn(x)
        x = self.seqDropout(x)

        # attention:
        v = torch.tanh(
            torch.matmul(torch.reshape(x, [-1, self.nHidden * 2]), self.attweight_w)
            + torch.reshape(self.attweight_b, [1, -1])
        )
        vu = torch.matmul(v, torch.reshape(self.attweight_u, [-1, 1]))
        exps = torch.reshape(torch.exp(vu), [-1, self.timeBins])
        alphas = exps / torch.reshape(torch.sum(exps, 1), [-1, 1])
        x = torch.sum(x * torch.reshape(alphas, [-1, self.timeBins, 1]), 1)
        return x

    def sequence_encoder(self, x):

        assert (
            x.shape[1] == self.L
        ), f"Shape: {x.shape}, expected (B, {self.L}, { self.nHidden * 2})"
        assert (
            x.shape[2] == self.nHidden * 2
        ), f"Shape: {x.shape}, expected (B, {self.L}, { self.nHidden * 2})"

        x, hn = self.seqRnn(x)
        x = self.seqDropout(x)

        return x

    def encode(self, x):
        # permute x in the correct dimension, _, L, n_chan, 29, 129 ->  _, L, 29, 129, nchan 
        
        x = x.permute(0, 2, 3, 4, 1)
        
        assert (
            x.shape[1] == self.L
        ), f"Shape: {x.shape}, expected (B, {self.L}, {self.timeBins}, 129, {self.nChan})"
        assert (
            x.shape[2] == self.timeBins
        ), f"Shape: {x.shape}, expected (B, {self.L}, {self.timeBins}, 129, {self.nChan})"
        assert (
            x.shape[3] == 129
        ), f"Shape: {x.shape}, expected (B, {self.L}, {self.timeBins}, 129, {self.nChan})"
        assert (
            x.shape[4] == self.nChan
        ), f"Shape: {x.shape}, expected (B, {self.L}, {self.timeBins}, 129, {self.nChan})"

        # from sequences to epochs:

        x = x.reshape(-1, self.timeBins, 129, self.nChan)

        x = self.epoch_encoder(x)

        # from epochs to sequences
        x = x.reshape(-1, self.L, self.nHidden * 2)

        x = self.sequence_encoder(x)

        y = x.reshape(-1, 2 * self.nHidden)
        y = self.fc(y)
        y = y.reshape(-1, self.L, 5)
        return x, y

    def forward(self, x):
        x, y = self.encode(x)
        return y

