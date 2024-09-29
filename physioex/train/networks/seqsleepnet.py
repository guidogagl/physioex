from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from physioex.train.networks.base import SleepModule

module_config = dict()


class SeqSleepNet(SleepModule):
    def __init__(self, module_config=module_config):

        module_config.update(
            {
                "T": 29,
                "F": 129,
                "D": 32,
                "nfft": 256,
                "lowfreq": 0,
                "highfreq": 50,
                "seqnhidden1": 64,
                "seqnlayer1": 1,
                "attentionsize1": 32,
                "seqnhidden2": 64,
                "seqnlayer2": 1,
            }
        )

        super(SeqSleepNet, self).__init__(Net(module_config), module_config)


class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()

        self.epoch_encoder = EpochEncoder(module_config)
        self.sequence_encoder = SequenceEncoder(module_config)

    def encode(self, x):

        batch, L, nchan, T, F = x.size()

        x = x.reshape(-1, nchan, T, F)

        x = self.epoch_encoder(x)

        x = x.reshape(batch, L, -1)

        x = self.sequence_encoder.encode(x)

        y = self.sequence_encoder.clf(x)

        return x, y

    def forward(self, x):
        x, y = self.encode(x)
        return y


class EpochEncoder(nn.Module):
    def __init__(self, module_config):
        super(EpochEncoder, self).__init__()
        self.F2_filtbank = LearnableFilterbank(
            module_config["in_channels"],
            module_config["F"],
            module_config["D"],
            module_config["nfft"],
            module_config["sf"],
            module_config["lowfreq"],
            module_config["highfreq"],
        )
        self.F2_birnn = nn.LSTM(
            input_size=module_config["D"] * module_config["in_channels"],
            hidden_size=module_config["seqnhidden1"],
            num_layers=module_config["seqnlayer1"],
            batch_first=True,
            bidirectional=True,
        )
        self.F2_attention = AttentionLayer(
            2 * module_config["seqnhidden1"], module_config["attentionsize1"]
        )
        self.attentionsize1 = module_config["attentionsize1"]
        self.T = module_config["T"]
        self.D = module_config["D"]
        self.F = module_config["F"]
        self.in_chans = module_config["in_channels"]
        return

    def forward(self, x):
        batch_size, in_chans, T, F = x.size()

        assert (
            in_chans == self.in_chans
        ), f"channels dimension mismatch, provided input size: {str(x.size())}"
        assert (
            T == self.T
        ), f"time dimension mismatch, provided input size: {str(x.size())}"
        assert (
            F == self.F
        ), f"frequency dimension mismatch, provided input size: {str(x.size())}"

        x = self.F2_filtbank(x)

        x = torch.reshape(x, (batch_size, self.T, self.D * self.in_chans))
        x, _ = self.F2_birnn(x)
        x = self.F2_attention(x)

        return x


class SequenceEncoder(nn.Module):
    def __init__(self, module_config):
        super(SequenceEncoder, self).__init__()

        self.LSTM = nn.GRU(
            input_size=2 * module_config["seqnhidden1"],
            hidden_size=module_config["seqnhidden2"],
            num_layers=module_config["seqnlayer2"],
            batch_first=True,
            bidirectional=True,
        )

        self.clf = nn.Linear(
            module_config["seqnhidden2"] * 2, module_config["n_classes"]
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.lin(x)
        return x

    def encode(self, x):
        x, _ = self.LSTM(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        attention_size: int = 32,
        time_major: bool = False,
    ):
        super().__init__()

        W_omega = torch.zeros((hidden_size, attention_size), dtype=torch.float32)
        b_omega = torch.zeros((attention_size), dtype=torch.float32)
        u_omega = torch.zeros((attention_size), dtype=torch.float32)

        self.W_omega = nn.Parameter(W_omega)
        self.b_omega = nn.Parameter(b_omega)
        self.u_omega = nn.Parameter(u_omega)

        nn.init.normal_(self.W_omega, std=0.1)
        nn.init.normal_(self.b_omega, std=0.1)
        nn.init.normal_(self.u_omega, std=0.1)

    def forward(self, x, r_alphas=False):
        batch_size, sequence_length, hidden_size = x.size()

        v = torch.tanh(
            torch.matmul(
                torch.reshape(x, [batch_size * sequence_length, hidden_size]),
                self.W_omega,
            )
            + torch.reshape(self.b_omega, [1, -1])
        )
        vu = torch.matmul(v, torch.reshape(self.u_omega, [-1, 1]))
        exps = torch.reshape(torch.exp(vu), [-1, sequence_length])
        alphas = exps / torch.reshape(torch.sum(exps, 1), [-1, 1])

        output = torch.sum(
            x * torch.reshape(alphas, [batch_size, sequence_length, 1]), 1
        )
        if r_alphas:
            return output, alphas
        return output


class LearnableFilterbank(nn.Module):
    def __init__(
        self,
        in_chan: int = 2,
        F: int = 129,
        nfilt: int = 32,
        nfft: int = 256,
        sf: int = 100,
        lowfreq: int = 0,
        highfreq: int = 50,
    ):
        super().__init__()
        self.F, self.D = F, nfilt

        S = torch.zeros((in_chan, F, nfilt), dtype=torch.float32)

        for i in range(in_chan):
            S[i] = self.lin_tri_filter_shape(nfilt, nfft, sf, lowfreq, highfreq)

        W = torch.zeros((in_chan, F, nfilt), dtype=torch.float32)

        self.W = nn.Parameter(W, requires_grad=True)
        self.S = nn.Parameter(S, requires_grad=False)

        nn.init.normal_(self.W)

    def forward(self, x):
        Wfb = torch.mul(torch.sigmoid(self.W), self.S)

        return torch.matmul(x, Wfb)

    def lin_tri_filter_shape(
        self, nfilt=20, nfft=512, sf=16000, lowfreq=0, highfreq=None
    ):
        highfreq = highfreq or sf / 2
        assert highfreq <= sf / 2, "highfreq is greater than sf/2"

        hzpoints = torch.linspace(lowfreq, highfreq, nfilt + 2)
        bin = torch.floor((nfft + 1) * hzpoints / sf)

        fbank = torch.zeros([nfilt, nfft // 2 + 1])
        for j in range(0, nfilt):
            for i in range(int(bin[j]), int(bin[j + 1])):
                fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
            for i in range(int(bin[j + 1]), int(bin[j + 2])):
                fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
        fbank = torch.transpose(fbank, 0, 1)
        return fbank.float()
