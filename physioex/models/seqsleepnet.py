import torch
from torch import nn
from torchinfo import summary

from physioex.train.trainer import Trainer
from physioex.data.dataset import PhysioExDataset


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


class SeqSleepNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 5,
        in_chan: int = 3,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        lowfreq: int = 0,
        highfreq: int = 50,
        fs: int = 100,
        seqnhidden1: int = 64,
        seqnlayer1: int = 4,
        attentionsize: int = 32,
        seqnhidden2: int = 64,
        seqnlayer2: int = 4,
    ):
        super().__init__()

        self.filterbank = LearnableFilterbank(
            in_chan, F, D, nfft, fs, lowfreq, highfreq
        )
        self.seqn1 = nn.LSTM(
            D * in_chan,
            seqnhidden1,
            num_layers=seqnlayer1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = AttentionLayer(2 * seqnhidden1, attentionsize)

        self.seqn2 = nn.GRU(
            2 * seqnhidden1,
            seqnhidden2,
            num_layers=seqnlayer2,
            batch_first=True,
            bidirectional=True,
        )

        self.clf = nn.Linear(2 * seqnhidden2, n_classes)

    def forward(self, x):

        batch_size, L, in_chans, T, F = x.size()

        x = x.reshape(batch_size * L, in_chans, T, F)
        x = self.filterbank(x)
        x = x.permute(0, 2, 1, 3)  # shape ( batch_size*L, T, in_chans, D )
        x = x.reshape(batch_size * L, T, -1)

        x, _ = self.seqn1(x)
        x = self.attention(x)

        x = x.reshape(batch_size, L, -1)

        x, _ = self.seqn2(x)

        x = x.reshape(batch_size * L, -1)
        x = self.clf(x)

        return x.reshape(batch_size, L, -1)


if __name__ == "__main__":

    dataset = PhysioExDataset(datasets=["sleepedf"])

    model = SeqSleepNet(
        in_chan=dataset.get_num_channels(),
    )

    print("SeqSleepNet summary:")

    summary(model, (32, 21, dataset.get_num_channels(), 29, 129))

    print("\nTraining SeqSleepNet...")

    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=10,
        lr=1e-5,
    )

    print("\nEvaluating SeqSleepNet...")

    results = Trainer.evaluate(
        model=model,
        dataset=dataset,
    )

    print(results)
