import torch
import torch.nn as nn

import torch.nn.functional as F


class HardAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_size: int,
        N: int = 1,  # number of elements to select
        temperature: float = 1.0,
    ):
        super(HardAttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.temperature = temperature

        self.N = N

        self.LSTM = nn.LSTM(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )

        self.W = nn.Linear(2 * hidden_size, attention_size)
        self.u = nn.Linear(attention_size, self.N, bias=False)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.size()

        # the layer need to choose attention_size elements from the sequence
        # depending on the temperature and the input elements

        # first we need to give the weights some sequetial context
        x_seq, _ = self.LSTM(x)

        x_seq = x_seq.reshape(-1, 2 * hidden_size)

        v = torch.tanh(self.W(x_seq))  # batch_size * sequence_length, attention_size
        vu = self.u(v)  # batch_size * sequence_length, N
        vu = torch.exp(vu).reshape(batch_size, sequence_length, self.N)
        vu = torch.permute(vu, (0, 2, 1))  # (batch_size, N, sequence_length)

        logits = vu / torch.sum(
            vu, dim=-1, keepdim=True
        )  # (batch_size, N, sequence_length)

        # logits is the probability distribution over the sequence
        # Reshape to apply the Gumbel-Softmax trick
        logits = logits.reshape(batch_size * self.N, sequence_length)

        # apply the Gumbel-Softmax trick
        alphas = F.gumbel_softmax(
            logits, tau=self.temperature, hard=True
        )  # (batch_size * N, sequence_length)

        # reshape to obtain the attention mask composed by 0 or 1 values
        alphas = alphas.reshape(batch_size, self.N, sequence_length)

        # select N elements from the sequence by multiplying the alphas
        x = torch.bmm(alphas, x)  # (batch_size, N, hidden_size)

        return x


class SoftAttentionLayer(nn.Module):
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
