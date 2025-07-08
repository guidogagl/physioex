import torch 
import torch.nn as nn

class ChannelsDropout(nn.Module):
    def __init__(self, dropout_prob=1.0):
        super().__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x, channel_acc):
        if not self.training or self.dropout_prob == 0.0:
            return x

        # x: (batch, nchan, hdim)
        batch, nchan, hdim = x.shape
        device = x.device

        # channel_acc: (nchan,) - accuratezza per canale (valori tra 0 e 1)
        acc = torch.tensor(channel_acc, device=device, dtype=torch.float32)
        proba = 1.0 - acc
        proba = proba / proba.sum()  # normalizzazione lineare

        # Applica lo shuffle solo a una frazione dei batch (dropout_prob)
        mask = torch.rand(batch, device=device) < self.dropout_prob

        # Per ogni elemento nel batch, campiona nchan indici secondo proba
        idx = torch.multinomial(proba.expand(batch, -1), nchan, replacement=True)  # (batch, nchan)
        batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, nchan)

        x_shuffled = x.clone()
        x_shuffled[mask] = x[batch_idx[mask], idx[mask], :]

        return x_shuffled


class TimeMasking(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        L: int = 29,
        temperature: float = 0.1,
    ):
        super(TimeMasking, self).__init__()
        self.temperature = temperature
        
        windows = []
        for n in range(1, (L // 2) + 2, 2):  # n = window length
            for start in range(L - n + 1):
                w = torch.zeros(L)
                w[start : start + n] = 1.0
                windows.append(w)
        windows = torch.stack(windows, dim=0)  # (num_windows, L)
        self.register_buffer("windows", windows)
        self.num_windows = windows.size(0)
        
        self.W = nn.Linear(hidden_size, self.num_windows, bias=False)

        W_omega = torch.zeros((self.num_windows, self.num_windows), dtype=torch.float32)
        b_omega = torch.zeros((self.num_windows), dtype=torch.float32)
        u_omega = torch.zeros((self.num_windows), dtype=torch.float32)

        self.W_omega = nn.Parameter(W_omega)
        self.b_omega = nn.Parameter(b_omega)
        self.u_omega = nn.Parameter(u_omega)

        nn.init.normal_(self.W_omega, std=0.1)
        nn.init.normal_(self.b_omega, std=0.1)
        nn.init.normal_(self.u_omega, std=0.1)
        
    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.size()
        
        w = self.W( x ) # batch, seq, num_windows
        
        v = torch.tanh(
            torch.matmul(
                torch.reshape(w, [batch_size * sequence_length, self.num_windows ]),
                self.W_omega,
            )
            + torch.reshape(self.b_omega, [1, -1])
        ) # 

        
        vu = torch.matmul(v, torch.reshape(self.u_omega, [-1, 1])) # (batch_size * num_windows, 1)
        exps = torch.reshape(torch.exp(vu), [-1, sequence_length])
        alphas = exps / torch.reshape(torch.sum(exps, 1), [-1, 1])

        alphas = alphas.reshape( batch_size, sequence_length)

        w = torch.einsum( "bs, bsl -> bl", alphas, w )  # batch, num_windows
        exps = torch.exp(w)
        alphas = exps / torch.reshape(torch.sum(exps, dim=1), [batch_size, 1])
        alphas = alphas.reshape(batch_size, self.num_windows)
        
        alphas = torch.nn.functional.gumbel_softmax(
            alphas, tau=self.temperature, hard=True
        )

        mask = torch.einsum("bs,sl -> bl", alphas, self.windows)
        
        # input masking
        x = torch.einsum("bs, bsl -> bl", mask, x)  / torch.sum(mask, dim=-1, keepdim=True)
        
        return x, mask

class ChannelSampler(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_size: int = 32,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature

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

        alphas = alphas.reshape( batch_size, sequence_length)

        alphas = torch.nn.functional.gumbel_softmax(
            alphas, tau=self.temperature, hard=True
        )
        
        x = torch.einsum("bs, bsh -> bh", alphas, x)

        if r_alphas:
            return x, alphas
        return x
