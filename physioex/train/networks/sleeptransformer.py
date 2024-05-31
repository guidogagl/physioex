import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from physioex.train.networks.base import SleepModule

module_config = dict()


class SleepTransformer(SleepModule):
    def __init__(self, module_config=module_config):
        super(SleepTransformer, self).__init__(Net(module_config), module_config)


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        config = SimpleNamespace(**config)

        self.config = config
        self.frame_transformer = Transformer_Encoder(
            d_model=config.F * config.in_channels,
            d_ff=config.frm_d_ff,
            num_blocks=config.frm_num_blocks,
            num_heads=config.frm_num_heads,
            maxlen=config.T,
            fc_dropout_rate=config.frm_fc_dropout,
            attention_dropout_rate=config.frm_attention_dropout,
            smoothing=config.frm_smoothing,
        )

        self.frame_attention = Attention(
            config.F * config.in_channels, config.attention_size
        )

        self.seq_transformer = Transformer_Encoder(
            d_model=config.F * config.in_channels,
            d_ff=config.seq_d_ff,
            num_blocks=config.seq_num_blocks,
            num_heads=config.seq_num_heads,
            maxlen=config.seq_len,
            fc_dropout_rate=config.seq_fc_dropout,
            attention_dropout_rate=config.seq_attention_dropout,
            smoothing=config.seq_smoothing,
        )

        self.fc1 = nn.Linear(config.F * config.in_channels, config.fc_hidden_size)
        self.fc2 = nn.Linear(config.fc_hidden_size, config.fc_hidden_size)
        self.fc3 = nn.Linear(config.fc_hidden_size, config.n_classes)

    def encode(self, x):
        x = x[..., : self.config.F]
        batch_size, L, nchan, T, f = x.size()

        x = x.permute(0, 1, 3, 2, 4)  # batch, L, T, nchan, F
        x = x.view(batch_size * L, T, nchan * f)

        x = self.frame_transformer(x)
        x, _ = self.frame_attention(x)

        x = x.view(batch_size, L, nchan * f)

        x = self.seq_transformer(x)

        x = x.view(batch_size * L, nchan * f)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.config.fc_dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.config.fc_dropout, training=self.training)
        y = self.fc3(x)

        return x.view(batch_size, L, -1), y.view(batch_size, L, -1)

    def forward(self, x):
        _, y = self.encode(x)
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create a positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension for the batch size and register the positional encoding matrix as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # Add positional encoding to the word embeddings

        x = x + self.pe
        return x


class Transformer_Encoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        num_blocks,
        num_heads,
        maxlen,
        fc_dropout_rate,
        attention_dropout_rate,
        smoothing,
    ):
        super(Transformer_Encoder, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.fc_dropout_rate = fc_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.smoothing = smoothing

        self.pos_encoder = PositionalEncoding(d_model, maxlen)
        encoder_layers = TransformerEncoderLayer(
            d_model, num_heads, d_ff, fc_dropout_rate
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_blocks)

    def forward(self, x):
        x *= self.d_model**0.5  # scale
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        self.W_omega = nn.Parameter(torch.randn(hidden_size, attention_size))
        self.b_omega = nn.Parameter(torch.randn(attention_size))
        self.u_omega = nn.Parameter(torch.randn(attention_size))

    def forward(self, inputs, time_major=False):
        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = inputs.transpose(0, 1)

        sequence_length = inputs.size(1)
        hidden_size = inputs.size(2)

        # Attention mechanism
        v = torch.tanh(
            torch.matmul(inputs.view(-1, hidden_size), self.W_omega) + self.b_omega
        )
        vu = torch.matmul(v, self.u_omega)
        exps = torch.exp(vu).view(-1, sequence_length)
        alphas = exps / torch.sum(exps, dim=1, keepdim=True)

        # Output of (Bi-)RNN is reduced with attention vector
        output = torch.sum(inputs * alphas.unsqueeze(-1), dim=1)

        return output, alphas
