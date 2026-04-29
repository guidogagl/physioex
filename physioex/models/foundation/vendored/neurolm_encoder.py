"""Vendored NeuroLM NeuralTransformer encoder.

Source: NeuroLM (Liu et al., 2024)
Only the VQ encoder (NeuralTransformer) is kept here — the decoder,
codebook, and instruction-tuning GPT2 head are discarded.

Licence: see the NeuroLM repository for licence terms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_tensor):
        return F.layer_norm(
            input_tensor, self.weight.shape, self.weight, self.bias, 1e-5
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, mask=None):
        batch, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head
        k = k.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        q = q.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        if mask is None:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_chans: int = 1, out_chans: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7)
        )
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()
        self.l = nn.Sequential(nn.Linear(400, 768), nn.GELU())

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, "b c n t -> b n (t c)")
        return self.l(x)


@dataclass
class NTConfig:
    block_size: int = 1024
    patch_size: int = 200
    num_classes: int = 0
    in_chans: int = 1
    out_chans: int = 16
    use_mean_pooling: bool = True
    init_scale: float = 0.001
    n_layer: int = 12
    n_head: int = 10
    n_embd: int = 400
    dropout: float = 0.0
    bias: bool = False


class NeuralTransformer(nn.Module):
    def __init__(self, config: NTConfig):
        super().__init__()
        self.patch_embed = (
            TemporalConv(out_chans=config.out_chans)
            if config.in_chans == 1
            else nn.Linear(config.in_chans, config.n_embd)
        )
        self.pos_embed = nn.Embedding(256, config.n_embd)
        self.time_embed = nn.Embedding(64, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = (
            nn.Identity()
            if config.use_mean_pooling
            else nn.LayerNorm(config.n_embd, eps=1e-6)
        )
        self.fc_norm = (
            nn.LayerNorm(config.n_embd, eps=1e-6) if config.use_mean_pooling else None
        )
        self.head = (
            nn.Linear(config.n_embd, config.num_classes)
            if config.num_classes > 0
            else nn.Identity()
        )
        self.pos_drop = nn.Dropout(p=config.dropout)
        self.apply(self._init_weights)
        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(config.init_scale)
            self.head.bias.data.mul_(config.init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.c_proj.weight.data, layer_id + 1)
            rescale(layer.mlp.c_proj.weight.data, layer_id + 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(
        self, x, input_chans=None, input_times=None, mask=None, return_all_tokens=False
    ):
        x = self.patch_embed(x)
        x = x + self.pos_embed(input_chans)
        x = x + self.time_embed(input_times)
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x) if return_all_tokens else self.fc_norm(x.mean(1))
        return x
