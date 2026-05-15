"""NeuroLM — Neural Language Model (Liu et al., 2024).

Patch-based tokenization at 200 Hz. Channels mapped to VQ vocabulary indices.
Embeddings: masked mean over valid tokens -> (B, 768).

Preprocessing (pure PyTorch, differentiable):
  1. x / 100 scaling
  2. Strip all-zero channels
  3. Tokenize: patchify signal + map channels to VQ vocabulary indices

Vendored from: NeuroLM (Liu et al., 2024)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from physioex.models.foundation_base import FoundationEncoder
from physioex.models.foundation_preproc import scale_div, strip_zero_channels

logger = logging.getLogger("physioex.foundation")

_SCALE_DIVISOR = 100.0

_STANDARD_1020 = [
    "FP1", "FPZ", "FP2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2",
    "AF4", "AF6", "AF8", "AF10", "F9", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "F10", "FT9", "FT7", "FC5", "FC3", "FC1",
    "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10", "T9", "T7", "C5", "C3",
    "C1", "CZ", "C2", "C4", "C6", "T8", "T10", "TP9", "TP7", "CP5",
    "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10", "P9", "P7",
    "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10", "PO9", "PO7",
    "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10", "O1",
    "OZ", "O2", "O9", "CB1", "CB2", "IZ", "O10", "T3", "T5", "T4", "T6",
    "M1", "M2", "A1", "A2", "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6",
    "CFC7", "CFC8", "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7",
    "CCP8", "T1", "T2", "FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h",
    "TPP10h", "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8",
    "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4",
    "F4-C4", "C4-P4", "P4-O2", "pad", "I1", "I2",
]

_PAD_IDX = _STANDARD_1020.index("pad")

_BIPOLAR_TO_STANDARD = {
    "C3-A2": "C3", "C4-A1": "C4", "O1-A2": "O1", "O2-A1": "O2",
    "F3-A2": "F3", "F4-A1": "F4", "FZ-CZ": "FZ", "CZ-PZ": "PZ",
}


def _channel_idx(name: str) -> int:
    upper = name.upper()
    if upper in _STANDARD_1020:
        return _STANDARD_1020.index(upper)
    std = _BIPOLAR_TO_STANDARD.get(name, name).upper()
    if std in _STANDARD_1020:
        return _STANDARD_1020.index(std)
    if name in _STANDARD_1020:
        return _STANDARD_1020.index(name)
    raise ValueError(
        f"NeuroLM: channel label {name!r} not in VQ vocabulary. "
        f"Known aliases: {sorted(_BIPOLAR_TO_STANDARD)!r}."
    )


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
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
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


def _extract_encoder_state(state: dict) -> tuple:
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    prefixes = ("_orig_mod.VQ.encoder.", "VQ.encoder.")
    extracted = {}
    used_prefix = None
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix) :]] = value
                used_prefix = prefix
                break
    if not extracted:
        raise KeyError("Could not find NeuroLM VQ encoder weights in checkpoint.")
    return extracted, used_prefix


class NeuroLMEncoder(FoundationEncoder):
    """NeuroLM with patch-based tokenization and VQ channel vocabulary."""

    MODEL_NAME = "neurolm"
    PIPELINE_PRESET = "neurolm"
    CHANNEL_STRATEGY = "all"

    def __init__(
        self,
        in_chan: int,
        checkpoint_path: str | None = None,
        channel_names=None,
        channel_map=None,
        **kwargs,
    ):
        from physioex.models.foundation_checkpoints import ensure_checkpoint

        self._channel_names = channel_names or []
        self._channel_map = channel_map or {}
        resolved = ensure_checkpoint("neurolm", checkpoint_path)
        self._ckpt_path = resolved
        self._resolved_names = (
            [self._channel_map.get(ch, ch) for ch in self._channel_names]
            if self._channel_names
            else []
        )

        ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
        self._encoder_conf = NTConfig(**ckpt["encoder_args"])
        self._encoder_state, _ = _extract_encoder_state(ckpt)

        super().__init__(
            in_chan=in_chan,
            checkpoint_path=resolved,
            **kwargs,
        )

    def _build_encoder(self, **kwargs) -> nn.Module:
        return NeuralTransformer(self._encoder_conf)

    def _get_embedding_dim(self) -> int:
        return 768

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        self.encoder.load_state_dict(self._encoder_state, strict=True)
        del self._encoder_state

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C', T) with /100 scaling + zero-strip."""
        x = scale_div(x, _SCALE_DIVISOR)
        if self._resolved_names:
            x_clean, kept = strip_zero_channels(x)
            self._kept_names = (
                [self._resolved_names[i] for i in kept]
                if len(kept) < len(self._resolved_names)
                else self._resolved_names
            )
        else:
            x_clean = x
            self._kept_names = self._resolved_names or ["FPZ"]
        self._clean_x = x_clean
        return x_clean

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C', T) -> (B, 768) embeddings via tokenization + transformer."""
        batch_size, n_chans, T = x.shape
        block_size = int(self._encoder_conf.block_size)
        patch_size = int(self._encoder_conf.patch_size)
        n_time_patches = T // patch_size

        ch_indices = [_channel_idx(c) for c in self._kept_names[:n_chans]]

        tokens = torch.zeros(
            batch_size, block_size, patch_size, device=x.device, dtype=torch.float32
        )
        input_chans = torch.full(
            (batch_size, block_size), _PAD_IDX, device=x.device, dtype=torch.long
        )
        input_times = torch.zeros(
            batch_size, block_size, device=x.device, dtype=torch.long
        )
        input_mask = torch.zeros(
            batch_size, block_size, device=x.device, dtype=torch.bool
        )

        time_idx = torch.arange(n_time_patches, device=x.device, dtype=torch.long)
        pos = 0
        for ch_i, ch_idx in enumerate(ch_indices):
            if pos + n_time_patches > block_size:
                break
            ch_signal = x[:, ch_i, :]
            patches = ch_signal.reshape(batch_size, n_time_patches, patch_size)
            tokens[:, pos : pos + n_time_patches] = patches
            input_chans[:, pos : pos + n_time_patches] = ch_idx
            input_times[:, pos : pos + n_time_patches] = time_idx
            input_mask[:, pos : pos + n_time_patches] = True
            pos += n_time_patches

        attn_mask = input_mask.unsqueeze(1).repeat(1, block_size, 1).unsqueeze(1)

        features = self.encoder.forward_features(
            tokens,
            input_chans=input_chans,
            input_times=input_times,
            mask=attn_mask,
            return_all_tokens=True,
        )

        valid = input_mask.unsqueeze(-1).to(features.dtype)
        pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled
