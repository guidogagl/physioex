"""XSeqSleepNet: SeqSleepNet with pluggable epoch- and sequence-level encoders.

Research scaffold for the question "is the hierarchical fold/unfold of
L-SeqSleepNet necessary, or does a *flat* recurrent model with a matrix memory
(xLSTM / mLSTM) recover it?". The model keeps SeqSleepNet's contract

    (B, L, C, T, F) spectrograms  ->  (B, L, n_classes) logits,   encode() -> (B, L, D)

and lets the two recurrent slots of SeqSleepNet be swapped independently:

* **epoch encoder** ``(N, C, T, F) -> (N, d_model)``: ``"seqsleepnet"`` (learnable
  filterbank + BiLSTM + attention, identical to :class:`~physioex.models.seqsleepnet.SeqSleepNet`)
  or ``"xlstm"`` (filterbank + bidirectional mLSTM stack + attention).
* **sequence encoder** ``(B, L, d_model) -> (B, L, d_out)``:

  ================  ======================================================================
  ``gru``           4-layer BiGRU, the SeqSleepNet sequence stage (parity baseline)
  ``gru_uni``       unidirectional GRU (baseline for the causal arm)
  ``gru_wrapped``   BiGRU inside the same pre-LN/residual/down-projection skeleton as an
                    mLSTM block (isolates the "wrapper" from the recurrence)
  ``gru_matched``   ``gru_wrapped`` with the hidden size solved so that the encoder has
                    at least as many parameters as the ``xlstm_bi`` encoder
  ``xlstm_bi``      two mLSTM stacks (forward / time-reversed), concatenated and merged
  ``xlstm_causal``  one causal mLSTM stack; supports single-pass whole-night inference
  ``xlstm_alt``     one mLSTM stack, alternating direction per block (Vision-LSTM style)
  ================  ======================================================================

xLSTM encoders wrap the official ``xlstm`` package (NX-AI, ``pip install physioex[xlstm]``);
only mLSTM blocks are used (the sLSTM CUDA kernel is not needed). The package is imported
lazily, so the module is importable without it.

Every sequence encoder exposes ``output_size`` and ``is_causal``.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
import torch.nn as nn

from physioex.models.seqsleepnet import AttentionLayer, LearnableFilterbank, SeqSleepNet

# ---------------------------------------------------------------------------
# xlstm (optional dependency)
# ---------------------------------------------------------------------------

# `xlstm/__init__` imports the sLSTM CUDA loader, which at import time only
# *computes* include paths from CUDA_HOME and raises if it is unset -- even
# though we never instantiate an sLSTM block. A placeholder is enough for the
# pure-torch mLSTM path used here (a real toolkit is only needed for sLSTM).
# Set at module import so that `import xlstm` works anywhere after importing us.
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")


def _import_xlstm():
    try:
        from xlstm import (
            mLSTMBlockConfig,
            mLSTMLayerConfig,
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
        )
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The xLSTM encoders need the `xlstm` package: pip install 'physioex[xlstm]'"
        ) from e
    return xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig


def build_mlstm_stack(
    d_model: int,
    num_blocks: int = 2,
    num_heads: int = 4,
    context_length: int = 2048,
    conv1d_kernel_size: int = 4,
    qkv_proj_blocksize: int = 4,
    proj_factor: float = 2.0,
    dropout: float = 0.0,
    bias: bool = False,
    add_post_blocks_norm: bool = True,
) -> nn.Module:
    """Build a causal mLSTM-only ``xLSTMBlockStack``: ``(B, S, d_model) -> (B, S, d_model)``.

    ``context_length`` is the maximum sequence length the stack accepts (it sizes
    the causal mask); set it >= the longest night you will feed in one pass.
    """
    xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig = _import_xlstm()
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=conv1d_kernel_size,
                qkv_proj_blocksize=qkv_proj_blocksize,
                num_heads=num_heads,
                proj_factor=proj_factor,
            )
        ),
        slstm_block=None,
        slstm_at=[],
        context_length=context_length,
        num_blocks=num_blocks,
        embedding_dim=d_model,
        add_post_blocks_norm=add_post_blocks_norm,
        bias=bias,
        dropout=dropout,
    )
    return xLSTMBlockStack(cfg)


def _n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Epoch encoders: (N, C, T, F) -> (N, d_model)
# ---------------------------------------------------------------------------


class SeqSleepNetEpochEncoder(nn.Module):
    """Filterbank -> BiLSTM (``num_layers``) -> attention. Same modules and
    hyper-parameters as the epoch stage of :class:`SeqSleepNet`."""

    def __init__(
        self,
        in_chan: int = 1,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        fs: int = 100,
        lowfreq: int = 0,
        highfreq: int = 50,
        hidden: int = 64,
        num_layers: int = 4,
        attention_size: int = 32,
    ):
        super().__init__()
        self.filterbank = LearnableFilterbank(in_chan, F, D, nfft, fs, lowfreq, highfreq)
        self.rnn = nn.LSTM(D * in_chan, hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(2 * hidden, attention_size)
        self.output_size = 2 * hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, F = x.shape
        x = self.filterbank(x).permute(0, 2, 1, 3).reshape(N, T, -1)  # (N, T, C*D)
        x, _ = self.rnn(x)
        return self.attention(x)


class XLSTMEpochEncoder(nn.Module):
    """Filterbank -> linear to ``d_model`` -> bidirectional mLSTM stack over the T
    frames -> attention. The "epoch-level" arm of the study."""

    def __init__(
        self,
        in_chan: int = 1,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        fs: int = 100,
        lowfreq: int = 0,
        highfreq: int = 50,
        d_model: int = 128,
        attention_size: int = 32,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        context_length: int = 64,
    ):
        super().__init__()
        self.filterbank = LearnableFilterbank(in_chan, F, D, nfft, fs, lowfreq, highfreq)
        self.proj_in = nn.Linear(D * in_chan, d_model)
        self.seq = XLSTMSequenceEncoder(
            d_model, num_blocks=num_blocks, num_heads=num_heads, direction="bi",
            dropout=dropout, context_length=context_length,
        )
        self.attention = AttentionLayer(d_model, attention_size)
        self.output_size = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, F = x.shape
        x = self.filterbank(x).permute(0, 2, 1, 3).reshape(N, T, -1)
        x = self.seq(self.proj_in(x))
        return self.attention(x)


# ---------------------------------------------------------------------------
# Sequence encoders: (B, L, d_model) -> (B, L, output_size)
# ---------------------------------------------------------------------------


class GRUSequenceEncoder(nn.Module):
    """Plain (Bi)GRU, as in SeqSleepNet's sequence stage (no norm, no residual)."""

    def __init__(self, d_model: int, hidden: int = 64, num_layers: int = 4, bidirectional: bool = True):
        super().__init__()
        self.rnn = nn.GRU(d_model, hidden, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.output_size = hidden * (2 if bidirectional else 1)
        self.is_causal = not bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out


class WrappedGRUSequenceEncoder(nn.Module):
    """BiGRU inside an mLSTM-block-like skeleton: ``x = x + drop(down(GRU(LN(x))))``
    per block, then a final LayerNorm. Controls for the pre-LN / residual /
    projection scaffolding that comes bundled with xLSTM blocks."""

    def __init__(
        self,
        d_model: int,
        hidden: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        n_dir = 2 if bidirectional else 1
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.rnns = nn.ModuleList(
            [nn.GRU(d_model, hidden, num_layers=1, batch_first=True, bidirectional=bidirectional) for _ in range(num_blocks)]
        )
        self.downs = nn.ModuleList([nn.Linear(n_dir * hidden, d_model) for _ in range(num_blocks)])
        self.drop = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(d_model)
        self.output_size = d_model
        self.is_causal = not bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ln, rnn, down in zip(self.norms, self.rnns, self.downs):
            h, _ = rnn(ln(x))
            x = x + self.drop(down(h))
        return self.post_norm(x)

    @classmethod
    def matched_to(
        cls,
        target_params: int,
        d_model: int,
        num_blocks: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = True,
        max_hidden: int = 4096,
    ) -> "WrappedGRUSequenceEncoder":
        """Smallest hidden size whose encoder has >= ``target_params`` parameters."""
        lo, hi = 1, max_hidden
        while lo < hi:
            mid = (lo + hi) // 2
            n = _n_params(cls(d_model, mid, num_blocks, dropout, bidirectional))
            if n >= target_params:
                hi = mid
            else:
                lo = mid + 1
        return cls(d_model, lo, num_blocks, dropout, bidirectional)


class XLSTMSequenceEncoder(nn.Module):
    """mLSTM stack(s) from the official ``xlstm`` package.

    direction:
        ``"causal"`` one stack, left-to-right (state constant in L; one forward
        pass over a whole night is a valid causal inference, given
        ``context_length`` >= night length).
        ``"bi"`` two stacks (forward and time-reversed), concatenated and merged
        by a linear layer back to ``d_model`` -- the strict analogue of a BiRNN.
        ``"alt"`` one stack whose odd blocks run on the reversed sequence
        (Vision-LSTM style); bidirectional receptive field after two blocks.
    """

    def __init__(
        self,
        d_model: int,
        num_blocks: int = 2,
        num_heads: int = 4,
        direction: str = "bi",
        context_length: int = 2048,
        dropout: float = 0.0,
        conv1d_kernel_size: int = 4,
        qkv_proj_blocksize: int = 4,
        proj_factor: float = 2.0,
    ):
        super().__init__()
        if direction not in ("causal", "bi", "alt"):
            raise ValueError(f"direction must be causal|bi|alt, got {direction!r}")
        self.direction = direction
        self.context_length = context_length
        kw = dict(
            num_blocks=num_blocks, num_heads=num_heads, context_length=context_length,
            conv1d_kernel_size=conv1d_kernel_size, qkv_proj_blocksize=qkv_proj_blocksize,
            proj_factor=proj_factor, dropout=dropout,
        )
        if direction == "bi":
            self.fwd = build_mlstm_stack(d_model, **kw)
            self.bwd = build_mlstm_stack(d_model, **kw)
            self.merge = nn.Linear(2 * d_model, d_model)
        else:
            self.stack = build_mlstm_stack(d_model, **kw)
        self.output_size = d_model
        self.is_causal = direction == "causal"

    def _check_length(self, S: int) -> None:
        if S > self.context_length:
            raise ValueError(
                f"sequence length {S} exceeds context_length={self.context_length}; "
                f"rebuild the encoder with a larger context_length."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_length(x.shape[1])
        if self.direction == "causal":
            return self.stack(x)
        if self.direction == "bi":
            f = self.fwd(x)
            b = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
            return self.merge(torch.cat([f, b], dim=-1))
        # alt: run the stack's blocks by hand, flipping odd blocks
        for i, block in enumerate(self.stack.blocks):
            if i % 2 == 1:
                x = torch.flip(block(torch.flip(x, dims=[1])), dims=[1])
            else:
                x = block(x)
        return self.stack.post_blocks_norm(x)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

EPOCH_ENCODERS = ("seqsleepnet", "paper", "xlstm")
SEQUENCE_ENCODERS = ("gru", "gru_uni", "gru_wrapped", "gru_matched", "xlstm_bi", "xlstm_causal", "xlstm_alt")


def make_epoch_encoder(name: str, **kwargs) -> nn.Module:
    if name == "seqsleepnet":
        return SeqSleepNetEpochEncoder(**kwargs)
    if name == "paper":
        # Phan's epoch encoder as specified in the papers: 1-layer BLSTM with
        # recurrent batch norm, attention size 64, dropout on the cell.
        from physioex.models.lseqsleepnet import PaperEpochEncoder

        kwargs = dict(kwargs)
        if "fs" in kwargs:
            kwargs["sf"] = kwargs.pop("fs")
        return PaperEpochEncoder(**kwargs)
    if name == "xlstm":
        return XLSTMEpochEncoder(**kwargs)
    raise ValueError(f"unknown epoch encoder {name!r}; choose from {EPOCH_ENCODERS}")


def make_sequence_encoder(name: str, d_model: int, **kwargs) -> nn.Module:
    """Build a sequence encoder by variant name. ``kwargs`` are variant-specific
    (``hidden``, ``num_layers``, ``num_blocks``, ``num_heads``, ``dropout``,
    ``context_length``; ``target_params`` or the xlstm kwargs for ``gru_matched``)."""
    if name == "gru":
        return GRUSequenceEncoder(d_model, bidirectional=True, **kwargs)
    if name == "gru_uni":
        return GRUSequenceEncoder(d_model, bidirectional=False, **kwargs)
    if name == "gru_wrapped":
        return WrappedGRUSequenceEncoder(d_model, **kwargs)
    if name == "gru_matched":
        target = kwargs.pop("target_params", None)
        wrapped_keys = {"num_blocks", "dropout", "bidirectional"}
        wrapped_kw = {k: v for k, v in kwargs.items() if k in wrapped_keys}
        if target is None:
            xkw = {k: v for k, v in kwargs.items() if k not in {"bidirectional", "hidden"}}
            target = _n_params(XLSTMSequenceEncoder(d_model, direction="bi", **xkw))
        return WrappedGRUSequenceEncoder.matched_to(target, d_model, **wrapped_kw)
    if name.startswith("xlstm_"):
        direction = {"xlstm_bi": "bi", "xlstm_causal": "causal", "xlstm_alt": "alt"}[name]
        return XLSTMSequenceEncoder(d_model, direction=direction, **kwargs)
    raise ValueError(f"unknown sequence encoder {name!r}; choose from {SEQUENCE_ENCODERS}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class XSeqSleepNet(nn.Module):
    """SeqSleepNet with swappable encoders. See module docstring.

    Args:
        n_classes, in_chan, F, D, nfft, lowfreq, highfreq, fs: as in SeqSleepNet.
        epoch_encoder: variant name (``"seqsleepnet"`` | ``"xlstm"``) or a module
            mapping ``(N, C, T, F) -> (N, d)`` with an ``output_size`` attribute.
        sequence_encoder: variant name (see ``SEQUENCE_ENCODERS``) or a module
            mapping ``(B, L, d) -> (B, L, d_out)`` with an ``output_size`` attribute.
        epoch_kwargs, seq_kwargs: forwarded to the factories.
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_chan: int = 1,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        lowfreq: int = 0,
        highfreq: int = 50,
        fs: int = 100,
        epoch_encoder: str | nn.Module = "seqsleepnet",
        sequence_encoder: str | nn.Module = "gru",
        epoch_kwargs: Optional[dict] = None,
        seq_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        if isinstance(epoch_encoder, str):
            epoch_encoder = make_epoch_encoder(
                epoch_encoder, in_chan=in_chan, F=F, D=D, nfft=nfft, fs=fs,
                lowfreq=lowfreq, highfreq=highfreq, **(epoch_kwargs or {}),
            )
        self.epoch_encoder = epoch_encoder
        d_model = epoch_encoder.output_size
        if isinstance(sequence_encoder, str):
            sequence_encoder = make_sequence_encoder(sequence_encoder, d_model, **(seq_kwargs or {}))
        self.sequence_encoder = sequence_encoder
        self.clf = nn.Linear(sequence_encoder.output_size, n_classes)

    @property
    def is_causal(self) -> bool:
        return bool(getattr(self.sequence_encoder, "is_causal", False))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C, T, F) -> (B, L, d_out) contextualised epoch embeddings."""
        B, L, C, T, F = x.shape
        h = self.epoch_encoder(x.reshape(B * L, C, T, F)).reshape(B, L, -1)
        return self.sequence_encoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        B, L, D = z.shape
        return self.clf(z.reshape(B * L, D)).reshape(B, L, -1)

    # -- parity with SeqSleepNet --------------------------------------------

    @classmethod
    def from_seqsleepnet(cls, model: SeqSleepNet) -> "XSeqSleepNet":
        """Rebuild a trained :class:`SeqSleepNet` as ``XSeqSleepNet(seqsleepnet, gru)``
        with identical weights (exact functional parity)."""
        fb, a = model.filterbank, model.attention
        in_chan = fb.W.shape[0]
        new = cls(
            n_classes=model.clf.out_features, in_chan=in_chan, F=fb.F, D=fb.D,
            epoch_encoder="seqsleepnet",
            epoch_kwargs=dict(
                hidden=model.seqn1.hidden_size, num_layers=model.seqn1.num_layers,
                attention_size=a.W_omega.shape[1],
            ),
            sequence_encoder="gru",
            seq_kwargs=dict(hidden=model.seqn2.hidden_size, num_layers=model.seqn2.num_layers),
        )
        remap = {"filterbank.": "epoch_encoder.filterbank.", "seqn1.": "epoch_encoder.rnn.",
                 "attention.": "epoch_encoder.attention.", "seqn2.": "sequence_encoder.rnn.", "clf.": "clf."}
        sd = {}
        for k, v in model.state_dict().items():
            for old, newp in remap.items():
                if k.startswith(old):
                    sd[newp + k[len(old):]] = v
                    break
            else:  # pragma: no cover
                raise KeyError(f"unexpected SeqSleepNet parameter {k}")
        new.load_state_dict(sd, strict=True)
        return new


if __name__ == "__main__":
    for name in ("gru", "gru_wrapped"):
        m = XSeqSleepNet(sequence_encoder=name)
        y = m(torch.randn(2, 20, 1, 29, 129))
        print(f"{name:12s} params={_n_params(m):,} out={tuple(y.shape)}")
