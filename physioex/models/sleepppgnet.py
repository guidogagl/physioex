"""SleepPPG-Net: PPG-based sleep staging.

Ported from the community replica "SPPPPG-Net" (author: Franz Ehrlich,
``sleep-is-all-you-need`` on GitLab, package ``SPPPPGNet`` 0.1.4, class
``SleepPPGNetTorch``), which reimplements the architecture of:

    Kotzen, C., Mollenhauer, M., Schirrmacher, T., & Schirrmeister, G. R. (2023).
    "SleepPPG-Net: A Deep Learning Model for Sleep Staging using Pulse
    Volume Recording Signals." IEEE J. Biomed. Health Inform. 27(4):924-932.
    arXiv:2202.05735.

The pre-trained weights on Hugging Face (``sleep-is-all-you-need/SPPPPG-Net``)
are distributed under a fair-noncommercial-research license: they may be used
for non-commercial research only.

This port keeps the replica's submodule names verbatim, so a replica
checkpoint loads with ``load_state_dict(..., strict=True)`` without any
remapping.  Only the "own" architecture is ported; the pyPhasesML/SleePyPhases
adapter and the alternative Wav2Net implementation are dropped.
"""
from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# ── Class order ────────────────────────────────────────────────────────────
# The replica's checkpoint was trained with the class order [Wake, REM, Light,
# Deep] (Light = N1+N2).  Physioex uses the AASM order W, N1, N2, N3, REM; for
# this 4-class model the closest mapping is [W, Light, Deep, REM].
#
#   replica  :  0=Wake  1=REM  2=Light  3=Deep
#   physioex :  0=W     1=Light  2=Deep  3=REM
#
# ORIGINAL_TO_PHYSIOEX[j] is the replica index of the logit that becomes
# physioex slot j.
ORIGINAL_TO_PHYSIOEX: torch.Tensor = torch.tensor([0, 2, 3, 1])

N_EPOCHS = 1200
SAMPLES_PER_EPOCH = 1024
FULL_LENGTH = N_EPOCHS * SAMPLES_PER_EPOCH  # 1_228_800 samples


def get_activation(name: str, **kwargs) -> nn.Module:
    """Return an activation function from its name."""
    if name == "relu":
        return nn.ReLU(**kwargs)
    elif name == "leaky":
        return nn.LeakyReLU(**kwargs)
    elif name == "gelu":
        return nn.GELU(**kwargs)
    elif name == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"{name=} is unsupported.")


def get_norm(name: str | None = "batch", *args, **kwargs) -> nn.Module:
    """Return a normalisation module from its name."""
    if name == "batch":
        return nn.BatchNorm1d(*args, **kwargs)
    elif name == "instance":
        return nn.InstanceNorm1d(*args, **kwargs)
    elif name is None:
        return nn.Identity()
    else:
        raise ValueError(f"Normalisation with {name=} unknown.")


class ResConvBlock(nn.Module):
    """Residual block of three 1-D convolutions followed by max-pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pool_size: int = 2,
        norm: str = "batch",
        activation: str = "leaky",
    ):
        super().__init__()

        use_bias = norm is None

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            bias=use_bias,
        )
        self.norm1 = get_norm(norm, num_features=out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            bias=use_bias,
        )
        self.norm2 = get_norm(norm, num_features=out_channels)

        self.conv3 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            bias=use_bias,
        )
        self.norm3 = get_norm(norm, num_features=out_channels)

        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)
        self.activation = get_activation(activation)

        # 1x1 conv matching the residual to the downsampled output.
        self.match_dimensions = nn.Conv1d(
            in_channels,
            out_channels,
            1,
            stride=pool_size,
            padding=0,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.match_dimensions(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.activation(out)

        out = self.pool(out)
        out = out + identity

        return out


class TCNBlock(nn.Module):
    """Temporal convolutional block of stacked dilated 1-D convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation_rates=(1, 2, 4, 8, 16, 32),
        dropout: float = 0.2,
        norm: str = "batch",
        activation: str = "leaky",
    ):
        super().__init__()
        self.dilated_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        use_bias = norm is None

        for rate in dilation_rates:
            padding = (kernel_size - 1) * rate // 2
            dilated_conv = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=rate,
                bias=use_bias,
            )
            self.dilated_convs.append(dilated_conv)
            self.norms.append(get_norm(norm, num_features=channels))

        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = x
        for conv, norm in zip(self.dilated_convs, self.norms):
            out = conv(out)
            out = norm(out)
            out = self.activation(out)

        out = out + identity
        out = self.dropout(out)
        out = self.activation(out)

        return out


class WindowLayer(nn.Module):
    """Reshape a 1-D feature map into a stack of fixed-size windows."""

    def __init__(self, input_size: int, num_windows: int, window_size: int):
        super().__init__()
        self.num_windows = num_windows
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, channels, length]
        batch_size, channels, _ = x.size()

        x = x.view(batch_size, channels, self.num_windows, -1)
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_windows, channels, window_size]
        x = x.reshape(batch_size, self.num_windows, -1)

        return x


class TimeDistributedDense(nn.Module):
    """A shared dense layer applied to every window of a time series."""

    def __init__(
        self, input_dim: int, output_dim: int, activation: str = "leaky", norm: str = "batch"
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = get_activation(activation)
        self.norm = get_norm(norm, num_features=output_dim) if norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, time_steps, input_dim]
        batch_size, time_steps, input_dim = x.size()

        x_reshaped = x.reshape(batch_size, -1, input_dim)
        y = self.dense(x_reshaped)

        if not isinstance(self.norm, nn.Identity):
            # BatchNorm1d over the channel axis.
            y = y.view(batch_size, time_steps, -1).transpose(1, 2)  # [B, out, T]
            y = self.norm(y)
            y = y.transpose(1, 2)  # [B, T, out]
        else:
            y = y.view(batch_size, time_steps, -1)

        y = self.activation(y)

        return y


class SleepPPGNet(nn.Module):
    """SleepPPG-Net with a physioex per-epoch interface.

    The submodule layout matches the replica's ``SleepPPGNetTorch``
    (``fe_layers`` x8, ``window_layer``, ``td_dense``, ``tcn1``, ``tcn2``,
    ``fc``) so that a replica checkpoint loads with ``strict=True`` and no
    remapping.

    Args:
        n_classes: number of output classes (4).
        n_epochs: number of 30 s epochs in a full night (1200).
        samples_per_epoch: samples per epoch (1024, i.e. 32 Hz).
        class_order: ``"physioex"`` permutes the replica's output
            [Wake, REM, Light, Deep] to [W, Light, Deep, REM];
            ``"original"`` returns the replica order unchanged.

    Input:
        A full night of ``n_epochs * samples_per_epoch`` (1_228_800) samples,
        given as ``(B, n_epochs, samples_per_epoch)``,
        ``(B, n_epochs, 1, samples_per_epoch)`` (the physioex per-epoch
        format), or ``(B, 1, n_epochs * samples_per_epoch)`` (the original
        format).

    Output:
        Logits ``(B, n_epochs, n_classes)``.
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_epochs: int = N_EPOCHS,
        samples_per_epoch: int = SAMPLES_PER_EPOCH,
        class_order: str = "physioex",
    ):
        super().__init__()
        if class_order not in ("physioex", "original"):
            raise ValueError(f"class_order must be 'physioex' or 'original', got {class_order!r}")
        if n_classes != 4:
            raise ValueError("SleepPPGNet is a 4-class model; n_classes must be 4")

        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.samples_per_epoch = samples_per_epoch
        self.class_order = class_order
        self.input_length = n_epochs * samples_per_epoch

        # Feature extractor (FE) — 8 stacked ResConvBlocks.
        self.fe_layers = nn.ModuleList()
        channels = [1, 16, 16, 32, 32, 64, 64, 128, 256]
        for i in range(len(channels) - 1):
            self.fe_layers.append(ResConvBlock(channels[i], channels[i + 1]))

        # 8 poolings of size 2 -> length divided by 256, 256 channels.
        self.fe_output_size = self.input_length // (2**8)
        self.fe_output_channels = channels[-1]

        # Windowing layer.  The replica hard-codes 1200 windows for a full
        # night; we keep that for the default config and otherwise split the
        # feature map evenly (num_windows * window_size == feature length).
        if self.input_length == FULL_LENGTH:
            self.num_windows = 1200
        else:
            self.num_windows = self.fe_output_size
        self.window_size = self.fe_output_size * self.fe_output_channels // self.num_windows
        self.window_layer = WindowLayer(
            self.fe_output_size, self.num_windows, self.window_size
        )

        # Time-distributed dense layer for compression.
        self.td_dense = TimeDistributedDense(self.window_size, 128)

        # FS — 2 stacked TCN blocks.
        self.tcn1 = TCNBlock(128, dilation_rates=[1, 2, 4, 8, 16, 32])
        self.tcn2 = TCNBlock(128, dilation_rates=[1, 2, 4, 8, 16, 32])

        # FC — final 1x1 convolution for predictions.
        self.fc = nn.Conv1d(128, n_classes, kernel_size=1)

    def _run(self, x1d: Tensor) -> Tensor:
        """Run the network on a flat ``(B, input_length)`` tensor.

        Returns logits of shape ``(B, n_classes, n_epochs)`` in the replica's
        original class order.
        """
        batch = x1d.shape[0]
        x = x1d.reshape(batch, 1, self.input_length)

        for layer in self.fe_layers:
            x = layer(x)

        x = self.window_layer(x)  # [B, num_windows, window_features]
        x = self.td_dense(x)  # [B, num_windows, 128]

        x = x.transpose(1, 2)  # [B, 128, num_windows]
        x = self.tcn1(x)
        x = self.tcn2(x)

        x = self.fc(x)  # [B, num_classes, num_windows]

        return x

    def forward(self, x: Tensor) -> Tensor:
        """Return per-epoch logits of shape ``(B, L, n_classes)``.

        ``x`` may be ``(B, L, 1, S)``, ``(B, L, S)`` (physioex per-epoch) or
        ``(B, 1, T)`` (original format, ``T = n_epochs * samples_per_epoch``).
        When ``L != n_epochs`` the input is zero-padded at the end (or
        truncated) to ``n_epochs`` epochs before the network is run, and only
        the first ``L`` epochs are returned.
        """
        if x.dim() == 4:
            batch, length, _chan, samples = x.shape
            if samples != self.samples_per_epoch:
                raise ValueError(
                    f"expected samples_per_epoch={self.samples_per_epoch}, got {samples}"
                )
            x1d = x.reshape(batch, length * samples)
        elif x.dim() == 3:
            if x.shape[1] == 1:
                batch, _, total = x.shape
                length = total // self.samples_per_epoch
                if total != length * self.samples_per_epoch:
                    raise ValueError(
                        f"input length {total} is not a multiple of "
                        f"samples_per_epoch={self.samples_per_epoch}"
                    )
                x1d = x.reshape(batch, total)
            else:
                batch, length, samples = x.shape
                if samples != self.samples_per_epoch:
                    raise ValueError(
                        f"expected samples_per_epoch={self.samples_per_epoch}, got {samples}"
                    )
                x1d = x.reshape(batch, length * samples)
        else:
            raise ValueError(
                f"expected input of shape (B, L, 1, S), (B, L, S) or (B, 1, T), got {x.shape}"
            )

        # Pad or truncate to exactly n_epochs epochs.
        if length > self.n_epochs:
            x1d = x1d[:, : self.input_length]
        elif length < self.n_epochs:
            pad = self.input_length - x1d.shape[1]
            x1d = torch.nn.functional.pad(x1d, (0, pad), value=0.0)

        n_out = min(length, self.n_epochs)
        logits = self._run(x1d)  # (B, n_classes, n_epochs)

        # (B, C, n_epochs) -> (B, n_epochs, C), then keep the first L epochs.
        logits = logits.transpose(1, 2)[:, :n_out, :]

        if self.class_order == "physioex":
            logits = logits[:, :, ORIGINAL_TO_PHYSIOEX]

        return logits

    @classmethod
    def from_original(
        cls, checkpoint_path: str, class_order: str = "physioex"
    ) -> "SleepPPGNet":
        """Load a replica checkpoint into a fresh ``SleepPPGNet``.

        The replica checkpoint is not a plain state dict, so it is loaded with
        ``weights_only=False``; a ``state_dict``/``model`` wrapper, if present,
        is unwrapped before a strict load.  The returned model is in eval mode.
        """
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            for key in ("state_dict", "model"):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break

        model = cls(class_order=class_order)
        model.load_state_dict(obj, strict=True)
        model.eval()
        return model

    def predict_night(self, signals: "dict[str, np.ndarray]", metadata: "dict | None" = None) -> "np.ndarray":
        """Classify one whole night, already preprocessed by the ``sleepppgnet`` preset.

        Args:
            signals: Mapping of signal name to a ``(L, 1024)`` float array of
                one night; the night's ``"PPG"`` channel is required. The
                night is zero-padded (or truncated) to the model's fixed
                ``n_epochs`` (1200 for the released model) grid before the
                network runs, as in the authors' code for nights shorter or
                longer than 10 h.
            metadata: Unused (kept for a common interface with the other
                cardiorespiratory models).

        Returns:
            int64 class indices of shape ``(L,)`` in the model's class order
            ([W, Light, Deep, REM] for ``class_order="physioex"``). Runs under
            ``torch.no_grad()`` in eval mode, batch of one, on the model's
            current device.
        """
        import numpy as np

        del metadata
        if "PPG" not in signals:
            raise ValueError(f"SleepPPGNet needs a 'PPG' signal, got {sorted(signals)}")
        arr = np.asarray(signals["PPG"], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.samples_per_epoch:
            raise ValueError(
                f"'PPG' must have shape (L, {self.samples_per_epoch} samples/epoch), got {arr.shape}"
            )
        L = arr.shape[0]
        x = torch.from_numpy(arr).reshape(1, L, 1, self.samples_per_epoch)
        x = x.to(next(self.parameters()).device)

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                preds = self.forward(x)[0].argmax(dim=-1).cpu().numpy()
        finally:
            if was_training:
                self.train()
        return preds.astype(np.int64)
