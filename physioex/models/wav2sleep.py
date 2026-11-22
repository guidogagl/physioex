"""Wav2Sleep multi-signal sleep staging network.

Port of the authors' implementation (MIT License, Copyright (c) Jonathan F.
Carter) from https://github.com/joncarter/wav2sleep, used under its MIT license.

Reference:
    Carter & Tarassenko, "Wav2Sleep: A multi-signal deep learning model for
    sleep staging", arXiv:2411.04644.

The module and attribute names are kept identical to the authors' code so
that the released ``state_dict.pth`` checkpoints load with
``load_state_dict(strict=True)`` without any key remapping.
"""
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# Mapping of signals to the number of samples in one 30-second epoch.
SAMPLES_PER_EPOCH: dict[str, int] = {
    "ABD": 256,
    "THX": 256,
    "ECG": 1024,
    "PPG": 1024,
    "EOG-L": 4096,
    "EOG-R": 4096,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    """Return an activation function from its name."""
    if name == "relu":
        return nn.ReLU(**kwargs)
    elif name == "leaky":
        return nn.LeakyReLU(**kwargs)
    elif name == "gelu":
        return nn.GELU(**kwargs)
    elif name == "silu" or name == "swish":
        return nn.SiLU(**kwargs)
    elif name == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"{name=} is unsupported.")


class ConvLayerNorm(nn.Module):
    """Layer norm for convolutional layers with channels first."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        self.eps = eps

    def forward(self, x_BCT: Tensor) -> Tensor:
        mu_BCT = x_BCT.mean(1, keepdim=True)
        sigma_BCT = (x_BCT - mu_BCT).pow(2).mean(1, keepdim=True)
        x_BCT = (x_BCT - mu_BCT) / torch.sqrt(sigma_BCT + self.eps)
        x_BCT = self.weight * x_BCT + self.bias
        return x_BCT


def get_norm(
    name: Optional[str] = "batch", causal: bool = False, *args, **kwargs
) -> nn.Module:
    """Return a normalization module from its name.

    ``norm_eps`` is accepted for any norm type but only instance norm uses it.
    """
    # Extract norm_eps - only used by instance norm, but may be passed for any norm type
    norm_eps = kwargs.pop("norm_eps", None)

    if name == "batch":
        return nn.BatchNorm1d(*args, **kwargs)
    elif name == "layer":
        return ConvLayerNorm(*args, **kwargs)
    elif name is None:
        return nn.Identity()
    elif name == "instance":
        if norm_eps is not None:
            kwargs["eps"] = norm_eps
        return nn.InstanceNorm1d(*args, **kwargs)
    else:
        raise ValueError(f"Normalisation with {name=} and {causal=} unknown.")


class ConvLayer1D(nn.Module):
    """Generic 1D convolutional layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        causal: bool = False,
        groups: int = 1,
        activation: str = "relu",
        bias: bool = False,
        norm: Optional[str] = "batch",
        norm_eps: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.causal = causal
        if causal:  # Create causal padding.
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = padding
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            groups=groups,
            bias=bias
            or norm is None,  # Make sure bias is used if no normalization is applied.
            dilation=dilation,
        )
        norm_kwargs = {"norm_eps": norm_eps} if norm_eps is not None else {}
        self.norm = get_norm(
            norm, causal=causal, num_features=output_dim, **norm_kwargs
        )
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        # In causal mode, remove the right-padding immediately after convolution to avoid skewing
        # normalization stats. When using stride > 1, trim slightly less to preserve correct
        # downsampled length so that residual paths (e.g. 1x1, stride-2) align.
        if self.causal and self.padding > 0:
            stride = (
                self.conv.stride[0]
                if isinstance(self.conv.stride, tuple)
                else self.conv.stride
            )
            right_trim = max(self.padding - (stride - 1), 0)
            if right_trim > 0:
                out = out[:, :, :-right_trim]
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class ConvBlock1D(nn.Module):
    """Three-layer convolutional block with downsampling by a factor of 2."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "leaky",
        norm: str = "batch",
        causal: bool = False,
        norm_eps: Optional[float] = None,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual

        def create_conv(
            input_dim: int, output_dim: int, stride: int = 1
        ) -> ConvLayer1D:
            return ConvLayer1D(
                input_dim=input_dim,
                output_dim=output_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=activation,
                norm=norm,
                dropout=dropout,
                causal=causal,
                norm_eps=norm_eps,
            )

        self.conv1 = create_conv(input_dim, output_dim)
        self.conv2 = create_conv(output_dim, output_dim)
        # Downsample within the convolutional path using a stride-2 conv for proper causal alignment.
        self.conv3 = create_conv(output_dim, output_dim, stride=2)
        self.activation = get_activation(activation)
        # Conventional ResNet downsampling - linear transformation with matching stride and output channels.
        if self.use_residual:
            self.downsample = nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False,
            )
        else:
            self.register_parameter("downsample", None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape [N, Cin, L]

        Returns:
            Tensor: shape [N, Cout, L]
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.use_residual:
            out = out + self.downsample(x)
        out = self.activation(out)
        return out


class DilatedConvBlock(nn.Module):
    """Dilated convolutional block.

    Uses a consistent channel dimension and progressively wider dilations to
    increase the context length for each epoch.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        dropout: float = 0.2,
        activation: str = "leaky",
        norm: str = "batch",
        kernel_size: int = 7,
        causal: bool = False,
        num_dilations: int = 6,
    ) -> None:
        super().__init__()
        blocks = []
        self.kernel_size = kernel_size
        self.dilations = [2**i for i in range(num_dilations)]
        for dilation in self.dilations:
            # Calculate effective kernel size to pad correctly.
            k_eff = kernel_size + (kernel_size - 1) * (dilation - 1)
            padding = k_eff // 2
            blocks.append(
                ConvLayer1D(
                    input_dim=feature_dim,
                    output_dim=feature_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    activation=activation,
                    norm=norm,
                    causal=causal,
                )
            )
        self.dropout = nn.Dropout(p=dropout)
        self.conv_layers = nn.Sequential(*blocks)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape [N, F, S]
        Returns:
            Tensor: shape [N, F, S]
        """
        out = self.conv_layers(x)
        out = self.dropout(out)
        out = out + x
        out = self.activation(out)
        return out


class Wav2Sleep(nn.Module):
    """Model for sleep staging.

    This model is used to classify sleep stages from time-series inputs.

    The network works as follows:
        1. Each input signal is passed through a signal encoder.
        2. Embeddings are mixed for each sleep epoch level.
        3. Sequence model mixes epoch features.
        4. Classifier is applied to output features.
    """

    def __init__(
        self,
        signal_encoders: "SignalEncoders",
        epoch_mixer: "MultiModalAttentionEmbedder",
        sequence_mixer: "SequenceCNN",
        num_classes: int,
    ):
        super().__init__()
        self.signal_encoders = signal_encoders
        self.epoch_mixer = epoch_mixer
        self.sequence_mixer = sequence_mixer
        self.feature_dim = self.epoch_mixer.feature_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(
            in_features=self.feature_dim, out_features=num_classes
        )

    @property
    def valid_signals(self) -> list[str]:
        """Return list of signals that can be used by the model."""
        return list(self.signal_encoders.signal_map.keys())

    @staticmethod
    def _to_flat_inputs(
        x: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], list[str], bool]:
        """Normalize the two accepted input formats to ``dict[str, Tensor(B, T_s)]``.

        Args:
            x: either ``dict[str, Tensor(B, T_s)]`` (the authors' format, with
                ``T_s = samples_per_epoch * S`` samples per signal) or
                ``dict[str, Tensor(B, L, T_e)]`` (the physioex format, per-epoch
                samples with ``T_e = samples_per_epoch`` for that signal).

        Returns:
            Tuple of (flat input dict, sorted signal names, whether any signal
            is absent (missing key or entirely -inf), as in the authors' code).
        """
        x_flat: dict[str, Tensor] = {}
        for signal_name, x_st in x.items():
            if signal_name not in SAMPLES_PER_EPOCH:
                raise ValueError(
                    f"Signal {signal_name!r} unrecognised. "
                    f"Expected one of {sorted(SAMPLES_PER_EPOCH)}."
                )
            t_e = SAMPLES_PER_EPOCH[signal_name]
            if x_st.ndim == 2:  # (B, T_s)
                if x_st.size(-1) % t_e != 0:
                    raise ValueError(
                        f"Input length {x_st.size(-1)} must be divisible by {t_e} "
                        f"for signal {signal_name!r}."
                    )
                x_flat[signal_name] = x_st
            elif x_st.ndim == 3:  # (B, L, T_e)
                if x_st.size(-1) != t_e:
                    raise ValueError(
                        f"Input length {x_st.size(-1)} must be {t_e} (samples per epoch) "
                        f"for signal {signal_name!r}."
                    )
                B, L, T = x_st.shape
                x_flat[signal_name] = x_st.reshape(B, L * T)
            else:
                raise ValueError(
                    f"Signal {signal_name!r} must have 2 or 3 dimensions, got {x_st.ndim}."
                )
        signals = sorted(x_flat.keys())
        if not signals:
            raise ValueError("At least one signal must be provided.")
        any_absent = any(bool(torch.isinf(x_flat[s]).all().item()) for s in signals)
        return x_flat, signals, any_absent

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        """Classify sleep stages from a dictionary of input signals.

        Returns per-epoch logits for each sleep stage. For ``num_classes=4``
        the class order is [W, N1+N2, N3, REM].

        Args:
            x: Dictionary of input tensors, keyed by signal name (ECG, PPG,
                ABD, THX, EOG-L, EOG-R). Each tensor has shape
                ``[batch_size, T_s]`` (``T_s = samples_per_epoch * S``) or
                ``[batch_size, L, T_e]`` (``T_e = samples_per_epoch``). A signal
                may be absent (missing key) or entirely ``-inf``; both mark it
                as unavailable for the attention padding mask.

        Returns:
            Tensor, shape [batch_size, S, num_classes]
        """
        x_flat, signals, any_absent = self._to_flat_inputs(x)
        # Create feature vectors for each signal.
        z_dict = self.signal_encoders(x_flat)
        # Mix features for each sleep epoch from different modalities.
        z_BSF = self.epoch_mixer(z_dict)
        # Mix unified features across the sequence.
        z_BSF = self.sequence_mixer(z_BSF)
        # Apply a classifier to features for each element in the sequence(s).
        logits_BSF = self.classifier(z_BSF)
        # Re-introduce the -inf sentinel for batch elements whose signals were
        # all unavailable, exactly as in the authors' code (missing or all -inf
        # input signals propagate through as an -inf logit row).
        if any_absent:
            B = x_flat[signals[0]].size(0)
            absent_B = torch.stack(
                [torch.isinf(x_flat[s]).all(dim=-1) for s in signals], dim=-1
            ).all(dim=-1)
            logits_BSF = torch.where(absent_B[:, None, None], float("-inf"), logits_BSF)
        return logits_BSF

    def predict(self, x: dict[str, Tensor]) -> Tensor:
        """Classify each epoch with the argmax class.

        Args:
            x: Dictionary of input tensors, see :meth:`forward`.

        Returns:
            Tensor, shape [batch_size, S]
        """
        logits_BSF = self.forward(x)
        return logits_BSF.argmax(axis=-1)

    def predict_night(
        self, signals: dict[str, "np.ndarray"], metadata: Optional[dict] = None
    ) -> "np.ndarray":
        """Classify one whole night, already preprocessed by the ``wav2sleep`` preset.

        Args:
            signals: Mapping of signal name (ECG, PPG, ABD, THX, ...) to a
                ``(L, T_e)`` float array of one night, with
                ``T_e = SAMPLES_PER_EPOCH[signal]`` samples per 30-s epoch.
                Keys the model does not know are ignored; known keys missing
                from ``signals`` are passed to the model as absent. The night
                is zero-padded (or truncated) to the model's fixed 1200-epoch
                (10 h) grid only when ``L`` differs from 1200, matching the
                authors' padding/truncation of shorter/longer nights.
            metadata: Unused (kept for a common interface with the other
                cardiorespiratory models).

        Returns:
            int64 class indices of shape ``(L,)`` in the model's own class
            order ([W, N1+N2, N3, REM] for the 4-class model). Runs under
            ``torch.no_grad()`` in eval mode, batch of one, on the model's
            current device.
        """
        import numpy as np

        del metadata
        L = next(iter(signals.values())).shape[0]
        n_epochs = 1200  # 10 h at 30-s epochs: the fixed grid the model was trained on
        device = next(self.parameters()).device

        x: dict[str, Tensor] = {}
        for sig in self.valid_signals:
            t_e = SAMPLES_PER_EPOCH[sig]
            if sig not in signals:
                x[sig] = torch.full((1, n_epochs * t_e), float("-inf"), device=device)
                continue
            arr = np.asarray(signals[sig], dtype=np.float32)
            if arr.shape != (L, t_e):
                raise ValueError(
                    f"signal {sig!r} has shape {arr.shape}; expected (L={L}, {t_e} samples/epoch)"
                )
            if L != n_epochs:
                target = n_epochs * t_e
                if L > n_epochs:
                    arr = arr[:n_epochs].reshape(-1)
                else:
                    pad = np.zeros((n_epochs - L) * t_e, dtype=np.float32)
                    arr = np.concatenate([arr.reshape(-1), pad])
            x[sig] = (
                torch.from_numpy(arr.astype(np.float32, copy=False))
                .reshape(1, -1)
                .to(device)
            )

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                preds = self.forward(x)[0, :L].argmax(dim=-1).cpu().numpy()
        finally:
            if was_training:
                self.train()
        return preds.astype(np.int64)

    @classmethod
    def from_original(cls, config_yaml_path: str, state_dict_path: str) -> "Wav2Sleep":
        """Build the model from the authors' ``config.yaml`` and checkpoint.

        The config is parsed with ``yaml.safe_load``; each nested dict's
        ``_target_`` names the corresponding authors' class and is mapped to
        the ported class of the same name before the ``_target_`` key is
        dropped. The state dict is loaded with ``strict=True``.

        Args:
            config_yaml_path: Path to the authors' ``config.yaml``.
            state_dict_path: Path to the authors' ``state_dict.pth``.

        Returns:
            The model in eval mode with the authors' weights.
        """
        cfg = yaml.safe_load(Path(config_yaml_path).read_text())
        known = {
            "Wav2Sleep": cls,
            "SignalEncoders": SignalEncoders,
            "MultiModalAttentionEmbedder": MultiModalAttentionEmbedder,
            "SequenceCNN": SequenceCNN,
        }

        def resolve(node):
            if isinstance(node, dict):
                target = node.pop("_target_", None)
                kwargs = {k: resolve(v) for k, v in node.items()}
                if target is not None:
                    name = target.rsplit(".", 1)[-1]
                    if name not in known:
                        raise ValueError(
                            f"Unknown _target_ class {target!r} in config."
                        )
                    return known[name](**kwargs)
                return kwargs
            if isinstance(node, list):
                return [resolve(v) for v in node]
            return node

        model = resolve(cfg)
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model


class SignalEncoders(nn.Module):
    """Class that handles multiple signal encoders."""

    def __init__(
        self,
        signal_map: dict[str, str],
        feature_dim: int,
        activation: str,
        norm: str = "instance",
        causal: bool = False,
        chunk_causal: bool = True,
        embed_signals: bool = False,
        initial_channels: int = 16,
        max_channels: int = 128,
        output_norm: bool = False,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.signal_map = signal_map
        self.causal = causal
        encoders = {}
        # Create encoders for the signals.
        # Multiple signals *can* map to the same encoder.
        for signal_name, encoder_name in self.signal_map.items():
            if encoder_name in encoders:
                continue
            if signal_name not in SAMPLES_PER_EPOCH:
                raise ValueError(
                    f"Column {signal_name} unrecognised. Doesn't have a sampling rate."
                )
            samples_per_epoch = SAMPLES_PER_EPOCH[signal_name]
            encoders[encoder_name] = SignalEncoder(
                input_dim=1,
                feature_dim=feature_dim,
                samples_per_epoch=samples_per_epoch,
                activation=activation,
                norm=norm,
                causal=causal,
                chunk_causal=chunk_causal,
                initial_channels=initial_channels,
                max_channels=max_channels,
                output_norm=output_norm,
                use_residual=use_residual,
            )
        self.encoders = nn.ModuleDict(encoders)
        # Optionally add embedding to indicate signal source.
        self.embed_signals = embed_signals
        self.sig_to_embedding_idx = {
            sig: i for i, sig in enumerate(sorted(signal_map.keys()))
        }
        if self.embed_signals:
            self.embedder = nn.Embedding(
                num_embeddings=len(signal_map), embedding_dim=self.feature_dim
            )
        else:
            self.register_parameter("embedder", None)

    def __len__(self) -> int:
        return len(self.encoders)

    def get_encoder(self, signal_name: str) -> "SignalEncoder":
        """Get the signal encoder for a signal."""
        if self.signal_map is not None:
            encoder_name = self.signal_map[signal_name]
        else:
            encoder_name = signal_name
        return self.encoders[encoder_name]  # type: ignore

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        z_dict: dict[str, Tensor] = {}
        # Apply signal encoder for each input signal.
        for signal_name, x_BT in x.items():
            mask_B = torch.isinf(
                x_BT[:, 0]
            )  # Identify inf. batch elements (missing signals during training).
            x_BT = torch.where(
                torch.isinf(x_BT), 0.0, x_BT
            )  # Set inf. values to zero for stability.
            z_BSF = self.get_encoder(signal_name)(x_BT)
            # Re-fill output batch elements with neg. inf. values so embedder knows what is missing.
            z_BSF = torch.where(mask_B[:, None, None], float("-inf"), z_BSF)
            if (
                self.embed_signals
            ):  # Add embedding to indicate signal source. (For shared encoders)
                e_key = torch.tensor(
                    [self.sig_to_embedding_idx[signal_name]],
                    device=z_BSF.device,
                    dtype=torch.int64,
                )
                e_1F = self.embedder(e_key)
                e_BSF = e_1F[None, :, :].repeat(z_BSF.size(0), z_BSF.size(1), 1)
                z_BSF += e_BSF
            z_dict[signal_name] = z_BSF
        return z_dict


class SignalEncoder(nn.Module):
    """Signal encoder layer.

    Progressively downsamples the input waveform, resulting in feature vector
    sequence for each sleep epoch. Then applies time-distributed dense layer to
    produce the final feature vector.

    When causal=True, each 30-second epoch is processed independently
    (quasi-causal), ensuring predictions for epoch t only use data from epoch
    t. This allows using standard instance normalization with improved
    training stability.
    """

    def __init__(
        self,
        input_dim: int = 1,
        feature_dim: int = 256,
        activation: str = "gelu",
        samples_per_epoch: int = 1024,
        norm: str = "instance",
        initial_channels: int = 16,
        max_channels: int = 128,
        causal: bool = False,
        chunk_causal: bool = True,
        output_norm: bool = False,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.samples_per_epoch = samples_per_epoch
        self.causal = causal
        self.chunk_causal = chunk_causal
        # Check that samples_per_epoch is a power of 2
        if samples_per_epoch & (samples_per_epoch - 1) != 0:
            raise ValueError(
                f"samples_per_epoch must be a power of 2, got {samples_per_epoch}"
            )
        # Calculate number of convolutional blocks to downsample the input to four feature vectors per sleep epoch (30 seconds).
        num_blocks = int(math.log2(samples_per_epoch)) - 2
        logger.debug(f"Creating {num_blocks=} convolutional blocks.")
        # Double number of channels every other block.
        channels = [
            min(initial_channels * 2 ** (i // 2), max_channels)
            for i in range(num_blocks)
        ]
        blocks = []
        # Note: When causal=True, we can achieve causality by chunking the input into 30-second epochs OR by using causal convolutions.
        _causal_conv_mode = causal and not chunk_causal
        for i, output_dim in enumerate(channels):
            if norm == "auto":
                if i < 2:  # Use instance norm in early layers.
                    _norm_i = "instance"
                else:
                    _norm_i = "layer"
            else:
                _norm_i = norm
            # Use a larger epsilon for instance norm to prevent NaN from low-variance feature maps.
            # This applies to both causal and non-causal modes for numerical stability.
            _norm_eps = 1e-2 if _norm_i == "instance" else None
            blocks.append(
                ConvBlock1D(
                    input_dim,
                    output_dim,
                    activation=activation,
                    norm=_norm_i,
                    norm_eps=_norm_eps,
                    causal=_causal_conv_mode,
                    use_residual=use_residual,
                )
            )
            input_dim = output_dim
        self.cnn = nn.Sequential(*blocks)
        self.epoch_dim = (
            channels[-1] * 4
        )  # Flattened dimension for time-distributed dense layer.
        self.linear = nn.Linear(self.epoch_dim, feature_dim)
        self.activation = get_activation(activation)
        # Optional output normalization - normalizes across feature_dim (causal-friendly).
        self.output_norm = nn.LayerNorm(feature_dim) if output_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape [B, T]

        Returns:
            Tensor: shape [B, S, f_dim]
        """
        if x.size(-1) % self.samples_per_epoch:
            raise ValueError(
                f"Input length {x.size(-1)} must be divisible by {self.samples_per_epoch=}."
            )
        B = x.size(0)
        S = x.size(-1) // self.samples_per_epoch

        if self.causal and self.chunk_causal:
            # Process each epoch independently for quasi-causal operation.
            # Reshape: [B, T] -> [B*S, 1, samples_per_epoch]
            y = x.view(B, S, self.samples_per_epoch)
            y = y.reshape(B * S, 1, self.samples_per_epoch)
            y = self.cnn(y)
            # y shape: [B*S, C, 4] -> [B*S, 4, C] -> [B, S, epoch_dim]
            y = y.transpose(-1, -2).reshape(B, S, self.epoch_dim)
        else:
            # Non-causal path: process entire sequence at once.
            y = x.unsqueeze(1)  # Add channel dim [B, 1, T]
            y = self.cnn(y)
            # Re-shape for time-distributed dense layer.
            y = y.transpose(-1, -2).reshape(B, -1, self.epoch_dim)

        # Apply FC layer.
        y = self.linear(y)
        y = self.activation(y)
        y = self.output_norm(y)
        return y


class MultiModalAttentionEmbedder(nn.Module):
    """Block that combines feature vectors from multiple signals using attention."""

    def __init__(
        self,
        feature_dim: int,
        layers: int = 4,
        dropout: float = 0.0,
        dim_ff: int = 512,
        activation: str = "gelu",
        norm_first: bool = True,
        nhead: int = 4,
        register_tokens: int = 0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            dim_feedforward=dim_ff,
            activation=get_activation(activation),
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            norm_first=norm_first,
        )
        self.num_layers = layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )
        # Learnable matrix of the CLS token + any register tokens.
        self.num_register_tokens = register_tokens
        self.register_tokens = nn.Parameter(
            torch.randn(1, 1, self.feature_dim, register_tokens + 1)
        )

    def forward(self, z_dict: dict[str, Tensor]) -> Tensor:
        """Turn multi-modal inputs into tokens.

        Patches is a dictionary mapping from each signal patch to the
        corresponding embedder. Signal patches may be of different sizes
        e.g. due to different sampling rates. However, each embedder must
        transform the patch to the same feature dimension.
        """
        z_stack: list[Tensor] = []  # Feature vectors for each modality
        m_stack: list[Tensor] = []  # Mask vectors for each modality
        signals = [signal_name for signal_name in sorted(z_dict.keys())]
        if len(signals) == 0:
            raise ValueError("No signals provided to MultiModalAttentionEmbedder.")
        # Embedding different modalities.
        for signal_name in signals:
            z_BSF = z_dict[signal_name]
            B, S, *_ = z_BSF.size()
            # Find where channels are missing within a batch, set to zero for stability.
            m_B = torch.isinf(z_BSF).any(dim=(1, 2))
            z_BSF = torch.where(m_B[:, None, None], 0.0, z_BSF)
            # Concat.
            z_stack.append(z_BSF)
            m_stack.append(m_B)
        z_BSFC = torch.stack(z_stack, dim=-1)
        m_BC = torch.stack(m_stack, dim=-1)  # True where signals unavailable.
        B, S, F, C = z_BSFC.size()
        if F != self.feature_dim:
            raise ValueError(
                f"Feature dimension {F} does not match {self.feature_dim=}."
            )
        # Add CLS and register tokens per signal patch. D = C + R (no. register_tokens + 1)
        z_BSFD = torch.cat([self.register_tokens.repeat(B, S, 1, 1), z_BSFC], dim=-1)
        B, S, F, D = z_BSFD.size()
        # Create padding mask so transformer can't attend to unavailable signals.
        # Can always attend to CLS and any register tokens.
        m_BR = (
            torch.zeros_like(m_BC[:, 0])
            .bool()[:, None]
            .repeat(1, self.num_register_tokens + 1)
        )
        m_BD = torch.cat([m_BR, m_BC], dim=-1)
        # Reshape features for signal-wise per-patch attention.
        z_NDF = z_BSFD.flatten(start_dim=0, end_dim=1).permute(dims=(0, 2, 1))
        # Reshape mask for signal-wise per-patch attention.
        m_BSD = m_BD[:, None, :].repeat(1, S, 1)
        m_ND = m_BSD.flatten(start_dim=0, end_dim=1)
        # Apply transformer to signal-wise per-patch features. Reshape.
        z_NDF = self.transformer_encoder(z_NDF, src_key_padding_mask=m_ND)
        z_BSFD = z_NDF.permute(dims=(0, 2, 1)).reshape(B, S, F, D)
        # Return just the CLS token per-patch as the feature vector.
        z_BSF = z_BSFD[:, :, :, 0]
        return z_BSF


class SequenceCNN(nn.Module):
    """Simple dilated CNN model for sequence mixing."""

    def __init__(
        self,
        feature_dim: int = 128,
        dropout: float = 0.2,
        num_layers: int = 2,
        activation: str = "gelu",
        norm: str = "batch",
        causal: bool = False,
        num_dilations: int = 6,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.dilated_convs = nn.Sequential(
            *[
                DilatedConvBlock(
                    feature_dim=feature_dim,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    causal=causal,
                    num_dilations=num_dilations,
                    kernel_size=kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x_BSF: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            x (Tensor): shape [N, S, F]
        Returns:
            Tensor: shape [N, S, F]
        """
        # Re-shape back to channel-first for dilated convolutions.
        x_BFS = x_BSF.transpose(-1, -2)
        x_BFS = self.dilated_convs(x_BFS)
        return x_BFS.transpose(-1, -2)


__all__ = [
    "Wav2Sleep",
    "SignalEncoders",
    "SignalEncoder",
    "MultiModalAttentionEmbedder",
    "SequenceCNN",
    "ConvBlock1D",
    "ConvLayer1D",
    "DilatedConvBlock",
    "ConvLayerNorm",
    "SAMPLES_PER_EPOCH",
    "get_activation",
    "get_norm",
]
