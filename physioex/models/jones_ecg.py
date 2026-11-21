"""JonesECGNet — ECG-only sleep staging (Jones, Itti & Sheth, 2024).

Clean re-implementation of the electrocardiography-only feed-forward network
from *Expert-level sleep staging using an electrocardiography-only feed-forward
neural network* (Comput. Biol. Med. 176:108545). The authors' code is
AGPL-3.0; this module is an independent MIT implementation written from the
published functional description. The authors' released checkpoints
(``adammj/ecg-sleep-staging`` on GitHub) are loaded at runtime with
:meth:`JonesECGNet.from_original` — the state-dict key names and shapes are
reproduced exactly (see ``tests/models/fixtures/cardioresp/state_dict_shapes.json``)
so they load with ``strict=True`` and no remapping.

Architecture (all 7 variants share the same hyper-parameters; only the TCN
convolution style and the auxiliary inputs differ):

* ``ecg_cnns``: 6 strided conv stages — per stage ``BatchNorm1d`` → (for all
  but the first stage) ``LeakyReLU`` → ``Conv1d(k=31, stride=2, no padding)``;
  channels 1→5, then 5→5. A 7680-sample epoch (256 Hz × 30 s) compresses to
  length 91.
* Pooling: global ``AvgPool1d``/``MaxPool1d`` (kernel 19, stride 3 on the
  91-length feature) → 25 features each; ``ecg_dense`` maps the 250-dim
  concatenation (avg first) to 40.
* ``additional_dense``: 4→4 on the per-night auxiliaries.
* ``inner_dense``: 4 residual dense blocks (44 → 25), then the padding mask
  zeroes padded epochs.
* ``tcn``: 12 dilated residual conv blocks over epochs (width 25).
  Symmetric variants: ``k=3`` non-causal with equal chomp at both ends;
  real-time variants: ``k=2`` causal with the chomp at the end only.
* ``outer_dense``: 4 residual dense blocks (25 → 5) → softmax.

Public API:

    model = JonesECGNet.from_original(".../epoch_0647_kappa_0.724.pt", "primary")
    logp = model(ecg)              # (B, L, 5) log-probabilities
    logp = model(ecg, additional, padding_mask)

Inputs:
    ecg: (B, L, 1, 7680) or (B, L, 7680) float32 in [-1, 1] (256 Hz epochs).
    additional: (B, L, 4); if ``None`` it is built with
    :meth:`JonesECGNet.build_additional` — real values when ``age``/``sex``/
    ``midnight_offset`` are available for the variant, standard-normal noise
    otherwise (as during the authors' inference of the input-occluded models).
    padding_mask: (B, L) or (B, L, 1), 1 = real epoch, 0 = padding.
"""

from __future__ import annotations

import datetime
import logging
import math
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

import torch
from torch import nn


# ── Variant registry ─────────────────────────────────────────────────

#: The seven released checkpoint variants: folder under
#: ``adammj/ecg-sleep-staging/models/``, TCN style and the metadata the model
#: was trained with (``needs``); columns absent from ``needs`` are filled with
#: standard-normal noise at inference.
VARIANTS: Dict[str, Dict] = {
    "primary": {"folder": "1_paper_models/primary_model", "causal": False, "cyclic": False, "needs": {"age", "sex", "start_time"}},
    "realtime": {"folder": "1_paper_models/real-time_model", "causal": True, "cyclic": False, "needs": {"age", "sex", "start_time"}},
    "no-demographics": {"folder": "2_post-paper_models/primary_model_no_demographics", "causal": False, "cyclic": False, "needs": {"start_time"}},
    "no-time": {"folder": "2_post-paper_models/primary_model_no_time", "causal": False, "cyclic": False, "needs": {"age", "sex"}},
    "only-ecg": {"folder": "2_post-paper_models/primary_model_only_ecg", "causal": False, "cyclic": False, "needs": set()},
    "cyclic": {"folder": "3_cyclic-time_models/primary_model", "causal": False, "cyclic": True, "needs": {"age", "sex", "start_time"}},
    "cyclic-realtime": {"folder": "3_cyclic-time_models/real-time_model", "causal": True, "cyclic": True, "needs": {"age", "sex", "start_time"}},
}

#: Number of 30-s epochs per day (256 Hz × 30 s).
_EPOCHS_PER_DAY = 2880


def _conv_out_len(n: int, kernel_size: int, stride: int, n_layers: int) -> int:
    """Output length of ``n_layers`` strided convs (no padding) applied to ``n`` samples."""
    for _ in range(n_layers):
        n = (n - kernel_size) // stride + 1
    return n


# ── ECG convolutional front-end ──────────────────────────────────────


class ECGConvNet(nn.Module):
    """Six strided-conv stages: ``BatchNorm1d`` → ``LeakyReLU`` → ``Conv1d(31, stride=2)``.

    The first stage has no activation (the input is the raw ECG). A
    7680-sample epoch (256 Hz, 30 s) comes out at length 91 with 5 channels.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 5, kernel_size: int = 31, n_layers: int = 6):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            layers.append(nn.BatchNorm1d(in_ch))
            if i > 0:
                layers.append(nn.LeakyReLU())
            layers.append(nn.Conv1d(in_ch, hidden_channels, kernel_size, stride=2, bias=True))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── Dense residual blocks ────────────────────────────────────────────


class DenseBlockRes(nn.Module):
    """One residual dense block: ``BatchNorm1d`` → [``Dropout``] → ``LeakyReLU`` → ``Linear`` plus a (downsampled) skip."""

    def __init__(self, n_in: int, n_out: int, p: float = 0.0):
        super().__init__()
        layers = [nn.BatchNorm1d(n_in)]
        if p > 0:
            layers.append(nn.Dropout(p))
        layers.extend([nn.LeakyReLU(), nn.Linear(n_in, n_out)])
        self.network = nn.Sequential(*layers)
        self.downsample = nn.Linear(n_in, n_out) if n_in != n_out else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return out + x


class LayeredDenseBlockRes(nn.Module):
    """A stack of :class:`DenseBlockRes` with an outer residual.

    ``count`` blocks map ``[c0, c1, c2]``: block 0 is ``c0→c1``
    (``c0→c2`` when ``count == 1``), middle blocks ``c1→c1``, the last block
    ``c1→c2``; the whole stack adds its (downsampled) input back in.
    """

    def __init__(self, count: int, channels, p: float = 0.0):
        super().__init__()
        c0, c1, c2 = channels
        blocks = []
        for i in range(count):
            if count == 1:
                n_in, n_out = c0, c2
            elif i == 0:
                n_in, n_out = c0, c1
            elif i == count - 1:
                n_in, n_out = c1, c2
            else:
                n_in, n_out = c1, c1
            blocks.append(DenseBlockRes(n_in, n_out, p=p))
        self.network = nn.Sequential(*blocks)
        self.downsample = nn.Linear(c0, c2) if c0 != c2 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return out + x


# ── TCN building blocks ──────────────────────────────────────────────


class Chomp(nn.Module):
    """Crop a 1-D convolution output back to the input length.

    Causal mode removes ``chomp`` (= padding) samples from the end;
    symmetric mode removes ``chomp`` (= padding // 2) samples from both ends.
    """

    def __init__(self, chomp: int, causal: bool):
        super().__init__()
        self.chomp = chomp
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            return x[:, :, :-self.chomp] if self.chomp > 0 else x
        return x[:, :, self.chomp : -self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    """One TCN stage: ``BatchNorm1d`` → ``LeakyReLU`` → dilated ``Conv1d`` → :class:`Chomp`, plus a skip.

    Channels-first over epochs: (B, C, L). Symmetric blocks use ``k=3`` and
    chomp ``d`` samples at each end (output length == input length); causal
    blocks use ``k=2`` and chomp ``d`` samples from the end only.
    """

    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, causal: bool):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        chomp = padding // 2 if not causal else padding
        self.network = nn.Sequential(
            nn.BatchNorm1d(n_in),
            nn.LeakyReLU(),
            nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation),
            Chomp(chomp, causal=causal),
        )
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return out + x


class TemporalConvNet(nn.Module):
    """Stack of dilated :class:`TemporalBlock` with an outer residual."""

    def __init__(self, width: int, n_layers: int, causal: bool):
        super().__init__()
        kernel_size = 2 if causal else 3
        blocks = [
            TemporalBlock(width, width, kernel_size, dilation=2**i, causal=causal)
            for i in range(n_layers)
        ]
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) + x


# ── The model ────────────────────────────────────────────────────────


class JonesECGNet(nn.Module):
    """ECG-only sleep staging network of Jones, Itti & Sheth (2024).

    Args:
        variant: one of :data:`VARIANTS` (default ``"primary"``).
        n_classes: number of sleep classes (5: W, N1, N2, N3, REM).

    Forward input:
        x: (B, L, 1, 7680) or (B, L, 7680) float32 ECG, 256 Hz, in [-1, 1].
        additional: (B, L, 4) auxiliary features; if ``None`` it is built with
            :meth:`build_additional`, filling the columns the variant was not
            trained with by standard-normal noise.
        padding_mask: (B, L) or (B, L, 1) bool/float, 1 = real epoch; default
            all ones.

    Forward returns:
        (B, L, n_classes) log-probabilities over [W, N1, N2, N3, REM].
    """

    def __init__(self, variant: str = "primary", n_classes: int = 5):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        info = VARIANTS[variant]
        self.variant = variant
        self.causal = info["causal"]
        self.cyclic = info["cyclic"]
        self.n_classes = n_classes

        # Per-epoch ECG front-end (7680 samples -> 91 frames x 5 channels)
        self.ecg_cnns = ECGConvNet(in_channels=1, hidden_channels=5, kernel_size=31, n_layers=6)
        # Global pools over the 91-frame output: stride 91 // 25 = 3, kernel 91 - 24*3 = 19
        in_len = _conv_out_len(7680, kernel_size=31, stride=2, n_layers=6)  # 91
        pool_stride = in_len // 25
        pool_kernel = in_len - (25 - 1) * pool_stride
        self.avg_pool = nn.AvgPool1d(pool_kernel, stride=pool_stride)
        self.max_pool = nn.MaxPool1d(pool_kernel, stride=pool_stride)
        # Dense projections
        self.ecg_dense = nn.Linear(2 * 25 * 5, 40)
        self.additional_dense = nn.Linear(4, 4)
        # Residual dense blocks + TCN
        self.inner_dense = LayeredDenseBlockRes(4, [44, 25, 25], p=0.2)
        self.tcn = TemporalConvNet(width=25, n_layers=12, causal=self.causal)
        self.outer_dense = LayeredDenseBlockRes(4, [25, 25, n_classes], p=0.2)

    # ── forward ─────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        additional: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run per-epoch inference; returns (B, L, n_classes) log-probabilities."""
        if x.dim() == 3:
            x = x.unsqueeze(2)
        if x.dim() != 4 or x.shape[2] != 1:
            raise ValueError(f"x must have shape (B, L, 1, 7680) or (B, L, 7680), got {tuple(x.shape)}")
        batch, seqlen, _, samples = x.shape

        if additional is None:
            additional = self.build_additional(seqlen, self.variant)
            additional = additional.unsqueeze(0).expand(batch, -1, -1)
        additional = additional.to(device=x.device, dtype=x.dtype)

        mask = padding_mask if padding_mask is not None else torch.ones(batch, seqlen, dtype=x.dtype, device=x.device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        mask = mask.to(x.dtype)

        # Per-epoch ECG features: (B, L, 40)
        y = x.reshape(batch * seqlen, 1, samples)
        y = self.ecg_cnns(y)
        pooled = torch.cat([self.avg_pool(y).flatten(1), self.max_pool(y).flatten(1)], dim=1)
        ecg_feats = self.ecg_dense(pooled)  # (B*L, 40)

        # Per-epoch 44-dim features (additional first)
        add_feats = self.additional_dense(additional.reshape(batch * seqlen, 4))  # (B*L, 4)
        feats = torch.cat([add_feats, ecg_feats], dim=1)  # (B*L, 44)

        # Inner residual dense blocks, then mask out padded epochs
        feats = self.inner_dense(feats)  # (B*L, 25)
        feats = feats.reshape(batch, seqlen, -1) * mask  # (B, L, 25)

        # Temporal TCN over epochs (channels-first)
        feats = self.tcn(feats.transpose(1, 2))  # (B, 25, L)
        feats = feats.transpose(1, 2).reshape(batch * seqlen, -1)  # (B*L, 25)

        # Outer residual dense blocks -> softmax -> log-probabilities
        logits = self.outer_dense(feats)  # (B*L, n_classes)
        probs = torch.softmax(logits, dim=-1)
        return torch.log(probs.clamp(min=1e-12)).reshape(batch, seqlen, self.n_classes)

    # ── auxiliary features ──────────────────────────────────────────

    @staticmethod
    def _fallback_additional(n_epochs: int, variant: str, device: "torch.device") -> "torch.Tensor":
        """Noise-filled (n_epochs, 4) auxiliaries for a variant whose required
        inputs are all missing.

        The draws follow :meth:`build_additional` exactly (one draw of size 2
        for demographics per night; one draw per epoch per column for the time
        columns), so the result is statistically identical to running the
        authors' inference of the input-occluded models.
        """
        info = VARIANTS[variant]
        generator = torch.Generator(device=device)
        if "age" in info["needs"] and "sex" in info["needs"]:
            demo = torch.randn(2, device=device, generator=generator)
        else:
            demo = torch.randn(2, device=device, generator=generator)
        if "start_time" in info["needs"] and info["cyclic"]:
            t = 2.0 * math.pi * torch.arange(
                n_epochs, dtype=torch.float32, device=device
            ) / _EPOCHS_PER_DAY
            time = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        else:
            time = torch.randn(n_epochs, 2, device=device, generator=generator)
        return torch.cat([demo.unsqueeze(0).expand(n_epochs, -1), time], dim=1)

    @staticmethod
    def midnight_offset(start: "datetime.time | datetime.datetime") -> float:
        """Signed offset in days of ``start`` from the nearest midnight.

        Positive when the start is in the (next) morning, negative in the
        (previous) evening: 01:30 → +1.5/24, 23:00 → -1/24.
        """
        if isinstance(start, datetime.datetime):
            start = start.time()
        minutes = start.hour * 60 + start.minute + start.second / 60.0
        offset = minutes / (24.0 * 60)
        if offset > 0.5:
            offset -= 1.0
        return offset

    @staticmethod
    def build_additional(
        n_epochs: int,
        variant: str,
        age: Optional[float] = None,
        sex: Optional[float] = None,
        midnight_offset: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Build the (n_epochs, 4) auxiliary-feature matrix for one night.

        Columns: [sex (0=F, 1=M), age/100, night_e (−1→1 across the night),
        midnight_e]. For cyclic variants the last two columns are replaced
        by (cos t_e, sin t_e) with ``t_e = 2π(e/2880 + midnight_offset mod 1)``.
        Columns the variant was not trained with (see :data:`VARIANTS`) are
        filled with standard-normal noise drawn from ``generator``: one draw
        per epoch per column, except demographics which are drawn once per
        night (size 2) and repeated over the epochs.

        Raises:
            ValueError: if ``age``, ``sex`` or ``midnight_offset`` is missing
                for a variant that was trained with it.
        """
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        info = VARIANTS[variant]

        missing = [
            name
            for name, value in (("age", age), ("sex", sex), ("start_time", midnight_offset))
            if value is None and name in info["needs"]
        ]
        if missing:
            raise ValueError(f"variant {variant!r} needs {', '.join(missing)} but it was not provided")

        def _noise(shape):
            return torch.randn(shape, generator=generator)

        # Demographics: real values, or ONE noise draw of size 2 per night
        # (repeated over the epochs) when the variant was trained without them.
        if "age" in info["needs"] and "sex" in info["needs"]:
            demo = torch.tensor([sex, age / 100.0], dtype=torch.float32)
        else:
            demo = _noise((2,))

        # Time columns
        if "start_time" in info["needs"]:
            if info["cyclic"]:
                t = 2.0 * math.pi * (torch.arange(n_epochs, dtype=torch.float32) / _EPOCHS_PER_DAY + (midnight_offset % 1.0))
                time = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
            else:
                e = torch.arange(n_epochs, dtype=torch.float32)
                night_e = e / ((n_epochs - 1) / 2.0) - 1.0 if n_epochs > 1 else torch.zeros(n_epochs)
                time = torch.stack([night_e, e / _EPOCHS_PER_DAY + midnight_offset], dim=1)
        else:
            time = _noise((n_epochs, 2))  # one draw per epoch per column

        return torch.cat([demo.unsqueeze(0).expand(n_epochs, -1), time], dim=1)

    # ── checkpoint loading ──────────────────────────────────────────

    @classmethod
    def from_original(cls, checkpoint_path: "str | Path", variant: str) -> "JonesECGNet":
        """Load one of the authors' released checkpoints.

        The checkpoints are plain dicts with a ``model_state_dict`` entry whose
        keys this re-implementation reproduces exactly, so the load is strict
        and needs no remapping.

        Args:
            checkpoint_path: path to the ``.pt`` checkpoint (e.g. downloaded
                from ``adammj/ecg-sleep-staging``).
            variant: variant name (used to build the matching architecture).

        Returns:
            The model in ``eval()`` mode.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        model = cls(variant=variant)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    # ── whole-night inference ─────────────────────────────────────────

    def predict_night(self, signals: "dict[str, np.ndarray]", metadata: "dict | None" = None) -> "np.ndarray":
        """Classify one whole night, already preprocessed by the ``jones_ecg`` preset.

        Args:
            signals: Mapping of signal name to a ``(L, 7680)`` float array of
                one night (256 Hz, 30-s epochs, in [-1, 1]); the night's
                ``"ECG"`` channel is required. Unlike the fixed-grid models,
                JonesECGNet accepts any number of epochs, so the night is not
                padded or truncated here.
            metadata: optional dict with ``"age"`` (years), ``"sex"``
                (``"M"``/``"F"``) and ``"recording_start"``
                (``datetime.time``). The auxiliary features are built with
                :meth:`build_additional` for the model's variant; a missing
                value that the variant was not trained with (see
                :data:`VARIANTS`) is filled with the variant's noise fallback,
                with one warning per missing key.

        Returns:
            int64 class indices of shape ``(L,)`` in AASM 5-class order
            [W, N1, N2, N3, REM]. Runs under ``torch.no_grad()`` in eval mode,
            batch of one, on the model's current device.
        """
        import numpy as np

        if "ECG" not in signals:
            raise ValueError(f"JonesECGNet needs an 'ECG' signal, got {sorted(signals)}")
        arr = np.asarray(signals["ECG"], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 7680:
            raise ValueError(f"'ECG' must have shape (L, 7680 samples/epoch), got {arr.shape}")
        L = arr.shape[0]

        metadata = metadata or {}
        age = metadata.get("age")
        sex_raw = metadata.get("sex")
        sex = {"M": 1.0, "F": 0.0}.get(str(sex_raw).upper()) if sex_raw is not None else None
        start = metadata.get("recording_start")
        if start is not None and not isinstance(start, (datetime.time, datetime.datetime)):
            raise ValueError(
                f"metadata['recording_start'] must be a datetime.time or datetime.datetime, "
                f"got {type(start).__name__}"
            )

        for name, value in (("age", age), ("sex", sex), ("recording_start", start)):
            if value is None:
                logger.warning(
                    "JonesECGNet.predict_night: metadata %r is missing for variant %r; "
                    "using the variant's noise fallback for that input",
                    name,
                    self.variant,
                )

        device = next(self.parameters()).device
        # Start from the variant's noise fallback (the authors' way of running
        # input-occluded models) and overwrite the columns whose inputs are
        # known, so a partly missing metadata dict never raises.
        additional = self._fallback_additional(L, self.variant, torch.device("cpu"))
        needs = VARIANTS[self.variant]["needs"]
        if age is not None and sex is not None and {"age", "sex"} <= needs:
            additional[:, 0] = float(sex)
            additional[:, 1] = float(age) / 100.0
        if start is not None and "start_time" in needs:
            full = self.build_additional(
                L, self.variant, age=0.0, sex=0.0, midnight_offset=self.midnight_offset(start)
            )
            additional[:, 2:] = full[:, 2:]
        additional = additional.unsqueeze(0)
        x = torch.from_numpy(arr).reshape(1, L, 1, arr.shape[1]).to(device)
        additional = additional.to(device)

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                preds = self.forward(x, additional=additional)[0].argmax(dim=-1).cpu().numpy()
        finally:
            if was_training:
                self.train()
        return preds.astype(np.int64)
