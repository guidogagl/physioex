"""LaBraM — Large Brain Model (Jiang et al., ICLR 2024).

Pure encoder: (B, L, C, T) -> (B, L, D)
Variable channel count with channel-aware position embeddings.
Maps input channels to LABRAM_CHANNEL_ORDER for correct pos embed slicing.

Embeddings: forward_features -> (B, 200).

Preprocessing (pure PyTorch, differentiable):
  1. Strip all-zero channels
  2. x / 100 scaling (0.1 mV range → ~[-1,1])
  3. Resolve channel indices in LABRAM_CHANNEL_ORDER
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from physioex.models.foundation_base import FoundationEncoder
from physioex.models.foundation_preproc import strip_zero_channels, scale_div

logger = logging.getLogger("physioex.foundation")

_PRETRAIN_SFREQ = 200.0
_PRETRAIN_PATCH = 200  # 1 s at 200 Hz
_SCALE_DIVISOR = 100.0

# Weight-load allowlist
_ALLOWED_MISSING: Set[str] = {
    "fc_norm.weight",
    "fc_norm.bias",
    "position_embedding",
    "temporal_embedding",
}
_ALLOWED_UNEXPECTED_PREFIXES = ("mask_token", "lm_head")
_ALLOWED_UNEXPECTED_SUBSTRINGS = (
    ".gamma_1",
    ".gamma_2",
    ".q_norm.",
    ".k_norm.",
    "logit_scale",
)


def _unwrap_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def _remap_labram_keys(state: dict) -> dict:
    # Check if the state dict uses the original student-prefixed format
    has_student_keys = any(k.startswith("student.") for k in state)
    if not has_student_keys:
        # Already in braindecode format (e.g. from braindecode/Labram-Braindecode)
        return dict(state)

    remapped = {}
    for k, v in state.items():
        if not k.startswith("student."):
            continue
        k = k[len("student.") :]
        k = k.replace("pos_embed", "position_embedding")
        k = k.replace("time_embed", "temporal_embedding")
        if k.startswith("patch_embed."):
            suffix = k[len("patch_embed.") :]
            if suffix.startswith(("conv", "norm")):
                k = "patch_embed.temporal_conv." + suffix
        k = k.replace(".mlp.fc1.", ".mlp.0.")
        k = k.replace(".mlp.fc2.", ".mlp.2.")
        remapped[k] = v
    return remapped


def _resolve_labram_channels(
    x: torch.Tensor,
    ch_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    from braindecode.models.labram import LABRAM_CHANNEL_ORDER

    labram_to_idx = {ch.upper(): i for i, ch in enumerate(LABRAM_CHANNEL_ORDER)}

    keep_src, labram_indices, matched, dropped = [], [], [], []
    for i, name in enumerate(ch_names):
        idx = labram_to_idx.get(name.upper())
        if idx is not None:
            keep_src.append(i)
            labram_indices.append(idx + 1)  # +1 for CLS at position 0
            matched.append(name)
        else:
            dropped.append(name)

    if not keep_src:
        raise ValueError(
            f"LaBraM: no input channels matched LABRAM_CHANNEL_ORDER. ch_names={ch_names!r}"
        )

    keep_t = torch.tensor(keep_src, device=x.device, dtype=torch.long)
    x_matched = x[:, keep_t, :]
    input_chans = torch.tensor([0] + labram_indices, dtype=torch.long, device=x.device)
    return x_matched, input_chans, matched, dropped


class LaBraMEncoder(FoundationEncoder):
    """LaBraM with channel-aware position embeddings; lazy model build."""

    MODEL_NAME = "labram"
    PIPELINE_PRESET = "labram"
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
        resolved = ensure_checkpoint("labram", checkpoint_path)
        self._ckpt_path = resolved
        self._resolved_names = (
            [self._channel_map.get(ch, ch) for ch in self._channel_names]
            if self._channel_names
            else []
        )

        # Pre-load checkpoint state for lazy model building
        raw = torch.load(resolved, map_location="cpu", weights_only=False)
        state = _unwrap_state_dict(raw)
        state = _remap_labram_keys(state)

        self._pretrained_pos_embed = state.pop("position_embedding").cpu()
        self._pretrained_temp_embed = (
            state.pop("temporal_embedding").cpu()
            if "temporal_embedding" in state
            else None
        )
        self._state_for_loading = state

        # Determine pretraining time patches from temporal_embedding
        if self._pretrained_temp_embed is not None:
            self._pretrain_n_time_patches = int(
                self._pretrained_temp_embed.shape[1] - 1
            )
        else:
            self._pretrain_n_time_patches = 30  # fallback: 30 patches = 30s

        self._built_n_chans: Optional[int] = None
        self._input_chans: Optional[torch.Tensor] = None

        super().__init__(
            in_chan=in_chan,
            checkpoint_path=resolved,
            **kwargs,
        )

    def _build_encoder(self, **kwargs) -> nn.Module:
        # LaBraM needs to know the actual channel count at init.
        # We build with the number of resolved names (after LABRAM matching).
        # If we don't know yet, use in_chan as default.
        n_chans = kwargs.get("in_chan", 2)
        return self._create_labram_model(n_chans)

    def _create_labram_model(self, n_chans: int) -> nn.Module:
        from braindecode.models import Labram

        n_times = 6000  # 30s × 200Hz
        n_time_patches = n_times // _PRETRAIN_PATCH

        model = Labram(
            n_times=n_times,
            n_chans=n_chans,
            sfreq=_PRETRAIN_SFREQ,
            n_outputs=0,
        )
        model.patch_embed.segment_patch.learned_patcher = False

        # Load weights (strict with allowlist)
        missing, unexpected = model.load_state_dict(
            self._state_for_loading, strict=False
        )
        real_missing = [k for k in missing if k not in _ALLOWED_MISSING]
        real_unexpected = [
            k
            for k in unexpected
            if not any(k.startswith(p) for p in _ALLOWED_UNEXPECTED_PREFIXES)
            and not any(s in k for s in _ALLOWED_UNEXPECTED_SUBSTRINGS)
        ]
        if real_missing or real_unexpected:
            logger.warning(
                f"LaBraM: unexpected state_dict deltas: missing={real_missing[:5]}, unexpected={real_unexpected[:5]}"
            )

        # Set position embedding
        model.position_embedding.data = self._pretrained_pos_embed.to(
            model.position_embedding.device
        )

        # Set temporal embedding (slice or interpolate)
        if self._pretrained_temp_embed is not None and hasattr(
            model, "temporal_embedding"
        ):
            needed = 1 + n_time_patches
            pretrain_rows = self._pretrained_temp_embed.shape[1]
            if pretrain_rows >= needed:
                temp_embed = self._pretrained_temp_embed[:, :needed, :]
            else:
                temp_embed = torch.nn.functional.interpolate(
                    self._pretrained_temp_embed.permute(0, 2, 1),
                    size=needed,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
            model.temporal_embedding.data = temp_embed.to(
                model.temporal_embedding.device
            )

        return model

    def _get_embedding_dim(self) -> int:
        return 200

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        pass  # Loading done in _build_encoder

    def _rebuild_if_needed(self, n_chans: int, device: torch.device) -> None:
        """Rebuild encoder if channel count changed."""
        if self._built_n_chans != n_chans:
            self.encoder = self._create_labram_model(n_chans)
            self.encoder.eval()
            self.encoder.to(device)
            for p in self.encoder.parameters():
                p.requires_grad = False
            self._built_n_chans = n_chans

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C', T) with zero-strip + /100 scaling + channel resolve."""
        # Strip all-zero channels and resolve names
        if self._resolved_names:
            x_np, kept = strip_zero_channels(x)
            kept_names = (
                [self._resolved_names[i] for i in kept]
                if len(kept) < len(self._resolved_names)
                else self._resolved_names
            )
        else:
            x_np = x
            kept_names = self._resolved_names

        # Scale: x / 100
        x_np = scale_div(x_np, _SCALE_DIVISOR)

        # Resolve channel indices in LABRAM_CHANNEL_ORDER
        if kept_names:
            x_np, input_chans, matched, dropped = _resolve_labram_channels(
                x_np, kept_names
            )
            self._input_chans = input_chans
        else:
            C = x_np.shape[1]
            self._input_chans = torch.arange(
                C + 1, device=x_np.device, dtype=torch.long
            )

        # Rebuild model if channel count changed
        self._rebuild_if_needed(x_np.shape[1], x_np.device)

        return x_np

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C', T) -> (B, 200) embeddings."""
        features = self.encoder.forward_features(x, input_chans=self._input_chans)
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        return features
