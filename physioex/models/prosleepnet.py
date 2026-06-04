"""ProtoSleepTransformer — Quantized and robust SleepTransformer for sleep staging.

Applies prototype-based vector quantization (SimVQ) and robust multi-channel
training (ChannelsDropout, ChannelMixer) on top of the SleepTransformer
backbone (EpochTransformer + SequenceTransformer).

Architecture:
  Input: (B, L, C, T, F) spectrograms

  Per-channel EpochTransformer (in_chan=1):
    Each channel processed independently → (B*L*C, d_model) epoch embeddings
    Per-channel classification (mcy) for accuracy tracking

  Robustness:
    ChannelsDropout — accuracy-weighted channel resampling during training
    ChannelMixer — residual TransformerEncoder for cross-channel communication
    Mean pooling across channels → (B*L, d_model)

  SimVQ Quantization:
    Maps epoch embeddings to codebook prototypes
    Post-quantization classification (proto_y)

  SequenceTransformer:
    Temporal context across epochs → (B, L, d_model)

  Classification Head (SleepTransformer-style):
    FC(d_model → d_clf) → ReLU → Dropout → FC → ReLU → Dropout → FC → n_classes
    Output: (B, L, n_classes)
"""

import torch
from torch import nn

from vector_quantize_pytorch import SimVQ

from physioex.models.sleeptransformer import EpochTransformer, SequenceTransformer
from physioex.train.metrics import accuracy_score
from physioex.train.trainer import Trainer
from physioex.data.dataset import PhysioExDataset


class RandomChannelDropout(nn.Module):
    """Zero out entire input channels randomly during training.

    For each sample, each channel is independently dropped with probability
    p_drop. At least one channel is always kept.

    Input:  (B, L, C, T, F)
    Output: same shape, with dropped channels zeroed
    """

    def __init__(self, p_drop: float = 0.5):
        super().__init__()
        self.p_drop = p_drop

    def forward(self, x):
        if not self.training or self.p_drop == 0.0:
            return x

        B, L, C, T, F = x.shape
        keep = torch.rand(B, 1, C, 1, 1, device=x.device) >= self.p_drop

        # Ensure at least one channel is kept per sample
        all_dropped = ~keep.any(dim=2, keepdim=True)
        if all_dropped.any():
            rescue = torch.zeros_like(keep)
            rescue[:, :, torch.randint(C, (1,)).item()] = True
            keep = keep | (all_dropped & rescue)

        return x * keep.float()


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

        if torch.is_tensor(channel_acc):
            acc = channel_acc.to(device=device, dtype=torch.float32)
        else:
            acc = torch.tensor(channel_acc, device=device, dtype=torch.float32)
        if acc.ndim != 1 or acc.numel() != nchan:
            raise ValueError(
                f"channel_acc must be 1D with length {nchan}, got shape {tuple(acc.shape)}"
            )
        if not torch.isfinite(acc).all():
            raise ValueError("channel_acc must contain only finite values")
        if (acc < 0).any() or (acc > 1).any():
            raise ValueError("channel_acc values must be in [0, 1]")
        proba = 1.0 - acc
        proba_sum = proba.sum()
        if proba_sum <= 0:
            raise ValueError("channel_acc must not be all ones")
        proba = proba / proba_sum

        mask = torch.rand(batch, device=device) < self.dropout_prob

        idx = torch.multinomial(
            proba.expand(batch, -1), nchan, replacement=True
        )  # (batch, nchan)
        batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, nchan)

        x_shuffled = x.clone()
        x_shuffled[mask] = x[batch_idx[mask], idx[mask], :]

        return x_shuffled


class ZeroEmbeddingDropout(nn.Module):
    """Drop channels by replacing their embeddings with the encoder's
    zero-input embedding. Probability is accuracy-weighted.

    Unlike ChannelsDropout (which resamples embeddings from other channels),
    this produces the same output as a zeroed input channel would, making
    training consistent with test-time channel occlusion.
    """

    def __init__(self, p_apply: float = 0.5):
        super().__init__()
        self.p_apply = p_apply

    def forward(self, x, zero_emb, channel_acc):
        if not self.training or self.p_apply == 0.0:
            return x

        # x: (batch, nchan, hdim)
        batch, nchan, hdim = x.shape
        device = x.device

        if torch.is_tensor(channel_acc):
            acc = channel_acc.to(device=device, dtype=torch.float32)
        else:
            acc = torch.tensor(channel_acc, device=device, dtype=torch.float32)

        proba = 1.0 - acc
        proba_sum = proba.sum()
        if proba_sum <= 0:
            return x
        proba = proba / proba_sum

        # Which samples get dropout applied
        apply_mask = torch.rand(batch, device=device) < self.p_apply

        # For each sample, sample which channels to DROP
        # Higher drop probability for low-accuracy channels
        n_to_drop = torch.randint(1, nchan, (batch,), device=device)  # drop 1..nchan-1

        result = x.clone()
        for i in range(batch):
            if not apply_mask[i]:
                continue
            nd = n_to_drop[i].item()
            drop_idx = torch.multinomial(proba, nd, replacement=False)
            result[i, drop_idx] = zero_emb

        return result


def _initialize_residual_transformer(encoder: nn.TransformerEncoder):
    """Initialize TransformerEncoder to act as identity at training start.

    For residual connection: output = input + encoder(input)
    We want encoder(input) ~ 0 initially, so output ~ input.
    """
    for layer in encoder.layers:
        torch.nn.init.constant_(layer.linear1.weight, 0.0)
        torch.nn.init.constant_(layer.linear1.bias, 0.0)
        torch.nn.init.constant_(layer.linear2.weight, 0.0)
        torch.nn.init.constant_(layer.linear2.bias, 0.0)

        torch.nn.init.constant_(layer.self_attn.out_proj.weight, 0.0)
        torch.nn.init.constant_(layer.self_attn.out_proj.bias, 0.0)

        torch.nn.init.normal_(layer.self_attn.in_proj_weight, mean=0.0, std=0.01)
        if layer.self_attn.in_proj_bias is not None:
            torch.nn.init.constant_(layer.self_attn.in_proj_bias, 0.0)

    return encoder


class ProtoSleepTransformer(nn.Module):
    """Quantized and robust SleepTransformer.

    Combines per-channel EpochTransformer encoding with ChannelsDropout,
    ChannelMixer, SimVQ prototype quantization, and SequenceTransformer.

    Input:  (B, L, C, T, F)
    Output: (B, L, n_classes)
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_chan: int = 3,
        # EpochTransformer params
        d_model: int = 128,
        n_heads: int = 8,
        n_epoch_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        attention_size: int = 128,
        # ChannelMixer params
        cdropout: float = 0.5,
        cm_n_heads: int = 4,
        cm_d_ff: int = 256,
        cm_n_layers: int = 1,
        # SimVQ params
        n_prototypes: int = 15,
        # SequenceTransformer params
        n_seq_layers: int = 4,
        # Classifier params
        d_clf: int = 1024,
        # Input shape (stored for explain/reconstruct compatibility)
        T: int = 29,
        F: int = 129,
        # Ablation flags
        use_channel_mixer: bool = True,
        use_prototypes: bool = True,
        random_input_dropout: float = 0.0,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.in_chan = in_chan
        self.d_model = d_model
        self.T = T
        self.F = F
        self.use_channel_mixer = use_channel_mixer
        self.use_prototypes = use_prototypes

        # Random input-level channel dropout (for ablation variant 2)
        if random_input_dropout > 0.0:
            self.input_dropout = RandomChannelDropout(random_input_dropout)

        # Per-channel epoch encoder (in_chan=1 so d_model stays as-is)
        self.epoch_encoder = EpochTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_epoch_layers,
            d_ff=d_ff,
            dropout=dropout,
            attention_size=attention_size,
        )

        # Per-channel classifier for mcy (needed for accuracy-weighted dropout)
        # and proto_y (needed for prototype loss)
        if use_channel_mixer or use_prototypes:
            self.clf = nn.Linear(d_model, n_classes)

        # Channel dropout: ZeroEmbeddingDropout for mixer variants,
        # ChannelsDropout for legacy/non-mixer
        if use_channel_mixer:
            self.zero_dropout = ZeroEmbeddingDropout(p_apply=cdropout)
        else:
            self.channels_dropout = ChannelsDropout(dropout_prob=cdropout)
        self.register_buffer(
            "channels_acc",
            torch.full((in_chan,), 0.7, dtype=torch.float32),
        )
        self.channels_acc_list = [
            torch.full((in_chan,), 0.7, dtype=torch.float32)
        ]

        # Channel mixer (residual transformer)
        if use_channel_mixer:
            t_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=cm_n_heads,
                dim_feedforward=cm_d_ff,
                batch_first=True,
            )
            self.channel_mixer = _initialize_residual_transformer(
                nn.TransformerEncoder(t_layer, num_layers=cm_n_layers)
            )

        # SimVQ prototype quantization
        if use_prototypes:
            self.prototype = SimVQ(
                dim=d_model,
                codebook_size=n_prototypes,
                rotation_trick=True,
                channel_first=False,
            )

        # Sequence encoder
        self.sequence_encoder = SequenceTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_seq_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Final classifier (SleepTransformer-style FC head)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_clf),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_clf, d_clf),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_clf, n_classes),
        )

        # Intermediate outputs for multi-loss training
        self.mcy = None
        self.commit_loss = None
        self.proto_y = None

    def forward(self, x):
        # Random input-level channel dropout (ablation variant 2)
        if hasattr(self, "input_dropout"):
            x = self.input_dropout(x)

        batch_size, L, in_chans, T, F = x.size()

        # [1] Per-channel epoch encoding
        # Concatenate a zero input for the zero-embedding (mixer variants)
        x_flat = x.reshape(batch_size * L * in_chans, 1, T, F)
        if self.use_channel_mixer:
            zero_input = torch.zeros(1, 1, T, F, device=x_flat.device, dtype=x_flat.dtype)
            x_flat = torch.cat([x_flat, zero_input], dim=0)

        embs = self.epoch_encoder(x_flat)

        if self.use_channel_mixer:
            x = embs[:-1]           # (B*L*C, d_model) — real embeddings
            zero_emb = embs[-1:]    # (1, d_model) — zero embedding
        else:
            x = embs

        # Per-channel classification (mcy) on real embeddings (pre-dropout)
        if hasattr(self, "clf"):
            self.mcy = self.clf(x).reshape(batch_size, L, in_chans, -1)
        else:
            self.mcy = None

        # [2] Robustness: channel dropout + mixer
        x = x.reshape(batch_size * L, in_chans, -1)  # (B*L, C, d_model)

        if self.training:
            if len(self.channels_acc_list) > 0:
                mean_acc = torch.stack(self.channels_acc_list, dim=0).mean(dim=0)
                self.channels_acc.copy_(mean_acc)
                self.channels_acc_list.clear()

            if self.use_channel_mixer:
                x = self.zero_dropout(x, zero_emb, self.channels_acc)
            else:
                x = self.channels_dropout(x, self.channels_acc)

        if self.use_channel_mixer:
            x = x + self.channel_mixer(x)  # residual

        x = x.mean(dim=1)  # (B*L, d_model) — pool channels

        # [3] SimVQ quantization (optional)
        if self.use_prototypes:
            x, indexes, commit_loss = self.prototype(x)
            self.commit_loss = commit_loss
            self.proto_y = self.clf(x).reshape(batch_size, L, -1)
        else:
            self.commit_loss = None
            self.proto_y = None

        # [4] Sequence encoding
        x = x.reshape(batch_size, L, -1)
        x = self.sequence_encoder(x)  # (B, L, d_model)

        # [5] Classification
        x = x.reshape(batch_size * L, -1)
        x = self.classifier(x)
        x = x.reshape(batch_size, L, -1)

        return x

    def get_metrics(self):
        return {
            "mcy": self.mcy,
            "commit_loss": self.commit_loss,
            "proto_y": self.proto_y,
        }

    def update_channel_acc(self, acc_list):
        acc_tensor = torch.tensor(
            acc_list, device=self.channels_acc.device, dtype=torch.float32
        )
        self.channels_acc_list.append(acc_tensor)


class ProtoSleepTransformerTrainer(Trainer):
    @staticmethod
    def _step(
        model: torch.nn.Module,
        batch: dict,
        loss_fn: torch.nn.Module,
        device: torch.device,
    ) -> tuple[float, float, dict]:

        # Support dict batches (new BasePhysioDataset) and tuple batches (legacy)
        if isinstance(batch, dict) and "signals" in batch:
            from physioex.data.collate import stack_channels

            inputs = stack_channels(batch).to(device)
            targets = batch["labels"].to(device)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

        # No autocast: SimVQ commit_loss and multi-loss accumulation
        # are numerically unstable in float16
        outputs = model(inputs.float())

        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1)

        metrics = model.get_metrics()

        mcy = metrics.get("mcy", None)
        commit_loss = metrics.get("commit_loss", None)
        proto_y = metrics.get("proto_y", None)

        main_loss = loss_fn(outputs, targets)
        proto_loss = loss_fn(proto_y.reshape(-1, proto_y.shape[-1]), targets)
        commit_loss_value = commit_loss if commit_loss is not None else 0.0
        loss = main_loss + proto_loss + commit_loss_value

        # Per-channel losses and accuracy
        mcy = mcy.reshape(outputs.shape[0], -1, outputs.shape[-1])
        in_chans = mcy.shape[1]

        chan_acc = []
        chan_losses = []
        for chan in range(in_chans):
            chan_outputs = mcy[:, chan, :].reshape(-1, mcy.shape[-1])
            chan_loss = loss_fn(chan_outputs, targets)
            acc = accuracy_score(
                chan_outputs,
                targets,
                ignore_index=getattr(loss_fn, "ignore_index", None),
            )
            chan_acc.append(acc)
            chan_losses.append(chan_loss)
            loss += chan_loss

        if not model.training:
            model.update_channel_acc(chan_acc)

        acc = accuracy_score(
            outputs, targets, ignore_index=getattr(loss_fn, "ignore_index", None)
        )

        del inputs, targets, outputs

        extra_metrics = {
            "loss_main": float(
                main_loss.detach().item()
                if isinstance(main_loss, torch.Tensor)
                else main_loss
            ),
            "loss_proto": float(
                proto_loss.detach().item()
                if isinstance(proto_loss, torch.Tensor)
                else proto_loss
            ),
            "loss_commit": float(
                commit_loss_value.detach().item()
                if isinstance(commit_loss_value, torch.Tensor)
                else commit_loss_value
            ),
            "acc_eeg": chan_acc[0] if len(chan_acc) > 0 else None,
            "acc_eog": chan_acc[1] if len(chan_acc) > 1 else None,
            "acc_emg": chan_acc[2] if len(chan_acc) > 2 else None,
        }
        if chan_losses:
            chan_sum = 0.0
            for idx, chan_loss in enumerate(chan_losses):
                value = float(
                    chan_loss.detach().item()
                    if isinstance(chan_loss, torch.Tensor)
                    else chan_loss
                )
                extra_metrics[f"loss_chan_{idx}"] = value
                chan_sum += value
            extra_metrics["loss_chan_sum"] = chan_sum

        return loss, acc, extra_metrics


class AblationTrainer(Trainer):
    """Trainer for ablation variants with per-channel loss but no prototype loss.

    Used for ProtoSleepTransformer with use_prototypes=False (variant 3: dropout + mixer).
    Computes main loss + per-channel losses, tracks per-channel accuracy for
    accuracy-weighted ChannelsDropout. No proto_loss, no commit_loss.
    """

    @staticmethod
    def _step(
        model: torch.nn.Module,
        batch: dict,
        loss_fn: torch.nn.Module,
        device: torch.device,
    ) -> tuple[float, float, dict]:

        if isinstance(batch, dict) and "signals" in batch:
            from physioex.data.collate import stack_channels

            inputs = stack_channels(batch).to(device)
            targets = batch["labels"].to(device)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

        outputs = model(inputs.float())

        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1)

        main_loss = loss_fn(outputs, targets)
        loss = main_loss

        # Per-channel losses and accuracy (if mcy available)
        metrics = model.get_metrics()
        mcy = metrics.get("mcy", None)

        extra_metrics = {
            "loss_main": float(main_loss.detach().item()),
        }
        chan_acc = []

        if mcy is not None:
            mcy = mcy.reshape(outputs.shape[0], -1, outputs.shape[-1])
            in_chans = mcy.shape[1]

            for chan in range(in_chans):
                chan_outputs = mcy[:, chan, :].reshape(-1, mcy.shape[-1])
                chan_loss = loss_fn(chan_outputs, targets)
                acc_chan = accuracy_score(
                    chan_outputs,
                    targets,
                    ignore_index=getattr(loss_fn, "ignore_index", None),
                )
                chan_acc.append(acc_chan)
                loss += chan_loss
                extra_metrics[f"loss_chan_{chan}"] = float(chan_loss.detach().item())

            if not model.training:
                model.update_channel_acc(chan_acc)

            extra_metrics["acc_eeg"] = chan_acc[0] if len(chan_acc) > 0 else None
            extra_metrics["acc_eog"] = chan_acc[1] if len(chan_acc) > 1 else None
            extra_metrics["acc_emg"] = chan_acc[2] if len(chan_acc) > 2 else None

        acc = accuracy_score(
            outputs, targets, ignore_index=getattr(loss_fn, "ignore_index", None)
        )

        del inputs, targets, outputs

        return loss, acc, extra_metrics


if __name__ == "__main__":
    from torchinfo import summary

    x = torch.randn(4, 21, 3, 29, 129)

    # --- Full ProtoSleepTransformer ---
    print("=== Full ProtoSleepTransformer ===")
    model = ProtoSleepTransformer(n_classes=5, in_chan=3)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    y = model(x)
    assert y.shape == (4, 21, 5)
    y.sum().backward()
    m = model.get_metrics()
    assert m["mcy"].shape == (4, 21, 3, 5)
    assert m["proto_y"].shape == (4, 21, 5)
    assert m["commit_loss"] is not None
    print("  OK")

    # --- Ablation: mixer only (no prototypes) ---
    print("=== ProtoSleepTransformer (mixer, no prototypes) ===")
    model2 = ProtoSleepTransformer(n_classes=5, in_chan=3, use_channel_mixer=True, use_prototypes=False)
    print(f"  params: {sum(p.numel() for p in model2.parameters()):,}")
    y2 = model2(x)
    assert y2.shape == (4, 21, 5)
    y2.sum().backward()
    m2 = model2.get_metrics()
    assert m2["mcy"].shape == (4, 21, 3, 5)
    assert m2["proto_y"] is None
    assert m2["commit_loss"] is None
    print("  OK")

    # --- Ablation: random input dropout (no mixer, no prototypes) ---
    print("=== ProtoSleepTransformer (random input dropout, no mixer, no prototypes) ===")
    model3 = ProtoSleepTransformer(
        n_classes=5, in_chan=3,
        random_input_dropout=0.5, cdropout=0.0,
        use_channel_mixer=False, use_prototypes=False,
    )
    print(f"  params: {sum(p.numel() for p in model3.parameters()):,}")
    model3.train()
    y3 = model3(x)
    assert y3.shape == (4, 21, 5)
    y3.sum().backward()
    m3 = model3.get_metrics()
    assert m3["mcy"] is None  # no clf when no mixer and no prototypes
    assert m3["proto_y"] is None
    print("  OK")

    # --- Ablation: baseline (no dropout, no mixer, no prototypes) ---
    print("=== ProtoSleepTransformer (baseline, no dropout) ===")
    model4 = ProtoSleepTransformer(
        n_classes=5, in_chan=3,
        cdropout=0.0, use_channel_mixer=False, use_prototypes=False,
    )
    print(f"  params: {sum(p.numel() for p in model4.parameters()):,}")
    y4 = model4(x)
    assert y4.shape == (4, 21, 5)
    y4.sum().backward()
    print("  OK")

    print("\nAll variants passed.")
