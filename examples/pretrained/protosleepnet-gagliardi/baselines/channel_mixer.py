"""Channel mixer wrapper with modality embeddings and attention pooling.

Wraps per-channel epoch encoders with:
  - Learned modality embeddings (EEG/EOG/EMG)
  - Accuracy-weighted ZeroEmbeddingDropout
  - TransformerEncoder for cross-channel mixing (no residual)
  - Attention pooling (learned channel weights)
  - Per-channel classification head for accuracy tracking

The wrapper is model-agnostic: pass any epoch_encoder, sequence_encoder,
and classifier. Training uses MixerTrainer which adds per-channel loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from physioex.train.trainer import Trainer
from physioex.train.metrics import accuracy_score


# ── ZeroEmbeddingDropout ─────────────────────────────────────────────


class ZeroEmbeddingDropout(nn.Module):
    """Drop channels by replacing embeddings with the encoder's zero-input
    embedding. Drop probability is accuracy-weighted: low-accuracy channels
    are dropped more often.

    Only active during training.

    Args:
        p_apply: probability of applying dropout to each sample.
    """

    def __init__(self, p_apply: float = 0.5):
        super().__init__()
        self.p_apply = p_apply

    def forward(self, x, zero_emb, channel_acc):
        """
        Args:
            x: (batch, n_channels, d_model)
            zero_emb: (1, d_model) — encoder output for zero input
            channel_acc: (n_channels,) — per-channel accuracy
        """
        if not self.training or self.p_apply == 0.0:
            return x

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

        apply_mask = torch.rand(batch, device=device) < self.p_apply
        n_to_drop = torch.randint(1, nchan, (batch,), device=device)

        result = x.clone()
        for i in range(batch):
            if not apply_mask[i]:
                continue
            nd = n_to_drop[i].item()
            drop_idx = torch.multinomial(proba, nd, replacement=False)
            result[i, drop_idx] = zero_emb

        return result


# ── SeqSleepNet epoch encoder (standalone) ───────────────────────────


class SeqEpochEncoder(nn.Module):
    """Single-channel epoch encoder using SeqSleepNet components.

    FilterBank(in_chan=1) → BiLSTM → Attention → (batch, 2*hidden)
    """

    def __init__(
        self,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        lowfreq: int = 0,
        highfreq: int = 50,
        fs: int = 100,
        seqnhidden1: int = 64,
        seqnlayer1: int = 4,
        attentionsize: int = 32,
    ):
        super().__init__()
        from physioex.models.seqsleepnet import LearnableFilterbank, AttentionLayer

        self.filterbank = LearnableFilterbank(
            in_chan=1, F=F, nfilt=D, nfft=nfft, sf=fs,
            lowfreq=lowfreq, highfreq=highfreq,
        )
        self.lstm = nn.LSTM(
            D, seqnhidden1, num_layers=seqnlayer1,
            batch_first=True, bidirectional=True,
        )
        self.attention = AttentionLayer(2 * seqnhidden1, attentionsize)
        self.d_model = 2 * seqnhidden1

    def forward(self, x):
        """
        Args:
            x: (batch, 1, T, F) — single-channel spectrogram
        Returns:
            (batch, d_model) — epoch embedding
        """
        x = self.filterbank(x)          # (batch, 1, T, D)
        x = x.permute(0, 2, 1, 3)       # (batch, T, 1, D)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch, T, D)
        x, _ = self.lstm(x)             # (batch, T, 2*hidden)
        x = self.attention(x)           # (batch, 2*hidden)
        return x


# ── ChannelMixerWrapper ─────────────────────────────────────────────


class ChannelMixerWrapper(nn.Module):
    """Wraps per-channel epoch encoders with cross-channel mixing.

    Pipeline:
        (B, L, C, T, F)
        → per-channel epoch encoding: (B*L, C, d_model)
        → + modality embeddings
        → per-channel classification (mcy, for loss + accuracy tracking)
        → ZeroEmbeddingDropout (accuracy-weighted)
        → TransformerEncoder (mixer, NO residual)
        → attention pooling: (B*L, d_model)
        → sequence encoder → classifier: (B, L, n_classes)

    Args:
        epoch_encoder: maps (N, 1, T, F) → (N, d_model)
        sequence_encoder: maps (B, L, d_model) → (B, L, d_seq)
        classifier: maps (B*L, d_seq) → (B*L, n_classes)
        n_channels: number of input channels (default 3)
        n_classes: number of output classes (default 5)
        d_model: epoch encoder output dimension
        d_seq: sequence encoder output dimension (if different from d_model)
        cdropout: channel dropout probability (default 0.5)
        cm_n_heads: mixer transformer heads (default 4)
        cm_d_ff: mixer transformer feedforward dim (default 256)
        cm_n_layers: mixer transformer layers (default 1)
    """

    def __init__(
        self,
        epoch_encoder: nn.Module,
        sequence_encoder: nn.Module,
        classifier: nn.Module,
        n_channels: int = 3,
        n_classes: int = 5,
        d_model: int = 128,
        d_seq: int = None,
        cdropout: float = 0.5,
        cm_n_heads: int = 4,
        cm_d_ff: int = 256,
        cm_n_layers: int = 1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.d_model = d_model
        d_seq = d_seq or d_model

        # Core components (from the original model)
        self.epoch_encoder = epoch_encoder
        self.sequence_encoder = sequence_encoder
        self.classifier = classifier

        # Modality embeddings
        self.modality_emb = nn.Embedding(n_channels, d_model)

        # Per-channel classifier (for accuracy tracking + auxiliary loss)
        self.mcy = nn.Linear(d_model, n_classes)

        # Accuracy-weighted channel dropout
        self.zero_emb_dropout = ZeroEmbeddingDropout(p_apply=cdropout)

        # Channel mixer (NO residual)
        mixer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cm_n_heads,
            dim_feedforward=cm_d_ff,
            batch_first=True,
        )
        self.channel_mixer = nn.TransformerEncoder(
            mixer_layer, num_layers=cm_n_layers,
        )

        # Attention pooling (learned channel weights)
        self.attn_pool = nn.Linear(d_model, 1)

        # Accuracy tracking buffers
        self.register_buffer(
            "channels_acc",
            torch.ones(n_channels) / n_classes,
        )
        self.channels_acc_list: list[torch.Tensor] = []

        # Store last mcy output for trainer
        self._last_mcy = None

    def update_channel_acc(self, acc_list):
        """Update per-channel accuracy buffer from validation results."""
        acc_tensor = torch.tensor(acc_list, dtype=torch.float32)
        self.channels_acc_list.append(acc_tensor)

    def get_metrics(self):
        """Return last per-channel classification outputs for the trainer."""
        return {"mcy": self._last_mcy}

    def forward(self, x):
        """
        Args:
            x: (B, L, C, T, F) — multi-channel spectrograms

        Returns:
            (B, L, n_classes) — per-epoch logits
        """
        B, L, C, T, F_dim = x.shape

        # ── 1. Per-channel epoch encoding ────────────────────────────
        # Reshape: (B*L*C, 1, T, F)
        x_flat = x.reshape(B * L * C, 1, T, F_dim)

        # Append zero input for zero_emb computation
        zero_input = torch.zeros(1, 1, T, F_dim, device=x.device, dtype=x.dtype)
        x_with_zero = torch.cat([x_flat, zero_input], dim=0)

        embs_with_zero = self.epoch_encoder(x_with_zero)
        embs = embs_with_zero[:-1]       # (B*L*C, d_model)
        zero_emb = embs_with_zero[-1:]   # (1, d_model)

        # Reshape to (B*L, C, d_model)
        embs = embs.reshape(B * L, C, self.d_model)

        # ── 2. Add modality embeddings ───────────────────────────────
        channel_ids = torch.arange(C, device=x.device)
        embs = embs + self.modality_emb(channel_ids).unsqueeze(0)

        # ── 3. Per-channel classification (mcy) ─────────────────────
        mcy_logits = self.mcy(embs)  # (B*L, C, n_classes)
        self._last_mcy = mcy_logits.reshape(B, L, C, self.n_classes)

        # ── 4. Update accuracy buffer (during training) ─────────────
        if self.training and len(self.channels_acc_list) > 0:
            mean_acc = torch.stack(self.channels_acc_list, dim=0).mean(dim=0)
            self.channels_acc.copy_(mean_acc)
            self.channels_acc_list.clear()

        # ── 5. ZeroEmbeddingDropout ──────────────────────────────────
        embs = self.zero_emb_dropout(embs, zero_emb, self.channels_acc)

        # ── 6. Channel mixer (NO residual) ───────────────────────────
        embs = self.channel_mixer(embs)  # (B*L, C, d_model)

        # ── 7. Attention pooling ─────────────────────────────────────
        weights = F.softmax(self.attn_pool(embs), dim=1)  # (B*L, C, 1)
        embs = (embs * weights).sum(dim=1)  # (B*L, d_model)

        # ── 8. Sequence encoder ──────────────────────────────────────
        embs = embs.reshape(B, L, -1)

        if isinstance(self.sequence_encoder, nn.GRU):
            embs, _ = self.sequence_encoder(embs)
        else:
            embs = self.sequence_encoder(embs)

        # ── 9. Classifier ────────────────────────────────────────────
        d_seq = embs.shape[-1]
        logits = self.classifier(embs.reshape(B * L, d_seq))
        logits = logits.reshape(B, L, -1)

        return logits


# ── MixerTrainer ─────────────────────────────────────────────────────


class MixerTrainer(Trainer):
    """Trainer subclass that adds per-channel auxiliary loss and accuracy tracking.

    Overrides _step to:
    1. Compute main classification loss
    2. Add per-channel losses from model.get_metrics()["mcy"]
    3. Track per-channel accuracy during eval for dropout weighting
    """

    @staticmethod
    def _step(model, batch, loss_fn, device):
        # Extract inputs and targets
        if isinstance(batch, dict) and "signals" in batch:
            from physioex.data.collate import stack_channels
            inputs = stack_channels(batch).to(device)
            targets = batch["labels"].to(device)
        elif isinstance(batch, dict) and "embeddings" in batch:
            inputs = batch["embeddings"].to(device)
            targets = batch["labels"].to(device)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

        with torch.autocast(device.type if "cuda" in device.type else "cpu"):
            outputs = model(inputs)

        B, L, n_classes = outputs.shape
        outputs_flat = outputs.reshape(-1, n_classes)
        targets_flat = targets.reshape(-1)

        # Main loss
        loss = loss_fn(outputs_flat, targets_flat)

        # Per-channel auxiliary loss
        metrics = model.get_metrics()
        mcy = metrics["mcy"]  # (B, L, C, n_classes)
        C = mcy.shape[2]
        for c in range(C):
            chan_out = mcy[:, :, c].reshape(-1, n_classes)
            loss = loss + loss_fn(chan_out, targets_flat)

        # Accuracy (main output)
        acc = accuracy_score(
            outputs_flat, targets_flat,
            ignore_index=getattr(loss_fn, "ignore_index", None),
        )

        # Track per-channel accuracy during eval (for dropout weighting)
        if not model.training:
            chan_accs = []
            for c in range(C):
                chan_out = mcy[:, :, c].reshape(-1, n_classes)
                chan_acc = accuracy_score(
                    chan_out, targets_flat,
                    ignore_index=getattr(loss_fn, "ignore_index", None),
                )
                if isinstance(chan_acc, torch.Tensor):
                    chan_acc = chan_acc.item()
                chan_accs.append(chan_acc)
            model.update_channel_acc(chan_accs)

        return loss, acc
