"""ProtoSleepNet — Template-based sleep staging model with residual connections.

A modular architecture that wraps any epoch_encoder + sequence_encoder backbone
with two key innovations:

1. **Residual sequence encoder**: z = h + seq(h), making epoch embeddings
   directly quantizable without accuracy loss.

2. **Channel mixer** (optional, for multi-channel input): per-channel epoch
   encoding → modality embeddings → accuracy-weighted dropout →
   TransformerEncoder mixer (with residual) → attention pooling.

Architecture (solution D — dual residual):

    Input: (B, L, C, T, F)

    [If C > 1: per-channel processing]
        epoch_encoder(x) per channel → x = (B*L, C, d_model)
        + modality embeddings
        → mcy classifier (for accuracy tracking + CE_channel loss)
        → ZeroEmbeddingDropout (accuracy-weighted)
        → x' = x + mixer(x)            ← RESIDUAL 1 (channel mixer)
        → attention pooling → h = (B*L, d_model)

    [If C == 1: direct processing]
        epoch_encoder(x) → h = (B*L, d_model)

    z = h + sequence_encoder(h)         ← RESIDUAL 2 (sequence encoder)
    logits = classifier(z)              ← single classifier

    Loss = CE_main(logits) + Σ CE_channel[c](mcy)

Usage:
    # With SleepTransformer backbone, single channel
    model = ProtoSleepNet.from_sleep_transformer(n_channels=1)

    # With SeqSleepNet backbone, 3 channels + mixer
    model = ProtoSleepNet.from_seq_sleep_net(n_channels=3, use_channel_mixer=True)

    # Custom backbone
    model = ProtoSleepNet(epoch_encoder=my_enc, sequence_encoder=my_seq, classifier=my_clf)
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
            zero_emb: (1, d_model)
            channel_acc: (n_channels,)
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


# ── ChannelMixer ─────────────────────────────────────────────────────


class ChannelMixer(nn.Module):
    """Cross-channel mixing with modality embeddings, residual transformer,
    and attention pooling.

    Pipeline:
        x (B*L, C, d) → + modality_emb → dropout → x + mixer(x) → attn_pool → (B*L, d)

    Args:
        n_channels: number of input channels
        n_classes: number of output classes (for mcy)
        d_model: embedding dimension
        cdropout: channel dropout probability
        n_heads: mixer transformer heads
        d_ff: mixer transformer feedforward dim
        n_layers: mixer transformer layers
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 5,
        d_model: int = 128,
        cdropout: float = 0.5,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.d_model = d_model

        # Modality embeddings
        self.modality_emb = nn.Embedding(n_channels, d_model)

        # Per-channel classifier (for accuracy tracking)
        self.mcy = nn.Linear(d_model, n_classes)

        # Accuracy-weighted dropout
        self.dropout = ZeroEmbeddingDropout(p_apply=cdropout)

        # Transformer mixer
        mixer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
        )
        self.mixer = nn.TransformerEncoder(mixer_layer, num_layers=n_layers)

        # Attention pooling
        self.attn_pool = nn.Linear(d_model, 1)

        # Accuracy tracking buffers
        self.register_buffer("channels_acc", torch.ones(n_channels) / n_classes)
        self.channels_acc_list: list[torch.Tensor] = []

    def update_channel_acc(self, acc_list):
        acc_tensor = torch.tensor(acc_list, dtype=torch.float32)
        self.channels_acc_list.append(acc_tensor)

    def forward(self, x, zero_emb):
        """
        Args:
            x: (BL, C, d_model) — per-channel epoch embeddings
            zero_emb: (1, d_model) — zero-input embedding

        Returns:
            h: (BL, d_model) — pooled embedding
            mcy_logits: (BL, C, n_classes) — per-channel logits
        """
        BL, C, d = x.shape

        # Add modality embeddings
        channel_ids = torch.arange(C, device=x.device)
        x = x + self.modality_emb(channel_ids).unsqueeze(0)

        # Per-channel classification
        mcy_logits = self.mcy(x)  # (BL, C, n_classes)

        # Update accuracy buffer
        if self.training and len(self.channels_acc_list) > 0:
            mean_acc = torch.stack(self.channels_acc_list, dim=0).mean(dim=0)
            self.channels_acc.copy_(mean_acc)
            self.channels_acc_list.clear()

        # Dropout
        x = self.dropout(x, zero_emb, self.channels_acc)

        # Mixer with residual
        x = x + self.mixer(x)  # RESIDUAL 1

        # Attention pooling
        weights = F.softmax(self.attn_pool(x), dim=1)  # (BL, C, 1)
        h = (x * weights).sum(dim=1)  # (BL, d_model)

        return h, mcy_logits


# ── ProtoSleepNet ────────────────────────────────────────────────────


class ProtoSleepNet(nn.Module):
    """Template-based sleep staging model with dual residual connections.

    Wraps any epoch_encoder + sequence_encoder with:
    - Optional channel mixer (for multi-channel input)
    - Residual sequence encoder: z = h + seq(h)
    - Single classifier

    Args:
        epoch_encoder: maps (N, 1, T, F) → (N, d_model) for per-channel,
                       or (N, C, T, F) → (N, d_model) for single-channel
        sequence_encoder: maps (B, L, d_model) → (B, L, d_model)
        classifier: maps (N, d_model) → (N, n_classes)
        n_channels: number of input channels (1 = no mixer)
        n_classes: number of output classes
        d_model: epoch encoder output dimension
        use_channel_mixer: enable channel mixer (requires n_channels > 1)
        cdropout: channel dropout probability
        cm_n_heads: mixer transformer heads
        cm_d_ff: mixer transformer feedforward dim
        cm_n_layers: mixer transformer layers
    """

    def __init__(
        self,
        epoch_encoder: nn.Module,
        sequence_encoder: nn.Module,
        classifier: nn.Module,
        n_channels: int = 1,
        n_classes: int = 5,
        d_model: int = 128,
        use_channel_mixer: bool = False,
        cdropout: float = 0.5,
        cm_n_heads: int = 4,
        cm_d_ff: int = 256,
        cm_n_layers: int = 1,
    ):
        super().__init__()
        self.epoch_encoder = epoch_encoder
        self.sequence_encoder = sequence_encoder
        self.classifier = classifier
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.d_model = d_model

        # Channel mixer (only for multi-channel)
        self.channel_mixer = None
        if use_channel_mixer and n_channels > 1:
            self.channel_mixer = ChannelMixer(
                n_channels=n_channels,
                n_classes=n_classes,
                d_model=d_model,
                cdropout=cdropout,
                n_heads=cm_n_heads,
                d_ff=cm_d_ff,
                n_layers=cm_n_layers,
            )

        self._last_mcy = None
        self._epoch_logits = None

        # VQ codebook (optional, set via set_codebook())
        self.register_buffer("codebook", None)

    def set_codebook(self, codebook):
        """Set VQ codebook for prototype quantization.

        Args:
            codebook: (M, d_model) numpy array or torch tensor.
        """
        import numpy as np
        if isinstance(codebook, np.ndarray):
            codebook = torch.from_numpy(codebook).float()
        # Ensure codebook is on the same device as the model
        device = next(self.parameters()).device
        self.codebook = codebook.to(device)

    @torch.no_grad()
    def _quantize(self, h):
        """Replace each embedding with its nearest codebook entry.

        Args:
            h: (N, d_model) epoch embeddings.
        Returns:
            (N, d_model) quantized embeddings.
        """
        dist = torch.cdist(h.unsqueeze(0), self.codebook.unsqueeze(0)).squeeze(0)
        idx = dist.argmin(dim=1)
        return self.codebook[idx]

    def update_channel_acc(self, acc_list):
        if self.channel_mixer is not None:
            self.channel_mixer.update_channel_acc(acc_list)

    def get_metrics(self):
        return {
            "mcy": self._last_mcy,             # (B, L, C, n_classes) or None
            "epoch_logits": self._epoch_logits,  # (B, L, n_classes)
        }

    def encode(self, x, quantize=False):
        """Encode input spectrograms to contextualized per-epoch embeddings.

        Full pipeline: epoch encoding → [channel mixer] → [VQ] →
        deep supervision → residual sequence encoding.

        Args:
            x: (B, L, C, T, F) spectrogram input.
            quantize: if True, replace epoch embeddings with nearest
                      codebook entry. Requires set_codebook() first.

        Returns:
            (B, L, d_model) contextualized epoch embeddings.
        """
        B, L, C, T, F_dim = x.shape

        # ── Epoch encoding ───────────────────────────────────────
        if self.channel_mixer is not None:
            # Per-channel: (B*L*C, 1, T, F) → (B*L*C, d_model)
            x_flat = x.reshape(B * L * C, 1, T, F_dim)

            # Append zero input for zero_emb computation
            zero_input = torch.zeros(1, 1, T, F_dim, device=x.device, dtype=x.dtype)
            x_with_zero = torch.cat([x_flat, zero_input], dim=0)

            embs_with_zero = self.epoch_encoder(x_with_zero)
            embs = embs_with_zero[:-1]      # (B*L*C, d_model)
            zero_emb = embs_with_zero[-1:]   # (1, d_model)

            embs = embs.reshape(B * L, C, self.d_model)

            # Channel mixer (RESIDUAL 1 inside)
            h, mcy_logits = self.channel_mixer(embs, zero_emb)
            self._last_mcy = mcy_logits.reshape(B, L, C, self.n_classes)
        else:
            # Single/multi-channel without mixer
            x_flat = x.reshape(B * L, C, T, F_dim)
            h = self.epoch_encoder(x_flat)  # (B*L, d_model)
            self._last_mcy = None

        # ── VQ quantization (optional, no grad) ─────────────────
        if quantize:
            assert self.codebook is not None, "Call set_codebook() before quantize=True"
            h = h.reshape(B * L, -1)
            h = self._quantize(h)
            h = h.reshape(B, L, -1)

        # ── Epoch-level classification (deep supervision) ────────
        h = h.reshape(B, L, -1)
        d = h.shape[-1]
        self._epoch_logits = self.classifier(h.reshape(B * L, d)).reshape(B, L, -1)

        # ── Residual sequence encoding ───────────────────────────
        if isinstance(self.sequence_encoder, nn.GRU):
            seq_out, _ = self.sequence_encoder(h)
        else:
            seq_out = self.sequence_encoder(h)

        z = h + seq_out  # RESIDUAL 2

        return z

    def forward(self, x, quantize=False):
        """Forward pass: encode → classify.

        Args:
            x: (B, L, C, T, F) spectrogram input.
            quantize: if True, replace epoch embeddings with nearest
                      codebook entry. Requires set_codebook() first.

        Returns:
            (B, L, n_classes) per-epoch logits.
        """
        embeddings = self.encode(x, quantize=quantize)  # (B, L, d_model)

        B, L, D = embeddings.shape
        logits = self.classifier(embeddings.reshape(B * L, D)).reshape(B, L, -1)

        return logits

    # ── Factory methods ──────────────────────────────────────────

    @classmethod
    def from_sleep_transformer(
        cls,
        n_channels: int = 1,
        n_classes: int = 5,
        d_model: int = 128,
        n_heads: int = 8,
        n_epoch_layers: int = 4,
        n_seq_layers: int = 4,
        d_ff: int = 1024,
        d_clf: int = 1024,
        dropout: float = 0.1,
        attention_size: int = 128,
        **mixer_kwargs,
    ) -> "ProtoSleepNet":
        """Build ProtoSleepNet with SleepTransformer backbone."""
        from physioex.models.sleeptransformer import EpochTransformer, SequenceTransformer

        epoch_encoder = EpochTransformer(
            d_model=d_model, in_chan=1, n_heads=n_heads,
            n_layers=n_epoch_layers, d_ff=d_ff, dropout=dropout,
            attention_size=attention_size,
        )
        sequence_encoder = SequenceTransformer(
            d_model=d_model, n_heads=n_heads, n_layers=n_seq_layers,
            d_ff=d_ff, dropout=dropout,
        )
        classifier = nn.Sequential(
            nn.Linear(d_model, d_clf),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_clf, d_clf),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_clf, n_classes),
        )

        return cls(
            epoch_encoder=epoch_encoder,
            sequence_encoder=sequence_encoder,
            classifier=classifier,
            n_channels=n_channels,
            n_classes=n_classes,
            d_model=d_model,
            **mixer_kwargs,
        )

    @classmethod
    def from_seq_sleep_net(
        cls,
        n_channels: int = 1,
        n_classes: int = 5,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        lowfreq: int = 0,
        highfreq: int = 50,
        fs: int = 100,
        seqnhidden1: int = 64,
        seqnlayer1: int = 4,
        attentionsize: int = 32,
        seqnhidden2: int = 64,
        seqnlayer2: int = 4,
        **mixer_kwargs,
    ) -> "ProtoSleepNet":
        """Build ProtoSleepNet with SeqSleepNet backbone."""
        from physioex.models.seqsleepnet import LearnableFilterbank, AttentionLayer

        d_model = 2 * seqnhidden1

        class _SeqEpochEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.filterbank = LearnableFilterbank(
                    in_chan=1, F=F, nfilt=D, nfft=nfft, sf=fs,
                    lowfreq=lowfreq, highfreq=highfreq,
                )
                self.lstm = nn.LSTM(
                    D, seqnhidden1, num_layers=seqnlayer1,
                    batch_first=True, bidirectional=True,
                )
                self.attention = AttentionLayer(2 * seqnhidden1, attentionsize)

            def forward(self, x):
                x = self.filterbank(x)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x, _ = self.lstm(x)
                x = self.attention(x)
                return x

        epoch_encoder = _SeqEpochEncoder()
        sequence_encoder = nn.GRU(
            d_model, seqnhidden2, num_layers=seqnlayer2,
            batch_first=True, bidirectional=True,
        )
        d_seq = 2 * seqnhidden2
        classifier = nn.Linear(d_seq, n_classes)

        return cls(
            epoch_encoder=epoch_encoder,
            sequence_encoder=sequence_encoder,
            classifier=classifier,
            n_channels=n_channels,
            n_classes=n_classes,
            d_model=d_model,
            **mixer_kwargs,
        )


# ── ProtoSleepNetTrainer ─────────────────────────────────────────────


class ProtoSleepNetTrainer(Trainer):
    """Trainer for ProtoSleepNet with optional per-channel auxiliary loss.

    Adds CE loss on per-channel predictions (mcy) when channel mixer is active.
    Also tracks per-channel accuracy during eval for dropout weighting.
    """

    @staticmethod
    def _step(model, batch, loss_fn, device):
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

        # Main loss (post-residual 2)
        loss = loss_fn(outputs_flat, targets_flat)

        # Epoch-level loss (deep supervision on h, pre-sequence)
        metrics = model.get_metrics()
        epoch_logits = metrics.get("epoch_logits")
        if epoch_logits is not None:
            epoch_flat = epoch_logits.reshape(-1, n_classes)
            loss = loss + loss_fn(epoch_flat, targets_flat)

        # Per-channel auxiliary loss (if mixer active)
        mcy = metrics.get("mcy")
        if mcy is not None:
            C = mcy.shape[2]
            for c in range(C):
                chan_out = mcy[:, :, c].reshape(-1, n_classes)
                loss = loss + loss_fn(chan_out, targets_flat)

            # Track per-channel accuracy during eval
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

        acc = accuracy_score(
            outputs_flat, targets_flat,
            ignore_index=getattr(loss_fn, "ignore_index", None),
        )

        return loss, acc
