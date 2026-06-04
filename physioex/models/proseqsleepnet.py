"""ProtoSeqSleepNet — Quantized and robust SeqSleepNet for sleep staging.

Applies prototype-based vector quantization (SimVQ) and robust multi-channel
training (ChannelsDropout, ChannelMixer) on top of the SeqSleepNet backbone
(LearnableFilterbank + BiLSTM + Attention + BiGRU).

Architecture:
  Input: (B, L, C, T, F) spectrograms

  Per-channel epoch encoding:
    LearnableFilterbank → BiLSTM → AttentionLayer → (B*L*C, hidden) embeddings

  Robustness:
    ChannelsDropout — accuracy-weighted channel resampling during training
    ChannelMixer — residual TransformerEncoder for cross-channel communication
    Mean pooling across channels → (B*L, hidden)

  SimVQ Quantization (optional):
    Maps epoch embeddings to codebook prototypes
    Post-quantization classification (proto_y)

  BiGRU:
    Temporal context across epochs → (B, L, hidden)

  Classification:
    Linear(hidden → n_classes)
    Output: (B, L, n_classes)
"""

import torch
from torch import nn

from vector_quantize_pytorch import SimVQ

from physioex.models.seqsleepnet import LearnableFilterbank, AttentionLayer
from physioex.models.prosleepnet import (
    RandomChannelDropout,
    ChannelsDropout,
    ZeroEmbeddingDropout,
    _initialize_residual_transformer,
    AblationTrainer,
)
from physioex.train.metrics import accuracy_score
from physioex.train.trainer import Trainer


class SeqSleepNetEpochEncoder(nn.Module):
    """Per-channel epoch encoder: LearnableFilterbank + BiLSTM + Attention.

    Input:  (N, 1, T, F) — N samples, 1 channel each
    Output: (N, hidden) — where hidden = 2 * seqnhidden1
    """

    def __init__(self, F, D, in_chan, nfft, fs, lowfreq, highfreq,
                 seqnhidden1, seqnlayer1, attentionsize):
        super().__init__()
        self.filterbank = LearnableFilterbank(1, F, D * in_chan, nfft, fs, lowfreq, highfreq)
        self.seqn1 = nn.LSTM(
            D * in_chan, seqnhidden1,
            num_layers=seqnlayer1, batch_first=True, bidirectional=True,
        )
        self.attention = AttentionLayer(2 * seqnhidden1, attentionsize)

    def forward(self, x):
        # x: (N, 1, T, F)
        T = x.shape[2]
        x = self.filterbank(x)        # (N, T, 1, D*in_chan)
        x = x.permute(0, 2, 1, 3)     # (N, 1, T, D*in_chan)
        x = x.reshape(x.shape[0], T, -1)  # (N, T, D*in_chan)
        x, _ = self.seqn1(x)          # (N, T, hidden)
        x = self.attention(x)         # (N, hidden)
        return x


class ProtoSeqSleepNet(nn.Module):
    """Quantized and robust SeqSleepNet.

    Combines per-channel SeqSleepNet encoding (filterbank + BiLSTM + attention)
    with ChannelsDropout, ChannelMixer, SimVQ prototype quantization, and BiGRU.

    Input:  (B, L, C, T, F)
    Output: (B, L, n_classes)
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_chan: int = 3,
        # Filterbank params
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        lowfreq: int = 0,
        highfreq: int = 50,
        fs: int = 100,
        # BiLSTM (epoch encoding) params
        seqnhidden1: int = 64,
        seqnlayer1: int = 4,
        # Attention params
        attentionsize: int = 32,
        # ChannelMixer params
        cdropout: float = 0.5,
        cm_n_heads: int = 4,
        cm_d_ff: int = 256,
        cm_n_layers: int = 1,
        # SimVQ params
        n_prototypes: int = 100,
        # BiGRU (sequence) params
        seqnhidden2: int = 64,
        seqnlayer2: int = 4,
        # Ablation flags
        use_channel_mixer: bool = True,
        use_prototypes: bool = True,
        random_input_dropout: float = 0.0,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.in_chan = in_chan
        self.D = D
        self.use_channel_mixer = use_channel_mixer
        self.use_prototypes = use_prototypes

        hidden = 2 * seqnhidden1  # BiLSTM output size

        # Random input-level channel dropout (ablation variant 2)
        if random_input_dropout > 0.0:
            self.input_dropout = RandomChannelDropout(random_input_dropout)

        # Per-channel epoch encoder (filterbank + BiLSTM + attention)
        self.epoch_encoder = SeqSleepNetEpochEncoder(
            F, D, in_chan, nfft, fs, lowfreq, highfreq,
            seqnhidden1, seqnlayer1, attentionsize,
        )

        # Per-channel classifier for mcy (needed for accuracy-weighted dropout)
        if use_channel_mixer or use_prototypes:
            self.clf = nn.Linear(hidden, n_classes)

        # Channel dropout: ZeroEmbeddingDropout for mixer variants
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
                d_model=hidden,
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
                dim=hidden,
                codebook_size=n_prototypes,
                rotation_trick=True,
                channel_first=False,
            )

        # Sequence encoder (BiGRU)
        self.seqn2 = nn.GRU(
            hidden,
            seqnhidden2,
            num_layers=seqnlayer2,
            batch_first=True,
            bidirectional=True,
        )

        # Final classifier
        self.classifier = nn.Linear(2 * seqnhidden2, n_classes)

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
        x_flat = x.reshape(batch_size * L * in_chans, 1, T, F)
        if self.use_channel_mixer:
            zero_input = torch.zeros(1, 1, T, F, device=x_flat.device, dtype=x_flat.dtype)
            x_flat = torch.cat([x_flat, zero_input], dim=0)

        embs = self.epoch_encoder(x_flat)

        if self.use_channel_mixer:
            x = embs[:-1]          # (B*L*C, hidden)
            zero_emb = embs[-1:]   # (1, hidden)
        else:
            x = embs

        # Per-channel classification (mcy) on real embeddings (pre-dropout)
        if hasattr(self, "clf"):
            self.mcy = self.clf(x).reshape(batch_size, L, in_chans, -1)
        else:
            self.mcy = None

        # [2] Robustness: channel dropout + mixer
        x = x.reshape(batch_size * L, in_chans, -1)  # (B*L, C, hidden)

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

        x = x.mean(dim=1)  # (B*L, hidden) — pool channels

        # [3] SimVQ quantization (optional)
        if self.use_prototypes:
            x, indexes, commit_loss = self.prototype(x)
            self.commit_loss = commit_loss
            self.proto_y = self.clf(x).reshape(batch_size, L, -1)
        else:
            self.commit_loss = None
            self.proto_y = None

        # [4] Sequence encoding (BiGRU)
        x = x.reshape(batch_size, L, -1)
        x, _ = self.seqn2(x)  # (B, L, 2*seqnhidden2)

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


class ProtoSeqSleepNetTrainer(Trainer):
    """Trainer for ProtoSeqSleepNet with multi-loss (main + proto + commit + per-channel)."""

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
        loss = main_loss

        extra_metrics = {
            "loss_main": float(main_loss.detach().item()),
        }

        if proto_y is not None:
            proto_loss = loss_fn(proto_y.reshape(-1, proto_y.shape[-1]), targets)
            loss = loss + proto_loss
            extra_metrics["loss_proto"] = float(proto_loss.detach().item())

        if commit_loss is not None:
            loss = loss + commit_loss
            extra_metrics["loss_commit"] = float(
                commit_loss.detach().item()
                if isinstance(commit_loss, torch.Tensor)
                else commit_loss
            )

        # Per-channel losses and accuracy
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
    x = torch.randn(4, 21, 3, 29, 129)

    # --- Full ProtoSeqSleepNet ---
    print("=== Full ProtoSeqSleepNet ===")
    model = ProtoSeqSleepNet(n_classes=5, in_chan=3)
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
    print("=== ProtoSeqSleepNet (mixer, no prototypes) ===")
    model2 = ProtoSeqSleepNet(n_classes=5, in_chan=3, use_channel_mixer=True, use_prototypes=False)
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
    print("=== ProtoSeqSleepNet (random dropout) ===")
    model3 = ProtoSeqSleepNet(
        n_classes=5, in_chan=3,
        random_input_dropout=0.5, cdropout=0.0,
        use_channel_mixer=False, use_prototypes=False,
    )
    print(f"  params: {sum(p.numel() for p in model3.parameters()):,}")
    model3.train()
    y3 = model3(x)
    assert y3.shape == (4, 21, 5)
    y3.sum().backward()
    assert model3.get_metrics()["mcy"] is None
    print("  OK")

    # --- Ablation: baseline (no dropout, no mixer, no prototypes) ---
    print("=== ProtoSeqSleepNet (baseline) ===")
    model4 = ProtoSeqSleepNet(
        n_classes=5, in_chan=3,
        cdropout=0.0, use_channel_mixer=False, use_prototypes=False,
    )
    print(f"  params: {sum(p.numel() for p in model4.parameters()):,}")
    y4 = model4(x)
    assert y4.shape == (4, 21, 5)
    y4.sum().backward()
    print("  OK")

    print("\nAll variants passed.")
