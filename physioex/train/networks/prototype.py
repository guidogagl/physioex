import torch
import torch.nn as nn
import torch.nn.functional as F

from physioex.train.networks.base import SleepModule
from physioex.train.networks.utils.proto_layers import *

from vector_quantize_pytorch import SimVQ


module_config = dict()


class ProtoSleepModule(SleepModule):
    def __init__(self, model: nn.Module, module_config: dict = module_config):
        super(ProtoSleepModule, self).__init__(model, module_config)

        self.loss = nn.CrossEntropyLoss()

    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):

        (commit_loss, coverage, mcy) = embeddings

        coverage = coverage.reshape(-1)

        self.log(
            f"{log}/cov/mean",
            coverage.mean() * (1 + (29 // 2)),
            prog_bar=True,
            sync_dist=True,
        )
        self.log(f"{log}/cov/var", coverage.var(), sync_dist=True)

        batch_size, seq_len, n_class = outputs.size()

        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)

        loss = self.loss(outputs, targets)
        self.log(f"{log}/loss", loss, prog_bar=True, sync_dist=True)

        loss = loss + commit_loss - 0.1 * coverage.var()

        self.log(
            f"{log}/acc", self.wacc(outputs, targets), prog_bar=True, sync_dist=True
        )

        if log == "val":
            self.log(f"{log}_acc", self.wacc(outputs, targets), sync_dist=True)

        if mcy is not None:
            mcy = mcy.reshape(batch_size * seq_len, -1, n_class)

            c0_acc = self.wacc(mcy[:, 0], targets)
            c1_acc = self.wacc(mcy[:, 1], targets)
            c2_acc = self.wacc(mcy[:, 2], targets)

            self.nn.channels_proba = [c0_acc, c1_acc, c2_acc]

            self.log(f"{log}/c0_acc", c0_acc, sync_dist=True)
            self.log(f"{log}/c1_acc", c1_acc, sync_dist=True)
            self.log(f"{log}/c2_acc", c2_acc, sync_dist=True)

        self.log(f"{log}/commit_loss", commit_loss, sync_dist=True)

        if log_metrics:
            self.log(f"{log}/f1", self.wf1(outputs, targets), sync_dist=True)
            self.log(f"{log}/ck", self.ck(outputs, targets), sync_dist=True)
            self.log(f"{log}/pr", self.pr(outputs, targets), sync_dist=True)
            self.log(f"{log}/rc", self.rc(outputs, targets), sync_dist=True)
            self.log(f"{log}/macc", self.macc(outputs, targets), sync_dist=True)
            self.log(f"{log}/mf1", self.mf1(outputs, targets), sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        if "val/loss" not in self.trainer.logged_metrics:
            self.log("val/loss", float("inf"))

        # Logica di training
        inputs, targets, subjects, dataset_idx = batch
        embeddings, outputs = self.encode(inputs)

        return self.compute_loss(embeddings, outputs, targets)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets, subjects, dataset_idx = batch

        embeddings, outputs = voting_strategy(self, inputs, self.L)

        return self.compute_loss(embeddings, outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets, subjects, dataset_idx = batch

        embeddings, outputs = voting_strategy(self, inputs, self.L)

        return self.compute_loss(embeddings, outputs, targets, "test", log_metrics=True)


class ProtoSleepNet(nn.Module):
    def __init__(self, module_config=module_config):
        super(ProtoSleepNet, self).__init__()

        self.time_masking = TimeMasking(
            hidden_size=128,  # hidden size of the epoch encoder
            L=29,  # length of the time masking window
            temperature=0.1,  # temperature for the softmax
        )

        try:
            self.weights = module_config[
                "weights"
            ]  # default equal weights for channels and mixed channels
        except:
            print(module_config)
            exit(0)

        print("Channel mixing weights :", self.weights)

        self.channel_mixer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, batch_first=True
        )

        self.channel_mixer = nn.Sequential(self.channel_mixer, nn.LayerNorm([3, 128]))

        self.prototype = SimVQ(
            dim=128,
            codebook_size=50,
            rotation_trick=True,  # use rotation trick from Fifty et al.
            channel_first=False,
        )

        self.channels_dropout = ChannelsDropout(dropout_prob=0.5)
        self.channels_proba = [0.7, 0.7, 0.7]

        self.channels_sampler = ChannelSampler(
            hidden_size=128,
            attention_size=32,
        )

        self.clf = nn.Linear(128, 5)

    def epoch_encoder(self, x):
        pass

    def sequence_encoder(self, x):
        pass

    def proto_encoder(self, x):
        batch, L, nchan, T, F = x.size()

        #### epoch encoding ####
        x = x.reshape(batch * L, nchan, T, F)
        x = self.epoch_encoder(x)

        x = [self.time_masking(x[:, i]) for i in range(nchan)]

        mask = [mask_.reshape(batch * L, 1, T) for _, mask_ in x]
        x = [x_.reshape(batch * L, 1, 128) for x_, _ in x]
        x, mask = torch.cat(x, dim=1), torch.cat(mask, dim=1)

        # x shape : (batch_size * seq_len, n_chan, 128)
        # mask shape : (batch_size * seq_len, n_chan, T)
        coverage = torch.sum(mask.reshape(-1, T), dim=-1) / ((T // 2) + 1)
        coverage = coverage.reshape(batch, L, nchan)

        #### channel mixing ####
        x = (self.weights[0] * x) + (
            self.weights[1] * self.channel_mixer(x)
        )  # (batch * L, n_chan, 128)

        return x

    def prototyping(self, x):
        batch, L, nchan, T, F = x.size()

        x = self.proto_encoder(x)

        # prototyping
        x = x.reshape(batch * L * nchan, 128)
        p, indexes, commit_loss = self.prototype(x)
        p = p.reshape(batch, L, nchan, 128)
        indexes = indexes.reshape(batch, L, nchan)

        # sequence encoding
        x = self.sequence_encoder(p.clone())

        # channels dropout
        x = x.reshape(batch * L, nchan, 128)
        x, alphas = self.channels_sampler(x, r_alphas=True)

        p = p.reshape(batch * L, nchan, 128)
        p = torch.einsum("bs, bsh -> bh", alphas, p)

        alphas = alphas.reshape(batch, L, nchan)
        return p, alphas

    def encode(self, x):
        # x shape : (batch_size, seq_len, n_chan, n_samp)
        batch, L, nchan, T, F = x.size()

        #### epoch encoding ####
        x = x.reshape(batch * L, nchan, T, F)
        x = self.epoch_encoder(x)

        x = [self.time_masking(x[:, i]) for i in range(nchan)]

        mask = [mask_.reshape(batch * L, 1, T) for _, mask_ in x]
        x = [x_.reshape(batch * L, 1, 128) for x_, _ in x]
        x, mask = torch.cat(x, dim=1), torch.cat(mask, dim=1)

        # x shape : (batch_size * seq_len, n_chan, 128)
        # mask shape : (batch_size * seq_len, n_chan, T)
        coverage = torch.sum(mask.reshape(-1, T), dim=-1) / ((T // 2) + 1)
        coverage = coverage.reshape(batch, L, nchan)

        #### channel mixing ####
        x = (self.weights[0] * x) + (
            self.weights[1] * self.channel_mixer(x)
        )  # (batch * L, n_chan, 128)

        # prototyping
        x = x.reshape(batch * L * nchan, 128)
        x, indexes, commit_loss = self.prototype(x)
        x = x.reshape(batch, L, nchan, 128)
        indexes = indexes.reshape(batch, L, nchan)

        # sequence encoding
        x = self.sequence_encoder(x)

        # multi channel classification
        if self.training:
            x = x.reshape(-1, nchan, 128)
            mcy = [
                self.clf(x[:, 0]).reshape(-1, 1, 5),
                self.clf(x[:, 1]).reshape(-1, 1, 5),
                self.clf(x[:, 2]).reshape(-1, 1, 5),
            ]
            mcy = torch.cat(mcy, dim=1).reshape(batch, L, nchan, 5)
            x = x.reshape(batch, L, nchan, 128)
        else:
            mcy = None
        # channels dropout
        x = x.reshape(batch * L, nchan, 128)
        x = self.channels_dropout(x, self.channels_proba)  # apply dropout to channels
        x = self.channels_sampler(x)

        # classification
        x = self.clf(x).reshape(batch, L, -1)

        return (commit_loss, coverage, mcy), x

    def forward(self, x):
        x, y = self.encode(x)

        return y

    def project(self, x):
        # x shape : (batch_size, seq_len, n_chan, n_samp)
        batch, L, nchan, T, F = x.size()

        #### epoch encoding ####
        x = x.reshape(batch * L, nchan, T, F)
        x = self.epoch_encoder(x)

        x = [self.time_masking(x[:, i]) for i in range(nchan)]

        mask = [mask_.reshape(batch * L, 1, T) for _, mask_ in x]
        x = [x_.reshape(batch * L, 1, 128) for x_, _ in x]
        x, mask = torch.cat(x, dim=1), torch.cat(mask, dim=1)

        # x shape : (batch_size * seq_len, n_chan, 128)
        # mask shape : (batch_size * seq_len, n_chan, T)
        coverage = torch.sum(mask.reshape(-1, T), dim=-1) / ((T // 2) + 1)
        coverage = coverage.reshape(batch, L, nchan)

        #### channel mixing ####
        x = (self.weights[0] * x) + (
            self.weights[1] * self.channel_mixer(x)
        )  # (batch * L, n_chan, 128)

        # prototyping
        x = x.reshape(batch * L * nchan, 128)
        x, indexes, commit_loss = self.prototype(x)
        x = x.reshape(batch, L, nchan, 128)
        indexes = indexes.reshape(batch, L, nchan)

        # sequence encoding
        x = self.sequence_encoder(x)

        # channels dropout
        x = x.reshape(batch * L, nchan, 128)
        x = self.channels_dropout(x, self.channels_proba)  # apply dropout to channels
        x = self.channels_sampler(x)

        return x.reshape(batch, L, -1)


def voting_strategy(model: torch.nn.Module, inputs: torch.Tensor, L: int):
    batch_size, night_length, n_channels, _, _ = inputs.size()

    outputs = torch.zeros(
        batch_size, night_length, 5, device=inputs.device, dtype=inputs.dtype
    )
    coverage = torch.zeros(
        batch_size, night_length, n_channels, device=inputs.device, dtype=inputs.dtype
    )

    commit_loss = 0

    # input shape is ( bach_size, night_length, n_channels, ... )
    # segment the input in self.L segments with a sliding window of stride 1 and size self.L
    for i in range(0, inputs.size(1) - L + 1, 1):
        input_segment = inputs[:, i : i + L]
        (seg_cl, seg_coverage, seg_mcy), seg_outputs = model.encode(input_segment)

        outputs[:, i : i + L] += torch.nn.functional.softmax(seg_outputs, dim=-1)
        coverage[:, i : i + L] += seg_coverage

        commit_loss += seg_cl

    # normalize the coverage values
    for i in range(L):
        coverage[:, i] = coverage[:, i] / (i + 1)
        coverage[:, -i - 1] = coverage[:, -i - 1] / (i + 1)

    coverage[:, L:-L] = coverage[:, L:-L] / L

    commit_loss = commit_loss / (inputs.size(1) - L + 1)

    return (commit_loss, coverage, None), outputs
