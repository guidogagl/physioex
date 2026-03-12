import torch
import torch.nn as nn
from vector_quantize_pytorch import SimVQ

from physioex.train.networks.base import SleepModule
from physioex.train.networks.utils.proto_layers import ChannelsDropout, TimeMasking

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
        # (commit_loss, coverage, dist, y_, mcy) = embeddings
        batch_size, seq_len, n_class = outputs.size()

        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)

        loss = self.loss(outputs, targets)
        self.log(f"{log}/loss", loss, prog_bar=True, sync_dist=True)

        acc = self.wacc(outputs, targets)
        self.log(f"{log}/acc", acc, prog_bar=True, sync_dist=True)

        commit_loss = self.nn.commit_loss
        mcy = self.nn.mcy.to(outputs.device)
        proto_y = self.nn.proto_y.to(outputs.device)

        self.log(f"{log}/commit_loss", commit_loss, sync_dist=True)

        proto_y = proto_y.reshape(batch_size * seq_len, n_class)
        mcy = mcy.reshape(batch_size * seq_len, -1, n_class)

        proto_loss = self.loss(proto_y, targets)

        loss = loss + proto_loss + commit_loss

        proto_acc = self.wacc(proto_y, targets)
        
        channel_acc = [self.wacc(mcy[:, i], targets) for i in range(self.nn.in_channels)]

        self.nn.channels_proba = channel_acc
        
        for i, c_acc in enumerate(channel_acc):
            self.log(f"{log}/c{i}_acc", c_acc, sync_dist=True)

        mc_loss = sum(self.loss(mcy[:, i], targets) for i in range(self.nn.in_channels))
        loss = loss + mc_loss

        self.log(f"{log}/p_acc", proto_acc, sync_dist=True)

        if log == "val":
            self.log(f"{log}_acc", self.wacc(outputs, targets), sync_dist=True)

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

        embeddings, outputs = self.nn.eval_on_night(inputs, self.L)

        return self.compute_loss(embeddings, outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets, subjects, dataset_idx = batch

        embeddings, outputs = self.nn.eval_on_night(inputs, self.L)

        return self.compute_loss(embeddings, outputs, targets, "test", log_metrics=True)


class ProtoSleepNet(nn.Module):
    def __init__(self, module_config=module_config):
        super(ProtoSleepNet, self).__init__()

        self.in_channels = module_config["in_channels"]

        self.time_masking = TimeMasking(
            hidden_size=128,  # hidden size of the epoch encoder
            L=29,  # length of the time masking window
            temperature=0.1,  # temperature for the softmax
        )

        t_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, batch_first=True
        )

        self.channel_mixer = nn.TransformerEncoder(t_layer, num_layers=4)

        self.channel_mixer = _initialize_residual_transformer(self.channel_mixer)

        self.n_prototypes = module_config["n_prototypes"]
        
        self.prototype = SimVQ(
            dim=128,
            codebook_size=self.n_prototypes,
            rotation_trick=True,  # use rotation trick from Fifty et al.
            channel_first=False,
        )

        self.channels_dropout = ChannelsDropout(dropout_prob=0.5)
        self.channels_proba = [0.7 for _ in range(self.in_channels)]

        self.clf = nn.Linear(128, 5)

        self.mcy = None
        self.commit_loss = None
        self.proto_y = None

    def epoch_encoder(self, x):
        pass

    def sequence_encoder(self, x):
        pass

    def eval_on_night(self, inputs: torch.Tensor, L: int = 21):
        batch_size, night_length, n_channels, T, F = inputs.size()

        x, mcy = self.f_ExE(inputs, return_mcy=True)

        # prototyping
        p, commit_loss = self.f_P(x, return_commit_loss=True)
        proto_y = self.clf(p.reshape(batch_size * night_length, -1)).reshape(
            batch_size, night_length, -1
        )

        y = torch.zeros(batch_size, night_length, 5, device=p.device, dtype=p.dtype)
        counts = torch.zeros(
            batch_size, night_length, device=p.device, dtype=torch.float32
        )

        for i in range(0, L):
            p_i = p[:, i:]
            night_length_i = p_i.size(1) - (p_i.size(1) % L)
            p_i = p_i[:, :night_length_i, :].reshape(-1, L, p_i.size(-1))
            p_i = self.f_ExS(p_i).reshape(batch_size * night_length_i, -1)

            y[:, i : i + night_length_i] += self.clf(p_i).reshape(
                batch_size, night_length_i, -1
            )
            counts[:, i : i + night_length_i] += 1

        y = y.reshape(batch_size, night_length, -1)  # (batch_size, night_length, 5)
        y = y / counts.unsqueeze(-1)  # average over the segments

        self.mcy = mcy
        self.commit_loss = commit_loss
        self.proto_y = proto_y

        return p, y

    def f_ExE(self, x, return_mask=False, return_mcy=False):
        # epoch-encoding function for x
        batch, L, nchan, T, F = x.size()

        #### epoch encoding ####
        x = x.reshape(batch * L, nchan, T, F)
        x = self.epoch_encoder(x)  # shape : (batch * L, nchan, T, 128)
        x = x.reshape(batch * L * nchan, T, -1)  # (nchan * batch * L, T, 128)

        x, mask = self.time_masking(x)
        x = x.reshape(batch * L, nchan, -1)  # (batch * L, nchan, 128)
        mask = mask.reshape(batch * L, nchan, T)  # (batch * L, nchan, T)

        mcy = self.clf(x.reshape(batch * L * nchan, -1)).reshape(batch, L, nchan, -1)

        ### dropout channels ###
        x = self.channels_dropout(x, self.channels_proba)  # apply dropout to channels

        #### channel mixing ####
        x = x + self.channel_mixer(x)

        x = x.reshape(batch, L, nchan, -1)
        mask = mask.reshape(batch, L, nchan, T)

        # average across channels
        x = x.mean(dim=2)  # (batch, L, 128)

        if return_mask and return_mcy:
            return x, mask, mcy

        if return_mask:
            return x, mask

        if return_mcy:
            return x, mcy

        return x

    def f_P(self, x, return_indexes=False, return_commit_loss=False):
        # prototyping utility function for x
        # x shape : ( batch_size, seq_len, nchan, hidden_size )
        batch, L, hidden = x.size()

        x = x.reshape(batch * L, -1)
        x, indexes, commit_loss = self.prototype(x)

        x = x.reshape(batch, L, -1)
        indexes = indexes.reshape(batch, L)

        if return_indexes and return_commit_loss:
            return x, indexes, commit_loss

        if return_indexes:
            return x, indexes

        if return_commit_loss:
            return x, commit_loss

        return x

    def f_ExS(self, x):
        # sequence encoding function for x
        # x shape : ( batch_size, seq_len, hidden_size )
        batch, L, hidden = x.size()

        x = x + self.sequence_encoder(x)

        return x.reshape(batch, L, hidden)

    def encode(self, x):
        # x shape : (batch_size, seq_len, n_chan, n_samp)
        batch, L, nchan, T, F = x.size()

        # epoch encoding
        x, mcy = self.f_ExE(x, return_mcy=True)

        # prototyping
        p, commit_loss = self.f_P(x, return_commit_loss=True)

        # proto-classification
        proto_y = self.clf(p.reshape(batch * L, -1)).reshape(batch, L, -1)

        # sequence encoding
        x = self.f_ExS(p)

        # classification
        y = self.clf(x).reshape(batch, L, -1)

        # save the elements for the loss computation
        self.mcy = mcy
        self.commit_loss = commit_loss
        self.proto_y = proto_y

        return p, y

    def forward(self, x):
        x, y = self.encode(x)

        return y


def voting_strategy(model: torch.nn.Module, inputs: torch.Tensor, L: int):
    batch_size, night_length, n_channels, _, _ = inputs.size()

    outputs = torch.zeros(
        batch_size, night_length, 5, device=inputs.device, dtype=inputs.dtype
    )

    for i in range(0, inputs.size(1) - L + 1, 1):
        input_segment = inputs[:, i : i + L]
        _, seg_outputs = model.encode(input_segment)

        outputs[:, i : i + L] += torch.nn.functional.softmax(seg_outputs, dim=-1)

    return None, outputs


def _initialize_residual_transformer(encoder: nn.TransformerEncoder):
    """
    Initialize the TransformerEncoder to act as identity at training start.
    For residual connection: output = input + encoder(input)
    We want encoder(input) ≈ 0 initially, so output ≈ input
    """
    for layer in encoder.layers:
        # Initialize feedforward layers to zero
        torch.nn.init.constant_(layer.linear1.weight, 0.0)
        torch.nn.init.constant_(layer.linear1.bias, 0.0)
        torch.nn.init.constant_(layer.linear2.weight, 0.0)
        torch.nn.init.constant_(layer.linear2.bias, 0.0)

        # Initialize attention output projection to zero
        torch.nn.init.constant_(layer.self_attn.out_proj.weight, 0.0)
        torch.nn.init.constant_(layer.self_attn.out_proj.bias, 0.0)

        # Initialize Q, K, V projections with small values for stability
        torch.nn.init.normal_(layer.self_attn.in_proj_weight, mean=0.0, std=0.01)
        if layer.self_attn.in_proj_bias is not None:
            torch.nn.init.constant_(layer.self_attn.in_proj_bias, 0.0)

    return encoder
