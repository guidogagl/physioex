import torch
from torch import nn
from torchinfo import summary

from vector_quantize_pytorch import SimVQ

from physioex.models.seqsleepnet import LearnableFilterbank
from physioex.train.metrics import accuracy_score

from physioex.train.trainer import Trainer
from physioex.data.dataset import PhysioExDataset


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

        # channel_acc: (nchan,) - accuratezza per canale (valori tra 0 e 1)
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
        proba = proba / proba_sum  # normalizzazione lineare

        # Applica lo shuffle solo a una frazione dei batch (dropout_prob)
        mask = torch.rand(batch, device=device) < self.dropout_prob

        # Per ogni elemento nel batch, campiona nchan indici secondo proba
        idx = torch.multinomial(
            proba.expand(batch, -1), nchan, replacement=True
        )  # (batch, nchan)
        batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, nchan)

        x_shuffled = x.clone()
        x_shuffled[mask] = x[batch_idx[mask], idx[mask], :]

        return x_shuffled


class TimeMasking(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        L: int = 29,
        temperature: float = 0.1,
    ):
        super(TimeMasking, self).__init__()
        self.temperature = temperature
        self.L = L

        windows = []
        for n in range(1, (L // 2) + 2, 2):  # n = window length
            for start in range(L - n + 1):
                w = torch.zeros(L)
                w[start : start + n] = 1.0
                windows.append(w)
        windows = torch.stack(windows, dim=0)  # (num_windows, L)
        self.register_buffer("windows", windows)
        self.num_windows = windows.size(0)

        self.W = nn.Linear(hidden_size, self.num_windows, bias=False)

        W_omega = torch.zeros((self.num_windows, self.num_windows), dtype=torch.float32)
        b_omega = torch.zeros((self.num_windows), dtype=torch.float32)
        u_omega = torch.zeros((self.num_windows), dtype=torch.float32)

        self.W_omega = nn.Parameter(W_omega)
        self.b_omega = nn.Parameter(b_omega)
        self.u_omega = nn.Parameter(u_omega)

        nn.init.normal_(self.W_omega, std=0.1)
        nn.init.normal_(self.b_omega, std=0.1)
        nn.init.normal_(self.u_omega, std=0.1)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.size()
        if sequence_length != self.L:
            raise ValueError(
                f"sequence_length ({sequence_length}) must match L ({self.L})"
            )

        w = self.W(x)  # batch, seq, num_windows

        v = torch.tanh(
            torch.matmul(
                torch.reshape(w, [batch_size * sequence_length, self.num_windows]),
                self.W_omega,
            )
            + torch.reshape(self.b_omega, [1, -1])
        )  #

        vu = torch.matmul(v, torch.reshape(self.u_omega, [-1, 1]))
        attn_logits = torch.reshape(vu, [-1, sequence_length])
        alphas = torch.softmax(attn_logits, dim=1).reshape(batch_size, sequence_length)

        w = torch.einsum("bs, bsl -> bl", alphas, w)  # batch, num_windows
        alphas = torch.nn.functional.gumbel_softmax(w, tau=self.temperature, hard=True)

        mask = torch.einsum("bs,sl -> bl", alphas, self.windows)

        # input masking
        x = torch.einsum("bs, bsl -> bl", mask, x) / torch.sum(
            mask, dim=-1, keepdim=True
        )

        return x, mask


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


class ProtoSleepNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 5,
        in_chan: int = 3,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        lowfreq: int = 0,
        highfreq: int = 50,
        fs: int = 100,
        seqnhidden1: int = 64,
        seqnlayer1: int = 4,
        n_prototypes: int = 15,
        tmL: int = 29,
        tmtemp: float = 0.1,
        cdropout: float = 0.5,
        cmhidden: int = 256,
        cmnheads: int = 4,
        cmnlayers: int = 1,
        seqnhidden2: int = 64,
        seqnlayer2: int = 4,
    ):
        super(ProtoSleepNet, self).__init__()

        self.filterbank = LearnableFilterbank(
            1, F, D * in_chan, nfft, fs, lowfreq, highfreq
        )
        self.seqn1 = nn.LSTM(
            D * in_chan,
            seqnhidden1,
            num_layers=seqnlayer1,
            batch_first=True,
            bidirectional=True,
        )

        tmhidden = (
            2 * seqnhidden1
        )  # hidden size of the epoch encoder, input to time masking

        self.time_masking = TimeMasking(
            hidden_size=tmhidden,  # hidden size of the epoch encoder
            L=tmL,  # length of the time masking window
            temperature=tmtemp,  # temperature for the softmax
        )

        t_layer = nn.TransformerEncoderLayer(
            d_model=tmhidden, nhead=cmnheads, dim_feedforward=cmhidden, batch_first=True
        )

        self.channel_mixer = nn.TransformerEncoder(t_layer, num_layers=cmnlayers)

        self.channel_mixer = _initialize_residual_transformer(self.channel_mixer)

        self.channels_dropout = ChannelsDropout(dropout_prob=cdropout)
        self.register_buffer(
            "channels_acc", torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32)
        )
        self.channels_acc_list = [
            torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32)
        ]  # lista per memorizzare le accuratezze per canale

        self.prototype = SimVQ(
            dim=tmhidden,
            codebook_size=n_prototypes,
            rotation_trick=True,  # use rotation trick from Fifty et al.
            channel_first=False,
        )

        self.seqn2 = nn.GRU(
            2 * seqnhidden1,
            seqnhidden2,
            num_layers=seqnlayer2,
            batch_first=True,
            bidirectional=True,
        )

        self.clf = nn.Linear(tmhidden, n_classes)

        self.mcy = None
        self.commit_loss = None
        self.proto_y = None
        self.mask = None

    def forward(self, x):
        batch_size, L, in_chans, T, F = x.size()

        x = x.reshape(batch_size * L * in_chans, 1, T, F)

        x = self.filterbank(x)

        x = x.permute(0, 2, 1, 3)  # shape ( batch_size*L*in_chan, T, 1, D )

        x = x.reshape(batch_size * L * in_chans, T, -1)

        x, _ = self.seqn1(x)

        x, self.mask = self.time_masking(x)

        self.mcy = self.clf(x.reshape(batch_size * L * in_chans, -1)).reshape(
            batch_size, L, in_chans, -1
        )

        x = x.reshape(batch_size * L, in_chans, -1)

        if self.training:
            # check if the channel accuracy list is not empty
            if len(self.channels_acc_list) > 0:
                # compute the mean accuracy for each channel across all batches seen so far
                mean_acc = torch.stack(self.channels_acc_list, dim=0).mean(dim=0)
                self.channels_acc.copy_(mean_acc)

                # reset the list to avoid using stale accuracy values
                self.channels_acc_list.clear()

            x = self.channels_dropout(x, self.channels_acc)

        x = x + self.channel_mixer(x)

        x = x.mean(dim=1)

        x = x.reshape(batch_size * L, -1)

        x, indexes, commit_loss = self.prototype(x)

        self.proto_y = self.clf(x.reshape(batch_size * L, -1)).reshape(
            batch_size, L, -1
        )

        self.commit_loss = commit_loss

        x = x.reshape(batch_size, L, -1)

        x, _ = self.seqn2(x)
        x = x.reshape(batch_size * L, -1)
        x = self.clf(x)
        x = x.reshape(batch_size, L, -1)
        return x

    def get_metrics(self):
        return {
            "mcy": self.mcy,
            "commit_loss": self.commit_loss,
            "proto_y": self.proto_y,
            "mask": self.mask,
        }

    def update_channel_acc(self, acc_list):
        acc_tensor = torch.tensor(
            acc_list, device=self.channels_acc.device, dtype=torch.float32
        )
        self.channels_acc_list.append(acc_tensor)


class ProtoSleepNetTrainer(Trainer):
    @staticmethod
    def _step(
        model: torch.nn.Module,
        batch: dict,
        loss_fn: torch.nn.Module,
        device: torch.device,
    ) -> tuple[float, float, dict]:

        inputs, targets = batch

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.autocast(device.type if "cuda" in device.type else "cpu"):
            outputs = model(inputs.to(device))

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

        # mcy is multiplied for in_chans
        mcy = mcy.reshape(outputs.shape[0], -1, outputs.shape[-1])
        in_chans = mcy.shape[1]

        # compute per channel accuracy:
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

        # check if the model is in training mode
        if not model.training:
            model.update_channel_acc(chan_acc)

        # compute accuracy (dummy here)
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


if __name__ == "__main__":

    dataset = PhysioExDataset(
        datasets=[
            "hmc",
            "sleepedf",
            "mass",
            "dcsm",
            "shhs",
            "mros",
            "mesa",
            "wsc",
            "PD/HOA",
            "AD/HOA",
            "PD/PD",
            "AD/AD",
        ]
    )

    model = ProtoSleepNet(
        in_chan=dataset.get_num_channels(),
        cdropout=1.0,
        cmnlayers=4,
    )

    print("ProtoSleepNet summary:")

    summary(model, (32, 21, dataset.get_num_channels(), 29, 129))

    print("\nTraining ProtoSleepNet...")

    model = ProtoSleepNetTrainer.train(
        model=model,
        dataset=dataset,
        max_epochs=50,
        lr=1e-3,
    )

    print("\nEvaluating ProtoSleepNet...")

    results = ProtoSleepNetTrainer.evaluate(
        model=model,
        dataset=dataset,
    )

    print(results)
