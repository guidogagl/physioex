from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from physioex.train.networks.seqsleepnet import SeqSleepNet
from physioex.train.networks.tinysleepnet import TinySleepNet
from physioex.train.networks.base import SleepModule

import matplotlib.pyplot as plt
import seaborn as sns


module_config = dict()


class SeqSexNet(SleepModule):
    def __init__(self, module_config=module_config):

        super(SeqSexNet, self).__init__(Net(module_config=module_config), module_config)


from physioex.train.networks.seqsleepnet import (
    EpochEncoder,
    SequenceEncoder,
    AttentionLayer,
)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.nn = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 16, (3, 3), stride=(2, 2))),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(16, 32, (3, 3), stride=(2, 2))),
                    ("relu2", nn.ReLU()),
                    ("conv3", nn.Conv2d(32, 64, (3, 3), stride=(2, 2))),
                    ("relu3", nn.ReLU()),
                    ("conv4", nn.Conv2d(64, 128, (2, 2), stride=(2, 2))),
                    ("relu4", nn.ReLU()),
                    ("conv5", nn.Conv2d(128, 256, (1, 3), stride=(1, 3))),
                    ("relu5", nn.ReLU()),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

    def forward(self, x):
        x = self.nn(x)
        x = x.reshape(x.size(0), 128, -1)
        return x.sum(dim=2)


class NightEncoder(nn.Module):
    def __init__(self, module_config):
        super(NightEncoder, self).__init__()

        self.LSTM = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = AttentionLayer(
            hidden_size=64 * 2,
            attention_size=64,
        )

        self.clf = nn.Linear(64 * 2, module_config["n_classes"])

        nn.init.constant_(self.clf.bias, 40)

        self.clf.bias.requires_grad = False

    def encode(self, x):
        x, _ = self.LSTM(x)
        x = self.attention(x)
        return x, self.clf(x)

    def forward(self, x):
        _, y = self.encode(x)
        return y


class WholeNightNet(nn.Module):
    def __init__(self, module_config=module_config):

        super().__init__()

        module_config.update(
            {
                "T": 29,
                "F": 129,
                "D": 32,
                "nfft": 256,
                "lowfreq": 0,
                "highfreq": 50,
                "seqnhidden1": 64,
                "seqnlayer1": 1,
                "attentionsize1": 32,
                "seqnhidden2": 64,
                "seqnlayer2": 1,
            }
        )
        self.epoch_encoder = EpochEncoder(module_config)
        self.conv_encoder = ConvNet()

        self.lin1 = nn.Linear(128, module_config["n_classes"])
        nn.init.constant_(self.lin1.bias, 40)
        self.lin1.bias.requires_grad = False

        self.night_encoder = NightEncoder(module_config)

    def encode(self, x):
        batch, seqlen, nchan, T, F = x.size()

        x = x.reshape(batch * seqlen, nchan, T, F)
        epoch_enc = self.epoch_encoder(x)
        conv_enc = self.conv_encoder(x)

        epoch_enc = epoch_enc.reshape(batch, seqlen, -1)
        conv_enc = conv_enc.reshape(batch, seqlen, -1)

        x = epoch_enc + conv_enc

        y1 = self.lin1(x).mean(dim=1).view(-1, 1)

        x, y2 = self.night_encoder.encode(x)

        y = (y1 + y2) / 2

        return x, y

    def forward(self, x):
        _, y = self.encode(x)
        return y


class WholeNightAgeNet(SleepModule):
    def __init__(self, module_config=module_config):

        super(WholeNightAgeNet, self).__init__(
            WholeNightNet(module_config=module_config), module_config
        )

        self.validation_step_outputs = []
        self.validation_step_targets = []

        self.training_step_outputs = []
        self.training_step_targets = []

        self.val_step = 0
        self.train_step = 0

    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        return self.opt

    def log_correlation(self, outputs, targets, log: str = "train"):

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=targets.view(-1).cpu().numpy(), y=outputs.view(-1).cpu().numpy()
        )
        plt.xlabel("True Age")
        plt.ylabel("Estimated Age")
        plt.title("Correlation between Estimated and True Age")

        log_id = self.train_step if log == "train" else self.val_step

        # convert the figure to a tensor
        self.logger.experiment.add_figure(f"{log}-age-corr", plt.gcf(), log_id)
        plt.close()

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs)
        loss = self.compute_loss(embeddings, outputs, targets, "val")

        self.validation_step_outputs.append(outputs)
        self.validation_step_targets.append(targets)

        return loss

    def training_step(self, batch, batch_idx):
        # get the logged metrics
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))

        # Logica di training
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs)

        loss = self.compute_loss(embeddings, outputs, targets, "train")

        # outputs = torch.mean( outputs, dim = 1)

        # clone the tensor and detach it from the graph
        outputs = outputs.clone().detach()
        targets = targets.clone().detach()

        self.training_step_outputs.append(outputs)
        self.training_step_targets.append(targets)

        if self.train_step % 40 == 0:
            all_preds = torch.cat(self.training_step_outputs).view(-1)
            all_targets = torch.cat(self.training_step_targets).view(-1)

            self.training_step_outputs.clear()
            self.training_step_targets.clear()

            self.log_correlation(all_preds, all_targets, "train")

        self.train_step += 1

        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs).view(-1)
        all_targets = torch.cat(self.validation_step_targets).view(-1)

        self.log_correlation(all_preds, all_targets, "val")

        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()
        self.val_step += 1

    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):

        # outputs : batch_size, 1
        # targets : batch_size, seqlen, 1

        # take the mean age estimate from the sequence
        # outputs = torch.mean( outputs, dim = 1)  # normalize it to 100

        self.log(f"{log}_targ", targets.mean().item(), prog_bar=True)
        self.log(f"{log}_pred", outputs.mean().item(), prog_bar=True)

        outputs = outputs.reshape(-1, 1, 1)  # add a dimension for the sequence
        targets = targets.unsqueeze(1)  # add a dimension for the sequence

        return super().compute_loss(embeddings, outputs, targets, log, log_metrics)


from physioex.train.networks.age.model import M_PSG2FEAT, Config


class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()

        config = Config()
        config.n_channels = module_config["in_channels"]
        self.module = M_PSG2FEAT(config)

    def forward(self, x):
        x, y = self.encode(x)
        return y

    def encode(self, x: torch.Tensor):

        batch_size, seqlen, nchan, nsamp = x.size()

        x = x.permute(0, 2, 1, 3).reshape(batch_size, nchan, seqlen * nsamp)

        # we need to add padding to the input to make it equal to 5*60*128
        pad = 5 * 60 * 128 - x.size(2)
        x = nn.functional.pad(x, (0, pad))

        x = self.module(x)

        return x["feat"], x["pred"]


class SeqAgeNet(SleepModule):
    def __init__(self, module_config=module_config):

        super(SeqAgeNet, self).__init__(Net(module_config), module_config)

        self.validation_step_outputs = []
        self.validation_step_targets = []

        self.training_step_outputs = []
        self.training_step_targets = []

        self.val_step = 0
        self.train_step = 0

    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        return self.opt

    def log_correlation(self, outputs, targets, log: str = "train"):

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=targets.view(-1).cpu().numpy(), y=outputs.view(-1).cpu().numpy()
        )
        plt.xlabel("True Age")
        plt.ylabel("Estimated Age")
        plt.title("Correlation between Estimated and True Age")

        log_id = self.train_step if log == "train" else self.val_step

        # convert the figure to a tensor
        self.logger.experiment.add_figure(f"{log}-age-corr", plt.gcf(), log_id)
        plt.close()

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs)
        loss = self.compute_loss(embeddings, outputs, targets, "val")

        self.validation_step_outputs.append(outputs)
        self.validation_step_targets.append(targets)

        return loss

    def training_step(self, batch, batch_idx):
        # get the logged metrics
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))

        # Logica di training
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs)

        loss = self.compute_loss(embeddings, outputs, targets, "train")

        # outputs = torch.mean( outputs, dim = 1)

        # clone the tensor and detach it from the graph
        outputs = outputs.clone().detach()
        targets = targets.clone().detach()

        self.training_step_outputs.append(outputs)
        self.training_step_targets.append(targets)

        if self.train_step % 250 == 0:
            all_preds = torch.cat(self.training_step_outputs).view(-1)
            all_targets = torch.cat(self.training_step_targets).view(-1)

            self.training_step_outputs.clear()
            self.training_step_targets.clear()

            self.log_correlation(all_preds, all_targets, "train")

        self.train_step += 1

        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs).view(-1)
        all_targets = torch.cat(self.validation_step_targets).view(-1)

        self.log_correlation(all_preds, all_targets, "val")

        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()
        self.val_step += 1

    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):

        # outputs : batch_size, 1
        # targets : batch_size, seqlen, 1

        # take the mean age estimate from the sequence
        # outputs = torch.mean( outputs, dim = 1)  # normalize it to 100

        self.log(f"{log}_targ", targets.mean().item(), prog_bar=True)
        self.log(f"{log}_pred", outputs.mean().item(), prog_bar=True)

        outputs = outputs.reshape(-1, 1, 1)  # add a dimension for the sequence
        targets = targets.unsqueeze(1)  # add a dimension for the sequence

        return super().compute_loss(embeddings, outputs, targets, log, log_metrics)
