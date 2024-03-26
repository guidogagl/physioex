from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, latent_dim: int):
        super(SequenceEncoder, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.clf = nn.Linear(input_dim, n_classes)


class SeqtoSeqModule(nn.Module):
    def __init__(self, epoch_encoder, sequence_encoder):
        super(SeqtoSeqModule, self).__init__()
        self.epoch_encoder = epoch_encoder
        self.sequence_encoder = sequence_encoder

    def forward(self, x):
        batch_size, sequence_lenght = x.size(0), x.size(1)

        x_shape = [batch_size * sequence_lenght] + list(x.size()[2:])

        x = x.reshape(x_shape)
        x = self.epoch_encoder(x)

        x = x.reshape(batch_size, sequence_lenght, -1)

        return self.sequence_encoder(x)

    def encode(self, x):
        batch_size, sequence_lenght = x.size(0), x.size(1)
        x_shape = [batch_size * sequence_lenght] + list(x.size()[2:])

        x = x.reshape(x_shape)
        x = self.epoch_encoder(x)

        x = x.reshape(batch_size, sequence_lenght, -1)

        return self.sequence_encoder.encode(x), self.sequence_encoder(x)


class SeqtoSeq(pl.LightningModule):
    def __init__(
        self, epoch_encoder: nn.Module, sequence_encoder: nn.Module, config: Dict
    ):
        super(SeqtoSeq, self).__init__()
        self.nn = SeqtoSeqModule(epoch_encoder, sequence_encoder)

        # metrics
        self.acc = tm.Accuracy(
            task="multiclass", num_classes=config["n_classes"], average="weighted"
        )
        self.f1 = tm.F1Score(
            task="multiclass", num_classes=config["n_classes"], average="weighted"
        )
        self.ck = tm.CohenKappa(task="multiclass", num_classes=config["n_classes"])
        self.pr = tm.Precision(
            task="multiclass", num_classes=config["n_classes"], average="weighted"
        )
        self.rc = tm.Recall(
            task="multiclass", num_classes=config["n_classes"], average="weighted"
        )

        # loss
        self.loss = config["loss_call"](config["loss_params"])

        self.module_config = config

    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=float(self.module_config["learning_rate"]),
            betas=(
                float(self.module_config["adam_beta_1"]),
                float(self.module_config["adam_beta_2"]),
            ),
            eps=float(self.module_config["adam_epsilon"]),
        )
        return self.opt

    def forward(self, x):
        return self.nn(x)

    def encode(self, x):
        return self.nn.encode(x)

    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):
        # print(targets.size())
        batch_size, seq_len, n_class = outputs.size()

        embeddings = embeddings.reshape(batch_size * seq_len, -1)
        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)

        loss = self.loss(embeddings, outputs, targets)

        self.log(f"{log}_loss", loss, prog_bar=True)
        self.log(f"{log}_acc", self.acc(outputs, targets), prog_bar=True)
        self.log(f"{log}_f1", self.f1(outputs, targets), prog_bar=True)

        if log_metrics:
            self.log(f"{log}_ck", self.ck(outputs, targets))
            self.log(f"{log}_pr", self.pr(outputs, targets))
            self.log(f"{log}_rc", self.rc(outputs, targets))

        return loss

    def training_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs)

        return self.compute_loss(embeddings, outputs, targets)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        embeddings, outputs = self.encode(inputs)

        return self.compute_loss(embeddings, outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch

        embeddings, outputs = self.encode(inputs)

        return self.compute_loss(embeddings, outputs, targets, "test", log_metrics=True)
