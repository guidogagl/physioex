import importlib
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm

def confusion_matrix_to_dict(confusion_matrix):
    """
    Convert a confusion matrix tensor to a dictionary of scalars.
    """
    cm = confusion_matrix.cpu().numpy()
    cm_dict = {}
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cm_dict[f"cm_{i}_{j}"] = cm[i, j]
    return cm_dict

class SleepModule(pl.LightningModule):
    def __init__(self, nn: nn.Module, config: Dict):
        super(SleepModule, self).__init__()
        self.save_hyperparameters(ignore=["nn"])
        self.nn = nn

        self.n_classes = config["n_classes"]

        if self.n_classes > 1:
            # classification experiment
            self.wacc = tm.Accuracy(
                task="multiclass", num_classes=config["n_classes"], average="weighted"
            )
            self.macc = tm.Accuracy(
                task="multiclass", num_classes=config["n_classes"], average="macro"
            )
            self.wf1 = tm.F1Score(
                task="multiclass", num_classes=config["n_classes"], average="weighted"
            )
            self.mf1 = tm.F1Score(
                task="multiclass", num_classes=config["n_classes"], average="macro"
            )
            self.ck = tm.CohenKappa(task="multiclass", num_classes=config["n_classes"])
            self.pr = tm.Precision(
                task="multiclass", num_classes=config["n_classes"], average="weighted"
            )
            self.rc = tm.Recall(
                task="multiclass", num_classes=config["n_classes"], average="weighted"
            )
            
            self.cm = tm.ConfusionMatrix(task="multiclass",
                                            num_classes=config["n_classes"],
                                            normalize=None)
        elif self.n_classes == 1:
            # regression experiment
            self.mse = tm.MeanSquaredError()
            self.mae = tm.MeanAbsoluteError()
            self.r2 = tm.R2Score()

        # loss
        loss_module, loss_class = config["loss"].split(":")
        self.loss = getattr(importlib.import_module(loss_module), loss_class)(
            **config["loss_kwargs"]
        )
        self.module_config = config

        # learning rate

        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]

    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="min",
            factor=0.5,
            patience=1,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            # verbose=True,
        )
        scheduler = {
            "scheduler": scheduler,
            "name": "lr_scheduler",
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [self.opt], [scheduler]

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

        batch_size, seq_len, n_class = outputs.size()

        embeddings = embeddings.reshape(batch_size * seq_len, -1)
        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)
        
    
        if self.n_classes > 1:
            loss = self.loss(embeddings, outputs, targets)

            self.log(f"{log}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{log}_acc", self.wacc(outputs, targets), prog_bar=True, sync_dist=True)
            self.log(f"{log}_f1", self.wf1(outputs, targets), prog_bar=True, sync_dist=True)
            
            conf_dict = confusion_matrix_to_dict( self.cm(outputs, targets) )            
            # TODO: log confusion matrix
            #self.log_dict(f"{log}_confmat", conf_dict, sync_dist=True) 
        else:
            outputs = outputs.view(-1)

            loss = self.loss(embeddings, outputs, targets)

            self.log(f"{log}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{log}_r2", self.r2(outputs, targets), prog_bar=True, sync_dist=True)
            self.log(f"{log}_mae", self.mae(outputs, targets), prog_bar=True, sync_dist=True)
            self.log(f"{log}_mse", self.mse(outputs, targets), prog_bar=True, sync_dist=True)

        if log_metrics and self.n_classes > 1:
            self.log(f"{log}_ck", self.ck(outputs, targets), sync_dist=True)
            self.log(f"{log}_pr", self.pr(outputs, targets), sync_dist=True)
            self.log(f"{log}_rc", self.rc(outputs, targets), sync_dist=True)
            self.log(f"{log}_macc", self.macc(outputs, targets), sync_dist=True)
            self.log(f"{log}_mf1", self.mf1(outputs, targets), sync_dist=True)
            self.cm['cm'].update(outputs, targets)

        return loss

    def training_step(self, batch, batch_idx):
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))

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