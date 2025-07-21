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

def voting_strategy( model : torch.nn.Module, inputs : torch.Tensor, L : int  ):

    batch_size, night_length, n_channels, _, _ = inputs.size()
    
    embeddings_sample, _ = model.encode(inputs[:, 0:L])   
    embeddings_dim = embeddings_sample.shape[-1]
    
    outputs = torch.zeros(
        batch_size, night_length, model.n_classes, device=inputs.device, dtype=inputs.dtype 
    )
    
    embeddings = torch.zeros(
        batch_size, night_length, embeddings_dim, device=inputs.device, dtype=inputs.dtype 
    )

    # input shape is ( bach_size, night_lenght, n_channels, ... )
    # segment the input in self.L segments with a sliding window of stride 1 and size self.L
    for i in range(0, inputs.size(1) - L + 1, 1):
        input_segment = inputs[:, i:i+L]
        seg_emb, seg_outputs = model.encode(input_segment)
        
        outputs[:, i:i+L] += torch.nn.functional.softmax( seg_outputs, dim=-1 )
        embeddings[:, i:i+L] += seg_emb 
    
    return embeddings, outputs

class SleepModule(pl.LightningModule):
    def __init__(self, nn: nn.Module, config: Dict):
        super(SleepModule, self).__init__()
        self.save_hyperparameters(ignore=["nn"])
        self.nn = nn
        self.L = config["sequence_length"]
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
            "monitor": "val/loss",
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

            self.log(f"{log}/loss", loss, prog_bar=True, sync_dist=True,  on_epoch = False if log == "train" else True)
            self.log(f"{log}/acc", self.wacc(outputs, targets), prog_bar=True, sync_dist=True,  on_epoch = False if log == "train" else True)
            self.log(f"{log}/f1", self.wf1(outputs, targets), prog_bar=True, sync_dist=True,  on_epoch = False if log == "train" else True)

            if log == "val":
                self.log(f"{log}_acc", self.wacc(outputs, targets), sync_dist=True,  on_epoch = False if log == "train" else True)

            conf_dict = confusion_matrix_to_dict(self.cm(outputs, targets))
            # TODO: log confusion matrix
            #self.log_dict(f"{log}_confmat", conf_dict, sync_dist=True) 
        else:
            outputs = outputs.view(-1)

            loss = self.loss(embeddings, outputs, targets)

            self.log(f"{log}/loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{log}/r2", self.r2(outputs, targets), prog_bar=True, sync_dist=True)
            self.log(f"{log}/mae", self.mae(outputs, targets), prog_bar=True, sync_dist=True)
            self.log(f"{log}/mse", self.mse(outputs, targets), prog_bar=True, sync_dist=True)

        if log_metrics and self.n_classes > 1:
            self.log(f"{log}/ck", self.ck(outputs, targets), sync_dist=True, on_epoch = False if log == "train" else True)
            self.log(f"{log}/pr", self.pr(outputs, targets), sync_dist=True, on_epoch = False if log == "train" else True)
            self.log(f"{log}/rc", self.rc(outputs, targets), sync_dist=True, on_epoch = False if log == "train" else True)
            self.log(f"{log}/macc", self.macc(outputs, targets), sync_dist=True, on_epoch = False if log == "train" else True)
            self.log(f"{log}/mf1", self.mf1(outputs, targets), sync_dist=True, on_epoch = False if log =="train" else True )

            self.cm['cm'].update(outputs, targets)

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
        
        embeddings , outputs = voting_strategy(self, inputs, self.L)
        
        return self.compute_loss(embeddings, outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets, subjects, dataset_idx = batch

        embeddings , outputs = voting_strategy(self, inputs, self.L)

        return self.compute_loss(embeddings, outputs, targets, "test", log_metrics=True)