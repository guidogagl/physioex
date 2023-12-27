import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
import torchmetrics as tm

from typing import Type, Dict

from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer

class SeqtoSeqModule( nn.Module ):
    def __init__(self, encoder, decoder):
        super(SeqtoSeqModule, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder

    def forward(self, x):
        batch_size, sequence_lenght, modalities, input_dim = x.size()        

        x = x.reshape(batch_size * sequence_lenght, modalities, input_dim)
        x = self.encoder(x)

        x = x.reshape( batch_size, sequence_lenght, -1)
        
        return self.decoder(x)

    def encode(self, x):
        batch_size, sequence_lenght, modalities, input_dim = x.size()        

        x = x.reshape(batch_size * sequence_lenght, modalities, input_dim)
        x = self.encoder(x)

        x = x.reshape( batch_size, sequence_lenght, -1)
        
        return self.decoder.encode(x), self.decoder(x)


class SeqtoSeq(pl.LightningModule):
    def __init__(
        self, 
        encoder : nn.Module, 
        decoder : nn.Module,
        config : Dict
    ):
        super(SeqtoSeq, self).__init__()
        self.nn = SeqtoSeqModule( encoder, decoder )

        # metrics
        self.acc = tm.Accuracy(task="multiclass", num_classes=config["n_classes"], average="weighted")
        self.f1 = tm.F1Score(task="multiclass", num_classes=config["n_classes"], average="weighted")
        self.ck = tm.CohenKappa(task="multiclass", num_classes=config["n_classes"])
        self.pr = tm.Precision(task="multiclass", num_classes=config["n_classes"], average="weighted")
        self.rc = tm.Recall(task="multiclass", num_classes=config["n_classes"], average="weighted")

        # loss
        self.loss = nn.CrossEntropyLoss()

        self.module_config = config

    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr= float(self.module_config["learning_rate"]),
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
        self, outputs, targets, log: str = "train", log_metrics: bool = False
    ):
        # print(targets.size())
        batch_size, seq_len, n_class = outputs.size()

        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)

        loss = self.loss(outputs, targets)

        self.log(log + "_loss", loss, prog_bar=True)
        self.log(log + "_acc", self.acc(outputs, targets), prog_bar=True)

        if log_metrics:
            self.log(log + "_f1", self.f1(outputs, targets))
            self.log(log + "_ck", self.ck(outputs, targets))
            self.log(log + "_pr", self.pr(outputs, targets))
            self.log(log + "_rc", self.rc(outputs, targets))
        return loss

    def training_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch
        outputs = self(inputs)

        return self.compute_loss(outputs, targets)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        outputs = self(inputs)
        return self.compute_loss(outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch
        outputs = self(inputs)

        return self.compute_loss(outputs, targets, "test", log_metrics=True)

    def predict_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        outputs = self(inputs)

        self.compute_loss(outputs, targets, "predict", log_metrics=True)

        return outputs


class ContrSeqtoSeqModule( SeqtoSeqModule ):
    def __init__(self, encoder, decoder, latent_space_dim, n_classes):
        super(ContrSeqtoSeqModule, self).__init__(encoder, decoder)
        self.ls_norm = nn.LayerNorm( latent_space_dim )
        self.clf = nn.Linear(latent_space_dim, n_classes)

    def forward(self, x):
        embeddings = super().forward(x)
        embeddings = nn.ReLU()( embeddings ) 
        embeddings = self.ls_norm( embeddings )
        
        batch_size, seq_len, ls_dim = embeddings.size()
        embeddings = embeddings.reshape(-1, ls_dim)

        outputs = self.clf( embeddings )

        embeddings = embeddings.reshape(batch_size, seq_len, ls_dim)
        outputs = outputs.reshape(batch_size, seq_len, -1)
        return embeddings, outputs


class ContrSeqtoSeq( SeqtoSeq ):
    def __init__(
        self, 
        encoder : nn.Module, 
        decoder : nn.Module,
        config : Dict
    ):
        super(ContrSeqtoSeq, self).__init__(None, None, config=config)
        
        self.miner = miners.MultiSimilarityMiner()
        self.contr_loss = losses.TripletMarginLoss(
                                    distance = CosineSimilarity(), 
                                    reducer = ThresholdReducer(high=0.3), 
                                    embedding_regularizer = LpRegularizer())
        
        self.nn = ContrSeqtoSeqModule(encoder, decoder, config["latent_space_dim"], config["n_classes"])
    
    def compute_loss(
        self, outputs, targets, log: str = "train", log_metrics: bool = False
    ):
        
        projections, predictions = outputs
        loss = super().compute_loss(predictions, targets, log, log_metrics)

        batch_size, seq_len, embedding_size = projections.size()

        projections = projections.reshape( batch_size * seq_len, -1)
        targets = targets.reshape(-1)

        indx = torch.randperm(len(targets))[: batch_size]

        projections, targets = projections[indx], targets[indx]

        hard_pairs = self.miner( projections, targets )
        contr_loss = self.contr_loss( projections, targets, hard_pairs)

        return loss + contr_loss



