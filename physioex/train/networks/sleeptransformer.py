from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from physioex.train.networks.base import SleepModule
from physioex.train.networks.seqsleepnet import AttentionLayer

import math
import torchmetrics as tm

module_config = dict()

class MiceTransformer(SleepModule):
    def __init__(self, module_config=module_config):
        super(MiceTransformer, self).__init__(Net(module_config), module_config)
        
        self.wacc = tm.Accuracy(
            task="multiclass", num_classes=3, average="weighted"
        )
        self.macc = tm.Accuracy(
            task="multiclass", num_classes=3, average="macro"
        )
        self.wf1 = tm.F1Score(
            task="multiclass", num_classes=3, average="weighted"
        )
        self.mf1 = tm.F1Score(
            task="multiclass", num_classes=3, average="macro"
        )
        self.ck = tm.CohenKappa(task="multiclass", num_classes=3)
        self.pr = tm.Precision(
            task="multiclass", num_classes=3, average="weighted"
        )
        self.rc = tm.Recall(
            task="multiclass", num_classes=3, average="weighted"
        )
        self.cm = {'cm': tm.ConfusionMatrix(task="multiclass",
                                            num_classes=3,
                                            normalize=None).cuda()
        }
        self.cm['cm5'] = tm.ConfusionMatrix(task="multiclass",
                                            num_classes=5,
                                            normalize=None).cuda()

        
        self.pe = PositionalEncoding( 128 )
        
        t_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        
        self.mice_to_human_encoder = nn.TransformerEncoder(
            t_layer, 
            num_layers=4
        )
        
        # set all the parameters of self.nn with require_grads = False
        for param in self.nn.parameters():
            param.requires_grad = False
            
    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):
        
        batch_size, seq_len, n_class = outputs.size()
        
        # from 5 classes W, N1, N2, N3, R
        # to 3 classes W, N, R
        # N = N1 + N2 + N3
        
        if log_metrics == True:
            self.cm['cm5'].update(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())
        
        W = outputs[:, :, 0]
        N = torch.maximum( outputs[:, :, 1], outputs[:, :, 2] )
        N = torch.maximum( N, outputs[:, :, 3] )

        R = outputs[:, :, 4]

        outputs = torch.stack([W, N, R], dim=2)
        
        return super(MiceTransformer, self).compute_loss(embeddings, outputs, targets, log, log_metrics)
    
    
    def encode( self, x ):
        batch, L, nchan, T, F = x.shape
        # T in mice is equal to 17, but in humans is equal to 29
        # let's make it 34
       
        x = torch.permute( x, (0, 1, 2, 4, 3)).reshape( batch, L, nchan, F, T, 1)
        x = x.expand(batch, L, nchan, F, T, 2).reshape( batch, L, nchan, F, -1 )
        x = torch.permute( x, (0, 1, 2, 4, 3))
        
        x = self.pe(x)
        x = self.mice_to_human_encoder(x)
        
        with torch.no_grad():
            return self.nn.encode(x)
        
    def forward(self, x):
        batch, L, nchan, T, F = x.shape
        # T in mice is equal to 17, but in humans is equal to 29
        # let's make it 34
       
        x = torch.permute( x, (0, 1, 2, 4, 3)).reshape( batch, L, nchan, F, T, 1)
        x = x.expand(batch, L, nchan, F, T, 2).reshape( batch, L, nchan, F, -1 )
        x = torch.permute( x, (0, 1, 2, 4, 3))
        
        x = self.pe(x)
        x = self.mice_to_human_encoder(x)
        
        with torch.no_grad():
            return self.nn(x)
        

class SleepTransformer(SleepModule):
    def __init__(self, module_config=module_config):
        super(SleepTransformer, self).__init__(Net(module_config), module_config)


class Net(nn.Module):
    def __init__(self, module_config=module_config):
        super().__init__()

        self.epoch_encoder = EpochEncoder(module_config)
        self.sequence_encoder = SequenceEncoder(module_config)

    def encode(self, x):

        batch, L, nchan, T, F = x.size()

        x = x.reshape(-1, nchan, T, F)

        x = self.epoch_encoder(x)

        x = x.reshape(batch, L, -1)

        x = self.sequence_encoder.encode(x)

        y = self.sequence_encoder.clf(x)

        return x, y

    def forward(self, x):
        x, y = self.encode(x)
        return y


class EpochEncoder(nn.Module):
    def __init__(self, module_config):
        super(EpochEncoder, self).__init__()

        self.pe = PositionalEncoding( 128 * module_config["in_channels"] )
        
        t_layer = nn.TransformerEncoderLayer(
            d_model=128 * module_config["in_channels"],
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        
        self.encoder = nn.TransformerEncoder(
            t_layer, 
            num_layers=4
        )
        
        self.attention = AttentionLayer(128 * module_config["in_channels"], 128)
        
    def forward(self, x):
        batch_size, in_chans, T, F = x.size()

        x = x[ ..., :128 ]
        x = x.permute(0, 2, 1, 3).reshape(batch_size, T, 128 * in_chans)        
                
        x = self.pe(x)

        x = self.encoder(x)
        x = self.attention(x)
        
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, module_config):
        super(SequenceEncoder, self).__init__()

        self.pe = PositionalEncoding( 128 * module_config["in_channels"] )
        t_layer = nn.TransformerEncoderLayer(
            d_model=128 * module_config["in_channels"],
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            t_layer, 
            num_layers=4
        )
        
        self.clf = nn.Linear(
            128 * module_config["in_channels"], module_config["n_classes"]
        )

    def forward(self, x):
        x = self.pe(x)
        x = self.encoder(x)
        x = self.clf(x)
        return x

    def encode(self, x):
        x = self.pe(x)
        x = self.encoder(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # shape x : (batch_size, seq_len, hidden_size)
        x = x + self.pe[:, : x.size(1)]
        return x
