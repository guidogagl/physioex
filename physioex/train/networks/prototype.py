from typing import Dict
import torch
import torch.nn as nn

import math

from collections import OrderedDict

from physioex.train.networks.base import SleepModule

# from 30-seconds of signal we need to extract the 5 second subsequence
# that is more relevant for the classification task

import seaborn as sns
import matplotlib.pyplot as plt

class ProtoSleepNet(SleepModule):
    def __init__(self, module_config : dict):
        super(ProtoSleepNet, self).__init__(NN(module_config), module_config)

        self.prototypes = self.nn.epoch_encoder.prototype

        self.step = 0

    def log_prototypes(self, log : str):        
        # prototypes shape : (N, hidden_size)
        # convert the model parameters to numpy
        prototypes = self.prototypes.prototypes.detach().cpu().numpy()

        # create the heatmap of the prototypes
        # y label equal to prototype index, x label equal to the feature index
        # display the cbar with diverging colors
        sns.heatmap(prototypes, cmap="RdPu", cbar=True)
        plt.xlabel("Features")
        plt.ylabel("Prototype")
        
        plt.title("Prototypes")
        self.logger.experiment.add_figure(f"{log}-age-corr", plt.gcf(), self.step )
        plt.close()
        

    
    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):
        self.step += 1
        
        if self.step % 250 == 0 and log == "train":
            self.log_prototypes(log)
        
        return super().compute_loss(embeddings, outputs, targets, log, log_metrics)
        
class NN(nn.Module):
    def __init__( self, 
        config : dict
    ):
        super(NN, self).__init__()
        
        assert 3000 % config["proto_lenght"] == 0, "The prototype lenght must be a divisor of 3000"
        
        
        self.epoch_encoder = EpochEncoder( 
            hidden_size = 3000 // config["proto_lenght"], 
            attention_size = 128, 
            N = config["N"], 
            n_prototypes = config["n_prototypes"], 
            temperature = 1.0 
        )
        
        self.sequence_encoder = SequenceEncoder(
            hidden_size = self.epoch_encoder.out_size,
            nhead = 4
        )
        
        self.clf = nn.Linear( self.epoch_encoder.out_size, config["n_classes"] )
    
    def encode( self, x ):
        # x shape : (batch_size, seq_len, n_chan, n_samp)
        batch_size, seq_len, _, _ = x.size()
        
        proto, _ = self.epoch_encoder( x )
        
        # proto shape : (batch_size, seq_len, N, hidden_size)
        # we selected N prototypes from each epoch in the sequence
        
        # sum the prototypes to get the epoch representation
        N = proto.size(2)
        proto = torch.sum( proto, dim=2 ) / N
        
        # proto shape : (batch_size, seq_len, hidden_size)
        # encode the sequence with the transformer        
        proto = self.sequence_encoder( proto ).reshape( batch_size*seq_len, -1 )
        
        clf = self.clf( proto ).reshape( batch_size, seq_len, -1 )
        proto = proto.reshape( batch_size, seq_len, -1 )        
        
        return proto, clf
            
    def forwad( self, x ):        
        x, y = self.encode( x )       
                
        return y


class SequenceEncoder( nn.Module ):
    def __init__(self,
        hidden_size : int,
        nhead : int,
    ):
        super(SequenceEncoder, self).__init__()
        assert  hidden_size % nhead == 0, "Hidden size must be a divisor of the number of heads"
        
        self.pe = PositionalEncoding(hidden_size, 1024)
        
        # autoregressive model for the sequence trasnformer based on the transformer architecture
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size, 
            dropout=0.25,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder( self.encoder, num_layers=1 )

    def forward( self, x):
        # x shape : (batch_size, seq_len, hidden_size)
        
        # encode the sequence with positional encoding
        x = self.pe(x)
        
        # encode the sequence with the transformer
        x = self.encoder(x)
        
        return x
    

class PrototypeLayer( nn.Module ):
    def __init__( self, 
        hidden_size : int,
        N : int = 1, # number of prototypes to learn
        ):
        super(PrototypeLayer, self).__init__()

        self.prototypes = nn.Parameter( torch.zeros( N, hidden_size ), requires_grad=True )
        
        #nn.init.uniform_(self.prototypes, 0, 1)
        
        # TODO inizilizzare a 0
        nn.init.constant_(self.prototypes, 0)
    
    def forward( self, x ):
        # x shape : (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.size()
        
        x = x.reshape( batch_size * seq_len, hidden_size )
        
        # calculate the  distance between each element of the sequence and the prototypes
        dist = torch.cdist( x, self.prototypes ) + 1e-20

        # dist shape : (batch_size * seq_len, N)
        # select the prototype with the minimum distance with gumbel softmax
        dist = - torch.log( dist )
        # dist shape : (batch_size * seq_len, N)
        
        
        logits = torch.nn.functional.gumbel_softmax(dist, tau=1.0, hard=True)
        
        dist = dist * logits
        
        print( "Dist shape: ", dist.shape)
        print( "Dist expected shape, " , (batch_size, 1) )

        # logits shape : (batch_size * seq_len, N)        
        # select the prototype with the maximum probability for each sample
        proto = torch.einsum( "bn, nh -> bh", logits, self.prototypes ) 
        
        # proto shape : (batch_size * seq_len, hidden_size)
        x = x - proto
        
        x = x.reshape( batch_size, seq_len, hidden_size )
        proto = proto.reshape( batch_size, seq_len, hidden_size )
        
        return proto, x
        
        
class EpochEncoder( nn.Module ):
    def __init__( self, 
        hidden_size : int,
        attention_size : int,
        N : int = 1, # number of elements to select from the epoch
        n_prototypes : int = 25, # number of elements to learn
        temperature : float = 1.0
        ):
        
        super(EpochEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.N = N
        
        # we need to extract frequency-related features from the signal
        self.conv1 = nn.Sequential( OrderedDict([
            ("conv1", nn.Conv1d(1, 32, 5)),
            ("relu1", nn.ReLU()),
            ("maxpool1", nn.MaxPool1d(5)),
            ("conv2", nn.Conv1d(32, 64, 5)),
            ("relu2", nn.ReLU()),
            ("maxpool2", nn.MaxPool1d(5)),
            ("conv3", nn.Conv1d(64, 128, 5)),
            ("relu3", nn.ReLU()),
            ("maxpool3", nn.MaxPool1d(5)),
            ("conv4", nn.Conv1d(128, 256, 3)),
            ("relu4", nn.ReLU()),
            ("flatten", nn.Flatten())
        ]))
        
        self.out_size = self.conv1(torch.randn(1, 1,  hidden_size)).shape[1]
        
        self.sampler = HardAttentionLayer(hidden_size, attention_size, N, temperature)
        
        self.prototype = PrototypeLayer( self.out_size, n_prototypes )
        
    def forward( self, x ):
        # x shape : (batch_size, seq_len, n_chan, n_samp)
        batch_size, seq_len, n_chan, n_samp = x.size()

        assert n_samp % self.hidden_size == 0, "Hidden size must be a divisor of the number of samples"

        x = x.reshape( batch_size * seq_len, n_chan*(n_samp//self.hidden_size), -1 )
        x = self.sampler( x ) # shape : (batch_size * seq_len, N, hidden_size)
        x = x.reshape( batch_size * seq_len * self.N, 1,  -1 )
        
        x = self.conv1( x ) # shape : (batch_size * seq_len * self.N, out_size)

        x = x.reshape( batch_size, seq_len*self.N, self.out_size )
        
        proto, residual = self.prototype( x )
        
        proto = proto.reshape( batch_size, seq_len, self.N, self.out_size )
        residual = residual.reshape( batch_size, seq_len, self.N, self.out_size )
        
        return proto, residual

class HardAttentionLayer(nn.Module):
    def __init__(self, 
            hidden_size : int,
            attention_size : int, 
            N : int = 1, # number of elements to select
            temperature : float = 1.0,
            encoding : nn.Module = None
        ):
        super(HardAttentionLayer, self).__init__()
        
        self.temperature = temperature
        
        self.pe = PositionalEncoding(hidden_size, 100)
        
        self.N = N

        self.Q = nn.Linear(hidden_size, attention_size * N, bias = False)
        self.K = nn.Linear(hidden_size, attention_size * N, bias = False)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.size()

        # encode the sequence with positional encoding
        pos_emb = self.pe(x)

        # calculate the query and key
        Q = self.Q(pos_emb)
        K = self.K(pos_emb)
        
        Q = Q.reshape( batch_size, sequence_length, self.N, -1 ).transpose(1, 2)
        K = K.reshape( batch_size, sequence_length, self.N, -1 ).transpose(1, 2)
        
        attention = torch.einsum( "bnsh,bnth -> bnst", Q, K ) / ( hidden_size ** (1/2) )
        attention = torch.sum(attention, dim=-1) / sequence_length

        # attention shape : (batch_size * N, sequence_length)
        logits = attention.reshape( batch_size * self.N, sequence_length )                
        # apply the Gumbel-Softmax trick to select the N most important elements
        alphas = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=True)
        alphas = alphas.reshape( batch_size, self.N, sequence_length )
        
        # select N elements from the sequence x using alphas
        x = torch.einsum( "bns, bsh -> bnh", alphas, x )
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # shape x : (batch_size, seq_len, hidden_size)
        x = x + self.pe[:, : x.size(1)]
        return x
