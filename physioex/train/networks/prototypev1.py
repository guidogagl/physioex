from typing import Dict
import torch
import torch.nn as nn

import math

from collections import OrderedDict

from physioex.train.networks.base import SleepModule

from physioex.train.networks.sleeptransformer import PositionalEncoding
from physioex.train.networks.seqsleepnet import AttentionLayer


import seaborn as sns
import matplotlib.pyplot as plt

module_config = dict()

import torch.distributions as dist
import torch.nn.functional as F

import time
import copy 

class ProtoLoss( nn.Module ):    
    def __init__(self):
        super(ProtoLoss, self).__init__()
        self.target_loss = nn.CrossEntropyLoss()
        self.multi_channels_loss = nn.CrossEntropyLoss()
    
        
    def forward(self, preds, targets, multi_channels_preds):
        
        batch, L, nchan, nclasses = multi_channels_preds.size()
         
        # target loss 
        tl = self.target_loss( preds.reshape( -1, nclasses), targets.reshape(-1) )
        
        # multi channel loss
        targets = targets.reshape( batch, L, 1).repeat( 1, 1, nchan )
        mcl = self.multi_channels_loss( multi_channels_preds.reshape( -1, nclasses ), targets.reshape( -1 ))
        
        return tl, mcl


def voting_strategy( model : torch.nn.Module, inputs : torch.Tensor, L : int  ):
    
    batch_size, night_lenght, n_channels, _, _ = inputs.size()

    (commit_loss, mcy), outputs = model.encode(inputs)   

    outputs = torch.zeros( batch_size, night_lenght, 5, device=inputs.device, dtype=inputs.dtype )
    mcy = torch.zeros( batch_size, night_lenght, n_channels, 5, device=inputs.device, dtype=inputs.dtype )   

    commit_loss = 0
    
    # input shape is ( bach_size, night_lenght, n_channels, ... )
    # segment the input in self.L segments with a sliding window of stride 1 and size self.L
    for i in range(0, inputs.size(1) - L + 1, 1):
        input_segment = inputs[:, i:i+L]
        (seg_cl, seg_mcy), seg_outputs = model.encode(input_segment)
        
        outputs[:, i:i+L] += torch.nn.functional.softmax( seg_outputs, dim=-1 )
        mcy[:, i:i+L] += torch.nn.functional.softmax( seg_mcy, dim=-1 )
        commit_loss += seg_cl

    commit_loss = commit_loss / (inputs.size(1) - L + 1)
    
    return (commit_loss, mcy), outputs

class ProtoSleepNet(SleepModule):
    def __init__(self, module_config: dict = module_config):
        super(ProtoSleepNet, self).__init__(NN(module_config), module_config)

        self.loss = ProtoLoss()
        
    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):
        
        commit_loss, multi_channels_preds = embeddings
        
        batch_size, seq_len, n_class = outputs.size()

        outputs = outputs.reshape(-1, n_class)
        targets = targets.reshape(-1)

        tl, mcl = self.loss( outputs, targets, multi_channels_preds )

        loss = commit_loss + mcl + tl
        
        self.log(f"{log}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{log}_acc", self.wacc(outputs, targets), prog_bar=True, sync_dist=True)

        nchan = multi_channels_preds.size(2)
        multi_channels_preds = multi_channels_preds.reshape( -1, nchan, n_class )
        
        for c in range( nchan ):
            self.log(f"{log}_{c}_acc", self.wacc(multi_channels_preds[:, c], targets), prog_bar=True, sync_dist=True)
            
        self.log(f"{log}_commit_loss", commit_loss, prog_bar=True, sync_dist=True)

        if log_metrics:
            self.log(f"{log}_ck", self.ck(outputs, targets), sync_dist=True)
            self.log(f"{log}_pr", self.pr(outputs, targets), sync_dist=True)
            self.log(f"{log}_rc", self.rc(outputs, targets), sync_dist=True)
            self.log(f"{log}_macc", self.macc(outputs, targets), sync_dist=True)
            self.log(f"{log}_mf1", self.mf1(outputs, targets), sync_dist=True)
        
        return loss


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


class SectionEncoder( nn.Module ):
    def __init__(self, 
        n_layers = 4,
        input_dim = 128,
        hidden_dim = 1024,
        dropout = 0.1, 
        activation_func = nn.GELU(),
    ):
        super(SectionEncoder, self).__init__()
        
        self.pe = PositionalEncoding(input_dim, 100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # each channel is processed as an independent sample, otherwise it would be 128 * in_channels
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation_func,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        self.additive_attention = AttentionLayer(
            128, 
            64,
        )
        
    def forward(self, x):
        # batch, seq_len, hidden_size = x.size()        
        x = self.pe(x)
        x = self.encoder(x)
        x = self.additive_attention(x)
        
        return x

class NN(nn.Module):
    def __init__(self, module_config = module_config):
        super(NN, self).__init__()

        from physioex.train.networks.sleeptransformer import SequenceEncoder
        
        from vector_quantize_pytorch import SimVQ
        
        self.in_channels = module_config["in_channels"]
        self.n_prototypes = module_config["n_prototypes"]
        
        # assert self.in_channels == len( self.n_prototypes ), "Err: Number of prototypes must match the number of channels of the model"
        
        self.S = module_config["S"]
        self.N = module_config["N"]

        self.encoder = SectionEncoder(
            input_dim = 128,
            hidden_dim = 1024,
            dropout = 0.1, 
            activation_func = nn.GELU(),
        )

        self.sampler = HardAttentionLayer(
            hidden_size = 128,
            attention_size = 1024, 
            N = self.N,           
        )
        
        self.prototype = SimVQ(
                    dim = 128,
                    codebook_size = self.n_prototypes,  
                    rotation_trick = True,  # use rotation trick from Fifty et al.
                    channel_first=False
        ) 
        
        # need to change n_channels to 1 to create sequence encoder, because each channel is processed independently
        modified_config = copy.deepcopy(module_config)
        modified_config["in_channels"] = 1
        self.sequence_encoder = SequenceEncoder( modified_config )
        
        if self.in_channels > 1:
            self.channels_sampler = HardAttentionLayer(
                hidden_size = 128,
                attention_size = 128,
                N = 1
            )

        self.clf = self.sequence_encoder.clf
        
    def encode(self, x):
        # x shape : (batch_size, seq_len, n_chan, n_samp)
        batch, L, nchan, T, F = x.size()
        
        #start_time = time.time()
        _, p, _, commit_loss, _ = self.get_prototypes( x ) # batch, L, nchan, N, 128
        #print(f"Prototypes computed in {time.time() - start_time:.2f} seconds")
        
        # average prototypes
        p = p.mean( dim = 3 ).reshape( batch, L, nchan, 128).permute( 0, 2, 1, 3)
        p = p.reshape( -1, L, 128 )        
        
        ### sequence learning ##### 
        start_time = time.time()
        p = self.sequence_encoder.encode( p ) # out -1, L, 128
        #print(f"Sequence encoding computed in {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()                
        mcy = self.clf( p.reshape( -1, 128) ).reshape( batch, nchan, L, -1).permute( 0, 2, 1, 3 )
        #print(f"Multi-channel classification computed in {time.time() - start_time:.2f} seconds")
                
        ### channel picking
        start_time = time.time()
        p = p.reshape( batch, nchan, L, 128).permute( 0, 2, 1, 3).reshape( -1, nchan, 128)
        
        if self.in_channels > 1:
            p, _ = self.channels_sampler( p ) 
        
        #print(f"Channel sampling computed in {time.time() - start_time:.2f} seconds")        
        p = p.reshape( batch*L, 128 )
         
        #start_time = time.time()
        y = self.clf(p).reshape( batch, L, -1)
        #print(f"Final classification computed in {time.time() - start_time:.2f} seconds")
                
        return (commit_loss, mcy), y 

    
    def get_prototypes(self, x):
        # x shape : (batch_size, seq_len, n_chan, n_samp)
        batch, L, nchan, T, _ = x.size()
        
        #### epoch encoding ####
        x = x.reshape( batch * L * nchan, T, -1 )
        x = x[ ..., :128 ]

        x = F.pad( x, (0, 0, 0, 1) ).reshape(-1, 6, 5, 128 ) 

        _, alphas = self.sampler( x.sum( dim = -2 ) )  
        
        x = torch.einsum("bns, bsh -> bnh", alphas, x.reshape( -1, 6, 5*128 ) ).reshape( -1, 5, 128 )
        
        # now we want to use a transformer encoder to encode the sequence
        x = self.encoder(x)

        prototypes, indexes, commit_loss = self.prototype(x)
        
        prototypes = prototypes.reshape( batch, L, nchan, self.N, 128 )
        indexes = indexes.reshape( batch, L, nchan, self.N )        

        return x, prototypes, indexes, commit_loss, alphas

    def forward(self, x):
        x, y = self.encode(x)

        return y




class HardAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_size: int,
        N: int = 1,  # number of elements to select
        temperature: float = 0.1,
    ):
        super(HardAttentionLayer, self).__init__()

        self.temperature = temperature

        self.pe = PositionalEncoding(hidden_size, 100)

        self.N = N

        self.Q = nn.Linear(hidden_size, attention_size * N, bias=False)
        self.K = nn.Linear(hidden_size, attention_size * N, bias=False)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.size()

        # encode the sequence with positional encoding
        pos_emb = self.pe(x)

        # calculate the query and key
        Q = self.Q(pos_emb)
        K = self.K(pos_emb)

        Q = Q.reshape(batch_size, sequence_length, self.N, -1).transpose(1, 2)
        K = K.reshape(batch_size, sequence_length, self.N, -1).transpose(1, 2)

        attention = torch.einsum("bnsh,bnth -> bnst", Q, K) / (hidden_size ** (1 / 2))
        attention = torch.sum(attention, dim=-1) / sequence_length

        # attention shape : (batch_size * N, sequence_length)
        logits = attention.reshape(batch_size * self.N, sequence_length)
        # apply the Gumbel-Softmax trick to select the N most important elements
        alphas = torch.nn.functional.gumbel_softmax(
            logits, tau=self.temperature, hard=True
        )
        alphas = alphas.reshape(batch_size, self.N, sequence_length)

        # select N elements from the sequence x using alphas
        x = torch.einsum("bns, bsh -> bnh", alphas, x)

        return x, alphas