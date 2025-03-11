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


class NN(nn.Module):
    def __init__(self, module_config = module_config):
        super(NN, self).__init__()

        from physioex.train.networks.sleeptransformer import EpochEncoder as SectionEncoder
        from physioex.train.networks.sleeptransformer import SequenceEncoder
        
        from vector_quantize_pytorch import SimVQ
        
        in_channels = module_config["in_channels"]
        module_config["in_channels"] = 1
        
        self.S = module_config["S"]
        self.N = module_config["N"]
                
        self.pe = PositionalEncoding( 128 )
        
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

        self.sampler = HardAttentionLayer(
            hidden_size = 128,
            attention_size = 1024, #129 * self.S,
            N = module_config["N"],            
        )
        
        self.prototype = SimVQ(
            dim = 128,
            codebook_size = 50,
            rotation_trick = True,  # use rotation trick from Fifty et al.
            channel_first=False
        )
                
        self.sequence_encoder = SequenceEncoder( module_config )
        
        if in_channels > 1:
            self.channels_sampler = HardAttentionLayer(
                hidden_size = 128,
                attention_size = 128,
                N = 1
            )
        else :
            self.channels_sampler = nn.Identity()

        self.clf = self.sequence_encoder.clf
        
    def encode(self, x):
        # x shape : (batch_size, seq_len, n_chan, n_samp)
        batch, L, nchan, T, F = x.size()
        
        start_time = time.time()
        p, _, commit_loss = self.get_prototypes( x ) # batch, L, nchan, N, 128
        end_time = time.time()
        #print( f"Prototype time: {end_time - start_time}")
        
        start_time = time.time()        
        # average prototypes
        p = p.mean( dim = 3 ).reshape( batch, L, nchan, 128).permute( 0, 2, 1, 3)
        p = p.reshape( -1, L, 128 )        
        end_time = time.time()
        #print( f"Prototype average time: {end_time - start_time}")
        
        ### sequence learning ##### 
        start_time = time.time()
        p = self.sequence_encoder.encode( p ) # out -1, L, 128
        end_time = time.time()
        #print( f"Sequence time: {end_time - start_time}")
        
        start_time = time.time()
        ### multichannel optimization:
        mcy = self.clf( p.reshape( -1, 128) ).reshape( batch, nchan, L, -1).permute( 0, 2, 1, 3 )
        # batch, L, nchan, nclasses
                
        ### channel picking
        p = p.reshape( batch, nchan, L, 128).permute( 0, 2, 1, 3).reshape( -1, nchan, 128)
 
        p = self.channels_sampler( p ).reshape( batch*L, 128 )
        y = self.clf(p).reshape( batch, L, -1)
        end_time = time.time()
        #print( f"Multichannel time: {end_time - start_time}")
        
        return (commit_loss, mcy) , y 

    
    def get_prototypes(self, x):
        # x shape : (batch_size, seq_len, n_chan, n_samp)
        batch, L, nchan, T, F = x.size()
        
        #### epoch encoding ####
        x = x.reshape( batch * L * nchan, T, F )
        x = x[ ..., :128 ]
        
        x = self.pe(x)
        x = self.encoder(x)
    
        # out : -1, 1, T, 128                
        x = self.sampler( x ) # out -1, N, 128
        
        p, indx, commit_loss = self.prototype( x.reshape(-1, 128) )                

        p = p.reshape(batch, L, nchan, self.N, 128 )
        indx = indx.reshape(batch, L, nchan, self.N )

        return p, indx, commit_loss

    def forwad(self, x):
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

        return x

