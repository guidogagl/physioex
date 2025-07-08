import torch
import torch.nn as nn

from physioex.train.networks.sleeptransformer import PositionalEncoding

from physioex.train.networks.prototype import ProtoSleepNet, ProtoSleepModule

module_config = dict()


class ProtoSleepTransformerNet(ProtoSleepModule):
    def __init__(self, module_config: dict = module_config):
        super(ProtoSleepTransformerNet, self).__init__(NN(module_config), module_config)


class NN( ProtoSleepNet ):
    def __init__(self, module_config = module_config):
        super(NN, self).__init__( module_config ) 

        self.e_encoder = EpochEncoder()
        self.s_encoder = SequenceEncoder()

    def epoch_encoder(self, x):
        return self.e_encoder(x)
    def sequence_encoder(self, x):
        return self.s_encoder(x)

class EpochEncoder( nn.Module ):
    def __init__(self):
        super(EpochEncoder, self).__init__()

        self.pe = PositionalEncoding( 128 )
        
        t_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            t_layer,
            num_layers=4,
        )

    def forward(self, x):
        batch, nchan, T, F = x.size()
        x = x.reshape( batch*nchan, T, F )[..., :128]

        x = self.pe(x)
        x = self.encoder(x)

        return x.reshape(batch, nchan, T, -1)  # (batch, nchan, 128)

class SequenceEncoder(nn.Module):
    def __init__(self):
        super(SequenceEncoder, self).__init__()

        self.pe = PositionalEncoding( 128 )
        t_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            t_layer, 
            num_layers=4
        )
    

    def forward(self, x):
        batch, L, nchan, _ = x.size()
        x = x.permute(0, 2, 1, 3).reshape(batch * nchan, L,  128) # batch*nchan, L, 128
        x = self.pe(x)
        x = self.encoder(x)
        return x.reshape(batch, nchan, L, -1).permute(0, 2, 1, 3)  # (batch, L, nchan, 128)
        
