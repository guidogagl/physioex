import torch
import torch.nn as nn

from physioex.train.networks.seqsleepnet import LearnableFilterbank

from physioex.train.networks.prototype import ProtoSleepNet, ProtoSleepModule



class ProtoSeqSleepNet(ProtoSleepModule):
    def __init__(self, module_config: dict ):
        super(ProtoSeqSleepNet, self).__init__(NN(module_config), module_config)


class NN( ProtoSleepNet ):
    def __init__(self, module_config : dict ):
        super(NN, self).__init__( module_config )

        self.e_encoder = EpochEncoder()
        self.s_encoder = SequenceEncoder()

    def epoch_encoder(self, x):
        return self.e_encoder(x)
    def sequence_encoder(self, x):
        return self.s_encoder(x)

class ChannelEncoder(nn.Module):
    def __init__(self):
        super(ChannelEncoder, self).__init__()
        self.filtbank = LearnableFilterbank(
            in_chan = 1,
            nfilt = 32
        )

        self.birnn = nn.LSTM(
            input_size = 32,
            hidden_size = 64,
            num_layers = 4,
            batch_first=True,
            bidirectional=True,
        )

        self.norm = nn.LayerNorm(128)
    
    def forward(self, x):
        batch, T, F = x.size()
        
        x = self.filtbank(x)
        x, _ = self.birnn(x)                         
        x = self.norm(x)

        return x

class EpochEncoder( nn.Module ):
    def __init__(self):
        super(EpochEncoder, self).__init__()

        self.eeg_encoder = ChannelEncoder()
        self.eog_encoder = ChannelEncoder()
        self.emg_encoder = ChannelEncoder()


    def forward(self, x):
        batch, nchan, T, F = x.size()

        eeg = self.eeg_encoder(x[:, 0])
        eog = self.eog_encoder(x[:, 1])
        emg = self.emg_encoder(x[:, 2])

        eeg = eeg.reshape(batch, 1, T, -1)
        eog = eog.reshape(batch, 1, T, -1)
        emg = emg.reshape(batch, 1, T, -1)

        x = torch.cat((eeg, eog, emg), dim=1)

        return x

class SequenceEncoder(nn.Module):
    def __init__(self):
        super(SequenceEncoder, self).__init__()

        self.eeg = nn.GRU(
            input_size= 128,
            hidden_size=64,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )
        self.eog = nn.GRU(
            input_size= 128,
            hidden_size=64,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )
        self.emg = nn.GRU(
            input_size= 128,
            hidden_size=64,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )
        
    

    def forward(self, x):
        batch, L, nchan, _ = x.size()

        eeg, _ = self.eeg(x[:,:, 0])
        eog, _ = self.eog(x[:,:, 1])
        emg, _ = self.emg(x[:,:, 2])

        eeg = eeg.reshape(batch, L, 1, -1)
        eog = eog.reshape(batch, L, 1, -1)
        emg = emg.reshape(batch, L, 1, -1)

        return torch.cat((eeg, eog, emg), dim=2)
