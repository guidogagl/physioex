from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
from physioex.train.networks.base import SeqtoSeq, ContrSeqtoSeq


module_config = dict()

def get_spectograms():
    return

inpunt_transforms = get_spectograms
target_transforms = None

class SeqtoSeqSleepNet( SeqtoSeq ):
    def __init__(self, module_config = module_config):
        super(SeqtoSeqSleepNet, self).__init__( None, None , module_config)

        self.nn = SeqtoSeqModule( EpochEncoder(module_config), SequenceEncoder(module_config) )

    
class ContrSeqtoSeqSleepNet( ContrSeqtoSeq ):
    def __init__(self, module_config = module_config):

        decoder_config = module_config.copy()
        decoder_config["n_classes"] = decoder_config["latent_space_dim"]
        super(ContrSeqtoSeqSleepNet, self).__init__( None, None , module_config)

        self.nn = ContrSeqtoSeqModule( EpochEncoder(module_config), SequenceEncoder(decoder_config), module_config["latent_space_dim"], module_config["n_classes"] )
    

class EpochEncoder(nn.Module):
    def __init__(self, module_config):
        super(EpochEncoder, self).__init__()
        self.F2_filtbank = LearnableFilterbank(module_config["in_chan"], module_config["F"], module_config["D"], module_config["nfft"], module_config["samplerate"], module_config["lowfreq"], module_config["highfreq"])
        self.F2_birnn = nn.LSTM( input_size = module_config["D"] * module_config["in_chan"], hidden_size = module_config["seqnhidden1"], num_layers = module_config["seqnlayer1"], batch_first = True, bidirectional = True)
        self.F2_attention = AttentionLayer( 2*module_config["seqnhidden1"], module_config["attentionsize1"])
        self.attentionsize1 = module_config["attentionsize1"]
        self.T = module_config["T"]
        self.D = module_config["D"]
        self.F = module_config["F"]
        self.in_chans = module_config["in_chan"]
        return
    
    def forward(self, x):
        batch_size, in_chans, T, F = x.size()

        assert in_chans == self.in_chans, "channels dimension mismatch, provided input size: " + str(x.size())
        assert T == self.T, "time dimension mismatch, provided input size: " + str(x.size())
        assert F == self.F, "frequency dimension mismatch, provided input size: " + str(x.size())

        x = self.F2_filtbank(x)

        x = torch.reshape( x, (batch_size, self.T, self.D*self.in_chans) )
        x, _ = self.F2_birnn( x )
        x = self.F2_attention( x )
        
        return x
    
class SequenceEncoder(nn.Module):
    def __init__(self, module_config):
        super(SequenceEncoder, self).__init__()
        
        self.LSTM = nn.LSTM( input_size = 2*module_config["seqnhidden1"], 
                            hidden_size = module_config["seqnhidden2"], 
                            num_layers = module_config["seqnlayer2"], 
                            batch_first = True, 
                            bidirectional = True
                            )

        self.lin = nn.Linear( module_config["seqnhidden2"] * 2, module_config["n_classes"])

    def forward(self, x):
        x, _ = self.LSTM( x )
        x = self.lin(x)        
        return x

    def encode(self, x):
        x, _ = self.LSTM( x )
        return x

class SeqtoSeqModule( nn.Module ):
    def __init__(self, encoder, decoder):
        super(SeqtoSeqModule, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder

    def forward(self, x):
        batch_size, sequence_lenght, n_chans, T, F = x.size()        

        x = x.reshape(batch_size * sequence_lenght, n_chans, T, F)
        x = self.encoder(x)

        x = x.reshape( batch_size, sequence_lenght, -1)
        
        return self.decoder(x)

    def encode(self, x):
        batch_size, sequence_lenght, modalities, input_dim = x.size()        

        x = x.reshape(batch_size * sequence_lenght, modalities, input_dim)
        x = self.encoder(x)

        x = x.reshape( batch_size, sequence_lenght, -1)
        
        return self.decoder.encode(x), self.decoder(x)

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
    
class AttentionLayer(nn.Module):
    def __init__(self,  hidden_size,
                        attention_size : int = 32, 
                        time_major : bool = False, 
                        ):
        super().__init__()

        W_omega = torch.zeros((hidden_size, attention_size), dtype = torch.float32 )
        b_omega = torch.zeros((attention_size), dtype = torch.float32 )
        u_omega = torch.zeros((attention_size), dtype = torch.float32 )

        self.W_omega = nn.Parameter( W_omega )
        self.b_omega = nn.Parameter( b_omega )    
        self.u_omega = nn.Parameter( u_omega ) 

        nn.init.normal_(self.W_omega, std = 0.1) 
        nn.init.normal_(self.b_omega, std = 0.1) 
        nn.init.normal_(self.u_omega, std = 0.1) 

    def forward(self, x, r_alphas = False):
        batch_size, sequence_length, hidden_size = x.size()

        v = torch.tanh(torch.matmul(torch.reshape(x, [batch_size * sequence_length, hidden_size]), self.W_omega) + torch.reshape(self.b_omega, [1, -1]))
        vu = torch.matmul(v, torch.reshape(self.u_omega, [-1, 1]))
        exps = torch.reshape(torch.exp(vu), [-1, sequence_length])
        alphas = exps / torch.reshape(torch.sum(exps, 1), [-1, 1])

        output = torch.sum(x * torch.reshape(alphas, [batch_size, sequence_length, 1]), 1)
        if r_alphas:
            return output, alphas
        return output 

class LearnableFilterbank(nn.Module):
    def __init__(self,  in_chan : int = 2,
                        F : int = 129, 
                        nfilt : int = 32, 
                        nfft : int = 256, 
                        samplerate : int = 100, 
                        lowfreq : int = 0, 
                        highfreq : int = 50
                        ):
        super().__init__()
        self.F, self.D = F, nfilt
        
        S = torch.zeros((in_chan, F, nfilt), dtype = torch.float32 )

        for i in range(in_chan):
            S[i] = self.lin_tri_filter_shape(nfilt, nfft, samplerate, lowfreq, highfreq)
        
        W = torch.zeros((in_chan, F, nfilt), dtype = torch.float32)
        
        self.W = nn.Parameter( W, requires_grad = True )
        self.S = nn.Parameter( S, requires_grad = False) 

        nn.init.normal_(self.W) 
        
    def forward(self, x):
        Wfb = torch.mul( torch.sigmoid( self.W ), self.S )

        return torch.matmul(x, Wfb) 

    def lin_tri_filter_shape(self, nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        highfreq = highfreq or samplerate/2
        assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

        hzpoints = torch.linspace(lowfreq,highfreq,nfilt+2)
        bin = torch.floor((nfft+1)*hzpoints/samplerate)

        fbank = torch.zeros([nfilt,nfft//2+1])
        for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        fbank = torch.transpose(fbank, 0 , 1)
        return fbank.float()
