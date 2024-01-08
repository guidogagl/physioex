from braindecode.models import SleepStagerChambon2018
from physioex.train.networks.base import SeqtoSeq 
import torch.optim as optim

from torch import nn 
import torch 

module_config = dict()

class SequenceEncoder( nn.Module ):
    def __init__(self, input_dim : int, n_classes : int, latent_dim : int ):
        super(SequenceEncoder, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.proj = nn.Linear( input_dim, latent_dim )
        self.norm = nn.LayerNorm( latent_dim )
        self.clf = nn.Linear( latent_dim, n_classes )

    def forward(self, x):
        x = self.encode(x)
        x = self.clf(x)
        return x

    def encode(self, x):
        batch_size, sequence_lenght, input_dim = x.size()
        x = x.reshape( batch_size, sequence_lenght * input_dim )
        x = self.drop(x)
        x = nn.ReLU()(self.proj(x))
        x = self.norm(x)
        return x
        
class Chambon2018Net( SeqtoSeq ):
    def __init__(self, module_config = module_config):
        super(Chambon2018Net, self).__init__(None, None, module_config)

        epoch_encoder = SleepStagerChambon2018(
            n_chans = module_config["n_channels"],
            sfreq = module_config["sfreq"], 
            n_outputs= module_config["n_classes"],
            n_times=module_config["n_times"],
            return_feats=True
        )
        
        sequence_encoder = SequenceEncoder( epoch_encoder.len_last_layer * module_config["seq_len"], module_config["n_classes"], module_config["latent_space_dim"] )

        super(Chambon2018Net, self).__init__(epoch_encoder, sequence_encoder, module_config)
        
    
    def compute_loss(
        self, embeddings, outputs, targets, log: str = "train", log_metrics: bool = False
    ):
        
        batch_size, n_class = outputs.size()
        outputs = outputs.reshape(batch_size, 1, n_class)
        embeddings = embeddings.reshape(batch_size, 1, -1)

        return super().compute_loss(embeddings, outputs, targets, log, log_metrics)
 
