from braindecode.models import SleepStagerChambon2018
from physioex.train.networks.base import SeqtoSeq, ContrSeqtoSeq, ContrSeqtoSeqModule
import torch.optim as optim

from torch import nn 
import torch 

module_config = {
    "n_classes": 5,
    "n_channels": 1,
    "sfreq": 100,
    "n_times": 3000,
    "seq_len": 3,
    "learning_rate": 1e-4,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "latent_space_dim": 32
}





class CustModule( nn.Module ):
    def __init__(self, encoder, decoder):
        super(CustModule, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder

    def forward(self, x):
        batch_size, sequence_lenght, modalities, input_dim = x.size()        

        x = x.reshape(batch_size * sequence_lenght, modalities, input_dim)
        x = self.encoder(x)

        x = x.reshape( batch_size, sequence_lenght, -1)
        x = x.reshape( batch_size, -1)
        
        return self.decoder(x)

class Chambon2018Net( SeqtoSeq ):
    def __init__(self, module_config = module_config):
        super(Chambon2018Net, self).__init__(None, None, module_config)

        encoder = SleepStagerChambon2018(
            n_channels = module_config["n_channels"],
            sfreq = module_config["sfreq"], 
            n_outputs= module_config["n_classes"],
            n_times=module_config["n_times"],
            return_feats=True
        )
        
        decoder = nn.Sequential(  # apply linear layer on concatenated feature vectors
            nn.Dropout(0.5),
            nn.Linear( encoder.len_last_layer * module_config["seq_len"], module_config["n_classes"] )
        )

        self.nn = CustModule( encoder=encoder, decoder=decoder)
    
    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=module_config["learning_rate"],
            betas=(
                module_config["adam_beta_1"],
                module_config["adam_beta_2"],
            ),
            eps=module_config["adam_epsilon"],
        )
        return self.opt
    
    def compute_loss(
        self, outputs, targets, log: str = "train", log_metrics: bool = False
    ):
        
        batch_size, n_class = outputs.size()
        outputs = outputs.reshape(batch_size, 1, n_class)

        return super().compute_loss(outputs, targets, log, log_metrics)

class ContrCustModule( CustModule ):
    def __init__(self, encoder, decoder, latent_space_dim, n_classes):
        super(ContrCustModule, self).__init__(encoder, decoder)
        self.ls_norm = nn.LayerNorm( latent_space_dim )
        self.clf = nn.Linear(latent_space_dim, n_classes)

    def forward(self, x):
        embeddings = super().forward(x)
        embeddings = nn.ReLU()( embeddings ) 
        embeddings = self.ls_norm( embeddings )

        outputs = self.clf( embeddings )

        return embeddings, outputs

class ContrChambon2018Net( ContrSeqtoSeq ):
    def __init__(self, module_config = module_config):
        encoder = SleepStagerChambon2018(
            n_channels = module_config["n_channels"],
            sfreq = module_config["sfreq"], 
            n_outputs= module_config["n_classes"],
            n_times=module_config["n_times"],
            return_feats=True
        )
        
        dec = nn.Sequential(  # apply linear layer on concatenated feature vectors
            nn.Dropout(0.5),
            nn.Linear( encoder.len_last_layer * module_config["seq_len"], module_config["latent_space_dim"] )
        )

        super(ContrChambon2018Net, self).__init__(None, None, module_config)

        self.nn = ContrCustModule(encoder, dec, module_config["latent_space_dim"], module_config["n_classes"])

    def compute_loss(
        self, outputs, targets, log: str = "train", log_metrics: bool = False
    ):
        
        projections, outputs = outputs

        outputs = outputs.unsqueeze(dim = 1)
        projections = projections.unsqueeze( dim = 1)

        return super().compute_loss( (projections, outputs), targets, log, log_metrics) 
    
    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=module_config["learning_rate"],
            betas=(
                module_config["adam_beta_1"],
                module_config["adam_beta_2"],
            ),
            eps=module_config["adam_epsilon"],
        )
        return self.opt 

