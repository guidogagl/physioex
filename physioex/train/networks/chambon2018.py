import torch
import torch.optim as optim
from braindecode.models import SleepStagerChambon2018
from torch import nn

from physioex.train.networks.base import SleepModule

module_config = dict()


class Net(nn.Module):
    def __init__(self, module_config = module_config):
        super().__init__()

        
        self.epoch_encoder = SleepStagerChambon2018(
            n_chans=module_config["in_channels"],
            sfreq=module_config["sfreq"],
            n_outputs=module_config["n_classes"],
            n_times=module_config["n_times"],
            return_feats=True,
        )
        
        self.clf = nn.Linear( self.epoch_encoder.len_last_layer * module_config["seq_len"], 5)
        
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x):
        x, y = self.encode(x)
        return y

    def encode(self, x):
        batch_size, seqlen, nchan, nsamp = x.size()
        
        x = self.epoch_encoder(x)
        x = x.reshape( batch_size, -1 )
        
        y = self.drop(x)
        y = self.clf(y)
        
        return x, y

class Chambon2018Net(SleepModule):
    def __init__(self, module_config=module_config):
        super(Chambon2018Net, self).__init__(Net(module_config), module_config)


    def compute_loss(
        self,
        embeddings,
        outputs,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):
        """
        Computes the loss for the Chambon2018Net model.

        Args:
            embeddings: The embeddings.
            outputs: The model outputs.
            targets: The target values.
            log (str): The logging information.
            log_metrics (bool): Whether to log metrics.

        Returns:
            torch.Tensor: The computed loss value.
        """
        batch_size, n_class = outputs.size()
        outputs = outputs.reshape(batch_size, 1, n_class)
        embeddings = embeddings.reshape(batch_size, 1, -1)

        return super().compute_loss(embeddings, outputs, targets, log, log_metrics)
