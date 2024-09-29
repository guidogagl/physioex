import torch
import torch.optim as optim
from braindecode.models import SleepStagerChambon2018
from torch import nn

from physioex.train.networks.base import SleepModule

module_config = dict()


class Net(nn.Module):
    def __init__(self, module_config=module_config):
        """
        The Net class extends nn.Module. This class implements the core network proposed by Chambon et al in 2018.
        The network consist of an encoder of epochs, a concatenation layer and a classification layer.
        Multiple epochs are concatenated and fed to a linear classifier which predicts the sleep stage of the middle epoch of the sequence.

        Args:
            module_config (dict): A dictionary containing the module configuration. Defaults to `module_config`.

        Attributes:
            epoch_encoder (SleepStagerChambon2018): The epoch encoder.
            clf (nn.Linear): The linear classifier.
            drop (nn.Dropout): A dropout module to prevent overfitting.
        """
        super().__init__()

        print(module_config["in_channels"])
        self.epoch_encoder = SleepStagerChambon2018(
            n_chans=module_config["in_channels"],
            sfreq=module_config["sf"],
            n_outputs=module_config["n_classes"],
            n_times=module_config["n_times"],
            return_feats=True,
        )

        self.clf = nn.Linear(
            self.epoch_encoder.len_last_layer * module_config["sequence_length"],
            module_config["n_classes"],
        )

        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        """
        Implements the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the module.
        """
        x, y = self.encode(x)
        return y

    def encode(self, x: torch.Tensor):
        """
        Encodes the input x using the epoch encoder, returns both the econdings and the classification outcome.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing the encoded input tensor and the output tensor of the module.
        """
        batch_size, seqlen, nchan, nsamp = x.size()

        x = x.reshape(-1, nchan, nsamp)

        x = self.epoch_encoder(x)

        x = x.reshape(batch_size, -1)

        y = self.drop(x)
        y = self.clf(y)

        return x, y


class Chambon2018Net(SleepModule):
    def __init__(self, module_config: dict = module_config):
        """
        The Chambon2018Net class extends SleepModule. This class is a wrapper for the core Chambon2018 network to be trained inside physioex.

        Args:
            module_config (dict): A dictionary containing the module configuration. Defaults to `module_config`.

        Attributes:
            all the attributes are from `SleepModule`
        """

        module_config["n_times"] = 3000
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
        Computes the loss for the Chambon2018Net model. This is necessary because Chambon2018 is a multi-input-single-output model, while the base class is a multi-input-multi-output model ( sequence-to-sequence ).

        Args:
            embeddings (torch.Tensor): The embeddings tensors.
            outputs (torch.Tensor): The model output tensors.
            targets (torch.Tensor): The target tensors.
            log (str): The logging information. Defaults to "train".
            log_metrics (bool): Whether to log metrics. Defaults to False.

        Returns:
            torch.Tensor: The computed loss value.
        """
        batch_size, n_class = outputs.size()
        outputs = outputs.reshape(batch_size, 1, n_class)
        embeddings = embeddings.reshape(batch_size, 1, -1)

        return super().compute_loss(embeddings, outputs, targets, log, log_metrics)
