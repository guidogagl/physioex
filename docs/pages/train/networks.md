# Networks Module Overview

The model implemented into the PhysioEx library are:

- [Chambon2018](https://ieeexplore.ieee.org/document/8307462) model for sleep stage classification ( raw time series as input).
- [TinySleepNet](https://github.com/akaraspt/tinysleepnet) model for sleep stage classification (raw time series as input).
- [SeqSleepNet](https://arxiv.org/pdf/1809.10932.pdf) model for sleep stage classification (time-frequency images as input).


| Model          | Model Name    | Input Transform | Target Transform |
|----------------|---------------|-----------------|------------------|
| Chambon2018    | chambon2018   | raw             | get_mid_label    |
| TinySleepNet   | tinysleepnet  | raw             | None             |
| SeqSleepNet    | seqsleepnet   | xsleepnet       | None             |

The models in PhysioEx are designed to receive input sequences of 30-second sleep epochs. These sequences can be either preprocessed or raw, depending on the specific requirements of the model. 
The preprocessing status of the input sequences is indicated by the "Input Transform" attribute. This attribute must match the "preprocessing" argument of the dataset to ensure that the model receives as input the correct information.

Similarly, models can be sequence-to-sequence (default) or sequence-to-epoch. In the last case a function that selects one epoch in the sequence needs to be added to the PhysioExDataModule pipeline to match the target data and the output of the model. These functions are implemented into the `physioex.train.networks.utils.target_transform` module. 

When implementing your own SleepModule, the Input Transform and Target Transform methods must be configurated properly, the best practice is to set them into a `.yaml` file as discussed in the train module documentation page.

# Extending the SleepModule

All the models compatible with PhysioEx are Pytorch Lightning Modules which extends the `physioes.train.networks.base.SleepModule`.

By extending the module you can implement your own custom sleep staging deep learning network. When extending the module use a dictionary `module_config: dict` as the argument to the construct to allow compatibility with all the library. Second define your custom `torch.nn.Module` and use  `module_config: dict` as its constructor argument too.

!!! example
    ```python
    import torch
    from physioex.train.networks.base import SleepModule

    class CustomNet( torch.nn.Module ):
        def __init__(self, module_config: dict):

            # tipycally here you have an epoch_encoder and a sequence_encoder
            self.epoch_encoder = ...
            self.sequence_encoder = ...

            pass

        def forward(self, x : torch.Tensor):
            encoding, preds = self.encode(x)
            return preds

        def encode(self, x : torch.Tensor):
            # get your latent-space encodings
            encodings = ...

            # get your predictions out of the encodings
            preds = ...

            return econdings, preds

    class CustomModule(SleepModule):
        def __init__(self, module_config: dict):
            super(CustomNet, self).__init__(CustomNet(module_config), module_config)

    ```

The SleepModule needs to know the `n_classes` ( for sleep staging this is tipycally 5 ) and the loss to be computed during training. By default the loss function in PhysioEx ( check `physioex.train.networks.utils.loss` ) take a python `dict` in its constructor, so you should always specify in your module_config the `n_classes` value, `loss_call` and `loss_params`.

`SleepModule`
::: network.SleepModule
    handler: python
    options:
      members:
        - __init__
        - configure_optimizers
        - forward
        - encode
        - compute_loss
        - training_step
        - validation_step
        - test_step
      show_root_heading: false
      show_source: false
	  heading_level: 3