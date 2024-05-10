# `Chambon2018` documentation

This page details the implementation of the `chambon2018` model published [here](https://ieeexplore.ieee.org/document/8307462).

To train the model one could use the `train -experiment chambon2018` command.

::: physioex.train.networks.chambon2018.Chambon2018Net
    handler: python
    options:
      members:
        - __init__
        - compute_loss
      show_root_heading: true
      show_source: true

::: physioex.train.networks.chambon2018.SequenceEncoder
    handler: python
    options:
      members:
        - __init__
        - forward
        - encode
      show_root_heading: true
      show_source: true