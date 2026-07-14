# `Chambon2018` documentation

This page details the implementation of the `chambon2018` model published [here](https://ieeexplore.ieee.org/document/8307462).

Train it with the CLI:

```bash
train --model physioex.models.chambon2018:Chambon2018Net \
      --dataset hmc --channels EEG EOG EMG --pipelines time_domain
```

::: physioex.models.chambon2018.Chambon2018Net
    handler: python
    options:
      members:
        - __init__
        - encode
        - forward
      show_root_heading: true
      show_source: true
