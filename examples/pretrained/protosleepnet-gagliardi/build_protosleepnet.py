"""Factory module for ProtoSleepNet model construction.

Used by extraction/testing scripts via --build_module build_protosleepnet.
"""
from physioex.models.protosleepnet import ProtoSleepNet


def build_model(backbone="seq", n_channels=3):
    """Build ProtoSleepNet with default mixer kwargs."""
    mixer_kwargs = dict(
        use_channel_mixer=True,
        cdropout=0.5,
        cm_n_heads=4,
        cm_d_ff=256,
        cm_n_layers=1,
    )
    if backbone == "seq":
        return ProtoSleepNet.from_seq_sleep_net(n_channels=n_channels, **mixer_kwargs)
    elif backbone == "st":
        return ProtoSleepNet.from_sleep_transformer(n_channels=n_channels, **mixer_kwargs)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
