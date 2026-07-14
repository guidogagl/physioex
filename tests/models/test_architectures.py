"""Forward-shape + param-count contract tests for the classic architectures.

Random init, tiny batches, CPU. SeqSleepNet and TinySleepNet already have
end-to-end smoke coverage (tests/test_e2e_smoke.py); this file covers the
remaining public nets: Chambon2018Net, TsinalisCNN, SleepTransformer,
LSeqSleepNet.
"""
import pytest
import torch


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


# ── Chambon2018Net: (B, L, C, T) -> (B, 1, n_classes) ────────────────

def test_chambon2018_forward_shape():
    from physioex.models.chambon2018 import Chambon2018Net

    model = Chambon2018Net(n_classes=5, in_channels=2, sf=100, n_times=3000)
    x = torch.randn(2, 5, 2, 3000)  # (B, L, C, T)
    out = model(x)
    assert out.shape == (2, 1, 5)
    assert _n_params(model) > 0


def test_chambon2018_encode_shape():
    from physioex.models.chambon2018 import Chambon2018Net

    model = Chambon2018Net(n_classes=5, in_channels=1, n_times=3000)
    feats = model.encode(torch.randn(2, 3, 1, 3000))
    assert feats.shape[0] == 2 and feats.shape[1] == 3  # (B, L, D)


# ── TsinalisCNN: (B, 5, 1, 3000) -> (B, 1, n_classes) ────────────────

def test_tsinalis_forward_shape():
    from physioex.models.tsinalis import TsinalisCNN

    model = TsinalisCNN(n_classes=5, sfreq=100)
    x = torch.randn(2, 5, 1, 3000)  # 5-epoch sequence
    out = model(x)
    assert out.shape == (2, 1, 5)
    assert _n_params(model) > 0


# ── SleepTransformer: (B, L, C, T, F) -> (B, L, n_classes) ───────────

def test_sleeptransformer_forward_shape():
    from physioex.models.sleeptransformer import SleepTransformer

    # small config to keep it light; spectrogram T=29 frames, F=129 bins
    model = SleepTransformer(
        n_classes=5, in_chan=1, d_model=32, n_heads=4,
        n_epoch_layers=1, n_seq_layers=1, d_ff=64, d_clf=64,
    )
    x = torch.randn(2, 3, 1, 29, 129)  # (B, L, C, T, F)
    out = model(x)
    assert out.shape == (2, 3, 5)


# ── LSeqSleepNet: (B, L, C, T, F) -> (B, L, n_classes) ───────────────

def test_lseqsleepnet_forward_shape():
    from physioex.models.lseqsleepnet import LSeqSleepNet

    model = LSeqSleepNet(n_classes=5, in_chan=1)
    x = torch.randn(2, 4, 1, 29, 129)  # (B, L, C, T, F)
    out = model(x)
    assert out.shape[0] == 2 and out.shape[-1] == 5
