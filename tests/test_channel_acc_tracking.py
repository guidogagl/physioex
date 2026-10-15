"""Opt-in tracking of the per-channel accuracy during training (``ChannelMixer.track_train_acc``).

The accuracy buffer drives the accuracy-weighted modality dropout. By default it is only refreshed from validation
batches (released behaviour); with ``track_train_acc`` it also follows the training batches, smoothed with an EMA.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from physioex.models.protosleepnet import ChannelMixer, ProtoSleepNet, ProtoSleepNetTrainer  # noqa: E402

pytestmark = pytest.mark.unit


def _mixer(**kw):
    torch.manual_seed(0)
    cm = ChannelMixer(n_channels=3, n_classes=5, d_model=16, n_heads=2, d_ff=32, n_layers=1)
    for k, v in kw.items():
        setattr(cm, k, v)
    return cm


def _run(cm, acc):
    cm.update_channel_acc(acc)
    cm(torch.randn(4, 3, 16), torch.zeros(1, 16))


def test_defaults_keep_released_behaviour():
    cm = _mixer()
    assert cm.track_train_acc is False
    assert cm.acc_momentum == 0.0
    assert torch.allclose(cm.channels_acc, torch.full((3,), 0.2))


def test_buffer_is_replaced_when_momentum_is_zero():
    cm = _mixer().train()
    _run(cm, [0.9, 0.5, 0.1])
    assert torch.allclose(cm.channels_acc, torch.tensor([0.9, 0.5, 0.1]))


def test_ema_follows_the_new_accuracy():
    cm = _mixer(acc_momentum=0.9).train()
    old = cm.channels_acc.clone()
    new = torch.tensor([0.9, 0.5, 0.1])
    _run(cm, new.tolist())
    assert torch.allclose(cm.channels_acc, 0.9 * old + 0.1 * new, atol=1e-6)
    _run(cm, new.tolist())
    again = 0.9 * (0.9 * old + 0.1 * new) + 0.1 * new
    assert torch.allclose(cm.channels_acc, again, atol=1e-6)


def test_eval_mode_never_updates_the_buffer():
    cm = _mixer(acc_momentum=0.5).eval()
    before = cm.channels_acc.clone()
    _run(cm, [0.9, 0.5, 0.1])
    assert torch.equal(cm.channels_acc, before)


def _tiny_batch(L=2):
    x = torch.randn(1, L, 3, 29, 129)
    y = torch.zeros(1, L, dtype=torch.long)
    return x, y


@pytest.mark.parametrize("track", [False, True])
def test_trainer_step_records_accuracy_in_training_only_when_enabled(track):
    torch.manual_seed(0)
    model = ProtoSleepNet.from_seq_sleep_net(
        n_channels=3, use_channel_mixer=True, cm_n_heads=2, cm_d_ff=32, cm_n_layers=1
    )
    model.channel_mixer.track_train_acc = track
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    ProtoSleepNetTrainer._step(model, _tiny_batch(), loss_fn, torch.device("cpu"))
    assert len(model.channel_mixer.channels_acc_list) == (1 if track else 0)


def test_trainer_step_in_eval_mode_always_records_accuracy():
    model = ProtoSleepNet.from_seq_sleep_net(
        n_channels=3, use_channel_mixer=True, cm_n_heads=2, cm_d_ff=32, cm_n_layers=1
    ).eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    with torch.no_grad():
        ProtoSleepNetTrainer._step(model, _tiny_batch(), loss_fn, torch.device("cpu"))
    assert len(model.channel_mixer.channels_acc_list) == 1
