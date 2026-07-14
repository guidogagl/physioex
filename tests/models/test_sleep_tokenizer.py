"""Unit tests for physioex.models.sleep_tokenizer (pure-torch components)."""
import pytest
import torch

from physioex.models.sleep_tokenizer import (
    infer_modality,
    build_modality_ids,
    ChannelUNet,
    SAB,
    PMA,
    PrototypicalClassifier,
)
from physioex.data.modality import ModalityType


# ── Modality helpers ─────────────────────────────────────────────────

def test_infer_modality_returns_int_and_delegates():
    m = infer_modality("EEG C4-M1")
    assert isinstance(m, int)
    assert m == int(ModalityType.EEG)
    # hint precedence carries through the shared implementation
    assert infer_modality("Chin1", hint="EEG") == int(ModalityType.EEG)


def test_build_modality_ids_from_channel_order():
    batch = {
        "channel_order": ["EEG_0", "EOG_1", "EMG_2"],
        "labels": torch.zeros(4, 5),  # B=4
    }
    ids = build_modality_ids(batch)
    assert ids.shape == (4, 3)
    assert ids.dtype == torch.long
    # every row identical; columns follow the modality of each channel
    assert ids[0].tolist() == [
        int(ModalityType.EEG), int(ModalityType.EOG), int(ModalityType.EMG)
    ]


def test_build_modality_ids_unknown_channel_is_other():
    batch = {"channel_order": ["WEIRD_0"], "labels": torch.zeros(2, 1)}
    ids = build_modality_ids(batch)
    assert ids.shape == (2, 1)
    assert ids[0, 0].item() == int(ModalityType.OTHER)


# ── Set-transformer components ───────────────────────────────────────

def test_sab_preserves_shape():
    sab = SAB(d_model=16, n_heads=4, ff_dim=32)
    x = torch.randn(3, 6, 16)  # (N, C, d_model)
    assert sab(x).shape == (3, 6, 16)


def test_pma_pools_to_fixed_seeds():
    pma = PMA(d_model=16, n_heads=4, n_seeds=4)
    # output is (N, n_seeds, d_model) regardless of the number of channels C
    out5 = pma(torch.randn(3, 5, 16))
    out9 = pma(torch.randn(3, 9, 16))
    assert out5.shape == (3, 4, 16)
    assert out9.shape == (3, 4, 16)


def test_channel_unet_encodes_to_vector():
    net = ChannelUNet()
    x = torch.randn(2, 1, 3072)  # (N, 1, T), T divisible by 2^depth
    out = net(x)
    assert out.ndim == 2 and out.shape[0] == 2


# ── Prototypical classifier ──────────────────────────────────────────

def test_prototypical_classifier_forward_shape():
    clf = PrototypicalClassifier(d_model=8, n_classes=5)
    logits = clf(torch.randn(10, 8))
    assert logits.shape == (10, 5)


def test_prototypical_classifier_update_centroids():
    clf = PrototypicalClassifier(d_model=8, n_classes=3)
    assert not clf.initialized.any()
    emb = torch.randn(30, 8)
    labels = torch.randint(0, 3, (30,))
    clf.update_centroids(emb, labels)
    # centroids for the classes present become initialized + unit-norm
    assert clf.initialized.any()
    present = torch.unique(labels)
    for k in present.tolist():
        assert torch.isclose(clf.centroids[k].norm(), torch.tensor(1.0), atol=1e-4)


def test_prototypical_classifier_ignores_negative_labels():
    clf = PrototypicalClassifier(d_model=4, n_classes=2)
    emb = torch.randn(5, 4)
    labels = torch.full((5,), -1)  # all unscored
    clf.update_centroids(emb, labels)
    assert not clf.initialized.any()  # nothing updated
