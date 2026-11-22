"""Tests for the SleepPPG-Net port (tests/models/sleepppgnet.py).

Covers the state-dict contract against the replica fixture, the forward
behaviour (input formats, padding/truncation, class permutation), and —
behind ``@pytest.mark.hf`` — golden parity with the authors' replica code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

FIXTURES = Path(__file__).parent / "fixtures" / "cardioresp"

# Replica / physioex class-order permutation (see the model docstring).
#   replica  :  0=Wake  1=REM  2=Light  3=Deep
#   physioex :  0=W     1=Light  2=Deep  3=REM
# physioex[:, j] = original[:, ORIGINAL_TO_PHYSIOEX[j]].
ORIGINAL_TO_PHYSIOEX = [0, 2, 3, 1]


def _state_dict_shapes(model):
    return {k: list(v.shape) for k, v in model.state_dict().items()}


# ── State-dict contract ────────────────────────────────────────────────────

@pytest.mark.unit
def test_state_dict_matches_fixture():
    from physioex.models.sleepppgnet import SleepPPGNet

    with open(FIXTURES / "state_dict_shapes.json") as f:
        expected = json.load(f)["sleepppgnet-replica"]

    model = SleepPPGNet()
    actual = _state_dict_shapes(model)

    assert set(actual) == set(expected), (
        f"key mismatch: only-in-model={set(actual) - set(expected)}, "
        f"only-in-fixture={set(expected) - set(actual)}"
    )
    mismatches = {
        k: (actual[k], expected[k])
        for k in actual
        if tuple(actual[k]) != tuple(expected[k])
    }
    assert not mismatches, f"shape mismatches: {mismatches}"


# ── Forward behaviour ──────────────────────────────────────────────────────

def _small_model(**kwargs):
    from physioex.models.sleepppgnet import SleepPPGNet

    # 10 epochs x 1024 samples: same architecture, shorter input.
    return SleepPPGNet(n_epochs=10, samples_per_epoch=1024, **kwargs).eval()


@pytest.mark.unit
def test_forward_all_formats_and_permutation():
    from physioex.models.sleepppgnet import SleepPPGNet

    torch.manual_seed(0)
    model = _small_model(class_order="physioex").eval()
    S, L = model.samples_per_epoch, model.n_epochs

    # A second model in the replica's original class order, with identical
    # weights, to check the class permutation.
    x_orig = _small_model(class_order="original").eval()
    x_orig.load_state_dict(model.state_dict())

    # One flat input reshaped into the three accepted layouts: the three
    # formats must produce identical outputs.
    base = torch.randn(2, L, S)
    x1 = base[:, :, None, :]  # (B, L, 1, S) physioex per-epoch
    x2 = base  # (B, L, S)
    x3 = base.reshape(2, 1, L * S)  # (B, 1, T) original format

    with torch.no_grad():
        out1_phys = model(x1)
        out2_phys = model(x2)
        out3_phys = model(x3)
        out1_orig = x_orig(x1)

    for out in (out1_phys, out2_phys, out3_phys, out1_orig):
        assert out.shape == (2, L, 4)

    # physioex output is the class permutation of the original output.
    assert torch.allclose(
        out1_phys, out1_orig[:, :, ORIGINAL_TO_PHYSIOEX], atol=1e-5, rtol=1e-5
    )
    # The three input formats are just reshapes of the same flat input.
    assert torch.allclose(out1_phys, out2_phys, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out1_phys, out3_phys, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_forward_padding_and_truncation():
    from physioex.models.sleepppgnet import SleepPPGNet

    torch.manual_seed(1)
    model = _small_model()  # n_epochs=10, class_order="physioex"
    S, n_epochs = model.samples_per_epoch, model.n_epochs
    with torch.no_grad():
        full = model(torch.randn(2, n_epochs, 1, S))

        # L < n_epochs: zero-pad at the end, return the first L epochs.
        short = model(torch.randn(2, 3, 1, S))
        assert short.shape == (2, 3, 4)

        # L > n_epochs: truncate to n_epochs, return n_epochs epochs.
        long = model(torch.randn(2, 13, 1, S))
        assert long.shape == (2, n_epochs, 4)

        # Feeding a 13-epoch input is equivalent to feeding its first
        # n_epochs epochs (truncation happens before the network).
        long_full = torch.randn(2, 13, 1, S)
        assert torch.allclose(
            model(long_full), model(long_full[:, :n_epochs]), atol=1e-5, rtol=1e-5
        )


# ── Golden parity (needs the authors' weights from HF) ─────────────────────

@pytest.mark.hf
def test_golden_parity():
    from huggingface_hub import hf_hub_download

    from physioex.models.sleepppgnet import SleepPPGNet

    ckpt = hf_hub_download(
        "sleep-is-all-you-need/SPPPPG-Net",
        "modelStatebe14cbce-76c1a81b-dd25c9a2-SleepPPGNet-857c0582-b900d29c-6864dcd6-39dc88ff-0--current",
    )
    model = SleepPPGNet.from_original(ckpt, class_order="original")
    assert model.training is False  # eval()

    # Input recipe from tests/models/fixtures/cardioresp/README.md.
    x = np.random.default_rng(200).standard_normal(1228800).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(x).reshape(1, 1, 1228800))

    with np.load(FIXTURES / "sleepppgnet-replica.npz") as data:
        ref = data["logits"]  # (1, 4, 1200), original class order

    # out is (1, 1200, 4) -> transpose to (1, 4, 1200).
    np.testing.assert_allclose(out.numpy().transpose(0, 2, 1), ref, atol=1e-4, rtol=1e-4)
