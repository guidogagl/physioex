"""Tests for the Jones et al. (2024) ECG-only sleep-staging network.

Cheap (default CI): state-dict key/shape parity with the authors' released
checkpoints (fixture ``state_dict_shapes.json``), forward-shape + padding-mask
semantics, and the ``build_additional`` feature builder.

Gated (@hf): golden parity against outputs produced by the authors' own code
(``fixtures/cardioresp/jones-ecg.npz``); each of the 7 checkpoints is
downloaded once from ``adammj/ecg-sleep-staging`` on GitHub.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from physioex.models.jones_ecg import VARIANTS, JonesECGNet

FIXTURES = Path(__file__).parent / "fixtures" / "cardioresp"
SAMPLES = 7680  # 256 Hz x 30 s

CHECKPOINT_FILES = {
    "primary": "epoch_0647_kappa_0.724.pt",
    "realtime": "epoch_0637_kappa_0.703.pt",
    "no-demographics": "epoch_0552_kappa_0.718.pt",
    "no-time": "epoch_0490_kappa_0.724.pt",
    "only-ecg": "epoch_0369_kappa_0.721.pt",
    "cyclic": "epoch_0548_kappa_0.725.pt",
    "cyclic-realtime": "epoch_0472_kappa_0.703.pt",
}

ALL_VARIANTS = list(VARIANTS)


def _variant_keys(variant):
    return [
        "1_paper_models/primary_model",
        "1_paper_models/real-time_model",
        "2_post-paper_models/primary_model_no_demographics",
        "2_post-paper_models/primary_model_no_time",
        "2_post-paper_models/primary_model_only_ecg",
        "3_cyclic-time_models/primary_model",
        "3_cyclic-time_models/real-time_model",
    ][["primary", "realtime", "no-demographics", "no-time", "only-ecg", "cyclic", "cyclic-realtime"].index(variant)]


# ── State-dict parity with the authors' checkpoints ──────────────────

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_state_dict_matches_fixture(variant):
    """Keys and shapes must match the original checkpoint exactly (strict load)."""
    with open(FIXTURES / "state_dict_shapes.json") as f:
        expected = json.load(f)["jones/" + _variant_keys(variant)]

    model = JonesECGNet(variant=variant)
    actual = {k: list(v.shape) for k, v in model.state_dict().items()}
    assert actual == expected, (
        f"missing: {set(expected) - set(actual)}\n"
        f"extra: {set(actual) - set(expected)}\n"
        f"shape mismatch: {{{', '.join(f'{k!r}: got {actual[k]} want {expected[k]}' for k in actual if k in expected and actual[k] != expected[k])}}}"
    )


# ── Forward contract ─────────────────────────────────────────────────

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_forward_shape_and_output_range(variant):
    """Random init: (B, L, 1, 7680) -> (B, L, 5) log-probabilities summing to ~0."""
    torch.manual_seed(0)
    model = JonesECGNet(variant=variant).eval()
    x = torch.randn(2, 3, 1, SAMPLES)
    # pass explicit additional so the test works for every variant
    additional = torch.randn(2, 3, 4)
    with torch.no_grad():
        out = model(x, additional=additional)
    assert out.shape == (2, 3, 5)
    # log-probabilities: finite, every entry <= 0
    assert torch.isfinite(out).all()
    assert (out <= 0).all()


def test_forward_accepts_3d_ecg_and_explicit_additional():
    model = JonesECGNet("primary").eval()
    x = torch.randn(1, 4, SAMPLES)
    additional = torch.randn(1, 4, 4)
    with torch.no_grad():
        out = model(x, additional=additional)
    assert out.shape == (1, 4, 5)


def test_padding_mask_zeroing_causal():
    """For the causal (real-time) variant, padded epochs do not change the outputs of real epochs."""
    torch.manual_seed(1)
    model = JonesECGNet("realtime").eval()
    n = 8
    x = torch.randn(1, n, 1, SAMPLES)
    additional = torch.randn(1, n, 4)
    with torch.no_grad():
        out_full = model(x, additional=additional)
        # same head, shorter sequence (no padding at all)
        out_head = model(x[:, : n // 2], additional=additional[:, : n // 2])
        # same sequence, second half marked as padding
        mask = torch.ones(1, n)
        mask[:, n // 2 :] = 0
        out_masked = model(x, additional=additional, padding_mask=mask)
    # causal convolutions only look at the past, and masked epochs carry the
    # zeroed features -> the real (first) epochs give the same output in all three runs
    assert torch.allclose(out_full[:, : n // 2], out_head, atol=1e-5)
    assert torch.allclose(out_masked[:, : n // 2], out_head, atol=1e-5)


def test_forward_rejects_bad_shapes():
    model = JonesECGNet("primary")
    with pytest.raises(ValueError, match="must have shape"):
        model(torch.randn(1, 2, 3, SAMPLES))
    with pytest.raises(ValueError, match="unknown variant"):
        JonesECGNet("bogus")


def test_forward_builds_additional_or_raises():
    """additional=None: noise-filled build works for only-ecg, raises for primary."""
    torch.manual_seed(2)
    x = torch.randn(1, 4, 1, SAMPLES)

    JonesECGNet("only-ecg").eval()(x)  # needs nothing -> OK
    for variant in ("primary", "realtime", "cyclic", "cyclic-realtime"):
        with pytest.raises(ValueError, match="age, sex, start_time"):
            JonesECGNet(variant).eval()(x)
    with pytest.raises(ValueError, match="start_time"):
        JonesECGNet("no-demographics").eval()(x)
    with pytest.raises(ValueError, match="age, sex"):
        JonesECGNet("no-time").eval()(x)


# ── midnight_offset ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "start,expected",
    [
        (datetime.time(0, 0), 0.0),
        (datetime.time(1, 30), 1.5 / 24),
        (datetime.time(23, 0), -1 / 24),
        (datetime.time(18, 0), -6 / 24),
        (datetime.datetime(2024, 5, 1, 1, 30), 1.5 / 24),
    ],
)
def test_midnight_offset(start, expected):
    assert JonesECGNet.midnight_offset(start) == pytest.approx(expected)


# ── build_additional ─────────────────────────────────────────────────

def test_build_additional_primary():
    n = 10
    t = JonesECGNet.build_additional(n, "primary", age=40, sex=1, midnight_offset=-0.5)
    assert t.shape == (n, 4) and t.dtype == torch.float32
    assert torch.allclose(t[:, 0], torch.ones(n))
    assert torch.allclose(t[:, 1], torch.full((n,), 0.4))
    night = torch.linspace(-1, 1, n)
    assert torch.allclose(t[:, 2], night, atol=1e-6)
    assert torch.allclose(t[:, 3], torch.arange(n, dtype=torch.float32) / 2880 - 0.5, atol=1e-6)


def test_build_additional_cyclic():
    n = 2880
    t = JonesECGNet.build_additional(n, "cyclic", age=25, sex=0, midnight_offset=0.0)
    assert torch.allclose(t[:, 0], torch.zeros(n))
    assert torch.allclose(t[:, 1], torch.full((n,), 0.25))
    cos_t = torch.cos(2 * np.pi * torch.arange(n, dtype=torch.float32) / 2880)
    sin_t = torch.sin(2 * np.pi * torch.arange(n, dtype=torch.float32) / 2880)
    assert torch.allclose(t[:, 2], cos_t, atol=1e-5)
    assert torch.allclose(t[:, 3], sin_t, atol=1e-5)
    # midnight_offset 0.25 -> phase shift by 1/4 turn
    t2 = JonesECGNet.build_additional(4, "cyclic", age=25, sex=0, midnight_offset=0.25)
    assert torch.allclose(t2[0, 2], torch.tensor(float(np.cos(2 * np.pi * 0.25))), atol=1e-5)


def test_build_additional_noise_columns():
    """Columns the variant was not trained with carry noise with the expected draw structure."""
    n = 12
    g = torch.Generator().manual_seed(7)
    t = JonesECGNet.build_additional(n, "no-demographics", midnight_offset=0.0, generator=g)
    # demographics: ONE draw of size 2 repeated over all epochs
    assert torch.equal(t[:, 0], t[0, 0].expand(n))
    assert torch.equal(t[:, 1], t[0, 1].expand(n))
    # time columns are real (not noise)
    assert torch.allclose(t[0, 2], torch.tensor(-1.0))
    assert torch.allclose(t[-1, 2], torch.tensor(1.0), atol=1e-6)

    g = torch.Generator().manual_seed(8)
    t = JonesECGNet.build_additional(n, "no-time", age=40, sex=0, generator=g)
    assert torch.allclose(t[:, 0], torch.zeros(n))
    assert torch.allclose(t[:, 1], torch.full((n,), 0.4))
    # noise per epoch: consecutive draws from the same generator
    assert not torch.allclose(t[0, 2], t[1, 2])

    g = torch.Generator().manual_seed(9)
    t = JonesECGNet.build_additional(n, "only-ecg", generator=g)
    # demographics = ONE draw of size 2 repeated over epochs -> columns 0,1 constant
    assert torch.equal(t[:, 0], t[0, 0].expand(n))
    assert torch.equal(t[:, 1], t[0, 1].expand(n))
    # time columns: noise per epoch -> they vary across epochs
    assert t[:, 2:].std(dim=0).min() > 0


def test_build_additional_missing_metadata_raises():
    with pytest.raises(ValueError, match="age, sex, start_time"):
        JonesECGNet.build_additional(4, "primary")
    with pytest.raises(ValueError, match="start_time"):
        JonesECGNet.build_additional(4, "no-demographics", age=40, sex=1)
    # no-time needs age+sex (not start_time)
    with pytest.raises(ValueError, match="age, sex"):
        JonesECGNet.build_additional(4, "no-time")
    # with age+sex provided it should build fine
    t = JonesECGNet.build_additional(4, "no-time", age=40, sex=1)
    assert t.shape == (4, 4)
    with pytest.raises(ValueError, match="unknown variant"):
        JonesECGNet.build_additional(4, "nope", age=40, sex=1, midnight_offset=0)


# ── Gated golden parity (authors' checkpoints + reference outputs) ──

@pytest.mark.hf
@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_golden_parity(variant, tmp_path):
    """exp(forward output) must match the authors' own code on the same inputs."""
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/adammj/ecg-sleep-staging/main/models/"
        f"{VARIANTS[variant]['folder']}/{CHECKPOINT_FILES[variant]}"
    )
    path = tmp_path / CHECKPOINT_FILES[variant]
    urllib.request.urlretrieve(url, path)
    model = JonesECGNet.from_original(path, variant)
    assert not model.training

    # Input recipe from fixtures/cardioresp/README.md (seeds 300/301)
    ecg = np.clip(np.random.default_rng(300).standard_normal(60 * SAMPLES).astype(np.float32) * 0.2, -1, 1).reshape(1, 60, 1, SAMPLES)
    additional = np.random.default_rng(301).random((1, 60, 4)).astype(np.float32)
    padding = np.ones((1, 60, 1), dtype=np.float32)

    with torch.no_grad():
        logp = model(torch.tensor(ecg), additional=torch.tensor(additional), padding_mask=torch.tensor(padding))
    probs = torch.exp(logp).numpy()[0]

    golden = np.load(FIXTURES / "jones-ecg.npz")
    ref = golden[_variant_keys(variant)][0]
    np.testing.assert_allclose(probs, ref, atol=1e-5)
