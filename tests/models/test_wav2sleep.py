"""Wav2Sleep (Carter & Tarassenko 2024) architecture and golden-output tests.

Golden outputs and state-dict shapes were produced once with the authors' own
code (see tests/models/fixtures/cardioresp/README.md). The ``@hf`` tests need
the authors' checkpoints from the HuggingFace Hub.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from physioex.models.wav2sleep import SAMPLES_PER_EPOCH, Wav2Sleep

FIXTURES = Path(__file__).parent / "fixtures" / "cardioresp"
CONFIGS = {
    "wav2sleep": FIXTURES / "wav2sleep.config.yaml",
    "wav2sleep-eog": FIXTURES / "wav2sleep-eog.config.yaml",
}
N_EPOCHS = 60


# ── (1) Build from the authors' config; state dict matches the fixture ──

@pytest.mark.parametrize("name", list(CONFIGS))
def test_state_dict_matches_fixture(name):
    """state_dict keys and shapes exactly equal the fixture JSON for both configs."""
    import yaml

    from physioex.models.wav2sleep import (
        MultiModalAttentionEmbedder,
        SequenceCNN,
        SignalEncoders,
    )

    cfg = yaml.safe_load(CONFIGS[name].read_text())
    for sub in ("signal_encoders", "epoch_mixer", "sequence_mixer"):
        cfg[sub].pop("_target_")
    cfg.pop("_target_")
    model = Wav2Sleep(
        signal_encoders=SignalEncoders(**cfg["signal_encoders"]),
        epoch_mixer=MultiModalAttentionEmbedder(**cfg["epoch_mixer"]),
        sequence_mixer=SequenceCNN(**cfg["sequence_mixer"]),
        num_classes=cfg["num_classes"],
    )
    expected = json.loads((FIXTURES / "state_dict_shapes.json").read_text())[name]
    actual = {k: list(v.shape) for k, v in model.state_dict().items()}
    assert actual == expected


# ── (2) Forward shapes; both input formats give identical outputs ──────

def _rand_inputs(model, batch: int = 2, n_epochs: int = 4, seed: int = 0):
    """Random signals for every signal the model knows, in both formats.

    Batch elements 1+ are made constant per signal (different offset per
    signal): the released checkpoints use instance norm (eps=1e-2) and its
    running stats are trained on real signals, so near-zero-variance batch
    elements would drift from the authors' behaviour.
    """
    x_flat: dict[str, torch.Tensor] = {}
    x_phys: dict[str, torch.Tensor] = {}
    for j, sig in enumerate(model.valid_signals):
        spe = SAMPLES_PER_EPOCH[sig]
        x0 = torch.randn(batch, n_epochs * spe, generator=torch.Generator().manual_seed(seed))
        if batch > 1:
            x0[1:] = float(j + 1)
        x_flat[sig] = x0
        x_phys[sig] = x0.reshape(batch, n_epochs, spe)
    return x_flat, x_phys


@pytest.mark.parametrize("name", list(CONFIGS))
def test_forward_shapes_and_format_equivalence(name):
    import yaml

    from physioex.models.wav2sleep import (
        MultiModalAttentionEmbedder,
        SequenceCNN,
        SignalEncoders,
    )

    cfg = yaml.safe_load(CONFIGS[name].read_text())
    for sub in ("signal_encoders", "epoch_mixer", "sequence_mixer"):
        cfg[sub].pop("_target_")
    cfg.pop("_target_")
    model = Wav2Sleep(
        signal_encoders=SignalEncoders(**cfg["signal_encoders"]),
        epoch_mixer=MultiModalAttentionEmbedder(**cfg["epoch_mixer"]),
        sequence_mixer=SequenceCNN(**cfg["sequence_mixer"]),
        num_classes=cfg["num_classes"],
    ).eval()
    x_flat, x_phys = _rand_inputs(model)
    with torch.no_grad():
        y_flat = model(x_flat)
        y_phys = model(x_phys)
    assert y_flat.shape == (2, 4, cfg["num_classes"])
    assert y_phys.shape == (2, 4, cfg["num_classes"])
    assert torch.equal(y_flat, y_phys)
    # predict() returns (B, S) class indices.
    assert model.predict(x_flat).shape == (2, 4)


# ── (3) Absent-signal handling: missing key == all -inf ────────────────

@pytest.mark.parametrize("absent", ["ABD", "THX"])
def test_absent_signal_missing_equals_all_neg_inf(absent):
    import yaml

    from physioex.models.wav2sleep import (
        MultiModalAttentionEmbedder,
        SequenceCNN,
        SignalEncoders,
    )

    cfg = yaml.safe_load(CONFIGS["wav2sleep"].read_text())
    for sub in ("signal_encoders", "epoch_mixer", "sequence_mixer"):
        cfg[sub].pop("_target_")
    cfg.pop("_target_")
    model = Wav2Sleep(
        signal_encoders=SignalEncoders(**cfg["signal_encoders"]),
        epoch_mixer=MultiModalAttentionEmbedder(**cfg["epoch_mixer"]),
        sequence_mixer=SequenceCNN(**cfg["sequence_mixer"]),
        num_classes=cfg["num_classes"],
    ).eval()
    x_flat, _ = _rand_inputs(model, batch=2, n_epochs=4)
    # One batch element has `absent` missing, the other has it all -inf.
    x_missing = {k: v.clone() for k, v in x_flat.items()}
    del x_missing[absent]
    x_neginf = {k: v.clone() for k, v in x_flat.items()}
    x_neginf[absent][1] = float("-inf")
    with torch.no_grad():
        y_missing = model(x_missing)
        y_neginf = model(x_neginf)
    # The all--inf row of x_neginf must match the missing row of x_missing:
    # a missing key is exactly the same as an all -inf signal. Both calls use
    # the same input shape, so the comparison is exact up to fp32 rounding in
    # the cuDNN transformer (nondeterministic reduction order).
    torch.testing.assert_close(y_missing[1], y_neginf[1], atol=1e-5, rtol=1e-5)
    # A batch element with every signal missing propagates as -inf.
    x_none = {k: torch.full_like(v, float("-inf")) for k, v in x_flat.items()}
    with torch.no_grad():
        y_none = model(x_none)
    assert torch.isinf(y_none).all()
    # No signals at all is an error.
    with pytest.raises(ValueError, match="At least one signal"):
        model({})


# ── (4) Golden parity against the authors' code (@hf: downloads needed) ─


@pytest.mark.hf
@pytest.mark.parametrize(
    ("name", "signals", "seeds", "cases"),
    [
        (
            "wav2sleep",
            ["ABD", "ECG", "PPG", "THX"],
            [100, 101, 102, 103],
            ["logits_all", "logits_ecg", "logits_ppg", "logits_ecg_thx"],
        ),
        (
            "wav2sleep-eog",
            ["EOG-L", "EOG-R"],
            [100, 101],
            ["logits_all"],
        ),
    ],
)
def test_golden_parity(name, signals, seeds, cases):
    """Our port matches the authors' code on the recipes in the fixture README."""
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(f"joncarter/{name}", "config.yaml")
    state_dict_path = hf_hub_download(f"joncarter/{name}", "state_dict.pth")
    model = Wav2Sleep.from_original(config_path, state_dict_path)

    npz = np.load(FIXTURES / f"{name}.npz")

    def case_logits(active_signals: list[str]) -> torch.Tensor:
        x: dict[str, torch.Tensor] = {}
        for i, sig in enumerate(signals):
            rng = np.random.default_rng(seeds[i])
            arr = rng.standard_normal(SAMPLES_PER_EPOCH[sig] * N_EPOCHS).astype(np.float32)
            x[sig] = (
                torch.from_numpy(arr).reshape(1, -1)
                if sig in active_signals
                else torch.full((1, SAMPLES_PER_EPOCH[sig] * N_EPOCHS), float("-inf"))
            )
        with torch.no_grad():
            return model(x)[0].cpu().numpy()

    expected_active = {
        "logits_all": signals,
        "logits_ecg": ["ECG"],
        "logits_ppg": ["PPG"],
        "logits_ecg_thx": ["ECG", "THX"],
    }
    for case in cases:
        got = case_logits(expected_active[case])
        want = npz[case][0]
        assert got.shape == want.shape, (case, got.shape, want.shape)
        np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-4, err_msg=case)
