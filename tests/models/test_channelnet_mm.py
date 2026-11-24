"""Tests of the multimodal ChannelNet (gated cross-modal connections) and of its three-modality reader."""

import json

import numpy as np
import pytest
import torch

from physioex.data.clean_cache import MultiSequenceDataset, load_multimodal_night
from physioex.models.channelnet_mm import (MODALITIES, CrossConnection, MultiChannelNet, occlusion_probs,
                                           point_names, sample_occlusion)
from physioex.models.seqsleepnet_clean import ChannelNet


def _branches(seed=0):
    torch.manual_seed(seed)
    return {m: ChannelNet(m).eval() for m in MODALITIES}


def _inputs(B=2, L=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = {m: torch.randn(B, L, 3000, generator=g) for m in MODALITIES}
    off = {m: torch.full((B,), 0.0 if m != "EMG" else 3.5) for m in MODALITIES}
    return x, off


def test_point_names():
    assert point_names({"F", "E", "Q"}) == ["F1", "F2", "F3", "F4", "E", "Q1", "Q2", "Q3", "Q4"]
    assert point_names({"E"}) == ["E"]
    assert point_names(set()) == []


@pytest.mark.parametrize("points", [(), ("E",), ("F", "E", "Q")])
def test_gates_zero_equal_pretrained(points):
    """With every gate at 0 and no occlusion, each auxiliary head reproduces its pretrained ChannelNet exactly."""
    br = _branches()
    x, off = _inputs()
    ref = {m: br[m](x[m], off[m])[0] for m in MODALITIES}
    mm = MultiChannelNet(br, points).eval()
    with torch.no_grad():
        for c in mm.conn.values():                 # non-trivial W, still gated off
            for lin in c.W.values():
                lin.weight.normal_()
        _, aux, out = mm(x, off)
    for m in MODALITIES:
        torch.testing.assert_close(aux[m], ref[m], rtol=0, atol=1e-6)


def test_gate_changes_only_the_target():
    br = _branches()
    x, off = _inputs()
    mm = MultiChannelNet(br, ("E",)).eval()
    with torch.no_grad():
        _, aux0, _ = mm(x, off)
        mm.conn["E"].g["EEG_EMG"].fill_(1.0)          # EEG -> EMG only
        _, aux1, _ = mm(x, off)
        assert not torch.allclose(aux1["EMG"], aux0["EMG"])
        torch.testing.assert_close(aux1["EEG"], aux0["EEG"], rtol=0, atol=0)
        torch.testing.assert_close(aux1["EOG"], aux0["EOG"], rtol=0, atol=0)
        mm.disabled = {"E"}                            # ablation restores the pretrained output
        _, aux2, _ = mm(x, off)
        torch.testing.assert_close(aux2["EMG"], aux0["EMG"], rtol=0, atol=0)


def test_occluded_modality_sends_nothing():
    """An occluded sender changes neither the other branches nor the fused logits, whatever its input."""
    br = _branches()
    x, off = _inputs()
    mm = MultiChannelNet(br, ("F", "E", "Q")).eval()
    with torch.no_grad():
        for c in mm.conn.values():
            for k in c.g:
                c.g[k].fill_(0.5)
        occ = {m: torch.tensor([m == "EEG", False]) for m in MODALITIES}   # sample 0: EEG hidden
        f1, a1, _ = mm(x, off, occ)
        x2 = dict(x)
        x2["EEG"] = torch.randn_like(x["EEG"])
        f2, a2, _ = mm(x2, off, occ)
    torch.testing.assert_close(f1[0], f2[0], rtol=0, atol=1e-6)
    for m in ("EOG", "EMG"):
        torch.testing.assert_close(a1[m][0], a2[m][0], rtol=0, atol=1e-6)
    assert not torch.allclose(f1[1], f2[1])            # sample 1 still uses EEG


def test_occlusion_sampling():
    probs = occlusion_probs({"EEG": 0.8, "EOG": 0.74, "EMG": 0.46})
    assert probs["EEG"] == pytest.approx(0.5) and probs["EMG"] == pytest.approx(0.5 * 0.46 / 0.8)
    occ = sample_occlusion(20000, probs, MODALITIES, "cpu")
    allo = torch.stack([occ[m] for m in MODALITIES], -1).all(-1)
    assert not allo.any()
    assert abs(occ["EEG"].float().mean() - 0.5) < 0.05


def test_parameter_groups():
    mm = MultiChannelNet(_branches(), ("E", "Q"))
    new = {id(p) for p in mm.new_parameters()}
    old = {id(p) for p in mm.branch_parameters()}
    assert not new & old
    assert new | old == {id(p) for p in mm.parameters()}
    assert len(CrossConnection(MODALITIES).pairs) == 6


def _write(root, ds, subj, chan, x, fs):
    d = root / ds / "signals" / subj / f"{chan}_clean_abc" / "pipe"
    d.mkdir(parents=True)
    np.save(d / "signal.npy", x.astype(np.float32))
    (d / "signal.meta.json").write_text(json.dumps({"fs_out": fs}))
    nd = root / ds / "signals" / subj / f"{chan}_clean_abc" / "night"     # continuous copy: must be ignored
    nd.mkdir()
    np.save(nd / "signal.npy", x.reshape(-1).astype(np.float32))


def test_multimodal_loader(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    eeg = rng.standard_normal((n, 3000))
    eog = rng.standard_normal((n + 3, 3000))           # longer: truncated
    emg = rng.standard_normal((n, 7680))               # 256 Hz
    eeg[:2] = 0.1                                      # flat start in EEG only: trimmed for all
    eog[12] = 0.1                                      # inner flat epoch in EOG: masked
    emg[20, 1000:1600] = 0.0                           # partly flat EMG: masked
    for c, x, fs in (("EEG", eeg, 100.0), ("EOG", eog, 100.0), ("EMG", emg, 256.0)):
        _write(tmp_path, "ds", "s1", c, x, fs)
    (tmp_path / "ds" / "labels").mkdir()
    y = rng.integers(0, 5, n + 5)
    np.save(tmp_path / "ds" / "labels" / "s1.npy", y)
    night, why = load_multimodal_night(tmp_path, "ds", "s1")
    assert why == "ok"
    assert all(v.shape == (n - 2, 3000) for v in night.x.values())
    assert night.y[12 - 2] == -1 and night.y[20 - 2] == -1
    np.testing.assert_array_equal(night.y[:5], y[2:7])
    np.testing.assert_allclose(night.x["EEG"][0], eeg[2].astype(np.float32))
    assert night.offset_db["EMG"] == pytest.approx(10 * np.log10(90 / 40)) and night.offset_db["EEG"] == 0.0
    ds = MultiSequenceDataset([night], 20, stride=1)
    xb, yb, ob = ds[0]
    assert set(xb) == set(MODALITIES) and xb["EMG"].shape == (20, 3000) and yb.shape == (20,)
    assert len(ds) == n - 2 - 20 + 1


def test_branch_dropout_off_in_eval_and_active_in_train():
    torch.manual_seed(0)
    br = {m: ChannelNet(m, dropout=0.3).eval() for m in MODALITIES}
    x, off = _inputs()
    ref = {m: br[m](x[m], off[m])[0] for m in MODALITIES}
    mm = MultiChannelNet(br, ("F", "E", "Q")).eval()
    with torch.no_grad():
        _, aux, _ = mm(x, off)
    for m in MODALITIES:
        torch.testing.assert_close(aux[m], ref[m], rtol=0, atol=1e-6)
    mm.train()
    with torch.no_grad():
        assert not torch.allclose(mm(x, off)[0], mm(x, off)[0])


def test_ssn3ch_baseline_shapes_and_scaling():
    from physioex.models.seqsleepnet_3ch_clean import SeqSleepNet3chClean
    x, off = _inputs(B=3, L=4)
    net = SeqSleepNet3chClean(dropout=0.1)
    net.fit_scaling([(x, off)])
    s = (net.spectrogram(x, off) - net.mean[:, None, :]) / net.std[:, None, :]
    assert s.shape == (3, 4, 3, 29, 129)
    assert abs(float(s.mean())) < 1e-3 and abs(float(s.std()) - 1) < 0.05     # standardised on the fit data
    net.eval()
    logits, aux, _ = net(x, off)
    assert logits.shape == (3, 4, 5) and aux == {}
    assert "mean" in net.state_dict() and "std" in net.state_dict()


def test_multi_random_offsets_aligned(tmp_path):
    from physioex.data.clean_cache import MultiRandomOffsetSequences
    rng = np.random.default_rng(1)
    n = 50
    base = rng.standard_normal((n, 3000))
    for c in ("EEG", "EOG", "EMG"):
        _write(tmp_path, "ds", "s1", c, base, 100.0)
    (tmp_path / "ds" / "labels").mkdir()
    np.save(tmp_path / "ds" / "labels" / "s1.npy", rng.integers(0, 5, n))
    night, _ = load_multimodal_night(tmp_path, "ds", "s1")
    ds = MultiRandomOffsetSequences([night], 20, max_shift=300, seed=0)
    for j in range(len(ds)):
        x, y, _ = ds[j]
        torch.testing.assert_close(x["EEG"], x["EOG"])         # one shift shared by all modalities
