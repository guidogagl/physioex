"""Tests of the corrected single-channel SeqSleepNet (ChannelNet) and of its cache-only reader."""

import json

import numpy as np
import pytest
import torch
from scipy.signal import spectrogram

from physioex.data.clean_cache import SequenceDataset, flat_epochs, load_night, to_100hz
from physioex.models.seqsleepnet_clean import (FREQS, MODALITY, ChannelNet, SplitRNN, StaticFilterbank,
                                               SpectralFrontend, filter_edges, scale_db)


def test_frontend_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3000).astype(np.float64)
    _, _, ref = spectrogram(x, fs=100, window="hamming", nperseg=200, noverlap=100, nfft=256,
                            detrend="constant", scaling="density", mode="psd")
    P = SpectralFrontend().double()(torch.from_numpy(x)[None]).numpy()[0]
    assert P.shape == (29, 129)
    np.testing.assert_allclose(P, ref.T, rtol=1e-4, atol=1e-12)   # float32 window buffer


@pytest.mark.parametrize("modality", ["EEG", "EOG", "EMG"])
def test_filterbank_convex_and_bounded(modality):
    cfg = MODALITY[modality]
    freqs = FREQS[cfg["bins"]]
    fb = StaticFilterbank(freqs, filter_edges(cfg["spacing"], cfg["nfilt"], float(freqs[0]), float(freqs[-1])))
    with torch.no_grad():
        fb.W.normal_(0, 3)                       # any learned W keeps columns convex
    w = fb.weights()
    torch.testing.assert_close(w.sum(0), torch.ones(cfg["nfilt"]))
    assert (w >= 0).all()
    nb = len(freqs)
    for level in (1e-12, 1e5):                   # far below the floor and far above the ceiling
        y = scale_db(fb(torch.full((1, 29, nb), level)))
        assert y.min() >= -1 and y.max() <= 1
    P = torch.rand(4, 29, nb) * 10
    s = fb(P)                                    # convex mean stays within the bin range of each frame
    assert (s <= P.max(-1, keepdim=True).values + 1e-5).all()
    assert (s >= P.min(-1, keepdim=True).values - 1e-5).all()


def test_split_lstm_equals_fused():
    torch.manual_seed(0)
    split = SplitRNN("lstm", 32, 16, 2, "L")
    fused = torch.nn.LSTM(32, 16, num_layers=2, batch_first=True, bidirectional=True)
    with torch.no_grad():
        for layer, rnn in enumerate(split.rnns):
            for name, p in rnn.named_parameters():
                getattr(fused, name.replace("_l0", f"_l{layer}")).copy_(p)
    x = torch.randn(3, 29, 32)
    out = {}
    split(x, out)
    # without the LayerNorm between layers the two stacks are identical
    h = x
    for rnn in split.rnns:
        h, _ = rnn(h)
    ref, _ = fused(x)
    torch.testing.assert_close(h, ref)
    torch.testing.assert_close(out["L1_raw"], split.rnns[0](x)[0])


def test_channelnet_outputs():
    m = ChannelNet("EEG").eval()
    logits, out = m(torch.randn(2, 21, 3000))
    assert logits.shape == (2, 21, 5)
    for k in ["F", "L1", "L4", "L1_raw", "alpha", "e", "G1", "G4", "q", "logits"]:
        assert k in out
    assert out["F"].abs().max() <= 1


def _write_night(root, ds, subj, chan, x, y, fs):
    d = root / ds / "signals" / subj / f"{chan}_clean_abc" / "pipe"
    d.mkdir(parents=True)
    np.save(d / "signal.npy", x.astype(np.float32))
    (d / "signal.meta.json").write_text(json.dumps({"fs_out": fs}))
    (root / ds / "labels").mkdir(parents=True, exist_ok=True)
    np.save(root / ds / "labels" / f"{subj}.npy", y)


def test_flat_epochs_trimmed_and_masked(tmp_path):
    rng = np.random.default_rng(1)
    n = 40
    x = rng.standard_normal((n, 3000))
    x[:3] = 0.2                                  # lights window at the start: trimmed
    x[-2:] = 0.2                                 # and at the end
    x[10] = 0.2                                  # inner flat epoch: masked
    x[20, 500:700] = 0.0                         # partly flat (2 s): masked
    y = rng.integers(0, 5, n)
    y[15] = 7                                    # unscored: masked
    _write_night(tmp_path, "ds", "s1", "EEG", x, np.r_[y, 0], 100.0)   # labels one epoch longer: truncated
    night, why = load_night(tmp_path, "ds", "s1", "EEG")
    assert why == "ok" and len(night.y) == n - 5
    assert night.y[10 - 3] == -1 and night.y[20 - 3] == -1 and night.y[15 - 3] == -1
    assert (night.y >= 0).sum() == n - 5 - 3
    full, part = flat_epochs(x, 100.0)
    assert full[10] and part[20] and not full[20]


def test_length_mismatch_truncated_at_end(tmp_path):
    x = np.random.default_rng(2).standard_normal((30, 3000))
    _write_night(tmp_path, "ds", "s1", "EEG", x, np.r_[np.arange(30) % 5, np.zeros(15, int)], 100.0)
    night, why = load_night(tmp_path, "ds", "s1", "EEG")
    assert why == "ok" and len(night.y) == 30
    np.testing.assert_array_equal(night.y, np.arange(30) % 5)   # aligned from the first epoch


def test_emg_resampled_to_100hz(tmp_path):
    fs, n = 256.0, 20
    t = np.arange(n * 30 * int(fs)) / fs
    x = np.sin(2 * np.pi * 20 * t).reshape(n, -1) + 0.01 * np.random.default_rng(3).standard_normal((n, 7680))
    _write_night(tmp_path, "ds", "s1", "EMG", x, np.ones(n, int), fs)
    night, why = load_night(tmp_path, "ds", "s1", "EMG")
    assert why == "ok" and night.x.shape == (n, 3000)
    assert night.offset_db == pytest.approx(10 * np.log10(90 / 40))
    f = np.fft.rfftfreq(3000, 1 / 100)
    spec = np.abs(np.fft.rfft(night.x[5].astype(np.float32)))
    assert abs(f[spec.argmax()] - 20) < 0.1    # the 20 Hz tone survives at the right frequency
    y = to_100hz(np.zeros((2, 3000)), 100.0)
    assert y.shape == (2, 3000)


def test_sequence_windows_cover_night(tmp_path):
    x = np.random.default_rng(4).standard_normal((50, 3000))
    _write_night(tmp_path, "ds", "s1", "EEG", x, np.ones(50, int), 100.0)
    night, _ = load_night(tmp_path, "ds", "s1", "EEG")
    ds = SequenceDataset([night], 21)
    starts = [s for _, s in ds.windows]
    assert starts == [0, 21, 29]
    xb, yb, off = ds[0]
    assert xb.shape == (21, 3000) and yb.shape == (21,) and float(off) == 0.0


def test_dropout_zero_unchanged_and_eval_deterministic():
    torch.manual_seed(0)
    ref = ChannelNet("EEG").eval()
    torch.manual_seed(0)
    reg = ChannelNet("EEG", dropout=0.3)
    assert set(reg.state_dict()) == set(ref.state_dict())      # no new parameters: old checkpoints load strict
    reg.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(2, 20, 3000)
    reg.eval()
    torch.testing.assert_close(reg(x)[0], ref(x)[0], rtol=0, atol=0)
    reg.train()
    a, b = reg(x)[0], reg(x)[0]
    assert not torch.allclose(a, b)                             # dropout active in train mode only


def test_random_offset_windows(tmp_path):
    from physioex.data.clean_cache import RandomOffsetSequences
    n, L = 107, 20
    x = np.arange(n * 3000, dtype=np.float64).reshape(n, 3000) % 977 + np.random.default_rng(5).standard_normal(
        (n, 3000))
    _write_night(tmp_path, "ds", "s1", "EEG", x, np.arange(n) % 5, 100.0)
    night, _ = load_night(tmp_path, "ds", "s1", "EEG")
    ds = RandomOffsetSequences([night], L, max_shift=0, seed=0)
    seen_sets = []
    for k in range(4):
        ds.set_epoch(k)
        cover = np.zeros(len(night.y), int)
        for _, s in ds.windows:
            cover[s:s + L] += 1
        assert (cover >= 1).all() and cover.max() <= 2       # every epoch once, twice only at the night edges
        seen_sets.append(tuple(s for _, s in ds.windows))
    assert len(set(seen_sets)) > 1                            # boundaries change across passes
    xb, yb, _ = ds[1]
    s = ds.windows[1][1]
    np.testing.assert_array_equal(xb.numpy(), night.x[s:s + L].astype(np.float32))   # no shift = plain window
    ds2 = RandomOffsetSequences([night], L, max_shift=300, seed=0)
    xs, ys, _ = ds2[1]
    s2 = ds2.windows[1][1]
    assert xs.shape == (L, 3000) and torch.equal(ys, torch.from_numpy(night.y[s2:s2 + L]))   # labels not shifted
