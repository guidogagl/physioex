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


def test_length_mismatch_skipped(tmp_path):
    x = np.random.default_rng(2).standard_normal((30, 3000))
    _write_night(tmp_path, "ds", "s1", "EEG", x, np.zeros(45, int), 100.0)
    night, why = load_night(tmp_path, "ds", "s1", "EEG")
    assert night is None and why == "length mismatch"


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
