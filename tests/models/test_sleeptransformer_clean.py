"""Tests of the SleepTransformer instance of the multimodal framework (xmai-lab/sleep-mechinterp#26)."""

import json

import numpy as np
import pytest
import torch

from physioex.data.clean_cache import CleanCacheMultiNights, CleanCacheNights, LazyArray, MultiSequenceDataset
from physioex.models.channelnet_mm import MODALITIES, MultiChannelNet
from physioex.models.seqsleepnet_clean import ChannelNet, scale_db
from physioex.models.sleeptransformer_clean import SleepTransformer3chClean, SleepTransformerChannel


def _inputs(B=2, L=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = {m: torch.randn(B, L, 3000, generator=g) for m in MODALITIES}
    off = {m: torch.full((B,), 0.0 if m != "EMG" else 3.5) for m in MODALITIES}
    return x, off


def _branches(kind, seed=0):
    torch.manual_seed(seed)
    make = SleepTransformerChannel if kind == "st" else ChannelNet
    return {m: make(m).eval() for m in MODALITIES}


def test_block_api_equals_sleeptransformer_forward():
    """The block-by-block branch is physioex SleepTransformer on the framework input (bins 1..128, fixed dB scale)."""
    torch.manual_seed(0)
    br = SleepTransformerChannel("EMG").eval()
    x, off = torch.randn(2, 21, 3000), torch.tensor([0.0, 3.5])
    with torch.no_grad():
        logits, out = br(x, off)
        ref = br.net(scale_db(br.frontend(x)[..., 1:129], off.view(2, 1, 1, 1)).unsqueeze(2))
    torch.testing.assert_close(logits, ref, rtol=0, atol=1e-5)
    assert out["L4"].shape == (2 * 21, 29, 128) and out["G4"].shape == (2, 21, 128) and out["alpha"].shape == (2, 21, 29)


@pytest.mark.parametrize("kind", ["cn", "st"])
def test_gates_zero_equal_pretrained(kind):
    br = _branches(kind)
    x, off = _inputs()
    with torch.no_grad():
        ref = {m: br[m](x[m], off[m])[0] for m in MODALITIES}
        mm = MultiChannelNet(br, ("F",)).eval()
        for c in mm.conn.values():
            for lin in c.W.values():
                lin.weight.normal_()
        _, aux, out = mm(x, off)
    for m in MODALITIES:
        torch.testing.assert_close(aux[m], ref[m], rtol=0, atol=1e-5)
    assert mm.sites == ["F1", "F2", "F3", "F4"]


@pytest.mark.parametrize("kind", ["cn", "st"])
def test_occluded_modality_sends_nothing(kind):
    mm = MultiChannelNet(_branches(kind), ("F",)).eval()
    x, off = _inputs()
    with torch.no_grad():
        for c in mm.conn.values():
            for k in c.g:
                c.g[k].fill_(0.5)
        occ = {m: torch.tensor([m == "EEG", False]) for m in MODALITIES}
        f1, a1, _ = mm(x, off, occ)
        x2 = dict(x)
        x2["EEG"] = torch.randn_like(x["EEG"])
        f2, a2, _ = mm(x2, off, occ)
    torch.testing.assert_close(f1[0], f2[0], rtol=0, atol=1e-5)
    for m in ("EOG", "EMG"):
        torch.testing.assert_close(a1[m][0], a2[m][0], rtol=0, atol=1e-5)
    assert not torch.allclose(f1[1], f2[1])


def test_frozen_branches_run_without_dropout():
    mm = MultiChannelNet(_branches("st"), ("F",))
    mm.frozen_branches = True
    mm.train()
    assert mm.fused.training and not any(b.training for b in mm.branches.values())
    x, off = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(mm(x, off)[1]["EEG"], mm(x, off)[1]["EEG"], rtol=0, atol=0)
    mm.frozen_branches = False
    mm.train()
    assert all(b.training for b in mm.branches.values())


def test_st3ch_shapes_and_scaling():
    x, off = _inputs(B=3, L=4)
    net = SleepTransformer3chClean()
    net.fit_scaling([(x, off)])
    s = (net.spectrogram(x, off) - net.mean[:, None, :]) / net.std[:, None, :]
    assert s.shape == (3, 4, 3, 29, 128)
    assert abs(float(s.mean())) < 1e-3 and abs(float(s.std()) - 1) < 0.05
    net.eval()
    logits, aux, _ = net(x, off)
    assert logits.shape == (3, 4, 5) and aux == {}
    assert {"mean", "std"} <= set(net.state_dict())


def _write(root, ds, subj, chan, x, fs):
    d = root / ds / "signals" / subj / f"{chan}_clean_abc" / "pipe"
    d.mkdir(parents=True)
    np.save(d / "signal.npy", x.astype(np.float16))
    (d / "signal.meta.json").write_text(json.dumps({"fs_out": fs}))


def _cache(tmp_path, n_subj=3, n=40):
    rng = np.random.default_rng(0)
    root = tmp_path / "cache"
    (root / "ds" / "labels").mkdir(parents=True)
    for k in range(n_subj):
        s = f"s{k}"
        for c, fs, E in (("EEG", 100.0, 3000), ("EOG", 100.0, 3000), ("EMG", 256.0, 7680)):
            _write(root, "ds", s, c, rng.standard_normal((n + k, E)), fs)
        np.save(root / "ds" / "labels" / f"{s}.npy", rng.integers(0, 5, n + k))
    split = {"ds": {"train": ["s0", "s1"], "valid": [], "test": ["s2"]}}
    return root, split


def test_local_dir_reader_matches_eager(tmp_path):
    root, split = _cache(tmp_path)
    eager = CleanCacheMultiNights(root, ["ds"], split=split, log=lambda *_: None)
    lazy = CleanCacheMultiNights(root, ["ds"], split=split, log=lambda *_: None, local_dir=tmp_path / "local",
                                 workers=2)
    for k in ("train", "test"):
        assert [n.subject for n in eager.nights[k]] == [n.subject for n in lazy.nights[k]]
        for a, b in zip(eager.nights[k], lazy.nights[k]):
            np.testing.assert_array_equal(a.y, b.y)
            assert a.offset_db == b.offset_db
            for m in MODALITIES:
                assert isinstance(b.x[m], LazyArray)
                np.testing.assert_array_equal(np.asarray(a.x[m]), np.asarray(b.x[m]))
    xa, ya, _ = MultiSequenceDataset(eager.nights["train"], 20)[3]
    xb, yb, _ = MultiSequenceDataset(lazy.nights["train"], 20)[3]
    torch.testing.assert_close(xa["EMG"], xb["EMG"])
    single = CleanCacheNights(root, ["ds"], "EEG", split=split, log=lambda *_: None, local_dir=tmp_path / "l1",
                              workers=2)
    np.testing.assert_array_equal(np.asarray(single.nights["train"][0].x, dtype=np.float32),
                                  np.asarray(eager.nights["train"][0].x["EEG"], dtype=np.float32))


def test_split_keeps_only_five_stage_nights(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "make_split", Path(__file__).resolve().parents[2] / "examples/pretrained/st-shhs/make_split.py")
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    root = tmp_path / "cache"
    (root / "ds" / "labels").mkdir(parents=True)
    labels = {"a": [0, 1, 2, 3, 4, -1], "b": [0, 1, 2, 4, 4], "c": [4, 3, 2, 1, 0], "d": [0, 1, 2, 3, -1, 4]}
    for s, y in labels.items():
        (root / "ds" / "signals" / s).mkdir(parents=True)
        np.save(root / "ds" / "labels" / f"{s}.npy", np.array(y, dtype=np.int16))
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"fold_0": {"train": ["a", "b", "z"], "valid": ["c"], "test": ["d"]}}))
    split, counts = ms.make_split(root, "ds", bench)
    assert split == {"ds": {"train": ["a"], "valid": ["c"], "test": ["d"]}}
    assert counts["train"] == {"benchmark": 3, "kept": 1, "not_in_cache": 1, "missing_stage": 1}
