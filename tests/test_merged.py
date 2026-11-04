"""physioex.data.merged: merged train/test sets over the ProtoSleepNet splits (fake datasets)."""
import json

import numpy as np
import pytest
import torch

from physioex.data import merged as M

N_EP = 10  # windows per fake night


class FakeDS(torch.utils.data.Dataset):
    """Minimal BasePhysioDataset stand-in: N_EP windows per subject, one channel at ``fs``."""

    def __init__(self, name, subjects, fs=100.0, missing=()):
        self.DATASET_NAME, self.sequence_length, self.fs = name, 3, fs
        self._subjects_ids = list(subjects)
        self._missing = set(missing)
        self.cache_enabled = False
        self._subject_ranges = [(s, i * N_EP, (i + 1) * N_EP) for i, s in enumerate(self._subjects_ids)]

    def get_subjects(self):
        return list(self._subjects_ids)

    def __len__(self):
        return N_EP * len(self._subjects_ids)

    def _subject_ids_to_flat_indices(self, ids):
        pos = {s: i for i, s in enumerate(self._subjects_ids)}
        return [pos[s] * N_EP + k for s in ids for k in range(N_EP)]

    def __getitem__(self, i):
        n = int(round(self.fs * 30))
        return {"signals": {"EMG_0": torch.zeros(self.sequence_length, n)}, "channel_order": ["EMG_0"],
                "labels": torch.zeros(self.sequence_length, dtype=torch.long),
                "channel_info": {"EMG_0": {"fs_out": self.fs}},
                "subject": {"subject_id": self._subjects_ids[i // N_EP]}}


def _splits(tmp_path):
    sp = {}
    for u in M.UNITS:
        site, n = u["site"], u["name"]
        d = sp.setdefault(site, {"train": [], "val": [], "test": []})
        for k, m in (("train", 6), ("val", 2), ("test", 2)):
            d[k] += [[n, f"{n}-{k}{i}"] for i in range(m)]
    p = tmp_path / "splits.json"
    p.write_text(json.dumps(sp))
    return str(p)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    made = {}

    def make(name, channels, pipelines, sequence_length, cache_dir, memmap_cache_size=200, **extra):
        subs = [f"{name}-{k}{i}" for k, m in (("train", 6), ("val", 2), ("test", 2)) for i in range(m)]
        if M.BY_NAME[name]["item"] is None:
            subs += [f"{name}-extra{i}" for i in range(3)]
        ds = FakeDS(name, subs, fs=256.0 if name.startswith("mesa") else 100.0)
        made[name] = ds
        return ds

    monkeypatch.setattr(M, "make_dataset", make)
    monkeypatch.setattr(M, "cached_subjects", lambda ds: set(ds.get_subjects()) - {s for s in ds.get_subjects() if s.endswith("train5") and ds.DATASET_NAME == "mesa"})
    return _splits(tmp_path), made


def _ids(md, idx):
    out = set()
    for i in idx:
        d, j = md.multi._resolve_index(int(i))
        out.add((md.unit_names[d], md.multi._datasets[d]._subjects_ids[j // N_EP]))
    return out


def test_units_and_pool():
    assert len(M.POOL) == 10
    assert set(M.units_of(["mass"])) == {"mass_ss02", "mass_ss03", "mass_ss04", "mass_ss05"}
    assert "mass_ss01" in M.test_only_units() and "mass_ss01" not in M.units_of(M.POOL)
    for u in M.test_only_units():
        assert M.BY_NAME[u]["item"] is None
    with pytest.raises(ValueError):
        M.units_of(["alzheimers_ad"])


def test_build_train_only_train_split(fake):
    splits, _ = fake
    md = M.build_train(["shhs", "mesa", "mass"], splits, ["EMG_clean"], None, 3, None, valid_per_item=15, memmap_cache_size=0)
    assert md.unit_names == M.units_of(["shhs", "mesa", "mass"])
    tr, va = _ids(md, md.train_idx), _ids(md, md.valid_idx)
    assert all("-train" in s for _, s in tr) and all("-val" in s for _, s in va)
    assert not ({s for _, s in tr} & {s for _, s in va})
    # mesa-train5 is not in the cache: dropped
    assert ("mesa", "mesa-train5") not in tr and md.report["mesa"]["train_nights"] == 5
    # validation: at most valid_per_item windows per item
    per_item = {}
    for u, _ in va:
        per_item[M.BY_NAME[u]["item"]] = per_item.get(M.BY_NAME[u]["item"], 0) + 1
    assert len(md.valid_idx) <= 3 * 15


def test_build_train_empty_item(fake, monkeypatch):
    splits, _ = fake
    monkeypatch.setattr(M, "cached_subjects", lambda ds: set() if ds.DATASET_NAME == "sleepedf" else set(ds.get_subjects()))
    with pytest.raises(ValueError):
        M.build_train(["sleepedf", "dcsm", "hmc"], splits, ["EMG_clean"], None, 3, None, memmap_cache_size=0)
    md = M.build_train(["sleepedf", "dcsm", "hmc"], splits, ["EMG_clean"], None, 3, None, allow_empty=True,
                       memmap_cache_size=0)
    assert md.report["empty_items"] == ["sleepedf"]


def test_site_sampler_equal_mass(fake):
    from physioex.data.sampling import site_balanced_weights

    splits, _ = fake
    md = M.build_train(["shhs", "dcsm", "mass"], splits, ["EMG_clean"], None, 3, None, memmap_cache_size=0)
    w, _ = site_balanced_weights(md.multi, md.train_idx, md.item_of_dataset)
    items = np.asarray([md.item_of_dataset[md.multi._resolve_index(int(i))[0]] for i in md.train_idx])
    mass = {it: float(w[items == it].sum() / w.sum()) for it in ("shhs", "dcsm", "mass")}
    assert all(abs(v - 1 / 3) < 1e-6 for v in mass.values()), mass


def test_build_test(fake):
    splits, _ = fake
    md = M.build_test(splits, ["EMG_clean"], None, 3, None, memmap_cache_size=0)
    by = {}
    for d, s, item, sub in md.test:
        by.setdefault(md.unit_names[d], []).append((s, item, sub))
    assert all("-test" in s for s, _, _ in by["dcsm"]) and len(by["dcsm"]) == 2
    # test-only units: every night (splits + extra), item None, own subgroup
    assert len(by["alzheimers_ad"]) == 13 and {it for _, it, _ in by["alzheimers_ad"]} == {None}
    assert {sub for _, _, sub in by["lsec_dre"]} == {"lsec_dre"}
    assert {sub for _, _, sub in by["mass_ss03"]} == {"mass_ss03"}


def test_resampled_view():
    t = np.arange(3 * 30 * 256) / 256.0
    x = np.sin(2 * np.pi * 20.0 * t).reshape(3, -1).astype(np.float32)

    class One(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return {"signals": {"EMG_0": torch.from_numpy(x.copy()), "EEG_0": torch.zeros(3, 3000)},
                    "channel_info": {"EMG_0": {"fs_out": 256.0}, "EEG_0": {"fs_out": 100.0}}}

    it = M.ResampledView(One(), "EMG", 100.0)[0]
    y = it["signals"]["EMG_0"].numpy()
    assert y.shape == (3, 3000) and it["channel_info"]["EMG_0"]["fs_out"] == 100.0
    assert it["signals"]["EEG_0"].shape == (3, 3000)
    f = np.fft.rfftfreq(y.size, 1 / 100.0)[np.argmax(np.abs(np.fft.rfft(y.reshape(-1))))]
    assert abs(f - 20.0) < 0.1
    # already at the rate: unchanged object
    z = np.zeros((3, 3000), np.float32)

    class Same(One):
        def __getitem__(self, i):
            return {"signals": {"EMG_0": torch.from_numpy(z)}, "channel_info": {"EMG_0": {"fs_out": 100.0}}}

    assert M.ResampledView(Same(), "EMG", 100.0)[0]["signals"]["EMG_0"].data_ptr() == torch.from_numpy(z).data_ptr()
