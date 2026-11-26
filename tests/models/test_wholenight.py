"""Whole-night sequence fine-tuning (xmai-lab/sleep-mechinterp#32)."""

import numpy as np
import pytest
import torch

from physioex.data.night_sequences import (MIXES, SequenceSampler, crop, cycle_cuts, junctions_ok, permute_night,
                                           segments, transition_matrix)
from physioex.models.channelnet_mm import MODALITIES, MultiChannelNet
from physioex.models.seqsleepnet_clean import ChannelNet
from physioex.models.wholenight import WholeNightNet
from physioex.train.bin.train_wholenight import CONFIGS

E = {"EEG": 3000, "EOG": 3000, "EMG": 3000}


def _mm(seed=0):
    torch.manual_seed(seed)
    mm = MultiChannelNet({m: ChannelNet(m) for m in MODALITIES}, ("F",)).eval()
    with torch.no_grad():
        for c in mm.conn.values():
            for k in c.g:
                c.g[k].fill_(0.3)
    return mm


def _wn(config, seed=0):
    _, _, loops, tied, _, night, freeze = CONFIGS[config]
    return WholeNightNet(_mm(seed), loops=loops, tied=tied, night_head=night, freeze_seq=freeze).eval()


def _x(B, L, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = {m: torch.randn(B, L, E[m], generator=g) for m in MODALITIES}
    off = {m: torch.full((B,), 0.0) for m in MODALITIES}
    return x, off


@pytest.mark.parametrize("config", list(CONFIGS))
def test_every_config_equals_s3f_at_init(config):
    wn = _wn(config)
    x, off = _x(1, 30)
    occ = {m: torch.tensor([m == "EOG"]) for m in MODALITIES}
    with torch.no_grad():
        ref, ref_aux, _ = wn.mm(x, off, occ)
        lg, aux, o = wn(x, off, occ)
        lg2, _, _ = wn(x, off, occ, lengths=torch.tensor([30]))
    torch.testing.assert_close(lg, ref, rtol=0, atol=1e-5)
    torch.testing.assert_close(lg2, ref, rtol=0, atol=1e-5)
    for m in MODALITIES:
        torch.testing.assert_close(aux[m], ref_aux[m], rtol=0, atol=1e-5)
    assert len(o["loops"]) == CONFIGS[config][2]


def test_whole_night_path_is_exactly_multichannelnet():
    wn = _wn("A-mix")
    x, off = _x(1, 25)
    with torch.no_grad():
        assert torch.equal(wn(x, off)[0], wn.mm(x, off)[0])


@pytest.mark.parametrize("config", ["A-mix", "B-untied", "C+A"])
def test_packed_equals_unpadded(config):
    wn = _wn(config)
    with torch.no_grad():                                   # make loops / night head active
        for p in wn.gain.values():
            p.fill_(0.5)
        if wn.night is not None:
            wn.night.c.fill_(0.5)
    xa, oa = _x(1, 30, seed=2)
    xb, ob = _x(1, 18, seed=3)
    xp = {m: torch.zeros(2, 30, E[m]) for m in MODALITIES}
    for m in MODALITIES:
        xp[m][0], xp[m][1, :18] = xa[m][0], xb[m][0]
    off = {m: torch.zeros(2) for m in MODALITIES}
    with torch.no_grad():
        lp, _, _ = wn(xp, off, lengths=torch.tensor([30, 18]))
        la, _, _ = wn(xa, oa)
        lb, _, _ = wn(xb, ob)
    torch.testing.assert_close(lp[0], la[0], rtol=0, atol=1e-5)
    torch.testing.assert_close(lp[1, :18], lb[0], rtol=0, atol=1e-5)


@pytest.mark.parametrize("config", ["A-mix", "B-untied", "C"])
def test_frozen_parameters_do_not_change(config):
    wn = _wn(config)
    before = {n: p.detach().clone() for n, p in wn.named_parameters()}
    trainable = {n for n, p in wn.named_parameters() if p.requires_grad}
    opt = torch.optim.Adam([p for p in wn.parameters() if p.requires_grad], lr=1e-2)
    x, off = _x(2, 12)
    wn.train()
    assert not wn.mm.training and not wn.mm.branches["EEG"].epoch_rnn.training
    assert wn.mm.branches["EEG"].seq_rnn.training == (config != "C")
    lg, aux, _ = wn(x, off, lengths=torch.tensor([12, 9]))
    y = torch.randint(0, 5, (2, 12))
    y[1, 9:] = -1
    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
    (crit(lg.reshape(-1, 5), y.reshape(-1)) + sum(crit(a.reshape(-1, 5), y.reshape(-1)) for a in aux.values())).backward()
    opt.step()
    for n, p in wn.named_parameters():
        if n not in trainable:
            assert torch.equal(p, before[n]), n
    frozen_prefixes = ("mm.branches.EEG.epoch_rnn", "mm.branches.EEG.attention", "mm.conn.F1")
    assert not any(n.startswith(frozen_prefixes) for n in trainable)
    if config == "C":
        assert not any(".seq_rnn." in n or ".clf." in n for n in trainable)
        assert any(n.startswith("night.") for n in trainable)
    else:
        assert any(".seq_rnn." in n for n in trainable)


def test_padding_labels_are_ignored():
    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
    lg = torch.randn(10, 5)
    y = torch.tensor([0, 1, 2, 3, 4, -1, -1, -1, -1, -1])
    assert torch.isclose(crit(lg, y), crit(lg[:5], y[:5]))


def _night_labels():
    # onset W, then 3 cycles ending with REM >= 10, a W bout, final W
    y = [0] * 8 + [1] * 5 + [2] * 30 + [3] * 20 + [4] * 12 + [2] * 25 + [3] * 10 + [4] * 15 + [0] * 6 + \
        [1] * 4 + [2] * 30 + [4] * 20 + [2] * 10 + [0] * 5
    return np.array(y)


def test_cycle_cuts_and_permutation():
    y = _night_labels()
    cuts = cycle_cuts(y)
    assert len(cuts) >= 2
    for c in cuts:
        assert y[c - 1] in (0, 4) and y[c] != y[c - 1]
    trans = transition_matrix([y])
    rng = np.random.default_rng(0)
    seen_perm = False
    for _ in range(20):
        idx = permute_night(y, trans, rng, "cycle")
        assert sorted(idx.tolist()) == list(range(len(y)))
        segs = segments(len(y), cuts)
        assert idx[:segs[0][1]].tolist() == list(range(segs[0][1]))     # first segment in place
        if not np.array_equal(idx, np.arange(len(y))):
            seen_perm = True
            new = []                                                     # rebuild the segment order from idx
            i = 0
            while i < len(idx):
                s = next(sg for sg in segs if sg[0] == idx[i])
                new.append(s)
                i += s[1] - s[0]
            assert junctions_ok(y, new, trans, 0.01)
    assert seen_perm
    idx = permute_night(y, trans, rng, "random")
    assert sorted(idx.tolist()) == list(range(len(y)))


def test_short_night_is_not_permuted_and_crop_bounds():
    y = np.array([0] * 10 + [2] * 40 + [0] * 10)
    rng = np.random.default_rng(0)
    assert np.array_equal(permute_night(y, transition_matrix([y]), rng), np.arange(len(y)))
    for _ in range(50):
        c = crop(500, rng)
        assert 20 <= len(c) <= 500 and c[0] >= 0 and c[-1] < 500


class _N:
    def __init__(self, n, k):
        rng = np.random.default_rng(k)
        self.x = {m: rng.standard_normal((n, E[m])).astype(np.float16) for m in MODALITIES}
        self.y = np.resize(_night_labels(), n)
        self.offset_db = {m: 0.0 for m in MODALITIES}


def test_sampler_batches_and_equal_budget_accounting():
    nights = [_N(240, 0), _N(200, 1), _N(260, 2)]
    trans = transition_matrix([n.y for n in nights])
    for mix in MIXES:
        it = iter(SequenceSampler(nights, MIXES[mix], trans, seed=0))
        for _ in range(3):
            b = next(it)
            B, L = b["y"].shape
            assert B == (32 if mix == "l20" else 4)
            assert int(b["lengths"].max()) == L and all(b["x"][m].shape[:2] == (B, L) for m in MODALITIES)
            for i, n in enumerate(b["lengths"].tolist()):
                assert (b["y"][i, n:] == -1).all() and (b["y"][i, :n] >= 0).all()
            if mix == "l20":
                assert L == 20


@pytest.mark.parametrize("config", list(CONFIGS))
def test_optimiser_groups_partition_trainable_parameters(config):
    wn = _wn(config)
    groups = [wn.seq_parameters(), wn.head_parameters(), wn.gain_parameters()]
    ids = [id(p) for g in groups for p in g]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in wn.parameters() if p.requires_grad}
