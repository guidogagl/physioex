"""EEGAR (next-token model on patch codes, xmai-lab/sleep-lens#20): sequence layouts, FIM positions, loss,
sliding-window scoring, surprise of a rare code, mapping of the 4 patch grids on the 0.5 s tokens, and the
spectral twin of the codes. CPU, tiny models."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from physioex.models.eeg_ar import (
    BOS, EEGAR, IGNORE, MID, N_SPECIAL, PRE, SUF, collate, make_example, score_codes, token_scores,
)
from physioex.models.patch_codec import SpectralTwin, patch_spectra

K, L = 8, 2


def _tiny():
    torch.manual_seed(0)
    return EEGAR(n_codes=K, levels=L, d_model=32, layers=2, heads=4)


def _codes(W, seed=0):
    return np.random.default_rng(seed).integers(0, K, (W, L))


@pytest.mark.parametrize("span", [None, (1, 2), (3, 7), (5, 9)])
def test_every_code_is_a_target_once_at_its_own_position(span):
    W = 10
    c = _codes(W)
    e = make_example(c, K, span)
    assert all(len(v) == W * L + 3 for v in e.values())
    ok = e["tgt"] != IGNORE
    assert sorted(e["where"][ok]) == list(range(W * L))  # each code predicted exactly once
    # the position of the input equals the flat index of the code it predicts, in both layouts
    assert np.array_equal(e["pos"][ok], e["where"][ok])
    assert np.array_equal(e["tgt"][ok], c.reshape(-1)[e["where"][ok]])
    assert np.array_equal(e["lev"][ok], e["where"][ok] % L)
    tok = e["inp"][(e["inp"] >= N_SPECIAL)]
    assert len(tok) == W * L
    if span is None:
        assert e["inp"][0] == BOS and (e["inp"][W * L + 1:] == 0).all()
    else:
        a, b = span
        assert e["inp"][0] == PRE
        j_suf, j_mid = np.flatnonzero(e["inp"] == SUF)[0], np.flatnonzero(e["inp"] == MID)[0]
        assert e["pos"][j_suf] == b * L and e["pos"][j_mid] == a * L
        assert len(e["inp"]) - j_mid - 1 == (b - a) * L  # the middle closes the sequence


def test_forward_loss_and_backward():
    m = _tiny()
    b = collate([make_example(_codes(6, s), K, None if s % 2 else (1, 3)) for s in range(4)])
    loss, per = m(b)
    assert torch.isfinite(loss) and 0 < float(loss) < 10
    loss.backward()
    assert m.emb.weight.grad is not None and float(per["n"].sum()) == 4 * 6 * L


def test_causal_hidden_ignores_the_future():
    m = _tiny().eval()
    e1, e2 = make_example(_codes(6, 1), K), make_example(_codes(6, 1), K)
    e2["inp"][8:] = N_SPECIAL  # change everything after position 7
    h1 = m.hidden(*(torch.from_numpy(e1[k])[None] for k in ("inp", "pos", "lev")))
    h2 = m.hidden(*(torch.from_numpy(e2[k])[None] for k in ("inp", "pos", "lev")))
    assert torch.allclose(h1[0, :8], h2[0, :8], atol=1e-5)


def test_score_codes_scores_every_patch_once():
    m = _tiny().eval()
    for P in (5, 23, 40):
        nll, ent = score_codes(m, _codes(P, P), window=10, hop=4, batch=3)
        assert nll.shape == ent.shape == (P, L)
        assert (nll > 0).all() and (ent > 0).all() and (ent <= np.log(K) + 1e-4).all()


def test_rare_code_gives_a_surprise_peak():
    """Trained on a periodic code sequence, the model is most surprised by a code it never saw."""
    m = _tiny()
    seq = np.stack([np.arange(400) % 4, (np.arange(400) + 1) % 4], 1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    for _ in range(150):
        ss = rng.integers(0, 400 - 12, 16)
        loss, _ = m(collate([make_example(seq[s:s + 12], K, None) for s in ss]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    test = seq[:60].copy()
    test[30, 0] = 7  # never seen
    nll, ent = score_codes(m, test, window=12, hop=6, batch=4)
    S = (nll - ent).sum(1)
    assert S.argmax() == 30
    assert S[30] > 5 * max(np.median(np.abs(S)), 0.1)


def test_infill_shapes_and_range():
    m = _tiny().eval()
    out = m.infill(torch.from_numpy(_codes(8)), 2, 5, n_samples=3, generator=torch.Generator().manual_seed(0))
    assert out.shape == (3, 3, L) and int(out.min()) >= 0 and int(out.max()) < K


def test_token_scores_average_the_covering_patches():
    n = 1000  # 20 tokens of 0.5 s
    per_grid = {o: np.zeros((n - o) // 200) for o in (0, 50, 100, 150)}
    per_grid[0][1] = 4.0  # samples 200-399 = tokens 4..7
    s = token_scores(per_grid, n)
    assert s.shape == (20,)
    # tokens 4..7 are covered by patch 1 of grid 0 and by one patch of each other grid
    assert np.allclose(s[4:8], 1.0) and np.allclose(np.delete(s, range(4, 8)), 0.0)
    # token 0 is covered by grid 0 only, tokens 1-3 by grids 0 and 50, token 4 by grid 50 only, 5-7 by none
    s2 = token_scores({0: np.array([2.0]), 50: np.array([4.0])}, 400)
    assert s2[0] == 2.0 and np.allclose(s2[1:4], 3.0) and s2[4] == 4.0 and np.isnan(s2[5:]).all()


def test_patch_spectra_of_a_sine():
    t = np.arange(1000) / 100
    x = torch.from_numpy(np.sin(2 * np.pi * 10 * t).astype(np.float32))
    f = patch_spectra(x)
    assert f.shape == (5, 37)
    bins = f[:, 6:36]  # 1 Hz bins 0.5-30.5: bin k covers k + 0.5 .. k + 1.5 Hz
    assert (bins.argmax(1) == 9).all()  # 9.5-10.5 Hz
    bands = f[:, :6]
    assert (bands.argmax(1) == 3).all()  # alpha


def test_spectral_twin_codes_and_r2():
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(3000, 37, generator=g) @ torch.randn(37, 37, generator=g)
    tw = SpectralTwin.fit(feats, n_codes=16, levels=4, iters=10)
    c = tw.codes(feats)
    assert c.shape == (3000, 4) and int(c.max()) < 16
    r2 = [tw.residual_r2(feats, l) for l in (1, 2, 4)]
    assert 0 < r2[0] < r2[1] < r2[2] < 1
    tw2 = SpectralTwin.from_state(tw.state_dict())
    assert torch.equal(tw2.codes(feats, 2), c[:, :2])
