"""Directional words on synthetic coordinates (07.11.2026): a direction planted on the events is the
top word and beats the rotation null; whole-row scoring keeps the 60-token rows."""
import numpy as np

from physioex.train.bin import concept_words as W


def _setup(seed=0, n_rows=400, D=8):
    rng = np.random.default_rng(seed)
    n = n_rows * W.T
    Z = rng.standard_t(8, (n, D)).astype(np.float32)
    y = np.repeat(rng.choice([1, 2], n_rows), W.T).astype(np.int8)
    ref = np.zeros(n, bool)
    for s in rng.choice(np.flatnonzero(y == 2), 150, replace=False):
        ref[s:s + 3] = True
    ref &= y == 2
    Z[ref, 3] += 3.0
    night = np.repeat(np.arange(n_rows) // 40, W.T)
    grp = np.array(["a", "b", "c", "d"])[night % 4]
    bases = {"pq1": {"groups": {"kurt4": [0, 1, 2, 3], "all": list(range(D))}}}
    refs = {"spindle_rule": (ref, np.ones(n, bool))}
    return {"pq1": Z}, bases, y, grp, night, refs


def test_planted_direction_is_top_word():
    Zs, bases, y, grp, night, refs = _setup()
    r = W.scan_concept("spindle", Zs, bases, y, grp, night, refs, n_rot=3, n_shift=2)
    assert r["top"][0]["word"] in ("pq1:d3+", "pq1:|d3|", "pq1:E[kurt4]~", "pq1:E[kurt4]")
    assert any(t["word"] == "pq1:d3+" for t in r["top"])
    assert r["loo_ap"] > 0.5 > r["null_rotation"]["q95"]
    assert r["loo_ap"] > r["null_shift"]["q95"]


def test_word_score_rows_and_names():
    Z = np.arange(2 * W.T * 2, dtype=np.float32).reshape(-1, 2)
    names = W.word_names(2, {"all": [0, 1]}, "pq1")
    assert len(names) == 2 * 3 + 3 and W.word_dirs("pq1:|d1|") == ("pq1", 1)
    e = W.word_score(Z, "pq1:E[all]~", {"all": [0, 1]})
    raw = (Z ** 2).sum(1)
    assert e[W.T - 1] == (raw[W.T - 2] + 2 * raw[W.T - 1]) / 3  # row edge repeated, no leak into the next row


def test_rows_rel_matches_lexicon_rel_change():
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "vql", pathlib.Path(W.__file__).parents[2] / "models" / "vq_lexicon.py")
    vql = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vql)
    e = np.random.default_rng(1).random(3 * W.T) + 0.1
    np.testing.assert_allclose(W._rows_rel(e), vql.rel_change(e, 10, 1, W.T))


def test_swap_tokens_uses_same_stage_donors_without_the_word():
    rng = np.random.default_rng(0)
    y = np.array([2, 2, 3, 2])
    P = np.stack([np.full((W.T, 2), e, np.float32) for e in range(4)])
    fire = np.zeros(4 * W.T, bool)
    fire[5:8] = True  # epoch 0
    Q = W.swap_tokens(P, y, fire, np.flatnonzero(fire), rng)
    got = Q.reshape(-1, 2)[5:8, 0]
    assert set(got) <= {1.0, 3.0} and (Q.reshape(-1, 2)[~fire] == P.reshape(-1, 2)[~fire]).all()


def test_fidelity_detects_a_word_the_stager_reads():
    """Fake stager: p(N2) = share of the epoch's tokens with d0 > 2; swapping the word must lower it more
    than swapping random tokens."""
    rng = np.random.default_rng(1)
    n = 40
    y = np.full(n, 2)
    P = rng.normal(0, 1, (n, W.T, 2)).astype(np.float32)
    P[::2, 10:16, 0] += 4
    bases = {"pq1": {"mean": np.zeros(2, np.float32), "unmix": np.eye(2, dtype=np.float32), "groups": {"all": [0, 1]}}}

    def logp(L):
        f = (L["eeg"]["P"][:, :, 0].astype(np.float32) > 2).mean(1)
        p = np.stack([0.1 + 0 * f, 0.1 + 0 * f, 0.2 + f, 0.3 - f / 2, 0.3 - f / 2], 1)
        return np.log(p / p.sum(1, keepdims=True))

    lat = {"eeg": {"P": P.astype(np.float16), "y": y}}
    r = W.fidelity_night(lat, logp, ["eeg"], bases, "pq1:d0+", "eeg", (2,), (2,), 0.05, 5, rng)
    assert r["dp"] > 0.02 and r["dp"] > max(r["ctrl_dp"])
