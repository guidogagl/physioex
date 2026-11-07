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
