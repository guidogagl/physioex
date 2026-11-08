"""Tests of the rationale probes (concept_rationale): context features, in-fold lift, matched test, planted P x E."""
import numpy as np

from physioex.train.bin import concept_rationale as cr


def test_rel_bg_and_iso_separate_isolated_wave_from_train():
    night = np.zeros(400, np.int32)
    F0 = np.zeros(400, np.float32)
    F0[100] = 3.0  # isolated big wave
    F0[300:310] = 3.0  # train of big waves
    rb, io = cr.rel_bg(F0, night), cr.iso(F0, night)
    assert rb[100] > 2.9 and io[100] < -2.9  # stands out of its background and of its neighbours
    assert io[305] >= 0  # inside the train the neighbours are as large
    assert rb[305] > 2.9  # the median over +-10 s still sees a quiet background
    F0[290:330] = 3.0  # a long train: the background itself is large
    assert abs(cr.rel_bg(F0, night)[310]) < 1e-6


def test_context_does_not_cross_nights():
    night = np.repeat([0, 1], 50).astype(np.int32)
    F0 = np.where(night == 1, 5.0, 0.0).astype(np.float32)
    assert np.allclose(cr.rel_bg(F0, night), 0) and np.allclose(cr.iso(F0, night), 0)
    P = np.repeat(np.arange(100, dtype=np.float32)[:, None], 2, 1)
    c = cr.ctx_mean(P, night, 2)
    assert np.isclose(c[10, 0], (8 + 9 + 11 + 12) / 4)  # token excluded
    assert np.isclose(c[49, 0], (47 + 48) / 2)  # night edge: no token of night 1
    assert np.isclose(c[50, 0], (51 + 52) / 2)


def test_lift_feature_unseen_keys_zero():
    k = np.array([0, 0, 0, 1, 1, 1])
    r = np.array([1, 1, 0, 0, 0, 0], bool)
    f = cr.lift_feature(k, r, np.array([0, 1, 7]), prior=0.0)
    assert f[0] > 0 > f[1] and f[2] == 0


def _planted(seed=0, n_groups=6, n=6000):
    """Event = large token (P) AND epoch of type 1 (E): the amplitude alone does not separate inside a bin."""
    rng = np.random.default_rng(seed)
    key = np.repeat([f"s{g}" for g in range(n_groups)], n)
    p = rng.normal(size=len(key)).astype(np.float32)
    e = rng.integers(0, 2, len(key))
    r = (p > 1.0) & (e == 1)
    y = np.full(len(key), 2, np.int8)
    return key, p, e, r, y


def test_planted_p_by_e_wins_only_with_e():
    key, p, e, r, y = _planted()
    groups = sorted(set(key))
    M = p[:, None]  # matched on the amplitude
    s_p = cr.fold_probe(p[:, None], [], r, key, groups)
    s_pe = cr.fold_probe(np.column_stack([p, e, p * e]), [], r, key, groups)
    mp, cov = cr.matched(s_p, r, key, y, M)
    mpe, _ = cr.matched(s_pe, r, key, y, M)
    assert cov > 0.5
    assert np.mean(list(mp.values())) < 0.15  # P alone ~ the matched amplitude
    assert np.mean(list(mpe.values())) > 0.5
    assert cr.wilcoxon_gt0([mpe[g] - mp[g] for g in mpe]) < 0.05


def test_codes_enter_through_in_fold_lift():
    key, p, e, r, y = _planted(1)
    groups = sorted(set(key))
    codes_e = np.column_stack([e * 3 + np.random.default_rng(2).integers(0, 3, len(e)), np.zeros(len(e), int)])
    s = cr.fold_probe(p[:, None], [codes_e], r, key, groups)
    m, _ = cr.matched(s, r, key, y, p[:, None])
    assert np.mean(list(m.values())) > 0.3


def test_run_spec_end_to_end():
    key, p, e, r, y = _planted(3, n_groups=5, n=3000)
    n = len(key)
    night = np.repeat(np.arange(5), 3000).astype(np.int32)
    F = np.column_stack([p, p] + [np.random.default_rng(4).normal(size=n)] * 7).astype(np.float32)
    D = {"P": np.column_stack([p, np.zeros(n)]).astype(np.float32), "F": F, "y": y, "night": night,
         "gname": key.astype(str), "Eq": np.column_stack([e, 1 - e]).astype(np.float32),
         "rel_bg": cr.rel_bg(F[:, 0], night), "iso": cr.iso(F[:, 0], night),
         "refs": {"toy": (r, np.ones(n, bool))}}
    spec = {"term": "toy", "ref": "toy", "stages": [2], "word_variants": [["P"], ["P", "Eq"], ["P", "bogus"]],
            "baselines": ["abs_amp", "stage"], "matched_bins": ["abs"]}
    out = cr.run_spec(D, spec, n_null=2)
    v = {"+".join(x["inputs"]): x for x in out["variants"]}
    assert out["skipped_inputs"] == ["bogus"]
    assert v["P+Eq"]["matched"]["mean"] > v["P"]["matched"]["mean"] + 0.1
    assert len(out["null_matched"]["values"]) == 2 and out["null_matched"]["inputs"] == ["P", "Eq"]
