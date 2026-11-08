"""Tests of the rationale probes (concept_rationale): context features, in-fold lift, matched test, twins,
planted P x E, epoch level."""
import numpy as np

from physioex.train.bin import concept_rationale as cr

T = cr.T


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


def test_rel_back_sees_only_the_past():
    night = np.zeros(100, np.int32)
    F0 = np.zeros(100, np.float32)
    F0[50:] = 2.0  # sustained rise (arousal-like)
    rb = cr.rel_back(F0, night)
    assert rb[49] == 0 and rb[50] == 2.0 and rb[80] == 0  # change vs the 10 s before, then adapted


def test_context_does_not_cross_nights():
    night = np.repeat([0, 1], 50).astype(np.int32)
    F0 = np.where(night == 1, 5.0, 0.0).astype(np.float32)
    assert np.allclose(cr.rel_bg(F0, night), 0) and np.allclose(cr.iso(F0, night), 0)
    P = np.repeat(np.arange(100, dtype=np.float32)[:, None], 2, 1)
    c = cr.ctx_mean(P, night, 2)
    assert np.isclose(c[10, 0], (8 + 9 + 11 + 12) / 4)  # token excluded
    assert np.isclose(c[49, 0], (47 + 48) / 2)  # night edge: no token of night 1
    assert np.isclose(c[50, 0], (51 + 52) / 2)
    b = cr.ctx_mean(P, night, 3, back=True)
    assert np.isclose(b[10, 0], 8) and np.isclose(b[51, 0], 50) and np.isclose(b[50, 0], 50)


def test_spec_windows_band_and_alignment():
    fs = 100
    t = np.arange(4 * 30 * fs) / fs
    x = np.sin(2 * np.pi * 12 * t) * (t > 60)  # sigma/alpha burst in the second half
    S = cr.spec_windows(x.reshape(4, -1), fs)
    assert S.shape == (4 * T, 7)
    late, early = S[-10], S[10]
    names = list(cr.SPEC_WIN)
    assert late[names.index("sigma")] > late[names.index("so")] + 5
    assert late[names.index("sigma")] > early[names.index("sigma")] + 5


def test_epoch_perm_same_stage_other_epoch():
    y = np.repeat(np.array([2, 2, 3, 2, 3, 2], np.int8), T)
    night = np.zeros(len(y), np.int32)
    p = cr.epoch_perm(y, night, np.random.default_rng(0))
    assert (y[p] == y).all() and (p // T != np.arange(len(y)) // T).all() and (p % T == np.arange(len(y)) % T).all()


def test_lift_feature_unseen_keys_zero():
    k = np.array([0, 0, 0, 1, 1, 1])
    r = np.array([1, 1, 0, 0, 0, 0], bool)
    f = cr.lift_feature(k, r, np.array([0, 1, 7]), prior=0.0)
    assert f[0] > 0 > f[1] and f[2] == 0


def _toy(seed=0, n_groups=5, n_ep=60, n_nights=2):
    """Event = large token (P) AND epoch of type 1 (E): the amplitude alone does not separate inside a cell."""
    rng = np.random.default_rng(seed)
    n = n_groups * n_nights * n_ep * T
    e = np.repeat(rng.integers(0, 2, n // T), T)
    p = rng.normal(size=n).astype(np.float32)
    r = (p > 1.0) & (e == 1)
    night = np.repeat(np.arange(n_groups * n_nights), n_ep * T).astype(np.int32)
    gname = np.array([f"s{g}" for g in night // n_nights])
    F = np.column_stack([p, p] + [rng.normal(size=n)] * 7).astype(np.float32)
    D = {"P": np.column_stack([p, rng.normal(size=n)]).astype(np.float32), "F": F, "S": F[:, 2:].copy(),
         "y": np.full(n, 2, np.int8), "night": night, "gname": gname, "set": np.full(n, "pool"),
         "logscale": np.zeros(n, np.float32), "Eq": np.column_stack([e, 1 - e]).astype(np.float32) + 0.01 * rng.normal(size=(n, 2)).astype(np.float32),
         "refs": {"toy": (r, np.ones(n, bool))}}
    cr.finish(D)
    return D, p, e, r


def test_planted_p_by_e_wins_only_with_e():
    D, p, e, r = _toy()
    spec = {"term": "toy", "mod": "eeg", "ref": "toy", "stages": [2], "word_variants": [["P"], ["P", "Eq"], ["P", "bogus"]],
            "baselines": ["abs_amp", "spec_tok"], "matched_bins": ["abs"]}
    out = cr.run_tok(D, spec, n_null=2)
    v = {"+".join(x["inputs"]): x for x in out["variants"]}
    assert out["skipped_inputs"] == ["bogus"]
    # P alone ~ the amplitude left inside a quintile: no better than the amplitude baseline
    assert v["P"]["vs_baseline"]["mean"] < 0.05
    assert v["P+Eq"]["matched"]["mean"] > 0.3
    assert v["P+Eq"]["vs_baseline"]["p_perm"] < 0.01 and v["P+Eq"]["vs_baseline"]["sign_ok"]
    assert v["P+Eq"]["added"]["Eq"]["mean"] > 0.2
    tw = v["P+Eq"]["vs_twins"]["twin_ep"]  # the epoch spectrum does not know the planted epoch type
    assert tw["mean"] > 0.2
    assert out["null_matched"]["inputs"] == ["P", "Eq"] and max(out["null_matched"]["values"]) < 0.15


def test_context_twin_and_candidates():
    D, p, e, r = _toy(1)
    # event = token whose 2 neighbours are large: the context helps, the permuted context does not
    nb = np.r_[p[1:], 0] + np.r_[0, p[:-1]]
    r2 = nb > 1.5
    D["refs"]["nb"] = (r2, np.ones(len(r2), bool))
    spec = {"term": "nb", "mod": "eeg", "ref": "nb", "stages": [2], "word_variants": [["P"], ["P", "P_ctx±1"]],
            "baselines": ["abs_amp"], "matched_bins": ["abs"]}
    v = {"+".join(x["inputs"]): x for x in cr.run_tok(D, spec, n_null=0)["variants"]}
    assert v["P+P_ctx±1"]["vs_twins"]["twin_perm"]["mean"] > 0.3
    spec_c = dict(spec, candidates={"var": "abs", "q": 0.5}, word_variants=[["P", "P_ctx±1"]])
    out = cr.run_tok(D, spec_c, n_null=0)
    assert out["variants"][0]["auc_nights"] > 0.8 and out["n_tok"] < 0.55 * len(r2)


def test_oscillation_needs_energy_terms():
    D, p, e, r = _toy(4)
    q = D["P"][:, 1]
    D["refs"]["osc"] = (np.abs(q) > 1.5, np.ones(len(q), bool))  # sign-symmetric: invisible to a linear probe
    spec = {"term": "osc", "mod": "eeg", "ref": "osc", "stages": [2], "word_variants": [["P"]],
            "baselines": ["abs_amp"], "matched_bins": ["abs"]}
    v = cr.run_tok(D, spec, n_null=0)["variants"][0]
    assert v["matched"]["mean"] > 0.5


def test_codes_enter_through_in_fold_lift():
    D, p, e, r = _toy(2)
    D["codes_E"] = np.column_stack([e * 3 + np.random.default_rng(2).integers(0, 3, len(e)), np.zeros(len(e), int)])
    spec = {"term": "toy", "mod": "eeg", "ref": "toy", "stages": [2], "word_variants": [["P", "codes_E"]],
            "baselines": ["abs_amp"], "matched_bins": ["abs"]}
    v = cr.run_tok(D, spec, n_null=0)["variants"][0]
    assert v["matched"]["mean"] > 0.3


def test_epoch_level_partial_spearman():
    D, p, e, r = _toy(3)
    spec = {"term": "frac", "mod": "eeg", "ref": "toy", "level": "epoch", "stages": [2],
            "word_variants": [["Eq"], ["P_ep"]], "baselines": ["abs_ep"]}
    out = cr.run_epoch(D, spec, n_null=2)
    v = {"+".join(x["inputs"]): x for x in out["variants"]}
    assert v["Eq"]["rho"]["mean"] > 0.5 and v["Eq"]["vs_baseline"]["p_perm"] < 0.01
    assert out["null_rho"]["p"] <= 1 / 3
    t, _ = cr.epoch_target(D, "post:toy:1")
    assert t[0] == 0 and set(np.unique(t)) <= {0.0, 1.0}


def test_onset_takes_first_two_tokens_per_run_and_night():
    night = np.array([0] * 6 + [1] * 4, np.int32)
    r = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 0], bool)  # night 1 starts inside a run
    D = {"night": night, "refs": {"a": (r, np.ones(len(r), bool))}}
    o, _ = cr.ref_mask(D, "onset:a")
    assert o.tolist() == [0, 1, 1, 0, 0, 0, 1, 1, 0, 0]


def test_twin_n_and_spec_nb():
    D, p, e, r = _toy(5)
    assert D["spec_nb"].shape == (len(p), 3 * D["spec_ep"].shape[1])  # label-free: mean, sd, step
    assert D["spec_nb_lab"].shape == (len(p), D["spec_ep"].shape[1] + 5)
    tw = cr.twins([["P", "Eq", "Nq"]], False)
    tn = [x for x in tw if x[1] == "twin_N"]
    assert tn and "Nq" not in tn[0][2] and "spec_nb" in tn[0][2] and "Eq" in tn[0][2] and "spec_ep" not in tn[0][2]
    tn = [x for x in cr.twins([["P", "Nq"]], False) if x[1] == "twin_N"]
    assert "spec_ep" in tn[0][2]  # N alone also carries the centre epoch


def test_forward_context_and_stage_label_twins():
    night = np.repeat([0, 1], 50).astype(np.int32)
    P = np.repeat(np.arange(100, dtype=np.float32)[:, None], 2, 1)
    f = cr.ctx_mean(P, night, 3, fwd=True)
    assert np.isclose(f[10, 0], 12) and np.isclose(f[49, 0], 49) and np.isclose(f[48, 0], 49)
    assert cr.CTX.fullmatch("P_fwd6") and cr.CTX.fullmatch("P_fwd6_perm")
    c = cr.ceiling_inputs({"baselines": ["abs_amp"], "stages": [2], "ref": "a"}, [["P", "P_fwd6", "P_back20"]])
    assert "spec_fwd6" in c and "spec_back20" in c
    assert not [x for x in cr.twins([["P", "P_ctx±1"]], True, stage_label=True) if x[1] == "twin_perm"]


def test_ceiling_reads_same_context_as_words():
    spec = {"baselines": ["abs_amp"], "stages": [2, 3], "ref": "toy"}
    c = cr.ceiling_inputs(spec, [["P", "P_ctx±2"], ["P", "P_back20", "Nq"]])
    assert c[:1] == ["abs_amp"] and {"spec_ctx2", "spec_back20", "spec_ep", "spec_nb", "stage"} <= set(c)


def test_match_across_stages_pools_cells():
    D, p, e, r = _toy(6)
    D["y"] = np.where(e == 1, 4, 0).astype(np.int8)  # the stage separates the event: within stage it is the amplitude
    spec = {"term": "toy", "mod": "eeg", "ref": "toy", "stages": [0, 4], "word_variants": [["P", "Eq"]],
            "baselines": ["abs_amp"], "matched_bins": ["abs"], "match_across_stages": True}
    v = cr.run_tok(D, spec, n_null=0)["variants"][0]
    assert v["matched"]["mean"] > 0.3 and v["vs_baseline"]["mean"] > 0.1


def test_epoch_without_controls():
    D, p, e, r = _toy(7)
    spec = {"term": "frac", "mod": "eeg", "ref": "toy", "level": "epoch", "stages": [2], "controls": [],
            "word_variants": [["Eq"]], "baselines": ["abs_ep"]}
    out = cr.run_epoch(D, spec, n_null=0)
    assert out["variants"][0]["rho"]["mean"] > 0.5
