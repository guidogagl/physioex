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
    grp = np.array(list("abcdefgh"))[night % 8]
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


def _lexicon_mod():
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("vql2", pathlib.Path(W.__file__).parents[2] / "models" / "vq_lexicon.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_frozen_words_fire_as_scan_scores(tmp_path):
    """Every word kind, converted to a Lexicon word, fires exactly where the scan score passes its
    threshold; a multimodal lexicon reads each modality's P; a single-array call still works."""
    vql = _lexicon_mod()
    rng = np.random.default_rng(2)
    P = {"eeg": rng.normal(0, 1, (6, W.T, 4)).astype(np.float32), "emg": rng.normal(0, 1, (6, W.T, 2)).astype(np.float32)}
    bases = {}
    for m, d in (("eeg", 4), ("emg", 2)):
        bases[m] = {"pq1": {"mean": np.zeros(d, np.float32), "unmix": rng.normal(0, 1, (d, d)).astype(np.float32),
                            "mix": np.eye(d, dtype=np.float32), "groups": {"all": list(range(d)), "p": [0, 1]}},
                    "pq3": {"mean": np.zeros(3 * d, np.float32), "unmix": rng.normal(0, 1, (d, 3 * d)).astype(np.float32),
                            "mix": np.zeros((3 * d, d), np.float32), "groups": {"all": list(range(d))}}}
    cases = [("eeg", "pq1:d1+"), ("eeg", "pq1:d2-"), ("eeg", "pq3:|d0|"), ("eeg", "pq3:E[all]~"), ("eeg", "pq1:E[p]rel"),
             ("emg", "pq1:E[p]"), ("emg", "pq1:d0-")]
    words, lb = [], {}
    for i, (m, w) in enumerate(cases):
        rep = w.split(":")[0]
        s = W.word_score(W.transform(P[m], bases[m][rep], rep), w, bases[m][rep]["groups"])
        thr = float(np.quantile(s, 0.9))
        words.append((W.lexicon_word(f"w{i}", w, bases[m], m, thr, {}), s >= thr))
        lb[f"{m}_{rep}"] = {"rep": rep, "mod": m, **{k: bases[m][rep][k] for k in ("mean", "unmix", "mix")}}
    lex = vql.Lexicon(lb, [w for w, _ in words])
    lex.save(str(tmp_path))
    fired = vql.Lexicon.load(str(tmp_path)).fire(P, continuous=False)
    for w, ref in words:
        assert (fired[w["name"]] == ref).all(), w["word"]
    only_eeg = vql.Lexicon({k: v for k, v in lb.items() if k.startswith("eeg")}, [w for w, _ in words if w["basis"].startswith("eeg")])
    single = only_eeg.fire(P["eeg"], continuous=False)
    assert all((single[w["name"]] == r).all() for w, r in words if w["basis"].startswith("eeg"))
    assert set(lex.fire({"eeg": P["eeg"]}, continuous=False)) == {w["name"] for w, _ in words if w["basis"].startswith("eeg")}


def test_event_f1_matching():
    ref = np.zeros(40, bool); ref[5:8] = True; ref[20:23] = True
    pred = np.zeros(40, bool); pred[8] = True; pred[30] = True  # first adjacent (tolerance 1 token), second false
    r = W.event_f1(pred, ref)
    assert r["n_pred"] == 2 and r["n_ref"] == 2 and abs(r["f1"] - 0.5) < 1e-9


def _codes_setup(seed=0, n_rows=400, L=2, n_codes=64):
    """Random P codes; the events of stage 2 use codes 5 and 6 of level 0 (a planted family)."""
    Zs, bases, y, grp, night, refs = _setup(seed, n_rows)
    rng = np.random.default_rng(seed + 1)
    C = rng.integers(0, n_codes, (len(y), L)).astype(np.int16)
    ref = refs["spindle_rule"][0]
    C[ref, 0] = rng.choice([5, 6], ref.sum())
    return Zs, bases, y, grp, night, refs, C, n_codes


def test_code_word_finds_planted_family_and_beats_nulls():
    Zs, bases, y, grp, night, refs, C, n_codes = _codes_setup()
    r = W.scan_concept("spindle", Zs, bases, y, grp, night, refs, n_rot=2, n_shift=19, codes=C, n_codes=n_codes)
    c = r["code"]
    assert c["loo_ap"] > 0.5 > c["null_shift"]["q95"] and c["null_shift"]["p"] <= 0.05
    assert c["k"] <= 10 and {(f["level"], f["codes"][0]) for f in c["family"][:2]} == {(0, 5), (0, 6)}
    assert c["stability_split"] >= 0.7 and c["p_vs_base"] < 0.05
    cand = W.candidates(r)["code"]
    assert W.gates("spindle", cand)["verdict"] == "valida"


def test_stage_only_code_is_rejected():
    Zs, bases, y, grp, night, refs, C, n_codes = _codes_setup()
    ref = refs["spindle_rule"][0]
    C[:, 0] = np.where(y == 2, 5, 9)  # the code says the stage, not the event
    C[ref, 0] = 5
    r = W.scan_concept("spindle", Zs, bases, y, grp, night, refs, n_rot=2, n_shift=19, codes=C, n_codes=n_codes)
    cand = W.candidates(r)["code"]
    assert W.gates("spindle", cand)["verdict"] in ("bocciata", "ampiezza")


def test_amplitude_baseline_catches_an_energy_reference():
    """When the token RMS of the raw signal predicts the reference as well as the word, the word only
    says amplitude, whatever its AP."""
    Zs, bases, y, grp, night, refs = _setup()
    ref = refs["spindle_rule"][0]
    rng = np.random.default_rng(5)
    F = rng.normal(0, 1, (len(y), len(W.FEATS))).astype(np.float32)
    F[:, 0] += 6 * ref
    F[:, 1] += 6 * ref
    r = W.scan_concept("spindle", Zs, bases, y, grp, night, refs, n_rot=2, n_shift=19, F=F)
    assert r["baseline"]["loo_ap"] > 0.95 and r["ceiling"]["loo_ap"] > 0.95
    v = W.gates("spindle", W.candidates(r)["ica"])
    assert v["verdict"] == "ampiezza" and v["p_vs_base"] > 0.05


def test_planted_ica_word_beats_the_baselines():
    Zs, bases, y, grp, night, refs = _setup()
    F = np.random.default_rng(6).normal(0, 1, (len(y), len(W.FEATS))).astype(np.float32)
    r = W.scan_concept("spindle", Zs, bases, y, grp, night, refs, n_rot=2, n_shift=19, F=F)
    c = W.candidates(r)["ica"]
    assert c["null_p"] <= 0.05 and c["p_vs_base"] < 0.05 and c["margin_nap"] > 0.3 and c["stable"]
    assert W.gates("spindle", c)["verdict"] == "valida" and r["ica"]["retention"] > 1


def test_shift_within_runs_keeps_the_stage():
    rng = np.random.default_rng(0)
    y = np.repeat([1, 2, 2, 3], 30)
    night = np.repeat([0, 0, 1, 1], 30)
    ref = rng.random(120) < 0.3
    s = W.shift_within_runs(ref, night, y, rng)
    for a in range(0, 120, 30):  # every (night, stage) run keeps its count
        assert s[a:a + 30].sum() == ref[a:a + 30].sum()


def test_gates_ignore_the_stager_and_use_q():
    c = {"null_p": 0.01, "p_vs_base": 0.001, "margin_nap": 0.3, "amp": False, "stable": True, "within": 5.0}
    assert W.gates("spindle", c)["verdict"] == "valida" and "L3" not in W.gates("spindle", c)
    assert W.gates("spindle", c, q=0.2)["verdict"] == "ampiezza"
    assert W.gates("spindle", {**c, "null_p": 0.2})["verdict"] == "bocciata"
    assert W.gates("spindle", {**c, "margin_nap": 0.02})["verdict"] == "debole"


def test_bh_and_choice():
    q = W.bh([0.01, 0.04, 0.03, 0.5])
    np.testing.assert_allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    ica, code = {"loo_ap": 0.5}, {"loo_ap": 0.52}
    assert W.choose({"ica": ica, "code": code}, 0.2) == "ica" and W.choose({"ica": ica, "code": code}, 0.01) == "code"


def test_compact_word_lexicon_round_trip(tmp_path):
    vql = _lexicon_mod()
    rng = np.random.default_rng(4)
    n_codes = 16
    C = rng.integers(0, n_codes, (5, W.T, 3)).astype(np.int16)
    K = W.prefix_keys(C.reshape(-1, 3), n_codes)
    ref = (C.reshape(-1, 3)[:, 0] == 3) | ((C.reshape(-1, 3)[:, 0] == 7) & (C.reshape(-1, 3)[:, 1] == 2))
    word = W.compact_word(W.prefix_table(K, ref, min_n=5), 5)
    assert 0 in word and 3 in word[0][0]
    for win in (1, 3):
        s = W.compact_score(K, word, win)
        thr = float(np.quantile(s, 0.8)) - 1e-6
        lex = vql.Lexicon({"eeg_code_x": {"rep": "code_prefix", "mod": "eeg",
                                          **{f"pref_keys{l}": word[l][0] for l in word},
                                          **{f"pref_w{l}": word[l][1] for l in word}}},
                          [{"name": "x", "event": "x", "basis": "eeg_code_x", "kind": "code_prefix", "n_codes": n_codes,
                            "win": win, "side": 1, "threshold": thr}])
        lex.save(str(tmp_path))
        back = vql.Lexicon.load(str(tmp_path))
        np.testing.assert_array_equal(back.fire(None, continuous=False, codes={"eeg": C})["x"], s >= thr)


def test_token_features_shapes_and_bands():
    fs = 100
    t = np.arange(4 * 3000) / fs
    x = (np.sin(2 * np.pi * 13.5 * t) * (t > 60)).reshape(4, 3000)  # sigma in the last two epochs
    F = W.token_features(x, fs)
    assert F.shape == (4 * W.T, len(W.FEATS))
    sig = W.FEATS.index("sigma")
    assert F[3 * W.T:, sig].mean() > F[:W.T, sig].mean() + 5


def test_code_word_lexicon_round_trip(tmp_path):
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "vql", pathlib.Path(W.__file__).parents[2] / "models" / "vq_lexicon.py")
    vql = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vql)
    rng = np.random.default_rng(3)
    C = rng.integers(0, 16, (4, W.T, 2)).astype(np.int16)
    lift = rng.random((2, 16)) + 0.5
    for win in (1, 3):
        s = W.code_score(C.reshape(-1, 2), lift, win)
        thr = float(np.quantile(s, 0.8))
        lex = vql.Lexicon({"eeg_code_x": {"rep": "code", "mod": "eeg", "loglift": np.log(lift)}},
                          [{"name": "x", "event": "x", "basis": "eeg_code_x", "kind": "code", "win": win,
                            "side": 1, "threshold": thr}])
        lex.save(str(tmp_path))
        back = vql.Lexicon.load(str(tmp_path))
        np.testing.assert_array_equal(back.fire(None, continuous=False, codes={"eeg": C})["x"], s >= thr)
        assert back.fire(np.zeros((4, W.T, 3), np.float32), continuous=False) == {}  # no codes: no code word
