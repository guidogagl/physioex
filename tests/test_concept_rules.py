import numpy as np

from physioex.explain import concept_rules as CR

T = CR.T


def _night(n_ep, k=0):
    return np.full(n_ep * T, k, np.int32)


def test_event_runs_merge_and_night_split():
    night = np.r_[_night(1, 0), _night(1, 1)]
    m = np.zeros(2 * T, bool)
    m[[3, 5, 6, T - 1, T]] = True  # 3 and 5 merge across one gap; T-1 | T split by the night
    runs = CR.event_runs(m, night, gap=1)
    assert runs == [(3, 7), (T - 1, T), (T, T + 1)]


def test_kc_low_swa_isolated_and_n3_high_swa():
    night = _night(2)
    word = np.zeros(2 * T, bool)
    word[20:22] = True            # epoch 0: isolated transient
    word[T + 20:T + 22] = True    # epoch 1: same transient
    swa = np.array([0.05, 0.40])
    alpha = np.array([0.1, 0.1])
    o = CR.kc_n3_rule(word, night, swa, alpha)
    assert o["kc"][20:22].all() and not o["kc"][T:].any()
    assert o["n3"][T + 20:T + 22].all() and not o["n3"][:T].any()


def test_kc_needs_isolation_and_no_wake_alpha():
    night = _night(2)
    word = np.zeros(2 * T, bool)
    word[[10, 11, 14, 15]] = True  # two events 3 tokens apart (gap 2 > merge gap): not isolated
    word[T + 30:T + 32] = True     # isolated, but in an alpha (W) epoch
    o = CR.kc_n3_rule(word, night, np.array([0.05, 0.05]), np.array([0.1, 0.8]))
    assert not o["kc"].any()
    o = CR.kc_n3_rule(word, night, np.array([0.05, 0.05]), np.array([0.1, 0.8]), use_iso=False, use_alpha=False)
    assert o["kc"][[10, 14, T + 30]].all()


def test_event_agreement():
    night = _night(1)
    det, ref = np.zeros(T, bool), np.zeros(T, bool)
    det[[2, 10, 30]] = True
    ref[[2, 3, 30, 50]] = True
    a = CR.event_agreement(det, ref, night)
    assert a["n_det"] == 3 and a["n_ref"] == 3
    assert np.isclose(a["purity"], 2 / 3) and np.isclose(a["coverage"], 2 / 3)


def test_permuted_predicate_lowers_agreement():
    rng = np.random.default_rng(0)
    n_ep = 200
    night = _night(n_ep)
    swa = np.where(rng.random(n_ep) < 0.5, 0.05, 0.5)
    word = np.zeros(n_ep * T, bool)
    word[np.arange(n_ep) * T + 30] = True  # one isolated transient per epoch
    truth = CR.kc_n3_rule(word, night, swa, np.zeros(n_ep))["kc"]
    f_true = CR.event_agreement(CR.kc_n3_rule(word, night, swa, np.zeros(n_ep))["kc"], truth, night)["f1"]
    perm = CR.permute_epochs(swa, night[::T], rng)
    f_perm = CR.event_agreement(CR.kc_n3_rule(word, night, perm, np.zeros(n_ep))["kc"], truth, night)["f1"]
    assert f_true == 1.0 and f_perm < 0.7


def test_shift_keeps_events_and_bootstrap_ci():
    rng = np.random.default_rng(1)
    night = np.r_[_night(10, 0), _night(10, 1)]
    m = np.zeros(len(night), bool)
    m[::37] = True
    s = CR.shift_within_nights(m, night, rng)
    for k in (0, 1):
        assert s[night == k].sum() == m[night == k].sum()
    counts = CR.per_unit_counts(m, m, night)
    lo, hi = CR.bootstrap_f1(counts, n_boot=50)
    assert lo == hi == 1.0
