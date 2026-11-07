"""Validation of the rule references (phase C, 08.11.2026): event matching, acceptance, site rates and
the leave-one-excerpt-out tuning of the K-complex rule."""
import numpy as np

from physioex.explain import concept_refs as R
from physioex.train.bin import concept_refs_eval as E

FS = 100.0


def _mask(n, iv):
    m = np.zeros(n, bool)
    for a, b in iv:
        m[a:b] = True
    return m


def test_match_is_one_to_one_at_iou():
    ref = _mask(100, [(10, 14), (30, 32), (60, 70)])
    det = _mask(100, [(11, 15), (31, 33), (40, 42), (60, 61)])  # IoU .6, .33, none, .1
    assert E.night_counts(det, ref) == (2, 4, 3)
    # one long detection covering two references counts once
    assert E.match(E.events(_mask(50, [(0, 20)])), E.events(_mask(50, [(2, 8), (10, 18)])), iou=0.2) == 1


def test_prf_bootstrap_and_accept():
    b = E.bootstrap([(6, 8, 10), (4, 5, 6)], n_boot=200)
    assert abs(b["precision"] - 10 / 13) < 1e-9 and abs(b["recall"] - 10 / 16) < 1e-9
    assert b["ci_f1"][0] <= b["f1"] <= b["ci_f1"][1]
    assert E.accept(0.7, 0.55) == "riferimento" and E.accept(0.7, 0.4) == "indicativo"
    assert E.accept(0.2, 0.9) == "fuori" and E.accept(float("nan"), 0.5) == "non valutabile"


def test_site_rates_flag_a_site_dependent_rule():
    y = np.full(7200 * 2, 2, np.int8)  # 2 h of N2 in tokens
    every = lambda k: _mask(len(y), [(i, i + 2) for i in range(0, len(y), k)])
    nights = [("a", y, {"x": every(100)}), ("b", y, {"x": every(110)}), ("c", y, {"x": every(1000)})]
    r = E.site_rates(nights, rules=("x",))["x"]
    assert r["stage"] == 2 and r["site_dependent"] and 9 < r["max_over_min"] < 10
    r = E.site_rates(nights[:2], rules=("x",))["x"]
    assert not r["site_dependent"]


def _kc(amp, dur=1.0):
    t = np.arange(int(dur * FS)) / FS
    return -amp * np.sin(2 * np.pi * t / dur)  # negative half then positive half


def test_kc_tuning_picks_a_threshold_that_rejects_small_waves():
    rng = np.random.default_rng(0)
    exc = []
    for e in range(4):
        n_ep = 20
        x = rng.normal(0, 5, n_ep * 3000)
        ref = np.zeros(n_ep * R.T, bool)
        for k, s in enumerate(range(500, n_ep * 3000 - 500, 1500)):
            big = k % 2 == 0
            x[s:s + 100] += _kc(120 if big else 5)
            if big:
                ref[s // 50:(s + 100) // 50] = True
        exc.append((f"e{e}", x.reshape(n_ep, 3000), np.arange(n_ep), np.full(n_ep, 2, np.int8), FS,
                    {"vis1": ref, "vis2": ref} if e < 2 else {"vis1": ref}))
    out = E.kc_tuning(exc, grid={"k_mad": (2.0, 15.0), "iso_s": (0.0,)})
    assert out["best_all"]["k_mad"] == 15.0
    assert out["heldout"]["vis2"]["n_nights"] == 2
    assert out["heldout"]["vis1"]["precision"] > 0.8 and out["heldout"]["vis1"]["recall"] > 0.8
