"""Concept scan on synthetic codes (07.11.2026): a code planted on the events is found with a high
lift; a code planted on the whole stage is not taken for the concept (stratified by stage)."""
import numpy as np

from physioex.train.bin import concept_scan as S


def _data(rng, plant_event=True, n_nights=8, n_tok=600):
    codes, y, night, group, ref = [], [], [], [], []
    for g in range(n_nights):
        yy = np.repeat(np.resize(rng.permutation(5), n_tok // 60), 60)  # every stage in every night
        r = np.zeros(n_tok, bool)
        for s in rng.choice(np.flatnonzero(yy == 2), 6):
            r[s:s + 2] = True
        r &= yy == 2
        c = rng.integers(0, 64, (n_tok, 2))
        c[yy == 2, 1] = rng.integers(0, 8, (yy == 2).sum())  # stage code: N2 uses few codes of column 1
        if plant_event:
            c[r, 0] = 63
        codes.append(c)
        y.append(yy)
        night.append(np.full(n_tok, g))
        group.append(np.full(n_tok, g % 4))
        ref.append(r)
    ref = np.concatenate(ref)
    return {"codes": np.concatenate(codes).astype(np.int16), "y": np.concatenate(y), "night": np.concatenate(night),
            "group": np.concatenate(group), "groups": ["a", "b", "c", "d"], "cols": ["P0", "P1"],
            "refs": {"spindle_rule": (ref, np.ones(len(ref), bool))}}


def test_planted_code_found_and_beats_null():
    D = _data(np.random.default_rng(0))
    r = S.scan_concept("spindle", D, 64, n_nulls=5)
    assert {"col": "P0", "code": 63} in r["selected"]
    assert r["f1"] > 0.9 > r["null_f1"]["q95"]
    assert r["describe"]["union"]["lift"] > 5
    assert all(f["f1"] > 0.8 for f in r["loo"]) and r["stability"]["P0:63"] == 1.0


def test_stage_code_not_taken_for_concept():
    D = _data(np.random.default_rng(1), plant_event=False)
    r = S.scan_concept("spindle", D, 64, n_nulls=5)
    # inside N2 the stage codes are everywhere: no lift, F1 not above the null
    assert r["describe"]["union"]["lift"] < 1.5
    assert r["f1"] <= r["null_f1"]["q95"] + 0.05


def test_shift_within_nights_keeps_counts():
    rng = np.random.default_rng(2)
    ref = rng.random(300) < 0.1
    night = np.repeat([0, 1, 2], 100)
    sh = S._shift_within_nights(ref, night, rng)
    for n in range(3):
        assert sh[night == n].sum() == ref[night == n].sum()
