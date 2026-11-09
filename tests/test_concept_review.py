"""Blind review: stratified pick (8 / 6 / 6 with fill), one PNG per example with a random A/B key, and the
aggregation that opens the key."""
import json
import os

import numpy as np

from physioex.train.bin import concept_review as RV
from tests.test_concept_examples import _toy_dump

T = RV.T


def _strata_dump(tmp, n=60):
    """Toy night: every 3rd epoch a planted KC seen by both; extra model-only and rule-only runs."""
    mk = _toy_dump(tmp, n=n)
    z = dict(np.load(mk))
    lex = z["lex_kc"].copy()
    for e in range(3, n - 3, 6):  # model only
        lex[e * T + 40:e * T + 42] = True
    z["lex_kc"] = lex
    np.savez(mk, **z)
    rp = os.path.join(tmp, "refs", "hmc", "SN001.npz")
    r = dict(np.load(rp))
    m = np.unpackbits(r["k_complex_rule"])[:int(r["_len_k_complex_rule"])].astype(bool)
    for e in range(4, n - 3, 9):  # rule only
        m[e * T + 50:e * T + 52] = True
    np.savez(rp, _len_k_complex_rule=np.array(len(m)), k_complex_rule=np.packbits(m))
    return mk


def test_pick_png_aggregate(tmp_path):
    mk = _strata_dump(str(tmp_path))
    out = tmp_path / "rev"
    RV.main(["pick", "--dump", str(tmp_path), "--masks", f"kc={mk}", "--out", str(out), "--concepts", "kc"])
    sel = json.load(open(out / "selection.json"))["concepts"]["kc"]
    got = {s: sum(r["stratum"] == s for r in sel["examples"]) for s in RV.STRATA}
    assert sum(got.values()) == min(20, sum(sel["candidates"].values()))
    assert got["rule_only"] <= 6 and got["model_only"] <= 6
    z = np.load(out / "kc" / "ex_00.npz")
    assert z["x_eeg"].shape == (3, 3000) and z["rule"].shape == z["model"].shape == (3 * T,)
    r0 = json.loads(str(z["meta"]))
    assert r0["stratum"] == "both" and z["model"][r0["a"]:r0["b"]].any() and z["rule"][r0["a"]:r0["b"]].any()
    RV.main(["png", "--out", str(out)])
    key = json.load(open(out / "key.json"))
    assert len(key) == len(sel["examples"]) and (out / "kc" / "ex_00.png").exists()
    # critics: lens "a" says yes to the rule panel, no to the model panel
    os.makedirs(out / "verdicts")
    ex = []
    for r in sel["examples"]:
        rp = key[f"kc/ex_{r['i']:02d}"]["rule_panel"]
        v = {p: {"verdict": "yes" if p == rp else "no"} for p in "AB"}
        ex.append({"example": r["i"], "panel_A": v["A"], "panel_B": v["B"]})
    for lens in ("a", "b"):
        json.dump({"examples": ex}, open(out / "verdicts" / f"kc_{lens}.json", "w"))
    RV.main(["aggregate", "--out", str(out)])
    res = json.load(open(out / "review.json"))["kc"]
    assert res["rule"]["all"]["support"] == 1.0 and res["model"]["all"]["support"] == 0.0


def test_select_fill_and_prefer():
    rng = np.random.default_rng(0)
    cands = {"both": [(f"n{i}", 0, 1, 2) for i in range(30)], "rule_only": [("r", 0, 1, 2)], "model_only": []}
    picks, short = RV.select(cands, {"n7"}, rng)
    assert short == {"rule_only": 5, "model_only": 6}
    assert len(picks["both"]) == 8 + 11 and picks["both"][0][0] == "n7"
    assert len({c[0] for c in picks["both"]}) == len(picks["both"])


def test_preferred():
    p = {"hpap/X1", "mass_ss01/01-01-0002"}
    assert RV.preferred("hpap_lab_full/X1", p) and RV.preferred("mass_ss01/01-01-0002", p)
    assert not RV.preferred("hpap_lab_full/X2", p) and not RV.preferred("mesa/X1", p)


def test_fleiss():
    assert RV.fleiss([[5, 0, 0], [0, 0, 5], [5, 0, 0]]) == 1.0
