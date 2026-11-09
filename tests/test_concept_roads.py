"""Road C (explanations via the decoded signal, toy codec and a fake stager) and road A (rule-supervised
semantic layer on the toy dump)."""
import numpy as np

from physioex.train.bin import concept_explain_decoded as CE
from physioex.train.bin import concept_rationale as cr
from physioex.train.bin import concept_semlayer as SL
from tests.test_concept_compose import _toy
from tests.test_concept_counterfactual import ToyCodec

T = cr.T


def test_explain_night_toy(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 16
    x = 0.05 * rng.normal(size=(n, 3000)).astype(np.float32)
    kc = np.zeros((n, T), bool)
    t = np.linspace(0, np.pi * 2, 100)
    for e in range(0, n, 2):
        x[e, 1000:1100] += -3.0 * np.sin(t)
        kc[e, 20:22] = True
    np.savez(tmp_path / "a.npz", x_eeg=x, y=np.array([2] * n), P_eeg=rng.normal(size=(n, T, 2)).astype(np.float32),
             logp_eeg=np.log(np.full((n, 5), 0.2, np.float32)))
    np.savez(tmp_path / "q.npz", Nq_eeg=np.zeros((n, 4), np.float32), Eq_eeg=np.zeros((n, 4), np.float32))

    def fake_probs(stagers, lat, lo, hi, dev):  # p(N2) grows with the energy of the P tokens
        e = np.abs(lat["P"][lo:hi]).mean((1, 2))
        p = np.full((hi - lo, 5), 0.1)
        p[:, 2] = 0.2 + 0.5 * np.tanh(e)
        return p / p.sum(1, keepdims=True)

    monkeypatch.setattr(CE, "stage_probs", fake_probs)
    z, q = np.load(tmp_path / "a.npz"), np.load(tmp_path / "q.npz")
    refs = {"k_complex_rule": kc.reshape(-1), "delta": kc.reshape(-1)}
    rows, info = CE.explain_night("a", z, q, refs, 50.0, 100.0, ({"eeg": object()}, None, None), ToyCodec(), "cpu",
                                  rng, max_ep=6, w=3)
    assert rows and info["n_epochs"] == 6 and info["replay_max_abs_dp"] is not None
    r = rows[0]
    assert set(r["edits"]) >= {"remove", "ctrl_random"} and "frac_after" in r["edits"]["remove"]
    s = CE.summarize(rows)
    assert "N2" in s and "kc" in s["N2"] and s["N2"]["kc"]["n"] >= 5


def test_semlayer_toy():
    D = _toy(seed=3, n_groups=6, n_nights=2, n_ep=30)
    D["S"][:, 0] = D["F"][:, 0]  # the toy spectrum is noise: give the twin the planted delta column
    D["_cache"] = {}
    D["SF"] = D["S"][:, :3].copy()
    res = SL.run(D, "spec", units=8, epochs=2, n_train=20000, max_fit=5000, n_folds=3)
    assert len(res["per_fold"]) == 3 and "delta" in res["mean"]
    assert res["mean"]["delta"]["probe_units_ap"] > 0.5 and res["mean"]["delta"]["best_unit_ap"] > 0.3
    assert res["site"]["site_from_units"] is not None
    res2 = SL.run(D, "spec+P", units=8, epochs=1, n_train=10000, max_fit=5000, n_folds=2)
    assert res2["dim"] > res["dim"] and len(res2["per_fold"]) == 2
