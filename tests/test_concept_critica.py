"""D1-D3 of the critical evaluation on the toy dump of ``test_concept_compose``: the spectral twin maps the
latent inputs, the MLP / linear probes run and return a verdict, the leak matrix and the swaps run on a toy
codec."""
import numpy as np
import torch

from physioex.train.bin import concept_leak as CL
from physioex.train.bin import concept_nonlinear as CN
from physioex.train.bin import concept_rationale as cr
from tests.test_concept_compose import _toy
from tests.test_concept_counterfactual import ToyCodec

T = cr.T


def _toy_spec(seed=0):
    D = _toy(seed)
    D["_map"] = "spec"
    return D


def test_mapped_spectral_twin():
    D = _toy_spec()
    assert cr.mapped(D, ["P", "P_ctx±1", "P_back3", "P_fwd2", "P@emg", "spec_tok"]) == \
        ["spec_tok", "spec_ctx1", "spec_back3", "spec_fwd2", "P@emg"]  # duplicates dropped
    D["_map"] = "specfine"
    D["SF"] = np.zeros((len(D["y"]), 3), np.float32)
    assert cr.mapped(D, ["P", "P_ctx±2"]) == ["spec_tok", "spec_fine", "spec_ctx2", "spec_fine_ctx2"]
    D.pop("_map")
    assert cr.mapped(D, ["P", "P_ctx±1"]) == ["P", "P_ctx±1"]
    X, _, bad = cr.build(_toy_spec(), ["P", "P_ctx±1"], np.arange(2 * T))
    assert not bad and X.shape[1] == 2 * D["S"].shape[1]  # no quadratic terms on the twin


def test_spec_fine_shape():
    x = np.random.default_rng(0).normal(size=(3, 3000)).astype(np.float32)
    sf = cr.spec_fine(x, 100.0)
    assert sf.shape == (3 * T, 30) and np.isfinite(sf).all()


def test_nonlinear_probe_runs():
    D = _toy()
    D["SF"] = D["S"][:, :3].copy()
    refs = {"k_complex_vis1": "dreams", "delta": "pool"}  # the toy P carries delta, not the K-complex context
    inputs = {"P": ["P"], "spec": ["spec_tok"], "specfine": ["spec_tok", "spec_fine"],
              "P+specfine": ["P", "spec_tok", "spec_fine"]}
    res = CN.run(D, refs, inputs, hidden=8, max_fit=2000, n_boot=5)
    row = res["delta"]
    assert row["fold_by"] == "site" and row["folds"] == 4
    assert row["ap"]["P"]["linear"] > 0.5 and row["ap"]["P"]["mlp"] > 0.5
    assert "P-specfine:mlp" in row["boot"] and "added_by_P:mlp" in row["boot"]
    assert res["k_complex_vis1"]["fold_by"] == "excerpt"
    assert set(CN.verdict(res).values()) <= {"readout", "content", "none", "linear_only"}


def test_leak_probes_toy():
    D = _toy()
    V = CL.epoch_views(D)
    assert V["P"].shape == (len(D["y"]) // T, 2 * D["P"].shape[1]) and "k_complex_rule" in V["frac"]
    out = CL.probes(V, n_folds=2)
    lv = out["levels"]
    assert set(lv) >= {"N", "E", "P", "N+E+P", "spec_ep"}
    assert lv["E"]["stage_kappa"] > 0.5  # the planted E carries the SWA, hence N2 against N3
    assert lv["N"]["site_balanced_acc"] is not None and lv["P"]["event_delta_spearman"] > 0.5


def test_leak_swaps_toy(tmp_path):
    rng = np.random.default_rng(0)
    keys = ["a", "b"]
    meta = {"nights": {k: {"site": s} for k, s in zip(keys, ("s0", "s1"))}, "fs": {"eeg": 100.0}}
    scale = {k: {"eeg": {"scale": 50.0}} for k in keys}
    n = 12
    for k in keys:
        x = 0.05 * rng.normal(size=(n, 3000)).astype(np.float32)
        y = np.array([2, 3] * (n // 2))
        np.savez(tmp_path / f"{k}.npz", x_eeg=x, y=y)
    rows = CL.swaps(ToyCodec(), str(tmp_path), meta, scale, keys, 100.0, n_ep=6, seed=0)
    names = {r["swap"] for r in rows}
    assert {"N_other_night", "E_other_stage", "P_other_stage", "E_same_stage", "P_perm"} <= names
    s = CL.summarize_swaps(rows)
    assert s["P_perm"]["n"] == 12 and "log_rms" in s["P_perm"] and s["P_perm"]["log_rms"]["moved"] >= 0
    assert all(r["donor_stage"] != r["stage"] for r in rows if r["swap"] == "E_other_stage")
