"""Page examples + extended V4 with the toy codec of the C8 test: a planted K-complex is picked as a true
positive, saved with its context, and removed by the P edit while the controls leave it."""
import json
import os

import numpy as np

from physioex.train.bin import concept_examples as CE
from tests.test_concept_counterfactual import ToyCodec

T = CE.T


class KCToy(ToyCodec):
    """Decoder writing a full K-complex-like wave (1 s) where a run of P tokens is large."""

    def decode(self, n, e, p, seed=0):
        amp = p[..., 0].numpy()
        w = 0.01 * np.random.default_rng(seed).normal(size=(len(amp), T * 50))
        wave = -np.sin(np.linspace(0, np.pi * 2, 100))
        for i, a in enumerate(amp):
            big = a > 1.0
            for t in np.flatnonzero(big & ~np.r_[False, big[:-1]]):
                s = t * 50
                w[i, s:s + 100] += a[t] * wave[:len(w[i, s:s + 100])]
        return w


def _pack(m):
    return {"_len_k_complex_rule": np.array(len(m)), "k_complex_rule": np.packbits(m)}


def _toy_dump(tmp, n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = 0.05 * rng.normal(size=(n, 3000)).astype(np.float32)
    kc = np.zeros(n * T, bool)
    t = np.linspace(0, np.pi * 2, 100)
    for e in range(2, n - 2, 3):
        s = e * 3000 + 1000
        x.reshape(-1)[s:s + 100] += -3.0 * np.sin(t)
        kc[e * T + 20:e * T + 22] = True
    key = "hmc/SN001"
    os.makedirs(os.path.join(tmp, "hmc"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "refs", "hmc"), exist_ok=True)
    np.savez(os.path.join(tmp, f"{key}.npz"), x_eeg=x)
    np.savez(os.path.join(tmp, "refs", f"{key}.npz"), **_pack(kc))
    json.dump({"fs": {"eeg": 100.0}, "nights": {key: {"unit": "hmc"}}}, open(os.path.join(tmp, "meta.json"), "w"))
    json.dump({key: {"eeg": {"scale": 50.0, "median": 0.0}}}, open(os.path.join(tmp, "scale.json"), "w"))
    mk = os.path.join(tmp, "kc_masks.npz")
    np.savez(mk, keys=np.array([key]), night_ep=np.zeros(n, np.int32), y_ep=np.full(n, 2, np.int8),
             set_ep=np.array(["pool"] * n), lex_kc=kc, lex_n3=np.zeros(n * T, bool), swa_ep=np.zeros(n))
    return mk


def test_examples_and_v4(tmp_path, monkeypatch):
    mk = _toy_dump(str(tmp_path))
    monkeypatch.setattr(CE, "Codec", lambda combo, dev, mod="eeg": KCToy())
    out = tmp_path / "ex"
    CE.main(["--combo-dir", "x", "--dump", str(tmp_path), "--masks", f"kc={mk}", "--out", str(out),
             "--concepts", "kc", "--n-v4", "8"])
    z = np.load(out / "kc_0.npz")
    meta = json.loads(str(z["meta"]))
    assert meta["kind"] == "tp" and z["x_eeg"].shape == (5, 3000) and z["lex"].shape == (5 * T,)
    assert z["lex"][meta["a"]:meta["b"]].any() and z["ref"][meta["a"]:meta["b"]].any()
    assert "cf_ed_remove" in z.files and z["cf_ed_remove"].shape == (5, 3000)
    v4 = json.load(open(out / "v4_extra.json"))["v4"]["kc"]["pool"]["remove"]
    assert v4["n"] >= 5 and v4["remove_gone"] > 0.8 and v4["ctrl_silent_gone"] < 0.2


def test_events_iou_and_edges():
    lex = np.zeros(10 * T, bool)
    ref = np.zeros(10 * T, bool)
    lex[5 * T + 10:5 * T + 14] = ref[5 * T + 11:5 * T + 13] = True  # IoU 0.5, centre epoch 5
    lex[0:4] = ref[0:4] = True  # epoch 0: too close to the edge
    lex[7 * T:7 * T + 30] = True
    ref[7 * T:7 * T + 2] = True  # IoU 2/30 < 0.2
    assert CE.events(lex, ref, 10, "tp") == [(5 * T + 10, 5 * T + 14, 5)]
    assert CE.events(lex, ref, 10, "fn") == []
