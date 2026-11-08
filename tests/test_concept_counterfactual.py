"""V0 / V4 plumbing with a toy codec: the decoder writes a K-complex-like wave where P carries it."""
import numpy as np
import torch

from physioex.train.bin import concept_counterfactual as CF

T = CF.T


class ToyCodec:
    half = 1

    def encode(self, xp, centres):
        x = np.stack([xp[t] for t in centres])  # (B, 3000)
        tok = x.reshape(len(x), T, -1)
        p = torch.from_numpy(np.stack([np.abs(tok).max(2), tok.std(2)], -1).astype(np.float32))
        z = torch.zeros(len(x), 4)
        return {"N": z, "E": z.clone(), "P": p}

    def requant(self, p):
        return p

    def decode(self, n, e, p, seed=0):
        t = np.linspace(0, np.pi * 2, 50)
        amp = p[..., 0].numpy()  # (B, T)
        w = (amp[..., None] * -np.sin(t)[None, None]).reshape(len(amp), -1)
        return w + 0.01 * np.random.default_rng(seed).normal(size=w.shape)


def test_night_and_summary_run():
    rng = np.random.default_rng(0)
    n = 40
    x = 0.05 * rng.normal(size=(n, 3000)).astype(np.float32)
    kc = np.zeros(n * T, bool)
    t = np.linspace(0, np.pi * 2, 100)
    for e in range(0, n, 2):
        s = e * 3000 + 1000
        x.reshape(-1)[s:s + 100] += -3.0 * np.sin(t)
        kc[e * T + 20:e * T + 22] = True
    v0, v4 = CF.night(ToyCodec(), {"x_eeg": x}, 50.0, 100.0, kc, rng, max_ep=40)
    assert len(v0) == 40 and any(r["kind"] == "kc" for r in v4)
    out = CF.summarize(v0, v4)
    assert "kc_NEP" in out["V0"] and "remove" in out["V4"]
