"""Q-Former stager: shapes, attention shares, training on separable latents, extraction windows."""
import numpy as np
import torch


def _fake(M=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, M)
    N = rng.standard_normal((M, 16)).astype(np.float16)
    E = rng.standard_normal((M, 16)).astype(np.float16)
    P = rng.standard_normal((M, 6, 4)).astype(np.float16)
    P[np.arange(M), y % 6, 0] += 4  # stage written in one P token
    return {"N": N, "E": E, "P": P, "y": y.astype(np.int8), "night": (np.arange(M) % 6).astype(np.int16)}


def test_shapes_and_attention():
    from physioex.models.qformer_stager import QFormerStager, attention_by_stage

    m = QFormerStager(dim=16, d_p=4, n_tok=6, d=32, k=3, layers=1, heads=2).eval()
    out = m(torch.randn(7, 16), torch.randn(7, 16), torch.randn(7, 6, 4))
    assert out.shape == (7, 5) and m.last_attn.shape == (7, 8)
    assert torch.allclose(m.last_attn.sum(1), torch.ones(7), atol=1e-5)
    a = attention_by_stage(m.last_attn, np.arange(7) % 5)
    assert abs(a["W"]["N"] + a["W"]["E"] + a["W"]["P"] - 1) < 1e-5 and len(a["W"]["P_tokens"]) == 6


def test_training_learns_stage_from_p():
    from physioex.models.qformer_stager import QFormerStager, standardize, train, predict, metrics

    data = standardize({"train": _fake(1200, 0), "valid": _fake(300, 1), "test": _fake(300, 2)})
    m = QFormerStager(dim=16, d_p=4, n_tok=6, d=32, k=4, layers=1, heads=2, dropout=0.0)
    train(m, data, epochs=25, lr=3e-3, batch=128, patience=25, log=lambda *_: None)
    r = metrics(predict(m, data["test"])[0].argmax(1).numpy(), data["test"]["y"].numpy(), data["test"]["night"])
    assert r["kappa_pooled"] > 0.8, r["kappa_pooled"]


def test_extract_latents_windows_and_labels():
    from physioex.models.qformer_stager import extract_latents

    class Sub:
        subject_id = "a"

    class DS:
        sequence_length, _subjects, _n_epochs, _main_period = 5, [Sub()], {"a": 12}, {"a": (1, 11)}

        def _build_item(self, s, i, n):
            x = torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 3000)
            y = torch.tensor([0, 1, 2, -1, 3, 4, 0, 1, 2, 3, 4, 0])
            return {"signals": {"EEG": x[None]}, "labels": y}

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def encode(self, x):  # N = first sample of the window's first epoch, E = centre epoch index
            b = len(x)
            return {"N": x[:, 0, 0, :1].repeat(1, 2), "E": x[:, 2, 0, :1].repeat(1, 2), "P": torch.zeros(b, 3, 2)}

    d = extract_latents(M(), DS(), ["a"], batch=2)
    assert d["epoch"].tolist() == [2, 4, 5, 6, 7, 8, 9]  # 3 unscored, context of 2 each side
    assert d["E"][:, 0].tolist() == d["epoch"].tolist() and d["N"][:, 0].tolist() == (d["epoch"] - 2).tolist()
    assert d["y"].tolist() == [2, 3, 4, 0, 1, 2, 3]
