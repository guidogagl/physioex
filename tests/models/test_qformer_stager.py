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

    torch.manual_seed(0)
    data = standardize({"train": _fake(1200, 0), "valid": _fake(300, 1), "test": _fake(300, 2)})
    m = QFormerStager(dim=16, d_p=4, n_tok=6, d=32, k=4, layers=1, heads=2, dropout=0.0)
    train(m, data, epochs=25, lr=3e-3, batch=128, patience=0, sched="cosine", evals_per_epoch=2, log=lambda *_: None)
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


def _fake_seq(M, seed, where):
    """Nights of M // 6 consecutive epochs; the stage of epoch t is written in the P of epoch t
    (``where="p_same"``) or in the E of epoch t - 1 (``"e_prev"``)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, M)
    N = rng.standard_normal((M, 16)).astype(np.float16)
    E = rng.standard_normal((M, 16)).astype(np.float16)
    P = rng.standard_normal((M, 6, 4)).astype(np.float16)
    if where == "p_same":
        P[np.arange(M), y, 0] += 4
    else:
        E[np.arange(M), y] += 4
        E = np.roll(E, 1, axis=0)
    return {"N": N, "E": E, "P": P, "y": y.astype(np.int8), "night": (np.arange(M) // (M // 6)).astype(np.int16),
            "epoch": np.arange(M) % (M // 6)}


def _fit_seq(arch, where):
    from physioex.models.qformer_stager import ARCHS, metrics, predict, standardize, train

    torch.manual_seed(0)
    data = standardize({s: _fake_seq(M, i, where) for i, (s, M) in enumerate((("train", 3000), ("valid", 300), ("test", 300)))})
    m = ARCHS[arch](dim=16, d_p=4, n_tok=6, d=32, k=4, layers=1, heads=2, dropout=0.0, seq_len=5, seq_layers=1)
    train(m, data, epochs=30, lr=3e-3, batch=160, patience=0, sched="cosine", evals_per_epoch=1, log=lambda *_: None)
    logits, attn = predict(m, data["test"])
    assert logits.shape == (300, 5) and torch.isfinite(logits).all()  # voting covers every epoch
    return metrics(logits.argmax(1).numpy(), data["test"]["y"].numpy(), data["test"]["night"])["kappa_pooled"], attn


def test_seq_linear_reads_the_previous_epoch():
    k, attn = _fit_seq("seqlinear", "e_prev")
    assert k > 0.8 and attn is None, k


def test_seq_qformer_trains_and_votes():
    k, attn = _fit_seq("seqqformer", "p_same")
    assert k > 0.8 and attn.shape == (300, 8), k


def _fake_mods(where_b="e_prev", drop_rows=0):
    """Two modalities on the same epochs: ``a`` noise (d_p 4), ``b`` carries the stage (d_p 2,
    other dims); ``drop_rows`` epochs of ``b`` missing in every split."""
    out = {"a": {}, "b": {}}
    for i, (s, M) in enumerate((("train", 3000), ("valid", 300), ("test", 300))):
        b = _fake_seq(M, i, where_b)
        rng = np.random.default_rng(10 + i)
        a = {k: v for k, v in b.items() if k in ("y", "night", "epoch")}
        a.update(N=rng.standard_normal((M, 16)).astype(np.float16), E=rng.standard_normal((M, 16)).astype(np.float16),
                 P=rng.standard_normal((M, 6, 4)).astype(np.float16))
        b = dict(b, N=b["N"][:, :8], E=b["E"], P=b["P"][:, :5, :2] if where_b == "e_prev" else b["P"])
        keep = np.arange(M) >= drop_rows
        out["a"][s], out["b"][s] = a, {k: v[keep] for k, v in b.items()}
    return out


def _fit_multi(drop=(), drop_rows=0):
    from physioex.models.qformer_stager import MultiSeqStager, metrics, pack_modalities, predict, train

    data, spec, missing = pack_modalities(_fake_mods(drop_rows=drop_rows), drop)
    torch.manual_seed(0)
    m = MultiSeqStager(spec, epoch="linear", d=32, heads=2, dropout=0.0, seq_len=5, seq_layers=1)
    train(m, data, epochs=30, lr=3e-3, batch=160, patience=0, sched="cosine", log=lambda *_: None)
    logits, _ = predict(m, data["test"])
    return metrics(logits.argmax(1).numpy(), data["test"]["y"].numpy(), data["test"]["night"])["kappa_pooled"], data, spec, missing


def test_pack_modalities_shapes_alignment_and_missing():
    from physioex.models.qformer_stager import pack_modalities

    data, spec, missing = pack_modalities(_fake_mods(drop_rows=7), ("a:P",))
    assert spec == [("a", 16, 4, 6), ("b", 8, 2, 5)]
    t = data["test"]
    assert t["N"].shape == (300, 24) and t["E"].shape == (300, 32) and t["P"].shape == (300, 11, 4)
    assert missing["test"] == {"a": 0, "b": 7}
    assert (t["E"][:7, 16:] == 0).all() and (t["E"][7:, 16:] != 0).any()  # missing epochs of b: zeros
    assert (t["P"][:, :6] == 0).all() and (t["P"][:, 6:, 2:] == 0).all()  # a:P dropped, b padded to d_p 4


def test_multi_stager_reads_the_right_modality_and_drop_removes_it():
    k, *_ = _fit_multi()
    assert k > 0.8, k
    k_drop, *_ = _fit_multi(drop=("b:E",))
    assert k_drop < 0.2, k_drop


def test_multi_early_fusion_reads_tokens_of_the_right_modality():
    """One Q-Former, k x 2 queries on the tokens of both modalities: learns the stage written in
    the P tokens of modality b (same epoch); late-fusion modules absent. On this toy (one modality is
    pure noise) early fusion reaches kappa 0.69, late fusion 0.42 (01.11.2026): threshold 0.5."""
    from physioex.models.qformer_stager import MultiSeqStager, metrics, pack_modalities, predict, train

    data, spec, _ = pack_modalities(_fake_mods(where_b="p_same"))
    torch.manual_seed(0)
    m = MultiSeqStager(spec, epoch="qformer", d=32, k=4, layers=1, heads=2, dropout=0.0, seq_len=5, seq_layers=1,
                       fusion="early")
    assert m.joint.queries.shape == (8, 32) and not hasattr(m, "fuse") and len(m.encs) == 0
    train(m, data, epochs=30, lr=3e-3, batch=160, patience=0, sched="cosine", log=lambda *_: None)
    logits, _ = predict(m, data["test"])
    k = metrics(logits.argmax(1).numpy(), data["test"]["y"].numpy(), data["test"]["night"])["kappa_pooled"]
    assert logits.shape == (300, 5) and k > 0.5, k
