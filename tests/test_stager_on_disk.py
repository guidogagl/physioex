"""Stager latents on disk (LatentWriter / LazyLevel / lazy_modalities) give the same inputs, statistics
and predictions as the in-memory path (standardize / pack_modalities / windowed predict), 07.11.2026."""
import numpy as np
import torch

from physioex.models import qformer_stager as Q


def _nights(rng, n_nights, dim=6, d_p=3, n_tok=4, skip=()):
    """Latents of ``n_nights`` nights (float16) as extract_latents stores them; ``skip``: (night, epoch) left out."""
    out = {k: [] for k in ("N", "E", "P", "y", "night", "epoch")}
    for g in range(n_nights):
        ep = np.array([e for e in range(30 + 3 * g) if (g, e) not in skip])
        n = len(ep)
        out["N"].append(rng.normal(1.0, 2.0, (n, dim)).astype(np.float16))
        out["E"].append(rng.normal(-1.0, 0.5, (n, dim)).astype(np.float16))
        out["P"].append(rng.normal(0.5, 1.5, (n, n_tok, d_p)).astype(np.float16))
        out["y"].append(rng.integers(0, 5, n).astype(np.int8))
        out["night"].append(np.full(n, g, np.int32))
        out["epoch"].append(ep.astype(np.int32))
    return {k: np.concatenate(v) for k, v in out.items()}


def _write(tmp, name, splits, chunk=7):
    d = str(tmp / name)
    for s, arr in splits.items():
        w = Q.LatentWriter(d, s, stats=(s == "train"))
        for i in range(0, len(arr["y"]), chunk):  # several appends, as night by night
            w.add(**{k: v[i:i + chunk] for k, v in arr.items()})
        st = w.close()
        if st is not None:
            Q.save_stats(f"{d}/stats.npz", st)
    return d


def test_writer_roundtrip_and_stats(tmp_path):
    rng = np.random.default_rng(0)
    tr = _nights(rng, 3)
    d = _write(tmp_path, "eeg", {"train": tr})
    back = Q.open_latents(d, "train")
    for k in tr:
        np.testing.assert_array_equal(np.asarray(back[k]), tr[k])
    st, ref = Q.load_stats(f"{d}/stats.npz"), Q.latent_stats(tr)
    for k in "NEP":
        np.testing.assert_allclose(st[k][0], ref[k][0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(st[k][1], ref[k][1], rtol=1e-4, atol=1e-5)


def test_lazy_single_modality_equals_standardize(tmp_path):
    rng = np.random.default_rng(1)
    splits = {"train": _nights(rng, 3), "valid": _nights(rng, 2)}
    d = _write(tmp_path, "eeg", splits)
    lazy, spec, _ = Q.lazy_modalities({"eeg": d}, stats={"eeg": Q.latent_stats(splits["train"])})
    eager = Q.standardize(splits)
    assert spec == [("eeg", 6, 3, 4)]
    idx = torch.tensor([[0, 1, 2], [5, 6, 7]])
    for s in splits:
        for k in "NEP":
            torch.testing.assert_close(lazy[s][k][idx], eager[s][k][idx])
            torch.testing.assert_close(lazy[s][k][2:9], eager[s][k][2:9])
        torch.testing.assert_close(lazy[s]["y"], eager[s]["y"])


def test_lazy_multimodal_equals_pack(tmp_path):
    rng = np.random.default_rng(2)
    eeg = {"train": _nights(rng, 3), "valid": _nights(rng, 2)}
    gaps = {(0, 3), (1, 0), (1, 10), (2, 31)}  # epochs without EOG: zero-filled
    eog = {"train": _nights(rng, 3, d_p=2, n_tok=5, skip=gaps), "valid": _nights(rng, 2, d_p=2, n_tok=5)}
    dirs = {"eeg": _write(tmp_path, "eeg", eeg), "eog": _write(tmp_path, "eog", eog)}
    drop = ["eog:E"]
    st = {"eeg": Q.latent_stats(eeg["train"]), "eog": Q.latent_stats(eog["train"])}
    lazy, spec, miss = Q.lazy_modalities(dirs, drop, stats=st)
    eager, spec2, miss2 = Q.pack_modalities({"eeg": eeg, "eog": eog}, drop)
    assert spec == spec2 and miss == miss2 and miss["train"]["eog"] == len(gaps)
    w = torch.from_numpy(Q.train_windows(Q.night_groups(eager["train"]), 5, np.random.default_rng(0)))
    for k in "NEP":
        assert lazy["train"][k].shape == tuple(eager["train"][k].shape)
        torch.testing.assert_close(lazy["train"][k][w], eager["train"][k][w])
        torch.testing.assert_close(lazy["valid"][k][:], eager["valid"][k])


def test_per_night_predict_equals_windowed(tmp_path):
    """New _predict_seq (one read per night) = the old one (all windows of all nights at once)."""
    rng = np.random.default_rng(3)
    v = _nights(rng, 3)
    data = Q.standardize({"train": v, "valid": v})["valid"]
    L = 5
    torch.manual_seed(0)
    model = Q.ARCHS["seqqformer"](dim=6, d_p=3, n_tok=4, d=16, k=2, layers=1, heads=2, dropout=0.0, seq_len=L,
                                  seq_layers=1).eval()
    logp, attn = Q._predict_seq(model, data, L, batch=4 * L, device="cpu")
    W = torch.from_numpy(Q.all_windows(Q.night_groups(data), L))
    M = len(data["y"])
    prob, cnt, att = torch.zeros(M, 5), torch.zeros(M), None
    with torch.no_grad():
        for i in range(0, len(W), 4):
            w = W[i:i + 4]
            out = model(data["N"][w], data["E"][w], data["P"][w]).float().softmax(-1)
            prob.index_add_(0, w.flatten(), out.flatten(0, 1))
            cnt.index_add_(0, w.flatten(), torch.ones(w.numel()))
            a = model.last_attn.float()
            att = torch.zeros(M, a.shape[-1]) if att is None else att
            att.index_add_(0, w.flatten(), a)
    torch.testing.assert_close(logp, (prob / cnt[:, None]).clamp_min(1e-8).log(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(attn, att / cnt[:, None], rtol=1e-5, atol=1e-5)


def test_train_on_disk_runs(tmp_path):
    """Training loop on LazyLevel inputs: a few steps, best checkpoint kept."""
    rng = np.random.default_rng(4)
    d = _write(tmp_path, "eeg", {"train": _nights(rng, 4), "valid": _nights(rng, 2)})
    data, spec, _ = Q.lazy_modalities({"eeg": d})
    torch.manual_seed(0)
    model = Q.ARCHS["seqqformer"](dim=6, d_p=3, n_tok=4, d=16, k=2, layers=1, heads=2, dropout=0.0, seq_len=5,
                                  seq_layers=1)
    hist = Q.train(model, data, epochs=2, lr=1e-3, batch=20, patience=0, sched="cosine", evals_per_epoch=2,
                   log=lambda *_, **__: None)
    assert len(hist) >= 2 and all(np.isfinite(h["valid_kappa"]) or np.isnan(h["valid_kappa"]) for h in hist)


def test_prefetch_threads_do_not_change_training(tmp_path):
    """Threaded read-ahead: same batches in the same order, same history as one thread."""
    rng = np.random.default_rng(5)
    d = _write(tmp_path, "eeg", {"train": _nights(rng, 4), "valid": _nights(rng, 2)})
    data, _, _ = Q.lazy_modalities({"eeg": d})
    hs = []
    for th in (1, 4):
        torch.manual_seed(0)
        model = Q.ARCHS["seqqformer"](dim=6, d_p=3, n_tok=4, d=16, k=2, layers=1, heads=2, dropout=0.1, seq_len=5,
                                      seq_layers=1)
        hs.append(Q.train(model, data, epochs=2, lr=1e-3, batch=20, patience=0, sched="cosine", evals_per_epoch=2,
                          log=lambda *_, **__: None, prefetch_threads=th))
    assert hs[0] == hs[1]
