"""MSAE-VQ: residual VQ, quantized hierarchy N -> E -> P, codes and bits, trainer step.

Random init, tiny batches, CPU.
"""
import math

import pytest
import torch

L = 5


def _batch(B=4):
    torch.manual_seed(0)
    return {
        "signals": {"EEG_0": torch.randn(B, L, 3000)},
        "channel_order": ["EEG_0"],
        "labels": torch.zeros(B, L, dtype=torch.long),
        "subject": [{"id": f"s{i}"} for i in range(B)],
        "epoch_indices": torch.arange(L)[None].repeat(B, 1),
    }


def _model(**kw):
    from physioex.models.msae_vq import MSAEVQ

    torch.manual_seed(0)
    kw = {"codes": 16, "codes_n": 8, "levels": (2, 3, 3), **kw}
    return MSAEVQ(sequence_length=L, width=8, dim=32, d_p=4, token_sec=0.5, **kw)


def _init_codebooks(model, B=8):
    """k-means on the latents of a random batch (what fit_vq_codebooks does on real data)."""
    from physioex.models.msae_vq import fit_vq_codebooks

    b = _batch(B)
    fit_vq_codebooks(model, [b], min_per_code=1, max_batches=1)
    return b


def test_residual_vq_error_decreases_and_prefix_is_sum():
    from physioex.models.msae_vq import ResidualVQ

    torch.manual_seed(0)
    rvq = ResidualVQ(4, 32, 8).eval()
    z = torch.randn(500, 8)
    rvq.kmeans_init(z)
    errs = [(rvq(z, k)[0] - z).pow(2).mean().item() for k in range(1, 5)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs
    zq, codes, _ = rvq(z, 3)
    assert codes.shape == (500, 3)
    assert torch.allclose(zq, rvq.lookup(codes), atol=1e-5)


def test_factorized_rvq_prefix_lookup_and_gradients():
    from physioex.models.msae_vq import ResidualVQ

    torch.manual_seed(0)
    rvq = ResidualVQ(3, 32, 16, factor_dim=4).eval()
    z = torch.randn(400, 16)
    rvq.kmeans_init(z)
    errs = [(rvq(z, k)[0] - z).pow(2).mean().item() for k in range(1, 4)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs
    zq, codes, _ = rvq(z, 3)
    assert torch.allclose(zq, rvq.lookup(codes), atol=1e-5)
    assert torch.allclose(rvq.levels[0].codebook.norm(dim=1), torch.ones(32), atol=1e-5)
    rvq.train()
    zg = z.clone().requires_grad_(True)
    zq, _, loss = rvq(zg)
    (zq.pow(2).sum() + loss).backward()
    assert rvq.proj_out[0].weight.grad.abs().sum() > 0 and rvq.proj_in[0].weight.grad.abs().sum() > 0
    assert zg.grad is not None


def test_residual_vq_ema_and_dead_code_restart():
    from physioex.models.msae_vq import ResidualVQ

    torch.manual_seed(0)
    rvq = ResidualVQ(1, 8, 2, restart_after=1)
    z = torch.randn(64, 2)
    rvq.kmeans_init(z)
    vq = rvq.levels[0]
    vq.codebook[0] = 100.0  # far from the data: never chosen
    vq.ema_sum[0] = 100.0
    rvq.train()
    for _ in range(6):
        rvq(z)
    assert vq.codebook[0].abs().max() < 10  # restarted onto a data vector
    assert vq.idle.max() <= vq.restart_after + 1


def test_quantizer_dropout_only_in_training():
    model = _model(q_dropout=1.0)
    x = torch.randn(2, L, 1, 3000)
    model.eval()
    assert model.encode(x)["codes"]["E"].shape[-1] == 3
    model.vq_levels = {"N": 1, "E": 2, "P": 1}
    c = model.encode(x)["codes"]
    assert c["N"].shape == (2, 1) and c["E"].shape == (2, 2) and c["P"].shape == (2, 60, 1)
    model.train()
    seen = {model.encode(x)["codes"]["E"].shape[-1] for _ in range(30)}
    assert seen <= {1, 2, 3} and len(seen) > 1


def test_shapes_codes_embed_tokens_bits():
    model = _model().eval()
    _init_codebooks(model)
    x = torch.randn(2, L, 1, 3000)
    out = model(x)
    assert out["N"].shape == (2, 32) and out["E"].shape == (2, 32) and out["P"].shape == (2, 60, 4)
    assert out["wave"].shape == (2, 3000) and model.tokens(x).shape == (2, 62, 32)
    c = model.codes(x)
    assert c["N"].shape == (2, 2) and c["E"].shape == (2, 3) and c["P"].shape == (2, 60, 3)
    z = model.embed(c)
    for s in ("N", "E", "P"):
        assert torch.allclose(z[s], out[s], atol=1e-5)
    assert model.bits_per_epoch() == round(2 * 3 + 3 * 4 + 60 * 3 * 4)
    assert model.bits_per_epoch({"N": 1, "E": 1, "P": 1}) == round(3 + 4 + 60 * 4)


def test_decode_from_codes_equals_forward():
    model = _model().eval()
    _init_codebooks(model)
    x = torch.randn(2, L, 1, 3000)
    out = model(x)
    z = model.embed(model.codes(x))
    assert torch.allclose(model.decode(z["N"], z["E"], z["P"]), out["wave"], atol=1e-4)


def test_factorized_n_decode_from_codes():
    model = _model(factor_dim_n=4).eval()
    _init_codebooks(model)
    x = torch.randn(2, L, 1, 3000)
    out = model(x)
    z = model.embed(model.codes(x))
    assert torch.allclose(z["N"], out["N"], atol=1e-5)
    assert torch.allclose(model.decode(z["N"], z["E"], z["P"]), out["wave"], atol=1e-4)


def test_n_does_not_depend_on_centre_epoch():
    model = _model().eval()
    _init_codebooks(model)
    x = torch.randn(2, L, 1, 3000)
    x2 = x.clone()
    x2[:, L // 2] = torch.randn(2, 1, 3000)
    a, b = model.codes(x), model.codes(x2)
    assert torch.equal(a["N"], b["N"])
    assert not torch.equal(a["P"], b["P"])


def test_e_and_p_see_n_only_through_its_code():
    """Same N code -> same conditioning: a continuous N moved without changing its code
    leaves E and P unchanged."""
    model = _model().eval()
    _init_codebooks(model)
    x = torch.randn(2, L, 1, 3000)
    a = model.encode(x)
    ctx = model.encode_context
    model.encode_context = lambda xs: ctx(xs) + 1e-6 * torch.randn(len(xs), model.dim)
    b = model.encode(x)
    assert torch.equal(a["codes"]["N"], b["codes"]["N"])
    assert torch.equal(a["E"], b["E"]) and torch.equal(a["P"], b["P"])


def test_loads_msae_weights():
    from physioex.models.msae import MSAE

    torch.manual_seed(1)
    src = MSAE(sequence_length=L, width=8, dim=32, d_p=4, token_sec=0.5)
    model = _model()
    missing, unexpected = model.load_state_dict(src.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("rvq.") for k in missing)
    assert torch.equal(model.e_film.weight, src.e_film.weight)


def test_trainer_step_finite_and_overfits():
    from physioex.models.msae_vq import MSAEVQTrainer

    model = _model(q_dropout=0.0)
    b = _init_codebooks(model, B=4)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(40):
        loss, r2, extra = MSAEVQTrainer._step(model, b, None, torch.device("cpu"))
        assert torch.isfinite(loss) and "commit" in extra and "perplexity_P1" in extra
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert min(losses[-5:]) < losses[0]


def test_vq_keys_and_grid():
    from physioex.models.msae_vq import default_vq_grid, parse_vq_key, vq_key

    lv = {"N": 4, "E": 8, "P": 6}
    assert vq_key(lv) == "vq_N4E8P6" and parse_vq_key("vq_N4E8P6") == lv == parse_vq_key("N4E8P6")
    grid = default_vq_grid({"N": 4, "E": 8, "P": 8})
    assert {"N": 4, "E": 8, "P": 8} in grid and {"N": 4, "E": 8, "P": 1} in grid and {"N": 1, "E": 8, "P": 8} in grid
    assert len(grid) == len({vq_key(g) for g in grid})
    with pytest.raises(ValueError):
        parse_vq_key("bits_8")


def test_noninferiority_reads_vq_arms(tmp_path):
    import json

    from physioex.models.noninferiority import BANDS5, compare

    nights = list(range(6))

    def arm(r2, kappa):
        pn = {"night": nights, "r2": [r2] * 6, "logpow5": {b: [0.97] * 6 for b in BANDS5}}
        return {"NEP": {"per_night": pn}, "NEP_det": {"per_night": pn},
                "spec_n": {"per_night": {"night": nights, "r2": [0.68] * 6}}}

    def probe(kappa):
        return {"latent_bits": 0, "staging": {"NE": {"nights": nights, "kappa_per_night": [kappa + 0.001 * i for i in nights]}},
                "spindle_proxy": {"auc_decoded_NEP_nights": nights, "auc_decoded_NEP_per_night": [0.99] * 6}}

    ref, cand = tmp_path / "ref", tmp_path / "cand"
    ref.mkdir(); cand.mkdir()
    json.dump({"model": {"bits_0": arm(0.96, 0.72)}}, open(ref / "recon_eval_valid.json", "w"))
    json.dump(probe(0.72), open(ref / "probe_b0_valid.json", "w"))
    json.dump({"model": {"vq_N4E8P6": arm(0.96, 0.72)}}, open(cand / "recon_eval_valid.json", "w"))
    json.dump(probe(0.72), open(cand / "probe_vq_N4E8P6_valid.json", "w"))
    r = compare(f"{ref}@0", f"{cand}@vq_N4E8P6", split="valid")
    assert not r["missing"] and r["noninferior"]


def test_vq_explain_atlas_bursts_swap_sequence_bag():
    import numpy as np

    from physioex.models import vq_explain as vx

    model = _model().eval()
    _init_codebooks(model)
    windows = np.random.default_rng(0).standard_normal((6, L, 3000)).astype(np.float32)
    z = vx.encode_all(model, windows, batch=3)
    assert z["codes_E"].shape == (6, 3) and z["codes_P"].shape == (6, 60, 3)
    centres = windows[:, L // 2]
    ae = vx.atlas_e(model, z, centres, labels=np.arange(6) % 5, min_count=1)
    assert sum(v["count"] for v in ae.values()) == 6
    assert all(v["prototype"].shape == (3000,) for v in ae.values())
    bursts = vx.bursts_per_token(np.zeros((6, 30), bool) | (np.arange(30) == 3), 60)
    assert bursts.shape == (6, 60) and bursts[:, 6:8].all() and bursts.sum() == 12
    ap = vx.atlas_p(z, centres, bursts, min_count=1)
    assert sum(v["count"] for v in ap.values()) == 360
    bc = vx.burst_codes(z["codes_P"][..., 0], bursts)
    assert abs(bc["base_rate"] - 12 / 360) < 1e-9
    codes = {k: torch.from_numpy(z[f"codes_{k}"][:1]) for k in ("N", "E", "P")}
    same = vx.swap_token(model, codes, token=7, new_code=int(codes["P"][0, 7, 0]))
    assert np.allclose(same["sigma_change_token"], 0, atol=1e-4)
    other = (int(codes["P"][0, 7, 0]) + 1) % 16
    sw = vx.swap_token(model, codes, token=7, new_code=other)
    assert not np.allclose(sw["wave_before"][0], sw["wave_after"][0])
    st = vx.sequence_stats(np.array([0, 0, 1, 1, 2, 2]), labels=np.array([3, 3, 1, 1, 0, 0]))
    assert abs(st["nmi_stages"] - 1) < 1e-9 and abs(st["stay"] - 0.6) < 1e-9
    bag = vx.bag_of_codes(z, model.n_codes)
    assert bag.shape == (6, 8 + 16 + 16) and np.allclose(bag[:, 24:].sum(1), 1)
    assert len(vx.bag_names(model.n_codes)) == bag.shape[1]
