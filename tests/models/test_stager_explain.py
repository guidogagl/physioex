"""Perturbation explanations of the multimodal stager (physioex.models.stager_explain)."""
import numpy as np
import torch

from physioex.models import stager_explain as sx
from physioex.models.qformer_stager import MultiSeqStager

SPEC = [("a", 6, 4, 5), ("b", 4, 2, 7)]
L = 5


def _stager():
    torch.manual_seed(0)
    return MultiSeqStager(SPEC, epoch="qformer", seq_len=L, seq_layers=1, d=16, heads=2, k=2, layers=1, dropout=0.0).eval()


def _window(seed=0):
    g = torch.Generator().manual_seed(seed)
    n, e = torch.randn(1, L, 10, generator=g), torch.randn(1, L, 10, generator=g)
    p = torch.randn(1, L, 12, 4, generator=g)
    p[..., 5:, 2:] = 0  # modality b has d_p 2: padding
    return n, e, p


def test_occlusion_of_a_zero_token_is_zero_and_shapes():
    st = _stager()
    n, e, p = _window()
    p[0, L // 2, 3] = 0  # token 3 of modality a already at the mean
    oc = sx.token_occlusion(st, n, e, p, SPEC)
    assert oc["a"]["P"].shape == (5,) and oc["b"]["P"].shape == (7,)
    assert abs(oc["a"]["P"][3]) < 1e-6 and np.abs(oc["a"]["P"]).max() > 0
    assert abs(oc["base"].sum() - 1) < 1e-5 and set(oc["a"]) == {"N", "E", "P"}
    d = sx.deletion_check(st, n, e, p, SPEC, oc, k=3, n_random=5)
    assert set(d) == {"top_drop", "random_drop", "random_drop_sd"}


def test_ablation_equals_manual_zeroing_and_zero_embedding():
    st = _stager()
    n, e, p = _window(1)
    ab = sx.modality_ablation(st, n, e, p, SPEC)
    n2, e2, p2 = n.clone(), e.clone(), p.clone()
    n2[0, L // 2, 6:] = 0
    e2[0, L // 2, 6:] = 0
    p2[0, L // 2, 5:] = 0
    ref = sx.probs(st, n2, e2, p2)[0].numpy()
    assert np.allclose(ab["b"]["probs"], ref, atol=1e-6)
    # the zeroed modality's embedding is the zero-input embedding used by the modality dropout
    with torch.no_grad():
        h = st.encs[1].embed(n2[0, L // 2:L // 2 + 1, 6:], e2[0, L // 2:L // 2 + 1, 6:], p2[0, L // 2:L // 2 + 1, 5:, :2])
        assert torch.allclose(h[0] + st.mod_emb[1], st.zero_embeddings()[1], atol=1e-5)
    abw = sx.modality_ablation(st, n, e, p, SPEC, whole_window=True)
    assert set(abw) == {"base", "target", "a", "b"}


def test_attention_collected_per_modality_without_changing_outputs():
    st = _stager()
    n, e, p = _window(2)
    with torch.no_grad():
        out1 = st(n, e, p)
        assert st.last_attn.shape == (L, (2 + 5) + (2 + 7))
        out2 = st(n, e, p)
    assert torch.equal(out1, out2)
    at = sx.attention_at(st, n, e, p, SPEC)
    for m, T in (("a", 5), ("b", 7)):
        assert at[m]["P"].shape == (T,)
        assert abs(at[m]["N"] + at[m]["E"] + at[m]["P"].sum() - 1) < 1e-4


def test_set_epoch_changes_only_the_target_slice():
    n, e, p = _window(3)
    P = np.ones((7, 2), np.float32)
    n2, e2, p2 = sx.set_epoch(n, e, p, SPEC, "b", 1, E=np.full(4, 9.0), P=P)
    assert torch.equal(n2, n)
    assert torch.all(e2[0, 1, 6:] == 9) and torch.equal(e2[0, 1, :6], e[0, 1, :6]) and torch.equal(e2[0, 0], e[0, 0])
    assert torch.all(p2[0, 1, 5:, :2] == 1) and torch.equal(p2[0, 1, :5], p[0, 1, :5]) and torch.equal(p2[0, 2], p[0, 2])
    assert abs(sx.rank_corr(np.arange(5.0), np.arange(5.0) * 2) - 1) < 1e-9
