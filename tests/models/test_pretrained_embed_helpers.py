"""Unit tests for pure helpers in physioex.models.pretrained / embed.

The heavy paths (load_from_pretrained, extract_embeddings, linear_probe) need
checkpoints / HuggingFace and are covered by gated tests elsewhere; here we
pin the small pure functions that underpin them.
"""
from pathlib import Path

import numpy as np
import pytest

from physioex.models.pretrained import _format_dataset_metrics, _resolve_class
from physioex.models.embed import _hf_repo_id, _load_npy_as_float32


# ── pretrained._resolve_class ────────────────────────────────────────

def test_resolve_class_valid_spec():
    cls = _resolve_class("physioex.models.chambon2018:Chambon2018Net")
    from physioex.models.chambon2018 import Chambon2018Net
    assert cls is Chambon2018Net


def test_resolve_class_bad_class_raises():
    with pytest.raises(AttributeError):
        _resolve_class("physioex.models.chambon2018:NoSuchClass")


def test_resolve_class_bad_module_raises():
    with pytest.raises(ModuleNotFoundError):
        _resolve_class("physioex.models.does_not_exist:Foo")


# ── embed._hf_repo_id ────────────────────────────────────────────────

def test_hf_repo_id_format():
    repo = _hf_repo_id("cbramod")
    assert repo.endswith("/cbramod-embeddings")
    assert "/" in repo


# ── embed._load_npy_as_float32 ───────────────────────────────────────

def test_load_npy_float_cast(tmp_path):
    p = tmp_path / "f.npy"
    np.save(p, np.array([[1.0, 2.0]], dtype=np.float64))
    out = _load_npy_as_float32(p)
    assert out.dtype == np.float32
    assert np.allclose(out, [[1.0, 2.0]])


def test_load_npy_int_preserved(tmp_path):
    p = tmp_path / "i.npy"
    np.save(p, np.array([1, 2, 3], dtype=np.int16))
    out = _load_npy_as_float32(p)
    assert out.dtype == np.int16  # int16 labels preserved, not cast
    assert out.tolist() == [1, 2, 3]


# ── pretrained._format_dataset_metrics ───────────────────────────────

def test_format_metrics_full_columns_unchanged_layout():
    m = {"hmc": dict(accuracy=0.8, f1_score=0.7, cohen_kappa=0.75, precision=0.71, recall=0.69)}
    lines = _format_dataset_metrics(m)
    assert lines[0].split() == ["Dataset", "Acc", "F1", "Kappa", "Prec", "Rec"]
    assert lines[1] == "-" * 60
    assert lines[2].split() == ["hmc", "0.8000", "0.7000", "0.7500", "0.7100", "0.6900"]


def test_format_metrics_skips_columns_nobody_reports():
    m = {"shhs": dict(accuracy=0.9, f1_score=0.8, cohen_kappa=0.85, n_subjects=842),
         "stages_GSLH": dict(accuracy=0.8, f1_score=0.7, cohen_kappa=0.7, n_subjects=3)}
    lines = _format_dataset_metrics(m)
    assert lines[0].split() == ["Dataset", "Acc", "F1", "Kappa", "N"]
    assert "0.0000" not in "\n".join(lines)
    # long site names widen the first column; rows stay aligned with the header
    assert len({len(x) for x in (lines[0], lines[2], lines[3])}) == 1
    assert lines[2].split()[-1] == "842"


def test_format_metrics_missing_value_is_a_dash_not_zero():
    m = {"a": dict(accuracy=0.9, cohen_kappa=0.8), "b": dict(accuracy=0.7)}
    rows = {l.split()[0]: l.split() for l in _format_dataset_metrics(m)[2:]}
    assert rows["b"][1:] == ["0.7000", "-"]


# ── pretrained.load_from_pretrained (offline, synthetic repo) ────────

def test_load_from_pretrained_factory_roundtrip(tmp_path, monkeypatch, capsys):
    """config.json with a ``factory`` + weights + a partial metrics.json, served from a local folder."""
    import json

    import huggingface_hub
    import torch

    from physioex.models.pretrained import load_from_pretrained
    from physioex.models.protosleepnet import ProtoSleepNet

    kwargs = dict(n_channels=3, use_channel_mixer=True, cdropout=0.5, cm_n_heads=4, cm_d_ff=256, cm_n_layers=1)
    torch.manual_seed(0)
    src = ProtoSleepNet.from_seq_sleep_net(**kwargs).eval()
    folder = tmp_path / "toy-model"
    folder.mkdir()
    torch.save(src.state_dict(), folder / "model.pt")
    (folder / "config.json").write_text(json.dumps({
        "model_class": "physioex.models.protosleepnet:ProtoSleepNet",
        "factory": "from_seq_sleep_net",
        "factory_kwargs": kwargs,
    }))
    (folder / "metrics.json").write_text(json.dumps(
        {"siteA": {"accuracy": 0.9, "f1_score": 0.8, "cohen_kappa": 0.85, "n_subjects": 5}}))

    def fake_download(repo_id, filename, **_):
        assert repo_id == "local/test"
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    model = load_from_pretrained("toy-model", repo_id="local/test", verbose=True)

    assert not model.training
    sd = model.state_dict()
    assert set(sd) == set(src.state_dict())
    assert all(torch.equal(sd[k], v) for k, v in src.state_dict().items())
    x = torch.randn(1, 3, 3, 29, 129)
    with torch.no_grad():
        assert torch.equal(model(x), src(x))
    out = capsys.readouterr().out
    assert "siteA" in out and "Prec" not in out and "0.0000" not in out


@pytest.mark.hf
@pytest.mark.parametrize("name,L", [("protosleepnet-gagliardi", 20), ("protosleeptransformer-gagliardi", 21)])
def test_released_prototype_models_load_and_run(name, L):
    """The two official ProtoSleepNet checkpoints on the HF Hub load with the default repo and give (B, L, 5) logits."""
    import torch

    from physioex.models import load_from_pretrained

    model = load_from_pretrained(name)
    assert not model.training
    with torch.no_grad():
        y = model(torch.randn(2, L, 3, 29, 129))
    assert tuple(y.shape) == (2, L, 5)
    assert torch.isfinite(y).all()


@pytest.mark.hf
@pytest.mark.parametrize("name", ["protosleepnet-gagliardi", "protosleeptransformer-gagliardi"])
def test_docs_page_tables_match_hub_metrics(name):
    """Every per-site number on the documentation page equals the ``metrics.json`` stored with the weights on the Hub."""
    import json
    import re

    from huggingface_hub import hf_hub_download

    page = (Path(__file__).resolve().parents[2] / "docs/pages/train/networks/protosleepnet.md").read_text()
    section = page.split(f"(`{name}`)", 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]
    rows = re.findall(r"\| `(\S+)` \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", section)
    with open(hf_hub_download("4rooms/physioex", f"{name}/metrics.json")) as f:
        metrics = json.load(f)
    assert {r[0] for r in rows} == set(metrics)
    for site, n, acc, kappa, f1 in rows:
        m = metrics[site]
        assert int(n) == m["n_subjects"]
        assert (acc, kappa, f1) == tuple(f"{m[k]:.3f}" for k in ("accuracy", "cohen_kappa", "f1_score"))
