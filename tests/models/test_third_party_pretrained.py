"""Tests for the third-party pretrained registry in physioex.models.pretrained.

All tests are offline: downloads (``ensure_checkpoint``, ``hf_hub_download``)
and the authors' ``from_original`` factories are monkeypatched with fakes.
The single real-download smoke tests are gated with ``@pytest.mark.hf``.
"""
from pathlib import Path

import pytest
import torch
from torch import nn

import physioex.models.pretrained as pretrained
from physioex.models.pretrained import (
    THIRD_PARTY_PRETRAINED,
    _jones_spec,
    _license_is_restrictive,
    load_from_pretrained,
    list_pretrained,
)

FIXTURES = Path(__file__).parent / "fixtures" / "cardioresp"

#: name -> (n_files, filename of file 0 that the loader should receive)
_EXPECTED_JONES = {
    "primary": "epoch_0647_kappa_0.724.pt",
    "realtime": "epoch_0637_kappa_0.703.pt",
    "no-demographics": "epoch_0552_kappa_0.718.pt",
    "no-time": "epoch_0490_kappa_0.724.pt",
    "only-ecg": "epoch_0369_kappa_0.721.pt",
    "cyclic": "epoch_0548_kappa_0.725.pt",
    "cyclic-realtime": "epoch_0472_kappa_0.703.pt",
}


class _FakeModel(nn.Module):
    """Placeholder standing in for the (parallel-written) authors' models."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


def _fake_factory(*args, **kwargs):
    model = _FakeModel()
    return model


# ── Fakes installed by the shared fixture ────────────────────────────


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """Patch downloads and from_original factories; record every call.

    The authors' ``from_original`` factories are not imported here (they are
    written in parallel); the registry loaders are swapped for a fake that
    records its arguments and returns a :class:`_FakeModel`.
    """
    calls = {
        "ensure": [],       # name per ensure_checkpoint call
        "hf": [],           # (repo_id, filename) per hf_hub_download call
        "from_original": [],  # (args, kwargs) per loader call
    }

    def fake_ensure(name, checkpoint_path=None):
        calls["ensure"].append(name)
        d = tmp_path / name / "checkpoint"
        d.mkdir(parents=True, exist_ok=True)
        # Lay down one file per filename so every loader path resolves.
        spec = THIRD_PARTY_PRETRAINED[THIRD_PARTY_PRETRAINED[name].get("alias", name)]
        src = spec["source"]
        for f in src.filenames:
            (d / f).write_text("fake")
        first = d / (src.local_filename or Path(src.filenames[0]).name)
        return str(first)

    def fake_hf_hub_download(repo_id, filename, **kw):
        calls["hf"].append((repo_id, filename))
        p = tmp_path / "hf" / repo_id / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fake")
        return str(p)

    def fake_from_original(*a, **k):
        calls["from_original"].append((a, k))
        return _fake_factory()

    monkeypatch.setattr(pretrained, "ensure_checkpoint", fake_ensure)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_hf_hub_download, raising=True
    )
    # Swap the registry loaders (dicts live in this module's namespace).
    for name in list(THIRD_PARTY_PRETRAINED):
        monkeypatch.setitem(
            THIRD_PARTY_PRETRAINED[name],
            "loader",
            lambda paths, n=name: fake_from_original(_paths=paths, _name=n),
        )
    return calls


# ── Registry consistency ─────────────────────────────────────────────


def test_registry_has_all_fields():
    required = {
        "source", "loader", "license", "license_note", "reference",
        "n_classes", "class_names", "input",
    }
    for name, spec in THIRD_PARTY_PRETRAINED.items():
        missing = required - set(spec)
        assert not missing, f"{name}: missing fields {sorted(missing)}"
        assert len(spec["class_names"]) == spec["n_classes"], name
        assert callable(spec["loader"]), name
        assert len(spec["reference"]) > 10, name


def test_registry_expected_entries():
    expected = (
        ["wav2sleep", "wav2sleep-eog", "sleepppgnet-replica", "jones-ecg"]
        + [f"jones-ecg-{v}" for v in _EXPECTED_JONES]
    )
    for name in expected:
        assert name in THIRD_PARTY_PRETRAINED, name
    # No surprises (11 entries: 3 + 7 variants + 1 alias).
    assert len(THIRD_PARTY_PRETRAINED) == 11


def test_jones_urls_well_formed():
    for v, filename, folder in [
        ("primary", "epoch_0647_kappa_0.724.pt", "1_paper_models/primary_model"),
        ("realtime", "epoch_0637_kappa_0.703.pt", "1_paper_models/real-time_model"),
        ("no-demographics", "epoch_0552_kappa_0.718.pt", "2_post-paper_models/primary_model_no_demographics"),
        ("no-time", "epoch_0490_kappa_0.724.pt", "2_post-paper_models/primary_model_no_time"),
        ("only-ecg", "epoch_0369_kappa_0.721.pt", "2_post-paper_models/primary_model_only_ecg"),
        ("cyclic", "epoch_0548_kappa_0.725.pt", "3_cyclic-time_models/primary_model"),
        ("cyclic-realtime", "epoch_0472_kappa_0.703.pt", "3_cyclic-time_models/real-time_model"),
    ]:
        url = _jones_spec(v, filename, folder).raw_urls[0]
        assert url.startswith("https://raw.githubusercontent.com/adammj/ecg-sleep-staging/main/models/")
        assert url.endswith(f"/{folder}/{filename}")
        assert "  " not in url and "\t" not in url
        assert url == f"https://raw.githubusercontent.com/adammj/ecg-sleep-staging/main/models/{folder}/{filename}"


def test_alias_resolves():
    alias = THIRD_PARTY_PRETRAINED["jones-ecg"]
    assert alias.get("alias") == "jones-ecg-primary"
    target = THIRD_PARTY_PRETRAINED["jones-ecg-primary"]
    # The alias shares the loader spec (same source / license / classes).
    assert alias["source"] is target["source"]
    assert alias["license"] == target["license"]
    assert alias["class_names"] == target["class_names"]
    assert alias["n_classes"] == target["n_classes"]


def test_class_orders_match_golden_fixtures():
    # class_names must agree with the golden fixtures (tests/models/fixtures/cardioresp/README.md)
    spec = THIRD_PARTY_PRETRAINED["wav2sleep"]
    assert spec["class_names"] == ("W", "Light", "Deep", "REM")
    assert len(THIRD_PARTY_PRETRAINED["wav2sleep-eog"]["class_names"]) == 5
    assert THIRD_PARTY_PRETRAINED["sleepppgnet-replica"]["class_names"] == ("W", "Light", "Deep", "REM")
    for v in _EXPECTED_JONES:
        assert THIRD_PARTY_PRETRAINED[f"jones-ecg-{v}"]["class_names"] == ("W", "N1", "N2", "N3", "REM")


# ── Routing ──────────────────────────────────────────────────────────


def test_third_party_name_routes_to_loader(fakes):
    model = load_from_pretrained("wav2sleep")
    assert isinstance(model, _FakeModel)
    assert model.training is False
    # ensure_checkpoint got the registry name; loader got both downloaded files.
    assert fakes["ensure"] == ["wav2sleep"]
    (args, kwargs) = fakes["from_original"][0]
    assert kwargs["_name"] == "wav2sleep"
    paths = kwargs["_paths"]
    assert [Path(p).name for p in paths] == ["config.yaml", "state_dict.pth"]
    assert all(Path(p).is_file() for p in paths)
    # No HF download happened.
    assert fakes["hf"] == []


@pytest.mark.parametrize("name,variant", [(f"jones-ecg-{v}", v) for v in _EXPECTED_JONES])
def test_jones_variants_route_to_loader(fakes, name, variant):
    model = load_from_pretrained(name)
    assert isinstance(model, _FakeModel)
    (args, kwargs) = fakes["from_original"][0]
    assert kwargs["_name"] == name
    assert [Path(p).name for p in kwargs["_paths"]] == [_EXPECTED_JONES[variant]]


def test_alias_loads_target(fakes):
    model = load_from_pretrained("jones-ecg")
    assert isinstance(model, _FakeModel)
    (args, kwargs) = fakes["from_original"][0]
    assert kwargs["_name"] == "jones-ecg-primary"  # alias resolved to canonical
    assert [Path(p).name for p in kwargs["_paths"]] == [
        _EXPECTED_JONES["primary"]
    ]


def test_non_registry_name_keeps_4rooms_behaviour(monkeypatch, tmp_path):
    # A name outside the registry must use hf_hub_download as before.
    calls = []

    def fake_hf_hub_download(repo_id, filename, **kw):
        calls.append((repo_id, filename))
        p = tmp_path / repo_id / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith("config.json"):
            p.write_text(
                '{"model_class": "physioex.models.chambon2018:Chambon2018Net", '
                '"model_kwargs": {"n_classes": 5, "in_channels": 2, "sf": 100, '
                '"n_times": 3000}}'
            )
        else:
            # Real state dict so torch.load succeeds.
            from physioex.models.chambon2018 import Chambon2018Net
            torch.save(
                Chambon2018Net(n_classes=5, in_channels=2, sf=100, n_times=3000)
                .state_dict(),
                p,
            )
        return str(p)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_hf_hub_download, raising=True
    )
    model = load_from_pretrained("seqsleepnet-huy")
    assert isinstance(model, nn.Module)
    assert model.training is False
    assert calls[0] == ("4rooms/physioex", "seqsleepnet-huy/config.json")
    assert ("4rooms/physioex", "seqsleepnet-huy/model.pt") in calls


# ── Cache re-use ─────────────────────────────────────────────────────


def test_second_call_does_not_download(fakes):
    load_from_pretrained("sleepppgnet-replica")
    load_from_pretrained("sleepppgnet-replica")
    assert fakes["ensure"] == ["sleepppgnet-replica", "sleepppgnet-replica"]
    assert len(fakes["from_original"]) == 2


def test_cache_reuse_of_existing_files(tmp_path, monkeypatch):
    # Pre-populate the cache dir; the real ensure_checkpoint must report a
    # cache hit and never reach the network.
    model_name = "jones-ecg-primary"
    src = THIRD_PARTY_PRETRAINED[model_name]["source"]
    model_dir = (
        tmp_path / "v1" / "foundation_models" / model_name / "checkpoint"
    )
    model_dir.mkdir(parents=True)
    (model_dir / src.filenames[0]).write_text("cached")

    monkeypatch.setenv("PHYSIOEX_CACHE_DIR", str(tmp_path))

    import physioex.models.foundation_checkpoints as fc

    assert fc._cache_root() == tmp_path
    assert fc.ensure_checkpoint(model_name) == str(model_dir / src.filenames[0])
    # No .part leftovers, no re-download.
    assert not list(model_dir.glob("*.part"))


def test_cache_registry_hits_for_all_third_party(tmp_path, monkeypatch):
    # Every registered name must resolve in the real download registry and,
    # with the cache pre-populated, return a path inside its own cache dir.
    import physioex.models.foundation_checkpoints as fc

    monkeypatch.setenv("PHYSIOEX_CACHE_DIR", str(tmp_path))
    for name in THIRD_PARTY_PRETRAINED:
        src = THIRD_PARTY_PRETRAINED[THIRD_PARTY_PRETRAINED[name].get("alias", name)]["source"]
        _precache(tmp_path, name, src)
        result = fc.ensure_checkpoint(name)
        assert result.startswith(str(tmp_path)), name
        assert name in fc.CHECKPOINT_REGISTRY


def _precache(tmp_path, name, src):
    d = tmp_path / "v1" / "foundation_models" / name / "checkpoint"
    d.mkdir(parents=True, exist_ok=True)
    # Lay down every filename; the primary file (local_filename or first
    # filename) is what ensure_checkpoint checks for a cache hit.
    for f in set(src.filenames) | {src.local_filename or Path(src.filenames[0]).name}:
        (d / f).write_text("cached")
    return str(d / (src.local_filename or Path(src.filenames[0]).name))


# ── License warnings ─────────────────────────────────────────────────


def test_license_warning_for_agpl(fakes, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="physioex.models"):
        load_from_pretrained("jones-ecg-primary")
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("AGPL-3.0" in m and "jones-ecg-primary" in m for m in msgs)


def test_license_warning_for_noncommercial(fakes, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="physioex.models"):
        load_from_pretrained("sleepppgnet-replica")
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("sleepppgnet-replica" in m for m in msgs)


def test_no_license_warning_for_mit(fakes, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="physioex.models"):
        load_from_pretrained("wav2sleep")
        load_from_pretrained("wav2sleep-eog")
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not msgs


def test_license_predicate():
    assert _license_is_restrictive("AGPL-3.0")
    assert _license_is_restrictive("fair-noncommercial-research")
    assert not _license_is_restrictive("MIT")
    assert not _license_is_restrictive("Apache-2.0")
    assert not _license_is_restrictive("BSD-3-Clause")


# ── list_pretrained ──────────────────────────────────────────────────


def test_list_pretrained_contents():
    entries = list_pretrained()
    by_name = {e["name"]: e for e in entries}
    for name in THIRD_PARTY_PRETRAINED:
        e = by_name[name]
        assert set(e) == {"name", "source", "license", "reference",
                          "n_classes", "class_names"}
        assert e["license"] == THIRD_PARTY_PRETRAINED[name]["license"]
        assert e["n_classes"] == THIRD_PARTY_PRETRAINED[name]["n_classes"]
        assert e["source"]  # non-empty
    # A couple of the documented 4rooms models are listed without network.
    for name in ("seqsleepnet-huy", "protosleepnet-gagliardi"):
        assert by_name[name]["source"].startswith("hf: 4rooms/physioex/")
        assert by_name[name]["license"] == "Apache-2.0"


def test_verbose_prints_metadata(fakes, capsys):
    load_from_pretrained("wav2sleep", verbose=True)
    out = capsys.readouterr().out
    assert "Model:     wav2sleep" in out
    assert "Reference:" in out
    assert "License:   MIT" in out
    assert "Classes:   W, Light, Deep, REM" in out
    assert "Input:" in out


# ── Real downloads (gated) ───────────────────────────────────────────


@pytest.mark.hf
def test_real_download_huggingface_file():
    # One real HF file download per source type: wav2sleep (2 small files).
    import physioex.models.foundation_checkpoints as fc

    name = "wav2sleep"
    try:
        result = fc.ensure_checkpoint(name)
        src = THIRD_PARTY_PRETRAINED[name]["source"]
        model_dir = Path(result).parent
        for f in src.filenames:
            assert (model_dir / f).exists(), f
        assert (model_dir / src.filenames[0]).stat().st_size > 0
    finally:
        fc.clear_cache(name)


@pytest.mark.hf
def test_real_download_raw_url():
    # One real raw-URL download: the Jones primary checkpoint.
    import physioex.models.foundation_checkpoints as fc

    name = "jones-ecg-primary"
    try:
        paths = fc.ensure_checkpoint(name)
        assert isinstance(paths, list)
        assert len(paths) == 1
        assert Path(paths[0]).stat().st_size > 1000
    finally:
        fc.clear_cache(name)
