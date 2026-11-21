"""Tests for the pretrained cardiorespiratory examples (wav2sleep, sleepppgnet, jones-ecg).

No network and no data are needed: the scripts are checked for syntax,
their ``paper_metrics.json`` for well-formedness, and their ``main()`` for
graceful skipping on an empty data root (the latter needs the not-yet-shipped
``physioex.train.replication`` module, so it is import-gated).
"""
import importlib.util
import json
import py_compile
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "pretrained"
)
FOLDERS = ("wav2sleep", "jones-ecg", "sleepppgnet")
SCRIPTS = ("quickstart.py", "verify.py")


def _folder(folder: str) -> Path:
    return EXAMPLES_DIR / folder


def _load_module(folder: str, script: str):
    """Import an example script as a standalone module (no package context)."""
    path = _folder(folder) / script
    spec = importlib.util.spec_from_file_location(
        f"example_{folder}_{script[:-3]}", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("folder", FOLDERS)
@pytest.mark.parametrize("script", SCRIPTS)
def test_script_compiles(folder, script):
    py_compile.compile(str(_folder(folder) / script), doraise=True)


@pytest.mark.parametrize("folder", FOLDERS)
def test_paper_metrics_well_formed(folder):
    header = json.loads((_folder(folder) / "paper_metrics.json").read_text())
    assert "model" in header and "targets" in header
    for target in header["targets"]:
        assert "dataset" in target
        assert "condition" in target
        assert target["value"] > 0 and target["value"] < 1, (
            f"{folder}: target value {target['value']} not in (0, 1)"
        )


@pytest.mark.parametrize("folder", FOLDERS)
def test_verify_build_parser(folder, tmp_path):
    mod = _load_module(folder, "verify.py")
    parser = mod.build_parser()
    args = parser.parse_args([])  # defaults must not raise
    assert args.out == "./verify_results.json"
    assert args.max_nights is None


@pytest.mark.parametrize("folder", FOLDERS)
def test_verify_main_skips_without_data(folder, tmp_path, monkeypatch):
    pytest.importorskip("physioex.train.replication")
    monkeypatch.chdir(tmp_path)  # keep the default ./verify_results.json local
    mod = _load_module(folder, "verify.py")
    data_root = tmp_path / "empty_data"
    data_root.mkdir()
    out = tmp_path / "verify_results.json"
    rc = mod.main(["--data-root", str(data_root), "--out", str(out)])
    assert rc == 0, "missing data must SKIP, not FAIL"
    result = json.loads(out.read_text())
    assert result["rows"], "expected at least one row"
    assert all(row["status"] == "SKIP" for row in result["rows"]), (
        f"all rows should be SKIP with an empty data root, got "
        f"{[row['status'] for row in result['rows']]}"
    )
    assert "run" in result and "per_record_kappa" in result


@pytest.mark.parametrize("folder", FOLDERS)
@pytest.mark.parametrize("script", SCRIPTS)
def test_docs_exist(folder, script):
    assert (_folder(folder) / "README.md").is_file()
    assert (_folder(folder) / "TRAINING.md").is_file()
