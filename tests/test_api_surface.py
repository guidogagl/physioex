"""API-surface guard: every advertised public symbol must be importable, and
every declared console script must resolve to a real callable.

This is the capillary net that catches a rename/move breaking a documented
export or an entry point, independent of the behavioural tests.
"""
import importlib

import pytest

# Packages that advertise a public API via __all__.
PACKAGES = [
    "physioex.data",
    "physioex.data.datasets",
    "physioex.data.steps",
    "physioex.data.readers",
    "physioex.models",
    "physioex.explain.foundational",
    "physioex.explain.posthoc",
    "physioex.explain.prototypes.posthoc",
]


@pytest.mark.parametrize("pkg_name", PACKAGES)
def test_all_exports_are_importable(pkg_name):
    pkg = importlib.import_module(pkg_name)
    exported = getattr(pkg, "__all__", None)
    assert exported, f"{pkg_name} declares no __all__"
    missing = [name for name in exported if not hasattr(pkg, name)]
    assert not missing, f"{pkg_name} __all__ lists unresolved symbols: {missing}"


def test_data_key_symbols_present():
    import physioex.data as d

    for name in (
        "BasePhysioDataset", "MultiDataset", "PreprocessingPipeline",
        "dict_collate_fn", "SleepEvent", "get_preset",
    ):
        assert hasattr(d, name)


def test_models_encoders_present():
    import physioex.models as m

    for name in (
        "FoundationEncoder", "CBraModEncoder", "BENDREncoder", "LaBraMEncoder",
        "BIOTEncoder", "SleepFMEncoder", "TFCEncoder", "REVEEncoder",
        "SJEEncoder", "NeuroLMEncoder", "load_from_pretrained",
    ):
        assert hasattr(m, name)


# ── Console-script entry points ──────────────────────────────────────

CONSOLE_SCRIPTS = {
    "train": ("physioex.train.bin.train", "train_script"),
    "finetune": ("physioex.train.bin.finetune", "finetune_script"),
    "test_model": ("physioex.train.bin.test", "test_script"),
}


@pytest.mark.parametrize("script", list(CONSOLE_SCRIPTS), ids=lambda s: s)
def test_console_scripts_resolve(script):
    module_path, func_name = CONSOLE_SCRIPTS[script]
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name, None)
    assert callable(fn), f"entry point {script} -> {module_path}:{func_name} not callable"
