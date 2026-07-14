# Testing suite architecture (Diagram B)

Living class/structure diagram of the **pytest** test suite. Kept in sync with
`tests/` at every phase and checked for congruence against the
[library diagram](../library/overview.md) (Diagram A). Legend: ✅ in place
(Fase A) · ⏳ planned (Fase B/C).

## Scaffolding: fixtures & factories

```mermaid
classDiagram
    class conftest {
        <<pytest>>
        +pytest_collection_modifyitems() ✅
        +data_dir() fixture ✅
        +cache_dir() fixture ✅
        +edf_factory() fixture ✅
        +fake_edf_dataset() fixture ✅
    }
    class factories_edf {
        <<module>>
        +write_fake_edf(path, n_channels, duration_sec) ✅
        +write_fake_annotations_edf(path, stages) ✅
    }
    class FakeEDFDataset {
        +__init__(root, subject_id) ✅
        +_list_subjects() ✅
        +_read_subject_labels(spec) ✅
    }
    class factories_datasets {
        <<module ⏳>>
        +make_sleepedf_subject() ⏳
        +make_dcsm_subject() ⏳
        +make_mass_subject() ⏳
        +make_nsrr_subject() ⏳
        +make_alzheimers_subject() ⏳
        +make_parkinsons_subject() ⏳
        +make_wsc_subject() ⏳
    }

    BasePhysioDataset <|-- FakeEDFDataset
    factories_edf ..> FakeEDFDataset : builds subject files
    conftest ..> factories_edf : wraps
    conftest ..> factories_datasets : wraps ⏳
    factories_datasets ..> factories_edf : reuse EDF writer
```

## Markers (gating)

```mermaid
classDiagram
    class Markers {
        <<pytest.ini>>
        +unit
        +integration
        +real_data  → skip unless PHYSIOEX_TEST_REAL_DATA=1
        +gpu        → skip unless torch.cuda.is_available()
        +hf         → skip unless PHYSIOEX_TEST_HF=1
        +slow
    }
```

## Target module layout ↔ library mapping

Each test module targets exactly the public symbols documented in Diagram A
(congruence rule). ✅ = scaffolding present, ⏳ = to be migrated/added.

```mermaid
flowchart LR
    subgraph tests
        CF["conftest.py ✅"]
        FAC["factories/ ✅"]
        TD["data/ ⏳"]
        TM["models/ ⏳"]
        TT["train/ ⏳"]
        TE["explain/ ⏳"]
        TA["test_api_surface.py ⏳"]
    end
    TD --> DATA["physioex.data"]
    TM --> MODELS["physioex.models"]
    TT --> TRAIN["physioex.train"]
    TE --> EXPLAIN["physioex.explain"]
    TA --> ALL["all __all__ exports"]
    FAC --> DATA
```

## Conventions

- **pytest-only**: plain `assert`, fixtures, `@pytest.mark.parametrize`; no
  `report()/passed/failed/__main__` scaffolding (removed in Fase B).
- **Single source of truth for synthetic data**: `tests/factories/` (the old
  `tests/test_raw_dataset_integration` re-exports become imports in Fase B).
- **Coverage**: measured over `physioex` (legacy modules omitted), gate enforced
  in CI (Fase D).
