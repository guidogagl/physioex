# Training Schema - ProtosleepNet-Gagliardi

## Overview

Questo documento definisce lo schema di divisione training/testing per il pretraining di SleepTokenizer, ottimizzato per massimizzare la copertura dell'eterogeneità dei canali PSG (Polysomnography) e garantire la generalizzazione su dataset mai visti.

**Obiettivo:** Il training set deve fornire la rappresentazione più completa possibile del problema per generalizzare su dataset con caratteristiche diverse (sampling rate, reference scheme, naming conventions, popolazioni).

**Data:** 2025-01-13

---

## Dataset Disponibili (Riepilogo)

| Dataset | Recordings | Sampling Rate | Elementi Unici |
|---------|-----------|---------------|----------------|
| SHHS | 8,444 | 125-128 Hz | Multi-center, longitudinal (2 visite), generic naming |
| MrOS | 3,135 | ~200 Hz | Maschi 65+, LEG EMG, LOC/ROC naming |
| MESA | 2,237 | 200-512 Hz | Numbered channels (EEG1), 4 etnie |
| STAGES | 1,914 | 200-512 Hz | 4 naming conventions, 13 siti, LEG EMG |
| MASS | 597 | 256 Hz | **CLE reference**, 5 cohort, 10-20 completo |
| WSC | ~4,000* | 512 Hz | Underscore naming, **average reference**, longitudinal (5 visite) |
| SleepEDF | 197 | 100 Hz | **100Hz**, **Fpz-Cz/Pz-Oz** (non-standard) |
| HomePAP | 373 | 200-512 Hz | Massima eterogeneità naming, lab+home |
| HMC | 151 | 256 Hz | AASM completo |
| DCSM | ~200 | 256 Hz | Standard europeo |
| Parkinson's | 86 night | 500 Hz | **500Hz**, mixed montage, patologia |
| Alzheimer's | 69 | 200 Hz | **Average reference**, patologia |

\*WSC: ~500 soggetti × 5 visite longitudinali

---

## Divisione Training/Testing

### Training Set (~15,000 recordings)

| Dataset | Sottogruppi | Recordings | Elementi di Eterogeneità |
|---------|-------------|-----------|-------------------------|
| **SHHS** | Visit 1 | 5,793 | Multi-center baseline, generic naming |
| **MESA** | Tutti | 2,237 | **Numbered channels**, diversità etnica (White/Hispanic) |
| **STAGES** | Tutti | 1,914 | **4 naming conventions** (Grael, hyphenated, concatenated, bare), 13 siti |
| **MrOS** | Tutti | 3,135 | **Maschi 65+**, LEG EMG, LOC/ROC naming |
| **WSC** | Visit 1 | ~1,500* | **512Hz**, underscore naming, longitudinal baseline |
| **Parkinson's** | Night + **HOA** (healthy) | ~40 | **500Hz**, mixed montage, healthy older adults |
| **Alzheimer's** | **HC** (healthy) | 32 | 200Hz, **average reference**, healthy controls |
| **SleepEDF** | Tutti | 197 | **100Hz**, **Fpz-Cz/Pz-Oz** (posizioni non-standard) |
| **HMC** | Tutti | 151 | AASM completo a 256Hz |
| **HomePAP** | Tutti | 373 | Massima eterogeneità naming (`C4 / M1`, `E1 / E2`) |

\*WSC-Visit 1 è la visita più grande (baseline, nessun attrition)

**Totale Training: ~15,400 recordings**

---

### Testing Set (~5,500 recordings)

| Dataset | Sottogruppi | Recordings | Generalizzazione Testata |
|---------|-------------|-----------|--------------------------|
| **SHHS** | Visit 2 | 2,651 | **Longitudinale** (stessi soggetti ~4 anni dopo) |
| **WSC** | Visit 2-5 | ~2,500 | **Long-term longitudinal** (visite successive) |
| **MASS** | Tutte (5 cohort) | 597 | **CLE reference**, SAF annotations (SS04), 20s epochs |
| **Parkinson's** | Night + **PD** (disease) | ~46 | **500Hz**, mixed montage, **patologia Parkinson** |
| **Alzheimer's** | **AD** (disease) | 37 | **Average reference**, **patologia Alzheimer** |
| **DCSM** | Tutti | ~200 | Standard europeo |

**Totale Testing: ~6,000 recordings**

---

## Copertura Eterogeneità

### Sampling Rate

| Rate (Hz) | Training | Testing | Coverage |
|-----------|----------|---------|----------|
| 100 | ✅ SleepEDF | — | Base |
| 125-128 | ✅ SHHS-1 | ✅ SHHS-2, WSC | Variazione |
| 200 | ✅ MESA, Parkinson, Alzheimer | ✅ DCSM | Standard |
| 256 | ✅ HMC, MESA, STAGES, HomePAP | ✅ MASS | Alta qualità |
| 500 | ✅ Parkinson (HOA) | ✅ Parkinson (PD) | Massima |
| 512 | ✅ WSC-1 | ✅ WSC-2-5 | Massima |

**Coverage:** 100-512 Hz completo

---

### Reference Schemes

| Reference | Training | Testing | Note |
|-----------|----------|---------|------|
| Bipolar (C3-M2) | ✅ HMC, SleepEDF | — | Standard |
| Generic (EEG) | ✅ SHHS | — | Multi-center |
| Underscore (C3_M2) | ✅ WSC | — | WSC style |
| Numbered (EEG1) | ✅ MESA | — | MESA style |
| Grael (EEG_C3-A2) | ✅ STAGES | — | STAGES style |
| Non-standard pos (Fpz-Cz) | ✅ SleepEDF | — | SleepEDF unique |
| **CLE (C3-CLE)** | ❌ No | ✅ MASS | **Tested generalization** |
| **Average (C3-REF)** | ✅ Alzheimer (HC), Parkinson | ✅ Alzheimer (AD) | Mixed in train, pure in test |
| **Mixed** | ✅ Parkinson | — | Mixed montage |

**Coverage:** Tutti gli schemi di reference rappresentati, con CLE testato per generalizzazione

---

### Naming Conventions

| Convention | Training | Esempi |
|------------|----------|--------|
| Standard hyphenated | ✅ HMC, SleepEDF | `C3-M2`, `Fpz-Cz` |
| Underscore | ✅ WSC | `C3_M2` |
| Numbered | ✅ MESA | `EEG1`, `EEG2` |
| Grael/Compumedics | ✅ STAGES | `EEG_C3-A2` |
| Concatenated | ✅ STAGES | `C3M2` |
| Generic | ✅ SHHS | `EEG`, `EOG` |
| LOC/ROC | ✅ MrOS, STAGES | `LOC`, `ROC` |
| Bare differential | ✅ HomePAP | `C3 / M1`, `E1 / E2` |
| CLE-based | ❌ No | `C3-CLE` (testing only) |
| Average-based | ✅ Alzheimer (HC) | `C3-REF` |

**Coverage:** Tutte le convenzioni di naming principali nel training

---

### Popolazioni e Stati di Salute

| Gruppo | Training | Testing |
|--------|----------|---------|
| Adulto generale | ✅ SHHS, MESA, STAGES, WSC, HMC, SleepEDF, HomePAP | — |
| Maschi 65+ | ✅ MrOS | — |
| Multi-etnia | ✅ MESA (White/Hispanic) | ✅ MESA (Black/Chinese in test se divisi) |
| Multi-center | ✅ SHHS, STAGES (13 siti) | — |
| Healthy (Parkinson) | ✅ HOA | — |
| Disease (Parkinson) | — | ✅ PD |
| Healthy (Alzheimer) | ✅ HC | — |
| Disease (Alzheimer) | — | ✅ AD |

**Note:**
- Parkinson e Alzheimer sono divisi per healthy/disease
- La generalizzazione da healthy a disease è testata nel testing set

---

### Longitudinalità

| Dataset | Training | Testing | Intervallo |
|---------|----------|---------|-----------|
| **SHHS** | Visit 1 | Visit 2 | ~4 anni |
| **WSC** | Visit 1 | Visit 2-5 | 4 anni tra visite |
| **Mass** | — | SS01-SS05 | Non longitudinale (cohort diverse) |

**Note:**
- Visit 1 = baseline (no attrition)
- Visit 2+ = attrition naturale + drift temporale
- La generalizzazione longitudinal è testata nel testing set

---

## Decisioni Chiave

### 1. Dataset longitudinali divisi per visita
- **SHHS:** Visit 1 → train, Visit 2 → test
- **WSC:** Visit 1 → train, Visit 2-5 → test
- **Razione:** Evita data leakage, testa generalizzazione longitudinal

### 2. Patologie divise per healthy/disease
- **Parkinson:** HOA (healthy) → train, PD (disease) → test
- **Alzheimer:** HC (healthy) → train, AD (disease) → test
- **Razione:** Testa generalizzazione da healthy a disease, mantiene caratteristiche di segnale (500Hz, avg reference) in training

### 3. MASS tutto in testing
- **Razione:** CLE reference è unico e non presente in training → ottimo test case per generalizzazione
- **SS04 (SAF-only)** è un edge case perfetto per testing

### 4. WSC-Visit 1 nel training
- **Razione:** È la visita più grande (~1,500), mantiene coverage 512Hz nel training

---

## Riferimenti

- **Canali table:** `/home/dev/physioex/physioex/data/datasets/channels_table.md`
- **NSRR:** [sleepdata.org](https://sleepdata.org)
- **PhysioNet:** [physionet.org](https://physionet.org)

---

## Note Implementative

### Per caricare il training set:
```python
# SHHS Visit 1
from physioex.data.datasets import SHHSDataset
shhs = SHHSDataset(visit=1, ...)

# WSC Visit 1
from physioex.data.datasets import WSCDataset
wsc = WSCDataset(visit=1, ...)

# MESA (White + Hispanic only se si vuole dividere per etnia)
from physioex.data.datasets import MESADataset
mesa = MESADataset(...)

# Parkinson (solo healthy)
from physioex.data.datasets import ParkinsonDataset
park_hoa = ParkinsonDataset(subject_ids=hoa_ids, ...)
```

### Per caricare il testing set:
```python
# SHHS Visit 2
shhs2 = SHHSDataset(visit=2, ...)

# WSC Visit 2-5
wsc_tests = [WSCDataset(visit=v, ...) for v in [2, 3, 4, 5]]

# MASS (tutte le cohort)
from physioex.data.datasets import MASSDataset
mass = MASSDataset(...)

# Parkinson (solo disease)
park_pd = ParkinsonDataset(subject_ids=pd_ids, ...)
```

---

*Generato da analisi canali PhysioEx v2.0 - 2025-01-13*
