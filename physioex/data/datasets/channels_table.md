# PhysioEx Dataset Channel Tables

Comprehensive reference of available channels, modality types, and sampling frequencies for all supported datasets in PhysioEx.

**Modality Type Codes:**
- `EEG=0, EOG=1, EMG=2, ECG=3, RESP=4, SPO2=5, HR=6, LEG=7, POS=8, ACCEL=9, LIGHT=10, SOUND=11, TEMP=12, DEVICE=13, OTHER=14`

**Usability Key:**
- ✅ **Usable**: Physiological signal suitable for sleep staging (≥50 Hz recommended)
- ⚠️ **Limited**: Low frequency or processed signal (may be useful for specific tasks)
- ❌ **Not usable**: Technical markers, event flags, or processed parameters

---

## 1. HMC (Haaglanden Medisch Centrum)

**Source:** [PhysioNet](https://physionet.org/content/hmc-sleep-staging/1.1/) - 151 recordings
**Native Sampling Rate:** 256 Hz (all channels)
**Epoch Length:** 30 seconds

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| EEG C4-M1 | EEG (0) | 256 | ✅ Usable | Primary central EEG |
| EEG C3-M2 | EEG (0) | 256 | ✅ Usable | Secondary central EEG |
| EEG F4-M1 | EEG (0) | 256 | ✅ Usable | Frontal EEG |
| EEG O2-M1 | EEG (0) | 256 | ✅ Usable | Occipital EEG |
| EOG E1-M2 | EOG (1) | 256 | ✅ Usable | Left EOG |
| EOG E2-M2 | EOG (1) | 256 | ✅ Usable | Right EOG |
| EMG chin | EMG (2) | 256 | ✅ Usable | Chin EMG |
| ECG | ECG (3) | 256 | ✅ Usable | Single modified lead II |

**Modality Assessment:** ✅ All modalities correctly classified. No issues.

**Available Modalities for Sleep Staging:** EEG, EOG, EMG, ECG

**Summary:** HMC provides the complete AASM-recommended montage at a uniform 256 Hz sampling rate. All channels are high-quality physiological signals suitable for sleep staging. The horizontal EOG can be derived by subtracting E1-M2 and E2-M2.

---

## 2. SleepEDF (Sleep Cassette)

**Source:** [PhysioNet](https://physionet.org/physiobank/database/sleep-edfx/) - 197 recordings
**Native Sampling Rate:** 100 Hz (EEG/EOG), 1 Hz (EMG envelope/RESP/TEMP)
**Epoch Length:** 30 seconds

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| EEG Fpz-Cz | EEG (0) | 100 | ✅ Usable | Frontal EEG (alternative to C4-A1) |
| EEG Pz-Oz | EEG (0) | 100 | ✅ Usable | Parietal-occipital EEG |
| EOG horizontal | EOG (1) | 100 | ✅ Usable | Horizontal EOG |
| EMG submental | EMG (2) | 1 | ⚠️ Limited | **Envelope** signal (1 Hz), not raw EMG |
| Resp oro-nasal | RESP (4) | 1 | ⚠️ Limited | Respiratory effort (1 Hz) |
| Temp rectal | TEMP (12) | 1 | ⚠️ Limited | Body temperature (1 Hz) |
| Event marker | DEVICE (13) | 1 | ❌ Not usable | Technical annotation channel |

**Modality Assessment:** ⚠️ **Important note**: The EMG channel in SleepEDF is a **1 Hz envelope signal**, not raw EMG. This is often misinterpreted. The envelope is computed by high-pass filtering, rectifying, and low-pass filtering the raw EMG.

**Missing/Incorrect Modalities:**
- EMG submental is correctly classified but should be marked as "processed envelope" not raw EMG

**Available Modalities for Sleep Staging:** EEG, EOG (EMG is envelope only, not suitable for raw signal analysis)

**Summary:** SleepEDF uses non-standard electrode placements (Fpz-Cz/Pz-Oz instead of C4-A1/C3-A2). The 1 Hz EMG envelope and respiratory/temperature channels are processed parameters, not raw physiological signals. Only EEG and EOG are at full physiological sampling rates.

---

## 3. MASS (Montreal Archive of Sleep Studies)

**Source:** [Borealis Data](https://borealisdata.ca) - SS01 to SS05 cohorts
**Native Sampling Rate:** 256 Hz (all cohorts)
**Epoch Length:** 30s (SS01, SS03) or 20s (SS02, SS04, SS05) → 30s with padding

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| EEG C3-CLE | EEG (0) | 256 | ✅ Usable | CLE = Contralateral Linked Ear reference |
| EEG C4-CLE | EEG (0) | 256 | ✅ Usable | Standard bipolar derivation |
| EEG Cz-CLE | EEG (0) | 256 | ✅ Usable | Central midline |
| EEG F3-CLE | EEG (0) | 256 | ✅ Usable | Left frontal |
| EEG F4-CLE | EEG (0) | 256 | ✅ Usable | Right frontal |
| EEG Fz-CLE | EEG (0) | 256 | ✅ Usable | Frontal midline |
| EEG O1-CLE | EEG (0) | 256 | ✅ Usable | Left occipital |
| EEG O2-CLE | EEG (0) | 256 | ✅ Usable | Right occipital |
| EEG Pz-CLE | EEG (0) | 256 | ✅ Usable | Parietal midline |
| EEG C3-LER | EEG (0) | 256 | ✅ Usable | LER = Linked Ear Reference (~6 subjects) |
| EEG C4-LER | EEG (0) | 256 | ✅ Usable | Alternative reference for C3-CLE |
| EOG Left Horiz | EOG (1) | 256 | ✅ Usable | Horizontal EOG (left) |
| EOG Right Horiz | EOG (1) | 256 | ✅ Usable | Horizontal EOG (right) |
| EOG Upper Vertic | EOG (1) | 256 | ✅ Usable | Vertical EOG (upper) |
| EOG Lower Vertic | EOG (1) | 256 | ✅ Usable | Vertical EOG (lower) |
| EMG Chin1 | EMG (2) | 256 | ✅ Usable | Chin EMG primary |
| EMG Chin2 | EMG (2) | 256 | ✅ Usable | Chin EMG secondary |
| EMG Chin3 | EMG (2) | 256 | ✅ Usable | Chin EMG tertiary |
| ECG I | ECG (3) | 256 | ✅ Usable | Lead I ECG |
| ECG II | ECG (3) | 256 | ✅ Usable | Lead II ECG |

**Modality Assessment:** ✅ All modalities correctly classified. MASS uses two reference schemes: CLE (Contralateral Linked Ear, majority) and LER (Linked Ear Reference, ~6 subjects in SS01).

**Cohort-Specific Notes:**
- SS01 (30s epochs): 200 subjects, AASM scoring
- SS02 (20s epochs): 100 subjects, R&K scoring
- SS03 (30s epochs): 100 subjects, AASM scoring
- SS04 (20s epochs): 97 subjects, R&K scoring, SAF annotations only
- SS05 (20s epochs): 100 subjects, R&K scoring, SAF annotations only

**Available Modalities for Sleep Staging:** EEG, EOG, EMG, ECG

**Summary:** MASS is the largest publicly available sleep dataset with full 10-20 EEG montage at 256 Hz. All channels are high-quality raw signals. The 20s cohorts use Phan's ±5s padding convention to create 30s windows for compatibility.

---

## 4. SHHS (Sleep Heart Health Study)

**Source:** [NSRR](https://sleepdata.org/datasets/shhs) - Visit 1: 5793 subjects, Visit 2: 2651 subjects
**Native Sampling Rate:** 125 Hz (SHHS-1), 125/128 Hz (SHHS-2, varies by recording)
**Epoch Length:** 30 seconds

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| EEG | EEG (0) | 125-128 | ✅ Usable | C4/A1 primary (bipolar) |
| EEG(sec) | EEG (0) | 125-128 | ✅ Usable | C3/A2 secondary |
| EOG(L) | EOG (1) | 125-128 | ✅ Usable | Left EOG |
| EOG(R) | EOG (1) | 125-128 | ✅ Usable | Right EOG |
| EMG | EMG (2) | 125-128 | ✅ Usable | Chin EMG |
| ECG | ECG (3) | 125-128 | ✅ Usable | ECG lead |

**Modality Assessment:** ✅ All modalities correctly classified. SHHS uses a simplified AASM montage.

**Available Modalities for Sleep Staging:** EEG, EOG, EMG, ECG

**Summary:** SHHS is the largest multi-center sleep study cohort. The montage follows AASM minimal recommendations. Channel names are generic ("EEG" not "EEG C4-A1") but represent standard bipolar derivations. Sampling rate is consistent but slightly lower than MASS/HMC.

---

## 5. MESA (Multi-Ethnic Study of Atherosclerosis)

**Source:** [NSRR](https://sleepdata.org/datasets/mesa) - 2,237 subjects
**Native Sampling Rate:** Variable (typically 200-512 Hz)
**Epoch Length:** 30 seconds

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| EEG1 | EEG (0) | 200-512 | ✅ Usable | Primary EEG (C4-M1 or similar) |
| EEG2 | EEG (0) | 200-512 | ✅ Usable | Secondary EEG |
| EEG3 | EEG (0) | 200-512 | ✅ Usable | Tertiary EEG |
| EEG C4-M1 | EEG (0) | 200-512 | ✅ Usable | Standard central derivation |
| EEG C3-M2 | EEG (0) | 200-512 | ✅ Usable | Alternative central derivation |
| EEG Cz-Oz | EEG (0) | 200-512 | ✅ Usable | Midline derivation |
| EOG-L | EOG (1) | 200-512 | ✅ Usable | Left EOG |
| EOG-R | EOG (1) | 200-512 | ✅ Usable | Right EOG |
| EMG | EMG (2) | 200-512 | ✅ Usable | Chin EMG (bipolar LCHIN-RCHIN) |
| ECG | ECG (3) | 200-512 | ✅ Usable | ECG lead |

**Modality Assessment:** ✅ All modalities correctly classified. MESA uses numbered channels (EEG1/2/3) and standard bipolar names.

**Available Modalities for Sleep Staging:** EEG, EOG, EMG, ECG

**Summary:** MESA focuses on underrepresented ethnic groups (Black, White, Hispanic, Chinese-American). Channel naming is heterogeneous with both numbered (EEG1) and anatomical (C4-M1) conventions. Sampling rate varies but is generally high (>200 Hz).

---

## 6. MrOS (Osteoporotic Fractures in Men Study)

**Source:** [NSRR](https://sleepdata.org/datasets/mros) - 3,135 subjects
**Native Sampling Rate:** ~200 Hz (varies)
**Epoch Length:** 30 seconds

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| C4 / A1 | EEG (0) | ~200 | ✅ Usable | Differential pair (referenced) |
| C3 / A2 | EEG (0) | ~200 | ✅ Usable | Differential pair (referenced) |
| C4-M1 | EEG (0) | ~200 | ✅ Usable | Pre-referenced variant |
| C3-M2 | EEG (0) | ~200 | ✅ Usable | Pre-referenced variant |
| EEG2 | EEG (0) | ~200 | ✅ Usable | Numbered variant |
| EEG3 | EEG (0) | ~200 | ✅ Usable | Numbered variant |
| LOC | EOG (1) | ~200 | ✅ Usable | Left outer canthus EOG |
| ROC | EOG (1) | ~200 | ✅ Usable | Right outer canthus EOG |
| EOG-L | EOG (1) | ~200 | ✅ Usable | Alternative left EOG name |
| EOG-R | EOG (1) | ~200 | ✅ Usable | Alternative right EOG name |
| EMG | EMG (2) | ~200 | ✅ Usable | Chin EMG (bipolar) |
| L Chin | EMG (2) | ~200 | ✅ Usable | Left chin electrode |
| R Chin | EMG (2) | ~200 | ✅ Usable | Right chin electrode |
| ECG L | ECG (3) | ~200 | ✅ Usable | Left ECG lead |
| ECG R | ECG (3) | ~200 | ✅ Usable | Right ECG lead |

**Modality Assessment:** ✅ All modalities correctly classified. MrOS uses both differential pairs and pre-referenced montages.

**Available Modalities for Sleep Staging:** EEG, EOG, EMG, ECG

**Summary:** MrOS focuses on older men (65+ years) studying sleep-disordered breathing relationships with falls, fractures, and vascular disease. The dataset uses standard AASM electrode positions with heterogeneous naming (LOC/ROC, EOG-L/R).

---

## 7. HomePAP (Home Positive Airway Pressure)

**Source:** [NSRR](https://sleepdata.org/datasets/homepap) - 373 subjects
**Native Sampling Rate:** Variable (200-512 Hz)
**Epoch Length:** 30 seconds

**Three subsets available:**
- **lab-full**: Full in-lab PSG
- **lab-split**: Split-night in-lab PSG
- **home**: Home-based portable monitoring

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| C4 / M1 | EEG (0) | 200-512 | ✅ Usable | Primary EEG (most common) |
| C3 / M2 | EEG (0) | 200-512 | ✅ Usable | Secondary EEG |
| C4-M1 | EEG (0) | 200-512 | ✅ Usable | Pre-referenced (lab-split, some lab-full) |
| C3-M2 | EEG (0) | 200-512 | ✅ Usable | Pre-referenced variant |
| C4 / A1 | EEG (0) | 200-512 | ✅ Usable | Alternative reference |
| C3 / A2 | EEG (0) | 200-512 | ✅ Usable | Alternative reference |
| C4 | EEG (0) | 200-512 | ⚠️ Check | Bare electrode (no ref, some subjects) |
| C3 | EEG (0) | 200-512 | ⚠️ Check | Bare electrode (no ref, some subjects) |
| E1 / E2 | EOG (1) | 200-512 | ✅ Usable | Dominant EOG pair (212 subjects) |
| E-1 / E-2 | EOG (1) | 200-512 | ✅ Usable | Hyphenated variant |
| E1-E2 | EOG (1) | 200-512 | ✅ Usable | Pre-referenced EOG |
| E1 / M2 | EOG (1) | 200-512 | ✅ Usable | Alternative EOG derivation |
| E2 / M1 | EOG (1) | 200-512 | ✅ Usable | Alternative EOG derivation |
| LOC / ROC | EOG (1) | 200-512 | ✅ Usable | Left/Right outer canthus |
| L-EOG / R-EOG | EOG (1) | 200-512 | ✅ Usable | Alternative EOG naming |
| Lchin / Cchin | EMG (2) | 200-512 | ✅ Usable | Standard chin EMG (125+ subjects) |
| Chin1 / Chin2 | EMG (2) | 200-512 | ✅ Usable | Numbered chin EMG |
| EMG1 / EMG2 | EMG (2) | 200-512 | ✅ Usable | Numbered variant (some lab-full) |
| ECG1 / ECG3 | ECG (3) | 200-512 | ✅ Usable | Standard ECG derivation (145 subjects) |
| ECG3-ECG1 | ECG (3) | 200-512 | ✅ Usable | Pre-referenced variant (29 subjects) |

**Modality Assessment:** ✅ All modalities correctly classified. HomePAP has the most heterogeneous channel naming with multiple conventions.

**Potential Issues:**
- Some subjects have bare electrodes (C4, C3) without explicit reference - may be average-referenced or need metadata verification
- MNE fallback required for home subset EDFs (non-ASCII in physical dimension field)

**Available Modalities for Sleep Staging:** EEG, EOG, EMG, ECG

**Summary:** HomePAP compares lab vs. home PSG for OSA diagnosis. Channel naming varies significantly across subsets and subjects. The most common montage is C4/M1 (EEG), E1-E2 (EOG), Lchin-Cchin (EMG), ECG1-ECG3. Some home recordings use non-compliant EDFs requiring MNE reader.

---

## 8. DCSM (Danish Center for Sleep Medicine)

**Source:** PhysioNet - ~200 recordings
**Native Sampling Rate:** 256 Hz
**Epoch Length:** 30 seconds

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| C3-M2 | EEG (0) | 256 | ✅ Usable | Standard central EEG |
| C4-M1 | EEG (0) | 256 | ✅ Usable | Standard central EEG |
| E1-M2 | EOG (1) | 256 | ✅ Usable | Left EOG |
| E2-M2 | EOG (1) | 256 | ✅ Usable | Right EOG |
| CHIN | EMG (2) | 256 | ✅ Usable | Chin EMG |
| ECG-II | ECG (3) | 256 | ✅ Usable | Lead II ECG |

**Modality Assessment:** ✅ All modalities correctly classified. DCSM uses standard bipolar montage.

**Available Modalities for Sleep Staging:** EEG, EOG, EMG, ECG

**Summary:** DCSM is a clinical sleep dataset from Denmark using standard AASM montage at 256 Hz. All channels are properly referenced bipolar derivations.

---

## 9. STAGES (Stanford Technology Analytics and Genomics in Sleep)

**Source:** Nature Scientific Data - 1,914 recordings across 13 clinical sites
**Native Sampling Rate:** 200-512 Hz (varies by site)
**Epoch Length:** 30 seconds

**13 Sites:** BOGN, GSBB, GSDV, GSLH, GSSA, GSSW, MSMI, MSNF, MSQW, MSTH, MSTR, STLK, STNF

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| **Grael/Compumedics style (GSBB, GSDV, GSLH, GSSA, GSSW, MSMI):** |||||
| EEG_C4-A1 | EEG (0) | 200-512 | ✅ Usable | Primary central |
| EEG_C3-A2 | EEG (0) | 200-512 | ✅ Usable | Secondary central |
| EEG_F4-A1 | EEG (0) | 200-512 | ✅ Usable | Frontal |
| EEG_F3-A2 | EEG (0) | 200-512 | ✅ Usable | Frontal |
| EEG_O2-A1 | EEG (0) | 200-512 | ✅ Usable | Occipital |
| EEG_O1-A2 | EEG (0) | 200-512 | ✅ Usable | Occipital |
| EOG_LOC-A2 | EOG (1) | 200-512 | ✅ Usable | Left EOG |
| EOG_ROC-A2 | EOG (1) | 200-512 | ✅ Usable | Right EOG |
| EMG_Chin | EMG (2) | 200-512 | ✅ Usable | Chin EMG |
| ECG_II | ECG (3) | 200-512 | ✅ Usable | Lead II |
| ECG_I | ECG (3) | 200-512 | ✅ Usable | Lead I |
| Leg_1 | LEG (7) | 200-512 | ✅ Usable | Left leg EMG (PLM) |
| Leg_2 | LEG (7) | 200-512 | ✅ Usable | Right leg EMG (PLM) |
| **Hyphenated bipolar (STLK):** |||||
| C4-M1 | EEG (0) | 200-512 | ✅ Usable | Central |
| C3-M2 | EEG (0) | 200-512 | ✅ Usable | Central |
| F4-M1 | EEG (0) | 200-512 | ✅ Usable | Frontal |
| F3-M2 | EEG (0) | 200-512 | ✅ Usable | Frontal |
| O2-M1 | EEG (0) | 200-512 | ✅ Usable | Occipital |
| O1-M2 | EEG (0) | 200-512 | ✅ Usable | Occipital |
| **Concatenated (BOGN):** |||||
| C4M1 | EEG (0) | 200-512 | ✅ Usable | No hyphen |
| C3M2 | EEG (0) | 200-512 | ✅ Usable | No hyphen |
| E1M2 | EOG (1) | 200-512 | ✅ Usable | No hyphen |
| E2M2 | EOG (1) | 200-512 | ✅ Usable | No hyphen |
| **Other variants (MSNF, MSQW, MSTH, MSTR, STNF):** |||||
| C4 / M1 | EEG (0) | 200-512 | ✅ Usable | Bare + differential pair |
| E1 | EOG (1) | 200-512 | ✅ Usable | Single channel |
| E2 | EOG (1) | 200-512 | ✅ Usable | Single channel |
| LOC | EOG (1) | 200-512 | ✅ Usable | Left outer canthus |
| ROC | EOG (1) | 200-512 | ✅ Usable | Right outer canthus |
| **Extended 10-20 (Grael sites):** |||||
| EEG_T4-A1 | EEG (0) | 200-512 | ✅ Usable | Temporal |
| EEG_T3-A2 | EEG (0) | 200-512 | ✅ Usable | Temporal |
| EEG_P4-A1 | EEG (0) | 200-512 | ✅ Usable | Parietal |
| EEG_P3-A2 | EEG (0) | 200-512 | ✅ Usable | Parietal |
| EEG_Fp2-A1 | EEG (0) | 200-512 | ✅ Usable | Pre-frontal |
| EEG_Fp1-A2 | EEG (0) | 200-512 | ✅ Usable | Pre-frontal |

**Modality Assessment:** ✅ All modalities correctly classified. STAGES is notable for its **4 different channel naming conventions** across sites, all representing the same bipolar montage.

**Additional Modalities Present:**
- **LEG (7)**: Leg EMG for periodic limb movement (PLM) detection
- **Position sensors**: May be present as separate channels

**Available Modalities for Sleep Staging:** EEG, EOG, EMG, ECG, LEG

**Summary:** STAGES is a multi-site clinical dataset with extreme heterogeneity in channel naming. All sites use the same underlying bipolar montage (C4-M1/C3-M2 equivalent) but different naming conventions. The Grael sites also have extended 10-20 coverage. LEG EMG channels enable PLM analysis.

---

## 10. WSC (Wisconsin Sleep Cohort)

**Source:** NSRR - 5 longitudinal visits
**Native Sampling Rate:** 512 Hz (typical)
**Epoch Length:** 30 seconds

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| C3_M2 | EEG (0) | 512 | ✅ Usable | Standard central (underscore) |
| C4_M1 | EEG (0) | 512 | ✅ Usable | Standard central (underscore) |
| C3_M1 | EEG (0) | 512 | ✅ Usable | Cross-diagonal derivation |
| C4_M2 | EEG (0) | 512 | ✅ Usable | Cross-diagonal derivation |
| F3_M2 | EEG (0) | 512 | ✅ Usable | Left frontal |
| F4_M1 | EEG (0) | 512 | ✅ Usable | Right frontal |
| O1_M2 | EEG (0) | 512 | ✅ Usable | Left occipital (visit3/4) |
| Fz_AVG | EEG (0) | 512 | ✅ Usable | Frontal average reference |
| C3_AVG | EEG (0) | 512 | ✅ Usable | Central average reference |
| E1 / E2 | EOG (1) | 512 | ✅ Usable | Differential EOG pair |
| E1 | EOG (1) | 512 | ✅ Usable | Single EOG |
| E2 | EOG (1) | 512 | ✅ Usable | Single EOG |
| chin | EMG (2) | 512 | ✅ Usable | Chin EMG (lowercase) |
| cchin_l | EMG (2) | 512 | ✅ Usable | Left chin variant |
| cchin_r | EMG (2) | 512 | ✅ Usable | Right chin variant |
| rchin_l | EMG (2) | 512 | ✅ Usable | Alternative chin EMG |

**Modality Assessment:** ✅ All modalities correctly classified. WSC uses underscore separators instead of hyphens.

**Available Modalities for Sleep Staging:** EEG, EOG, EMG

**Summary:** WSC is a longitudinal study with 5 visits per subject. Channel naming uses underscores (C3_M2) instead of hyphens (C3-M2). Includes average-referenced channels (Fz_AVG, C3_AVG). High sampling rate (512 Hz) captures detailed signal features.

---

## 11. Alzheimer's Disease Dataset (UZ Leuven)

**Source:** UZ Leuven - 69 subjects (37 AD + 32 HC)
**Native Sampling Rate:** 200 Hz
**Epoch Length:** 30 seconds

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| EEG C4-REF | EEG (0) | 200 | ✅ Usable | Average reference |
| EEG C3-REF | EEG (0) | 200 | ✅ Usable | Average reference |
| EEG F4-REF | EEG (0) | 200 | ✅ Usable | Average reference |
| EEG F3-REF | EEG (0) | 200 | ✅ Usable | Average reference |
| EEG O2-REF | EEG (0) | 200 | ✅ Usable | Average reference |
| EEG O1-REF | EEG (0) | 200 | ✅ Usable | Average reference |
| EEG Fz-REF | EEG (0) | 200 | ✅ Usable | Average reference |
| EEG Pz-REF | EEG (0) | 200 | ✅ Usable | Average reference |
| EEG EOG1-REF | EOG (1) | 200 | ✅ Usable | EOG with average ref (differential) |
| EEG EOG2-REF | EOG (1) | 200 | ✅ Usable | EOG with average ref (differential) |
| EMG Chin | EMG (2) | 200 | ✅ Usable | Chin EMG (36 subjects, newer hardware) |
| EMG1 | EMG (2) | 200 | ✅ Usable | Numbered EMG (33 subjects, older) |
| EMG2 | EMG (2) | 200 | ✅ Usable | Numbered EMG (33 subjects, older) |
| ECG V1 | ECG (3) | 200 | ✅ Usable | Precordial lead V1 |

**Modality Assessment:** ✅ All modalities correctly classified. This dataset uses **average reference** montage instead of bipolar.

**Hardware Variants:**
- Newer (36 subjects): EMG Chin
- Older (33 subjects): EMG1, EMG2 (numbered)

**Available Modalities for Sleep Staging:** EEG, EOG, EMG, ECG

**Summary:** Alzheimer's dataset compares patients vs. healthy controls. Uses average reference montage (unlike most clinical datasets). EOG is differential (EOG1-REF minus EOG2-REF). Two hardware variants for EMG. Full 10-20 coverage at 200 Hz.

---

## 12. Parkinson's Disease Dataset (UZ Leuven)

**Source:** UZ Leuven - 87 subjects (40 HOA + 48 PD)
**Native Sampling Rate:** 500 Hz
**Epoch Length:** 30 seconds

| Channel Name | Modality Type | FS (Hz) | Usability | Notes |
|--------------|---------------|---------|-----------|-------|
| **Night recordings (35-36 channels @ 500 Hz):** |||||
| EEG C3-A2 | EEG (0) | 500 | ✅ Usable | Bipolar standard |
| EEG C4-A1 | EEG (0) | 500 | ✅ Usable | Bipolar standard |
| EEG C3-REF | EEG (0) | 500 | ✅ Usable | Average reference variant |
| EEG C4-REF | EEG (0) | 500 | ✅ Usable | Average reference variant |
| EOG Left | EOG (1) | 500 | ✅ Usable | Left EOG (46 subjects) |
| EOG right | EOG (1) | 500 | ✅ Usable | Right EOG (46 subjects) |
| EOG1 | EOG (1) | 500 | ✅ Usable | Numbered variant (40 subjects) |
| EOG2 | EOG (1) | 500 | ✅ Usable | Numbered variant (40 subjects) |
| EOG LOC-A2 | EOG (1) | 500 | ✅ Usable | Left outer canthus |
| EOG ROC-A1 | EOG (1) | 500 | ✅ Usable | Right outer canthus |
| EMG Chin | EMG (2) | 500 | ✅ Usable | Chin EMG (46 subjects) |
| EMG1 | EMG (2) | 500 | ✅ Usable | Numbered variant (40 subjects) |
| ECG V1 | ECG (3) | 500 | ✅ Usable | Precordial lead V1 |
| **Nap recordings (15 channels @ 500 Hz):** |||||
| *(Same channels, reduced montage)* | | | | |

**Modality Assessment:** ✅ All modalities correctly classified. Uses **bipolar montage** (unlike Alzheimer's average ref).

**Hardware Variants:**
- Variant 1 (~46 subjects): "EOG Left"/"EOG right"/"EMG Chin"
- Variant 2 (~40 subjects): "EOG1"/"EOG2"/"EMG1"

**Available Modalities for Sleep Staging:** EEG, EOG, EMG, ECG

**Summary:** Parkinson's dataset compares patients vs. healthy older adults. Includes both night (86) and nap (71) recordings. Highest sampling rate (500 Hz) among all datasets. Bipolar montage with two hardware variants.

---

## Cross-Dataset Summary

### Sampling Rate Summary

| Dataset | Native FS (Hz) | Notes |
|---------|----------------|-------|
| SleepEDF | 100 (EEG/EOG), 1 (EMG/RESP/TEMP) | 1 Hz channels are processed parameters |
| SHHS | 125-128 | Slight variation between recordings |
| MESA | 200-512 | Variable |
| MrOS | ~200 | Approximate |
| HomePAP | 200-512 | Variable by subset |
| DCSM | 256 | Uniform |
| HMC | 256 | Uniform |
| MASS | 256 | All cohorts uniform |
| STAGES | 200-512 | Variable by site |
| WSC | 512 | Uniform high frequency |
| Alzheimer's | 200 | Uniform |
| Parkinson's | 500 | Highest uniform frequency |

### Modality Type Coverage

| Modality | Datasets with ✅ Usable Channels |
|----------|----------------------------------|
| EEG (0) | All 12 datasets |
| EOG (1) | All 12 datasets |
| EMG (2) | All 12 datasets (SleepEDF is 1 Hz envelope only) |
| ECG (3) | All 12 datasets |
| RESP (4) | SleepEDF (1 Hz), some STAGES (via events) |
| LEG (7) | STAGES (Leg_1, Leg_2), MrOS (LAT/RAT), WSC |
| TEMP (12) | SleepEDF (1 Hz rectal) |
| DEVICE (13) | SleepEDF (Event marker) |
| OTHER (14) | Various position/light/sound channels |

### Recommended Modalities for Sleep Staging

**Primary (AASM recommended):**
- EEG: Essential for staging (N1, N2, N3 discrimination)
- EOG: Essential for REM detection
- EMG: Essential for REM vs Wake discrimination

**Secondary (useful but not required):**
- ECG: Useful for arousal detection, HRV analysis
- LEG: Useful for PLM detection (STAGES, MrOS, WSC)
- RESP: Useful for respiratory event detection

**Not Recommended (processed/low-freq):**
- SleepEDF EMG (1 Hz envelope): Not raw EMG
- SleepEDF RESP/TEMP (1 Hz): Too low frequency for waveform analysis
- Event markers: Technical annotations only

### Issues and Recommendations

1. **SleepEDF EMG**: The 1 Hz envelope signal should be clearly marked as "processed" not raw EMG. It's calculated by filtering, rectifying, and envelope-detecting the raw EMG.

2. **Montage heterogeneity**: Datasets use different referencing schemes:
   - Bipolar (C3-M2, C4-M1): Most common
   - Average reference (C3-REF, C4-REF): Alzheimer's
   - Contralateral ear (C3-CLE): MASS
   - Numbered channels (EEG1, EOG1): MESA, MrOS

3. **Channel naming variations**: Same physical channel can have different names:
   - C4-M1 vs C4/M1 vs C4_M1
   - EOG(L) vs LOC vs E1-M2
   - CHIN vs EMG Chin vs EMG1

4. **Reference electrode naming**:
   - A1/M1: Left mastoid (same)
   - A2/M2: Right mastoid (same)

5. **Modality classification accuracy**: The `infer_channel_modality()` function in `modality.py` correctly identifies all standard PSG channels. No missing or incorrect classifications found for physiological signals.

### References

- [HMC on PhysioNet](https://physionet.org/content/hmc-sleep-staging/1.1/)
- [SleepEDF on PhysioNet](https://physionet.org/physiobank/database/sleep-edfx/)
- [SHHS on NSRR](https://sleepdata.org/datasets/shhs)
- [MESA on NSRR](https://sleepdata.org/datasets/mesa)
- [MrOS on NSRR](https://sleepdata.org/datasets/mros)
- [HomePAP on NSRR](https://sleepdata.org/datasets/homepap)
- [MASS on Borealis Data](https://borealisdata.ca)
- [STAGES dataset](https://www.nature.com/articles/sdata2018108) - Nature Scientific Data, 2018
- [PhysioEx library](https://github.com/physioex)

---

*Generated by PhysioEx data pipeline analysis. Last updated: 2025-01-13*
