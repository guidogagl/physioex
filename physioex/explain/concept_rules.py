"""Composite concepts of the lexicon (plan of 09.11.2026, "revise and update" cycle).

A clinical concept is written as the manuals write it: a local word of P (shape, 0.5 s tokens) AND a state
predicate read from the epoch / neighbourhood terms of the lexicon AND a temporal relation computed on the
sequence of words of the night (isolation, before / after, duration, density over a window). The predicates
come from lexicon terms, never from the stager's stage; the thresholds come from the manuals and are fixed
before any evaluation:

- N3 = slow waves (0.5-2 Hz, >= 75 uV) over >= 20 % of the epoch (AASM);
- W = alpha over > 50 % of the epoch (AASM);
- K-complex = a large slow transient that stands out from a background without slow-wave sleep, isolated.

Everything here works on flat token arrays (``T`` tokens per epoch) with a ``night`` index per token, like the
dump loaded by ``concept_rationale.load``. Event metrics are per event (overlap), with purity = fraction of the
detected events that overlap a reference event and coverage = fraction of the reference events overlapped.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

T = 60
N3_SWA = 0.20        # AASM: slow waves over >= 20 % of the epoch
W_ALPHA = 0.50       # AASM: alpha over > 50 % of the epoch
EYE_PRESENT = 0.05   # an eye-word "is present" in the epoch: >= 5 % of its tokens (1.5 s), fixed a priori
ATONIA_EPOCH = 0.50  # low chin tone: atonia over > 50 % of the epoch (the reference's epoch rule)
KC_ISO_TOK = 4       # isolation of a K-complex: no other slow event within +-2 s (fixed a priori)
MERGE_GAP = 1        # word tokens closer than this gap (tokens) form one event


# ---------------------------------------------------------------------------- events on token sequences
def event_runs(mask: np.ndarray, night: np.ndarray, gap: int = MERGE_GAP) -> List[Tuple[int, int]]:
    """Runs ``[a, b)`` of True tokens, merged across gaps <= ``gap`` tokens, never across nights."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    idx = np.flatnonzero(m)
    cut = (np.diff(idx) > gap + 1) | (night[idx[1:]] != night[idx[:-1]])
    starts = np.r_[idx[0], idx[1:][cut]]
    ends = np.r_[idx[:-1][cut], idx[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def runs_to_mask(runs: List[Tuple[int, int]], n: int) -> np.ndarray:
    out = np.zeros(n, bool)
    for a, b in runs:
        out[a:b] = True
    return out


def event_ids(runs: List[Tuple[int, int]], n: int) -> np.ndarray:
    """Token -> event index (-1 outside events)."""
    out = np.full(n, -1, np.int64)
    for i, (a, b) in enumerate(runs):
        out[a:b] = i
    return out


def isolated(runs: List[Tuple[int, int]], night: np.ndarray, k: int = KC_ISO_TOK) -> np.ndarray:
    """Per event: True when no other event of the same night starts or ends within +-``k`` tokens of it."""
    ok = np.ones(len(runs), bool)
    for i, (a, b) in enumerate(runs):
        for j in (i - 1, i + 1):
            if 0 <= j < len(runs) and night[runs[j][0]] == night[a]:
                a2, b2 = runs[j]
                gap = a2 - b if j > i else a - b2
                if gap <= k:
                    ok[i] = False
    return ok


def epoch_of(runs: List[Tuple[int, int]]) -> np.ndarray:
    """Epoch of every event: the epoch of its centre token."""
    return np.array([((a + b - 1) // 2) // T for a, b in runs], np.int64)


# ---------------------------------------------------------------------------- composite rules
def wake_rule(alpha_ep: np.ndarray, blink_ep: np.ndarray = None, eye_ep: np.ndarray = None,
              atonia_ep: np.ndarray = None) -> np.ndarray:
    """AASM wake per epoch from lexicon fractions: alpha > ``W_ALPHA``, or eye blinks present, or rapid eye
    movements present with a chin tone that is not low. Missing modalities (NaN) drop their clause, so a night
    without EOG / EMG falls back to the alpha clause."""
    w = np.nan_to_num(alpha_ep, nan=0.0) > W_ALPHA
    if blink_ep is not None:
        w |= np.nan_to_num(blink_ep, nan=0.0) >= EYE_PRESENT
    if eye_ep is not None:
        tone_low = np.nan_to_num(atonia_ep, nan=1.0) > ATONIA_EPOCH if atonia_ep is not None else np.zeros(len(w), bool)
        w |= (np.nan_to_num(eye_ep, nan=0.0) >= EYE_PRESENT) & ~tone_low
    return w


def kc_n3_rule(word: np.ndarray, night: np.ndarray, swa_ep: np.ndarray, alpha_ep: np.ndarray,
               use_swa: bool = True, use_alpha: bool = True, use_iso: bool = True,
               iso_k: int = KC_ISO_TOK, wake_ep: np.ndarray = None) -> Dict[str, np.ndarray]:
    """K-complex and N3 slow wave from the slow-transient word (token mask), the epoch slow-wave fraction and
    the epoch alpha fraction (one value per epoch).

    - ``kc``: word event in an epoch with ``swa < N3_SWA`` and ``alpha <= W_ALPHA``, isolated (+-``iso_k``);
    - ``n3``: word event in an epoch with ``swa >= N3_SWA``.

    With ``wake_ep`` (boolean per epoch, ``wake_rule``) the alpha clause is replaced by "not wake" and applies
    to both concepts (C1c). The ``use_*`` switches drop one component (ablation, V3): the dropped predicate
    counts as satisfied (``use_alpha`` drops the wake clause too)."""
    runs = event_runs(word, night)
    n = len(word)
    if not runs:
        z = np.zeros(n, bool)
        return {"kc": z, "n3": z.copy(), "runs": []}
    ep = epoch_of(runs)
    low_swa = swa_ep[ep] < N3_SWA if use_swa else np.ones(len(runs), bool)
    if wake_ep is not None:
        no_w = ~wake_ep[ep] if use_alpha else np.ones(len(runs), bool)
    else:
        no_w = alpha_ep[ep] <= W_ALPHA if use_alpha else np.ones(len(runs), bool)
    iso = isolated(runs, night, iso_k) if use_iso else np.ones(len(runs), bool)
    kc = low_swa & no_w & iso
    n3 = swa_ep[ep] >= N3_SWA if use_swa else np.zeros(len(runs), bool)
    if wake_ep is not None:
        n3 &= no_w
    return {"kc": runs_to_mask([r for r, k in zip(runs, kc) if k], n),
            "n3": runs_to_mask([r for r, k in zip(runs, n3) if k], n), "runs": runs}


def rem_w_rule(word: np.ndarray, night: np.ndarray, aton_ep: np.ndarray, alpha_ep: np.ndarray,
               use_tone: bool = True, use_alpha: bool = True) -> Dict[str, np.ndarray]:
    """Rapid eye movement of REM against the same movement in wake (C2), from the rapid-eye word (token mask),
    the epoch atonia fraction (chin tone low relative to the night) and the epoch alpha fraction.

    - ``rem``: eye-word event in an epoch with atonia > ``ATONIA_EPOCH`` and alpha <= ``W_ALPHA``;
    - ``wake``: eye-word event in an epoch with tone not low or alpha > ``W_ALPHA`` (AASM wake).

    A dropped component (``use_*``, V3) counts as satisfied for REM and as absent for wake."""
    runs = event_runs(word, night)
    n = len(word)
    if not runs:
        z = np.zeros(n, bool)
        return {"rem": z, "wake": z.copy(), "runs": []}
    ep = epoch_of(runs)
    low = np.nan_to_num(aton_ep[ep], nan=0.0) > ATONIA_EPOCH if use_tone else np.ones(len(runs), bool)
    hi_a = np.nan_to_num(alpha_ep[ep], nan=0.0) > W_ALPHA if use_alpha else np.zeros(len(runs), bool)
    rem = low & ~hi_a
    wake = ~rem
    return {"rem": runs_to_mask([r for r, k in zip(runs, rem) if k], n),
            "wake": runs_to_mask([r for r, k in zip(runs, wake) if k], n), "runs": runs}


STATES = ("W", "REM", "NREM")


def state_rule(wake_ep: np.ndarray, aton_ep: np.ndarray, swa_ep: np.ndarray) -> np.ndarray:
    """Coarse state per epoch from lexicon predicates (C3): 0 = W (``wake_rule``), 1 = REM (not wake, atonia
    > ``ATONIA_EPOCH``, slow waves < ``N3_SWA``), 2 = NREM (the rest). No stage enters."""
    low = np.nan_to_num(aton_ep, nan=0.0) > ATONIA_EPOCH
    st = np.full(len(wake_ep), 2, np.int8)
    st[~wake_ep & low & (swa_ep < N3_SWA)] = 1
    st[wake_ep] = 0
    return st


def in_state(word: np.ndarray, night: np.ndarray, state_ep: np.ndarray) -> Dict[str, np.ndarray]:
    """Word events labelled by the state of their epoch: one token mask per state in ``STATES``."""
    runs = event_runs(word, night)
    n = len(word)
    if not runs:
        return {k: np.zeros(n, bool) for k in STATES}
    ep = epoch_of(runs)
    return {k: runs_to_mask([r for r, e in zip(runs, ep) if state_ep[e] == i], n) for i, k in enumerate(STATES)}


# ---------------------------------------------------------------------------- event metrics
def event_agreement(det: np.ndarray, ref: np.ndarray, night: np.ndarray, sel: np.ndarray = None) -> Dict:
    """Event purity (precision), coverage (recall) and F1 by overlap. ``sel``: tokens in scope (events of
    either side that start outside it are ignored)."""
    if sel is not None:
        det, ref = det & sel, ref & sel
    rd, rr = event_runs(det, night, gap=0), event_runs(ref, night, gap=0)
    if not rd or not rr:
        return {"n_det": len(rd), "n_ref": len(rr), "purity": np.nan if not rd else 0.0,
                "coverage": np.nan if not rr else 0.0, "f1": 0.0 if (rd or rr) else np.nan}
    hit_d = np.array([ref[a:b].any() for a, b in rd])
    hit_r = np.array([det[a:b].any() for a, b in rr])
    p, c = float(hit_d.mean()), float(hit_r.mean())
    return {"n_det": len(rd), "n_ref": len(rr), "purity": p, "coverage": c,
            "f1": 0.0 if p + c == 0 else 2 * p * c / (p + c)}


def per_unit_counts(det: np.ndarray, ref: np.ndarray, night: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
    """Per night: (detected events hitting ref, detected events, ref events hit, ref events), for bootstrap."""
    out = {}
    for n in np.unique(night):
        m = night == n
        d, r, nn = det[m], ref[m], night[m]
        rd, rr = event_runs(d, nn, gap=0), event_runs(r, nn, gap=0)
        out[int(n)] = (sum(r[a:b].any() for a, b in rd), len(rd), sum(d[a:b].any() for a, b in rr), len(rr))
    return out


def f1_from_counts(c: np.ndarray) -> float:
    hd, nd, hr, nr = c.sum(0)
    if nd == 0 or nr == 0:
        return 0.0
    p, r = hd / nd, hr / nr
    return 0.0 if p + r == 0 else float(2 * p * r / (p + r))


def bootstrap_f1(counts: Dict[int, Tuple[int, int, int, int]], n_boot: int = 2000, seed: int = 0) -> List[float]:
    """95 % CI of the pooled event F1, resampling the units (nights / excerpts)."""
    c = np.array(list(counts.values()), float)
    if len(c) < 2:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    f = [f1_from_counts(c[rng.integers(0, len(c), len(c))]) for _ in range(n_boot)]
    return [float(np.quantile(f, 0.025)), float(np.quantile(f, 0.975))]


# ---------------------------------------------------------------------------- nulls
def permute_epochs(v_ep: np.ndarray, night_ep: np.ndarray, rng) -> np.ndarray:
    """Epoch values permuted inside each night: the predicate keeps its distribution, loses its place."""
    out = v_ep.copy()
    for n in np.unique(night_ep):
        m = np.flatnonzero(night_ep == n)
        out[m] = v_ep[rng.permutation(m)]
    return out


def shift_within_nights(mask: np.ndarray, night: np.ndarray, rng, min_shift: int = 2 * T) -> np.ndarray:
    """Token mask rolled circularly inside each night by a random shift >= ``min_shift``: the word keeps its
    events and their spacing, loses their place."""
    out = np.zeros_like(mask)
    for n in np.unique(night):
        m = np.flatnonzero(night == n)
        L = len(m)
        if L <= 2 * min_shift:
            out[m] = mask[m]
            continue
        out[m] = np.roll(mask[m], int(rng.integers(min_shift, L - min_shift)))
    return out
