"""Per-dataset recipe for the clean channels: which electrodes to read and how to combine them.

A :class:`CleanMontage` lives on each dataset class as ``CLEAN_MONTAGE``, next to
``CHANNEL_PREFERENCES``, which it does not replace: the existing channels and
caches are untouched. For every output (EEG, EOG, chin, legs, ECG) it lists
candidates in order of preference; the first one whose labels are all in the
recording wins. A candidate is

* ``"C4-M1"``: one label, read as is;
* ``("C4", "M1")``: a difference, ``C4 - M1``, as in ``CHANNEL_PREFERENCES``;
* ``{"E1-M2": 1, "E2-M1": -1, "M1-M2": -1}``: any linear combination of labels,
  for derivations that need more than two terms;
* :class:`Fallback` around any of the above: accepted, but the night drops to a
  lower cleaning level (the recipe says why in ``note``).

``extra`` maps a name to its own candidate list: channels the encoder does not
see but that the night-level cleaning uses (source separation, regression).
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

LEVELS = ("rich", "poor", "envelope", "missing")  # best first
OUTPUTS = ("eeg", "eog", "chin", "leg_l", "leg_r", "ecg")
# A night is "rich" (source separation feasible) only with at least this many
# EEG derivations (canonical + extra EEG); below it falls back to "poor".
MIN_EEG_RICH = 4


@dataclass(frozen=True)
class Fallback:
    candidate: Union[str, Tuple[str, str], Mapping[str, float]]
    level: str = "poor"
    note: str = ""


Candidate = Union[str, Tuple[str, str], Mapping[str, float], Fallback]


@dataclass(frozen=True)
class CleanMontage:
    eeg: Sequence[Candidate]
    eog: Sequence[Candidate]
    chin: Sequence[Candidate] = ()
    leg_l: Sequence[Candidate] = ()
    leg_r: Sequence[Candidate] = ()
    ecg: Sequence[Candidate] = ()
    extra: Mapping[str, Sequence[Candidate]] = field(default_factory=dict)
    # Names in ``extra`` that are EEG (count towards MIN_EEG_RICH); the others
    # (e.g. a vertical EOG) are still used by the cleaning.
    extra_eeg: Sequence[str] = ()
    level: str = "rich"
    mains: float = 50.0  # power-line frequency (Hz) of the recording site
    notes: str = ""

    def spec(self) -> str:
        """Stable hash of the recipe, part of the cache key of the clean channels."""

        def norm(c):
            if isinstance(c, Fallback):
                return ("fallback", norm(c.candidate), c.level)
            if isinstance(c, Mapping):
                return tuple(sorted((str(k), float(v)) for k, v in c.items()))
            return c

        payload = repr(
            (
                [tuple(norm(c) for c in getattr(self, o)) for o in OUTPUTS],
                sorted((k, tuple(norm(c) for c in v)) for k, v in self.extra.items()),
                tuple(self.extra_eeg),
                self.level,
                self.mains,
            )
        )
        return "CleanMontage:" + hashlib.sha1(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class ResolvedDerivation:
    name: str  # output or extra name
    candidate_index: int  # position of the winning candidate in its list
    terms: Tuple[
        Tuple[str, float, int], ...
    ]  # (label as in the file, weight, EDF index)
    fs: float
    level: str  # "rich" unless the candidate is a Fallback

    @property
    def label(self) -> str:
        """Readable form, e.g. ``E1-M2 - E2-M1 - M1-M2``."""
        out = ""
        for i, (lab, w, _) in enumerate(self.terms):
            s = f"({lab})" if len(self.terms) > 2 and "-" in lab else lab
            coef = "" if abs(w) == 1 else f"{abs(w):g}*"
            sign = (" - " if w < 0 else " + ") if i else ("-" if w < 0 else "")
            out += sign + coef + s
        return out


@dataclass
class ResolvedMontage:
    derivations: Dict[
        str, Optional[ResolvedDerivation]
    ]  # every output, None if missing
    extra: Dict[str, ResolvedDerivation]  # only the extras found
    level: str
    reasons: List[str]


def _as_terms(c) -> Tuple[Tuple[str, float], ...]:
    if isinstance(c, str):
        return ((c, 1.0),)
    if isinstance(c, tuple):
        if len(c) != 2:
            raise ValueError(
                f"a tuple candidate is a difference of two labels, got {c!r}"
            )
        return ((c[0], 1.0), (c[1], -1.0))
    if isinstance(c, Mapping):
        return tuple((str(k), float(v)) for k, v in c.items())
    raise TypeError(f"unsupported candidate {c!r}")


def _resolve_one(name, candidates, upper, fs_map) -> Optional[ResolvedDerivation]:
    for i, c in enumerate(candidates):
        level = "rich"
        if isinstance(c, Fallback):
            level, c = c.level, c.candidate
        terms = []
        for lab, w in _as_terms(c):
            j = upper.get(lab.upper().strip())
            if j is None:
                break
            terms.append((j, w))
        else:
            labels = [upper.labels[j] for j, _ in terms]
            fss = {float(fs_map[l]) for l in labels}
            if (
                len(fss) != 1
            ):  # terms at different rates cannot be summed sample by sample
                continue
            return ResolvedDerivation(
                name,
                i,
                tuple((l, w, j) for l, (j, w) in zip(labels, terms)),
                fss.pop(),
                level,
            )
    return None


class _UpperIndex(dict):
    def __init__(self, available: Sequence[str]):
        super().__init__()
        self.labels = list(available)
        for i, lab in enumerate(available):
            self.setdefault(lab.upper().strip(), i)


def _worse(a: str, b: str) -> str:
    return a if LEVELS.index(a) > LEVELS.index(b) else b


def resolve_montage(
    montage: CleanMontage, available: Sequence[str], fs_map: Mapping[str, float]
) -> ResolvedMontage:
    """Pick, for every output and extra, the first candidate present in the recording.

    The night level starts at ``montage.level`` and can only get worse: a
    Fallback candidate brings its own level; EEG or EOG missing gives
    ``"missing"``; a "rich" recipe with fewer than MIN_EEG_RICH EEG derivations
    in this recording becomes "poor". Extras identical to the canonical EEG
    (same terms) are dropped, so a recipe can list C3 both as fallback EEG and
    as extra.
    """
    upper = _UpperIndex(available)
    level, reasons = montage.level, []
    der = {o: _resolve_one(o, getattr(montage, o), upper, fs_map) for o in OUTPUTS}
    for o in ("eeg", "eog"):
        if der[o] is None:
            level = "missing"
            reasons.append(f"{o}: no candidate in the recording")
    for o, d in der.items():
        if d is not None and d.level != "rich":
            level = _worse(level, d.level)
            reasons.append(f"{o}: fallback {d.label!r} ({d.level})")
    extra = {}
    canon = der["eeg"].terms if der["eeg"] is not None else None
    for k, cands in montage.extra.items():
        d = _resolve_one(k, cands, upper, fs_map)
        if d is not None and d.terms != canon:
            extra[k] = d
    n_eeg = (der["eeg"] is not None) + sum(k in extra for k in montage.extra_eeg)
    if level == "rich" and n_eeg < MIN_EEG_RICH:
        level = "poor"
        reasons.append(
            f"{n_eeg} EEG derivations < {MIN_EEG_RICH}: no source separation"
        )
    return ResolvedMontage(der, extra, level, reasons)


def read_derivation(edf_reader, d: ResolvedDerivation) -> Tuple[np.ndarray, float]:
    """Read a resolved derivation from an open pyedflib.EdfReader, in µV (float64)."""
    from physioex.data.readers.edf import _unit_scale_to_uv

    xs = []
    for _, w, j in d.terms:
        x = edf_reader.readSignal(j).astype(np.float64)
        xs.append(w * _unit_scale_to_uv(edf_reader.getPhysicalDimension(j)) * x)
    n = min(len(x) for x in xs)
    return np.sum([x[:n] for x in xs], axis=0), d.fs
