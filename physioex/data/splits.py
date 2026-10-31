"""Subject-grouped 80/10/10 splits per site (ported from ``protosleepnet-train/splits/build_splits.py``).

The same person across visits or recordings gets the same *group key* and always lands in one
split. Groups are sorted, shuffled with a site-specific seed (``seed + sum(ord(site))``, order
independent) and cut 80/10/10. This is the split of the ProtoSleepNet multi-dataset release
(``outputs/psn-multi/splits_29sites.json`` on Sofia); :func:`grouped_splits` reproduces it for
one site from the subject ids alone, :func:`load_site_splits` reads it from the JSON.

``BaseDataset.get_splits`` (random 70/15/15 on the dataset's subject order) stays the fallback.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

# --- group-key functions: subject id -> person id ---


def _identity(sid: str) -> str:
    return sid


def _shhs(sid: str) -> str:  # shhs1-200077 == shhs2-200077
    m = re.match(r"^shhs\d+-(\d+)$", sid)
    return m.group(1) if m else sid


def _wsc(sid: str) -> str:  # wsc-visitK-<id>
    m = re.match(r"^wsc-visit\d+-(.+)$", sid)
    return m.group(1) if m else sid


def _hpap(sid: str) -> str:  # nsrrid is the trailing number, shared by lab-full / lab-split
    m = re.search(r"(\d+)$", sid)
    return m.group(1) if m else sid


def _stages(sid: str) -> str:  # XXXX00003_2 -> XXXX00003
    m = re.match(r"^([A-Za-z]{4}\d+)(?:_\d+)?$", sid)
    return m.group(1) if m else sid


def _parkinsons(sid: str) -> str:  # Sub_XXXX (night/nap suffixes dropped)
    m = re.search(r"(Sub_?\d+)", sid, flags=re.I)
    return m.group(1).lower().replace("_", "") if m else sid


def _mass(sid: str) -> str:  # 01-03-0001 / 01-03-0001 PSG -> 01-03-0001
    m = re.search(r"(\d{2}-\d{2}-\d{4})", sid)
    return m.group(1) if m else sid


GROUP_FNS: Dict[str, Callable[[str], str]] = {
    "identity": _identity, "shhs": _shhs, "wsc": _wsc, "hpap": _hpap,
    "stages": _stages, "parkinsons": _parkinsons, "mass": _mass,
}


def split_groups(groups: Sequence[str], seed: int, frac=(0.8, 0.1, 0.1)):
    """Shuffle sorted groups with ``seed`` and cut 80/10/10 (with >= 3 groups, val and test get
    >= 1 group each)."""
    g = sorted(groups)
    random.Random(seed).shuffle(g)
    n = len(g)
    n_val = max(1, round(frac[1] * n)) if n >= 3 else 0
    n_test = max(1, round(frac[2] * n)) if n >= 3 else 0
    n_train = n - n_val - n_test
    return g[:n_train], g[n_train:n_train + n_val], g[n_train + n_val:]


def site_seed(site: str, seed: int = 42) -> int:
    return seed + sum(map(ord, site))


def grouped_splits(subject_ids: Sequence[str], site: str, group: str = "identity",
                   seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """``(train, valid, test)`` subject ids of one site (a single unit), grouped by person."""
    gfn = GROUP_FNS[group]
    tr, va, te = split_groups(sorted({gfn(s) for s in subject_ids}), site_seed(site, seed))
    where = {g: k for k, gs in ((0, tr), (1, va), (2, te)) for g in gs}
    out: Tuple[List[str], List[str], List[str]] = ([], [], [])
    for s in sorted(subject_ids):
        out[where[gfn(s)]].append(s)
    return out


def build_splits(units: Dict[str, object], unit_cfgs: List[dict], seed: int = 42):
    """Several units per site (``unit_cfgs`` = ``units`` of ``sites.yaml``). Returns
    ``(splits, table)`` with ``splits[site] = {"train"|"val"|"test": [[unit, sid], ...]}``."""
    cfg_by_unit = {u["name"]: u for u in unit_cfgs}
    unknown = sorted(set(units) - set(cfg_by_unit))
    if unknown:
        raise KeyError(f"units not in sites.yaml: {unknown}")
    site_members: Dict[str, List[Tuple[str, str, str, int]]] = defaultdict(list)
    for unit, subj in units.items():
        c = cfg_by_unit[unit]
        gfn = GROUP_FNS[c.get("group", "identity")]
        items = subj.items() if isinstance(subj, dict) else ((s, 0) for s in subj)
        for sid, nw in items:
            site_members[c["site"]].append((unit, sid, gfn(sid), int(nw)))
    splits, table = {}, {}
    for site in sorted(site_members):
        mem = site_members[site]
        tr, va, te = split_groups(sorted({m[2] for m in mem}), site_seed(site, seed))
        where = {g: k for k, gs in (("train", tr), ("val", va), ("test", te)) for g in gs}
        out = {"train": [], "val": [], "test": []}
        for unit, sid, g, _ in sorted(mem):
            out[where[g]].append([unit, sid])
        splits[site] = out
        table[site] = {"recordings": len(mem), "groups": len(where), **{k: len(v) for k, v in out.items()},
                       "windows": sum(m[3] for m in mem) or None}
    return splits, table


def check_no_overlap(splits: Dict[str, dict], unit_cfgs: List[dict]) -> None:
    cfg = {u["name"]: u for u in unit_cfgs}
    for site, sp in splits.items():
        seen = {}
        for k in ("train", "val", "test"):
            for unit, sid in sp[k]:
                g = GROUP_FNS[cfg[unit].get("group", "identity")](sid)
                assert seen.setdefault(g, k) == k, f"{site}: group {g} in {seen[g]} and {k}"


def load_site_splits(path: str, site: str) -> Tuple[List[str], List[str], List[str]]:
    """``(train, valid, test)`` subject ids of ``site`` from a splits JSON (``build_splits`` format)."""
    sp = json.loads(Path(path).read_text())[site]
    return tuple([sid for _, sid in sp[k]] for k in ("train", "val", "test"))
