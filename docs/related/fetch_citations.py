#!/usr/bin/env python3
"""Refresh ``works.yaml`` with the publications/preprints that cite PhysioEx.

Semi-automatic maintenance helper — run it by hand when you want to update the
"Related Works" page:

    python docs/related/fetch_citations.py

It queries the OpenAlex API (free, no key) for every work that cites the
PhysioEx paper (by DOI), then MERGES the result into ``works.yaml``:

* entries marked ``manual: true`` are never touched;
* for auto entries, curated fields (``note``, ``thumbnail``) are preserved
  across refreshes; the rest (title, authors, venue, year, preview, links) are
  refreshed from OpenAlex;
* new citing works are appended; nothing is deleted.

Only stdlib (``urllib``) + PyYAML (a core PhysioEx dependency) are used, and the
network is touched ONLY here — never at documentation build time.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

import yaml

DOI = "10.1088/1361-6579/adaf73"          # the PhysioEx paper
MAILTO = "guido.gagliardi@kuleuven.be"     # OpenAlex "polite pool"
API = "https://api.openalex.org"
YAML_PATH = os.path.join(os.path.dirname(__file__), "works.yaml")
SNIPPET_CHARS = 280

PREPRINT_SOURCES = {
    "arxiv", "biorxiv", "medrxiv", "techrxiv", "research square",
    "ssrn", "chemrxiv", "preprints.org", "osf",
}


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": f"physioex-docs ({MAILTO})"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _short_id(openalex_id: str) -> str:
    return (openalex_id or "").rstrip("/").split("/")[-1]


def _reconstruct_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    text = " ".join(w for _, w in positions)
    if len(text) > SNIPPET_CHARS:
        text = text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " …"
    return text


def _is_preprint(work: dict) -> bool:
    if (work.get("type") or "") in {"preprint", "posted-content"}:
        return True
    src = ((work.get("primary_location") or {}).get("source") or {})
    if (src.get("type") or "") == "repository":
        return True
    name = (src.get("display_name") or "").lower()
    return any(p in name for p in PREPRINT_SOURCES)


def _entry_from_work(work: dict) -> dict:
    src = ((work.get("primary_location") or {}).get("source") or {})
    doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
    authors = [
        (a.get("author") or {}).get("display_name")
        for a in (work.get("authorships") or [])
    ]
    authors = [a for a in authors if a]
    landing = (work.get("primary_location") or {}).get("landing_page_url")
    return {
        "id": _short_id(work.get("id")),
        "title": work.get("title") or work.get("display_name") or "Untitled",
        "authors": authors,
        "year": work.get("publication_year"),
        "venue": src.get("display_name"),
        "type": "preprint" if _is_preprint(work) else "published",
        "doi": doi,
        "url": (f"https://doi.org/{doi}" if doi else landing),
        "preview": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "note": "",
        "thumbnail": "",
        "manual": False,
    }


def _fetch_citing(work_id: str) -> list[dict]:
    out, cursor = [], "*"
    while cursor:
        q = urllib.parse.urlencode({
            "filter": f"cites:{work_id}",
            "per-page": 200,
            "cursor": cursor,
            "mailto": MAILTO,
        })
        data = _get(f"{API}/works?{q}")
        out.extend(data.get("results", []))
        cursor = (data.get("meta") or {}).get("next_cursor")
        time.sleep(0.2)
    return out


def _key(entry: dict) -> str:
    return (entry.get("doi") or entry.get("id") or entry.get("title") or "").lower()


def merge(existing: list[dict], fetched: list[dict]) -> list[dict]:
    by_key = {_key(e): e for e in existing if isinstance(e, dict)}
    for new in fetched:
        k = _key(new)
        old = by_key.get(k)
        if old is None:
            by_key[k] = new
        elif old.get("manual"):
            continue  # fully curated → leave untouched
        else:
            new["note"] = old.get("note") or ""
            new["thumbnail"] = old.get("thumbnail") or ""
            by_key[k] = new
    merged = list(by_key.values())
    merged.sort(key=lambda e: (-(e.get("year") or 0), (e.get("title") or "").lower()))
    return merged


def main() -> int:
    print(f"Resolving DOI {DOI} on OpenAlex …")
    try:
        paper = _get(f"{API}/works/https://doi.org/{DOI}?mailto={MAILTO}")
        wid = _short_id(paper.get("id"))
        print(f"  → {wid} ({paper.get('title')})")
        fetched_works = _fetch_citing(wid)
    except Exception as exc:  # network / API problems must not be silent
        print(f"ERROR: OpenAlex query failed: {exc}", file=sys.stderr)
        return 1

    fetched = [_entry_from_work(w) for w in fetched_works]
    print(f"OpenAlex reports {len(fetched)} citing work(s).")

    existing = []
    if os.path.isfile(YAML_PATH):
        with open(YAML_PATH, "r", encoding="utf-8") as fh:
            existing = yaml.safe_load(fh) or []

    merged = merge(existing, fetched)
    with open(YAML_PATH, "w", encoding="utf-8") as fh:
        fh.write("# Auto-refreshed by fetch_citations.py from OpenAlex.\n")
        fh.write("# Set `manual: true` on an entry to protect it from overwrites,\n")
        fh.write("# or add entries by hand (e.g. preprints OpenAlex hasn't indexed).\n")
        yaml.safe_dump(merged, fh, allow_unicode=True, sort_keys=False, width=100)
    print(f"Wrote {len(merged)} entry(ies) → {YAML_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
