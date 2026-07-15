#!/usr/bin/env python3
"""Refresh ``works.yaml`` with the publications/preprints that cite PhysioEx.

Semi-automatic maintenance helper — run it by hand when you want to update the
"Related Works" page:

    python docs/related/fetch_citations.py

It unions two open citation sources (no API key needed):

* **OpenAlex**  — citations of the PhysioEx paper (by DOI);
* **Semantic Scholar** — same, for broader coverage (recovers works one index
  misses; together they approximate what Google Scholar shows — Scholar has no
  usable API).

Then it MERGES the result into ``works.yaml``:

* entries marked ``manual: true`` are never touched;
* curated fields (``note``, ``thumbnail``) are preserved across refreshes;
* the rest (title, authors, venue, year, preview, links) are refreshed;
* new citing works are appended; nothing is deleted.

As a best effort it also fills a **graphical preview** per card by reading the
landing page's ``og:image`` (often the graphical abstract / a figure); override
it by dropping an image in ``_thumbs/`` and setting ``thumbnail:`` by hand.

Anything Scholar shows but both APIs miss can be added by hand as a
``manual: true`` entry.

Only stdlib (``urllib``) + PyYAML (a core dep) are used, and the network is
touched ONLY here — never at documentation build time.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import yaml

DOI = "10.1088/1361-6579/adaf73"          # the PhysioEx paper
MAILTO = "guido.gagliardi@kuleuven.be"     # OpenAlex "polite pool"
OA = "https://api.openalex.org"
S2 = "https://api.semanticscholar.org/graph/v1"
YAML_PATH = os.path.join(os.path.dirname(__file__), "works.yaml")
SNIPPET_CHARS = 280
UA = f"physioex-docs (+{MAILTO})"

PREPRINT_SOURCES = {
    "arxiv", "biorxiv", "medrxiv", "techrxiv", "research square",
    "ssrn", "chemrxiv", "preprints.org", "osf",
}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _short_id(openalex_id: str) -> str:
    return (openalex_id or "").rstrip("/").split("/")[-1]


def _snippet(text: str) -> str:
    text = (text or "").strip()
    if len(text) > SNIPPET_CHARS:
        text = text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " …"
    return text


def _reconstruct_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = [(i, w) for w, idxs in inv.items() for i in idxs]
    pos.sort()
    return _snippet(" ".join(w for _, w in pos))


def _is_preprint(venue: str, types) -> bool:
    types = [t.lower() for t in (types or [])]
    if any(t in {"preprint", "posted-content"} for t in types):
        return True
    name = (venue or "").lower()
    return any(p in name for p in PREPRINT_SOURCES)


def _blank_entry() -> dict:
    return {"note": "", "thumbnail": "", "manual": False}


# ---- OpenAlex --------------------------------------------------------------

def _entry_from_openalex(work: dict) -> dict:
    src = ((work.get("primary_location") or {}).get("source") or {})
    doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
    authors = [(a.get("author") or {}).get("display_name")
               for a in (work.get("authorships") or [])]
    venue = src.get("display_name")
    landing = (work.get("primary_location") or {}).get("landing_page_url")
    e = _blank_entry()
    e.update({
        "id": _short_id(work.get("id")),
        "title": work.get("title") or work.get("display_name") or "Untitled",
        "authors": [a for a in authors if a],
        "year": work.get("publication_year"),
        "venue": venue,
        "type": "preprint" if (
            _is_preprint(venue, [work.get("type")])
            or (src.get("type") == "repository")
        ) else "published",
        "doi": doi,
        "url": (f"https://doi.org/{doi}" if doi else landing),
        "preview": _reconstruct_abstract(work.get("abstract_inverted_index")),
    })
    return e


def _fetch_openalex() -> list[dict]:
    paper = _get_json(f"{OA}/works/https://doi.org/{DOI}?mailto={MAILTO}")
    wid = _short_id(paper.get("id"))
    print(f"OpenAlex: paper is {wid} ({paper.get('title')})")
    out, cursor = [], "*"
    while cursor:
        q = urllib.parse.urlencode({
            "filter": f"cites:{wid}", "per-page": 200,
            "cursor": cursor, "mailto": MAILTO,
        })
        data = _get_json(f"{OA}/works?{q}")
        out += [_entry_from_openalex(w) for w in data.get("results", [])]
        cursor = (data.get("meta") or {}).get("next_cursor")
        time.sleep(0.2)
    print(f"OpenAlex: {len(out)} citing work(s)")
    return out


# ---- Semantic Scholar ------------------------------------------------------

def _entry_from_s2(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    doi = ext.get("DOI")
    authors = [a.get("name") for a in (p.get("authors") or [])]
    venue = (p.get("publicationVenue") or {}).get("name") or p.get("venue")
    e = _blank_entry()
    e.update({
        "id": p.get("paperId"),
        "title": p.get("title") or "Untitled",
        "authors": [a for a in authors if a],
        "year": p.get("year"),
        "venue": venue,
        "type": "preprint" if _is_preprint(venue, p.get("publicationTypes")) else "published",
        "doi": doi,
        "arxiv": ext.get("ArXiv"),
        "url": (f"https://doi.org/{doi}" if doi
                else (f"https://arxiv.org/abs/{ext['ArXiv']}" if ext.get("ArXiv") else p.get("url"))),
        "preview": _snippet(p.get("abstract") or ""),
    })
    return e


def _fetch_s2() -> list[dict]:
    fields = "title,year,authors,externalIds,abstract,venue,publicationVenue,publicationTypes,url"
    out, offset = [], 0
    try:
        while True:
            q = urllib.parse.urlencode({"fields": fields, "limit": 1000, "offset": offset})
            data = _get_json(f"{S2}/paper/DOI:{DOI}/citations?{q}")
            batch = data.get("data", [])
            out += [_entry_from_s2(d["citingPaper"]) for d in batch if d.get("citingPaper")]
            nxt = data.get("next")
            if not batch or nxt is None:
                break
            offset = nxt
            time.sleep(1.0)
        print(f"Semantic Scholar: {len(out)} citing work(s)")
    except Exception as exc:
        print(f"Semantic Scholar: skipped ({exc})", file=sys.stderr)
    return out


# ---- og:image (graphical preview, best-effort) -----------------------------

_OG_RE = re.compile(
    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', re.I)


def _fetch_og_image(url: str | None) -> str:
    if not url:
        return ""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read(200_000).decode("utf-8", "ignore")
        m = _OG_RE.search(html)
        return m.group(1).strip() if m else ""
    except Exception:
        return ""


# ---- merge -----------------------------------------------------------------

def _key(e: dict) -> str:
    return (e.get("doi") or e.get("id") or e.get("title") or "").lower()


def _union(*sources: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}
    for src in sources:
        for e in src:
            k = _key(e)
            if not k:
                continue
            cur = by_key.get(k)
            if cur is None:
                by_key[k] = e
            else:  # fill blanks / prefer a real abstract
                for f, v in e.items():
                    if not cur.get(f) and v:
                        cur[f] = v
                if len(e.get("preview") or "") > len(cur.get("preview") or ""):
                    cur["preview"] = e["preview"]
    return list(by_key.values())


def merge(existing: list[dict], fetched: list[dict]) -> list[dict]:
    by_key = {_key(e): e for e in existing if isinstance(e, dict)}
    for new in fetched:
        k = _key(new)
        old = by_key.get(k)
        if old is None:
            by_key[k] = new
        elif old.get("manual"):
            continue  # fully curated → untouched
        else:
            new["note"] = old.get("note") or new.get("note") or ""
            new["thumbnail"] = old.get("thumbnail") or new.get("thumbnail") or ""
            by_key[k] = new
    merged = list(by_key.values())
    merged.sort(key=lambda e: (-(e.get("year") or 0), (e.get("title") or "").lower()))
    return merged


def main() -> int:
    print(f"Resolving citations of DOI {DOI} …")
    try:
        oa = _fetch_openalex()
    except Exception as exc:
        print(f"ERROR: OpenAlex query failed: {exc}", file=sys.stderr)
        return 1
    s2 = _fetch_s2()
    fetched = _union(oa, s2)
    print(f"Union: {len(fetched)} unique citing work(s)")

    for e in fetched:
        if not e.get("thumbnail"):
            img = _fetch_og_image(e.get("url"))
            if img:
                e["thumbnail"] = img

    existing = []
    if os.path.isfile(YAML_PATH):
        with open(YAML_PATH, "r", encoding="utf-8") as fh:
            existing = yaml.safe_load(fh) or []

    merged = merge(existing, fetched)
    with open(YAML_PATH, "w", encoding="utf-8") as fh:
        fh.write("# Auto-refreshed by fetch_citations.py (OpenAlex + Semantic Scholar).\n")
        fh.write("# Set `manual: true` to protect a hand-curated entry from overwrites,\n")
        fh.write("# or add entries by hand (e.g. works only Google Scholar indexes).\n")
        yaml.safe_dump(merged, fh, allow_unicode=True, sort_keys=False, width=100)
    print(f"Wrote {len(merged)} entry(ies) → {YAML_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
