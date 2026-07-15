"""Sphinx extension: a ``related-works`` directive.

Renders a card gallery of the works that cite PhysioEx, read from a curated
YAML file (single source of truth, see ``related_works_yaml`` in conf.py).
The cards are produced as ``sphinx-design`` grids so the look matches the rest
of the site; entries are grouped into *Published* and *Preprints*.

The directive is invoked from a MyST page, so it generates **MyST** markup
(colon-fence ``:::{grid-item-card}`` …) and nested-parses it — RST directive
syntax would be silently dropped by the MyST parser.

The YAML is refreshed semi-automatically by ``docs/related/fetch_citations.py``
(OpenAlex); this directive only reads it at build time — no network access.
"""

from __future__ import annotations

import os

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

_SPECIAL = ("\\", "`", "*", "_", "[", "]", "<", ">", "|")
_PLACEHOLDER = "_thumbs/_placeholder.svg"   # under docs/related/


def _thumb_src(entry: dict) -> str:
    """Resolve a card image: a remote URL as-is, a repo path made absolute from
    the source root, or a neutral placeholder when none is set."""
    t = (entry.get("thumbnail") or "").strip() or _PLACEHOLDER
    if t.startswith(("http://", "https://")):
        return t
    return "/related/" + t.lstrip("/")


def _esc(text) -> str:
    """Escape MyST/Markdown inline-markup characters so arbitrary titles and
    abstracts can't break the build or the layout."""
    if text is None:
        return ""
    text = str(text).replace("\n", " ").strip()
    for ch in _SPECIAL:
        text = text.replace(ch, "\\" + ch)
    return text


def _authors(entry: dict) -> str:
    a = entry.get("authors") or []
    if isinstance(a, str):
        a = [a]
    a = [x for x in a if x]
    if not a:
        return ""
    if len(a) > 4:
        a = a[:4] + ["et al."]
    return ", ".join(a)


def _meta_line(entry: dict) -> str:
    bits = [b for b in (_authors(entry), entry.get("venue"), entry.get("year")) if b]
    return " · ".join(_esc(b) for b in bits)


def _card(entry: dict) -> list[str]:
    """One MyST ``grid-item-card`` colon-fence as a list of lines."""
    title = _esc(entry.get("title") or "Untitled")
    lines = [f":::{{grid-item-card}} {title}", f":img-top: {_thumb_src(entry)}", ""]

    meta = _meta_line(entry)
    if meta:
        lines += [meta, ""]

    preview = _esc(entry.get("preview") or "")
    if preview:
        lines += [preview, ""]
    note = entry.get("note")
    if note:
        lines += [f"*{_esc(note)}*", ""]

    # Footer (after +++): type badge + link badges.
    is_pre = (entry.get("type") or "").lower() == "preprint"
    footer = ["{bdg-warning}`Preprint`" if is_pre else "{bdg-success}`Published`"]
    url = entry.get("url") or (
        f"https://doi.org/{entry['doi']}" if entry.get("doi") else None
    )
    if url:
        label = "DOI" if entry.get("doi") else "Link"
        footer.append(f"{{bdg-link-primary}}`{label} <{url}>`")
    if entry.get("arxiv"):
        footer.append(
            f"{{bdg-link-info}}`arXiv <https://arxiv.org/abs/{entry['arxiv']}>`"
        )
    lines += ["+++", " ".join(footer), ":::", ""]
    return lines


def _grid(entries: list[dict]) -> list[str]:
    # 1 column on mobile, 2 from tablet up — wider, more readable cards.
    out = ["::::{grid} 1 1 2 2", ":gutter: 3", ":margin: 2", ""]
    for e in entries:
        out += _card(e)
    out += ["::::", ""]
    return out


class RelatedWorksDirective(Directive):
    has_content = False

    def run(self):
        env = self.state.document.settings.env
        rel = env.config.related_works_yaml
        path = os.path.join(env.srcdir, rel)
        env.note_dependency(path)

        entries = []
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or []
            if isinstance(data, list):
                entries = [e for e in data if isinstance(e, dict) and e.get("title")]

        if not entries:
            note = nodes.note()
            note += nodes.paragraph(
                text="No citing works are indexed yet. This page updates as "
                "publications and preprints citing PhysioEx are picked up."
            )
            return [note]

        def _year(e):
            try:
                return int(e.get("year") or 0)
            except (TypeError, ValueError):
                return 0

        published = sorted(
            [e for e in entries if (e.get("type") or "").lower() != "preprint"],
            key=_year, reverse=True,
        )
        preprints = sorted(
            [e for e in entries if (e.get("type") or "").lower() == "preprint"],
            key=_year, reverse=True,
        )

        md: list[str] = []
        if published:
            md += ["```{rubric} Published", "```", ""] + _grid(published)
        if preprints:
            md += ["```{rubric} Preprints", "```", ""] + _grid(preprints)

        container = nodes.container()
        self.state.nested_parse(
            StringList(md, source="related-works"), self.content_offset, container
        )
        return container.children


def setup(app):
    app.add_config_value("related_works_yaml", "related/works.yaml", "env")
    app.add_directive("related-works", RelatedWorksDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
