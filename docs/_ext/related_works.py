"""Sphinx extension: a ``related-works`` directive.

Renders a card gallery of the works that cite PhysioEx, read from a curated
YAML file (single source of truth, see ``related_works_yaml`` in conf.py).
The cards are produced as ``sphinx-design`` grids so the look matches the rest
of the site; entries are grouped into *Published* and *Preprints*.

The YAML is refreshed semi-automatically by ``docs/related/fetch_citations.py``
(OpenAlex); this directive only reads it at build time — no network access.
"""

from __future__ import annotations

import os

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList


def _esc(text: str) -> str:
    """Escape the RST inline-markup characters so arbitrary titles/abstracts
    (which may contain ``*``, backticks, ``|`` …) can't break the build."""
    if text is None:
        return ""
    text = str(text).replace("\n", " ").strip()
    for ch in ("\\", "*", "`", "|", "_"):
        text = text.replace(ch, "\\" + ch)
    return text


def _authors(entry: dict) -> str:
    a = entry.get("authors") or []
    if isinstance(a, str):
        a = [a]
    if not a:
        return ""
    if len(a) > 4:
        a = a[:4] + ["et al."]
    return ", ".join(a)


def _meta_line(entry: dict) -> str:
    bits = [b for b in (_authors(entry), entry.get("venue"), entry.get("year")) if b]
    return " · ".join(_esc(b) for b in bits)


def _card(entry: dict) -> list[str]:
    """One ``grid-item-card`` block as a list of RST lines."""
    title = _esc(entry.get("title") or "Untitled")
    lines = [f".. grid-item-card:: {title}", ""]

    meta = _meta_line(entry)
    if meta:
        lines += [f"   {meta}", ""]

    preview = _esc(entry.get("preview") or entry.get("note") or "")
    if preview:
        lines += [f"   {preview}", ""]
    note = entry.get("note")
    if note and entry.get("preview"):
        lines += [f"   *{_esc(note)}*", ""]

    # Footer: type badge + link badges.
    is_pre = (entry.get("type") or "").lower() == "preprint"
    badge = ":bdg-warning:`Preprint`" if is_pre else ":bdg-success:`Published`"
    footer = [badge]
    url = entry.get("url") or (
        f"https://doi.org/{entry['doi']}" if entry.get("doi") else None
    )
    if url:
        footer.append(f":bdg-link-primary:`DOI <{url}>`" if entry.get("doi")
                      else f":bdg-link-primary:`Link <{url}>`")
    if entry.get("arxiv"):
        footer.append(f":bdg-link-info:`arXiv <https://arxiv.org/abs/{entry['arxiv']}>`")
    lines += ["   +++", "   " + " ".join(footer)]
    return lines


def _grid(entries: list[dict]) -> list[str]:
    out = [".. grid:: 1 2 2 3", "   :gutter: 3", ""]
    for e in entries:
        out += ["   " + ln if ln else "" for ln in _card(e)]
        out += [""]
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
            para = nodes.paragraph(
                text="No citing works are indexed yet. This page updates as "
                "publications and preprints citing PhysioEx are picked up."
            )
            note += para
            return [note]

        def _year(e):
            try:
                return int(e.get("year") or 0)
            except (TypeError, ValueError):
                return 0

        published = sorted([e for e in entries if (e.get("type") or "").lower() != "preprint"],
                           key=_year, reverse=True)
        preprints = sorted([e for e in entries if (e.get("type") or "").lower() == "preprint"],
                          key=_year, reverse=True)

        rst: list[str] = []
        if published:
            rst += [".. rubric:: Published", ""] + _grid(published) + [""]
        if preprints:
            rst += [".. rubric:: Preprints", ""] + _grid(preprints) + [""]

        container = nodes.container()
        self.state.nested_parse(
            StringList(rst, source="related-works"), self.content_offset, container
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
