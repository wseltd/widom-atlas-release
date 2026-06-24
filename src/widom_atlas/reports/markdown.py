"""Render Markdown reports via Jinja2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined


def _env() -> Environment:
    return Environment(
        loader=PackageLoader("widom_atlas.reports", "templates"),
        autoescape=False,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


def render_markdown_report(context: dict[str, Any], out_path: Path) -> Path:
    """Render ``report.md`` from the canonical template into ``out_path``."""
    env = _env()
    template = env.get_template("report.md.j2")
    text = template.render(**context)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


__all__ = ["render_markdown_report"]
