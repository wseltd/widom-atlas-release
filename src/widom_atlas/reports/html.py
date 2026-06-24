"""Render HTML reports via Jinja2 (autoescape on, no external resources)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape


def _env() -> Environment:
    return Environment(
        loader=PackageLoader("widom_atlas.reports", "templates"),
        autoescape=select_autoescape(
            enabled_extensions=("html", "html.j2"),
            default_for_string=True,
            default=True,
        ),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


def render_html_report(context: dict[str, Any], out_path: Path) -> Path:
    """Render ``report.html`` mirroring the markdown report sections."""
    env = _env()
    template = env.get_template("report.html.j2")
    text = template.render(**context)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


__all__ = ["render_html_report"]
