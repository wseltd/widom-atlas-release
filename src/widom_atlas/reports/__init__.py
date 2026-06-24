"""Report artefacts: manifest, tables, figures, Markdown + HTML rendering."""

from widom_atlas.reports.figures import (
    plot_basin_centroids,
    plot_density_slices,
    plot_robustness_bar,
)
from widom_atlas.reports.html import render_html_report
from widom_atlas.reports.manifest import build_manifest, write_manifest
from widom_atlas.reports.markdown import render_markdown_report
from widom_atlas.reports.tables import (
    write_basins_csv,
    write_basins_json,
    write_symmetry_groups_json,
)

__all__ = [
    "build_manifest",
    "plot_basin_centroids",
    "plot_density_slices",
    "plot_robustness_bar",
    "render_html_report",
    "render_markdown_report",
    "write_basins_csv",
    "write_basins_json",
    "write_manifest",
    "write_symmetry_groups_json",
]
