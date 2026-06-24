"""Tests for reports module: manifest, tables, figures, markdown, html (T036–T040)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from jinja2.exceptions import UndefinedError

from widom_atlas.core.models import Basin, DensityGrid, RobustnessMetrics, SymmetryGroup
from widom_atlas.reports.figures import (
    plot_basin_centroids,
    plot_density_slices,
    plot_robustness_bar,
)
from widom_atlas.reports.html import render_html_report
from widom_atlas.reports.manifest import _sha256_file, build_manifest, write_manifest
from widom_atlas.reports.markdown import render_markdown_report
from widom_atlas.reports.tables import (
    BASIN_CSV_COLUMNS,
    write_basins_csv,
    write_basins_json,
    write_symmetry_groups_json,
)


def _basin(idx: int = 0, **kw) -> Basin:
    base = {
        "basin_id": idx,
        "count": 10,
        "weight": 0.5,
        "centroid_frac": (0.25, 0.25, 0.25),
        "centroid_cart_A": (1.0, 2.0, 3.0),
        "mean_energy_eV": -0.5,
        "std_energy_eV": 0.01,
        "accessible_fraction": 1.0,
        "spread_A": 0.4,
    }
    base.update(kw)
    return Basin(**base)


def _density(shape: tuple[int, int, int] = (4, 4, 4)) -> DensityGrid:
    grid = np.ones(shape) / float(np.prod(shape))
    return DensityGrid(
        grid=grid,
        shape=shape,
        cell_A=np.eye(3) * 5.0,
        spacing_A=tuple(5.0 / s for s in shape),
        temperature_K=298.15,
        gas="CO2",
        n_source_samples=100,
    )


def _sg(idx: int = 0, **kw) -> SymmetryGroup:
    base = {
        "group_id": idx,
        "member_basin_ids": (0, 1),
        "space_group_symbol": "Fm-3m",
        "space_group_number": 225,
        "n_operations_used": 48,
        "tolerances": {
            "symprec": 1e-2,
            "angle_tolerance_deg": 5.0,
            "basin_match_tol_A": 0.35,
            "energy_match_tol_kJmol": 2.0,
        },
        "grouping_confidence": 0.9,
    }
    base.update(kw)
    return SymmetryGroup(**base)


# --- T036 manifest ----------------------------------------------------------


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    p = tmp_path / "foo.bin"
    p.write_bytes(b"abc")
    assert _sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_build_manifest_captures_versions_and_hashes(tmp_path: Path) -> None:
    sp = tmp_path / "samples.npz"
    sp.write_bytes(b"sample-bytes")
    stp = tmp_path / "structure.cif"
    stp.write_bytes(b"structure-bytes")
    m = build_manifest(
        structure_id="X",
        gas="CO2",
        temperature_K=298.15,
        sample_path=sp,
        structure_path=stp,
        parameters={"symprec": 1e-2},
    )
    assert m.gas == "CO2"
    assert m.python_version
    assert "numpy" in m.dependency_versions
    assert m.structure_sha256 == hashlib.sha256(b"structure-bytes").hexdigest()


def test_build_manifest_handles_missing_dependency_version(tmp_path: Path) -> None:
    sp = tmp_path / "s.npz"
    stp = tmp_path / "x.cif"
    sp.write_bytes(b"a")
    stp.write_bytes(b"b")
    m = build_manifest(
        structure_id="X",
        gas="CO2",
        temperature_K=298.15,
        sample_path=sp,
        structure_path=stp,
        parameters={},
    )
    assert all(v != "" for v in m.dependency_versions.values())


def test_write_manifest_is_deterministic_sorted_json(tmp_path: Path) -> None:
    sp = tmp_path / "s.npz"
    stp = tmp_path / "x.cif"
    sp.write_bytes(b"a")
    stp.write_bytes(b"b")
    m = build_manifest(
        structure_id="X",
        gas="CO2",
        temperature_K=298.15,
        sample_path=sp,
        structure_path=stp,
        parameters={"b": 1, "a": 2},
    )
    p1 = tmp_path / "manifest1.json"
    p2 = tmp_path / "manifest2.json"
    write_manifest(m, p1)
    write_manifest(m, p2)
    assert p1.read_text() == p2.read_text()


def test_build_manifest_includes_license_metadata_when_provided(tmp_path: Path) -> None:
    sp = tmp_path / "s.npz"
    stp = tmp_path / "x.cif"
    sp.write_bytes(b"a")
    stp.write_bytes(b"b")
    m = build_manifest(
        structure_id="X",
        gas="CO2",
        temperature_K=298.15,
        sample_path=sp,
        structure_path=stp,
        parameters={},
        license_metadata={"license": "CC BY 4.0"},
    )
    assert m.dataset_license == "CC BY 4.0"


# --- T037 tables ------------------------------------------------------------


def test_write_basins_csv(tmp_path: Path) -> None:
    p = tmp_path / "basins.csv"
    write_basins_csv([_basin(0), _basin(1)], p)
    text = p.read_text(encoding="utf-8")
    header = text.splitlines()[0].split(",")
    assert tuple(header) == BASIN_CSV_COLUMNS


def test_write_basins_json(tmp_path: Path) -> None:
    p = tmp_path / "basins.json"
    write_basins_json([_basin(1), _basin(0)], p)
    payload = json.loads(p.read_text())
    assert [b["basin_id"] for b in payload["basins"]] == [0, 1]


def test_write_symmetry_groups_json(tmp_path: Path) -> None:
    p = tmp_path / "groups.json"
    write_symmetry_groups_json([_sg(1), _sg(0)], p)
    payload = json.loads(p.read_text())
    assert [g["group_id"] for g in payload["symmetry_groups"]] == [0, 1]


def test_basins_csv_column_order_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "basins.csv"
    write_basins_csv([_basin()], p)
    header = p.read_text().splitlines()[0]
    assert header == ",".join(BASIN_CSV_COLUMNS)


def test_basins_json_is_deterministic(tmp_path: Path) -> None:
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    write_basins_json([_basin(0), _basin(1)], p1)
    write_basins_json([_basin(0), _basin(1)], p2)
    assert p1.read_text() == p2.read_text()


def test_symmetry_groups_json_consumes_T029_model(tmp_path: Path) -> None:
    p = tmp_path / "g.json"
    write_symmetry_groups_json([_sg()], p)
    payload = json.loads(p.read_text())
    assert "space_group_number" in payload["symmetry_groups"][0]


# --- T038 figures -----------------------------------------------------------


def test_plot_density_slices_writes_png(tmp_path: Path) -> None:
    out = plot_density_slices(_density(), tmp_path / "density.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_basin_centroids_writes_png(tmp_path: Path) -> None:
    out = plot_basin_centroids([_basin(0), _basin(1, centroid_frac=(0.7, 0.7, 0.7))], None, tmp_path / "b.png")
    assert out.exists()


def test_plot_robustness_bar_writes_png(tmp_path: Path) -> None:
    metrics = RobustnessMetrics(
        delta_ln_KH=-0.1,
        delta_Qads_kJmol=-0.5,
        basin_count_pristine=4,
        basin_count_perturbed=4,
        basin_count_change=0,
        basin_persistence_fraction=0.9,
        basin_splitting_count=0,
        mean_basin_displacement_A=0.1,
        accessibility_change=0.0,
    )
    out = plot_robustness_bar(metrics, tmp_path / "r.png")
    assert out.exists()


def test_figures_use_agg_backend() -> None:
    assert matplotlib.get_backend().lower() in {"agg", "module://matplotlib_inline.backend_inline"}


def test_plot_density_slices_consumes_T021_DensityGrid(tmp_path: Path) -> None:
    out = plot_density_slices(_density(), tmp_path / "out.png", axis="a", n_slices=2)
    assert out.exists()


def test_plot_robustness_bar_consumes_T035_RobustnessMetrics(tmp_path: Path) -> None:
    metrics = RobustnessMetrics(
        delta_ln_KH=None,
        delta_Qads_kJmol=None,
        basin_count_pristine=2,
        basin_count_perturbed=2,
        basin_count_change=0,
        basin_persistence_fraction=1.0,
        basin_splitting_count=0,
        mean_basin_displacement_A=0.05,
        accessibility_change=0.0,
    )
    out = plot_robustness_bar(metrics, tmp_path / "r.png")
    assert out.exists()


# --- T039 markdown / T040 html ----------------------------------------------


def _ctx() -> dict:
    return {
        "structure_metadata": {"structure_id": "ToyCell", "cell_matrix": [[10, 0, 0], [0, 10, 0], [0, 0, 10]]},
        "gas": "CO2",
        "temperature_K": 298.15,
        "samples_summary": {"n_samples": 100, "input_hash": "abc", "mean_energy_eV": -0.5},
        "density_summary": {"shape": [4, 4, 4], "spacing_A": [2.5, 2.5, 2.5], "smoothing_sigma_A": 0.0},
        "basins": [_basin().model_dump(mode="json")],
        "symmetry_groups": [_sg().model_dump(mode="json")],
        "perturbation_summary": [{"kind": "isotropic", "label": "iso1", "notes": ""}],
        "robustness_metrics": {
            "delta_ln_KH": -0.1,
            "delta_Qads_kJmol": -0.5,
            "basin_persistence_fraction": 0.9,
            "mean_basin_displacement_A": 0.1,
        },
        "caveats": ["TOY OUTPUT — not chemically meaningful"],
        "uncertainty_notes": ["centroid_stderr_A may be coarse"],
        "figure_paths": {
            "density_slices": "../figures/density_slices.png",
            "basin_centroids": "../figures/basin_centroids.png",
            "robustness_bar": "../figures/robustness_bar.png",
        },
    }


def test_render_markdown_report_produces_file(tmp_path: Path) -> None:
    p = render_markdown_report(_ctx(), tmp_path / "report.md")
    assert p.exists()
    text = p.read_text()
    assert "widom-atlas report" in text
    assert "Structure & Conditions" in text
    assert "Sample Summary" in text


def test_render_markdown_report_strict_undefined(tmp_path: Path) -> None:
    bad = _ctx()
    del bad["caveats"]
    with pytest.raises(UndefinedError):
        render_markdown_report(bad, tmp_path / "x.md")


def test_render_markdown_report_includes_caveats(tmp_path: Path) -> None:
    p = render_markdown_report(_ctx(), tmp_path / "report.md")
    assert "TOY OUTPUT" in p.read_text()


def test_render_markdown_report_section_order_is_fixed(tmp_path: Path) -> None:
    p = render_markdown_report(_ctx(), tmp_path / "report.md")
    text = p.read_text()
    expected_order = [
        "Structure & Conditions",
        "Sample Summary",
        "Density Map",
        "Basins",
        "Symmetry Grouping",
        "Perturbations",
        "Robustness",
        "Caveats & Uncertainty",
    ]
    last = -1
    for header in expected_order:
        idx = text.find(header)
        assert idx > last, f"section out of order: {header}"
        last = idx


def test_render_markdown_report_consumes_T037_basins_json(tmp_path: Path) -> None:
    ctx = _ctx()
    ctx["basins"] = [_basin().model_dump(mode="json"), _basin(idx=1).model_dump(mode="json")]
    p = render_markdown_report(ctx, tmp_path / "report.md")
    text = p.read_text()
    assert "| 0 |" in text and "| 1 |" in text


def test_render_markdown_report_links_T038_figures(tmp_path: Path) -> None:
    p = render_markdown_report(_ctx(), tmp_path / "report.md")
    text = p.read_text()
    assert "../figures/density_slices.png" in text
    assert "../figures/basin_centroids.png" in text


def test_render_html_report_produces_file(tmp_path: Path) -> None:
    p = render_html_report(_ctx(), tmp_path / "report.html")
    assert p.exists()
    text = p.read_text()
    assert "<html" in text and "Structure" in text


def test_render_html_report_escapes_user_strings(tmp_path: Path) -> None:
    ctx = _ctx()
    ctx["structure_metadata"]["structure_id"] = "<script>alert(1)</script>"
    p = render_html_report(ctx, tmp_path / "report.html")
    text = p.read_text()
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text


def test_render_html_report_section_parity_with_markdown(tmp_path: Path) -> None:
    md_text = render_markdown_report(_ctx(), tmp_path / "r.md").read_text()
    html_text = render_html_report(_ctx(), tmp_path / "r.html").read_text()
    for header in (
        "Structure",
        "Sample Summary",
        "Density Map",
        "Basins",
        "Symmetry Grouping",
        "Perturbations",
        "Robustness",
        "Caveats",
    ):
        assert header in md_text
        assert header in html_text


def test_render_html_report_strict_undefined(tmp_path: Path) -> None:
    bad = _ctx()
    del bad["robustness_metrics"]
    with pytest.raises(UndefinedError):
        render_html_report(bad, tmp_path / "r.html")


def test_render_html_report_has_no_external_references(tmp_path: Path) -> None:
    p = render_html_report(_ctx(), tmp_path / "report.html")
    text = p.read_text()
    for prefix in ("https://cdn", "http://cdn", "//unpkg", "//jsdelivr"):
        assert prefix not in text


def test_render_html_report_accepts_same_context_as_T039(tmp_path: Path) -> None:
    ctx = _ctx()
    render_markdown_report(ctx, tmp_path / "r.md")
    render_html_report(ctx, tmp_path / "r.html")
