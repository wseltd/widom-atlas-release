"""T017: geometry primitive tests (atom relabeller, Fm-3m transform, site-truth)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from widom_atlas.v04.cif.fm3m_to_primitive import (
    CellMatrix,
    fm3m_cell,
    fm3m_frac_to_rhombohedral_frac,
    rhombohedral_frac_to_fm3m_frac,
    rhombohedral_primitive_from_fm3m,
)
from widom_atlas.v04.cif.relabel_selftest import run_relabel_selftest
from widom_atlas.v04.geometry.site_truth import (
    evaluate_geometry_self_test,
    minimum_image_distance,
    reconstruct_site_distance,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_fm3m_cell_volume() -> None:
    a = 26.3224
    cell = fm3m_cell(a)
    vol = abs(np.linalg.det(cell.matrix))
    assert vol == pytest.approx(a ** 3)


def test_rhombohedral_primitive_volume_is_quarter_of_fm3m() -> None:
    a = 26.3224
    rho = rhombohedral_primitive_from_fm3m(a)
    vol_rho = abs(np.linalg.det(rho.matrix))
    fm3m = fm3m_cell(a)
    vol_fm3m = abs(np.linalg.det(fm3m.matrix))
    assert vol_rho == pytest.approx(vol_fm3m / 4, rel=1e-12)


def test_fm3m_to_rhombohedral_round_trip() -> None:
    a = 26.3224
    frac = np.array([0.1446, 0.1446, 0.072])  # Wu 2010 Site I C atom
    rho_frac = fm3m_frac_to_rhombohedral_frac(frac, a)
    back = rhombohedral_frac_to_fm3m_frac(rho_frac, a)
    assert np.allclose(back, frac, atol=1e-12)


def test_minimum_image_distance_within_cell() -> None:
    cell = CellMatrix.from_abc_angles(10.0, 10.0, 10.0, 90.0, 90.0, 90.0).matrix
    a = np.array([0.1, 0.1, 0.1])
    b = np.array([0.2, 0.1, 0.1])
    d = minimum_image_distance(a, b, cell)
    assert d == pytest.approx(1.0)


def test_minimum_image_wraps_across_boundary() -> None:
    cell = CellMatrix.from_abc_angles(10.0, 10.0, 10.0, 90.0, 90.0, 90.0).matrix
    a = np.array([0.05, 0.5, 0.5])
    b = np.array([0.95, 0.5, 0.5])
    d = minimum_image_distance(a, b, cell)
    assert d == pytest.approx(1.0)


def test_reconstruct_site_distance() -> None:
    cell = CellMatrix.from_abc_angles(14.62823, 14.62823, 14.62823, 90.0, 90.0, 90.0).matrix
    # Na-Rho Na-OC1 distance per Lozinska
    na_frac = np.array([0.4408, 0.0, 0.0])
    oc1_frac = np.array([0.3828, 0.0, 0.0])
    d = reconstruct_site_distance(na_frac, oc1_frac, cell)
    assert d == pytest.approx(0.058 * 14.62823, abs=0.01)


def test_geometry_self_test_5b_Na_OC1_distance() -> None:
    cell = CellMatrix.from_abc_angles(14.62823, 14.62823, 14.62823, 90.0, 90.0, 90.0).matrix
    na = np.array([0.4408, 0.0, 0.0])
    oc1 = np.array([0.3828, 0.0, 0.0])
    res = evaluate_geometry_self_test(
        branch_id="5b",
        metal_frac=na,
        site_frac=oc1,
        cell_matrix=cell,
        target_distance_angstrom=0.058 * 14.62823,
        tolerance_angstrom=0.20,
    )
    assert res.passes


def test_relabel_mg_mof74_self_test_passes() -> None:
    path = REPO_ROOT / "docs/research/dataset-research-for-v0.4/15/core-mof-sep2014/core-mof-july2014/VOGTIV_clean_h.cif"
    if not path.exists():
        pytest.skip("VOGTIV CIF missing")
    res = run_relabel_selftest(path)
    assert res.passes, res.reason


def test_geometry_self_test_failure_on_wrong_distance() -> None:
    """Tolerance violation: returns passes=False."""
    cell = CellMatrix.from_abc_angles(10.0, 10.0, 10.0, 90.0, 90.0, 90.0).matrix
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.5, 0.0, 0.0])
    res = evaluate_geometry_self_test(
        branch_id="test",
        metal_frac=a,
        site_frac=b,
        cell_matrix=cell,
        target_distance_angstrom=10.0,  # impossible for unit-cell-bound coords
        tolerance_angstrom=0.1,
    )
    assert not res.passes
