"""Tests for InsertionSamples, Basin, DensityGrid, SymmetryGroup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from widom_atlas.core.models import (
    Basin,
    DensityGrid,
    InsertionSamples,
    SymmetryGroup,
)


def _samples_kwargs(n: int = 3) -> dict:
    return {
        "positions_cart": np.zeros((n, 3)),
        "positions_frac": np.linspace(0.0, 0.9, n)[:, None] * np.ones((1, 3)),
        "energies_eV": np.linspace(-0.3, -0.1, n),
        "accessible": np.array([True] * n),
        "temperature_K": 298.15,
        "gas": "CO2",
    }


# --- InsertionSamples (T005) ---------------------------------------------------


def test_insertion_samples_accepts_valid_minimal_inputs() -> None:
    s = InsertionSamples(**_samples_kwargs(3))
    assert s.n_samples == 3
    assert s.gas == "CO2"


def test_insertion_samples_rejects_mismatched_lengths() -> None:
    kw = _samples_kwargs(3)
    kw["energies_eV"] = np.array([-0.1, -0.2])  # length 2 vs 3
    with pytest.raises(ValidationError):
        InsertionSamples(**kw)


def test_insertion_samples_rejects_unwrapped_frac() -> None:
    kw = _samples_kwargs(3)
    kw["positions_frac"] = np.array([[0.1, 0.1, 1.5], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
    with pytest.raises(ValidationError):
        InsertionSamples(**kw)


def test_insertion_samples_rejects_nonfinite_energy() -> None:
    kw = _samples_kwargs(3)
    kw["energies_eV"] = np.array([-0.1, np.nan, -0.3])
    with pytest.raises(ValidationError):
        InsertionSamples(**kw)


def test_insertion_samples_rejects_invalid_gas() -> None:
    kw = _samples_kwargs(3)
    kw["gas"] = "H2O"
    with pytest.raises(ValidationError):
        InsertionSamples(**kw)


def test_insertion_samples_npz_roundtrip_preserves_arrays(tmp_path: Path) -> None:
    s = InsertionSamples(**_samples_kwargs(4), metadata={"src": "unit-test"})
    p = tmp_path / "samples.npz"
    s.to_npz(p)
    loaded = InsertionSamples.from_npz(p)
    np.testing.assert_array_equal(loaded.positions_cart, s.positions_cart)
    np.testing.assert_array_equal(loaded.positions_frac, s.positions_frac)
    np.testing.assert_array_equal(loaded.energies_eV, s.energies_eV)
    np.testing.assert_array_equal(loaded.accessible, s.accessible)
    assert loaded.temperature_K == s.temperature_K
    assert loaded.gas == s.gas
    assert loaded.metadata == s.metadata


# --- Basin (T006) --------------------------------------------------------------


def _basin(**overrides) -> dict:
    base = {
        "basin_id": 0,
        "count": 10,
        "weight": 0.4,
        "centroid_frac": (0.1, 0.2, 0.3),
        "centroid_cart_A": (1.0, 2.0, 3.0),
        "mean_energy_eV": -0.25,
        "std_energy_eV": 0.01,
        "accessible_fraction": 1.0,
        "spread_A": 0.5,
    }
    base.update(overrides)
    return base


def test_basin_rejects_weight_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Basin(**_basin(weight=1.5))
    with pytest.raises(ValidationError):
        Basin(**_basin(weight=-0.1))


def test_basin_rejects_unwrapped_centroid_frac() -> None:
    with pytest.raises(ValidationError):
        Basin(**_basin(centroid_frac=(1.2, 0.2, 0.3)))


def test_basin_rejects_negative_spread() -> None:
    with pytest.raises(ValidationError):
        Basin(**_basin(spread_A=-0.01))


def test_basin_rejects_nonfinite_mean_energy() -> None:
    with pytest.raises(ValidationError):
        Basin(**_basin(mean_energy_eV=float("nan")))


def test_basin_as_row_contains_all_scalar_fields() -> None:
    b = Basin(**_basin())
    row = b.as_row()
    expected_keys = {
        "basin_id",
        "count",
        "weight",
        "centroid_frac_a",
        "centroid_frac_b",
        "centroid_frac_c",
        "centroid_cart_x_A",
        "centroid_cart_y_A",
        "centroid_cart_z_A",
        "mean_energy_eV",
        "std_energy_eV",
        "accessible_fraction",
        "spread_A",
    }
    assert set(row.keys()) == expected_keys


# --- DensityGrid (T007) --------------------------------------------------------


def _density_kwargs(grid: np.ndarray | None = None) -> dict:
    if grid is None:
        grid = np.ones((4, 4, 4)) / 64.0
    return {
        "grid": grid,
        "shape": tuple(grid.shape),
        "cell_A": np.eye(3) * 5.0,
        "spacing_A": (1.25, 1.25, 1.25),
        "temperature_K": 298.15,
        "gas": "CO2",
        "n_source_samples": 1000,
    }


def test_density_grid_accepts_valid_uniform_grid() -> None:
    d = DensityGrid(**_density_kwargs())
    assert d.shape == (4, 4, 4)
    assert np.isclose(d.grid.sum(), 1.0)


def test_density_grid_rejects_unnormalised_grid() -> None:
    grid = np.ones((4, 4, 4))  # sum != 1
    with pytest.raises(ValidationError):
        DensityGrid(**_density_kwargs(grid=grid))


def test_density_grid_rejects_negative_values() -> None:
    grid = np.ones((4, 4, 4)) / 64.0
    grid[0, 0, 0] = -0.001
    with pytest.raises(ValidationError):
        DensityGrid(**_density_kwargs(grid=grid))


def test_density_grid_rejects_shape_mismatch() -> None:
    kw = _density_kwargs()
    kw["shape"] = (8, 8, 8)
    with pytest.raises(ValidationError):
        DensityGrid(**kw)


def test_density_grid_rejects_singular_cell() -> None:
    kw = _density_kwargs()
    cell = np.eye(3) * 5.0
    cell[2, 2] = 0.0
    kw["cell_A"] = cell
    with pytest.raises(ValidationError):
        DensityGrid(**kw)


def test_density_grid_rejects_inconsistent_spacing() -> None:
    kw = _density_kwargs()
    kw["spacing_A"] = (2.0, 2.0, 2.0)  # cell norm 5 / 4 = 1.25, not 2.0
    with pytest.raises(ValidationError):
        DensityGrid(**kw)


def test_density_grid_npz_roundtrip_preserves_grid_and_cell(tmp_path: Path) -> None:
    d = DensityGrid(**_density_kwargs())
    p = tmp_path / "density.npz"
    d.to_npz(p)
    loaded = DensityGrid.from_npz(p)
    np.testing.assert_allclose(loaded.grid, d.grid)
    np.testing.assert_allclose(loaded.cell_A, d.cell_A)
    assert loaded.shape == d.shape
    assert loaded.spacing_A == d.spacing_A
    assert loaded.gas == d.gas


# --- SymmetryGroup (T008) ------------------------------------------------------


def _sg_kwargs(**overrides) -> dict:
    base = {
        "group_id": 0,
        "member_basin_ids": (0, 1, 2, 3),
        "space_group_symbol": "Fm-3m",
        "space_group_number": 225,
        "n_operations_used": 48,
        "tolerances": {
            "symprec": 1e-2,
            "angle_tolerance_deg": 5.0,
            "basin_match_tol_A": 0.35,
            "energy_match_tol_kJmol": 2.0,
        },
        "grouping_confidence": 0.95,
    }
    base.update(overrides)
    return base


def test_symmetry_group_accepts_valid_high_confidence_group() -> None:
    sg = SymmetryGroup(**_sg_kwargs())
    assert sg.space_group_number == 225
    assert len(sg.member_basin_ids) == 4


def test_symmetry_group_rejects_empty_members() -> None:
    with pytest.raises(ValidationError):
        SymmetryGroup(**_sg_kwargs(member_basin_ids=()))


def test_symmetry_group_rejects_unsorted_or_duplicate_members() -> None:
    with pytest.raises(ValidationError):
        SymmetryGroup(**_sg_kwargs(member_basin_ids=(2, 1, 0)))
    with pytest.raises(ValidationError):
        SymmetryGroup(**_sg_kwargs(member_basin_ids=(0, 1, 1)))


def test_symmetry_group_rejects_space_group_number_out_of_range() -> None:
    with pytest.raises(ValidationError):
        SymmetryGroup(**_sg_kwargs(space_group_number=0))
    with pytest.raises(ValidationError):
        SymmetryGroup(**_sg_kwargs(space_group_number=231))


def test_symmetry_group_rejects_missing_tolerance_keys() -> None:
    with pytest.raises(ValidationError):
        SymmetryGroup(**_sg_kwargs(tolerances={"symprec": 1e-2}))


def test_symmetry_group_rejects_invalid_uncertainty_flag() -> None:
    with pytest.raises(ValidationError):
        SymmetryGroup(**_sg_kwargs(uncertainty_flags=("not_a_real_flag",)))


def test_symmetry_group_rejects_low_confidence_without_flags() -> None:
    with pytest.raises(ValidationError):
        SymmetryGroup(**_sg_kwargs(grouping_confidence=0.2, uncertainty_flags=()))


def test_symmetry_group_accepts_low_confidence_with_explanatory_flag() -> None:
    sg = SymmetryGroup(
        **_sg_kwargs(
            grouping_confidence=0.3,
            uncertainty_flags=("tolerance_ambiguous",),
        )
    )
    assert sg.uncertainty_flags == ("tolerance_ambiguous",)
