"""Tests for perturb module: strain, defects, apply_perturbation (T030–T032)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from widom_atlas.core.models import PerturbationSpec
from widom_atlas.io.from_arrays import from_arrays
from widom_atlas.perturb.api import apply_perturbation
from widom_atlas.perturb.defects import DefectRecord, remove_atoms
from widom_atlas.perturb.strain import apply_strain


def _atoms_h2o(cell_diag: float = 10.0) -> Atoms:
    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [0.0, 0.95, 0.0]],
        cell=np.eye(3) * cell_diag,
        pbc=True,
    )


def _ai(atoms: Atoms) -> object:
    rng = np.random.default_rng(0)
    n = 30
    frac = rng.random((n, 3))
    e = rng.normal(-0.2, 0.05, n)
    return from_arrays(structure=atoms, positions_frac=frac, energies_eV=e, temperature_K=298.15, gas="CO2")


# --- T030 strain ------------------------------------------------------------


def test_apply_strain_isotropic_scales_cell_uniformly() -> None:
    a = _atoms_h2o()
    a2 = apply_strain(a, mode="isotropic", value=0.01)
    np.testing.assert_allclose(a2.get_cell().array, a.get_cell().array * 1.01, atol=1e-12)


def test_apply_strain_uniaxial_only_changes_specified_axis() -> None:
    a = _atoms_h2o()
    a2 = apply_strain(a, mode="uniaxial", value=0.02, axis="a")
    new_cell = a2.get_cell().array
    old_cell = a.get_cell().array
    np.testing.assert_allclose(new_cell[0], old_cell[0] * 1.02, atol=1e-12)
    np.testing.assert_allclose(new_cell[1], old_cell[1], atol=1e-12)
    np.testing.assert_allclose(new_cell[2], old_cell[2], atol=1e-12)


def test_apply_strain_affine_accepts_full_matrix() -> None:
    a = _atoms_h2o()
    M = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -0.01]])
    a2 = apply_strain(a, mode="affine", value=M)
    expected = (np.eye(3) + M) @ a.get_cell().array
    np.testing.assert_allclose(a2.get_cell().array, expected, atol=1e-12)


def test_apply_strain_volume_preserving_keeps_determinant() -> None:
    a = _atoms_h2o()
    M = np.diag([0.005, -0.002, -0.003])  # small strain so 2nd-order error stays under tolerance
    a2 = apply_strain(a, mode="volume_preserving", value=M)
    det_old = float(np.linalg.det(a.get_cell().array))
    det_new = float(np.linalg.det(a2.get_cell().array))
    rel = abs(det_new - det_old) / det_old
    assert rel < 1e-4, f"volume change too large: rel={rel:g}"


def test_apply_strain_preserves_fractional_coordinates() -> None:
    a = _atoms_h2o()
    f1 = a.get_scaled_positions()
    a2 = apply_strain(a, mode="isotropic", value=0.01)
    f2 = a2.get_scaled_positions()
    np.testing.assert_allclose(f1, f2, atol=1e-12)


def test_apply_strain_does_not_mutate_input_structure() -> None:
    a = _atoms_h2o()
    original_cell = a.get_cell().array.copy()
    apply_strain(a, mode="isotropic", value=0.01)
    np.testing.assert_array_equal(a.get_cell().array, original_cell)


def test_apply_strain_uniaxial_requires_explicit_axis() -> None:
    a = _atoms_h2o()
    with pytest.raises(ValueError):
        apply_strain(a, mode="uniaxial", value=0.01)


def test_apply_strain_rejects_invalid_mode() -> None:
    a = _atoms_h2o()
    with pytest.raises(ValueError):
        apply_strain(a, mode="bogus", value=0.01)  # type: ignore[arg-type]


def test_apply_strain_triclinic_cell_handled_correctly() -> None:
    cell = np.array([[5.0, 0.0, 0.0], [1.0, 6.0, 0.0], [0.5, 0.5, 7.0]])
    a = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    a2 = apply_strain(a, mode="isotropic", value=0.01)
    np.testing.assert_allclose(a2.get_cell().array, cell * 1.01, atol=1e-12)


# --- T031 defects -----------------------------------------------------------


def test_remove_atoms_returns_structure_without_specified_indices() -> None:
    a = _atoms_h2o()
    a2, _ = remove_atoms(a, [1])
    assert len(a2) == len(a) - 1


def test_remove_atoms_returns_defect_record_with_removed_indices() -> None:
    a = _atoms_h2o()
    _, rec = remove_atoms(a, [1])
    assert isinstance(rec, DefectRecord)
    assert rec.removed_indices == (1,)
    assert rec.original_species == ("H",) or rec.original_species == ("O",)


def test_remove_atoms_rejects_out_of_range_index() -> None:
    a = _atoms_h2o()
    with pytest.raises(ValueError):
        remove_atoms(a, [99])


def test_remove_atoms_rejects_negative_index() -> None:
    a = _atoms_h2o()
    with pytest.raises(ValueError):
        remove_atoms(a, [-1])


def test_remove_atoms_rejects_duplicate_indices() -> None:
    a = _atoms_h2o()
    with pytest.raises(ValueError):
        remove_atoms(a, [0, 0])


def test_remove_atoms_rejects_empty_indices() -> None:
    a = _atoms_h2o()
    with pytest.raises(ValueError):
        remove_atoms(a, [])


def test_remove_atoms_does_not_mutate_input() -> None:
    a = _atoms_h2o()
    n_before = len(a)
    remove_atoms(a, [1])
    assert len(a) == n_before


def test_remove_atoms_preserves_remaining_species_and_positions() -> None:
    a = _atoms_h2o()
    a2, _ = remove_atoms(a, [1])
    remaining_idx = [i for i in range(len(a)) if i != 1]
    np.testing.assert_allclose(a2.positions, a.positions[remaining_idx], atol=1e-12)
    assert a2.get_chemical_symbols() == [a.get_chemical_symbols()[i] for i in remaining_idx]


# --- T032 apply_perturbation ------------------------------------------------


def test_apply_perturbation_strain_returns_new_atlas_input_with_strained_cell() -> None:
    a = _atoms_h2o()
    ai = _ai(a)
    spec = PerturbationSpec(kind="isotropic", magnitude=0.01, label="iso1")
    ai2 = apply_perturbation(ai, spec)
    np.testing.assert_allclose(ai2.cell_matrix_A, ai.cell_matrix_A * 1.01, atol=1e-12)


def test_apply_perturbation_defects_clears_samples() -> None:
    a = _atoms_h2o()
    ai = _ai(a)
    spec = PerturbationSpec(kind="atom_removal", removed_atom_indices=[1], label="rm")
    ai2 = apply_perturbation(ai, spec)
    assert ai2.n_samples == 0
    assert ai2.metadata.get("samples_cleared_due_to_perturbation") is True


def test_apply_perturbation_defects_returns_atlas_input_with_reduced_atom_count() -> None:
    a = _atoms_h2o()
    ai = _ai(a)
    spec = PerturbationSpec(kind="atom_removal", removed_atom_indices=[1, 2], label="rm")
    ai2 = apply_perturbation(ai, spec)
    assert ai2.metadata.get("removed_atom_indices") == [1, 2]


def test_apply_perturbation_strain_then_defects_records_history() -> None:
    a = _atoms_h2o()
    ai = _ai(a)
    spec1 = PerturbationSpec(kind="isotropic", magnitude=0.01, label="iso1")
    spec2 = PerturbationSpec(kind="atom_removal", removed_atom_indices=[1], label="rm")
    ai2 = apply_perturbation(ai, [spec1, spec2])
    history = ai2.metadata.get("perturbation_history")
    assert isinstance(history, list)
    assert len(history) == 2


def test_apply_perturbation_strain_then_defects_applies_in_correct_order() -> None:
    a = _atoms_h2o()
    ai = _ai(a)
    spec1 = PerturbationSpec(kind="isotropic", magnitude=0.01, label="iso1")
    spec2 = PerturbationSpec(kind="atom_removal", removed_atom_indices=[1], label="rm")
    ai2 = apply_perturbation(ai, [spec1, spec2])
    history = ai2.metadata["perturbation_history"]
    assert history[0]["kind"] == "isotropic"
    assert history[1]["kind"] == "atom_removal"


def test_apply_perturbation_sets_samples_cleared_metadata_flag() -> None:
    a = _atoms_h2o()
    ai = _ai(a)
    spec = PerturbationSpec(kind="isotropic", magnitude=0.01, label="iso1")
    ai2 = apply_perturbation(ai, spec)
    assert ai2.metadata.get("samples_cleared_due_to_perturbation") is True


def test_apply_perturbation_does_not_mutate_input() -> None:
    a = _atoms_h2o()
    ai = _ai(a)
    snap_meta = dict(ai.metadata)
    snap_cell = ai.cell_matrix_A.copy()
    spec = PerturbationSpec(kind="isotropic", magnitude=0.01, label="iso1")
    apply_perturbation(ai, spec)
    np.testing.assert_array_equal(ai.cell_matrix_A, snap_cell)
    assert ai.metadata == snap_meta


def test_apply_perturbation_rejects_unknown_kind() -> None:
    a = _atoms_h2o()
    ai = _ai(a)
    with pytest.raises(Exception):
        bogus = PerturbationSpec(kind="unknown_kind", label="x")  # type: ignore[arg-type]
        apply_perturbation(ai, bogus)


def test_apply_perturbation_propagates_strain_validation_error() -> None:
    a = _atoms_h2o()
    ai = _ai(a)
    with pytest.raises(Exception):
        bad = PerturbationSpec(kind="affine", strain_matrix=None, label="bad")  # validator should reject
        apply_perturbation(ai, bad)


def test_apply_perturbation_propagates_defect_validation_error() -> None:
    a = _atoms_h2o()
    ai = _ai(a)
    with pytest.raises(Exception):
        bad = PerturbationSpec(kind="atom_removal", removed_atom_indices=[], label="bad")
        apply_perturbation(ai, bad)


def test_apply_perturbation_appends_provenance_to_metadata_history() -> None:
    a = _atoms_h2o()
    ai = _ai(a)
    spec = PerturbationSpec(kind="isotropic", magnitude=0.01, label="iso1", notes="check provenance")
    ai2 = apply_perturbation(ai, spec)
    history = ai2.metadata["perturbation_history"]
    assert history[-1]["notes"] == "check provenance"
    assert history[-1]["kind"] == "isotropic"
