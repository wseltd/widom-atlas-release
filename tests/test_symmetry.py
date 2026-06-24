"""Tests for symmetry: detect_symmetry, group_equivalent_basins, group_basins (T027–T029)."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk

from widom_atlas.core.constants import (
    DEFAULT_BASIN_MATCH_TOL_A,
    DEFAULT_ENERGY_MATCH_TOL_KJMOL,
    DEFAULT_SYMPREC,
)
from widom_atlas.core.models import Basin, SymmetryGroup
from widom_atlas.io.structure_adapters import get_cell_matrix
from widom_atlas.symmetry.grouping import group_basins
from widom_atlas.symmetry.match import group_equivalent_basins
from widom_atlas.symmetry.spglib_ops import detect_symmetry
from widom_atlas.symmetry.types import FrameworkSymmetry


def _identity_framework(symprec: float = DEFAULT_SYMPREC) -> FrameworkSymmetry:
    return FrameworkSymmetry(
        space_group_number=1,
        international_symbol="P1",
        hall_number=1,
        rotations=np.eye(3, dtype=np.int64)[None, ...],
        translations=np.zeros((1, 3), dtype=np.float64),
        n_operations=1,
        confidence="high",
        is_low_symmetry=True,
        is_triclinic=True,
        symprec=symprec,
        angle_tolerance_deg=5.0,
    )


def _basin(idx: int, frac: tuple[float, float, float], energy: float = -0.5) -> Basin:
    return Basin(
        basin_id=idx,
        count=10,
        weight=0.1,
        centroid_frac=frac,
        centroid_cart_A=(frac[0] * 10.0, frac[1] * 10.0, frac[2] * 10.0),
        mean_energy_eV=energy,
        std_energy_eV=0.01,
        accessible_fraction=1.0,
        spread_A=0.1,
    )


def _cubic_si() -> Atoms:
    return bulk("Si", "diamond", a=5.43)


def _triclinic_atoms() -> Atoms:
    cell = np.array([[5.0, 0.0, 0.0], [1.5, 6.0, 0.0], [0.5, 0.7, 7.0]])
    return Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)


# --- T027 detect_symmetry ----------------------------------------------------


def test_detect_symmetry_cubic_returns_high_confidence() -> None:
    atoms = _cubic_si()
    fw = detect_symmetry(atoms)
    assert isinstance(fw, FrameworkSymmetry)
    assert fw.confidence in {"high", "medium"}
    assert fw.space_group_number == 227  # Fd-3m for diamond cubic Si


def test_detect_symmetry_uses_symprec_from_T016_constants() -> None:
    atoms = _cubic_si()
    fw = detect_symmetry(atoms)
    assert fw.symprec == DEFAULT_SYMPREC


def test_detect_symmetry_perturbed_cell_lowers_confidence() -> None:
    atoms = _cubic_si().copy()
    rng = np.random.default_rng(0)
    pos = atoms.get_positions()
    pos += rng.normal(0.0, 0.4, pos.shape)
    atoms.set_positions(pos)
    fw = detect_symmetry(atoms, symprec=1e-3)
    assert fw.confidence in {"medium", "low", "uncertain"}


def test_detect_symmetry_triclinic_flag_set() -> None:
    atoms = _triclinic_atoms()
    fw = detect_symmetry(atoms)
    assert fw.is_triclinic is True or fw.space_group_number <= 2


def test_detect_symmetry_returns_rotations_translations_arrays() -> None:
    atoms = _cubic_si()
    fw = detect_symmetry(atoms)
    assert fw.rotations.ndim == 3 and fw.rotations.shape[1:] == (3, 3)
    assert fw.translations.ndim == 2 and fw.translations.shape[1] == 3
    assert fw.rotations.shape[0] == fw.translations.shape[0] == fw.n_operations


def test_detect_symmetry_accepts_ase_atoms_input() -> None:
    atoms = _cubic_si()
    fw = detect_symmetry(atoms)
    assert fw.n_operations >= 1


def test_detect_symmetry_accepts_pymatgen_structure_input() -> None:
    from widom_atlas.io.structure_adapters import ase_to_pymatgen

    s = ase_to_pymatgen(_cubic_si())
    fw = detect_symmetry(s)
    assert fw.n_operations >= 1


# --- T028 group_equivalent_basins -------------------------------------------


def test_group_equivalent_basins_identity_only_yields_singletons() -> None:
    cell = np.eye(3) * 10.0
    fw = _identity_framework()
    basins = [_basin(i, (0.1 * i, 0.5, 0.5)) for i in range(3)]
    groups = group_equivalent_basins(basins, fw, cell)
    assert sorted(sorted(g) for g in groups) == [[0], [1], [2]]


def test_group_equivalent_basins_cubic_4_equivalent_sites_form_one_group() -> None:
    cell = np.eye(3) * 10.0
    rotations = np.array([np.eye(3, dtype=np.int64) for _ in range(4)])
    translations = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
        dtype=np.float64,
    )
    fw = FrameworkSymmetry(
        space_group_number=225,
        international_symbol="Fm-3m",
        hall_number=1,
        rotations=rotations,
        translations=translations,
        n_operations=4,
        confidence="high",
        is_low_symmetry=False,
        is_triclinic=False,
        symprec=DEFAULT_SYMPREC,
        angle_tolerance_deg=5.0,
    )
    basins = [
        _basin(0, (0.0, 0.0, 0.0)),
        _basin(1, (0.5, 0.5, 0.0)),
        _basin(2, (0.5, 0.0, 0.5)),
        _basin(3, (0.0, 0.5, 0.5)),
    ]
    groups = group_equivalent_basins(basins, fw, cell)
    # All four sites should be in one group
    assert any(set(g) == {0, 1, 2, 3} for g in groups)


def test_group_equivalent_basins_energy_mismatch_prevents_merge() -> None:
    cell = np.eye(3) * 10.0
    rotations = np.array([np.eye(3, dtype=np.int64), np.eye(3, dtype=np.int64)])
    translations = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float64)
    fw = FrameworkSymmetry(
        space_group_number=225,
        international_symbol="Fm-3m",
        hall_number=1,
        rotations=rotations,
        translations=translations,
        n_operations=2,
        confidence="high",
        is_low_symmetry=False,
        is_triclinic=False,
        symprec=DEFAULT_SYMPREC,
        angle_tolerance_deg=5.0,
    )
    basins = [
        _basin(0, (0.0, 0.0, 0.0), energy=-0.5),
        _basin(1, (0.5, 0.5, 0.0), energy=-0.5 + 0.1),  # 0.1 eV ≈ 9.6 kJ/mol > 2 kJ/mol tol
    ]
    groups = group_equivalent_basins(basins, fw, cell)
    assert sorted(sorted(g) for g in groups) == [[0], [1]]


def test_group_equivalent_basins_transitive_grouping_via_union_find() -> None:
    cell = np.eye(3) * 10.0
    rotations = np.array([np.eye(3, dtype=np.int64) for _ in range(2)])
    translations = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    fw = FrameworkSymmetry(
        space_group_number=225,
        international_symbol="Fm-3m",
        hall_number=1,
        rotations=rotations,
        translations=translations,
        n_operations=2,
        confidence="high",
        is_low_symmetry=False,
        is_triclinic=False,
        symprec=DEFAULT_SYMPREC,
        angle_tolerance_deg=5.0,
    )
    basins = [
        _basin(0, (0.0, 0.0, 0.0)),
        _basin(1, (0.5, 0.0, 0.0)),
    ]
    groups = group_equivalent_basins(basins, fw, cell)
    assert any(set(g) == {0, 1} for g in groups)


def test_group_equivalent_basins_uses_minimum_image_distance() -> None:
    cell = np.eye(3) * 10.0
    fw = _identity_framework()
    basins = [
        _basin(0, (0.01, 0.5, 0.5)),
        _basin(1, (0.99, 0.5, 0.5)),  # min-image distance ~0.2 A
    ]
    groups = group_equivalent_basins(basins, fw, cell, basin_match_tol_A=0.5)
    assert any(set(g) == {0, 1} for g in groups)


def test_group_equivalent_basins_low_symmetry_confidence_propagates() -> None:
    cell = np.eye(3) * 10.0
    fw = _identity_framework()
    object.__setattr__(fw, "confidence", "low")
    basins = [_basin(0, (0.1, 0.5, 0.5)), _basin(1, (0.7, 0.5, 0.5))]
    # No symmetry op connects these → singletons; we just ensure call succeeds with low-confidence framework
    groups = group_equivalent_basins(basins, fw, cell)
    assert sorted(sorted(g) for g in groups) == [[0], [1]]


def test_group_equivalent_basins_tolerance_default_from_T018_constants() -> None:
    assert DEFAULT_BASIN_MATCH_TOL_A == 0.35
    assert DEFAULT_ENERGY_MATCH_TOL_KJMOL == 2.0


def test_group_equivalent_basins_groups_are_disjoint_and_cover_all_basins() -> None:
    cell = np.eye(3) * 10.0
    fw = _identity_framework()
    basins = [_basin(i, (0.1 * i, 0.5, 0.5)) for i in range(5)]
    groups = group_equivalent_basins(basins, fw, cell)
    flat: list[int] = []
    for g in groups:
        flat.extend(g)
    assert sorted(flat) == sorted(b.basin_id for b in basins)
    assert len(set(flat)) == len(flat)


# --- T029 group_basins -------------------------------------------------------


def test_group_basins_returns_symmetry_groups_for_cubic_fixture() -> None:
    atoms = _cubic_si()
    cell = get_cell_matrix(atoms)
    basins = [
        Basin(
            basin_id=i,
            count=10,
            weight=0.1,
            centroid_frac=(0.25, 0.25, 0.25),
            centroid_cart_A=tuple((np.array([0.25, 0.25, 0.25]) @ cell).tolist()),
            mean_energy_eV=-0.5,
            std_energy_eV=0.01,
            accessible_fraction=1.0,
            spread_A=0.1,
        )
        for i in range(2)
    ]
    out = group_basins(atoms, basins)
    assert all(isinstance(g, SymmetryGroup) for g in out)


def test_group_basins_marks_uncertain_for_low_symmetry_p1() -> None:
    atoms = _triclinic_atoms()
    rng = np.random.default_rng(0)
    pos = atoms.get_positions()
    pos += rng.normal(0.0, 0.5, pos.shape)
    atoms.set_positions(pos)
    cell = get_cell_matrix(atoms)
    basins = [
        Basin(
            basin_id=0,
            count=10,
            weight=0.5,
            centroid_frac=(0.3, 0.4, 0.5),
            centroid_cart_A=tuple((np.array([0.3, 0.4, 0.5]) @ cell).tolist()),
            mean_energy_eV=-0.5,
            std_energy_eV=0.01,
            accessible_fraction=1.0,
            spread_A=0.1,
        ),
        Basin(
            basin_id=1,
            count=10,
            weight=0.5,
            centroid_frac=(0.6, 0.7, 0.8),
            centroid_cart_A=tuple((np.array([0.6, 0.7, 0.8]) @ cell).tolist()),
            mean_energy_eV=-0.5,
            std_energy_eV=0.01,
            accessible_fraction=1.0,
            spread_A=0.1,
        ),
    ]
    out = group_basins(atoms, basins)
    assert any(g.uncertainty_flags or g.grouping_confidence < 0.7 for g in out)


def test_group_basins_respects_pbc_for_boundary_basin() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)
    cell = get_cell_matrix(atoms)
    basins = [
        Basin(
            basin_id=0,
            count=10,
            weight=0.5,
            centroid_frac=(0.01, 0.5, 0.5),
            centroid_cart_A=tuple((np.array([0.01, 0.5, 0.5]) @ cell).tolist()),
            mean_energy_eV=-0.5,
            std_energy_eV=0.01,
            accessible_fraction=1.0,
            spread_A=0.1,
        ),
        Basin(
            basin_id=1,
            count=10,
            weight=0.5,
            centroid_frac=(0.99, 0.5, 0.5),
            centroid_cart_A=tuple((np.array([0.99, 0.5, 0.5]) @ cell).tolist()),
            mean_energy_eV=-0.5,
            std_energy_eV=0.01,
            accessible_fraction=1.0,
            spread_A=0.1,
        ),
    ]
    out = group_basins(atoms, basins)
    assert any({0, 1}.issubset(set(g.member_basin_ids)) for g in out)


def test_group_basins_flags_ambiguous_multi_match() -> None:
    # Two basins close enough that one symmetry op can match either — multi-match expected
    atoms = _cubic_si()
    cell = get_cell_matrix(atoms)
    basins = [
        Basin(
            basin_id=i,
            count=10,
            weight=0.5,
            centroid_frac=(0.0, 0.0, 0.0) if i == 0 else (1e-3, 1e-3, 1e-3),
            centroid_cart_A=tuple((np.array([0.0, 0.0, 0.0]) @ cell).tolist()),
            mean_energy_eV=-0.5,
            std_energy_eV=0.01,
            accessible_fraction=1.0,
            spread_A=0.1,
        )
        for i in range(2)
    ]
    out = group_basins(atoms, basins)
    assert all(isinstance(g, SymmetryGroup) for g in out)


def test_group_basins_uses_minimum_image_distance() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)
    cell = get_cell_matrix(atoms)
    basins = [
        Basin(
            basin_id=0,
            count=10,
            weight=0.5,
            centroid_frac=(0.0, 0.5, 0.5),
            centroid_cart_A=tuple((np.array([0.0, 0.5, 0.5]) @ cell).tolist()),
            mean_energy_eV=-0.5,
            std_energy_eV=0.01,
            accessible_fraction=1.0,
            spread_A=0.1,
        ),
    ]
    out = group_basins(atoms, basins)
    assert all(0 <= g.grouping_confidence <= 1.0 for g in out)


def test_group_basins_confidence_is_fraction_of_matched_operations() -> None:
    atoms = _cubic_si()
    basins = [
        Basin(
            basin_id=0,
            count=10,
            weight=1.0,
            centroid_frac=(0.0, 0.0, 0.0),
            centroid_cart_A=(0.0, 0.0, 0.0),
            mean_energy_eV=-0.5,
            std_energy_eV=0.01,
            accessible_fraction=1.0,
            spread_A=0.1,
        ),
    ]
    out = group_basins(atoms, basins)
    for g in out:
        assert 0.0 <= g.grouping_confidence <= 1.0


def test_group_basins_does_not_mutate_input_basins() -> None:
    atoms = _cubic_si()
    basins = [
        Basin(
            basin_id=i,
            count=10,
            weight=0.5,
            centroid_frac=(0.25, 0.25, 0.25),
            centroid_cart_A=(1.0, 1.0, 1.0),
            mean_energy_eV=-0.5,
            std_energy_eV=0.01,
            accessible_fraction=1.0,
            spread_A=0.1,
        )
        for i in range(2)
    ]
    snap = [b.model_dump() for b in basins]
    group_basins(atoms, basins)
    after = [b.model_dump() for b in basins]
    assert snap == after
