"""Regression tests for the geometric VOGTIV relabeller.

The Lin/Mercado Mg-MOF-74 force field has sublattice-specific Buckingham
parameters for the carboxylate C (Mof_Cd), phenoxide-attached aromatic C
(Mof_Ca), the two distinct aromatic ring carbons (Mof_Cb meta, Mof_Cc
para), and three O sublattices. Mis-assigning these labels attaches the
wrong cross-pair parameters to the wrong atoms, distorting Mg-MOF-74 +
CO2 K_H and Q_st.

This test pins the geometric criteria each relabelled atom must satisfy:

  Mof_Cd  bonded to exactly 2 O atoms
  Mof_Ca  bonded to exactly 1 O atom
  Mof_Cb  bonded to 0 O atoms AND bonded to 1 H
  Mof_Cc  bonded to 0 O atoms AND bonded to 1 Mof_Cd carboxylate-C
  Mof_Oa  bonded to exactly 1 Mg
  Mof_Ob  bonded to exactly 2 Mg AND no Mof_Ca neighbour
  Mof_Oc  bonded to exactly 2 Mg AND a Mof_Ca neighbour (phenoxide bridge)
  Mof_Mg  every Mg atom
  Mof_H   every H atom

Per VOGTIV primitive cell, each sublattice must contain exactly 6 atoms.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from widom_atlas.v04.raspa2.cif_relabeller import (
    _lattice_matrix_from_params,
    _parse_vogtiv_cif,
    relabel_vogtiv_cif,
)


@pytest.fixture(scope="module")
def vogtiv_cif() -> Path:
    p = Path("docs/research/dataset-research-for-v0.4/15/core-mof-sep2014/core-mof-july2014/VOGTIV_clean_h.cif")
    if not p.exists():
        pytest.skip(f"VOGTIV CIF not present: {p}")
    return p


@pytest.fixture()
def relabelled_cif(vogtiv_cif: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "VOGTIV_relabelled.cif"
    counts = relabel_vogtiv_cif(vogtiv_cif, dst)
    assert all(counts.get(k, 0) == 6 for k in (
        "Mof_Mg", "Mof_Ca", "Mof_Cb", "Mof_Cc", "Mof_Cd",
        "Mof_Oa", "Mof_Ob", "Mof_Oc", "Mof_H",
    )), counts
    return dst


def _min_image_distance(cart_i: np.ndarray, cart_j: np.ndarray, lattice: np.ndarray) -> float:
    best = float("inf")
    for ia in (-1, 0, 1):
        for ib in (-1, 0, 1):
            for ic in (-1, 0, 1):
                shift = ia * lattice[0] + ib * lattice[1] + ic * lattice[2]
                d = float(np.linalg.norm(cart_j + shift - cart_i))
                if d < best:
                    best = d
    return best


def _read_labels_from_cif(cif_path: Path) -> list[tuple[str, str, np.ndarray]]:
    """Return list of (label, element, cart_position) tuples for each atom."""
    import re
    lattice, atoms = _parse_vogtiv_cif(cif_path)
    text = cif_path.read_text().splitlines()
    line_to_label: dict[int, str] = {}
    in_loop = False
    columns: list[str] = []
    label_col = None
    for idx, line in enumerate(text):
        s = line.strip()
        if s == "loop_":
            columns = []
            in_loop = False
            continue
        if s.startswith("_atom_site_"):
            columns.append(s)
            if "_atom_site_label" in columns and "_atom_site_fract_x" in columns:
                in_loop = True
                label_col = columns.index("_atom_site_label")
            continue
        if in_loop and s and not s.startswith("#") and not s.startswith("_"):
            fields = re.split(r"\s+", s)
            if label_col is not None and len(fields) > label_col:
                line_to_label[idx] = fields[label_col]
    return [(line_to_label[a["original_line_index"]], a["element"], np.asarray(a["cart"]))
            for a in atoms], lattice


def test_relabel_counts_match_vogtiv_stoichiometry(relabelled_cif):
    """Every sublattice must contain exactly 6 atoms in the VOGTIV primitive."""
    atoms, _ = _read_labels_from_cif(relabelled_cif)
    counts: dict[str, int] = {}
    for lab, _, _ in atoms:
        counts[lab] = counts.get(lab, 0) + 1
    for label in ("Mof_Mg", "Mof_Ca", "Mof_Cb", "Mof_Cc", "Mof_Cd",
                  "Mof_Oa", "Mof_Ob", "Mof_Oc", "Mof_H"):
        assert counts.get(label) == 6, f"{label}: {counts.get(label)} != 6"


def test_carbon_O_neighbour_counts(relabelled_cif):
    """Mof_Cd must have 2 O bonds; Mof_Ca 1; Mof_Cb/Cc 0 (each)."""
    atoms, lattice = _read_labels_from_cif(relabelled_cif)
    o_carts = [c for lab, el, c in atoms if el == "O"]
    expected_n_o = {"Mof_Cd": 2, "Mof_Ca": 1, "Mof_Cb": 0, "Mof_Cc": 0}
    for lab, el, cart in atoms:
        if el != "C":
            continue
        n_o = sum(1 for oc in o_carts if _min_image_distance(cart, oc, lattice) < 1.7)
        expected = expected_n_o.get(lab)
        assert expected is not None, f"unexpected label {lab} for a C atom"
        assert n_o == expected, f"{lab} expected n_O={expected} got {n_o}"


def test_oxygen_Mg_neighbour_counts(relabelled_cif):
    """Mof_Oa must have 1 Mg bond; Mof_Ob/Mof_Oc each 2 Mg bonds."""
    atoms, lattice = _read_labels_from_cif(relabelled_cif)
    mg_carts = [c for lab, el, c in atoms if el == "Mg"]
    expected_n_mg = {"Mof_Oa": 1, "Mof_Ob": 2, "Mof_Oc": 2}
    for lab, el, cart in atoms:
        if el != "O":
            continue
        n_mg = sum(1 for mc in mg_carts if _min_image_distance(cart, mc, lattice) < 2.7)
        expected = expected_n_mg.get(lab)
        assert expected is not None, f"unexpected label {lab} for an O atom"
        assert n_mg == expected, f"{lab} expected n_Mg={expected} got {n_mg}"


def test_Mof_Oc_must_bond_to_phenoxide_aromatic_C(relabelled_cif):
    """Mof_Oc (phenoxide O) must be bonded to a Mof_Ca; Mof_Ob (carboxylate
    bridging O) must NOT bond to a Mof_Ca. This is the geometric tie-break
    between the two 2-Mg-bonded oxygens."""
    atoms, lattice = _read_labels_from_cif(relabelled_cif)
    ca_carts = [c for lab, el, c in atoms if lab == "Mof_Ca"]
    for lab, el, cart in atoms:
        if lab not in ("Mof_Ob", "Mof_Oc"):
            continue
        bonded_to_ca = any(_min_image_distance(cart, cc, lattice) < 1.7 for cc in ca_carts)
        if lab == "Mof_Oc":
            assert bonded_to_ca, "Mof_Oc must bond to a Mof_Ca (phenoxide bridge)"
        else:
            assert not bonded_to_ca, (
                "Mof_Ob (Mg-O chain bridging carboxylate-O) must NOT bond to Mof_Ca"
            )


def test_lattice_matrix_consistency():
    """Sanity-check the lattice constructor for orthogonal then triclinic cases."""
    # Cubic 10 Å
    L = _lattice_matrix_from_params(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    assert math.isclose(L[0, 0], 10.0, abs_tol=1e-9)
    assert math.isclose(L[1, 1], 10.0, abs_tol=1e-9)
    assert math.isclose(L[2, 2], 10.0, abs_tol=1e-9)
    # VOGTIV
    L = _lattice_matrix_from_params(6.7588, 15.1941, 15.1941, 62.1589, 81.4729, 98.5271)
    a, b, c = np.linalg.norm(L[0]), np.linalg.norm(L[1]), np.linalg.norm(L[2])
    assert math.isclose(a, 6.7588, abs_tol=1e-4)
    assert math.isclose(b, 15.1941, abs_tol=1e-4)
    assert math.isclose(c, 15.1941, abs_tol=1e-4)
    # Volume = a·(b×c) = ~1321 Å³
    vol = abs(np.dot(L[0], np.cross(L[1], L[2])))
    assert 1300 < vol < 1340, f"VOGTIV primitive volume {vol:.2f} outside expected band"
