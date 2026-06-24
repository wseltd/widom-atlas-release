"""Unit tests for the native site-truth extractor."""
from __future__ import annotations

import math
import numpy as np

from widom_atlas.v04.native.runner import StrongestInsertion
from widom_atlas.v04.native.site_truth import (
    _TARGET_KEY_SPECS,
    _closest_pair_distance,
    extract_site_truth_verdict,
)


def _make_strongest(probe_types, probe_carts, framework_types, framework_carts,
                    cell=None):
    if cell is None:
        cell = np.eye(3) * 30.0
    return StrongestInsertion(
        U_K=-5000.0,
        probe_types=list(probe_types),
        probe_cartesian_angstrom=np.array(probe_carts, dtype=float),
        framework_types=list(framework_types),
        framework_cartesian_angstrom=np.array(framework_carts, dtype=float),
        supercell_matrix_angstrom=cell,
        temperature_K=298.0,
        seed=42,
    )


def test_closest_pair_distance_finds_correct_atom():
    """Two Mg atoms at known positions; the strongest insertion places O_co2
    at (0, 0, 2.27) — the closest Mg-O(CO2) pair is the one at the origin."""
    strongest = _make_strongest(
        probe_types=["O_co2", "C_co2", "O_co2"],
        probe_carts=[[0, 0, 2.27], [0, 0, 3.41], [0, 0, 4.55]],
        framework_types=["Mof_Mg", "Mof_Mg"],
        framework_carts=[[0, 0, 0], [10, 0, 0]],
    )
    pi, fi, d = _closest_pair_distance(strongest, "O_co2", "Mof_Mg", 0)
    assert pi == 0  # first O_co2 of the CO2 probe
    assert fi == 0  # Mg at origin
    assert math.isclose(d, 2.27, abs_tol=1e-6)


def test_extract_site_truth_for_mg_mof74_passing_geometry():
    """O_co2 at 2.27 Å from Mg should PASS with tolerance 0.10."""
    strongest = _make_strongest(
        probe_types=["O_co2", "C_co2", "O_co2"],
        probe_carts=[[0, 0, 2.27], [0, 0, 3.42], [0, 0, 4.57]],
        framework_types=["Mof_Mg"],
        framework_carts=[[0, 0, 0]],
    )
    site_truth_block = {
        "enabled": True,
        "tolerance_angstrom": 0.10,
        "target_geometry": {"Mg_O_CO2_distance_angstrom": 2.27},
        "reference": {"primary": "Wu_2010_JPCL_SI_Table_S1", "primary_doi": "10.1021/jz100558r"},
    }
    out = extract_site_truth_verdict(strongest, site_truth_block)
    assert out["passes_site_truth"] is True
    assert out["distances"][0]["label"] == "Mg_O_CO2_distance_angstrom"
    assert math.isclose(out["distances"][0]["atlas_angstrom"], 2.27, abs_tol=1e-6)
    assert out["distances"][0]["passes"] is True


def test_extract_site_truth_for_mg_mof74_failing_geometry():
    """O_co2 at 2.65 Å from Mg (the +0.38 Å drift we found for 1a) should FAIL."""
    strongest = _make_strongest(
        probe_types=["O_co2", "C_co2", "O_co2"],
        probe_carts=[[0, 0, 2.65], [0, 0, 3.80], [0, 0, 4.95]],
        framework_types=["Mof_Mg"],
        framework_carts=[[0, 0, 0]],
    )
    site_truth_block = {
        "enabled": True,
        "tolerance_angstrom": 0.10,
        "target_geometry": {"Mg_O_CO2_distance_angstrom": 2.27},
        "reference": {"primary": "Wu_2010_JPCL_SI_Table_S1", "primary_doi": "10.1021/jz100558r"},
    }
    out = extract_site_truth_verdict(strongest, site_truth_block)
    assert out["passes_site_truth"] is False
    d = out["distances"][0]
    assert math.isclose(d["atlas_angstrom"], 2.65, abs_tol=1e-6)
    assert math.isclose(d["delta_angstrom"], 0.38, abs_tol=1e-6)
    assert d["passes"] is False


def test_extract_site_truth_disabled_returns_skipped():
    out = extract_site_truth_verdict(
        strongest=_make_strongest(["X"], [[0, 0, 0]], ["Y"], [[1, 1, 1]]),
        branch_site_truth_block={"enabled": False},
    )
    assert out["passes_site_truth"] is None
    assert out["skipped"] is True


def test_extract_site_truth_no_strongest_returns_skipped():
    out = extract_site_truth_verdict(
        strongest=None,
        branch_site_truth_block={
            "enabled": True,
            "tolerance_angstrom": 0.10,
            "target_geometry": {"Mg_O_CO2_distance_angstrom": 2.27},
        },
    )
    assert out["passes_site_truth"] is None
    assert out["skipped"] is True


def test_extract_site_truth_for_na_rho_two_distances():
    """5b Na-Rho has two distinct Na-OC distances per the Lozinska site-truth.
    Verifies rank=0 → nearest Na-O(CO2); rank=1 → second-nearest.
    Geometry: two Na atoms 6.0 Å apart along x; CO2 bridging between them at
    z=2.7 Å, positioned so one O is closer to Na@x=0 (d≈2.85) and the other
    O is farther from both Na atoms.
    """
    # Break the Na-Na symmetry so the rank=0 and rank=1 distances differ:
    # O_co2[0] at (1.0, 0, 2.65); O_co2[2] at (4.5, 0, 2.65).
    # Na@(0,0,0) is 2.832 from O_co2[0] and 5.224 from O_co2[2].
    # Na@(6,0,0) is 5.659 from O_co2[0] and 2.978 from O_co2[2].
    # rank=0 → 2.832 (O_co2[0] → Na@0); rank=1 → 2.978 (O_co2[2] → Na@6).
    strongest = _make_strongest(
        probe_types=["O_co2", "C_co2", "O_co2"],
        probe_carts=[[1.0, 0, 2.65], [2.75, 0, 2.65], [4.5, 0, 2.65]],
        framework_types=["Na", "Na"],
        framework_carts=[[0, 0, 0], [6.0, 0, 0]],
    )
    site_truth_block = {
        "enabled": True,
        "tolerance_angstrom": 0.40,
        "target_geometry": {
            "Na_OC1_distance_angstrom": 2.88,
            "Na_OC3_distance_angstrom": 2.58,
        },
        "reference": {"primary": "Lozinska_2012", "primary_doi": "10.1021/ja300034j"},
    }
    out = extract_site_truth_verdict(strongest, site_truth_block)
    labels = {d["label"]: d for d in out["distances"]}
    assert "Na_OC1_distance_angstrom" in labels
    assert "Na_OC3_distance_angstrom" in labels
    rank0 = labels["Na_OC1_distance_angstrom"]["atlas_angstrom"]
    rank1 = labels["Na_OC3_distance_angstrom"]["atlas_angstrom"]
    assert math.isclose(rank0, 2.832, abs_tol=1e-2), f"rank0={rank0}"
    assert math.isclose(rank1, 3.045, abs_tol=1e-2), f"rank1={rank1}"
    assert rank1 > rank0
