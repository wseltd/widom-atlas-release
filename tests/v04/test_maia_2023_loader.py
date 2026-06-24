"""Regression tests for Maia 2023 UiO-66 loader (3b execution)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from widom_atlas.v04.native.maia_2023_loader import (
    MAIA_2023_UA_LJ,
    TRAPPE_CO2_BOND_LENGTH_A,
    TRAPPE_CO2_CHARGES_E,
    TRAPPE_CO2_SELF_LJ,
    MAIA_2023_EHq_CHARGES_E,
    MAIA_2023_EHq_LJ,
    MAIA_2023_UAq_CHARGES_E,
    _classify_atom_by_ddec_charge,
    _parse_cif_with_charges,
    _verify_classification_counts,
    load_3b_native_maia_2023,
)

REPO = Path(__file__).resolve().parents[2]
RUBTAK01 = REPO / "fixtures" / "v04" / "RUBTAK01_SL_DDEC.cif"


def test_maia_table_2_verbatim_LJ():
    """Table 2 of Maia 2023 (UA / UAq) verbatim — guards against typos."""
    expected = {
        "Maia_Zr":  (34.72, 2.78),
        "Maia_O1":  (55.00, 2.80),
        "Maia_O25": (93.00, 3.02),
        "Maia_O29": (55.00, 2.80),
        "Maia_C1":  (41.00, 3.90),
        "Maia_C13": (21.00, 3.88),
        "Maia_C25": (48.00, 3.74),
        "Maia_H25": (0.00,  0.00),
    }
    assert expected == MAIA_2023_UA_LJ


def test_maia_table_3_EHq_LJ_differences_from_UAq():
    """Table 3 (EHq) keeps Zr/C1/C13/O1/O25/O29 same as UAq, reparam C25 + new H1."""
    assert MAIA_2023_EHq_LJ["Maia_Zr"] == MAIA_2023_UA_LJ["Maia_Zr"]
    assert MAIA_2023_EHq_LJ["Maia_C1"] == MAIA_2023_UA_LJ["Maia_C1"]
    assert MAIA_2023_EHq_LJ["Maia_C13"] == MAIA_2023_UA_LJ["Maia_C13"]
    assert MAIA_2023_EHq_LJ["Maia_O1"] == MAIA_2023_UA_LJ["Maia_O1"]
    assert MAIA_2023_EHq_LJ["Maia_O25"] == MAIA_2023_UA_LJ["Maia_O25"]
    assert MAIA_2023_EHq_LJ["Maia_O29"] == MAIA_2023_UA_LJ["Maia_O29"]
    # C25 EHq is smaller (H is explicit)
    assert MAIA_2023_EHq_LJ["Maia_C25"] == (30.70, 3.60)
    # New H1 in EHq
    assert MAIA_2023_EHq_LJ["Maia_H1"] == (25.45, 2.36)


def test_maia_UAq_charges_electroneutral_per_zr6_cluster():
    """One Zr6 cluster + 6 BDC ligands per primitive cell sums to electroneutral."""
    per_cluster = {
        "Maia_Zr":  6,
        "Maia_O25": 4,
        "Maia_O29": 4,
        "Maia_O1":  24,
        "Maia_C1":  12,
        "Maia_C13": 12,
        "Maia_C25": 24,
        "Maia_H25": 4,
    }
    total = sum(
        MAIA_2023_UAq_CHARGES_E[label] * count
        for label, count in per_cluster.items()
    )
    assert abs(total) < 1e-9, f"UAq charges not electroneutral: sum = {total}"


def test_maia_EHq_charges_electroneutral_per_zr6_cluster():
    """Same per_cluster counts but with H1 aromatic H accounted for."""
    per_cluster = {
        "Maia_Zr":  6,
        "Maia_O25": 4,
        "Maia_O29": 4,
        "Maia_O1":  24,
        "Maia_C1":  12,
        "Maia_C13": 12,
        "Maia_C25": 24,
        "Maia_H25": 4,
        "Maia_H1":  24,
    }
    total = sum(
        MAIA_2023_EHq_CHARGES_E[label] * count
        for label, count in per_cluster.items()
    )
    assert abs(total) < 1e-9, f"EHq charges not electroneutral: sum = {total}"


def test_trappe_co2_constants_match_maia_table_1():
    assert TRAPPE_CO2_SELF_LJ["C_co2"] == (27.0, 2.80)
    assert TRAPPE_CO2_SELF_LJ["O_co2"] == (79.0, 3.05)
    assert TRAPPE_CO2_CHARGES_E["C_co2"] == +0.70
    assert TRAPPE_CO2_CHARGES_E["O_co2"] == -0.35
    assert TRAPPE_CO2_BOND_LENGTH_A == 1.16


def test_charge_bucket_classifier_O_boundaries():
    assert _classify_atom_by_ddec_charge("O", -1.196) == "Maia_O25"
    assert _classify_atom_by_ddec_charge("O", -1.057) == "Maia_O29"
    assert _classify_atom_by_ddec_charge("O", -0.593) == "Maia_O1"
    assert _classify_atom_by_ddec_charge("O", -1.10) == "Maia_O25"
    assert _classify_atom_by_ddec_charge("O", -0.80) == "Maia_O29"


def test_charge_bucket_classifier_C_boundaries():
    assert _classify_atom_by_ddec_charge("C", +0.694) == "Maia_C1"
    assert _classify_atom_by_ddec_charge("C", -0.032) == "Maia_C13"
    assert _classify_atom_by_ddec_charge("C", -0.097) == "Maia_C25"


def test_charge_bucket_classifier_H_boundaries():
    assert _classify_atom_by_ddec_charge("H", +0.504) == "Maia_H25"
    assert _classify_atom_by_ddec_charge("H", +0.108) == "Maia_H_aromatic"


def test_classification_counts_RUBTAK01_verified():
    """RUBTAK01_SL_DDEC.cif must classify into the expected Zr6O4(OH)4(BDC)6 per-cluster counts."""
    _, elements, _, ddec = _parse_cif_with_charges(RUBTAK01)
    classifications = [
        _classify_atom_by_ddec_charge(el, q)
        for el, q in zip(elements, ddec, strict=True)
    ]
    counts: dict[str, int] = {}
    for t in classifications:
        counts[t] = counts.get(t, 0) + 1
    expected = {
        "Maia_Zr": 6,
        "Maia_O25": 4,
        "Maia_O29": 4,
        "Maia_O1": 24,
        "Maia_C1": 12,
        "Maia_C13": 12,
        "Maia_C25": 24,
        "Maia_H25": 4,
        "Maia_H_aromatic": 24,
    }
    assert counts == expected


def test_load_UA_drops_aromatic_H_and_zeros_charges():
    sys = load_3b_native_maia_2023(REPO, cif_path=RUBTAK01, variant="UA")
    assert sys.n_framework_atoms == 90  # 114 - 24 aromatic H
    assert sys.framework_charges_e is not None
    assert float(np.abs(sys.framework_charges_e).sum()) == 0.0
    # Probe charges zeroed in UA (see loader docstring)
    assert float(np.abs(sys.probe.charges_e).sum()) == 0.0


def test_load_UAq_keeps_90_atoms_and_charges_neutral():
    sys = load_3b_native_maia_2023(REPO, cif_path=RUBTAK01, variant="UAq")
    assert sys.n_framework_atoms == 90  # aromatic H still dropped (united atom)
    assert sys.framework_charges_e is not None
    assert abs(float(sys.framework_charges_e.sum())) < 1e-9
    # Probe charges retain TraPPE-CO2 values
    assert sys.probe.charges_e[1] == +0.70  # central C
    assert sys.probe.charges_e[0] == -0.35  # O


def test_load_EHq_keeps_all_114_atoms_and_charges_neutral():
    sys = load_3b_native_maia_2023(REPO, cif_path=RUBTAK01, variant="EHq")
    assert sys.n_framework_atoms == 114  # all atoms
    assert sys.framework_charges_e is not None
    assert abs(float(sys.framework_charges_e.sum())) < 1e-9


def test_classifier_counts_fail_message_includes_each_failing_label():
    """If per-cluster counts don't match, classifier raises with specific labels."""
    bad_counts = {
        "Maia_Zr": 6,
        "Maia_O25": 100,  # wrong
        "Maia_O29": 4,
        "Maia_O1": 24,
        "Maia_C1": 12,
        "Maia_C13": 12,
        "Maia_C25": 24,
        "Maia_H25": 4,
        "Maia_H_aromatic": 24,
    }
    status = _verify_classification_counts(bad_counts)
    assert status["Maia_O25"].startswith("FAIL:")
    assert status["Maia_Zr"] == "OK"


def test_invalid_variant_raises():
    with pytest.raises(ValueError, match="variant must be"):
        load_3b_native_maia_2023(REPO, cif_path=RUBTAK01, variant="NOT_A_REAL_VARIANT")
