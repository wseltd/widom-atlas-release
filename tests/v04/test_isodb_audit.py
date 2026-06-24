"""Regression tests for ISODB CO2 isotherm audit (5c branches)."""
from __future__ import annotations

from pathlib import Path

import pytest

from widom_atlas.v04.refs.isodb_audit import (
    BAR_TO_PA,
    MMOL_PER_G_TO_MOL_PER_KG,
    HenryFit,
    audit_isodb_isotherm,
    render_audit_json,
)

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "docs/research/dataset-research-for-v0.4/5c_replacement_branches"


def test_unit_conversion_constants():
    """K_H unit conversion: mmol/g/bar -> mol/kg/bar (identity) -> mol/kg/Pa (1e-5 ratio)."""
    assert MMOL_PER_G_TO_MOL_PER_KG == 1.0
    assert BAR_TO_PA == 1.0e5

    # Round-trip: 1 mol/kg/bar -> 1e-5 mol/kg/Pa -> back
    K_mol_per_kg_per_bar = 125.07
    K_mol_per_kg_per_Pa = K_mol_per_kg_per_bar / BAR_TO_PA
    K_back = K_mol_per_kg_per_Pa * BAR_TO_PA
    assert abs(K_back - K_mol_per_kg_per_bar) < 1e-9


def test_audit_NaZK5_primary_5c_branch():
    """Na-ZK-5 primary 5c branch matches operator-reported K_H sensitivity range."""
    path = BASE / "na_zk5_pham_lobo_2013/pham_lobo_2013_isotherm49_na_zk5_co2_303K.json"
    assert path.exists()
    audit = audit_isodb_isotherm(path)
    assert audit.adsorbent == "Zeolite Na-ZK-5"
    assert audit.temperature_K == 303.0
    assert audit.n_points == 9
    method_kh = audit.K_H_method_values_mol_per_kg_per_bar()
    # 1pt slope at p=0.00103 bar should be ~125 mol/(kg.bar)
    assert 124.0 < method_kh["1pt_slope"] < 126.0
    # Sensitivity range: 1pt slope is highest, virial lowest (saturation)
    min_K, max_K = audit.K_H_sensitivity_range_mol_per_kg_per_bar()
    assert max_K / min_K > 3.0  # wide spread due to mild saturation


def test_audit_zeolite_5a_298K_is_clean_fallback():
    """5A + CO2 at 298 K should be a clean fallback (narrow sensitivity, adequate Henry)."""
    path = BASE / "zeolite_5a_wang_levan_2009/wang_levan_2009_isotherm8_5a_co2_298K.json"
    audit = audit_isodb_isotherm(path)
    assert audit.adsorbent == "Zeolite 5A"
    assert audit.temperature_K == 298.0
    assert audit.n_points >= 40
    min_K, max_K = audit.K_H_sensitivity_range_mol_per_kg_per_bar()
    # Narrow sensitivity range — within 15% across all 4 methods
    assert max_K / min_K < 1.15
    assert "adequate" in audit.henry_regime_adequacy


def test_audit_4a_273K_has_step_like_spread():
    """4A at 273 K has step-like uptake — wide sensitivity spread expected."""
    path = BASE / "zeolite_4a_hefti_2020/hefti_2020_isotherm35_4a_co2_273K.json"
    audit = audit_isodb_isotherm(path)
    min_K, max_K = audit.K_H_sensitivity_range_mol_per_kg_per_bar()
    assert max_K / min_K > 2.5


def test_render_audit_json_schema():
    """render_audit_json must produce a stable schema for downstream consumers."""
    path = BASE / "zeolite_13x_wang_levan_2009/wang_levan_2009_isotherm27_13x_co2_273K.json"
    audit = audit_isodb_isotherm(path)
    rendered = render_audit_json(audit)
    assert "source_file" in rendered
    assert "doi" in rendered
    assert "adsorbent" in rendered
    assert "temperature_K" in rendered
    assert "unit_conversion" in rendered
    assert rendered["unit_conversion"]["bar_to_Pa"] == BAR_TO_PA
    assert rendered["unit_conversion"]["mmol_per_g_to_mol_per_kg"] == MMOL_PER_G_TO_MOL_PER_KG
    assert "K_H_fits" in rendered
    assert "K_H_sensitivity" in rendered
    assert "henry_regime_adequacy" in rendered
    for fit in rendered["K_H_fits"]:
        assert "K_H_mol_per_kg_per_bar" in fit
        assert "K_H_mol_per_kg_per_Pa" in fit
        # Round-trip check
        assert abs(
            fit["K_H_mol_per_kg_per_Pa"] - fit["K_H_mol_per_kg_per_bar"] / BAR_TO_PA
        ) < 1e-12


def test_unit_conversion_per_data_point():
    """Every isotherm row's p_Pa = p_bar * 1e5, q_mol_per_kg = q_mmol_per_g."""
    path = BASE / "na_zk5_pham_lobo_2013/pham_lobo_2013_isotherm49_na_zk5_co2_303K.json"
    audit = audit_isodb_isotherm(path)
    rendered = render_audit_json(audit)
    for row in rendered["isotherm_data"]:
        assert abs(row["p_Pa"] - row["p_bar"] * 1e5) < 1e-12
        assert row["q_mol_per_kg"] == row["q_mmol_per_g"]


def test_invalid_units_raise():
    """Non-bar / non-mmol/g units raise ValueError."""
    import json
    import tempfile

    fake_data = {
        "DOI": "fake",
        "pressureUnits": "kPa",   # wrong
        "adsorptionUnits": "mmol/g",
        "adsorbent": {"name": "fake"},
        "adsorbates": [{"name": "CO2"}],
        "temperature": 298,
        "isotherm_data": [{"pressure": 0.001, "total_adsorption": 0.1}],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fp:
        json.dump(fake_data, fp)
        tmp_path = Path(fp.name)
    with pytest.raises(ValueError, match="unsupported pressure units"):
        audit_isodb_isotherm(tmp_path)


def test_HenryFit_dataclass_fields_present():
    fit = HenryFit(
        method="test",
        K_H_mmol_per_g_per_bar=1.0,
        K_H_mol_per_kg_per_bar=1.0,
        K_H_mol_per_kg_per_Pa=1e-5,
        n_points_used=3,
        p_max_used_bar=0.01,
    )
    assert fit.method == "test"
    assert fit.notes == []  # default empty list
