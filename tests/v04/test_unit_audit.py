"""Forensic unit-conversion tests.

These tests catch factor-of-10 / atm-bar / STP-vs-non-STP / units-cm3-vs-mol
mistakes when converting literature Henry coefficients into the canonical
mol·kg⁻¹·bar⁻¹ used by the verdict evaluator.

The 6c MFI+Ar audit (2026-05-14) found that the YAML literature reference
K_H = 2.003 mol/(kg·bar) was off by ~10× from a correctly-unit-converted
Talu-Myers second-virial B = 4.35 cm³(STP)/g/atm at 305.75 K. These tests
make that audit reproducible and prevent regression.
"""
from __future__ import annotations

import pytest

from widom_atlas.v04.units import (
    BAR_PER_ATM,
    MOL_PER_CM3_STP,
    V_M_STP_CM3_PER_MOL,
    KH_cm3STP_per_g_per_atm_to_mol_per_kg_per_bar,
    KH_mol_per_kg_per_atm_to_mol_per_kg_per_bar,
    energy_K_to_kjmol,
    energy_kjmol_to_K,
    vant_hoff_KH_correction,
)

# ----------------------------------------------------------------------------
# Sanity: STP molar volume + atm/bar conversion factors
# ----------------------------------------------------------------------------

def test_molar_volume_at_STP_matches_textbook() -> None:
    """V_m(STP) = RT/P at 273.15 K, 101325 Pa = 0.022414 m³/mol = 22413.96 cm³/mol."""
    assert pytest.approx(22413.96, abs=0.1) == V_M_STP_CM3_PER_MOL
    assert pytest.approx(4.4615e-5, rel=1e-4) == MOL_PER_CM3_STP


def test_bar_per_atm_constant() -> None:
    """1 atm = 1.01325 bar, exactly (definitional)."""
    assert pytest.approx(1.01325, abs=1e-6) == BAR_PER_ATM


# ----------------------------------------------------------------------------
# Forensic conversion: cm³(STP)/g/atm → mol/kg/bar
# ----------------------------------------------------------------------------

def test_convert_cm3STP_g_atm_to_mol_kg_bar_smoke() -> None:
    """B = 1 cm³(STP)/g/atm should give ~0.044 mol/(kg·bar).

    Walk-through:
      1 cm³(STP)/g/atm  ×  (1 mol / 22414 cm³(STP))   ≈ 4.46e-5 mol/g/atm
                       ×  1000 g/kg                    ≈ 0.0446 mol/kg/atm
                       ×  (1 atm / 1.01325 bar)        ≈ 0.0441 mol/kg/bar
    """
    result = KH_cm3STP_per_g_per_atm_to_mol_per_kg_per_bar(1.0)
    assert result == pytest.approx(0.04404, abs=1e-4)


def test_talu_myers_ar_conversion_matches_forensic_audit() -> None:
    """Talu-Myers 2001 Ar in silicalite: B = 4.35 cm³(STP)/g/atm at 305.75 K.

    Operator-supplied target (forensic audit): K_H(305.75 K) ≈ 0.19 mol/(kg·bar).
    With Q_st = 15.7 kJ/mol van't Hoff to 298.15 K: ≈ 0.22 mol/(kg·bar).
    """
    K_H_at_305 = KH_cm3STP_per_g_per_atm_to_mol_per_kg_per_bar(4.35)
    assert K_H_at_305 == pytest.approx(0.1915, abs=0.005), (
        f"4.35 cm³(STP)/g/atm should give ≈0.19 mol/(kg·bar), got {K_H_at_305}"
    )

    K_H_at_298 = vant_hoff_KH_correction(
        K_H_at_T1=K_H_at_305, T1_K=305.75, T2_K=298.15, Q_st_kJ_per_mol=15.7,
    )
    assert K_H_at_298 == pytest.approx(0.224, abs=0.01), (
        f"van't Hoff to 298.15 K should give ≈0.22 mol/(kg·bar), got {K_H_at_298}"
    )


def test_pre_erratum_value_2003_would_be_factor_of_9_off() -> None:
    """Historical sentinel: the v04.2 YAML originally stored K_H = 2.003 mol/(kg·bar)
    for 6c. The forensic audit (V04_UNIT_AUDIT.md, 2026-05-14) showed this was off
    by ~9× from the correctly-reconstructed 0.224 mol/(kg·bar). This test pins the
    historical ratio so the pre-erratum value cannot return undetected — see
    `test_6c_YAML_K_H_reference_matches_forensic_reconstruction` for the live YAML
    check post-erratum."""
    pre_erratum_yaml_value = 2.003
    reconstructed = vant_hoff_KH_correction(
        K_H_at_T1=KH_cm3STP_per_g_per_atm_to_mol_per_kg_per_bar(4.35),
        T1_K=305.75, T2_K=298.15, Q_st_kJ_per_mol=15.7,
    )
    ratio = pre_erratum_yaml_value / reconstructed
    assert 8.0 < ratio < 11.0, (
        f"Pre-erratum YAML value {pre_erratum_yaml_value} should be ~9-10× the "
        f"reconstructed {reconstructed:.4f}; got ratio={ratio:.2f}"
    )


# ----------------------------------------------------------------------------
# Generic catch-all: factor-of-10 / atm-bar / STP mistakes
# ----------------------------------------------------------------------------

def test_atm_vs_bar_conversion_is_not_a_noop() -> None:
    """Converting from atm to bar must NOT be identity. K_H expressed per atm
    must be multiplied by ~1.013 to become per-bar (atm is larger than bar)."""
    K_H_atm = 1.0  # mol/kg/atm
    K_H_bar = KH_mol_per_kg_per_atm_to_mol_per_kg_per_bar(K_H_atm)
    assert K_H_bar != pytest.approx(K_H_atm, rel=1e-6), \
        "atm→bar conversion should not be the identity"
    assert K_H_bar == pytest.approx(1.0 / 1.01325, abs=1e-5)


def test_STP_step_is_not_omitted() -> None:
    """Converting cm³(STP)/g/atm to mol/kg/atm MUST multiply by (1/V_m(STP)) × 1000.
    A bare conversion that forgets the STP step gives a number ~22414× too large.
    """
    # If the STP step is OMITTED, the implementer would just do:
    #   B_wrong = 4.35 cm³/g/atm  ×  1000 g/kg  =  4350 cm³/kg/atm  (NOT a Henry slope)
    # The correct mol/kg/atm number is:
    correct_mol_kg_atm = 4.35 * MOL_PER_CM3_STP * 1000.0
    assert correct_mol_kg_atm == pytest.approx(0.1941, abs=1e-4)
    # That same number expressed as a fraction of an "STP-omitted" wrong value
    # would be off by exactly 22414×.
    wrong_with_omitted_STP_factor = 4.35 * 1000.0
    assert wrong_with_omitted_STP_factor / correct_mol_kg_atm == pytest.approx(22414, rel=1e-3)


def test_vant_hoff_round_trip() -> None:
    """K_H_low at T_low → K_H_high at T_high → back to K_H_low at T_low."""
    K0 = 0.5
    T0 = 305.0
    T1 = 280.0
    Qst = 16.0
    K1 = vant_hoff_KH_correction(K_H_at_T1=K0, T1_K=T0, T2_K=T1, Q_st_kJ_per_mol=Qst)
    K0_back = vant_hoff_KH_correction(K_H_at_T1=K1, T1_K=T1, T2_K=T0, Q_st_kJ_per_mol=Qst)
    assert K0_back == pytest.approx(K0, rel=1e-9)
    # Q_st > 0 and T1 < T0 => K1 > K0
    assert K1 > K0


def test_energy_round_trip() -> None:
    """kJ/mol ↔ K round-trip."""
    Q = 20.9  # CH4 Q_st in MFI
    Q_K = energy_kjmol_to_K(Q)
    Q_back = energy_K_to_kjmol(Q_K)
    assert Q_back == pytest.approx(Q, rel=1e-12)


# ----------------------------------------------------------------------------
# Sanity tests on the locked YAML's literature K_H references
# ----------------------------------------------------------------------------

def test_6c_YAML_K_H_reference_matches_forensic_reconstruction() -> None:
    """The YAML stored 6c reference K_H MUST agree with the correctly-converted
    reconstruction from the raw Talu-Myers Table 3 B = 4.35 cm^3(STP)/g/atm,
    after van't Hoff to 298.15 K with Q_st = 15.7 kJ/mol.

    Was xfail-marked prior to the 2026-05-14 unit-erratum (when the YAML stored
    2.003 mol/(kg·bar), off by ~9× from the reconstruction). Now a passing
    regression test — if anyone re-introduces the off-by-10 error in the YAML,
    this test will fail loudly.
    """
    from pathlib import Path

    import yaml as _yaml

    repo_root = Path(__file__).resolve().parent.parent.parent
    matrix = _yaml.safe_load((repo_root / "v04_case_matrix.yaml").read_text())
    yaml_6c_K_H = None
    for case in matrix.get("cases", []):
        for branch in case.get("branches", []):
            if branch.get("branch_id") == "6c":
                yaml_6c_K_H = branch["references"]["K_H"]["value"]
                break
    assert yaml_6c_K_H is not None, "Could not locate 6c K_H reference in YAML"

    K_H_reconstructed = vant_hoff_KH_correction(
        K_H_at_T1=KH_cm3STP_per_g_per_atm_to_mol_per_kg_per_bar(4.35),
        T1_K=305.75, T2_K=298.15, Q_st_kJ_per_mol=15.7,
    )
    ratio = yaml_6c_K_H / K_H_reconstructed
    assert 0.80 < ratio < 1.25, (
        f"6c YAML K_H = {yaml_6c_K_H} mol/(kg·bar) disagrees with the forensic "
        f"reconstruction {K_H_reconstructed:.4f} mol/(kg·bar) by ratio {ratio:.3f}. "
        f"Allowed band is 0.80–1.25 (i.e. within 25%). "
        f"If you are re-introducing the pre-2026-05-14 value of 2.003, STOP — "
        f"see V04_UNIT_AUDIT.md and V04_LOCKED_SPEC_CHANGELOG.md (erratum 2026-05-14)."
    )


def test_6c_YAML_K_H_acceptance_window_is_strict_pm_0p10_logK() -> None:
    """The 6c K_H acceptance window must be strict ±0.10 log10 around the value."""
    from pathlib import Path

    import yaml as _yaml

    repo_root = Path(__file__).resolve().parent.parent.parent
    matrix = _yaml.safe_load((repo_root / "v04_case_matrix.yaml").read_text())
    for case in matrix.get("cases", []):
        for branch in case.get("branches", []):
            if branch.get("branch_id") == "6c":
                kh = branch["references"]["K_H"]
                value = kh["value"]
                wmin = kh["acceptance_window_min"]
                wmax = kh["acceptance_window_max"]
                # Strict ±0.10 log10 → window = value / 10^0.10 .. value × 10^0.10
                expected_min = value / 10**0.10
                expected_max = value * 10**0.10
                assert wmin == pytest.approx(expected_min, abs=0.005), (
                    f"6c acceptance_window_min={wmin} expected ≈ {expected_min:.3f}"
                )
                assert wmax == pytest.approx(expected_max, abs=0.005), (
                    f"6c acceptance_window_max={wmax} expected ≈ {expected_max:.3f}"
                )
                return
    raise AssertionError("branch 6c not found in YAML")


def test_6c_YAML_erratum_block_present() -> None:
    """The 6c K_H reference must carry a structured `erratum` block documenting
    the 2026-05-14 unit correction. This makes the correction self-describing
    in the YAML and detectable by downstream consumers."""
    from pathlib import Path

    import yaml as _yaml

    repo_root = Path(__file__).resolve().parent.parent.parent
    matrix = _yaml.safe_load((repo_root / "v04_case_matrix.yaml").read_text())
    for case in matrix.get("cases", []):
        for branch in case.get("branches", []):
            if branch.get("branch_id") == "6c":
                erratum = branch["references"]["K_H"].get("erratum")
                assert erratum is not None, "6c K_H erratum block missing"
                assert erratum["corrected_from"] == 2.003
                assert erratum["corrected_to"] == pytest.approx(0.224, abs=0.001)
                assert erratum["correction_kind"] == "literature_reference_unit_correction"
                assert "V04_UNIT_AUDIT.md" in erratum.get("correction_evidence", "")
                return
    raise AssertionError("branch 6c not found in YAML")
