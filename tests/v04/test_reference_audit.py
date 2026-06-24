"""Reference-audit regression tests (V04_REFERENCE_AUDIT.md, 2026-05-14).

Each test re-extracts a literature K_H from the primary data we have on disk
and asserts the YAML reference value matches within tolerance. Designed to
catch the same class of unit/decimal errors that the 6c audit (V04_UNIT_AUDIT.md)
turned up.

Branches covered:
 - 6a MFI + CH4 (Hufton 1993 local isotherm JSON)
 - 4a Si-CHA + CO2 (Maghsoudi 2013 Toth parameters in YAML)
 - 3a UiO-66 + CO2 (Cmarik 2012 SI Table S6 — direct read this session)
 - 2a HKUST-1 + CO2 (Simmons / Carne-Sanchez / Asadi NIST isodb JSONs)
 - 6b MFI + Kr — REFERENCE_BLOCKED sentinel
 - 5b Na-Rho + CO2 — REFERENCE_BLOCKED sentinel (loading-peak vs zero-coverage)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml as _yaml
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _matrix() -> dict:
    return _yaml.safe_load((REPO_ROOT / "v04_case_matrix.yaml").read_text())


def _branch(branch_id: str) -> dict:
    for case in _matrix().get("cases", []):
        for b in case.get("branches", []):
            if b.get("branch_id") == branch_id:
                return b
    raise KeyError(branch_id)


# ----------------------------------------------------------------------------
# 6a — Hufton 1993 MFI + CH4
# ----------------------------------------------------------------------------

def test_6a_Hufton1993_K_H_from_local_fixture() -> None:
    """Re-extract K_H from the local Hufton 1993 silicalite+CH4 isotherm and
    assert the YAML reference value 0.89 mol/(kg·bar) agrees within 5%."""
    fixture = REPO_ROOT / "fixtures/v04/hufton_1993/Isotherm3.json"
    d = json.loads(fixture.read_text())
    assert d["temperature"] == 298
    assert d["adsorbent"]["name"] == "Silicalite MFI"
    assert d["adsorbates"][0]["name"] == "Methane"
    assert d["adsorptionUnits"] == "mmol/g"
    assert d["pressureUnits"] == "bar"

    PS = np.array([p["pressure"] for p in d["isotherm_data"]])
    NS = np.array([p["total_adsorption"] for p in d["isotherm_data"]])

    # Henry+virial, force-zero, p < 0.6 bar
    mask = PS < 0.6
    A = np.column_stack([PS[mask], PS[mask] ** 2])
    sol, *_ = np.linalg.lstsq(A, NS[mask], rcond=None)
    K_H_henry_virial = float(sol[0])

    # Langmuir q_sat·K on all points
    def lang(p, qs, k):
        return qs * k * p / (1 + k * p)
    popt, _ = curve_fit(lang, PS, NS, p0=[1.5, 1.0])
    K_H_langmuir = float(popt[0] * popt[1])

    yaml_K_H = _branch("6a")["references"]["K_H"]["value"]
    assert yaml_K_H == pytest.approx(0.89, abs=0.005)

    # Both reconstructions should be within 10% of YAML
    assert K_H_henry_virial == pytest.approx(yaml_K_H, rel=0.10), (
        f"Henry+virial K_H={K_H_henry_virial} vs YAML {yaml_K_H}"
    )
    assert K_H_langmuir == pytest.approx(yaml_K_H, rel=0.10), (
        f"Langmuir K_H={K_H_langmuir} vs YAML {yaml_K_H}"
    )


# ----------------------------------------------------------------------------
# 4a — Maghsoudi 2013 Si-CHA + CO2 (Toth)
# ----------------------------------------------------------------------------

def test_4a_Maghsoudi_Toth_K_H_from_YAML_parameters() -> None:
    """The Toth low-loading limit K_H = q_sat × b (with b in 1/kPa → ×100 to
    1/bar) must reproduce the YAML K_H within 1%. Cross-check temperatures
    must also self-consistently reproduce Q_st via van't Hoff."""
    branch = _branch("4a")
    kh_block = branch["references"]["K_H"]
    yaml_K_H = kh_block["value"]
    assert yaml_K_H == pytest.approx(2.432, abs=0.005)

    # Derivation: "q_sat (4.824) x b (5.04e-3 1/kPa) x 100 (kPa/bar)"
    # YAML cross_check_temperatures carries 323 K and 348 K Toth params
    qsat_298, b_298 = 4.824, 5.04e-3
    KH_298 = qsat_298 * b_298 * 100
    assert pytest.approx(yaml_K_H, rel=0.005) == KH_298

    crosses = kh_block["cross_check_temperatures"]
    # Find 348 K cross-check
    cross_348 = next(c for c in crosses if c["T_K"] == 348.0)
    KH_348 = cross_348["q_sat"] * cross_348["b_inv_kPa"] * 100
    assert pytest.approx(cross_348["K_H"], rel=0.005) == KH_348

    # Van't Hoff 298 ↔ 348 → Q_st ≈ 23.5 kJ/mol (matches YAML cross-check note)
    Q_vH = (
        math.log(KH_298 / KH_348)
        / (1 / 298.0 - 1 / 348.0)
        * 8.314462618 / 1000.0
    )
    assert Q_vH == pytest.approx(23.5, abs=0.5), (
        f"Maghsoudi Toth van't Hoff Q_st={Q_vH} vs expected 23.5 kJ/mol"
    )


# ----------------------------------------------------------------------------
# 3a — Cmarik 2012 UiO-66 + CO2 (Toth, b directly in 1/bar)
# ----------------------------------------------------------------------------

def test_3a_Cmarik2012_Toth_K_H_consistency() -> None:
    """Per direct read of Cmarik 2012 SI Table S6: P column is 'Bar', so the
    Toth b values (0.284, 0.200, 0.158 at 298/308/318 K) are in 1/bar. K_H =
    q_sat × b directly (no factor 100).

    This test guards against a future drift where someone re-uses the 4a
    pattern (Maghsoudi b in 1/kPa) and accidentally multiplies the 3a
    Cmarik b by 100.
    """
    branch = _branch("3a")
    yaml_K_H = branch["references"]["K_H"]["value"]
    assert yaml_K_H == pytest.approx(5.14, abs=0.01)

    # Cmarik 2012 SI Table S6 at 298 K: q_sat = 18.080 mol/kg, b = 0.284 [1/bar]
    K_H_reconstructed = 18.080 * 0.284
    assert K_H_reconstructed == pytest.approx(yaml_K_H, rel=0.005), (
        f"Cmarik Toth q_sat*b={K_H_reconstructed} vs YAML 3a K_H {yaml_K_H}"
    )

    # If a regression accidentally treats b as 1/kPa and applies ×100, the
    # result would be ~514 mol/(kg·bar) — physically absurd. Guard:
    if_misconverted = 18.080 * 0.284 * 100
    assert not (4.0 < if_misconverted < 7.0), (
        "Sentinel: 3a Toth b should be in 1/bar, not 1/kPa. "
        "If this fails, someone may have re-applied the 4a kPa→bar pattern by mistake."
    )

    # Cross-check Q_st from Cmarik's 298/308/318 K Toth fits
    KH_298 = 18.080 * 0.284
    KH_318 = 16.576 * 0.158
    Q_vH = (
        math.log(KH_298 / KH_318)
        / (1 / 298.0 - 1 / 318.0)
        * 8.314462618 / 1000.0
    )
    yaml_Q = branch["references"]["Q_st"]["value"]
    assert Q_vH == pytest.approx(yaml_Q, abs=0.5)


# ----------------------------------------------------------------------------
# 2a — HKUST-1 triangulation
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,path,T_K,loading_unit,expect_in_window",
    [
        ("Simmons_2011", "docs/research/dataset-research-for-v0.4/16/10.1039C0ee00700e.Isotherm16.json",
         300.0, "mmol/g", True),
        ("Carne_Sanchez_2014", "docs/research/dataset-research-for-v0.4/16/10.1002adma.201403827.Isotherm5.json",
         295.0, "cm3(STP)/g", True),
    ],
)
def test_2a_HKUST1_triangulation_within_window(
    label: str, path: str, T_K: float, loading_unit: str, expect_in_window: bool,
) -> None:
    """The two NIST-isodb primary isotherms (Simmons 2011 + Carné-Sánchez 2014)
    must produce Langmuir-fit K_H values inside the YAML acceptance window
    [5.5, 9.0] mol/(kg·bar)."""
    fixture = REPO_ROOT / path
    if not fixture.exists():
        pytest.skip(f"primary isotherm not in local data: {path}")
    d = json.loads(fixture.read_text())
    assert d["temperature"] == pytest.approx(T_K, abs=1.0)
    assert d["adsorptionUnits"] == loading_unit
    PS = np.array([p["pressure"] for p in d["isotherm_data"]])
    NS = np.array([p["total_adsorption"] for p in d["isotherm_data"]])
    if loading_unit == "cm3(STP)/g":
        NS_mol_kg = NS / 22413.96 * 1000.0
    elif loading_unit == "mmol/g":
        NS_mol_kg = NS.copy()
    else:
        pytest.fail(f"unhandled loading unit {loading_unit}")

    def lang(p, qs, k):
        return qs * k * p / (1 + k * p)
    popt, _ = curve_fit(lang, PS, NS_mol_kg, p0=[max(NS_mol_kg) * 1.2, 1.0], maxfev=5000)
    K_H = float(popt[0] * popt[1])

    branch = _branch("2a")
    wmin = branch["references"]["K_H"]["acceptance_window_min"]
    wmax = branch["references"]["K_H"]["acceptance_window_max"]
    assert wmin == 5.5 and wmax == 9.0
    if expect_in_window:
        assert wmin <= K_H <= wmax, (
            f"2a {label} reconstructed K_H={K_H:.3f} outside YAML window [{wmin}, {wmax}]"
        )


# ----------------------------------------------------------------------------
# 6b — REFERENCE_BLOCKED sentinel
# ----------------------------------------------------------------------------

def test_6b_reference_blocked_pending_primary_data() -> None:
    """Sentinel: 6b cannot be audited until the Talu-Myers 2001 PDF is bundled
    locally. This test passes (with a recorded skip-reason) until that file
    appears, at which point it must be replaced with a real reconstruction
    test like 6a / 3a / 2a above.
    """
    expected_paths = [
        REPO_ROOT / "docs/research/dataset-research-for-v0.4/talu_myers_2001.pdf",
        REPO_ROOT / "docs/research/dataset-research-for-v0.4/9/talu_myers_2001.pdf",
        REPO_ROOT / "docs/research/dataset-research-for-v0.4/5/talu_myers_2001.pdf",
        REPO_ROOT / "fixtures/v04/talu_myers_2001.pdf",
        REPO_ROOT / "fixtures/v04/golden_sircar_1994.pdf",
    ]
    have_primary = any(p.exists() for p in expected_paths)
    if not have_primary:
        pytest.skip(
            "6b is REFERENCE_BLOCKED: neither Talu-Myers 2001 nor Golden-Sircar 1994 "
            "PDF is in local data. See V04_REFERENCE_AUDIT.md §2. Resolve by "
            "supplying one of those PDFs and replacing this test with a real "
            "reconstruction (mirror of test_6a_Hufton1993_K_H_from_local_fixture)."
        )
    raise AssertionError(
        "Primary 6b reference data appeared on disk — replace this sentinel "
        "with a real K_H reconstruction test now."
    )


# ----------------------------------------------------------------------------
# 6b R2 erratum (2026-05-17) — K_H unit-corrected from operator-supplied
# Talu Table 4 verbatim; Q_st reclassified as fitted_van_t_Hoff (NOT calorimetric)
# ----------------------------------------------------------------------------

def test_6b_remains_in_scope_after_R2_erratum() -> None:
    """6b is RETAINED IN SCOPE — the R2 erratum unit-corrects the K_H reference
    but the branch is NOT dropped. Branch count: 15 → 17 → 18 (2026-05-19
    pass-5 R6 added 1d Mercado 2016 Model 4) → 22 (2026-06-01 final pivot
    added 4 5c replacement-scalar branches)."""
    branches = []
    for case in _matrix().get("cases", []):
        for b in case.get("branches", []):
            branches.append(b["branch_id"])
    assert "6b" in branches
    assert len(branches) == 22


def test_6b_K_H_unit_corrected_to_0p806_mol_per_kg_per_bar() -> None:
    """6b K_H reference must now be 0.806 mol/(kg·bar), the unit-corrected
    value from Talu Table 4 raw 0.00691 mol/(kg·kPa) at 305.45 K with van't
    Hoff extrapolation to 298.15 K."""
    branch = _branch("6b")
    kh = branch["references"]["K_H"]
    assert kh.get("value") == pytest.approx(0.806, abs=0.005), (
        f"6b K_H value should be ≈ 0.806 mol/(kg·bar) post-R2 erratum, got {kh.get('value')}"
    )
    assert kh.get("units") == "mol/(kg*bar)"
    assert kh.get("temperature_K") == pytest.approx(298.15, abs=0.01)


def test_6b_K_H_acceptance_window_is_strict_pm_0p10_logK() -> None:
    """Strict ±0.10 log10 around 0.806 mol/(kg·bar) → window ≈ [0.640, 1.015]."""
    branch = _branch("6b")
    kh = branch["references"]["K_H"]
    value = kh.get("value")
    wmin = kh.get("acceptance_window_min")
    wmax = kh.get("acceptance_window_max")
    expected_min = value / 10**0.10
    expected_max = value * 10**0.10
    assert wmin == pytest.approx(expected_min, abs=0.005), (
        f"6b acceptance_window_min={wmin} expected ≈ {expected_min:.3f}"
    )
    assert wmax == pytest.approx(expected_max, abs=0.005), (
        f"6b acceptance_window_max={wmax} expected ≈ {expected_max:.3f}"
    )


def test_6b_K_H_reference_unit_correction_from_Talu_Table_4() -> None:
    """Reconstruct K_H from Talu Table 4 raw values (mol/(kg·kPa)) + van't Hoff.
    Catches future drift: if anyone breaks the kPa→bar (×100) or van't Hoff
    helpers, this test fails.

    Operator-supplied Table 4 verbatim (2026-05-17):
      Kr T=305.45 K: K_H_exp = 0.00691 mol/(kg·kPa)
      Kr T=342.55 K: K_H_exp = 0.00349 mol/(kg·kPa)
    """
    import math
    K_low_kPa = 0.00691
    K_high_kPa = 0.00349
    T_low = 305.45
    T_high = 342.55

    # Step 1: mol/(kg·kPa) → mol/(kg·bar) is ×100 (1 bar = 100 kPa). This is the
    # exact factor the YAML pre-erratum was missing (8.064 = 0.00691 × ~1166 was
    # off by ~10× from the correct 0.691; recall the erratum analysis).
    assert K_low_kPa * 100 == pytest.approx(0.691, abs=0.001)
    assert K_high_kPa * 100 == pytest.approx(0.349, abs=0.001)

    # Step 2: van't Hoff slope between 305.45 K and 342.55 K → Q_st ≈ 16.02 kJ/mol
    R = 8.314462618e-3  # kJ/(mol·K)
    Q_vH = R * math.log(K_low_kPa / K_high_kPa) / (1 / T_low - 1 / T_high)
    assert Q_vH == pytest.approx(16.02, abs=0.05), (
        f"Van't Hoff Q_st from Talu Table 4 two K_H values = {Q_vH:.3f} kJ/mol; "
        f"expected ≈ 16.02"
    )

    # Step 3: extrapolate K_H to 298.15 K
    T_ref = 298.15
    K_298_kPa = K_low_kPa * math.exp(Q_vH / R * (1 / T_ref - 1 / T_low))
    K_298_bar = K_298_kPa * 100
    assert K_298_bar == pytest.approx(0.806, abs=0.005), (
        f"K_H(298.15 K) extrapolated from Talu Table 4 = {K_298_bar:.4f} mol/(kg·bar); "
        f"expected ≈ 0.806"
    )

    # Cross-check against the YAML value
    yaml_K_H = _branch("6b")["references"]["K_H"]["value"]
    assert yaml_K_H == pytest.approx(K_298_bar, rel=0.01)


def test_6b_pre_erratum_8p064_value_was_factor_of_10_error() -> None:
    """Sentinel: the pre-R2 YAML value 8.064 mol/(kg·bar) was off by ~10× from
    the correctly-unit-converted 0.806. This catches accidental re-introduction
    of the factor-of-10 error during future refactors.
    """
    pre_erratum_yaml_value = 8.064
    post_erratum_yaml_value = _branch("6b")["references"]["K_H"]["value"]
    ratio = pre_erratum_yaml_value / post_erratum_yaml_value
    assert 9.5 < ratio < 10.5, (
        f"Pre-erratum 6b K_H {pre_erratum_yaml_value} should be ~10× the "
        f"corrected {post_erratum_yaml_value}; got ratio {ratio:.3f}"
    )


def test_6b_Q_st_classified_as_fitted_not_calorimetric() -> None:
    """Classification history (per V04_LOCKED_SPEC_CHANGELOG.md):
      2026-05-18: reference_blocked_pending_golden_sircar_1994
      2026-05-19 pass-2 R3-audit-trail: reference_blocked_secondary_heat_value_found_zero_coverage_method_unresolved
      2026-05-19 pass-3 R4: reference_anchored_secondary (operator-promoted)

    R4 promotes Q_st reference to 16.39 kJ/mol from Ads@UC secondary database
    (NOT primary-PDF-verbatim). The fitted-van't-Hoff method classification
    is preserved (Q_st_method_compatibility with the atlas two_point_van_t_Hoff
    remains COMPATIBLE).
    """
    branch = _branch("6b")
    q = branch["references"]["Q_st"]
    cls = q.get("classification") or ""
    assert cls == "reference_anchored_secondary", (
        f"6b Q_st classification expected 'reference_anchored_secondary' "
        f"after 2026-05-19 pass-3 R4 promotion; got '{cls}'."
    )
    # Provisional van't Hoff zero-coverage estimate kept as cross-corroboration
    assert q.get("provisional_fitted_van_t_Hoff_value") == pytest.approx(16.02, abs=0.01)
    # Value is now the operator-promoted Ads@UC secondary-anchored figure
    assert q.get("value") == pytest.approx(16.39, abs=0.01)
    assert q.get("method") == "fitted_van_t_Hoff_from_Talu_Table_4"
    assert q.get("source_type") == "fitted_van_t_Hoff_from_primary_experimental_Henry_constants"
    assert q.get("not_calorimetric") is True
    # Provenance must explicitly tag secondary_anchored, NOT primary_pdf_verbatim
    prov = q.get("provenance", {})
    assert prov.get("provenance_tag") == "secondary_anchored"
    assert prov.get("not_primary_pdf_verbatim") is True
    assert prov.get("source_secondary_database") == "Ads@UC"
    # Calorimetric note must explain why
    calorimetric_note = q.get("calorimetric_note", "")
    assert "Table 5" in calorimetric_note
    assert "absent" in calorimetric_note.lower() or "kr is" in calorimetric_note.lower()


def test_6b_Q_st_2026_05_19_pass_2_audit_trail_records_secondary_corroboration_and_independent_verification() -> None:
    """2026-05-19 pass-2 R3-audit-trail erratum: 6b Q_st YAML must record
    (a) Ads@UC secondary corroboration -ΔH = 16.39 kJ/mol at q = 0.58 mol/kg
        explicitly tagged as finite_loading_NOT_zero_coverage; AND
    (b) Independent ReadKong verification of the two Talu Table 4 Kr K_H rows
        (305.45 K and 342.55 K) with the operator-supplied archive SHA.
    """
    branch = _branch("6b")
    q = branch["references"]["Q_st"]

    sec = q.get("secondary_corroboration_ads_uc_2026_05_19", {})
    assert sec.get("minus_delta_H_kJ_per_mol") == pytest.approx(16.39, abs=0.01), (
        "Ads@UC secondary value 16.39 kJ/mol must be recorded verbatim."
    )
    assert sec.get("loading_mol_per_kg") == pytest.approx(0.58, abs=0.01)
    assert sec.get("regime_classification") == "finite_loading_NOT_zero_coverage"
    assert sec.get("archive_evidence_sha256") == (
        "4ab23101f69ac325ea664c487ddaaf62d322239a7afb7e9d092721f2fc448d14"
    )

    ver = q.get("independent_verification_of_K_H_table_4_2026_05_19", {})
    rows = ver.get("verified_rows", [])
    assert len(rows) == 2
    assert rows[0]["temperature_C"] == pytest.approx(32.3, abs=0.05)
    assert rows[0]["B_over_kT_experimental_mol_per_kg_per_kPa"] == pytest.approx(0.00691, abs=1e-5)
    assert rows[1]["temperature_C"] == pytest.approx(69.4, abs=0.05)
    assert rows[1]["B_over_kT_experimental_mol_per_kg_per_kPa"] == pytest.approx(0.00349, abs=1e-5)
    assert ver.get("archive_evidence_sha256") == (
        "4ab23101f69ac325ea664c487ddaaf62d322239a7afb7e9d092721f2fc448d14"
    )


def test_6b_Q_st_acceptance_window_strict_pm_2p0_kJ_per_mol() -> None:
    """Strict ±2.0 kJ/mol around the operator-promoted reference value.
    Window history (R4 erratum):
      2026-05-17 R2: [14.0, 18.0] (centered on 16.02 van't Hoff)
      2026-05-19 R4: [14.39, 18.39] (centered on 16.39 Ads@UC secondary-anchored).
    """
    branch = _branch("6b")
    q = branch["references"]["Q_st"]
    wmin = q.get("acceptance_window_min")
    wmax = q.get("acceptance_window_max")
    assert wmin == pytest.approx(14.39, abs=0.01)
    assert wmax == pytest.approx(18.39, abs=0.01)
    # Window width is strict ±2.0 kJ/mol (must not loosen)
    assert (wmax - wmin) == pytest.approx(4.0, abs=0.01)


def test_6b_erratum_block_carries_raw_table_values() -> None:
    """The 6b K_H erratum block must include the raw Talu Table 4 values
    verbatim with their printed units, for full audit traceability."""
    branch = _branch("6b")
    erratum = branch["references"]["K_H"]["erratum"]
    raw = erratum.get("raw_table_values") or {}
    assert raw.get("kr_305p45K_exp_mol_per_kg_per_kPa") == pytest.approx(0.00691, abs=1e-6)
    assert raw.get("kr_342p55K_exp_mol_per_kg_per_kPa") == pytest.approx(0.00349, abs=1e-6)
    assert raw.get("pressure_basis") == "kPa"
    assert raw.get("conversion_factor_to_bar") == 100
    # Table 4 heading verbatim
    heading = raw.get("table_heading", "")
    assert "Henry constants" in heading
    assert "B/kT" in heading


def test_6b_K_H_erratum_metadata_complete() -> None:
    """The K_H erratum block must include: corrected_from, corrected_to,
    correction_factor, correction_kind, correction_evidence, correction_date_utc,
    correction_revision, and a test pointer."""
    erratum = _branch("6b")["references"]["K_H"]["erratum"]
    assert erratum["corrected_from"] == 8.064
    assert erratum["corrected_to"] == pytest.approx(0.806, abs=0.001)
    assert erratum["correction_kind"] == "literature_reference_unit_correction"
    assert erratum["correction_date_utc"] == "2026-05-17"
    assert erratum.get("correction_revision") == "R2"
    assert "V04_REFERENCE_AUDIT.md" in erratum.get("correction_evidence", "")


def test_6b_branch_scope_retained_in_v04() -> None:
    """The 6b erratum must explicitly mark branch_scope=retained_in_v04 and
    record the 2026-05-18 split scalar verdict: K_H active (Talu-Myers Table 4
    verbatim) + Q_st reference_blocked (pending Golden-Sircar 1994)."""
    vm = _branch("6b").get("verdict_machinery") or {}
    assert vm.get("branch_scope") == "retained_in_v04"
    assert vm.get("scalar_verdict") == "K_H_active_Q_st_reference_blocked"


# ----------------------------------------------------------------------------
# 5b — REFERENCE_BLOCKED sentinel (loading-peak vs zero-coverage mismatch)
# ----------------------------------------------------------------------------

def test_5b_K_H_reference_is_ill_conditioned_two_point_Langmuir() -> None:
    """5b K_H = 252.5 is derived from a 2-point Langmuir at 0.1 and 0.2 bar
    where the loading is 87 % / 93 % of q_sat. This test pins the
    sensitivity at the audit-documented level (±30-60% per 2% data
    perturbation) and FAILS if anyone replaces 252.5 with an apparently
    'better' value that is still derived the same way.
    """
    from scipy.optimize import fsolve
    p1, n1 = 0.1, 3.274
    p2, n2 = 0.2, 3.501
    def F(x, dn1=0.0):
        qs, K = x
        return [qs * K * p1 / (1 + K * p1) - n1 * (1 + dn1),
                qs * K * p2 / (1 + K * p2) - n2]
    sol0 = fsolve(F, [4.0, 50.0], args=(0.0,))
    qs0, K0 = sol0
    KH0 = qs0 * K0
    # Should reproduce 252.5 within rounding
    assert pytest.approx(252.5, abs=0.5) == KH0
    # +2% perturbation: K_H should jump > 30%
    sol_up = fsolve(F, [4.0, 50.0], args=(0.02,))
    KH_up = sol_up[0] * sol_up[1]
    drift = abs(KH_up - KH0) / KH0
    assert drift > 0.30, (
        f"5b K_H sensitivity is suspiciously low ({drift:.1%}). "
        f"Audit (V04_REFERENCE_AUDIT.md §5) found ±30-60% sensitivity from 2% "
        f"loading perturbation. If this fails, either Langmuir fit is using a "
        f"DIFFERENT p_1/n_1 pair or the regime is no longer near-saturation."
    )


# ----------------------------------------------------------------------------
# 5b erratum (2026-05-17) — branch retained, scalar reclassified
# ----------------------------------------------------------------------------

def test_5b_remains_in_scope_after_erratum() -> None:
    """5b is RETAINED IN SCOPE — the erratum reclassifies the scalar verdict
    but the branch is not dropped from the matrix. This guards against
    silent scope reduction. Branch count: 15 → 17 → 18 (2026-05-19 pass-5
    R6 added 1d Mercado 2016 Model 4) → 22 (2026-06-01 final pivot added 4
    5c replacement-scalar branches; 5b itself unchanged)."""
    branches = []
    for case in _matrix().get("cases", []):
        for b in case.get("branches", []):
            branches.append(b["branch_id"])
    assert "5b" in branches
    assert len(branches) == 22


def test_5b_site_truth_remains_enabled() -> None:
    """5b site-truth verdict MUST remain active — geometry (Na-OC1, Na-OC3)
    is well-defined regardless of the Henry-regime question."""
    branch = _branch("5b")
    assert branch["site_truth"]["enabled"] is True
    vm = branch.get("verdict_machinery") or {}
    assert vm.get("site_truth_verdict") == "active"


def test_5b_scalar_K_H_is_reference_blocked() -> None:
    """The 5b scalar K_H reference must be reclassified to reference_blocked
    (trapdoor zeolite, no Henry regime). The HISTORICAL value 252.5 is
    retained for traceability but NOT used as strict scalar truth."""
    branch = _branch("5b")
    vm = branch.get("verdict_machinery") or {}
    assert vm.get("scalar_verdict") == "reference_blocked"
    kh = branch["references"]["K_H"]
    assert kh.get("classification") == "reference_blocked_trapdoor_no_Henry_regime"
    assert kh.get("affects_v04_scalar_verdict") is False
    # Historical value retained
    assert kh.get("value") == 252.5
    # Erratum block present
    erratum = kh.get("erratum")
    assert erratum is not None
    assert erratum["correction_kind"] == "reference_observable_mismatch_trapdoor_no_Henry_regime"
    assert "V04_REFERENCE_AUDIT.md" in erratum.get("correction_evidence", "")
    assert erratum.get("correction_date_utc") == "2026-05-17"


def test_5b_scalar_Q_st_is_reference_blocked() -> None:
    """The 5b scalar Q_st reference must be reclassified to
    reference_blocked (45 kJ/mol is loading-peak, not zero-coverage)."""
    branch = _branch("5b")
    q = branch["references"]["Q_st"]
    assert q.get("classification") == "reference_blocked_loading_peak_not_zero_coverage"
    assert q.get("affects_v04_scalar_verdict") is False
    assert q.get("value") == 45.0   # historical, retained for traceability
    erratum = q.get("erratum")
    assert erratum is not None
    assert erratum["correction_kind"] == "reference_observable_mismatch_loading_peak_not_zero_coverage"
    assert erratum.get("correction_date_utc") == "2026-05-17"


def test_5b_historical_252p5_value_documented_as_provisional() -> None:
    """The old 252.5 mol/(kg·bar) must be tagged as PROVISIONAL/historical
    in the derivation field — never as strict scalar truth."""
    branch = _branch("5b")
    kh = branch["references"]["K_H"]
    derivation = kh.get("derivation") or ""
    assert "PROVISIONAL" in derivation or "provisional" in derivation, (
        "5b K_H derivation must explicitly mark 252.5 as PROVISIONAL/historical"
    )
    # Old window values renamed to historical_*
    assert "historical_acceptance_window_min" in kh
    assert "historical_acceptance_window_max" in kh
    # Live acceptance window keys should not be present (so they cannot be
    # accidentally picked up by the verdict emitter)
    assert kh.get("acceptance_window_min") is None
    assert kh.get("acceptance_window_max") is None


def test_5b_branch_scope_retained_in_v04() -> None:
    """The erratum must explicitly mark branch_scope=retained_in_v04* so any
    future reader (or aggregator) can confirm the branch was not silently
    dropped. The 2026-05-18 update tightened the scope string to
    `retained_in_v04_site_truth_only` to reflect that the scalar K_H/Q_st
    axis is METHOD-blocked (closed-state Widom vs open-state Henry) while
    the site-truth track remains active."""
    vm = _branch("5b").get("verdict_machinery") or {}
    assert (vm.get("branch_scope") or "").startswith("retained_in_v04")
    # 5b scalar is METHOD-blocked (different observable than the experimental
    # Langmuir K_H). The exact missing method must be recorded.
    assert vm.get("scalar_block_kind") == "method_blocked"
    missing = vm.get("scalar_block_missing_method") or ""
    assert "GCMC" in missing or "trapdoor" in missing.lower()


def test_5b_scientific_validation_treats_reference_blocked_as_unresolved() -> None:
    """A REFERENCE_BLOCKED verdict on 5b must NOT count as PASS or BROAD_PASS,
    and the overall scientific_validation_pass must still be False until 5b
    is resolved."""
    from widom_atlas.v04.audit.pass_criteria import scientific_validation_pass
    # Synthetic verdicts: every strict branch passes, except 5b is reference_blocked.
    synth = {bid: {"verdict": "PASS"} for bid in
             ("1a", "1b", "2a", "3a", "4a", "6a", "6b", "6c")}
    synth["5b"] = {"verdict": "REFERENCE_BLOCKED"}
    ok, detail = scientific_validation_pass(synth)
    assert not ok
    assert "5b" in detail
    assert "REFERENCE_BLOCKED" in detail


def test_5b_Q_st_reference_is_loading_peak_not_zero_coverage() -> None:
    """Lozinska 2014 SI Figure S2.2 shows q_st(loading) for CO2 on Na-Rho.
    Zero-coverage q_st ≈ 15-20 kJ/mol; loading-peak q_st ≈ 42 kJ/mol at
    ~2 mmol/g (≈ 1 CO2 per Na cation). The YAML's 45 kJ/mol is the peak,
    NOT the Widom-comparable zero-coverage value. Sentinel: ensure this
    mismatch is documented in the YAML (either via classification field or
    a clear note) — otherwise the audit will silently produce a
    physically-mismatched 'FAIL'.
    """
    branch = _branch("5b")
    q = branch["references"]["Q_st"]
    yaml_Q = q["value"]
    assert yaml_Q == 45.0
    # If the erratum has been applied, the classification field flags the issue.
    classification = q.get("classification") or ""
    note = (q.get("classification_note") or "") + (q.get("derivation") or "")
    has_documentation = (
        "reference_blocked" in classification
        or "loading" in note.lower()
        or "peak" in note.lower()
        or "configuration_conditional" in classification
    )
    assert has_documentation, (
        "5b Q_st = 45 kJ/mol is the loading-PEAK heat per Lozinska 2014 SI "
        "Figure S2.2, NOT the Widom-zero-coverage observable. The YAML must "
        "document this observable mismatch — either via "
        "'classification: reference_blocked_loading_peak_not_zero_coverage' "
        "(proposed erratum) or a note in 'derivation'. See V04_REFERENCE_AUDIT.md §5."
    )


# ============================================================================
# 2026-05-19 pass-2 regression tests — operator-supplied literature data
# (archive SHA-256 4ab23101f69ac325ea664c487ddaaf62d322239a7afb7e9d092721f2fc448d14)
# ============================================================================

TALU_TABLE_4_KR = [
    {"T_C": 32.3, "T_K": 305.45, "B_over_kT_mol_per_kg_per_kPa": 0.00691},
    {"T_C": 69.4, "T_K": 342.55, "B_over_kT_mol_per_kg_per_kPa": 0.00349},
]


def test_6b_Kr_K_H_van_t_Hoff_reconstruction_matches_YAML() -> None:
    """Reconstruct 6b Kr K_H at 298.15 K from the two Talu-Myers Table 4 Kr rows
    (verbatim ReadKong open-source values 0.00691 mol/(kg.kPa) at 305.45 K and
    0.00349 at 342.55 K) and assert the YAML 6b K_H matches within 1%.

    Confirms operator's R2 erratum (2026-05-17) that set 6b K_H = 0.806 from
    operator-supplied Talu Table 4 verbatim. The independent ReadKong
    transcription matches the YAML value to 0.05%.
    """
    R = 8.314
    T1 = TALU_TABLE_4_KR[0]["T_K"]
    T2 = TALU_TABLE_4_KR[1]["T_K"]
    KH1_bar = TALU_TABLE_4_KR[0]["B_over_kT_mol_per_kg_per_kPa"] * 100.0
    KH2_bar = TALU_TABLE_4_KR[1]["B_over_kT_mol_per_kg_per_kPa"] * 100.0
    Q_st_J = R * math.log(KH1_bar / KH2_bar) / (1.0 / T1 - 1.0 / T2)
    T_target = 298.15
    ln_ratio = (Q_st_J / R) * (1.0 / T_target - 1.0 / T1)
    KH_298_reconstructed = KH1_bar * math.exp(ln_ratio)

    branch = _branch("6b")
    KH_yaml = branch["references"]["K_H"]["value"]
    rel_err = abs(KH_298_reconstructed - KH_yaml) / KH_yaml
    assert rel_err < 0.01, (
        f"6b K_H reconstruction from Talu Table 4 verbatim gives {KH_298_reconstructed:.4f} "
        f"mol/(kg.bar) at 298.15 K; YAML has {KH_yaml}; relative error {rel_err*100:.2f}% "
        "exceeds 1% tolerance. The R2 erratum (2026-05-17) must be checked."
    )


def test_6b_Kr_Q_st_van_t_Hoff_reconstruction_yields_16_kJ_per_mol() -> None:
    """Re-derive zero-coverage Kr Q_st via van't Hoff from the two Talu Table 4
    Kr K_H rows and assert the YAML provisional_fitted_van_t_Hoff_value matches.

    This is a regression test for the conversion math, not for whether 6b Q_st
    is REFERENCE_BLOCKED on the strict axis. The value 16.02 ± 0.05 kJ/mol is
    derivable from primary K_H data; the operator's 2026-05-19 pass-2 Ads@UC
    secondary corroboration (16.39 kJ/mol at q=0.58 mol/kg) is finite-loading
    and does not change the strict axis.
    """
    R = 8.314e-3  # kJ/(mol·K)
    T1 = TALU_TABLE_4_KR[0]["T_K"]
    T2 = TALU_TABLE_4_KR[1]["T_K"]
    K1 = TALU_TABLE_4_KR[0]["B_over_kT_mol_per_kg_per_kPa"]
    K2 = TALU_TABLE_4_KR[1]["B_over_kT_mol_per_kg_per_kPa"]
    Q_st_reconstructed = R * math.log(K1 / K2) / (1.0 / T1 - 1.0 / T2)

    branch = _branch("6b")
    Q_st_yaml = branch["references"]["Q_st"]["provisional_fitted_van_t_Hoff_value"]
    assert abs(Q_st_reconstructed - Q_st_yaml) < 0.05, (
        f"6b Q_st van't Hoff reconstruction = {Q_st_reconstructed:.3f} kJ/mol; "
        f"YAML provisional_fitted_van_t_Hoff_value = {Q_st_yaml}; "
        "deviation exceeds 0.05 kJ/mol. Conversion: "
        "Q_st = R · ln(K1/K2) / (1/T1 - 1/T2) with the two Talu Table 4 Kr rows."
    )
    assert 15.9 < Q_st_yaml < 16.1, (
        f"YAML 6b Q_st provisional value {Q_st_yaml} should be ~16.02 kJ/mol "
        "(zero-coverage van't Hoff from primary Henry-regime K_H data)."
    )


def test_1a_1b_Mason_K_H_pinned_at_381_no_silent_drift() -> None:
    """Regression test guarding against silent drift of 1a/1b Mason K_H
    reference. Historical values: 187 (initial), 216 (post first Mason audit),
    381 (Mason 2011 SI Table S6 DSL derivation 2026-05-18, current).
    Vandenbrande 2018 Table 2 reports Mason at 384 — within 1% of YAML 381,
    corroborating the current DSL derivation.

    Any future change to this value requires explicit erratum (changelog +
    source documentation) — this test must be updated in lockstep with the
    erratum, NOT silently.
    """
    EXPECTED_MASON_K_H = 381.0
    DRIFT_TOLERANCE = 5.0  # mol/(kg.bar) — accommodates Vandenbrande Table 2 Mason 384
    OUTDATED_FORBIDDEN = {187.0, 216.0}
    for bid in ("1a", "1b"):
        branch = _branch(bid)
        kh = branch["references"]["K_H"]["value"]
        assert abs(kh - EXPECTED_MASON_K_H) <= DRIFT_TOLERANCE, (
            f"{bid} Mason K_H reference drifted to {kh}; expected "
            f"{EXPECTED_MASON_K_H} ± {DRIFT_TOLERANCE} mol/(kg.bar). "
            "If this is intentional, update this test in the same commit "
            "as a documented erratum in V04_LOCKED_SPEC_CHANGELOG.md."
        )
        assert kh not in OUTDATED_FORBIDDEN, (
            f"{bid} K_H = {kh} matches an outdated historical value "
            f"(initial 187 or first-audit 216). Mason DSL derivation pins it at "
            f"{EXPECTED_MASON_K_H}. See V04_MASON_2011_REFERENCE_AUDIT.md."
        )


# ============================================================================
# 2026-05-19 pass-3 R4 promotion regression tests — 6b Q_st reference_anchored_secondary
# Per operator directive (verbatim): "Add regression tests for:
#   0.00691 mol/(kg kPa) -> 0.691 mol/(kg bar)
#   0.00349 mol/(kg kPa) -> 0.349 mol/(kg bar)
#   Q_st reference = 16.39 kJ/mol
#   no kPa/bar/Pa mix-up"
# ============================================================================


def test_6b_Kr_K_H_kPa_to_bar_conversion_0p00691_yields_0p691() -> None:
    """Conversion check: Talu Table 4 Kr row at 32.3 °C reports
    B/kT = 0.00691 mol/(kg.kPa). Multiplying by 100 (kPa/bar) gives
    0.691 mol/(kg.bar). No kPa->bar->Pa->kPa unit cycling allowed.
    """
    B_over_kT_kPa = 0.00691
    K_H_bar = B_over_kT_kPa * 100.0
    assert K_H_bar == pytest.approx(0.691, abs=1e-4), (
        f"kPa->bar conversion: 0.00691 mol/(kg.kPa) * 100 kPa/bar = "
        f"{K_H_bar} mol/(kg.bar); expected 0.691."
    )


def test_6b_Kr_K_H_kPa_to_bar_conversion_0p00349_yields_0p349() -> None:
    """Conversion check: Talu Table 4 Kr row at 69.4 °C reports
    B/kT = 0.00349 mol/(kg.kPa). Multiplying by 100 (kPa/bar) gives
    0.349 mol/(kg.bar).
    """
    B_over_kT_kPa = 0.00349
    K_H_bar = B_over_kT_kPa * 100.0
    assert K_H_bar == pytest.approx(0.349, abs=1e-4), (
        f"kPa->bar conversion: 0.00349 mol/(kg.kPa) * 100 kPa/bar = "
        f"{K_H_bar} mol/(kg.bar); expected 0.349."
    )


def test_6b_Kr_Q_st_reference_value_16p39_kJ_per_mol_with_secondary_provenance() -> None:
    """Operator R4 promotion: 6b Q_st reference = 16.39 kJ/mol from Ads@UC
    secondary database. Provenance must be tagged secondary_anchored (NOT
    primary_pdf_verbatim) with the full source chain (Golden-Sircar 1994
    primary, Talu-Myers 2001 cross-citation, Ads@UC secondary database).
    """
    branch = _branch("6b")
    q = branch["references"]["Q_st"]
    assert q.get("value") == pytest.approx(16.39, abs=0.01), (
        f"6b Q_st reference value expected 16.39 kJ/mol (Ads@UC); "
        f"got {q.get('value')}. See V04_LOCKED_SPEC_CHANGELOG.md R4 erratum."
    )
    assert q.get("classification") == "reference_anchored_secondary"
    prov = q.get("provenance", {})
    assert prov.get("source_primary_doi") == "10.1006/jcis.1994.1023"
    assert prov.get("source_cross_citation_doi") == "10.1016/S0927-7757(01)00628-8"
    assert prov.get("source_secondary_database") == "Ads@UC"
    assert prov.get("provenance_tag") == "secondary_anchored"
    assert prov.get("not_primary_pdf_verbatim") is True
    # Caveat must explicitly mention finite-loading q = 0.58 mol/kg
    caveat = (prov.get("caveat") or "").lower()
    assert "0.58" in caveat and "finite" in caveat


def test_6b_Kr_K_H_no_kPa_bar_Pa_mixup() -> None:
    """Sanity check: 6b K_H = 0.806 mol/(kg.bar) at 298.15 K is in `bar` units;
    must NOT be 0.806e-5 mol/(kg.Pa) (Pa), 80.6 mol/(kg.kPa) (kPa), 806
    mol/(kg.kPa), or any other unit mix-up. The conversion chain is:
      Talu Table 4 raw: 0.00691 mol/(kg.kPa) at 305.45 K
      -> *100 kPa/bar -> 0.691 mol/(kg.bar) at 305.45 K
      -> van't Hoff extrapolation to 298.15 K -> 0.806 mol/(kg.bar).
    """
    branch = _branch("6b")
    kh = branch["references"]["K_H"]
    value = kh.get("value")
    units = (kh.get("units") or "").lower()
    assert "bar" in units and "kg" in units, (
        f"6b K_H units expected to contain 'kg' and 'bar'; got '{units}'."
    )
    # value must be ~0.806 (van't Hoff to 298.15 K), not the raw kPa-basis number
    assert 0.7 < value < 0.95, (
        f"6b K_H value {value} should be ~0.806 mol/(kg.bar) at 298.15 K. "
        "Suspect kPa<->bar unit mix-up: if value is ~80 or ~8e-6, the conversion "
        "is wrong (raw kPa value should be ~0.00691, bar value should be ~0.691 "
        "at 305.45 K, ~0.806 at 298.15 K). Note: NIST 8e-6 mol/(kg.Pa) row from "
        "MFI+CH4 fixture is for a different gas; do not import it here."
    )
    # Forbid known unit-mix-up sentinel values
    FORBIDDEN_UNIT_MIXUP_VALUES = {
        8.064,    # pre-R2-erratum value (was 10x too high — kPa/Pa mix-up)
        0.00806,  # 0.806e-2 if someone mistook the kPa value for bar
        80.6,     # 0.806 * 100 (extra kPa->bar)
        0.806e-5, # bar->Pa drift (with 1 bar = 1e5 Pa)
    }
    assert value not in FORBIDDEN_UNIT_MIXUP_VALUES, (
        f"6b K_H value {value} matches a known unit-mix-up sentinel."
    )


# ============================================================================
# 2026-05-28 5b finalisation packet regression tests
# (Lozinska 2012 SI + CIF + Brandani 2021 review + Brandani Excel inspected;
# 5b scalar finalised as method_blocked_or_reference_blocked; site-truth active)
# ============================================================================

LOZINSKA_MAIN_PDF_SHA256 = "1373dff266e6592adbefed8c3a83aa993476f0e59d903710bea4111bd73079dd"
LOZINSKA_SI_SHA256 = "fce096fc0574e18719216c9a8b8f8da4690d9e801c4c37e5081ff8900e246216"
LOZINSKA_CIF_SHA256 = "c5161958f210148e26fd174436c05a5e5e26866f384019c8d4e000c70d0d647e"
BRANDANI_REVIEW_SHA256 = "9945a3b770c7a7f2319ce020a8bb8744270e257b89a0242c07ef2c2f9650d486"
BRANDANI_EXCEL_SHA256 = "6d68da1465557e2898e8d514fd93ea2b3f80120fd4e220f21ce69789596deaa4"


def test_5b_scalar_is_method_blocked_after_lozinska_main_and_si_inspection() -> None:
    """After 2026-05-28 packet inspection (Lozinska main paper text + SI + CIF
    + Brandani review + Brandani Excel), 5b K_H and Q_st are finalised as
    method_blocked_or_reference_blocked. No admissible Henry-regime scalar
    was found. Per operator: 'finalise 5b Na-Rho as scalar METHOD_BLOCKED /
    REFERENCE_BLOCKED, with site-truth active'.
    """
    branch = _branch("5b")
    finalisation = branch.get("lozinska_2012_packet_2026_05_28_finalisation")
    assert finalisation is not None, (
        "5b must carry a 'lozinska_2012_packet_2026_05_28_finalisation' block "
        "documenting the 2026-05-28 packet ingestion + finalisation."
    )
    assert finalisation.get("K_H_final_classification") == "method_blocked_or_reference_blocked"
    assert finalisation.get("Q_st_final_classification") == "method_blocked_or_reference_blocked"
    assert finalisation.get("site_truth_status") == "active"
    exact_missing = finalisation.get("exact_missing_variable") or ""
    assert "Na9.8" in exact_missing or "Na9.8Al9.8Si38.2O96" in exact_missing, (
        "exact_missing_variable must name pure Na9.8 framework explicitly."
    )
    assert "Henry" in exact_missing
    assert "zero-coverage" in exact_missing or "Q_st" in exact_missing


def test_5b_site_truth_remains_active() -> None:
    """5b site-truth axis stays active per operator finalisation. Tolerance
    ±0.20 Å for Na-O(CO2) distances (Sites A and B from Lozinska 2012
    Table S11.2)."""
    branch = _branch("5b")
    site_truth = branch.get("site_truth") or {}
    assert site_truth.get("enabled") is True, (
        "5b site_truth must be enabled per 2026-05-28 finalisation."
    )
    assert site_truth.get("tolerance_angstrom") == 0.20
    target = site_truth.get("target_geometry") or {}
    assert target.get("site_A_Na_O_distance_angstrom") == 2.88
    assert target.get("site_B_Na_O_distance_angstrom") == 2.58


def test_5b_does_not_convert_0p1bar_uptake_to_henry() -> None:
    """The 0.1 bar Na-Rho CO2 uptake (3.07 mmol/g per Lozinska 2012 main text)
    is past the trapdoor onset; do NOT convert it to a Henry K_H. The
    do_not_use list must explicitly forbid this conversion.
    """
    branch = _branch("5b")
    finalisation = branch.get("lozinska_2012_packet_2026_05_28_finalisation") or {}
    do_not_use = finalisation.get("do_not_use_for_5b_scalar") or []
    do_not_use_text = " ".join(do_not_use).lower()
    assert "0.1 bar uptake" in do_not_use_text and "henry" not in do_not_use_text.split("uptake")[0].split("0.1 bar")[-1][:20].lower(), (
        "do_not_use list must explicitly forbid 0.1 bar uptake as Henry K_H source."
    )
    assert any("0.1 bar uptake" in item.lower() for item in do_not_use), (
        "do_not_use must list '0.1 bar uptake' explicitly."
    )


def test_5b_does_not_fit_two_point_langmuir_from_lozinska_finite_uptake() -> None:
    """The pre-pass-1 K_H = 252.5 mol/(kg·bar) was a 2-point Langmuir fit
    through Lozinska's 0.1 and 0.2 bar loadings — well past saturation. The
    2026-05-28 finalisation must forbid this fit explicitly.
    """
    branch = _branch("5b")
    finalisation = branch.get("lozinska_2012_packet_2026_05_28_finalisation") or {}
    do_not_use = finalisation.get("do_not_use_for_5b_scalar") or []
    assert any("two-point" in item.lower() or "2-point" in item.lower() or "two point" in item.lower()
               for item in do_not_use), (
        "do_not_use must explicitly forbid two-point Langmuir K_H fit from Lozinska finite uptake."
    )
    # The historical 252.5 value must remain in references.K_H.value as
    # traceability (NOT a verdict-affecting value) — already pinned by the
    # pass-1 erratum 2026-05-17.
    kh = (branch.get("references") or {}).get("K_H") or {}
    assert kh.get("affects_v04_scalar_verdict") is False, (
        "Historical K_H=252.5 must remain non-verdict-affecting (traceability only)."
    )


def test_brandani_excel_is_provenance_only_not_scalar_source() -> None:
    """Brandani 2021 supplementary Excel is a bibliography/index entry for
    Lozinska 2012; it is NOT raw data and must not be used as a strict
    scalar source."""
    branch = _branch("5b")
    finalisation = branch.get("lozinska_2012_packet_2026_05_28_finalisation") or {}
    sources_inspected = finalisation.get("sources_inspected_2026_05_28") or []
    brandani_excel = next(
        (s for s in sources_inspected
         if "Brandani 2021 supplementary Excel" in (s.get("source") or "")),
        None,
    )
    assert brandani_excel is not None, (
        "Brandani 2021 supplementary Excel must be recorded under sources_inspected."
    )
    assert brandani_excel.get("do_not_use_as_raw_data") is True, (
        "Brandani Excel must carry do_not_use_as_raw_data: true."
    )
    use_case = (brandani_excel.get("use_case") or "").lower()
    assert "bibliography" in use_case or "index" in use_case, (
        "Brandani Excel use_case must say bibliography/index, not raw data."
    )


def test_lozinska_sources_recorded_with_sha256() -> None:
    """The 2026-05-28 packet sources must be recorded with SHA-256 hashes in
    the YAML 5b finalisation block, matching the SHA256 manifest at
    docs/research/dataset-research-for-v0.4/5b_na_rho/SHA256_MANIFEST.md.
    """
    branch = _branch("5b")
    finalisation = branch.get("lozinska_2012_packet_2026_05_28_finalisation") or {}
    sources_inspected = finalisation.get("sources_inspected_2026_05_28") or []
    sha256_by_source = {}
    for s in sources_inspected:
        src = s.get("source") or ""
        sha = s.get("sha256")
        if sha:
            sha256_by_source[src] = sha
    # Lozinska main JACS PDF (private-provenance; added 2026-05-28)
    main_entry = next((v for k, v in sha256_by_source.items() if "Lozinska 2012 JACS main" in k), None)
    assert main_entry == LOZINSKA_MAIN_PDF_SHA256, (
        f"Lozinska 2012 JACS main PDF SHA-256 must match {LOZINSKA_MAIN_PDF_SHA256}; got {main_entry}."
    )
    # Lozinska SI
    si_entry = next((v for k, v in sha256_by_source.items() if "Lozinska 2012 JACS SI" in k), None)
    assert si_entry == LOZINSKA_SI_SHA256, (
        f"Lozinska SI SHA-256 must match {LOZINSKA_SI_SHA256}; got {si_entry}."
    )
    # Lozinska CIF
    cif_entry = next((v for k, v in sha256_by_source.items() if "CIF" in k and "Lozinska 2012" in k), None)
    assert cif_entry == LOZINSKA_CIF_SHA256
    # Brandani review
    review_entry = next((v for k, v in sha256_by_source.items() if "Brandani 2021 ZLC review" in k), None)
    assert review_entry == BRANDANI_REVIEW_SHA256
    # Brandani Excel
    excel_entry = next((v for k, v in sha256_by_source.items() if "Brandani 2021 supplementary Excel" in k), None)
    assert excel_entry == BRANDANI_EXCEL_SHA256


def test_5b_lozinska_main_pdf_is_archived_private_provenance() -> None:
    """The Lozinska 2012 JACS main paper is a purchased ACS article archived
    2026-05-28 in the private-provenance subdir. YAML must record:
      - status = archived (not 'NOT in repo')
      - repo_path under lozinska_2012_jacs/
      - sha256 matching the manifest
      - do_not_redistribute_publicly = True
    """
    branch = _branch("5b")
    finalisation = branch.get("lozinska_2012_packet_2026_05_28_finalisation") or {}
    sources = finalisation.get("sources_inspected_2026_05_28") or []
    main = next((s for s in sources if "Lozinska 2012 JACS main" in (s.get("source") or "")), None)
    assert main is not None, "Lozinska main paper entry must be present."
    status = main.get("status") or ""
    assert "archived" in status.lower(), (
        f"Lozinska main paper status must say 'archived'; got '{status}'."
    )
    assert main.get("repo_path", "").endswith("lozinska_2012_jacs_main.pdf"), (
        "Lozinska main paper repo_path must end with 'lozinska_2012_jacs_main.pdf'."
    )
    assert main.get("sha256") == LOZINSKA_MAIN_PDF_SHA256
    assert main.get("do_not_redistribute_publicly") is True, (
        "Lozinska main paper must carry do_not_redistribute_publicly: True (purchased article)."
    )
