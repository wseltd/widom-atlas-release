"""Q_st method compatibility rules.

Different sources measure Q_st via different operational definitions; the
strict ±2.0 kJ/mol threshold is only meaningful when atlas and reference
sample the same observable. This module classifies each (atlas_method,
ref_method) pair and emits one of three results:

  COMPATIBLE        — atlas and ref measure the same observable; strict
                       threshold applies.
  COMPATIBLE_NOTE   — atlas and ref measure related but not identical
                       observables; strict threshold still applies but the
                       verdict carries a note explaining the residual offset.
  METHOD_BLOCKED    — atlas and ref measure different observables; the
                       Q_st axis is reported as METHOD_BLOCKED with no
                       PASS/FAIL decision.

Canonical Q_st-method tags:

  direct_widom_boltzmann_weighted
      Boltzmann-weighted ⟨U_test⟩_B from Widom insertions, converted to
      Q_st via -⟨U⟩_B + RT. Native and RASPA2 emit this directly.

  two_point_van_t_Hoff
      Q_st = R · d ln K_H / d(1/T) computed from K_H at two temperatures.
      RASPA3 emits this by default in v0.4.

  multi_point_van_t_Hoff_regression
      Same as above but fit across ≥3 temperatures (e.g. Mason 2011 xls
      19-T van't Hoff regression).

  isosteric_from_isotherms
      Clausius-Clapeyron applied to multi-T isotherms at fixed loading.

  calorimetric_low_loading
      Direct microcalorimetric measurement (e.g. Tian-Calvet, IGC) at
      controlled low loading (typically below 0.1 saturation).

  calorimetric_zero_coverage
      Calorimetric measurement extrapolated to N → 0.

  fitted_van_t_Hoff_from_primary_experimental_Henry_constants
      Two-point van't Hoff fit between published primary-experimental K_H
      values (e.g. Talu-Myers 2001 Table 4 for Kr). Compatible with
      atlas two_point_van_t_Hoff at the same temperatures.
"""
from __future__ import annotations

from typing import Literal

# Compatibility outcomes
COMPATIBLE = "COMPATIBLE"
COMPATIBLE_NOTE = "COMPATIBLE_NOTE"
METHOD_BLOCKED = "METHOD_BLOCKED"

# Common aliases that map to the canonical tags. Keys are normalised
# lowercase substrings; we look up by substring containment so a verbose
# `source` string like "Mason_2011_EES Table 1 (calorimetric low-coverage)"
# also matches `calorimetric_low_loading`.
_TAG_ALIASES: dict[str, str] = {
    # direct Widom
    "direct_widom": "direct_widom_boltzmann_weighted",
    "boltzmann_weighted": "direct_widom_boltzmann_weighted",
    # two-point van't Hoff
    "two_point_van_t_hoff": "two_point_van_t_Hoff",
    "van_t_hoff_two_point": "two_point_van_t_Hoff",
    # multi-point regression
    "multi_point_van_t_hoff": "multi_point_van_t_Hoff_regression",
    "vant_hoff_regression": "multi_point_van_t_Hoff_regression",
    "van_t_hoff_regression": "multi_point_van_t_Hoff_regression",
    # generic van't Hoff (treated as two-point unless explicitly multi)
    "van_t_hoff": "two_point_van_t_Hoff",
    "vant_hoff": "two_point_van_t_Hoff",
    # isosteric from isotherms (Clausius-Clapeyron)
    "isosteric_from_isotherms": "isosteric_from_isotherms",
    "clausius_clapeyron": "isosteric_from_isotherms",
    # calorimetric
    "calorimetric_zero_coverage": "calorimetric_zero_coverage",
    "calorimetric_low_loading": "calorimetric_low_loading",
    "calorimetric_low_coverage": "calorimetric_low_loading",
    "tian_calvet": "calorimetric_low_loading",
    "tian-calvet": "calorimetric_low_loading",
    "microcalorimetric": "calorimetric_low_loading",
    "calorimetric": "calorimetric_low_loading",
    "calorimetry": "calorimetric_low_loading",
    # fitted van't Hoff between published K_H values
    "fitted_van_t_hoff_from_primary": "fitted_van_t_Hoff_from_primary_experimental_Henry_constants",
    # fitted isosteric heat from a dual-site Langmuir (or other multi-site isotherm)
    # fit across multiple temperatures — same conceptual family as multi-point
    # van't Hoff regression. Used by Mason 2011 SI Table S6 for Mg-MOF-74 + CO2.
    "fitted_isosteric_heat_dual_site_langmuir": "fitted_isosteric_heat_dual_site_Langmuir",
    "dual_site_langmuir": "fitted_isosteric_heat_dual_site_Langmuir",
    "dual-site_langmuir": "fitted_isosteric_heat_dual_site_Langmuir",
}


def normalise_q_st_method(raw: str | None) -> str | None:
    """Map an arbitrary YAML `method` or `source` string to a canonical tag."""
    if not raw:
        return None
    s = str(raw).lower()
    s = s.replace(" ", "_").replace("’", "'").replace("-", "_")
    s = s.replace("'", "")
    # Direct substring match against the alias table.
    for needle, tag in _TAG_ALIASES.items():
        if needle in s:
            return tag
    return None


def assess_compatibility(
    atlas_method: str | None,
    ref_method: str | None,
) -> tuple[Literal["COMPATIBLE", "COMPATIBLE_NOTE", "METHOD_BLOCKED"], str]:
    """Decide whether (atlas_method, ref_method) sample the same observable.

    Returns (outcome, note). The note is empty for COMPATIBLE and explains
    the offset / mismatch for the other two outcomes.
    """
    a = normalise_q_st_method(atlas_method)
    r = normalise_q_st_method(ref_method)

    if a is None or r is None:
        return (
            METHOD_BLOCKED,
            f"unknown Q_st method: atlas={atlas_method!r} ref={ref_method!r}",
        )

    if a == r:
        return COMPATIBLE, ""

    # Equivalent pairs (the two methodologies agree at zero coverage)
    EQUIVALENT_PAIRS = {
        frozenset(("direct_widom_boltzmann_weighted", "calorimetric_zero_coverage")),
        frozenset(("two_point_van_t_Hoff", "multi_point_van_t_Hoff_regression")),
        frozenset(("two_point_van_t_Hoff", "fitted_van_t_Hoff_from_primary_experimental_Henry_constants")),
        frozenset(("multi_point_van_t_Hoff_regression", "fitted_van_t_Hoff_from_primary_experimental_Henry_constants")),
        frozenset(("two_point_van_t_Hoff", "fitted_isosteric_heat_dual_site_Langmuir")),
        frozenset(("multi_point_van_t_Hoff_regression", "fitted_isosteric_heat_dual_site_Langmuir")),
    }
    if frozenset((a, r)) in EQUIVALENT_PAIRS:
        return COMPATIBLE, ""

    # Compatible-with-note pairs (related but not identical observables)
    NOTE_PAIRS = {
        frozenset(("direct_widom_boltzmann_weighted", "calorimetric_low_loading")):
            "Atlas is zero-coverage Boltzmann-weighted Widom; reference is calorimetric at low (but non-zero) loading. For single-strong-site systems these agree within a few kJ/mol; for multi-site systems the calorimetric value reflects the strongest-site Q_st while the Widom value averages over all accessible sites. Strict threshold applied with this caveat.",
        frozenset(("two_point_van_t_Hoff", "calorimetric_low_loading")):
            "Atlas is van't Hoff slope across two temperatures; reference is calorimetric at low loading. These differ by R T (PV term) and by the loading-dependence of the calorimetric measurement. Typical offset 1-5 kJ/mol.",
        frozenset(("direct_widom_boltzmann_weighted", "two_point_van_t_Hoff")):
            "Both are derived from the same Widom partition function but use different statistical estimators; agreement within a few kJ/mol at zero coverage.",
        frozenset(("direct_widom_boltzmann_weighted", "fitted_isosteric_heat_dual_site_Langmuir")):
            "Atlas is zero-coverage Boltzmann-weighted Widom; reference is the isosteric heat fitted by a dual-site Langmuir model (E_A typically corresponds to the strongest-site low-loading heat). For single-strong-site systems these agree within a few kJ/mol; for multi-site systems the DSL E_A reflects the strongest-site occupancy. Strict threshold applied with this caveat.",
        frozenset(("direct_widom_boltzmann_weighted", "multi_point_van_t_Hoff_regression")):
            "Atlas is zero-coverage Boltzmann-weighted Widom; reference is the multi-point van't Hoff regression slope (multiple temperatures, low-coverage isotherm fits). Both extract zero-coverage Q_st but via different statistical estimators on related observables. Typical offset 1-3 kJ/mol; strict threshold applied with this caveat.",
    }
    if frozenset((a, r)) in NOTE_PAIRS:
        return COMPATIBLE_NOTE, NOTE_PAIRS[frozenset((a, r))]

    # Anything else is method-blocked.
    return (
        METHOD_BLOCKED,
        f"atlas Q_st method {a!r} and reference Q_st method {r!r} are not "
        f"comparable as the same observable. Either supply a matching-method "
        f"reference or extend `assess_compatibility` with a justification.",
    )
