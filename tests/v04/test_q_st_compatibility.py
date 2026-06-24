"""Unit tests for Q_st_method compatibility."""
from __future__ import annotations

from widom_atlas.v04.audit.q_st_compatibility import (
    COMPATIBLE,
    COMPATIBLE_NOTE,
    METHOD_BLOCKED,
    assess_compatibility,
    normalise_q_st_method,
)


def test_normalise_recognises_canonical_tags():
    assert normalise_q_st_method("direct_widom_boltzmann_weighted") == "direct_widom_boltzmann_weighted"
    assert normalise_q_st_method("two_point_van_t_Hoff") == "two_point_van_t_Hoff"
    assert normalise_q_st_method("calorimetric_low_loading") == "calorimetric_low_loading"


def test_normalise_recognises_aliases_and_verbose_sources():
    assert normalise_q_st_method("DIRECT calorimetric (Tian-Calvet)") == "calorimetric_low_loading"
    assert normalise_q_st_method("Cmarik_2012_vant_Hoff_298_to_318_K") == "two_point_van_t_Hoff"
    assert normalise_q_st_method("Mason_2011 Tian Calvet microcalorimetric") == "calorimetric_low_loading"
    assert normalise_q_st_method("Maghsoudi_2013_section_4.3_zero_coverage_van_t_Hoff") == "two_point_van_t_Hoff"


def test_compatible_when_methods_match():
    outcome, note = assess_compatibility("direct_widom_boltzmann_weighted", "direct_widom_boltzmann_weighted")
    assert outcome == COMPATIBLE
    assert note == ""


def test_equivalent_pair_two_point_vs_multi_point_van_t_hoff():
    outcome, note = assess_compatibility("two_point_van_t_Hoff", "multi_point_van_t_Hoff_regression")
    assert outcome == COMPATIBLE
    assert note == ""


def test_compatible_note_direct_widom_vs_calorimetric_low_loading():
    """1a / 1b case: atlas (RASPA2 or native direct Widom) vs Mason calorimetric Table 1."""
    outcome, note = assess_compatibility("direct_widom_boltzmann_weighted", "calorimetric_low_loading")
    assert outcome == COMPATIBLE_NOTE
    assert "single-strong-site" in note or "calorimetric" in note.lower()


def test_method_blocked_isosteric_vs_calorimetric():
    """Clausius-Clapeyron and direct calorimetry are different observables."""
    outcome, note = assess_compatibility("isosteric_from_isotherms", "calorimetric_low_loading")
    assert outcome == METHOD_BLOCKED


def test_method_blocked_when_either_method_is_unknown():
    outcome, note = assess_compatibility("some_made_up_method", "calorimetric_low_loading")
    assert outcome == METHOD_BLOCKED


def test_compatible_with_note_van_t_hoff_vs_calorimetric():
    """Common cross-method case (e.g. RASPA3 emits van't Hoff, ref is Tian-Calvet)."""
    outcome, _ = assess_compatibility("two_point_van_t_Hoff", "calorimetric_low_loading")
    assert outcome == COMPATIBLE_NOTE
