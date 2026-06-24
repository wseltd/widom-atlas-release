"""Tests for RobustnessMetrics (T010)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from widom_atlas.core.models import RobustnessMetrics


def _kw(**overrides) -> dict:
    base = {
        "delta_ln_KH": -0.05,
        "delta_Qads_kJmol": -0.4,
        "basin_count_pristine": 4,
        "basin_count_perturbed": 5,
        "basin_count_change": 1,
        "basin_persistence_fraction": 0.8,
        "basin_splitting_count": 1,
        "mean_basin_displacement_A": 0.42,
        "accessibility_change": 0.0,
        "ambiguity_flags": [],
        "missing_data_flags": [],
    }
    base.update(overrides)
    return base


def test_robustness_metrics_accepts_missing_kh_and_qads() -> None:
    m = RobustnessMetrics(**_kw(delta_ln_KH=None, delta_Qads_kJmol=None))
    assert m.delta_ln_KH is None and m.delta_Qads_kJmol is None


def test_robustness_metrics_basin_count_change_consistency() -> None:
    with pytest.raises(ValidationError):
        RobustnessMetrics(**_kw(basin_count_change=99))


def test_robustness_metrics_persistence_fraction_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        RobustnessMetrics(**_kw(basin_persistence_fraction=1.5))
    with pytest.raises(ValidationError):
        RobustnessMetrics(**_kw(basin_persistence_fraction=-0.1))


def test_robustness_metrics_accessibility_change_range() -> None:
    with pytest.raises(ValidationError):
        RobustnessMetrics(**_kw(accessibility_change=2.0))
    with pytest.raises(ValidationError):
        RobustnessMetrics(**_kw(accessibility_change=-2.0))


def test_robustness_metrics_negative_displacement_rejected() -> None:
    with pytest.raises(ValidationError):
        RobustnessMetrics(**_kw(mean_basin_displacement_A=-0.01))


def test_robustness_metrics_missing_data_flag_auto_populated() -> None:
    m = RobustnessMetrics(**_kw(delta_ln_KH=None, delta_Qads_kJmol=None))
    assert "KH_unavailable" in m.missing_data_flags
    assert "Qads_unavailable" in m.missing_data_flags
