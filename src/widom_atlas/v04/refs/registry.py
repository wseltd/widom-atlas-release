"""T030: Literature reference registry.

Provides a single lookup table per branch_id → LiteratureReference. All
values come from the locked case matrix YAML (single source of truth);
this module merely formalises access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LiteratureReference:
    branch_id: str
    case_id: str
    K_H_value: float | None
    K_H_units: str
    K_H_window_min: float | None
    K_H_window_max: float | None
    K_H_source: str
    K_H_doi: str
    Q_st_value: float | None
    Q_st_low_loading: float | None
    Q_st_high_loading: float | None
    Q_st_units: str
    Q_st_window_min: float | None
    Q_st_window_max: float | None
    Q_st_source: str
    Q_st_doi: str
    temperature_K: float
    verdict_tier: str
    numeric_thresholds: str


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_registry(case_matrix: dict) -> dict[str, LiteratureReference]:
    out: dict[str, LiteratureReference] = {}
    for case in case_matrix.get("cases", []):
        for branch in case.get("branches", []):
            bid = branch["branch_id"]
            refs = branch.get("references") or {}
            kh = refs.get("K_H") or {}
            qst = refs.get("Q_st") or {}
            out[bid] = LiteratureReference(
                branch_id=bid,
                case_id=case["case_id"],
                K_H_value=_f(kh.get("value")),
                K_H_units=str(kh.get("units", "")),
                K_H_window_min=_f(kh.get("acceptance_window_min")),
                K_H_window_max=_f(kh.get("acceptance_window_max")),
                K_H_source=str(kh.get("source", "")),
                K_H_doi=str(kh.get("primary_doi") or kh.get("source_doi", "")),
                Q_st_value=_f(qst.get("value")),
                Q_st_low_loading=_f(qst.get("low_loading_value")),
                Q_st_high_loading=_f(qst.get("high_loading_value")),
                Q_st_units=str(qst.get("units", "")),
                Q_st_window_min=_f(qst.get("acceptance_window_min")),
                Q_st_window_max=_f(qst.get("acceptance_window_max")),
                Q_st_source=str(qst.get("source", "")),
                Q_st_doi=str(qst.get("primary_doi") or qst.get("source_doi", "")),
                temperature_K=_f(kh.get("temperature_K")) or 298.0,
                verdict_tier=str(branch.get("verdict_tier", "")),
                numeric_thresholds=str(branch.get("numeric_thresholds", "")),
            )
    return out
