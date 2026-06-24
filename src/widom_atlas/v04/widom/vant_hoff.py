"""Q_st via van't Hoff slope of ln K_H vs 1/T.

RASPA3 v3.0.29 does not expose ⟨U⟩_Widom directly in its Widom output; the
canonical-ensemble identity Q_st = -R · d(ln K_H)/d(1/T) recovers Q_st from
two (or more) Henry-coefficient runs at different temperatures.

This is the documented Widom-based Q_st estimator (Frenkel & Smit,
"Understanding Molecular Simulation", 2nd ed., §10.2.1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_R_KJ_MOL_K = 8.314462618e-3  # kJ / (mol K)


@dataclass(frozen=True)
class VantHoffResult:
    Q_st_kJ_per_mol: float
    Q_st_uncertainty_kJ_per_mol: float | None
    T_low_K: float
    T_high_K: float
    K_H_low: float
    K_H_high: float
    method: str  # "two_point_van_t_Hoff"


def vant_hoff_two_point(
    K_H_low: float, T_low_K: float, K_H_high: float, T_high_K: float,
    K_H_low_unc: float | None = None, K_H_high_unc: float | None = None,
) -> VantHoffResult:
    """Q_st = R · (ln K_low - ln K_high) / (1/T_low - 1/T_high)

    Sign convention: K_H decreases with rising T (adsorption is exothermic),
    so ln(K_low/K_high) > 0 when T_low < T_high, and 1/T_low - 1/T_high > 0,
    so Q_st > 0 (positive-exothermic).

    K_H must be in the same arbitrary units for both points.
    """
    if K_H_low <= 0 or K_H_high <= 0:
        raise ValueError("Q_st van't Hoff requires positive K_H")
    if T_low_K >= T_high_K:
        T_low_K, T_high_K = T_high_K, T_low_K
        K_H_low, K_H_high = K_H_high, K_H_low
        K_H_low_unc, K_H_high_unc = K_H_high_unc, K_H_low_unc
    slope = (math.log(K_H_low) - math.log(K_H_high)) / (1.0 / T_low_K - 1.0 / T_high_K)
    Q_st = _R_KJ_MOL_K * slope
    unc: float | None
    if K_H_low_unc is not None and K_H_high_unc is not None:
        d_low = K_H_low_unc / K_H_low
        d_high = K_H_high_unc / K_H_high
        d_slope = math.sqrt(d_low**2 + d_high**2) / abs(1.0 / T_low_K - 1.0 / T_high_K)
        unc = _R_KJ_MOL_K * d_slope
    else:
        unc = None
    return VantHoffResult(
        Q_st_kJ_per_mol=Q_st,
        Q_st_uncertainty_kJ_per_mol=unc,
        T_low_K=T_low_K,
        T_high_K=T_high_K,
        K_H_low=K_H_low,
        K_H_high=K_H_high,
        method="two_point_van_t_Hoff",
    )
