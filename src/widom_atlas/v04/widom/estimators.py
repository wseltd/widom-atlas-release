"""T022: Henry coefficient K_H and isosteric heat Q_st estimators from WidomAccumulator."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KHQst:
    K_H_mol_per_kg_per_Pa: float
    Q_st_kJ_per_mol: float
    n_seeds: int
    n_insertions_total: int


def estimate_KH_Qst_from_widom(
    henry_excess_K: float,
    mean_energy_K: float,
    temperature_K: float,
    framework_mass_kg_per_uc: float,
    uc_volume_m3: float,
    n_seeds: int,
    n_insertions_total: int,
) -> KHQst:
    """Compute K_H and Q_st from Widom output.

    Q_st_zero_loading = R*T - <U>_Boltzmann
    """
    from ..units import energy_K_to_kjmol
    from .driver import henry_excess_to_K_H_mol_per_kg_per_Pa

    k_h = henry_excess_to_K_H_mol_per_kg_per_Pa(
        henry_excess_K=henry_excess_K,
        temperature_K=temperature_K,
        framework_mass_kg_per_uc=framework_mass_kg_per_uc,
        uc_volume_m3=uc_volume_m3,
    )
    Q_st_K = temperature_K - mean_energy_K  # zero-coverage approximation
    Q_st_kJmol = abs(energy_K_to_kjmol(Q_st_K))
    return KHQst(
        K_H_mol_per_kg_per_Pa=k_h,
        Q_st_kJ_per_mol=Q_st_kJmol,
        n_seeds=n_seeds,
        n_insertions_total=n_insertions_total,
    )
