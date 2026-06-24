"""T009: unit conversion helpers.

Canonical internal units:
- Energy:  Kelvin (K)
- Energy:  kJ/mol  (paper-facing, Q_st)
- Length:  Angstrom (Å)
- Charge:  electron charges (e)
- Henry:   mol/(kg*Pa) internally; convertible to mol/(kg*bar), mol/(kg*kPa)

Additional helpers below support reconstructing literature Henry coefficients
from raw experimental units (cm³(STP)·g⁻¹·atm⁻¹, mol·kg⁻¹·atm⁻¹), with the
van't Hoff temperature correction. These are deliberately verbose so that a
unit-audit test can verify each conversion step in isolation.
"""
from __future__ import annotations

import math

R_GAS_CONSTANT_J_PER_MOL_PER_K = 8.314462618  # CODATA 2022
R_GAS_CONSTANT_KJ_PER_MOL_PER_K = R_GAS_CONSTANT_J_PER_MOL_PER_K / 1000.0
KB_K_PER_KJ_PER_MOL = 1000.0 / R_GAS_CONSTANT_J_PER_MOL_PER_K  # 120.272...
PA_PER_BAR = 100000.0
PA_PER_KPA = 1000.0
PA_PER_ATM = 101325.0
ATM_PER_BAR = PA_PER_BAR / PA_PER_ATM  # ≈ 0.986923  (1 bar in atm)
BAR_PER_ATM = PA_PER_ATM / PA_PER_BAR  # ≈ 1.013250  (1 atm in bar)
T_STP_K = 273.15
V_M_STP_M3_PER_MOL = R_GAS_CONSTANT_J_PER_MOL_PER_K * T_STP_K / PA_PER_ATM  # ≈ 0.022414
V_M_STP_CM3_PER_MOL = V_M_STP_M3_PER_MOL * 1e6                              # ≈ 22413.96
MOL_PER_CM3_STP = 1.0 / V_M_STP_CM3_PER_MOL                                  # ≈ 4.4615e-5


def energy_kjmol_to_K(value_kjmol: float) -> float:
    """kJ/mol → K (per Boltzmann constant + Avogadro's number)."""
    return value_kjmol * KB_K_PER_KJ_PER_MOL


def energy_K_to_kjmol(value_K: float) -> float:
    """K → kJ/mol."""
    return value_K / KB_K_PER_KJ_PER_MOL


def KH_mol_per_kg_per_bar_to_mol_per_kg_per_Pa(value_kg_bar: float) -> float:
    return value_kg_bar / PA_PER_BAR


def KH_mol_per_kg_per_Pa_to_mol_per_kg_per_bar(value_kg_Pa: float) -> float:
    return value_kg_Pa * PA_PER_BAR


def KH_mol_per_kg_per_kPa_to_mol_per_kg_per_bar(value_kg_kPa: float) -> float:
    return value_kg_kPa * (PA_PER_BAR / PA_PER_KPA)


def KH_mol_per_kg_per_kPa_to_mol_per_kg_per_Pa(value_kg_kPa: float) -> float:
    return value_kg_kPa / PA_PER_KPA


def KH_mol_per_kg_per_atm_to_mol_per_kg_per_bar(value_kg_atm: float) -> float:
    """1 atm = 1.01325 bar → multiply by atm/bar to get per-bar value."""
    return value_kg_atm / BAR_PER_ATM


def KH_cm3STP_per_g_per_atm_to_mol_per_kg_per_bar(B_cm3STP_g_atm: float) -> float:
    """Forensic helper: convert a literature B (second virial / Henry slope)
    reported in cm³(STP)·g⁻¹·atm⁻¹ to mol·kg⁻¹·bar⁻¹.

    Pipeline:
      cm³(STP)/g/atm  ×  (1 mol / 22413.96 cm³(STP))         → mol/g/atm
                      ×  1000 g/kg                            → mol/kg/atm
                      ×  1 atm / 1.01325 bar                  → mol/kg/bar
    """
    return B_cm3STP_g_atm * MOL_PER_CM3_STP * 1000.0 / BAR_PER_ATM


def vant_hoff_KH_correction(
    K_H_at_T1: float, T1_K: float, T2_K: float, Q_st_kJ_per_mol: float
) -> float:
    """K_H(T2) = K_H(T1) · exp[Q_st/R · (1/T2 - 1/T1)].

    Sign convention: Q_st > 0 for exothermic adsorption (standard chemistry
    convention). For T2 < T1, K_H(T2) > K_H(T1).
    """
    return K_H_at_T1 * math.exp(
        Q_st_kJ_per_mol / R_GAS_CONSTANT_KJ_PER_MOL_PER_K * (1.0 / T2_K - 1.0 / T1_K)
    )


def positive_exothermic_Qads(signed_Q_kjmol: float) -> float:
    """Convert a thermodynamic ΔH (negative for exothermic adsorption) to
    positive-exothermic-magnitude convention used throughout v0.4 (Q_ads > 0
    for binding)."""
    return abs(signed_Q_kjmol)
