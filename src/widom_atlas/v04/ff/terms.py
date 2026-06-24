"""T004: typed FFTerm dataclasses for the 4 supported pair-interaction kinds.

LJ_12_6:                U(r) = 4*epsilon * ((sigma/r)^12 - (sigma/r)^6)
BUCKINGHAM_A_EXP_C6:    U(r) = A * exp(-B*r) - S_g * C / r^6   (Lin/Mercado Model 4)
DZUBAK_A_EXP_C5_D6:     U(r) = A * exp(-B*r) - C5 / r^5 - D6 / r^6   (Dzubak 2012)
HARD_SPHERE:            U(r) = +inf for r < r_cut else 0
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class FunctionalForm(StrEnum):
    LJ_12_6 = "LJ_12_6"
    BUCKINGHAM_A_EXP_C6 = "BUCKINGHAM_A_EXP_C6"
    DZUBAK_A_EXP_C5_D6 = "DZUBAK_A_EXP_C5_D6"
    HARD_SPHERE = "HARD_SPHERE"


@dataclass(frozen=True)
class LJ126:
    """LJ 12-6 in Kelvin / Angstrom units."""
    epsilon_K: float
    sigma_angstrom: float

    def energy(self, r: float) -> float:
        sr6 = (self.sigma_angstrom / r) ** 6
        return 4.0 * self.epsilon_K * (sr6 * sr6 - sr6)

    @property
    def form(self) -> FunctionalForm:
        return FunctionalForm.LJ_12_6


@dataclass(frozen=True)
class BuckinghamLinMercado:
    """Lin/Mercado modified Buckingham: U(r) = A * exp(-B*r) - S_g * C / r^6.

    `A` in K, `B` in 1/Å, `C` is the dispersion coefficient in K*Å^6.

    Per professor blocker 5: Mercado tables already store C × S_g (pre-scaled).
    The implementation must NOT re-multiply when C_already_scaled=True.
    """
    A_K: float
    B_per_angstrom: float
    C_K_angstrom6: float
    S_g: float = 1.0
    C_already_scaled: bool = True

    def energy(self, r: float) -> float:
        c_eff = self.C_K_angstrom6 if self.C_already_scaled else self.S_g * self.C_K_angstrom6
        return self.A_K * math.exp(-self.B_per_angstrom * r) - c_eff / (r ** 6)

    @property
    def form(self) -> FunctionalForm:
        return FunctionalForm.BUCKINGHAM_A_EXP_C6


@dataclass(frozen=True)
class DzubakAExpC5D6:
    """Dzubak 2012 two-attraction form: U(r) = A * exp(-B*r) - C5/r^5 - D6/r^6.

    `A` in K, `B` in 1/Å, `C5` in K*Å^5, `D6` in K*Å^6.
    """
    A_K: float
    B_per_angstrom: float
    C5_K_angstrom5: float
    D6_K_angstrom6: float

    def energy(self, r: float) -> float:
        return (
            self.A_K * math.exp(-self.B_per_angstrom * r)
            - self.C5_K_angstrom5 / (r ** 5)
            - self.D6_K_angstrom6 / (r ** 6)
        )

    @property
    def form(self) -> FunctionalForm:
        return FunctionalForm.DZUBAK_A_EXP_C5_D6


@dataclass(frozen=True)
class HardSphere:
    r_cut_angstrom: float

    def energy(self, r: float) -> float:
        return math.inf if r < self.r_cut_angstrom else 0.0

    @property
    def form(self) -> FunctionalForm:
        return FunctionalForm.HARD_SPHERE


# Union type for the typed pair table
FFTerm = LJ126 | BuckinghamLinMercado | DzubakAExpC5D6 | HardSphere
