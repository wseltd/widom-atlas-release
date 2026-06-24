"""T019: internal Wolf-summation backend (screening/atlas only, never strict).

Wolf summation is a damped truncation approximation to the full Ewald
electrostatic sum. It is FAST and is used only for:
- Atlas-grid energy maps (where exhaustive Widom over a grid would be slow)
- Screening passes that propose hot regions for the Ewald-strict re-evaluation

NEVER used as the strict scalar reference. The locked spec mandates
RASPA3 Ewald for verdict-affecting branches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

COULOMB_KELVIN_PREFACTOR_PER_ANGSTROM = 167101.001  # e^2/(4*pi*eps0*kB*Å) in K·Å


@dataclass(frozen=True)
class WolfConfig:
    alpha_per_angstrom: float  # damping
    cutoff_angstrom: float
    fmt: str = "wolf_v1"


def wolf_pair_energy_K(
    qi: float, qj: float, r: float, cfg: WolfConfig
) -> float:
    """Pair Coulomb energy in Kelvin using Wolf summation.

    U(r) = q_i*q_j * (erfc(alpha*r)/r - erfc(alpha*r_c)/r_c)
    for r <= r_c, else 0.

    Returns 0 outside the cutoff. Returned units: K (after the
    Kelvin/Angstrom Coulomb prefactor).
    """
    if r >= cfg.cutoff_angstrom or r <= 0:
        return 0.0
    a = cfg.alpha_per_angstrom
    rc = cfg.cutoff_angstrom
    term = math.erfc(a * r) / r - math.erfc(a * rc) / rc
    return qi * qj * COULOMB_KELVIN_PREFACTOR_PER_ANGSTROM * term
