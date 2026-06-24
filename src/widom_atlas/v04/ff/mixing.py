"""T008: Lorentz-Berthelot mixing helper (LJ_12_6 only).

For unlike-pair LJ_12_6 cross-terms:
    sigma_ij  = (sigma_i + sigma_j) / 2
    epsilon_ij = sqrt(epsilon_i * epsilon_j)

This helper is restricted to LJ_12_6 pairs. Buckingham/Dzubak pairs must
provide explicit cross-terms; Lorentz-Berthelot is not defined for them.
"""
from __future__ import annotations

import math

from .terms import LJ126


def lorentz_berthelot(a: LJ126, b: LJ126) -> LJ126:
    sigma_ij = 0.5 * (a.sigma_angstrom + b.sigma_angstrom)
    eps_ij = math.sqrt(a.epsilon_K * b.epsilon_K)
    return LJ126(epsilon_K=eps_ij, sigma_angstrom=sigma_ij)
