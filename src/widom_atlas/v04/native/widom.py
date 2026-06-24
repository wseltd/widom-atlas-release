"""Widom insertion accumulator with log-sum-exp.

Inserts a rigid probe molecule at random positions+orientations inside a
periodic framework, accumulates the Boltzmann factors of the interaction
energies, and returns the Henry coefficient K_H plus an estimate of the
isosteric heat Q_st via the Boltzmann-weighted <U_gh> minus the
ideal-gas reference.

Numerical convention: instead of summing `exp(-βU)` directly, we keep a
running log-sum-exp accumulator. This avoids both (1) overflow when an
insertion lands in a strongly attractive site (U very negative, so
exp(-βU) is astronomically large) and (2) underflow when most insertions
hit core overlap (U → +∞).

K_H in mol/(kg·Pa) follows the same convention as RASPA2/RASPA3:

    K_H = β / M_framework_kg × <exp(-βU)>

with the average taken over random insertions inside the simulation cell
(uniform in r, uniform in orientation). The Boltzmann-weighted U is the
mean interaction energy of the "successful" insertions, used to compute
Q_st via:

    Q_st = -<U>_Boltzmann - k_B·T

(positive-exothermic convention).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

K_B_J_PER_K = 1.380649e-23
N_AVOGADRO = 6.02214076e23
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27


@dataclass
class WidomAccumulator:
    """Streaming log-sum-exp accumulator for Widom insertions.

    Tracks two quantities:

      log Z  = log <exp(-βU)>      (Henry-coefficient prefactor)
      <U>_B  = <U·exp(-βU)> / Z    (Boltzmann-weighted U; gives Q_st)

    Updated via numerically stable log-sum-exp recurrence.
    """

    n_samples: int = 0
    # Running max of -βU (for log-sum-exp shift); start at -inf
    max_log_w: float = -math.inf
    # Sum of exp(-βU - max_log_w) — the "shifted" Z
    Z_shifted: float = 0.0
    # Sum of U · exp(-βU - max_log_w) — the "shifted" Boltzmann numerator for <U>
    U_shifted_numer: float = 0.0
    # Variance accumulator
    Z_shifted_sq: float = 0.0

    def update(self, U_arr_K: np.ndarray, beta_inv_K: float) -> None:
        """Update accumulator with a batch of insertion energies in K.

        U_arr_K: 1D array of insertion energies in K (= U/k_B with k_B=1).
        beta_inv_K: temperature in K (β = 1/T in units where k_B=1).
        """
        if U_arr_K.size == 0:
            return
        log_w = -U_arr_K / beta_inv_K
        # Only keep finite entries (rejection of insertion overlaps)
        log_w_finite = log_w[np.isfinite(log_w)]
        if log_w_finite.size == 0:
            self.n_samples += int(U_arr_K.size)
            return

        batch_max = float(np.max(log_w_finite))
        new_max = max(self.max_log_w, batch_max)
        if math.isfinite(self.max_log_w) and new_max != self.max_log_w:
            shift = self.max_log_w - new_max
            self.Z_shifted *= math.exp(shift)
            self.Z_shifted_sq *= math.exp(2 * shift)
            self.U_shifted_numer *= math.exp(shift)
        self.max_log_w = new_max

        w = np.exp(log_w_finite - new_max)
        # The W array is bounded ≤ 1 by construction.
        self.Z_shifted += float(np.sum(w))
        self.Z_shifted_sq += float(np.sum(w * w))
        # For Q_st we need <U exp(-βU)>; use U values (in K) on the same finite entries.
        u_finite = U_arr_K[np.isfinite(log_w)]
        self.U_shifted_numer += float(np.sum(u_finite * w))
        self.n_samples += int(U_arr_K.size)

    def mean_boltzmann_factor(self) -> float:
        """<exp(-βU)> over all insertion attempts (including overlaps).

        Returns +inf if the max log-Boltzmann shift is too large for math.exp
        (i.e. some insertion landed in a Coulomb-singularity well below the
        hard-core; the runner should be guarding against this with an
        electrostatic-hard-core check).
        """
        if self.n_samples == 0 or not math.isfinite(self.max_log_w):
            return 0.0
        if self.max_log_w > 700.0:
            return float("inf")
        return math.exp(self.max_log_w) * self.Z_shifted / self.n_samples

    def mean_boltzmann_U_K(self) -> float:
        """Boltzmann-weighted <U> in K. Returns 0 if no successful insertions."""
        if self.Z_shifted <= 0.0:
            return 0.0
        return self.U_shifted_numer / self.Z_shifted

    def K_H_mol_per_kg_per_Pa(
        self, T_K: float, M_framework_kg: float, V_supercell_m3: float,
    ) -> float:
        """Henry coefficient: K_H = <e^{-βU}> × V / (M · R · T).

        Derivation from N = K_H · P · M_framework (mol of adsorbate per
        kg of framework) and the ideal-gas access limit N = P · V_pore /
        (R · T): the Widom Henry constant is ⟨exp(-βU)⟩ times the
        ideal-gas Henry constant V / (M · R · T). All quantities SI; the
        Boltzmann-weighted ⟨exp(-βU)⟩ is dimensionless.
        """
        Z = self.mean_boltzmann_factor()
        if Z <= 0.0:
            return 0.0
        R = K_B_J_PER_K * N_AVOGADRO  # 8.314 J/(mol·K)
        return Z * V_supercell_m3 / (M_framework_kg * R * T_K)

    def Q_st_kJ_per_mol(self, T_K: float, U_gas_K: float = 0.0) -> float:
        """Isosteric heat via Widom: Q_st = -<U_gh>_B + RT - <U_g>.

        Standard Widom convention (matches RASPA2/RASPA3 output):
            dH_ads = <U_gh>_1 - <U_h>_0 - <U_g> - RT          (NEGATIVE if exothermic)
            Q_st   = -dH_ads = -<U_gh>_B + <U_g> + RT          (POSITIVE if exothermic)

        With rigid Ar / CH4 / CO2 in the ideal-gas reference, <U_g> = 0,
        so this reduces to Q_st = -<U_gh>_B + RT. Output in kJ/mol.
        """
        u_K = self.mean_boltzmann_U_K()
        RT_K = T_K
        return float(-u_K + RT_K - U_gas_K) * 1e-3 * K_B_J_PER_K * N_AVOGADRO


def framework_mass_kg(types: list[str], type_to_mass_amu: dict[str, float]) -> float:
    """Total framework mass in kg, given a per-atom type list."""
    total = 0.0
    for t in types:
        m = type_to_mass_amu.get(t)
        if m is None:
            raise KeyError(f"no atomic mass for framework type {t!r}")
        total += float(m)
    return total * ATOMIC_MASS_UNIT_KG
