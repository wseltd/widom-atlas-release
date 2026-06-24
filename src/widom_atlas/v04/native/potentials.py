"""Typed pair potentials for the native Widom evaluator.

Three functional forms are supported, each a separate class. All return
the per-pair energy in K (k_B·T at T = 1 K = 1 K) given a NumPy array of
pair distances in Å. Output is `+inf` for r below the hard-core cutoff
(if any), so the Boltzmann factor evaluates to zero and the insertion is
correctly excluded.

  LennardJones12_6   : V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
  BuckinghamAExpC6   : V(r) = A·exp(-B·r) - C/r^6
  DzubakAExpC5D6     : V(r) = A·exp(-B·r) - C/r^5 - D/r^6

Units:
  ε : K
  σ : Å
  A : K
  B : Å^-1
  C : K·Å^5 (Dzubak) or K·Å^6 (Buckingham)
  D : K·Å^6 (Dzubak only)

The (typed pair) -> potential mapping lives in :class:`PairTable`, which
exposes a single `pair_energy(type_i, type_j, distances_angstrom)` API
so the Widom accumulator stays form-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PairPotential(Protocol):
    """Common interface: takes pair distances in Å, returns energies in K."""

    def energy(self, r_angstrom: np.ndarray) -> np.ndarray: ...

    def cutoff_angstrom(self) -> float: ...


@dataclass(frozen=True)
class LennardJones12_6:
    """Lennard-Jones 12-6 potential with optional shifted-truncated form.

    `shifted = False` (default): pure truncated, V(r > r_cut) = 0 and the
    in-cutoff value is the bare LJ. RASPA3 v3.0.29 calls this convention
    "truncated".

    `shifted = True`: shifted-truncated, V(r) - V(r_cut) is used so the
    potential is continuous at r_cut. RASPA3 v3.0.29 calls this convention
    "shifted_truncated". The YAML's `lj_treatment` field for 6c (MFI+Ar)
    selects this convention; selecting it here makes the native evaluator
    consistent with RASPA3's pair-energy convention.
    """

    epsilon_K: float
    sigma_angstrom: float
    cutoff_A: float = 12.8
    shifted: bool = False

    def energy(self, r_angstrom: np.ndarray) -> np.ndarray:
        # mask out r=0 to avoid divide-by-zero at the framework atom itself
        r = np.where(r_angstrom > 1e-10, r_angstrom, 1e-10)
        sr6 = (self.sigma_angstrom / r) ** 6
        sr12 = sr6 * sr6
        v = 4.0 * self.epsilon_K * (sr12 - sr6)
        if self.shifted:
            sr6_cut = (self.sigma_angstrom / self.cutoff_A) ** 6
            sr12_cut = sr6_cut * sr6_cut
            v_at_cut = 4.0 * self.epsilon_K * (sr12_cut - sr6_cut)
            v = v - v_at_cut
        # Hard wall at r < sigma/4 (only kicks in at unphysical contact); keeps
        # the Widom accumulator finite without affecting binding statistics.
        very_close = r_angstrom < (self.sigma_angstrom / 4.0)
        v = np.where(very_close, np.inf, v)
        # Truncate at cutoff
        v = np.where(r_angstrom > self.cutoff_A, 0.0, v)
        return v

    def cutoff_angstrom(self) -> float:
        return self.cutoff_A


@dataclass(frozen=True)
class BuckinghamAExpC6:
    """V(r) = A·exp(-B·r) - C/r^6 with a short-range hard core."""

    A_K: float
    B_inv_angstrom: float
    C_K_angstrom6: float
    hardcore_angstrom: float = 1.0
    cutoff_A: float = 12.8

    def energy(self, r_angstrom: np.ndarray) -> np.ndarray:
        r = np.where(r_angstrom > 1e-10, r_angstrom, 1e-10)
        v = self.A_K * np.exp(-self.B_inv_angstrom * r) - self.C_K_angstrom6 / (r ** 6)
        # Hard core: any r < hardcore_angstrom rejected.
        v = np.where(r_angstrom < self.hardcore_angstrom, np.inf, v)
        # Truncate at cutoff
        v = np.where(r_angstrom > self.cutoff_A, 0.0, v)
        return v

    def cutoff_angstrom(self) -> float:
        return self.cutoff_A


@dataclass(frozen=True)
class DzubakAExpC5D6:
    """V(r) = A·exp(-B·r) - C/r^5 - D/r^6 (Dzubak 2012 form)."""

    A_K: float
    B_inv_angstrom: float
    C_K_angstrom5: float
    D_K_angstrom6: float
    hardcore_angstrom: float = 1.0
    cutoff_A: float = 12.8

    def energy(self, r_angstrom: np.ndarray) -> np.ndarray:
        r = np.where(r_angstrom > 1e-10, r_angstrom, 1e-10)
        v = (
            self.A_K * np.exp(-self.B_inv_angstrom * r)
            - self.C_K_angstrom5 / (r ** 5)
            - self.D_K_angstrom6 / (r ** 6)
        )
        v = np.where(r_angstrom < self.hardcore_angstrom, np.inf, v)
        v = np.where(r_angstrom > self.cutoff_A, 0.0, v)
        return v

    def cutoff_angstrom(self) -> float:
        return self.cutoff_A


@dataclass(frozen=True)
class OngariAExpC6C8:
    """V(r) = A*exp(-B*r) - C6/r^6 - C8/r^8 — special case of the RASPA "generic"
    potential (RASPA manual equation 3.94) used in Ongari 2017 for HKUST-1
    Cu-O(CO2).

    Per 2026-05-19 pass-8 source-provenance update (operator-supplied via
    Europe PMC supplementaryFiles ZIP route for PMC5523115 -> jp7b02302_si_001.pdf
    SI page S8 section 4), the full RASPA generic equation 3.94 is:

        U(r) = p0 * exp(-p1*r) - p2/r^4 - p3/r^6 - p4/r^8 - p5/r^10

      with units per the RASPA manual excerpt in the Ongari SI:
        p0/k_B in K
        p1     in 1/Angstrom
        p2/k_B in K*Angstrom^4
        p3/k_B in K*Angstrom^6
        p4/k_B in K*Angstrom^8
        p5/k_B in K*Angstrom^10

    For Ongari 2017 Cu-O(CO2) specifically, the SI lists p2 = p5 = 0,
    reducing the form to:

        U(r) = A*exp(-B*r) - C6/r^6 - C8/r^8

    with verified coefficients:
        A  = p0 = 1.0e8 K
        B  = p1 = 4.19 1/Angstrom
        C6 = p3 = 3.196e4 K*Angstrom^6
        C8 = p4 = 5.0e6 K*Angstrom^8

    This class implements that reduced form. The Ongari SI also specifies a
    hard-core rule: for r < 1.8 A, set V = 1.0e15 K. Implemented here via
    hardcore_angstrom=1.8 (returns inf instead of 1e15, semantically
    equivalent for Boltzmann-weighted Widom).

    Future RASPA-generic potentials with non-zero p2 (r^-4 term) or non-zero
    p5 (r^-10 term) would need a more general class. This class is named
    "OngariAExpC6C8" to make the special-case scope explicit.

    Independent verification of the Ongari SI coefficient block was obtained
    pass-8 via Europe PMC supplementaryFiles ZIP for PMC5523115 (operator-
    supplied access route). The 2026-05-19 pass-6 R7 implementation used
    operator-supplied verbatim coefficients which match the Europe PMC SI
    reading exactly — no value change required.
    """

    A_K: float
    B_inv_angstrom: float
    C6_K_angstrom6: float
    C8_K_angstrom8: float
    hardcore_angstrom: float = 1.8
    cutoff_A: float = 13.0

    def energy(self, r_angstrom: np.ndarray) -> np.ndarray:
        r = np.where(r_angstrom > 1e-10, r_angstrom, 1e-10)
        v = (
            self.A_K * np.exp(-self.B_inv_angstrom * r)
            - self.C6_K_angstrom6 / (r ** 6)
            - self.C8_K_angstrom8 / (r ** 8)
        )
        v = np.where(r_angstrom < self.hardcore_angstrom, np.inf, v)
        v = np.where(r_angstrom > self.cutoff_A, 0.0, v)
        return v

    def cutoff_angstrom(self) -> float:
        return self.cutoff_A


@dataclass
class PairTable:
    """A symmetric type-pair → PairPotential lookup table.

    Use `set(type_a, type_b, potential)` and `get(type_a, type_b)`. Order
    of the two type names is irrelevant — the table is symmetric.
    """

    entries: dict[frozenset[str], PairPotential]

    def __init__(self) -> None:
        self.entries = {}

    def set(self, type_a: str, type_b: str, potential: PairPotential) -> None:
        key = frozenset((type_a, type_b)) if type_a != type_b else frozenset((type_a,))
        self.entries[key] = potential

    def get(self, type_a: str, type_b: str) -> PairPotential | None:
        key = frozenset((type_a, type_b)) if type_a != type_b else frozenset((type_a,))
        return self.entries.get(key)

    def pair_energy(self, type_a: str, type_b: str, r_angstrom: np.ndarray) -> np.ndarray:
        pot = self.get(type_a, type_b)
        if pot is None:
            return np.zeros_like(r_angstrom)
        return pot.energy(r_angstrom)

    def max_cutoff_angstrom(self) -> float:
        if not self.entries:
            return 0.0
        return max(p.cutoff_angstrom() for p in self.entries.values())

    def __len__(self) -> int:
        return len(self.entries)
