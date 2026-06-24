"""Wolf summation Coulomb pair-potential (numpy-only, no FFTs, no extra deps).

This module is the electrostatic complement to
:mod:`widom_atlas.backends.parameterised_lj`. Adding a real-space Wolf sum
on top of the existing LJ NeighborList loop turns the package's calculator
from "blind to charges" (the §A.6 finding in
``BACKEND_FAILURE_ANALYSIS.md``) into a genuine Coulomb-LJ calculator that
can reproduce charge-quadrupole / charge-induced-dipole binding when the
operator supplies partial charges.

Wolf summation reference
========================

Wolf, Keblinski, Phillpot, Eggebrecht. *J. Chem. Phys.* **110**, 8254 (1999).
DOI ``10.1063/1.478738``. The damped shifted-force form used here is

.. math::

   E_\\text{Wolf}(r_{ij}) = \\frac{1}{4\\pi\\epsilon_0} q_i q_j
   \\left( \\frac{\\mathrm{erfc}(\\alpha r_{ij})}{r_{ij}}
          - \\frac{\\mathrm{erfc}(\\alpha r_c)}{r_c} \\right)

with damping parameter ``alpha`` (default 0.20 Å⁻¹) and cutoff ``r_c``.
Each atom additionally carries a self-interaction correction term

.. math::

   E_\\text{self}(i) = -\\frac{q_i^2}{4\\pi\\epsilon_0}
                       \\left( \\frac{\\mathrm{erfc}(\\alpha r_c)}{2 r_c}
                               + \\frac{\\alpha}{\\sqrt{\\pi}} \\right)

The Wolf form is approximate but converges to within ~0.1 — 1 kJ/mol per
insertion at cutoff = 12 Å for typical MOF cells, which is sufficient for
Henry-coefficient screening. Operators who need higher accuracy can wire
the existing :class:`widom_atlas.backends.external.ExternalSamplesBackend`
to a RASPA3 / OpenMM run (Ewald / PME).

Units
=====

Internal unit convention is **eV for energy, Å for length**, matching the
rest of widom-atlas. The Coulomb prefactor in those units is

.. math::

   k_C = \\frac{1}{4\\pi\\epsilon_0} \\;\\to\\; 14.399645478\\;\\mathrm{eV\\,\\AA / e^2}

(CODATA 2018; verified against ``scipy.constants`` —
``e**2/(4*pi*eps_0)`` in eV·Å).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.special import erfc

# CODATA 2018 — k_C = e²/(4π ε₀) in eV·Å when q is in elementary charges.
COULOMB_PREFACTOR_EV_A: Final[float] = 14.399645478


@dataclass(frozen=True)
class WolfParameters:
    """Damping + cutoff for the Wolf real-space sum."""

    alpha_inv_A: float = 0.20  # damping in Å⁻¹; typical 0.10 — 0.30
    cutoff_A: float = 12.0  # real-space cutoff in Å


def wolf_pair_energy(
    r: np.ndarray,
    q_i: np.ndarray,
    q_j: np.ndarray,
    params: WolfParameters,
) -> np.ndarray:
    """Damped shifted Wolf pair potential, vectorised over arrays.

    All arrays must have the same length. ``r`` is the pair separation in
    Å; ``q_i, q_j`` are partial charges in elementary charges. Returns the
    pair energy in eV per pair.
    """
    rc = params.cutoff_A
    alpha = params.alpha_inv_A
    erfc_rc = float(erfc(alpha * rc) / rc)
    pair_term = erfc(alpha * r) / r - erfc_rc
    return COULOMB_PREFACTOR_EV_A * q_i * q_j * pair_term


def wolf_self_energy(charges: np.ndarray, params: WolfParameters) -> float:
    """Wolf self-correction summed over all atoms in a cell.

    Used by the calling calculator to subtract a per-atom self-term so the
    pair sum gives the correct interaction energy.
    """
    rc = params.cutoff_A
    alpha = params.alpha_inv_A
    inv_term = float(erfc(alpha * rc) / (2.0 * rc) + alpha / math.sqrt(math.pi))
    q = np.asarray(charges, dtype=np.float64)
    return float(-COULOMB_PREFACTOR_EV_A * (q * q).sum() * inv_term)


def cross_only_wolf_energy(
    positions: np.ndarray,
    charges: np.ndarray,
    tags: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    params: WolfParameters,
    framework_tag: int,
    gas_tag: int,
) -> float:
    """Inter-tag Wolf interaction energy (framework × gas pairs only).

    Mirrors the LJ inter-tag pattern in
    :func:`widom_atlas.backends.parameterised_lj.ParameterisedLJCalculator._interaction_energy`
    so the two contributions can be added in the same calculator. Per-tag
    self-energies cancel because we are computing an *interaction* energy,
    not a total energy.
    """
    n = len(positions)
    if n < 2 or cell is None:
        return 0.0
    pos = np.asarray(positions, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    cell_arr = np.asarray(cell, dtype=np.float64)
    fw_idx = np.flatnonzero(tags == framework_tag)
    gas_idx = np.flatnonzero(tags == gas_tag)
    if fw_idx.size == 0 or gas_idx.size == 0:
        return 0.0

    rc2 = params.cutoff_A * params.cutoff_A
    inv_cell = np.linalg.inv(cell_arr)

    e_total = 0.0
    for j in gas_idx:
        if q[j] == 0.0:
            continue
        rel = pos[fw_idx] - pos[j]  # (n_fw, 3)
        # Minimum image
        if any(pbc):
            frac = rel @ inv_cell
            frac -= np.round(frac)
            rel = frac @ cell_arr
        r2 = np.einsum("ij,ij->i", rel, rel)
        mask = (r2 > 1e-12) & (r2 < rc2) & (q[fw_idx] != 0.0)
        if not mask.any():
            continue
        r = np.sqrt(r2[mask])
        q_fw = q[fw_idx[mask]]
        e_pair = wolf_pair_energy(r, q_fw, np.full_like(q_fw, q[j]), params)
        e_total += float(e_pair.sum())
    return e_total


__all__ = [
    "COULOMB_PREFACTOR_EV_A",
    "WolfParameters",
    "cross_only_wolf_energy",
    "wolf_pair_energy",
    "wolf_self_energy",
]
