"""System representation for the native Widom evaluator.

A `NativeSystem` bundles:

  - a periodic framework: per-atom (type, fractional coordinate) + 3×3 cell matrix
  - a typed pair table mapping (type_a, type_b) → :class:`PairPotential`
  - a per-type mass map (used to compute the total framework mass for K_H)
  - a probe molecule: rigid set of (type, body-frame position, partial charge)

The system is the "physics" payload; the runner (:mod:`runner`) drives random
insertions, evaluates the framework-probe interaction energy via the pair table
(direct-space cutoff only; an Ewald addition is left as a follow-on), and
streams the energies into a :class:`WidomAccumulator`.

All coordinates are Cartesian Å; energies are K (k_B·T at T = 1 K is 1 K).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .potentials import PairTable


@dataclass(frozen=True)
class ProbeMolecule:
    """Rigid probe molecule used for the Widom insertion.

    `body_positions` are the per-atom positions in the molecule's own body
    frame in Å (centroid at origin). Orientation during insertion is sampled
    uniformly on SO(3).
    """

    name: str
    types: list[str]
    body_positions: np.ndarray  # (n_atoms, 3) in Å
    charges_e: np.ndarray | None = None  # optional partial charges in e

    def n_atoms(self) -> int:
        return len(self.types)


@dataclass
class NativeSystem:
    """A native-evaluator system: periodic framework + pair potentials + probe."""

    framework_types: list[str]
    framework_cart_angstrom: np.ndarray  # (N_atoms, 3)
    framework_charges_e: np.ndarray | None
    cell_matrix_angstrom: np.ndarray  # (3, 3)
    pair_table: PairTable
    probe: ProbeMolecule
    type_to_mass_amu: dict[str, float]
    supercell_replicas: tuple[int, int, int] = (1, 1, 1)
    energy_cutoff_angstrom: float = 12.8

    n_framework_atoms: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_framework_atoms = self.framework_cart_angstrom.shape[0]

    def supercell_cell(self) -> np.ndarray:
        """Effective simulation cell after the supercell replication."""
        na, nb, nc = self.supercell_replicas
        return np.array([
            self.cell_matrix_angstrom[0] * na,
            self.cell_matrix_angstrom[1] * nb,
            self.cell_matrix_angstrom[2] * nc,
        ])

    def supercell_positions(self) -> tuple[list[str], np.ndarray]:
        """Replicate the framework atoms according to supercell_replicas."""
        na, nb, nc = self.supercell_replicas
        all_types: list[str] = []
        all_carts: list[np.ndarray] = []
        a_vec = self.cell_matrix_angstrom[0]
        b_vec = self.cell_matrix_angstrom[1]
        c_vec = self.cell_matrix_angstrom[2]
        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    shift = ia * a_vec + ib * b_vec + ic * c_vec
                    all_carts.append(self.framework_cart_angstrom + shift)
                    all_types.extend(self.framework_types)
        return all_types, np.vstack(all_carts)


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix in SO(3) via QR-decomposed Gaussian."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    # Make rotation proper (det = +1)
    d = np.diag(np.sign(np.diag(r)))
    q = q @ d
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def insert_probe_at(
    probe: ProbeMolecule,
    centre_cart: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Return the probe's per-atom Cartesian positions at `centre_cart`."""
    return centre_cart + (probe.body_positions @ rotation.T)
