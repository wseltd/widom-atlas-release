"""T015: Fm-3m → P1 rhombohedral primitive transformer (HKUST-1 case 2a).

The simulation CIF is P1 rhombohedral primitive (a≈18.74 Å, α≈60°).
Wu 2010 site-truth coordinates are in Fm-3m (a≈26.32 Å, α=90°).
To compare site-truth distances against the simulation cell, fractional
coordinates must be transformed between frames.

The transformation is the standard one between Fm-3m (face-centred
cubic, 4 primitive cells per conventional cell) and the rhombohedral
primitive:

    a_p = (a_c/2)*(b_hat + c_hat)
    b_p = (a_c/2)*(a_hat + c_hat)
    c_p = (a_c/2)*(a_hat + b_hat)

The inverse takes fractional rhombohedral coords back to Cartesian, and
the resulting Cartesian point can be expressed in the conventional cubic
frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CellMatrix:
    """3x3 row-vectors a, b, c (Å)."""

    matrix: np.ndarray  # shape (3,3), each ROW is a lattice vector

    @classmethod
    def from_abc_angles(
        cls,
        a: float,
        b: float,
        c: float,
        alpha_deg: float,
        beta_deg: float,
        gamma_deg: float,
    ) -> CellMatrix:
        alpha = np.radians(alpha_deg)
        beta = np.radians(beta_deg)
        gamma = np.radians(gamma_deg)
        ax = a
        ay = 0.0
        az = 0.0
        bx = b * np.cos(gamma)
        by = b * np.sin(gamma)
        bz = 0.0
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
        return cls(np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]]))

    def fractional_to_cartesian(self, frac: np.ndarray) -> np.ndarray:
        return frac @ self.matrix

    def cartesian_to_fractional(self, cart: np.ndarray) -> np.ndarray:
        return cart @ np.linalg.inv(self.matrix)


def fm3m_cell(a_cubic: float) -> CellMatrix:
    return CellMatrix.from_abc_angles(a_cubic, a_cubic, a_cubic, 90.0, 90.0, 90.0)


def rhombohedral_primitive_from_fm3m(a_cubic: float) -> CellMatrix:
    """The rhombohedral primitive cell of an Fm-3m lattice with conventional a_cubic.

    Vectors:
      a_p = (a_c/2)*(0,1,1)
      b_p = (a_c/2)*(1,0,1)
      c_p = (a_c/2)*(1,1,0)
    """
    half = a_cubic / 2.0
    return CellMatrix(
        matrix=np.array(
            [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]]
        )
    )


def fm3m_frac_to_rhombohedral_frac(
    fm3m_frac: np.ndarray, a_cubic: float
) -> np.ndarray:
    """Convert a fractional-Fm3m coordinate to fractional-rhombohedral-primitive."""
    fm3m = fm3m_cell(a_cubic)
    rho = rhombohedral_primitive_from_fm3m(a_cubic)
    cart = fm3m.fractional_to_cartesian(np.atleast_2d(fm3m_frac))
    rho_frac = rho.cartesian_to_fractional(cart)
    return rho_frac.reshape(fm3m_frac.shape)


def rhombohedral_frac_to_fm3m_frac(
    rho_frac: np.ndarray, a_cubic: float
) -> np.ndarray:
    fm3m = fm3m_cell(a_cubic)
    rho = rhombohedral_primitive_from_fm3m(a_cubic)
    cart = rho.fractional_to_cartesian(np.atleast_2d(rho_frac))
    fm3m_frac = fm3m.cartesian_to_fractional(cart)
    return fm3m_frac.reshape(rho_frac.shape)
