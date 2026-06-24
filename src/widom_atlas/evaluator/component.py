"""Rigid multi-site gas geometries used by the Widom evaluator.

A ``Component`` is a small frozen dataclass that fully specifies a rigid
gas molecule:

- ``site_offsets`` : (n_sites, 3) array of offsets from the centre of mass,
  in Angstrom
- ``site_labels``  : list of n_sites labels matching UserParameterFile.gas_sites[].label
- ``site_charges`` : (n_sites,) array of partial charges in elementary charge units
- ``site_masses``  : (n_sites,) array of atomic masses in amu
- ``rotational``   : whether the component should be rotated by random Euler
  angles at each insertion (False for Lennard-Jones spheres like Kr/CH4 single-site)

The evaluator's job is to call ``orient_random(rng)`` for rotational
components and read the offsets verbatim for atomic / single-site ones.

Built-in profiles for the v0.4 release gate:
- ``co2_garcia_sanchez_2009`` (CO2, 3 sites, charges +0.6512 / -0.3256)
- ``n2_trappe`` (N2, 3 sites incl. virtual COM, charges -0.482 / 0 / -0.482; CO2-style)
- ``ch4_trappe_united_atom`` (CH4, single LJ centre)
- ``kr_lj`` (Kr, single LJ centre)

This module does NOT bundle RASPA's database — these profiles are
literature constants used downstream of UserParameterFile validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Component:
    """Rigid multi-site gas description."""

    name: str
    site_labels: list[str]
    site_offsets: np.ndarray  # (n_sites, 3), Angstrom
    site_charges: np.ndarray  # (n_sites,), e
    site_masses: np.ndarray   # (n_sites,), amu
    rotational: bool
    notes: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def n_sites(self) -> int:
        return self.site_offsets.shape[0]

    @property
    def total_mass(self) -> float:
        return float(self.site_masses.sum())

    def __post_init__(self) -> None:
        n = self.site_offsets.shape[0]
        if not (
            len(self.site_labels) == n
            and self.site_charges.shape == (n,)
            and self.site_masses.shape == (n,)
        ):
            raise ValueError(
                f"Component {self.name}: shape mismatch n_sites={n} but "
                f"labels={len(self.site_labels)} charges={self.site_charges.shape} masses={self.site_masses.shape}"
            )
        net = float(self.site_charges.sum())
        if abs(net) > 1e-6:
            object.__setattr__(self, "warnings", [*self.warnings, f"non-neutral component net charge {net:+.4e} e"])


def _zero(n: int) -> np.ndarray:
    return np.zeros(n, dtype=float)


def co2_garcia_sanchez_2009() -> Component:
    """García-Sánchez 2009 CO2 (3-site rigid linear): charges +0.6512 / -0.3256.

    C-O distance = 1.149 A, total mass ≈ 44.01 amu, linear along z-axis at origin.
    """
    d = 1.149
    offsets = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, +d],
        [0.0, 0.0, -d],
    ])
    return Component(
        name="CO2",
        site_labels=["C_co2", "O_co2", "O_co2"],
        site_offsets=offsets,
        site_charges=np.array([+0.6512, -0.3256, -0.3256]),
        site_masses=np.array([12.011, 15.999, 15.999]),
        rotational=True,
        notes="García-Sánchez et al., J. Phys. Chem. C 2009, 113, 8814 (TraPPE-style)",
    )


def n2_trappe() -> Component:
    """TraPPE-N2 (Potoff-Siepmann 2001): two N atoms at +/- 0.55 A from COM,
    plus a massless COM virtual site carrying +0.964 e; N atoms each carry -0.482 e.
    """
    offsets = np.array([
        [0.0, 0.0, +0.55],
        [0.0, 0.0, -0.55],
        [0.0, 0.0, 0.0],   # COM virtual
    ])
    return Component(
        name="N2",
        site_labels=["N_n2", "N_n2", "N_com"],
        site_offsets=offsets,
        site_charges=np.array([-0.482, -0.482, +0.964]),
        site_masses=np.array([14.007, 14.007, 0.0]),
        rotational=True,
        notes="TraPPE-N2 (Potoff & Siepmann, AIChE J. 2001) — N_com is a charge-only virtual site",
    )


def ch4_trappe_ua() -> Component:
    """TraPPE-united-atom CH4: single LJ centre, no charge, no rotation."""
    return Component(
        name="CH4",
        site_labels=["CH4_ua"],
        site_offsets=np.zeros((1, 3)),
        site_charges=_zero(1),
        site_masses=np.array([16.043]),
        rotational=False,
        notes="TraPPE-UA single Lennard-Jones centre",
    )


def kr_lj() -> Component:
    """Single LJ centre for Kr."""
    return Component(
        name="Kr",
        site_labels=["Kr"],
        site_offsets=np.zeros((1, 3)),
        site_charges=_zero(1),
        site_masses=np.array([83.798]),
        rotational=False,
        notes="Kr single Lennard-Jones centre",
    )


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed 3x3 rotation matrix (Shoemake 1992 quaternion method)."""
    u1, u2, u3 = rng.random(3)
    q0 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q1 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    return np.array([
        [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q3 * q0), 2 * (q1 * q3 + q2 * q0)],
        [2 * (q1 * q2 + q3 * q0), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q1 * q0)],
        [2 * (q1 * q3 - q2 * q0), 2 * (q2 * q3 + q1 * q0), 1 - 2 * (q1 * q1 + q2 * q2)],
    ])


def orient(component: Component, rng: np.random.Generator) -> np.ndarray:
    """Return rotated site offsets for a single insertion."""
    if not component.rotational:
        return component.site_offsets.copy()
    R = random_rotation_matrix(rng)
    return component.site_offsets @ R.T


__all__ = [
    "Component",
    "ch4_trappe_ua",
    "co2_garcia_sanchez_2009",
    "kr_lj",
    "n2_trappe",
    "orient",
    "random_rotation_matrix",
]
