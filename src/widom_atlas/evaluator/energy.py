"""Periodic LJ + Wolf-Coulomb energy kernel for the Widom evaluator.

The kernel takes:

- a fixed framework (positions, types) in a cell
- a single insertion (centre + rotated site offsets, gas types)

and returns ``(E_LJ_K, E_coul_K)`` in Kelvin (RASPA's native energy unit
for force-fields). Heavy lifting:

LJ
--
12-6 with shifted-and-truncated cutoff at ``r_cut`` (default 12 A or
``cell.min_face_distance / 2`` whichever is smaller, applying the
minimum-image convention). No tail correction because Widom insertions
are at infinite dilution.

Coulomb (Wolf 1999)
-------------------
Real-space damped sum:

  V_wolf(r) = q_i q_j * [erfc(α r) / r - erfc(α r_cut) / r_cut
                        + (erfc(α r_cut) / r_cut^2 + 2α / sqrt(π) * exp(-α^2 r_cut^2) / r_cut)
                          * (r - r_cut)]   (for r < r_cut, else 0)

with α = 0.20 / Å (canonical Wolf damping), and ``r_cut`` matched to the
LJ cutoff. Ke = 1389.354... K·Å/e² is the Coulomb constant in this unit
system (matches RASPA3's internal units). Wolf gives ≈0.5 % accuracy
for well-converged insertions and is faster than Ewald for
single-particle insertions where there's no host-host term to converge.

PBC
---
We use the standard minimum-image convention for orthorhombic and
non-orthorhombic cells — for each (host, guest) pair we compute the
fractional displacement, wrap to (-0.5, 0.5], then transform back to
Cartesian. This is exact for cubic cells and a good approximation for
near-cubic cells; widely-distorted unit cells (eg long thin) need a
larger supercell first, which the ``runner`` enforces via
``min_cell_face_distance_check``.
"""

from __future__ import annotations

import math

import numpy as np

COULOMB_K_PER_E2_PER_A: float = 1389.354_577_2  # K·Å / e²; matches RASPA3 unit system


def cell_face_distances(cell: np.ndarray) -> np.ndarray:
    """Perpendicular distances between opposite faces of a triclinic cell, in A."""
    a, b, c = cell[0], cell[1], cell[2]
    vol = abs(np.dot(a, np.cross(b, c)))
    da = vol / np.linalg.norm(np.cross(b, c))
    db = vol / np.linalg.norm(np.cross(a, c))
    dc = vol / np.linalg.norm(np.cross(a, b))
    return np.array([da, db, dc])


def minimum_image_displacements(
    pos_host: np.ndarray, pos_guest_site: np.ndarray, cell: np.ndarray, cell_inv: np.ndarray
) -> np.ndarray:
    """Return (n_host, 3) minimum-image displacement vectors.

    ``pos_host`` is (n_host, 3) Cartesian; ``pos_guest_site`` is (3,) Cartesian.
    """
    delta = pos_host - pos_guest_site[None, :]              # (n_host, 3)
    frac = delta @ cell_inv.T                                # (n_host, 3) fractional
    frac -= np.round(frac)
    return frac @ cell


def lj_energy_K(
    delta_norms: np.ndarray, sigma_A: np.ndarray, epsilon_K: np.ndarray, r_cut_A: float
) -> float:
    """Truncated-and-shifted 12-6 Lennard-Jones, in Kelvin.

    All inputs are 1-D arrays of length n_pairs.
    """
    mask = (delta_norms < r_cut_A) & (delta_norms > 1e-8)
    if not np.any(mask):
        return 0.0
    r = delta_norms[mask]
    sig = sigma_A[mask]
    eps = epsilon_K[mask]
    sr6 = (sig / r) ** 6
    e_full = 4.0 * eps * (sr6 * sr6 - sr6)
    sr6_cut = (sig / r_cut_A) ** 6
    e_shift = 4.0 * eps * (sr6_cut * sr6_cut - sr6_cut)
    return float(np.sum(e_full - e_shift))


def wolf_coulomb_energy_K(
    delta_norms: np.ndarray, q_pair: np.ndarray, r_cut_A: float, alpha_inv_A: float = 0.20
) -> float:
    """Wolf-summation Coulomb, in Kelvin (q_pair already includes q_i * q_j)."""
    mask = (delta_norms < r_cut_A) & (delta_norms > 1e-8)
    if not np.any(mask):
        return 0.0
    r = delta_norms[mask]
    q = q_pair[mask]
    alpha = alpha_inv_A
    erfc_ar = np.erfc(alpha * r) if hasattr(np, "erfc") else _np_erfc(alpha * r)
    erfc_arc = math.erfc(alpha * r_cut_A)
    term_self = erfc_ar / r
    term_shift = erfc_arc / r_cut_A
    deriv_at_cut = (
        erfc_arc / (r_cut_A * r_cut_A)
        + (2.0 * alpha / math.sqrt(math.pi))
        * math.exp(-alpha * alpha * r_cut_A * r_cut_A)
        / r_cut_A
    )
    e = q * (term_self - term_shift + deriv_at_cut * (r - r_cut_A))
    return float(np.sum(e) * COULOMB_K_PER_E2_PER_A)


def _np_erfc(x: np.ndarray) -> np.ndarray:
    """Vectorised erfc via math.erfc for older NumPy."""
    out = np.empty_like(x, dtype=float)
    flat = out.ravel()
    for i, v in enumerate(np.asarray(x).ravel()):
        flat[i] = math.erfc(float(v))
    return out


def insertion_energy_K(
    *,
    framework_positions: np.ndarray,           # (n_fw, 3) Cartesian
    framework_atom_ids: np.ndarray,            # (n_fw,)
    cell: np.ndarray,                          # (3, 3)
    cell_inv: np.ndarray,                      # (3, 3); matches np.linalg.inv(cell.T) convention used here
    insertion_centre: np.ndarray,              # (3,)
    site_offsets: np.ndarray,                  # (n_sites, 3)
    site_atom_ids: np.ndarray,                 # (n_sites,)
    sigma_pair: np.ndarray,                    # (n_fw_atom_types, n_gas_atom_types)
    epsilon_pair: np.ndarray,                  # (n_fw_atom_types, n_gas_atom_types)
    fw_charges: np.ndarray,                    # (n_fw,) per-atom in e
    gas_charges: np.ndarray,                   # (n_sites,) per-site in e
    r_cut_A: float,
    coulomb: str,
    alpha_inv_A: float = 0.20,
) -> tuple[float, float]:
    """Compute (E_LJ, E_coul) in K for one insertion."""
    e_lj_total = 0.0
    e_coul_total = 0.0
    for s in range(site_offsets.shape[0]):
        site_pos = insertion_centre + site_offsets[s]
        delta = minimum_image_displacements(framework_positions, site_pos, cell, cell_inv)
        d_norm = np.linalg.norm(delta, axis=1)

        sig_arr = sigma_pair[framework_atom_ids, site_atom_ids[s]]
        eps_arr = epsilon_pair[framework_atom_ids, site_atom_ids[s]]
        e_lj_total += lj_energy_K(d_norm, sig_arr, eps_arr, r_cut_A)

        if coulomb in ("Wolf", "wolf"):
            q_pair = fw_charges * gas_charges[s]
            e_coul_total += wolf_coulomb_energy_K(d_norm, q_pair, r_cut_A, alpha_inv_A=alpha_inv_A)
        elif coulomb in ("none", "None", ""):
            pass
        elif coulomb in ("Ewald", "ewald", "external_engine"):
            raise NotImplementedError(
                f"coulomb={coulomb!r} not supported by the internal evaluator. "
                "Map to 'Wolf' (the v0.4 default) or use the RASPA3 backend for the host run."
            )
        else:
            raise ValueError(f"unknown coulomb method {coulomb!r}")
    return e_lj_total, e_coul_total


__all__ = [
    "COULOMB_K_PER_E2_PER_A",
    "cell_face_distances",
    "insertion_energy_K",
    "lj_energy_K",
    "minimum_image_displacements",
    "wolf_coulomb_energy_K",
]
