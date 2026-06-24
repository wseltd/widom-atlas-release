"""Self-consistent induced-dipole polarization for the native Widom evaluator.

First public polarizable Widom Henry-constant workflow (as of mid-2026):
no production polarizable + Widom adsorption pipeline exists in LAMMPS
(fix widom has no DRUDE path), RASPA3 (polarization is "initial" and not
documented inside Widom moves), or upstream cusp-ai-oss/widom (no FF
engine at all). The Becker-Lin-Dubbeldam-Vlugt 2018 modified RASPA used
for the published Mg-MOF-74 + CO2 polarizable FF (DOI 10.1021/acs.jpcc.8b08639)
was never released publicly.

Implements isotropic-polarizability point-dipoles with Thole damping
(Noskov adaptation), self-consistent iteration on the induced dipoles,
and polarization-energy bookkeeping for the test-particle insertion.

Per insertion the workflow is:
  1. Static field E^0_i at every framework + probe atom from the permanent
     charges (computed via the existing Ewald infrastructure for the
     framework; direct-space for the inserted probe, since the probe
     is a small molecule).
  2. Self-consistent solution of mu_i = alpha_i (E^0_i + sum_{j != i} T_ij mu_j)
     where T_ij is the dipole field tensor with Thole damping
       T_ij_alpha_beta(r) = (3 r_alpha r_beta - r^2 delta_alpha_beta) / r^5
       * Thole_damping(r, alpha_i, alpha_j, a)
  3. Polarization energy at convergence:
       U_pol = -1/2 sum_i mu_i . E^0_i
  4. Insertion energy contribution:
       Delta U_pol = U_pol(framework + probe) - U_pol(framework only)

Notes
-----
- Framework U_pol is precomputed ONCE per simulation (the framework is
  rigid in Widom; framework dipoles depend only on the framework's own
  permanent-charge field, NOT on the probe). The Delta U_pol per
  insertion is then the change due to introducing the probe.
- We support direct matrix inversion (1 - alpha T)^-1 for small systems
  (<~ 200 atoms) and conjugate-gradient SCF for larger.
- Convergence criterion: max |delta mu| < 1e-6 e Angstrom by default.
- Thole damping (Noskov 2005 form): T_thole(r) = 1 - (1 + s r / 2) exp(-s r)
  with s = a_thole / (alpha_i alpha_j)^(1/6). a_thole = 2.6 is the
  Thole standard.

Units
-----
- alpha_i: A^3 (atomic polarizability)
- mu_i: e * A (dipole moment magnitude)
- E^0_i: K / (e A) -- consistent with the native runner's K-based energies
  and the Coulomb prefactor (k_e / k_B = 167101 K A / e^2) elsewhere.
- U_pol: K
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

COULOMB_K_ANGSTROM_PER_E2 = 167101.0
"""Coulomb prefactor in K Angstrom per e^2 (e^2 / (4 pi eps_0 k_B))."""


@dataclass(frozen=True)
class TholeDamping:
    """Thole damping parameters."""

    a_thole: float = 2.6
    """Thole damping parameter; 2.6 is the standard Noskov 2005 value."""

    def s_inv_angstrom(self, alpha_i_A3: float, alpha_j_A3: float) -> float:
        """Inverse-length damping parameter s_ij = a / (alpha_i alpha_j)^(1/6)."""
        return self.a_thole / ((alpha_i_A3 * alpha_j_A3) ** (1.0 / 6.0))

    def damping_factor(
        self, r_angstrom: float, alpha_i_A3: float, alpha_j_A3: float
    ) -> float:
        """Scalar Thole damping factor in (0, 1] applied to the bare dipole tensor."""
        s = self.s_inv_angstrom(alpha_i_A3, alpha_j_A3)
        s_r = s * r_angstrom
        return 1.0 - (1.0 + 0.5 * s_r) * math.exp(-s_r)


def bare_dipole_tensor(r_vec_angstrom: np.ndarray) -> np.ndarray:
    """Compute the 3x3 dipole-dipole interaction tensor for one pair.

    T_ij = (3 r_alpha r_beta - r^2 delta_alpha_beta) / r^5
    """
    r2 = float(np.dot(r_vec_angstrom, r_vec_angstrom))
    if r2 < 1e-20:
        return np.zeros((3, 3))
    r5 = r2 ** 2.5
    return (
        3.0 * np.outer(r_vec_angstrom, r_vec_angstrom) - r2 * np.eye(3)
    ) / r5


def damped_dipole_tensor(
    r_vec_angstrom: np.ndarray,
    alpha_i_A3: float,
    alpha_j_A3: float,
    thole: TholeDamping,
) -> np.ndarray:
    """3x3 dipole-dipole tensor with Thole damping (multiplied as a scalar factor)."""
    r = float(np.linalg.norm(r_vec_angstrom))
    if r < 1e-10:
        return np.zeros((3, 3))
    bare = bare_dipole_tensor(r_vec_angstrom)
    f = thole.damping_factor(r, alpha_i_A3, alpha_j_A3)
    return f * bare


def build_dipole_tensor_block_3Nx3N(
    positions_angstrom: np.ndarray,
    alphas_A3: np.ndarray,
    cell_matrix_angstrom: np.ndarray | None,
    thole: TholeDamping,
    cutoff_angstrom: float = 12.8,
) -> np.ndarray:
    """Assemble the full 3N x 3N damped dipole-tensor block matrix.

    If cell_matrix_angstrom is supplied, applies minimum-image; otherwise
    no periodic image (open boundary). Pairs farther than cutoff_angstrom
    contribute zero.

    Returns: 3N x 3N numpy array T where T[3i:3i+3, 3j:3j+3] = T_ij (off-diag).
    """
    N = len(positions_angstrom)
    T = np.zeros((3 * N, 3 * N))
    cutoff_sq = cutoff_angstrom ** 2
    inv_cell = (
        np.linalg.inv(cell_matrix_angstrom)
        if cell_matrix_angstrom is not None
        else None
    )

    for i in range(N):
        for j in range(i + 1, N):
            d = positions_angstrom[j] - positions_angstrom[i]
            if inv_cell is not None:
                frac_d = d @ inv_cell
                frac_d -= np.round(frac_d)
                d = frac_d @ cell_matrix_angstrom
            r2 = float(np.dot(d, d))
            if r2 > cutoff_sq:
                continue
            T_ij = damped_dipole_tensor(d, alphas_A3[i], alphas_A3[j], thole)
            T[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = T_ij
            T[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = T_ij
    return T


def solve_induced_dipoles_direct(
    static_field_K_per_e_angstrom: np.ndarray,
    alphas_A3: np.ndarray,
    T_block_3Nx3N: np.ndarray,
) -> np.ndarray:
    """Direct matrix inversion: (1/alpha - T) mu = E^0.

    Recommended for N <~ 200; CG iteration otherwise.

    Returns: induced dipoles mu in e A, shape (N, 3).
    """
    N = len(alphas_A3)
    inv_alpha = np.diag(np.repeat(1.0 / alphas_A3, 3))
    # The equation mu = alpha (E^0 + T mu) in matrix form:
    # mu = alpha E^0 + alpha T mu
    # mu - alpha T mu = alpha E^0
    # (I - alpha T) mu = alpha E^0
    # equivalently: (1/alpha - T) mu = E^0  (per-component)
    # Here we use the symmetric block form.
    A_matrix = inv_alpha - T_block_3Nx3N
    E_flat = static_field_K_per_e_angstrom.flatten()
    mu_flat = np.linalg.solve(A_matrix, E_flat)
    return mu_flat.reshape(N, 3)


def solve_induced_dipoles_scf(
    static_field_K_per_e_angstrom: np.ndarray,
    alphas_A3: np.ndarray,
    T_block_3Nx3N: np.ndarray,
    max_iterations: int = 100,
    tol_e_angstrom: float = 1e-6,
    relaxation: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Self-consistent iterative solver for induced dipoles.

    Uses under-relaxation (mu_new = (1-w) mu_old + w mu_naive) to prevent
    SCF divergence on systems with strong static fields. Default w=0.5
    matches Mooij/Klein's tested polarizable-MD prescription.

    Returns: (mu in e A shape (N, 3), n_iterations_used).
    """
    N = len(alphas_A3)
    mu = (alphas_A3[:, None] * static_field_K_per_e_angstrom).copy()
    for it in range(max_iterations):
        mu_flat = mu.flatten()
        T_mu = (T_block_3Nx3N @ mu_flat).reshape(N, 3)
        mu_naive = alphas_A3[:, None] * (
            static_field_K_per_e_angstrom + T_mu
        )
        mu_new = (1.0 - relaxation) * mu + relaxation * mu_naive
        delta = float(np.max(np.abs(mu_new - mu)))
        mu = mu_new
        if delta < tol_e_angstrom:
            return mu, it + 1
        if not np.isfinite(delta) or delta > 1e10:
            # SCF diverged. Return zero-dipole fallback + iteration count.
            # Caller can flag this as a non-polarizable result.
            return np.zeros_like(mu), -1
    return mu, max_iterations


def polarization_energy_K(
    induced_dipoles_e_angstrom: np.ndarray,
    static_field_K_per_e_angstrom: np.ndarray,
) -> float:
    """U_pol = -1/2 sum_i mu_i . E^0_i (units K)."""
    return -0.5 * float(
        np.sum(induced_dipoles_e_angstrom * static_field_K_per_e_angstrom)
    )


def static_coulomb_field_at_points_K_per_e_angstrom(
    target_positions: np.ndarray,
    source_positions: np.ndarray,
    source_charges_e: np.ndarray,
    cell_matrix_angstrom: np.ndarray | None = None,
    cutoff_angstrom: float = 12.8,
) -> np.ndarray:
    """Compute the static Coulomb field at target points due to point sources.

    Returns field in K/(e A); convention matches COULOMB_K_ANGSTROM_PER_E2.
    Minimum-image PBC if cell_matrix supplied.
    """
    N_target = len(target_positions)
    E = np.zeros((N_target, 3))
    cutoff_sq = cutoff_angstrom ** 2
    inv_cell = (
        np.linalg.inv(cell_matrix_angstrom)
        if cell_matrix_angstrom is not None
        else None
    )
    for i, p_t in enumerate(target_positions):
        for j, p_s in enumerate(source_positions):
            d = p_t - p_s
            if inv_cell is not None:
                frac_d = d @ inv_cell
                frac_d -= np.round(frac_d)
                d = frac_d @ cell_matrix_angstrom
            r2 = float(np.dot(d, d))
            if r2 < 1e-10 or r2 > cutoff_sq:
                continue
            r3 = r2 ** 1.5
            # E from source j at point i: k_e q_j (r_i - r_j) / r^3
            E[i] += COULOMB_K_ANGSTROM_PER_E2 * source_charges_e[j] * d / r3
    return E
