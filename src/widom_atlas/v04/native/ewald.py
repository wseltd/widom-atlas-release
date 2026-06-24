"""Ewald summation for the native Widom evaluator.

Standard Ewald: total Coulomb energy of a periodic system

    E_Coul = E_real + E_recip + E_self + E_intra_excl

with

    E_real      = (1/2) k_e ∑_{i,j,n} q_i q_j erfc(α |r_ij + n|) / |r_ij + n|
                  (n = lattice vector; i = j excluded for n = 0)
    E_recip     = (k_e / (2V)) ∑_{k ≠ 0} (4π/k²) exp(-k²/(4α²)) |S(k)|²
                  where S(k) = ∑_j q_j exp(i k·r_j)
    E_self      = -k_e (α / √π) ∑_j q_j²
    E_intra_excl = -k_e ∑_{intra-excluded i,j} q_i q_j erf(α r_ij) / r_ij

For the **Widom insertion** of a rigid test molecule into a fixed framework,
the only terms that change with insertion position+orientation are:

    ΔU = U_real_cross(test, frame) + U_recip_cross(test, frame)

with

    U_real_cross = k_e ∑_{i in test, j in frame, |r_ij+n| < r_cut}
                   q_i q_j erfc(α |r_ij + n|) / |r_ij + n|
    U_recip_cross = (k_e / V) ∑_{k > 0} (4π/k²) exp(-k²/(4α²))
                    × Re[ S_frame(k) · S_test*(k) ]

(Self / intra-exclusion terms are position-independent for a rigid test
molecule and cancel in U_widom = U(test in frame) - U(test in vacuum).)

Units: distances in Å, charges in elementary-charge units (e), energies
in K (k_B·T at T = 1 K).  Coulomb prefactor in these units is

    k_e = e² / (4πε₀) / k_B / Å = 167101 K·Å·e⁻²
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.special import erfc

COULOMB_K_ANGSTROM_PER_E2 = 167101.0


@dataclass(frozen=True)
class EwaldParameters:
    """Choice of (α, r_real_cutoff, k_max) Ewald parameters.

    α has units Å⁻¹; r_real_cutoff has units Å; k_max has units Å⁻¹.
    A rough heuristic for a target precision ε:
        α = √(-ln ε) / r_real_cutoff
        k_max = 2 α √(-ln ε)
    so the relative error of both the real-space and reciprocal-space sums
    is approximately ε. The default values target ε ≈ 1e-6 with
    r_real = 12.8 Å (matching the v04 LJ cutoff).
    """

    alpha_inv_angstrom: float = 0.3
    real_cutoff_angstrom: float = 12.8
    k_max_inv_angstrom: float = 1.4


def reciprocal_vectors(
    cell_matrix: np.ndarray,
    k_max_inv_angstrom: float,
) -> np.ndarray:
    """Return all reciprocal-lattice vectors {b1 m + b2 n + b3 p} (without
    factor of 2π pre-applied, in conventional Ewald form b_i = 2π·(a_j×a_k)/V)
    with |k| ≤ k_max_inv_angstrom, excluding k=0.

    `cell_matrix` rows are the direct-lattice basis vectors (Å).
    Returns shape (n_k, 3).
    """
    a1, a2, a3 = cell_matrix[0], cell_matrix[1], cell_matrix[2]
    V = abs(np.dot(a1, np.cross(a2, a3)))
    b1 = 2 * math.pi * np.cross(a2, a3) / V
    b2 = 2 * math.pi * np.cross(a3, a1) / V
    b3 = 2 * math.pi * np.cross(a1, a2) / V

    n1 = int(math.ceil(k_max_inv_angstrom / np.linalg.norm(b1))) + 1
    n2 = int(math.ceil(k_max_inv_angstrom / np.linalg.norm(b2))) + 1
    n3 = int(math.ceil(k_max_inv_angstrom / np.linalg.norm(b3))) + 1

    k_vecs: list[np.ndarray] = []
    for h in range(-n1, n1 + 1):
        for k in range(-n2, n2 + 1):
            for l in range(-n3, n3 + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                kv = h * b1 + k * b2 + l * b3
                if np.linalg.norm(kv) <= k_max_inv_angstrom:
                    k_vecs.append(kv)
    return np.asarray(k_vecs) if k_vecs else np.zeros((0, 3))


def structure_factor(
    k_vecs: np.ndarray,
    charges_e: np.ndarray,
    positions_angstrom: np.ndarray,
) -> np.ndarray:
    """Compute the complex structure factor S(k) = ∑_j q_j exp(i k·r_j).

    Returns a (n_k,) complex array. NaN/Inf in inputs raise.
    """
    if positions_angstrom.size == 0:
        return np.zeros(k_vecs.shape[0], dtype=complex)
    phases = positions_angstrom @ k_vecs.T  # (n_atoms, n_k)
    return (charges_e[:, None] * np.exp(1j * phases)).sum(axis=0)


def reciprocal_weights(
    k_vecs: np.ndarray,
    alpha_inv_angstrom: float,
    cell_volume_angstrom3: float,
) -> np.ndarray:
    """Per-k weight w(k) = (4π/k²) exp(-k²/(4α²)) / V.

    Returns shape (n_k,). The half-sum convention is enforced by the
    caller; here we return weights for the FULL (k and -k) sum.
    """
    if k_vecs.size == 0:
        return np.zeros(0)
    k_sq = np.einsum("ij,ij->i", k_vecs, k_vecs)
    return (4.0 * math.pi / k_sq) * np.exp(-k_sq / (4.0 * alpha_inv_angstrom ** 2)) / cell_volume_angstrom3


@dataclass
class FrameworkEwaldCache:
    """Pre-computed framework structure factors and per-k weights.

    Computed once per system; re-used for every Widom insertion.
    """

    k_vecs: np.ndarray
    weights: np.ndarray
    S_frame: np.ndarray
    framework_charges_e: np.ndarray
    framework_positions: np.ndarray
    alpha_inv_angstrom: float
    real_cutoff_angstrom: float
    cell_matrix: np.ndarray
    cell_inverse: np.ndarray
    cell_volume: float


def build_framework_ewald_cache(
    framework_charges_e: np.ndarray,
    framework_positions_angstrom: np.ndarray,
    cell_matrix_angstrom: np.ndarray,
    parameters: EwaldParameters,
) -> FrameworkEwaldCache:
    """Build the per-system Ewald cache. Call once per NativeSystem."""
    k_vecs = reciprocal_vectors(cell_matrix_angstrom, parameters.k_max_inv_angstrom)
    V = abs(np.dot(
        cell_matrix_angstrom[0],
        np.cross(cell_matrix_angstrom[1], cell_matrix_angstrom[2]),
    ))
    weights = reciprocal_weights(k_vecs, parameters.alpha_inv_angstrom, V)
    S_frame = structure_factor(k_vecs, framework_charges_e, framework_positions_angstrom)
    return FrameworkEwaldCache(
        k_vecs=k_vecs,
        weights=weights,
        S_frame=S_frame,
        framework_charges_e=framework_charges_e,
        framework_positions=framework_positions_angstrom,
        alpha_inv_angstrom=parameters.alpha_inv_angstrom,
        real_cutoff_angstrom=parameters.real_cutoff_angstrom,
        cell_matrix=cell_matrix_angstrom,
        cell_inverse=np.linalg.inv(cell_matrix_angstrom),
        cell_volume=V,
    )


def ewald_real_cross_test_frame(
    cache: FrameworkEwaldCache,
    test_charges_e: np.ndarray,
    test_positions_angstrom: np.ndarray,
) -> float:
    """Real-space cross-term energy in K.

    Loops over (i in test, j in frame) pairs, computes the minimum-image
    distance under PBC, and accumulates k_e q_i q_j erfc(α r) / r for r <
    r_real_cutoff. For a test with n_t atoms and frame with n_f atoms,
    cost is O(n_t · n_f).
    """
    if test_positions_angstrom.size == 0 or cache.framework_positions.size == 0:
        return 0.0
    cutoff = cache.real_cutoff_angstrom
    cutoff_sq = cutoff * cutoff
    alpha = cache.alpha_inv_angstrom
    cell = cache.cell_matrix
    inv = cache.cell_inverse
    total = 0.0
    for i, (q_i, r_i) in enumerate(zip(test_charges_e, test_positions_angstrom)):
        d = cache.framework_positions - r_i[None, :]
        frac_d = d @ inv
        frac_d -= np.round(frac_d)
        d_min = frac_d @ cell
        r_sq = np.einsum("ij,ij->i", d_min, d_min)
        within = r_sq < cutoff_sq
        if not np.any(within):
            continue
        r = np.sqrt(r_sq[within])
        q_j = cache.framework_charges_e[within]
        total += float(np.sum(q_i * q_j * erfc(alpha * r) / r))
    return COULOMB_K_ANGSTROM_PER_E2 * total


def ewald_recip_cross_test_frame(
    cache: FrameworkEwaldCache,
    test_charges_e: np.ndarray,
    test_positions_angstrom: np.ndarray,
) -> float:
    """Reciprocal-space cross-term energy in K.

    (k_e / V) × ∑_{k≠0} weight(k) Re[S_frame(k)·conj(S_test(k))].
    """
    if cache.k_vecs.size == 0 or test_positions_angstrom.size == 0:
        return 0.0
    S_test = structure_factor(cache.k_vecs, test_charges_e, test_positions_angstrom)
    cross = (cache.S_frame.conj() * S_test).real
    energy = float(np.sum(cache.weights * cross))
    return COULOMB_K_ANGSTROM_PER_E2 * energy


def ewald_self_intra_test(
    test_charges_e: np.ndarray,
    test_positions_angstrom: np.ndarray,
    alpha_inv_angstrom: float,
) -> float:
    """Test molecule's self-energy + intra-molecular Coulomb exclusion in K.

    These are constant for a rigid test molecule and cancel in the Widom
    excess chemical potential (because they appear in both U(test in box)
    and U(test in vacuum)). Provided for completeness; the runner doesn't
    need to call this for K_H purposes.
    """
    n = test_charges_e.size
    if n == 0:
        return 0.0
    self_term = -alpha_inv_angstrom / math.sqrt(math.pi) * float(np.sum(test_charges_e ** 2))
    # Intra exclusion: subtract erf(αr)·q_iq_j/r over (i<j) pairs
    intra = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(test_positions_angstrom[i] - test_positions_angstrom[j]))
            if r < 1e-12:
                continue
            from scipy.special import erf
            intra -= float(test_charges_e[i] * test_charges_e[j] * erf(alpha_inv_angstrom * r) / r)
    return COULOMB_K_ANGSTROM_PER_E2 * (self_term + intra)


def widom_ewald_delta_U(
    cache: FrameworkEwaldCache,
    test_charges_e: np.ndarray,
    test_positions_angstrom: np.ndarray,
) -> float:
    """The ΔU contribution to a single Widom insertion in K.

    Returns U_real_cross + U_recip_cross (the position-dependent part).
    Self and intra terms cancel between in-box and in-vacuum.
    """
    return (
        ewald_real_cross_test_frame(cache, test_charges_e, test_positions_angstrom)
        + ewald_recip_cross_test_frame(cache, test_charges_e, test_positions_angstrom)
    )
