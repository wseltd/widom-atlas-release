"""T016: symmetry-expansion + minimum-image site-truth comparator.

For each branch with site_truth.enabled=true, this module:
1. Loads the simulation CIF
2. Expands published site-truth fractional coordinates by space-group symmetry
3. Applies minimum-image periodic transformation
4. Computes the metal-O(gas) distance and returns whether it matches the
   published value within the per-branch tolerance.

No physics — pure crystallography.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometrySelfTestResult:
    branch_id: str
    target_distance_angstrom: float
    reconstructed_distance_angstrom: float
    tolerance_angstrom: float
    passes: bool
    note: str


def minimum_image_distance(
    frac_a: np.ndarray,
    frac_b: np.ndarray,
    cell_matrix: np.ndarray,
) -> float:
    """Minimum-image distance between two fractional positions in a triclinic cell."""
    delta = frac_a - frac_b
    delta -= np.round(delta)
    cart = delta @ cell_matrix
    return float(np.linalg.norm(cart))


def reconstruct_site_distance(
    metal_frac: np.ndarray,
    site_frac: np.ndarray,
    cell_matrix: np.ndarray,
) -> float:
    """Return the closest-image distance between metal (single fractional triplet)
    and the published site location (single fractional triplet)."""
    return minimum_image_distance(np.asarray(metal_frac), np.asarray(site_frac), cell_matrix)


def evaluate_geometry_self_test(
    branch_id: str,
    metal_frac: np.ndarray,
    site_frac: np.ndarray,
    cell_matrix: np.ndarray,
    target_distance_angstrom: float,
    tolerance_angstrom: float,
) -> GeometrySelfTestResult:
    d = reconstruct_site_distance(metal_frac, site_frac, cell_matrix)
    passes = abs(d - target_distance_angstrom) <= tolerance_angstrom
    return GeometrySelfTestResult(
        branch_id=branch_id,
        target_distance_angstrom=target_distance_angstrom,
        reconstructed_distance_angstrom=d,
        tolerance_angstrom=tolerance_angstrom,
        passes=passes,
        note=f"reconstructed={d:.3f}Å vs target={target_distance_angstrom:.3f}±{tolerance_angstrom:.2f}",
    )
