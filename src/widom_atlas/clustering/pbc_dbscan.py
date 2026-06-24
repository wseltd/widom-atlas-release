"""PBC-aware DBSCAN over minimum-image distances.

Implementation note: we build a precomputed pairwise minimum-image distance
matrix and feed it to ``sklearn.cluster.DBSCAN`` with ``metric='precomputed'``.
This is O(N^2) memory-wise but bullet-proof on triclinic + boundary-crossing
basins, which is the failure mode the package must avoid; for v1 sample sizes
(thousands of points) the matrix is well under 100 MB.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from widom_atlas.pbc.minimum_image import min_image_distance


def _pbc_neighbours(
    positions_frac: np.ndarray, cell_matrix_A: np.ndarray, eps_A: float
) -> np.ndarray:
    """Pairwise PBC distance matrix; entries above ``eps_A`` are set to ``np.inf``-like sentinel."""
    n = positions_frac.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    a = positions_frac[:, None, :]  # (n, 1, 3)
    b = positions_frac[None, :, :]  # (1, n, 3)
    a2, b2 = np.broadcast_arrays(a, b)
    dist = min_image_distance(a2, b2, cell_matrix_A)
    # symmetrise to suppress numerical asymmetry
    return 0.5 * (dist + dist.T)


def pbc_dbscan(
    positions_frac: np.ndarray,
    cell_matrix_A: np.ndarray,
    eps_A: float,
    min_samples: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Run DBSCAN over PBC distances.

    Returns an integer label array of length ``len(positions_frac)`` with
    ``-1`` denoting noise. Optional Boltzmann ``weights`` enable a weighted
    core-point criterion (sum of neighbour weights >= ``min_samples * mean_weight``).
    """
    if eps_A <= 0:
        raise ValueError(f"eps_A must be > 0; got {eps_A}")
    cell = np.asarray(cell_matrix_A, dtype=np.float64)
    if cell.shape != (3, 3):
        raise ValueError(f"cell_matrix_A must be (3,3); got {cell.shape}")
    cell_norms = np.linalg.norm(cell, axis=1)
    if eps_A >= 0.5 * float(cell_norms.min()):
        raise ValueError(
            "eps_A=%g exceeds 0.5 * min cell width=%g; PBC neighbour search is ambiguous"
            % (eps_A, 0.5 * float(cell_norms.min()))
        )

    pos = np.asarray(positions_frac, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions_frac must be (N,3); got {pos.shape}")

    n = pos.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    dist = _pbc_neighbours(pos, cell, eps_A)

    if weights is None:
        labels = DBSCAN(eps=eps_A, min_samples=int(min_samples), metric="precomputed").fit_predict(dist)
        return labels.astype(np.int64)

    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights shape {w.shape} does not match positions_frac shape ({n},3)")
    mean_w = float(w.mean())
    if mean_w <= 0.0:
        raise ValueError(f"mean weight is non-positive: {mean_w}")
    sample_weight = w / mean_w  # normalised so sum(sample_weight) ≈ n; min_samples remains an int
    labels = DBSCAN(eps=eps_A, min_samples=int(min_samples), metric="precomputed").fit_predict(
        dist, sample_weight=sample_weight
    )
    return labels.astype(np.int64)
