"""27-image (3×3×3) expansion + cluster-label collapse for boundary-crossing basins."""

from __future__ import annotations

import numpy as np

from widom_atlas.pbc.wrap import _validate_cell

_OFFSETS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=np.float64,
)
_N_IMAGES = 27


def expand_27_images(
    frac: np.ndarray, energies: np.ndarray, cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate each fractional point to all 27 neighbouring image cells.

    Returns ``(frac_expanded, cart_expanded, parent_indices)`` with first
    dimension ``27 * N``. ``parent_indices[k]`` gives the index in the original
    array of the source point of expanded row ``k``.
    """
    cell_arr = _validate_cell(cell)
    frac_arr = np.asarray(frac, dtype=np.float64)
    en_arr = np.asarray(energies, dtype=np.float64)
    if frac_arr.ndim != 2 or frac_arr.shape[1] != 3:
        raise ValueError(f"frac must have shape (N,3); got {frac_arr.shape}")
    n = frac_arr.shape[0]
    if en_arr.shape != (n,):
        raise ValueError(f"energies must have shape ({n},); got {en_arr.shape}")

    frac_expanded = (frac_arr[:, None, :] + _OFFSETS[None, :, :]).reshape(n * _N_IMAGES, 3)
    cart_expanded = frac_expanded @ cell_arr
    parent_indices = np.repeat(np.arange(n, dtype=np.int64), _N_IMAGES)
    return frac_expanded, cart_expanded, parent_indices


def collapse_to_primary(
    labels: np.ndarray, parent_indices: np.ndarray
) -> np.ndarray:
    """Consolidate cluster labels across image replicas via the parent-index map.

    For each parent index, take the maximum non-noise label assigned to its
    images (DBSCAN convention: ``-1`` is noise). Image rows with the resulting
    consolidated label override the per-row label so a basin spanning a
    periodic boundary receives one ID.
    """
    labels_arr = np.asarray(labels, dtype=np.int64).copy()
    parent_arr = np.asarray(parent_indices, dtype=np.int64)
    if labels_arr.shape != parent_arr.shape:
        raise ValueError(f"labels and parent_indices shape mismatch: {labels_arr.shape} vs {parent_arr.shape}")
    n_parents = int(parent_arr.max()) + 1 if parent_arr.size else 0
    if n_parents == 0:
        return labels_arr
    consolidated = np.full(n_parents, -1, dtype=np.int64)
    for parent in range(n_parents):
        mask = parent_arr == parent
        if not mask.any():
            continue
        non_noise = labels_arr[mask][labels_arr[mask] >= 0]
        if non_noise.size:
            consolidated[parent] = int(non_noise.max())
    out = np.where(consolidated[parent_arr] >= 0, consolidated[parent_arr], labels_arr)
    return out.astype(np.int64)
