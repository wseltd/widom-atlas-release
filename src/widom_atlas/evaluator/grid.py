"""Insertion-position generators for the Widom evaluator.

Two strategies, both deterministic given a seed:

- ``deterministic_uniform_grid(cell, n_per_axis)`` — fixed-density grid in
  fractional coordinates, then transform to Cartesian. Reproducible across
  runs and machines.
- ``stochastic_uniform_random(cell, n_samples, rng)`` — uniform random in
  fractional coordinates, then transform.

For the v0.4 release gate we use ``deterministic_uniform_grid`` for the
RASPA3 parity check (where a random seed in the GCMC run would make
strict scalar parity infeasible) and stochastic for the Henry coefficient
estimator on real frameworks.
"""

from __future__ import annotations

import numpy as np


def deterministic_uniform_grid(cell: np.ndarray, n_per_axis: int) -> np.ndarray:
    """Return ``(n_per_axis**3, 3)`` Cartesian positions on a uniform fractional grid.

    The grid is offset by ``(0.5/n_per_axis)`` so neither face nor corner of the
    unit cell is sampled — gives well-spaced bulk-of-cell coverage.
    """
    if n_per_axis < 1:
        raise ValueError(f"n_per_axis must be >= 1, got {n_per_axis}")
    offset = 0.5 / n_per_axis
    coords = (np.arange(n_per_axis) + 0.5) / n_per_axis
    fa, fb, fc = np.meshgrid(coords, coords, coords, indexing="ij")
    fractional = np.stack([fa.ravel(), fb.ravel(), fc.ravel()], axis=1)  # (N, 3)
    fractional = (fractional + offset) % 1.0
    return fractional @ cell


def stochastic_uniform_random(
    cell: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniform random fractional positions transformed to Cartesian."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    fractional = rng.random((n_samples, 3))
    return fractional @ cell


__all__ = ["deterministic_uniform_grid", "stochastic_uniform_random"]
