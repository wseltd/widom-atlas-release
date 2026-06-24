"""Toy two-basin example — runs the full widom-atlas pipeline on synthetic samples.

This file is the runnable counterpart of the example notebook. It uses
synthetic Boltzmann-distributed samples on a cubic toy cell — outputs are
**toy** and not chemically meaningful (verdict §13.J).

Run with:

    uv run python examples/notebooks/toy_two_basin.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from widom_atlas.core.pipeline import PipelineParams, run_atlas
from widom_atlas.io.from_arrays import from_arrays


def main(out_dir: Path = Path("examples/reports/toy_two_basin")) -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)
    rng = np.random.default_rng(0)
    n_per_basin = 200
    target_a = np.array([0.25, 0.5, 0.5])
    target_b = np.array([0.75, 0.5, 0.5])
    blob_a = rng.normal(target_a, 0.01, (n_per_basin, 3)) % 1.0
    blob_b = rng.normal(target_b, 0.01, (n_per_basin, 3)) % 1.0
    frac = np.vstack([blob_a, blob_b])
    energies = np.concatenate(
        [
            rng.normal(-0.5, 0.01, n_per_basin),
            rng.normal(-0.4, 0.01, n_per_basin),
        ]
    )
    atlas_input = from_arrays(
        structure=atoms,
        positions_frac=frac,
        energies_eV=energies,
        temperature_K=298.15,
        gas="CO2",
        metadata={
            "example": "toy_two_basin",
            "warning": "synthetic toy samples — not chemically meaningful (verdict §13.J)",
        },
    )
    params = PipelineParams(n_grid=(32, 32, 32), dbscan_eps_A=0.5, min_samples=8)
    result = run_atlas(atlas_input, params, out_dir, structure=atoms)
    print(f"basins extracted: {len(result.basins)}")
    print(f"manifest: {result.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
