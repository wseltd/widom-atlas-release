"""Example pipeline run for one MOF (UiO-66 stand-in) and one zeolite (MFI stand-in).

Uses the deterministic Si-diamond stand-in from the real-structures fixture
loader when no license-clean CIF is committed; otherwise loads the committed
file. Outputs are toy-shaped and explicitly tagged as such.

Run with:

    uv run python examples/notebooks/uio66_zif8_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from widom_atlas.benchmarks.runner import _toy_lj_samples
from widom_atlas.core.pipeline import PipelineParams, run_atlas
from widom_atlas.io.from_arrays import from_arrays


def _atlas_input(atoms, gas: str = "CO2", n: int = 600, seed: int = 0):
    cell = np.asarray(atoms.get_cell().array, dtype=np.float64)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    frac, energies, accessible = _toy_lj_samples(cell, positions, numbers, n_samples=n, seed=seed)
    return from_arrays(
        structure=atoms,
        positions_frac=frac,
        energies_eV=energies,
        accessible=accessible,
        temperature_K=298.15,
        gas=gas,
        metadata={
            "example": "uio66_zif8_pipeline",
            "samples_origin": "synthetic_toy_lj",
            "warning": "toy LJ output — not chemically meaningful (verdict §13.J)",
        },
    )


def main(out_root: Path = Path("examples/reports/uio66_zif8")) -> None:
    from widom_atlas.tests.fixtures_loader_passthrough import load_real_material

    params = PipelineParams(n_grid=(16, 16, 16), dbscan_eps_A=1.0, min_samples=6)
    for material_id in ("UiO-66", "MFI"):
        atoms, meta = load_real_material(material_id)
        atlas_input = _atlas_input(atoms, gas="CO2", seed=hash(material_id) % (1 << 32))
        out = out_root / material_id
        result = run_atlas(atlas_input, params, out, structure=atoms)
        print(
            f"{material_id}: stand_in={meta.get('stand_in')} "
            f"basins={len(result.basins)} manifest={out / 'manifest.json'}"
        )


if __name__ == "__main__":
    main()
