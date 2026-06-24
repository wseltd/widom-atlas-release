"""End-to-end real-data demonstration: CoRE-MOF UiO-66 → CuspAI Widom → widom-atlas.

This script is the canonical proof that the package consumes real CuspAI
Widom output on a real porous-material structure. It:

1. Loads UiO-66 from the bundled CoRE-MOF 2019 dataset (CSD refcode RUBTAK04_clean, CC BY 4.0).
2. Runs a small CuspAI Widom insertion campaign on it with an ASE
   Lennard-Jones calculator (verdict §G smoke-test calculator).
3. Converts the ``WidomInsertionResults`` to an :class:`AtlasInput` via the
   convenience adapter ``from_widom_result``.
4. Runs the full widom-atlas pipeline (density → basins → symmetry → reports).
5. Prints the artefacts produced and the Widom scalars carried in metadata.

Output goes under ``examples/reports/real_uio66_widom/``.

Run with:

    uv run python examples/notebooks/real_uio66_widom.py
"""

from __future__ import annotations

import json
from pathlib import Path

import CoRE_MOF
from ase.calculators.lj import LennardJones
from ase.io import read
from widom import run_widom_insertion

from widom_atlas.core.pipeline import PipelineParams, run_atlas
from widom_atlas.io.from_widom_result import from_widom_result


def main(out_dir: Path = Path("examples/reports/real_uio66_widom"), num_insertions: int = 400) -> None:
    with CoRE_MOF.get_CIF_structure_file("2019-ASR", "RUBTAK04_clean") as cif:
        atoms = read(str(cif))
    atoms.set_pbc(True)
    print(f"UiO-66 loaded: {len(atoms)} atoms; cell diag (Å) = {atoms.get_cell().array.diagonal()}")

    calc = LennardJones(epsilon=0.01, sigma=3.0)
    print(f"Running CuspAI Widom (num_insertions={num_insertions})...")
    results = run_widom_insertion(
        calculator=calc,
        structure=atoms,
        gas="CO2",
        temperature=298.15,
        model_outputs_interaction_energy=False,
        num_insertions=num_insertions,
        random_seed=42,
    )
    print(
        "Widom done: "
        f"KH={results.henry_coefficient:.3e} ± {results.henry_coefficient_std:.3e} mol/(kg·Pa); "
        f"Q_ads={results.heat_of_adsorption:.2f} kJ/mol; "
        f"accessible={sum(results.is_accessible)}/{len(results.is_accessible)}"
    )

    atlas_input = from_widom_result(
        results,
        gas="CO2",
        temperature_K=298.15,
        structure=atoms,
        metadata={
            "demo": "real_uio66_widom",
            "calculator": "ase.calculators.lj.LennardJones(epsilon=0.01, sigma=3.0)",
            "core_mof_refcode": "RUBTAK04_clean",
            "core_mof_dataset": "2019-ASR",
            "license": "CC BY 4.0",
        },
    )
    print(
        f"AtlasInput: structure_id={atlas_input.structure_id} "
        f"n_samples={atlas_input.n_samples} input_hash={atlas_input.input_hash[:16]}..."
    )

    params = PipelineParams(n_grid=(32, 32, 32), dbscan_eps_A=2.0, min_samples=4)
    result = run_atlas(atlas_input, params, out_dir, structure=atoms)
    print(f"\nPipeline complete. {len(result.basins)} basins extracted, {len(result.symmetry_groups)} symmetry groups.")
    for b in result.basins:
        print(
            f"  basin {b.basin_id}: count={b.count} weight={b.weight:.3f} "
            f"E={b.mean_energy_eV:.4f} eV  spread={b.spread_A:.2f} Å  accessible={b.accessible_fraction:.2f}"
        )

    artefacts = sorted(p for p in out_dir.rglob("*") if p.is_file())
    print(f"\nArtefacts ({len(artefacts)} files in {out_dir}):")
    for p in artefacts:
        print("  -", p.relative_to(out_dir))

    manifest = json.loads((out_dir / "manifest.json").read_text())
    print("\nmanifest.run_id:", manifest["run_id"])
    print("manifest.dependency_versions['widom']:", manifest["dependency_versions"].get("widom", "-"))


if __name__ == "__main__":
    main()
