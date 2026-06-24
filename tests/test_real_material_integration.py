"""Layer 2 — real-material integration tests.

After REPAIR-1: license-clean CIFs for UiO-66, ZIF-8, Mg-MOF-74, MOF-5 are
committed under ``tests/fixtures/real_structures/`` (CC BY 4.0 via the bundled
CoRE-MOF 2019-ASR dataset). The Layer 2 suite must run on those real cells —
not Si-diamond stand-ins — for the four committed materials. MFI / CHA fall
through to the stand-in path because IZA bulk-redistribution license is unclear
(verdict §6); Layer 2 documents this and skips zeolite-specific tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fixtures.real_structures import REAL_BENCHMARK_IDS, load_real_material

from widom_atlas.benchmarks.runner import _toy_lj_samples
from widom_atlas.core.pipeline import PipelineParams, run_atlas
from widom_atlas.io.from_arrays import from_arrays
from widom_atlas.io.structure_adapters import ase_to_pymatgen, pymatgen_to_ase
from widom_atlas.perturb.strain import apply_strain
from widom_atlas.robustness.compare import build_robustness_report

pytestmark = pytest.mark.real_material

_PARAMS = PipelineParams(n_grid=(8, 8, 8), dbscan_eps_A=0.5, min_samples=4)

# Materials with committed CC BY 4.0 CIFs (REPAIR-1).
COMMITTED_REAL_MATERIALS: tuple[str, ...] = ("UiO-66", "ZIF-8", "Mg-MOF-74", "MOF-5")
# Materials whose CIFs are *not* bundled because IZA bulk-redistribution is unclear.
DEFERRED_ZEOLITES: tuple[str, ...] = ("MFI", "CHA")
_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "real_structures"


def _atlas_input(atoms, gas: str = "CO2", n: int = 60, seed: int = 0):
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
        metadata={"layer": "layer2_integration"},
    )


@pytest.mark.parametrize("material_id", COMMITTED_REAL_MATERIALS)
def test_real_material_loads_real_cif_not_stand_in(material_id: str) -> None:
    """Each committed material must load its real CIF (not the Si stand-in)."""
    atoms, meta = load_real_material(material_id)
    assert meta["stand_in"] is False, (
        f"{material_id}: expected stand_in=False but got {meta['stand_in']}; "
        f"is the CIF committed at tests/fixtures/real_structures/{material_id}.cif?"
    )
    assert meta["source"] == "committed_cif"
    assert atoms.pbc.all()
    # Real MOFs have many atoms — the Si stand-in has 2.
    assert len(atoms) > 30, f"{material_id} has only {len(atoms)} atoms; smells like a stand-in"
    # And a real cell — Si diamond is cubic 5.43 Å.
    cell_diag = np.linalg.norm(atoms.get_cell().array, axis=1)
    if material_id in {"UiO-66", "MOF-5"}:
        # cubic ≈ 20–26 Å
        assert all(15.0 < x < 30.0 for x in cell_diag), cell_diag
    elif material_id == "ZIF-8":
        # cubic ≈ 17 Å
        assert all(10.0 < x < 25.0 for x in cell_diag), cell_diag


@pytest.mark.parametrize("material_id", COMMITTED_REAL_MATERIALS)
def test_real_material_provenance_records_full_metadata(material_id: str) -> None:
    """Every committed CIF must ship with a provenance.json carrying source / license / sha256 / citation / DOI."""
    prov_path = _FIXTURES_DIR / f"{material_id}.provenance.json"
    assert prov_path.exists(), f"missing {prov_path}"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    for key in (
        "material_id",
        "source_dataset",
        "source_package",
        "source_identifier",
        "license",
        "citation",
        "doi",
        "sha256",
        "redistribution_note",
    ):
        assert key in prov, f"{material_id}.provenance.json missing key {key}"
    assert prov["license"] == "CC BY 4.0"
    assert len(prov["sha256"]) == 64


@pytest.mark.parametrize("material_id", COMMITTED_REAL_MATERIALS)
def test_pipeline_runs_end_to_end_on_real_committed_material(
    material_id: str, tmp_path: Path
) -> None:
    """Full pipeline runs end-to-end on each committed real CIF."""
    atoms, meta = load_real_material(material_id)
    assert meta["stand_in"] is False
    ai = _atlas_input(atoms)
    result = run_atlas(ai, _PARAMS, tmp_path / material_id, structure=atoms)
    assert (tmp_path / material_id / "manifest.json").exists()
    assert (tmp_path / material_id / "basins.json").exists()
    assert (tmp_path / material_id / "config.json").exists()
    assert (tmp_path / material_id / "robustness" / "metrics.json").exists()
    assert result.density.grid.sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("material_id", DEFERRED_ZEOLITES)
def test_zeolite_loader_falls_back_to_stand_in(material_id: str) -> None:
    """MFI / CHA: until a license-clean CIF is supplied, the loader uses a Si stand-in.

    This test documents that behaviour explicitly so the suite cannot drift into
    pretending the zeolite path is validated when no real CIF is present.
    """
    atoms, meta = load_real_material(material_id)
    if (_FIXTURES_DIR / f"{material_id}.cif").exists():
        # Operator dropped a CIF in — that's allowed; real path then exercised.
        assert meta["stand_in"] is False
    else:
        assert meta["stand_in"] is True, meta
        assert meta["source"] == "synthetic_stand_in"
        assert "warning" in meta and material_id in meta["warning"]


def test_ase_pymatgen_roundtrip_preserves_real_material_cell() -> None:
    atoms, meta = load_real_material("UiO-66")
    assert meta["stand_in"] is False
    s = ase_to_pymatgen(atoms)
    a2 = pymatgen_to_ase(s)
    np.testing.assert_allclose(atoms.get_cell().array, a2.get_cell().array, atol=1e-10)


def test_strain_pipeline_on_real_material_cell() -> None:
    atoms, meta = load_real_material("ZIF-8")
    assert meta["stand_in"] is False
    strained = apply_strain(atoms, mode="isotropic", value=0.01)
    np.testing.assert_allclose(strained.get_cell().array, atoms.get_cell().array * 1.01, atol=1e-12)


def test_robustness_compare_on_real_material(tmp_path: Path) -> None:
    atoms, meta = load_real_material("MOF-5")
    assert meta["stand_in"] is False
    pristine = _atlas_input(atoms, seed=0)
    perturbed = _atlas_input(atoms, seed=1)
    pri_dir = tmp_path / "pristine"
    per_dir = tmp_path / "perturbed"
    run_atlas(pristine, _PARAMS, pri_dir, structure=atoms)
    perturbed_atoms = apply_strain(atoms, mode="isotropic", value=0.01)
    run_atlas(perturbed, _PARAMS, per_dir, structure=perturbed_atoms)
    manifest_path = per_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["perturbation_spec"] = {"kind": "isotropic", "magnitude": 0.01, "label": "iso1"}
    manifest["cell_matrix"] = perturbed_atoms.get_cell().array.tolist()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = build_robustness_report(pri_dir, per_dir)
    assert report.metrics_per_perturbation


def test_real_material_atlas_metadata_records_layer() -> None:
    atoms, _meta = load_real_material("UiO-66")
    ai = _atlas_input(atoms)
    assert ai.metadata.get("layer") == "layer2_integration"


def test_real_material_density_normalised_per_material() -> None:
    atoms, meta = load_real_material("Mg-MOF-74")
    assert meta["stand_in"] is False
    from widom_atlas.density.grid import build_density_grid

    ai = _atlas_input(atoms)
    g = build_density_grid(ai.samples, atoms, n_grid=(8, 8, 8))
    assert abs(float(g.grid.sum()) - 1.0) < 1e-9


def test_real_material_basin_extraction_handles_pbc() -> None:
    atoms, meta = load_real_material("UiO-66")
    assert meta["stand_in"] is False
    ai = _atlas_input(atoms, n=120, seed=2)
    from widom_atlas.clustering.basins import extract_basins

    basins = extract_basins(ai.samples, atoms, eps_A=0.5, min_samples=4)
    for b in basins:
        for x in b.centroid_frac:
            assert 0.0 <= x < 1.0


def test_real_material_symmetry_runs_and_reports_confidence() -> None:
    atoms, meta = load_real_material("ZIF-8")
    assert meta["stand_in"] is False
    ai = _atlas_input(atoms, n=80, seed=3)
    from widom_atlas.clustering.basins import extract_basins
    from widom_atlas.symmetry.grouping import group_basins

    basins = extract_basins(ai.samples, atoms, eps_A=0.5, min_samples=4)
    groups = group_basins(atoms, basins)
    for g in groups:
        assert 0.0 <= g.grouping_confidence <= 1.0


def test_layer_2_covers_all_six_named_materials() -> None:
    """Sanity check that the audit-named six materials are accounted for."""
    declared = set(REAL_BENCHMARK_IDS)
    handled = set(COMMITTED_REAL_MATERIALS) | set(DEFERRED_ZEOLITES)
    missing = declared - handled
    extra = handled - declared
    assert missing == set(), f"audit materials with no Layer 2 path: {missing}"
    assert extra == set(), f"unexpected Layer 2 materials: {extra}"
