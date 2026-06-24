"""Tests for the CuspAI Widom convenience adapter (``from_widom_result``).

The adapter is the *convenience* constructor — the stable foundation remains
``from_arrays``. These tests use a small, real CuspAI Widom run on a real
CoRE-MOF UiO-66 CIF (one execution, ~10 s) so they actually exercise the
schema we install against, not a stand-in.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("widom") is None or importlib.util.find_spec("CoRE_MOF") is None:
    pytest.skip("CuspAI Widom or CoRE-MOF not installed", allow_module_level=True)

import CoRE_MOF
from ase.calculators.lj import LennardJones
from ase.io import read
from widom import run_widom_insertion

from widom_atlas.io import AtlasInput, from_widom_result
from widom_atlas.io.from_widom_result import _gas_positions_to_cart, _parse_cif_to_atoms

pytestmark = pytest.mark.real_material


@pytest.fixture(scope="module")
def real_widom_run():
    with CoRE_MOF.get_CIF_structure_file("2019-ASR", "RUBTAK04_clean") as path:
        atoms = read(str(path))
    atoms.set_pbc(True)
    calc = LennardJones(epsilon=0.01, sigma=3.0)
    results = run_widom_insertion(
        calculator=calc,
        structure=atoms,
        gas="CO2",
        temperature=298.15,
        model_outputs_interaction_energy=False,
        num_insertions=120,
        random_seed=0,
    )
    return atoms, results


def test_from_widom_result_returns_atlas_input(real_widom_run) -> None:
    atoms, results = real_widom_run
    ai = from_widom_result(results, gas="CO2", temperature_K=298.15, structure=atoms)
    assert isinstance(ai, AtlasInput)


def test_from_widom_result_preserves_sample_count(real_widom_run) -> None:
    _atoms, results = real_widom_run
    ai = from_widom_result(results, gas="CO2", temperature_K=298.15, structure=_atoms)
    assert ai.n_samples == sum(results.is_valid)


def test_from_widom_result_writes_widom_scalars_to_metadata(real_widom_run) -> None:
    _atoms, results = real_widom_run
    ai = from_widom_result(results, gas="CO2", temperature_K=298.15, structure=_atoms)
    sc = ai.metadata["widom_scalars"]
    assert sc["henry_coefficient"] == pytest.approx(results.henry_coefficient)
    assert sc["heat_of_adsorption_kJmol"] == pytest.approx(results.heat_of_adsorption)


def test_from_widom_result_uses_centroid_by_default(real_widom_run) -> None:
    _atoms, results = real_widom_run
    arr = np.asarray(results.gas_positions, dtype=np.float64)
    expected = arr.mean(axis=1)
    centroid = _gas_positions_to_cart(arr, atom_index=None)
    np.testing.assert_allclose(centroid, expected, atol=1e-12)


def test_from_widom_result_atom_index_pick(real_widom_run) -> None:
    _atoms, results = real_widom_run
    arr = np.asarray(results.gas_positions, dtype=np.float64)
    pick0 = _gas_positions_to_cart(arr, atom_index=0)
    np.testing.assert_array_equal(pick0, arr[:, 0, :])


def test_from_widom_result_parses_cif_when_no_structure_supplied(real_widom_run) -> None:
    _atoms, results = real_widom_run
    parsed = _parse_cif_to_atoms(results.optimized_structure_cif)
    assert parsed.pbc.all()
    assert len(parsed) > 0


def test_from_widom_result_rejects_h2o_gas(real_widom_run) -> None:
    _atoms, results = real_widom_run
    with pytest.raises(ValueError):
        from_widom_result(results, gas="H2O", temperature_K=298.15, structure=_atoms)


def test_from_widom_result_accessible_mask_matches_widom(real_widom_run) -> None:
    _atoms, results = real_widom_run
    ai = from_widom_result(results, gas="CO2", temperature_K=298.15, structure=_atoms)
    expected = sum(a for a, v in zip(results.is_accessible, results.is_valid, strict=False) if v)
    assert sum(ai.accessible) == expected
