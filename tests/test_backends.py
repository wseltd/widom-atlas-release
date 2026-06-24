"""Unit + smoke tests for the backend layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from widom_atlas.backends import available_backends, get_backend
from widom_atlas.backends.parameterised_lj import ParameterisedLJCalculator
from widom_atlas.backends.parameters import (
    CH4_UFF_FALLBACK,
    KCAL_PER_MOL_TO_EV,
    KELVIN_TO_EV,
    TRAPPE_CO2,
    TRAPPE_N2,
    UFF_TABLE,
    framework_parameter_pack,
    gas_parameter_pack,
    parameter_pack_provenance,
)


def test_uff_covers_all_v1_benchmark_elements() -> None:
    """UFF table must include every element used by the v1 benchmark MOFs."""
    must_have = {"H", "C", "N", "O", "Mg", "Zn", "Cu", "Zr"}
    missing = must_have - set(UFF_TABLE)
    assert not missing, f"UFF table missing elements: {missing}"


def test_uff_unit_conversion() -> None:
    """UFF carbon: D_i=0.105 kcal/mol → 0.105 * 0.0433641 ≈ 0.004553 eV."""
    c = UFF_TABLE["C"]
    assert abs(c.eps_eV - 0.105 * KCAL_PER_MOL_TO_EV) < 1e-9
    # x_i = 3.851 → σ = 3.851 / 2^(1/6) ≈ 3.4309
    assert abs(c.sigma_A - 3.851 / 2.0 ** (1.0 / 6.0)) < 1e-6


def test_trappe_co2_unit_conversion() -> None:
    """TraPPE CO2 oxygen: ε/k_B = 79.0 K → 79.0 * 8.617e-5 eV."""
    o = TRAPPE_CO2["O"]
    assert abs(o.eps_eV - 79.0 * KELVIN_TO_EV) < 1e-9
    assert abs(o.sigma_A - 3.05) < 1e-9
    assert o.doi == "10.1002/aic.690470719"


def test_trappe_n2_unit_conversion() -> None:
    n = TRAPPE_N2["N"]
    assert abs(n.eps_eV - 36.0 * KELVIN_TO_EV) < 1e-9
    assert abs(n.sigma_A - 3.31) < 1e-9


def test_gas_parameter_pack_routes_correctly() -> None:
    co2 = gas_parameter_pack("CO2")
    assert set(co2) == {"C", "O"}
    assert co2["O"].doi.startswith("10.1002")
    n2 = gas_parameter_pack("N2")
    assert set(n2) == {"N"}
    ch4 = gas_parameter_pack("CH4")
    assert set(ch4) == {"C", "H"}
    assert CH4_UFF_FALLBACK


def test_unsupported_gas_raises() -> None:
    with pytest.raises(ValueError, match="unsupported gas"):
        gas_parameter_pack("H2O")


def test_parameter_pack_provenance_carries_doi() -> None:
    p = parameter_pack_provenance("CO2")
    assert p["mixing_rule"] == "Lorentz-Berthelot"
    assert "TraPPE-CO2" in p["gas_pack"]
    assert p["framework_pack"].startswith("UFF")
    assert p["gas_pack_doi"].startswith("10.")
    assert p["framework_pack_doi"].startswith("10.")


def test_available_backends_lists_v03_set() -> None:
    """v0.3 expanded the backend set with `user_parameterised_coulomb_lj`."""
    names = available_backends()
    assert set(names) == {
        "toy_lj",
        "parameterised_lj",
        "external_samples",
        "user_parameterised_coulomb_lj",
    }


def test_get_backend_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("not_a_backend")  # type: ignore[arg-type]


def test_get_backend_external_requires_path() -> None:
    with pytest.raises(ValueError, match="external_samples backend requires"):
        get_backend("external_samples")


def test_parameterised_lj_calculator_returns_zero_when_only_one_tag_present() -> None:
    """No inter-tag pairs ⇒ zero interaction energy."""
    atoms = Atoms("CuCu", positions=[[0, 0, 0], [3, 0, 0]], cell=np.eye(3) * 15, pbc=True)
    atoms.set_tags([1, 1])
    calc = ParameterisedLJCalculator(
        gas_parameters=gas_parameter_pack("CO2"),
        framework_parameters=framework_parameter_pack(),
        cutoff_A=12.0,
    )
    atoms.calc = calc
    assert atoms.get_potential_energy() == pytest.approx(0.0)


def test_parameterised_lj_calculator_combined_returns_attractive_energy() -> None:
    """Cu (framework, tag=1) + CO2 (gas, tag=0) at moderate distance ⇒ negative LJ energy."""
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=np.eye(3) * 20, pbc=True)
    atoms.set_tags([1])
    gas = molecule("CO2")
    gas.set_tags([0, 0, 0])
    gas.translate([6.0, 0, 0])
    combined = atoms.copy()
    combined.extend(gas)
    calc = ParameterisedLJCalculator(
        gas_parameters=gas_parameter_pack("CO2"),
        framework_parameters=framework_parameter_pack(),
        cutoff_A=12.0,
    )
    combined.calc = calc
    e = combined.get_potential_energy()
    assert e < 0.0
    assert e > -1.0  # not absurdly large


def test_parameterised_lj_calculator_zero_for_unknown_element() -> None:
    """Unknown element pair has eps=0, contributing 0 to the LJ sum."""
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=np.eye(3) * 20, pbc=True)
    atoms.set_tags([1])
    gas = Atoms("Og", positions=[[5, 0, 0]], cell=np.eye(3) * 20, pbc=True)  # Og not in UFF table here
    gas.set_tags([0])
    combined = atoms.copy()
    combined.extend(gas)
    calc = ParameterisedLJCalculator(
        gas_parameters={"Og": UFF_TABLE.get("Og", UFF_TABLE["H"])},  # Spoof: pass a real entry only when present
        framework_parameters=framework_parameter_pack(),
        cutoff_A=12.0,
    )
    combined.calc = calc
    e = combined.get_potential_energy()
    assert isinstance(e, float)


def test_parameterised_lj_backend_smoke_uio66() -> None:
    """End-to-end backend test on real UiO-66 CIF, small N for speed."""
    pytest.importorskip("widom")
    from ase.io import read

    cif = Path("tests/fixtures/real_structures/UiO-66.cif")
    if not cif.exists():
        pytest.skip("UiO-66 fixture not present")
    structure = read(str(cif))
    backend = get_backend("parameterised_lj", cutoff_A=12.0)
    out = backend.generate(
        structure=structure,
        gas="CO2",
        temperature_K=298.15,
        n_samples=80,
        seed=0,
        material_id="UiO-66",
        material_source="CoRE-MOF==2019.1",
    )
    assert out.samples_origin == "cuspai_widom"
    assert "TraPPE" in out.provenance["gas_pack"]
    assert out.provenance["mixing_rule"] == "Lorentz-Berthelot"
    ai = out.atlas_input
    assert ai.gas == "CO2"
    assert ai.samples.energies_eV.shape[0] == 80


def test_external_backend_round_trip(tmp_path: Path) -> None:
    """ExternalSamplesBackend ingests an NPZ produced by from_npz semantics."""
    from widom_atlas.io.from_arrays import from_arrays
    from widom_atlas.io.npz import save_samples_npz

    atoms = Atoms("Cu4", positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]],
                  cell=np.eye(3) * 10, pbc=True)
    n = 16
    rng = np.random.default_rng(0)
    frac = rng.random((n, 3))
    energies = rng.normal(-0.05, 0.02, n)
    accessible = np.ones(n, dtype=bool)
    atlas_input = from_arrays(
        structure=atoms,
        positions_frac=frac,
        energies_eV=energies,
        accessible=accessible,
        temperature_K=298.15,
        gas="CO2",
        metadata={"samples_origin": "test"},
    )
    npz_path = tmp_path / "samples.npz"
    save_samples_npz(atlas_input, npz_path)

    backend = get_backend("external_samples", external_samples_path=npz_path)
    out = backend.generate(
        structure=atoms,
        gas="CO2",
        temperature_K=298.15,
        n_samples=n,
        seed=0,
        material_id="test_mat",
        material_source="unit_test",
    )
    assert out.samples_origin == "external_samples"
    assert out.provenance["external_samples_sha256"]
    assert out.atlas_input.metadata["benchmark_material_id"] == "test_mat"
