"""v0.3 backend strategy tests — schema strictness, units, Wolf, charge refusal,
RASPA3 ingest, comparison report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from widom_atlas.backends import (
    KELVIN_TO_EV,
    KJ_PER_MOL_PER_EV,
    KJ_PER_MOL_PER_KCAL_MOL,
    SAMPLE_FORMAT_VERSION,
    available_backends,
    get_backend,
    to_eV,
)
from widom_atlas.backends.coulomb import (
    COULOMB_PREFACTOR_EV_A,
    WolfParameters,
    cross_only_wolf_energy,
    wolf_pair_energy,
)
from widom_atlas.backends.schema import (
    ExternalSampleManifest,
)
from widom_atlas.backends.units import GAS_CONSTANT_R_J_PER_MOL_K, raise_missing_units
from widom_atlas.backends.user_parameterised import (
    UserChargeAwareBackend,
    UserParameterFile,
    load_user_parameter_file,
)

# ---------------------------------------------------------------- units


def test_unit_constants_match_codata() -> None:
    assert GAS_CONSTANT_R_J_PER_MOL_K == 8.314462618
    assert abs(KJ_PER_MOL_PER_EV - 96.48533212) < 1e-7
    assert KJ_PER_MOL_PER_KCAL_MOL == 4.184
    assert abs(KELVIN_TO_EV - 8.617333262e-5) < 1e-12


def test_to_eV_conversion_table() -> None:
    one = np.array([1.0])
    assert abs(to_eV(one, "eV")[0] - 1.0) < 1e-12
    assert abs(to_eV(one, "K")[0] - KELVIN_TO_EV) < 1e-12
    assert abs(to_eV(one, "kJ_mol")[0] - 1.0 / KJ_PER_MOL_PER_EV) < 1e-12
    assert abs(to_eV(one, "kcal_mol")[0] - KJ_PER_MOL_PER_KCAL_MOL / KJ_PER_MOL_PER_EV) < 1e-12


def test_to_eV_rejects_unknown_unit() -> None:
    with pytest.raises(ValueError, match="unknown energy unit"):
        to_eV(np.array([1.0]), "Hartree")


def test_raise_missing_units_message_actionable() -> None:
    with pytest.raises(ValueError, match="energy_unit"):
        raise_missing_units("foo")


# --------------------------------------------------------------- schema


def _good_manifest_dict() -> dict:
    return {
        "sample_format_version": "0.3",
        "framework": "UiO-66",
        "gas": "CO2",
        "temperature_K": 298.15,
        "backend": "raspa3_external",
        "backend_version": "3.0.4",
        "n_insertions": 10000,
        "random_seed": 42,
        "energy_unit": "K",
        "parameter_mode": "external_samples",
        "force_field": {
            "framework_lj": "UFF",
            "framework_charges": "user_supplied",
            "gas_model": "TraPPE-CO2",
            "mixing_rules": "Lorentz-Berthelot",
            "electrostatics": "Ewald",
        },
        "citations": [
            {"role": "gas_model", "doi": "10.1002/aic.690470719", "source": "Potoff & Siepmann 2001"},
        ],
        "redistribution_status": "user_supplied_not_bundled",
        "warnings": [],
        "suitable_for_quantitative_interpretation": True,
    }


def test_external_manifest_round_trip() -> None:
    m = ExternalSampleManifest.model_validate(_good_manifest_dict())
    assert m.sample_format_version == SAMPLE_FORMAT_VERSION
    assert m.energy_unit == "K"
    assert m.gas == "CO2"
    assert len(m.citations) == 1


def test_external_manifest_rejects_missing_units() -> None:
    bad = _good_manifest_dict()
    bad.pop("energy_unit")
    with pytest.raises(Exception):
        ExternalSampleManifest.model_validate(bad)


def test_external_manifest_rejects_unknown_unit() -> None:
    bad = _good_manifest_dict()
    bad["energy_unit"] = "Hartree"
    with pytest.raises(Exception):
        ExternalSampleManifest.model_validate(bad)


def test_external_manifest_rejects_unknown_field() -> None:
    bad = _good_manifest_dict()
    bad["surprise"] = 42
    with pytest.raises(Exception):
        ExternalSampleManifest.model_validate(bad)


def test_external_manifest_charge_aware_consistency() -> None:
    bad = _good_manifest_dict()
    bad["parameter_mode"] = "user_parameterised_coulomb_lj"
    bad["force_field"]["electrostatics"] = "none"
    with pytest.raises(Exception, match="electrostatics"):
        ExternalSampleManifest.model_validate(bad)
    bad["force_field"]["electrostatics"] = "Wolf"
    bad["force_field"]["framework_charges"] = "none"
    with pytest.raises(Exception, match="framework_charges"):
        ExternalSampleManifest.model_validate(bad)


# -------------------------------------------------------------- coulomb


def test_coulomb_prefactor_matches_scipy() -> None:
    from scipy import constants as C

    expected = (C.e**2 / (4 * np.pi * C.epsilon_0)) / C.e * 1e10
    assert abs(COULOMB_PREFACTOR_EV_A - expected) < 1e-6


def test_wolf_pair_energy_attractive_for_opposite_charges() -> None:
    params = WolfParameters(alpha_inv_A=0.20, cutoff_A=12.0)
    e = wolf_pair_energy(np.array([2.4]), np.array([1.2]), np.array([-0.3]), params)
    assert e[0] < 0


def test_cross_only_wolf_energy_zero_when_one_tag_only() -> None:
    params = WolfParameters(alpha_inv_A=0.20, cutoff_A=12.0)
    pos = np.array([[0, 0, 0], [3, 0, 0]], dtype=np.float64)
    q = np.array([0.5, -0.5])
    tags = np.array([1, 1], dtype=np.int64)  # both framework
    cell = np.eye(3) * 20.0
    pbc = np.array([True, True, True])
    e = cross_only_wolf_energy(pos, q, tags, cell, pbc, params, framework_tag=1, gas_tag=0)
    assert e == 0.0


# ---------------------------------------------------------- backends API


def test_available_backends_contains_v03_backends() -> None:
    names = available_backends()
    assert "user_parameterised_coulomb_lj" in names
    assert "external_samples" in names


def test_get_backend_user_parameterised_requires_params() -> None:
    with pytest.raises(ValueError, match="user_parameterised_coulomb_lj backend requires"):
        get_backend("user_parameterised_coulomb_lj")


# ------------------------------------------ user-parameterised refusal contract


def _params_with_charges() -> dict:
    return {
        "framework_atom_types": [
            {"label": "Mg", "charge_e": 1.2, "sigma_A": 2.69, "epsilon_K": 55.85, "source": "user"},
            {"label": "O", "charge_e": -0.6, "sigma_A": 3.12, "epsilon_K": 30.19, "source": "user"},
            {"label": "C", "charge_e": 0.3, "sigma_A": 3.43, "epsilon_K": 52.84, "source": "user"},
            {"label": "H", "charge_e": 0.1, "sigma_A": 2.57, "epsilon_K": 22.14, "source": "user"},
        ],
        "gas_sites": [
            {"label": "C", "charge_e": 0.7, "sigma_A": 2.80, "epsilon_K": 27.0, "source": "TraPPE-CO2", "doi": "10.1002/aic.690470719"},
            {"label": "O", "charge_e": -0.35, "sigma_A": 3.05, "epsilon_K": 79.0, "source": "TraPPE-CO2", "doi": "10.1002/aic.690470719"},
        ],
        "mixing_rules": "Lorentz-Berthelot",
        "electrostatics": "Wolf",
        "redistribution_status": "user_supplied_not_bundled",
        "hybrid_warning": "TraPPE-CO2 + user-supplied DDEC framework charges + UFF LJ — hybrid approximation",
    }


def _params_no_charges() -> dict:
    p = _params_with_charges()
    for e in p["framework_atom_types"] + p["gas_sites"]:
        e["charge_e"] = 0.0
    return p


def test_user_parameter_file_round_trip(tmp_path: Path) -> None:
    pf = tmp_path / "params.json"
    pf.write_text(json.dumps(_params_with_charges()))
    pfile = load_user_parameter_file(pf)
    assert isinstance(pfile, UserParameterFile)
    assert pfile.electrostatics == "Wolf"


def test_user_parameter_file_rejects_unknown_field(tmp_path: Path) -> None:
    bad = _params_with_charges()
    bad["surprise"] = 42
    pf = tmp_path / "bad.json"
    pf.write_text(json.dumps(bad))
    with pytest.raises(Exception):
        load_user_parameter_file(pf)


def test_user_parameterised_backend_refuses_no_charges(tmp_path: Path) -> None:
    """Refuse to run when charges are zero AND --allow-neutral-fallback is False."""
    pytest.importorskip("widom")
    pf = tmp_path / "no_charges.json"
    pf.write_text(json.dumps(_params_no_charges()))
    backend = UserChargeAwareBackend(
        parameter_file=pf,
        allow_neutral_fallback=False,
    )
    atoms = Atoms("Mg", positions=[[0, 0, 0]], cell=np.eye(3) * 10, pbc=True)
    with pytest.raises(ValueError, match="does not declare non-zero partial charges"):
        backend.generate(
            structure=atoms,
            gas="CO2",
            temperature_K=298.15,
            n_samples=10,
            seed=0,
            material_id="Mg",
            material_source="test",
        )


def test_user_parameterised_backend_neutral_fallback_records_warning(tmp_path: Path) -> None:
    """Allowing neutral fallback emits an explicit warning into the manifest."""
    pytest.importorskip("widom")
    from ase.io import read

    cif = Path("tests/fixtures/real_structures/UiO-66.cif")
    if not cif.exists():
        pytest.skip("UiO-66 fixture not present")
    structure = read(str(cif))

    pf = tmp_path / "no_charges.json"
    pf.write_text(json.dumps(_params_no_charges()))
    backend = UserChargeAwareBackend(
        parameter_file=pf,
        allow_neutral_fallback=True,
    )
    out = backend.generate(
        structure=structure,
        gas="CO2",
        temperature_K=298.15,
        n_samples=20,
        seed=0,
        material_id="UiO-66",
        material_source="test",
    )
    warnings = out.atlas_input.metadata["warnings"]
    assert any("neutral_fallback" in w for w in warnings)


# --------------------------------------------------------- RASPA3 ingest


def _write_fake_raspa(tmp: Path) -> Path:
    """Build a minimal-but-realistic RASPA3 directory layout for parser tests."""
    raspa_dir = tmp / "raspa_run"
    raspa_dir.mkdir()
    (raspa_dir / "simulation.input").write_text(
        "SimulationType        MonteCarlo\n"
        "FrameworkName         UiO-66\n"
        "Forcefield            UFF\n"
        "Component0            CO2\n"
        "ExternalTemperature   298.15\n"
        "NumberOfCycles        1000\n"
        "NumberOfInitializationCycles 0\n",
        encoding="utf-8",
    )
    out_sys = raspa_dir / "Output" / "System_0"
    out_sys.mkdir(parents=True)
    (out_sys / "output_UiO-66_298.data").write_text(
        "RASPA version: 3.0.4\n"
        "Some preamble here.\n"
        "Henry coefficient   1.234e-06   [mol/kg/Pa]\n"
        "Heat of adsorption  -3050.0     [K]\n"
        "End of simulation.\n",
        encoding="utf-8",
    )
    return raspa_dir


def test_raspa3_scalar_parse(tmp_path: Path) -> None:
    from widom_atlas.backends.raspa3_ingest import parse_raspa3_scalars

    raspa_dir = _write_fake_raspa(tmp_path)
    result = parse_raspa3_scalars(raspa_dir)
    assert result.framework_name == "UiO-66"
    assert result.gas == "CO2"
    assert result.temperature_K == 298.15
    assert result.henry_coefficient_mol_per_kg_per_Pa is not None
    assert abs(result.henry_coefficient_mol_per_kg_per_Pa - 1.234e-6) < 1e-12
    # -3050 K → -3050 * R / 1000 ≈ -25.36 kJ/mol
    assert result.heat_of_adsorption_kJ_per_mol is not None
    assert abs(result.heat_of_adsorption_kJ_per_mol - (-3050.0 * GAS_CONSTANT_R_J_PER_MOL_K / 1000.0)) < 0.01
    assert result.raspa_version == "3.0.4"
    assert any(k.endswith("output_UiO-66_298.data") for k in result.output_files_sha256)


def test_write_scalar_only_sidecar(tmp_path: Path) -> None:
    from widom_atlas.backends.raspa3_ingest import (
        parse_raspa3_scalars,
        write_scalar_only_sidecar,
    )

    raspa_dir = _write_fake_raspa(tmp_path)
    sc = parse_raspa3_scalars(raspa_dir)
    out = write_scalar_only_sidecar(
        out_path=tmp_path / "raspa.scalar.json",
        scalar_result=sc,
        force_field_label="UFF",
        framework_charge_source="user_supplied",
        gas_model="TraPPE-CO2",
        citations=[{"role": "framework_charges", "doi": "10.1234/x", "source": "test"}],
        warnings=[],
    )
    payload = json.loads(out.read_text())
    assert payload["framework"] == "UiO-66"
    assert payload["atlas_input"] is False
    assert "output_files_sha256" in payload


# ---------------------------------------------- external_samples + manifest


def test_external_samples_unit_conversion_via_manifest(tmp_path: Path) -> None:
    """Manifest declares energy_unit='K'; backend converts to eV at ingest."""
    from widom_atlas.io.from_arrays import from_arrays
    from widom_atlas.io.npz import save_samples_npz

    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=np.eye(3) * 10, pbc=True)
    n = 4
    energies_K = np.array([-100.0, -50.0, +500.0, -200.0])
    accessible = np.ones(n, dtype=bool)
    frac = np.zeros((n, 3))
    frac[:, 0] = np.linspace(0.1, 0.4, n)

    # Producer writes energies_eV slot but actually fills it with K values; the
    # manifest declares unit=K so the backend converts.
    ai = from_arrays(
        structure=atoms,
        positions_frac=frac,
        energies_eV=energies_K,  # K values stored under the eV slot deliberately
        accessible=accessible,
        temperature_K=298.15,
        gas="CO2",
        metadata={"samples_origin": "test"},
    )
    npz_p = tmp_path / "s.npz"
    save_samples_npz(ai, npz_p)
    manifest = {
        "sample_format_version": "0.3",
        "framework": "test_mat",
        "gas": "CO2",
        "temperature_K": 298.15,
        "backend": "external_samples",
        "n_insertions": n,
        "energy_unit": "K",
        "parameter_mode": "external_samples",
        "force_field": {
            "framework_lj": "user_supplied",
            "framework_charges": "user_supplied",
            "gas_model": "user_supplied",
            "mixing_rules": "user_supplied",
            "electrostatics": "external_engine",
        },
        "citations": [],
        "redistribution_status": "user_supplied_not_bundled",
        "warnings": [],
        "suitable_for_quantitative_interpretation": True,
    }
    (tmp_path / "s.npz.manifest.json").write_text(json.dumps(manifest))

    backend = get_backend("external_samples", external_samples_path=npz_p)
    out = backend.generate(
        structure=atoms,
        gas="CO2",
        temperature_K=298.15,
        n_samples=n,
        seed=0,
        material_id="test_mat",
        material_source="unit_test",
    )
    assert out.atlas_input.metadata["external_sample_manifest"] is not None
    energies_after_conversion = np.asarray(out.atlas_input.energies_eV)
    expected_eV = energies_K * KELVIN_TO_EV
    np.testing.assert_allclose(energies_after_conversion, expected_eV, rtol=1e-9)


# --------------------------------------------------------- comparison report


def test_comparison_report_collects_rows(tmp_path: Path) -> None:
    from widom_atlas.backends.comparison import write_comparison_report

    # Fake a benchmark_run.json + scalar_comparison.json to exercise the parser
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "benchmark_run.json").write_text(
        json.dumps(
            {
                "run_id": "x",
                "set_name": "small",
                "gas": "CO2",
                "materials": [
                    {
                        "material_id": "UiO-66",
                        "gas": "CO2",
                        "temperature_K": 298.15,
                        "status": "ok",
                        "backend": "parameterised_lj",
                        "henry_coefficient": 1e-4,
                        "heat_of_adsorption_kJmol": -27.0,
                        "basins_count": 1,
                    }
                ],
            }
        )
    )
    (bench / "scalar_comparison").mkdir()
    (bench / "scalar_comparison" / "scalar_comparison.json").write_text(
        json.dumps(
            [
                {
                    "material_id": "UiO-66",
                    "gas": "CO2",
                    "KH_ref": 5e-7,
                    "Qads_ref_kJmol": -25.0,
                    "source_url": "https://doi.org/10.1021/la3046309",
                }
            ]
        )
    )

    out_dir = tmp_path / "compare_out"
    json_path, md_path = write_comparison_report([bench], out_dir)
    payload = json.loads(json_path.read_text())
    assert any(r["material"] == "UiO-66" for r in payload)
    md = md_path.read_text()
    assert "UiO-66 + CO2" in md
    assert "KH log_ratio" in md
