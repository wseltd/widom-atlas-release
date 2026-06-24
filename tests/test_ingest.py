"""Tests for the v0.4 ingestion layer."""

from __future__ import annotations

import hashlib
import tarfile
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def fixture_root() -> Path:
    return Path(__file__).parent / "fixtures"


def test_raspa3_ff_smoke_round_trip(fixture_root: Path) -> None:
    from widom_atlas.ingest.raspa3_ff import (
        parse_raspa3_input_directory,
        to_user_parameter_file,
    )

    bundle = fixture_root / "raspa3_mfi_henry"
    parsed = parse_raspa3_input_directory(
        force_field_path=bundle / "force_field.json",
        simulation_path=bundle / "simulation.json",
        component_paths={"CO2": bundle / "CO2.json"},
    )
    assert parsed.framework_name == "MFI_SI"
    assert parsed.charge_method.lower() in {"ewald", "wolf"}
    upf = to_user_parameter_file(parsed, gas_name="CO2")
    labels = [pa.label for pa in upf.framework_atom_types]
    assert "O" in labels and "Si" in labels
    co2_labels = [a.label for a in upf.gas_sites]
    assert any("C" in lb or "O" in lb for lb in co2_labels)


def test_mofxdb_parse_minimal_record() -> None:
    from widom_atlas.ingest.mofxdb import parse_mofxdb_record

    payload = {
        "id": 12345,
        "name": "MFI",
        "database": "iza",
        "doi": "10.1006/mfix.example",
        "heats": [
            {
                "force_field_id": 1,
                "force_field": "TraPPE",
                "temperature": 298.0,
                "adsorbate": [{"name": "CO2"}],
                "value": 25.0,
                "units": "kJ/mol",
                "simin": "SimulationType MonteCarlo\nNumberOfCycles 1000\n",
            },
            {
                "force_field_id": 2,
                "force_field": "Garcia-Sanchez",
                "temperature": 298.0,
                "adsorbate": [{"name": "CO2"}],
                "value": 26.0,
                "units": "kJ/mol",
                "simin": "SimulationType MonteCarlo\nNumberOfCycles 2000\n",
            },
        ],
    }
    distilled = parse_mofxdb_record(payload)
    recs = distilled["simin_records"]
    assert isinstance(recs, list) and len(recs) == 2
    assert all(len(r.simin_sha256) == 64 for r in recs)
    assert recs[0].framework_name == "MFI"
    assert recs[0].provenance_kind == "zeolite"


def test_mofxdb_select_deterministic_records() -> None:
    from widom_atlas.ingest.mofxdb import (
        MofxdbSiminRecord,
        select_deterministic_simin_records,
    )

    records = [
        MofxdbSiminRecord(
            mofx_record_id=i,
            mofx_database="coremof2019",
            provenance_kind="experimental",
            framework_name=f"MOF-{i}",
            component_names=["CO2"],
            force_field_id=ff_id,
            force_field_name=f"FF{ff_id}",
            gas="CO2",
            temperature_K=298.0,
            KH_value=1.0,
            KH_units="mol/kg/Pa",
            Qads_value=20.0,
            Qads_units="kJ/mol",
            simin_text=f"sim_{i}",
            simin_sha256=hashlib.sha256(f"sim_{i}".encode()).hexdigest(),
            source_doi=None,
            warnings=[],
        )
        for i, ff_id in enumerate([1, 2, 3, 1, 4, 5, 6, 7])
    ]
    picked = select_deterministic_simin_records(records, n=5, seed=42)
    assert len(picked) == 5
    ff_ids = [r.force_field_id for r in picked]
    assert len(set(ff_ids)) == 5


def test_mofxdb_select_excludes_hypothetical_by_default() -> None:
    from widom_atlas.ingest.mofxdb import (
        MofxdbSiminRecord,
        select_deterministic_simin_records,
    )

    records = [
        MofxdbSiminRecord(
            mofx_record_id=1,
            mofx_database="hmof",
            provenance_kind="hypothetical",
            framework_name="hMOF-1",
            component_names=["CO2"],
            force_field_id=1,
            force_field_name="FF1",
            gas="CO2",
            temperature_K=298.0,
            KH_value=1.0,
            KH_units="mol/kg/Pa",
            Qads_value=20.0,
            Qads_units="kJ/mol",
            simin_text="x",
            simin_sha256="0" * 64,
            source_doi=None,
            warnings=[],
        ),
        MofxdbSiminRecord(
            mofx_record_id=2,
            mofx_database="coremof2019",
            provenance_kind="experimental",
            framework_name="MOF-2",
            component_names=["CO2"],
            force_field_id=1,
            force_field_name="FF1",
            gas="CO2",
            temperature_K=298.0,
            KH_value=1.0,
            KH_units="mol/kg/Pa",
            Qads_value=20.0,
            Qads_units="kJ/mol",
            simin_text="y",
            simin_sha256="1" * 64,
            source_doi=None,
            warnings=[],
        ),
    ]
    picked = select_deterministic_simin_records(records, n=2, seed=0, require_distinct_force_fields=False)
    kinds = {r.provenance_kind for r in picked}
    assert "hypothetical" not in kinds


def test_nist_isodb_parse_henry_regime() -> None:
    from widom_atlas.ingest.nist_isodb import parse_nist_isotherm

    payload = {
        "filename": "10.1234.fake.isotherm0",
        "DOI": "10.1234/fake",
        "adsorbates": [{"formula": "CO2", "name": "Carbon dioxide"}],
        "adsorbent": {"name": "Mg-MOF-74"},
        "temperature": 298.15,
        "pressureUnits": "bar",
        "adsorptionUnits": "mmol/g",
        "isotherm_data": [
            {"pressure": 0.001, "total_adsorption": 0.6},
            {"pressure": 0.002, "total_adsorption": 1.2},
            {"pressure": 0.005, "total_adsorption": 3.0},
        ],
    }
    scalar = parse_nist_isotherm(payload)
    assert scalar.gas == "CO2"
    assert scalar.material == "Mg-MOF-74"
    assert scalar.KH_estimator_mol_per_kg_per_Pa is not None
    assert scalar.KH_estimator_mol_per_kg_per_Pa > 0
    assert scalar.n_points_used_for_KH == 3


def test_nist_isodb_unsupported_units_raises() -> None:
    from widom_atlas.ingest.nist_isodb import parse_nist_isotherm

    payload = {
        "filename": "x",
        "adsorbates": [{"formula": "CO2"}],
        "adsorbent": {"name": "MFI"},
        "temperature": 298.0,
        "pressureUnits": "psia",
        "adsorptionUnits": "mol/kg",
        "isotherm_data": [{"pressure": 1.0, "total_adsorption": 0.1}],
    }
    scalar = parse_nist_isotherm(payload)
    assert scalar.warnings  # unsupported pressure unit yields a warning


def test_crafted_unpack_safe_tar(tmp_path: Path) -> None:
    from widom_atlas.ingest.crafted import (
        summarise_unpacked_crafted,
        unpack_crafted_archive,
    )

    archive = tmp_path / "fake-crafted.tar.xz"
    extract_dir = tmp_path / "out"
    src = tmp_path / "src"
    src.mkdir()
    (src / "MOF-A_CO2_298.0_UFF_DDEC.csv").write_text("p,n\n1,0.1\n", encoding="utf-8")
    (src / "MOF-B_N2_273.0_UFF_DDEC.csv").write_text("p,n\n1,0.05\n", encoding="utf-8")
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(src, arcname=".")
    unpack_crafted_archive(archive, extract_dir)
    summary = summarise_unpacked_crafted(extract_dir)
    assert summary.n_csv_files == 2
    assert summary.materials_seen == 2
    assert "CO2" in summary.gases_seen


def test_crafted_refuses_unsafe_tar(tmp_path: Path) -> None:
    from widom_atlas.ingest.crafted import unpack_crafted_archive

    archive = tmp_path / "evil.tar.xz"
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.csv").write_text("x", encoding="utf-8")
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(src / "ok.csv", arcname="../../escape.csv")
    with pytest.raises(ValueError, match="unsafe tar member"):
        unpack_crafted_archive(archive, tmp_path / "out")


def test_core_mof_unpack_zip(tmp_path: Path) -> None:
    from widom_atlas.ingest.core_mof import unpack_core_mof_zip

    archive = tmp_path / "fake-core-mof.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("MOF1.cif", "data_MOF1\n_cell_length_a 10\n")
        zf.writestr("MOF2.cif", "data_MOF2\n_cell_length_a 20\n")
    result = unpack_core_mof_zip(archive, tmp_path / "out")
    assert result.n_cifs == 2


def test_core_mof_ddec6_parse_charges_from_cif(tmp_path: Path) -> None:
    from widom_atlas.ingest.core_mof_ddec6 import parse_ddec6_cif, to_user_parameter_file_dict

    cif = tmp_path / "FAKE.cif"
    cif.write_text(
        "data_FAKE\n"
        "_cell_length_a 10.0\n"
        "_cell_length_b 10.0\n"
        "_cell_length_c 10.0\n"
        "_cell_angle_alpha 90.0\n"
        "_cell_angle_beta 90.0\n"
        "_cell_angle_gamma 90.0\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "_atom_site_charge\n"
        "Zn1 Zn 0.0 0.0 0.0 1.50\n"
        "O1 O 0.5 0.0 0.0 -0.75\n"
        "O2 O 0.0 0.5 0.0 -0.75\n",
        encoding="utf-8",
    )
    table = parse_ddec6_cif(cif)
    assert table.n_atoms == 3
    assert "Zn" in table.elements and "O" in table.elements
    assert table.charges_e and len(table.charges_e) == 3
    upf_frag = to_user_parameter_file_dict(table)
    assert "framework_atom_types" in upf_frag


def test_qmof_unpack_zip(tmp_path: Path) -> None:
    from widom_atlas.ingest.qmof import unpack_qmof_zip

    archive = tmp_path / "fake-qmof.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("entry1/structure.cif", "data_e1")
        zf.writestr("entry1/metadata.json", '{"id": "e1"}')
    result = unpack_qmof_zip(archive, tmp_path / "out")
    assert result.n_cifs == 1
    assert result.n_json == 1


def test_eqeq_pacmof2_missing_binaries_return_structured_error() -> None:
    from widom_atlas.ingest.eqeq import is_eqeq_available, run_eqeq
    from widom_atlas.ingest.pacmof2 import is_pacmof2_available

    if not is_eqeq_available():
        from pathlib import Path as _Path

        run = run_eqeq(_Path("/tmp/nonexistent.cif"), _Path("/tmp/eqeq_out"))
        assert "not on PATH" in run.notes
    if not is_pacmof2_available():
        with pytest.raises(RuntimeError, match="pacmof2 binary not found"):
            from widom_atlas.ingest.pacmof2 import run_pacmof2

            run_pacmof2(Path("/tmp/nonexistent.cif"), Path("/tmp/pacmof_out"))


def test_odac_archive_missing() -> None:
    from widom_atlas.ingest.odac import verify_odac23_archive

    status = verify_odac23_archive(Path("/tmp/this-does-not-exist.tar.gz"), expected_md5=None)
    assert status.matches is False
    assert "missing" in status.notes


def test_ccdc_cif_extracts_co2_centroid(tmp_path: Path) -> None:
    from widom_atlas.ingest.ccdc_cif import (
        extract_gas_centroid_from_cif,
        to_site_reference_entry_dict,
    )

    cif = tmp_path / "fake-mof-co2.cif"
    cif.write_text(
        "data_FAKE\n"
        "_cell_length_a 10.0\n_cell_length_b 10.0\n_cell_length_c 10.0\n"
        "_cell_angle_alpha 90.0\n_cell_angle_beta 90.0\n_cell_angle_gamma 90.0\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Mg1 Mg 0.0 0.0 0.0\n"
        "C1 C 0.5 0.5 0.5\n"
        "O1 O 0.55 0.5 0.5\n"
        "O2 O 0.45 0.5 0.5\n",
        encoding="utf-8",
    )
    rec = extract_gas_centroid_from_cif(
        cif,
        material_id="MgMOF74",
        gas="CO2",
        site_label="OMS_Mg_CO2",
        gas_element_set={"C", "O"},
    )
    assert abs(rec.centroid_frac[0] - 0.5) < 1e-3
    payload = to_site_reference_entry_dict(rec, source_doi="10.1021/test")
    assert payload["material_id"] == "MgMOF74"
    assert payload["site_kind"] == "open_metal_site"
