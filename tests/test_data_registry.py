"""Schema strictness + round-trip tests for widom_atlas.data_registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from widom_atlas.data_registry import (
    DatasetRegistryEntry,
    ScalarReferenceEntry,
    SiteReferenceEntry,
    ValidationThresholds,
    list_datasets,
    list_scalar_references,
    list_site_references,
    load_dataset,
    load_scalar_reference,
    load_site_reference,
    load_validation_thresholds,
)
from widom_atlas.data_registry.registry import dataset_status, load_threshold_set

# ---------- registry round-trips ----------


def test_datasets_round_trip() -> None:
    ds = list_datasets()
    # v0.4 follow-up brought the registry to 16 entries: original 7 + ODAC25,
    # NIST-ISODB, MOFX-DB, CoRE-MOF-DFT-2014-DDEC6, Dzubak-MgMOF74-CO2-FF,
    # MACE-MP-0, ODAC25-MACE, UMA-FAIR-Chem, RASPA3-templates-MFI-henry.
    assert len(ds) >= 14
    names = {d.name for d in ds}
    assert {"CRAFTED", "CoRE-MOF-2019", "QMOF", "ODAC25", "NIST-ISODB", "MOFX-DB"} <= names


def test_crafted_uses_zenodo_doi_and_cdla_license() -> None:
    """v0.4 follow-up correction: CRAFTED's authoritative archive is on
    Zenodo (10.5281/zenodo.10120180) under CDLA-Sharing-1.0, not the
    Nature data-descriptor's CC-BY licence."""
    crafted = load_dataset("CRAFTED")
    assert crafted.primary_doi == "10.5281/zenodo.10120180"
    assert crafted.license == "CDLA-Sharing-1.0"


def test_v04_strict_threshold_set_present() -> None:
    """v0.4 follow-up brief §9 added a stricter threshold set."""
    ts = load_threshold_set("v0_4_strict")
    assert ts.KH_relative_error_upper == 0.05
    assert ts.Qads_abs_error_kJmol_upper == 2.0
    assert ts.basin_centroid_max_distance_A == 1.0


def test_load_dataset_lookup() -> None:
    crafted = load_dataset("CRAFTED")
    assert crafted.kind == "scalar_adsorption"
    assert crafted.license == "CDLA-Sharing-1.0"  # v0.4 follow-up correction
    assert crafted.redistribution_status == "open_access_with_attribution"
    assert crafted.primary_doi == "10.5281/zenodo.10120180"  # Zenodo authoritative archive
    assert crafted.content_summary.n_materials == 690
    assert "CO2" in crafted.content_summary.gases and "N2" in crafted.content_summary.gases
    assert 273.0 in crafted.content_summary.temperatures_K


def test_load_dataset_missing_raises() -> None:
    with pytest.raises(KeyError, match="unknown dataset"):
        load_dataset("DoesNotExist")


def test_scalars_carry_doi_and_provenance() -> None:
    refs = list_scalar_references()
    assert len(refs) >= 10
    for r in refs:
        assert r.provenance.citation.doi.startswith("10.")
        assert r.provenance.citation.source
        assert r.provenance.measurement_method
        assert r.provenance.redistribution_status


def test_load_scalar_reference_lookup() -> None:
    rows = load_scalar_reference("Mg-MOF-74", "CO2", temperature_K=298.15)
    # We have at least 3 reference values for this exact triple (Caskey, Mancini, Pandey)
    assert len(rows) >= 3
    dois = {r.provenance.citation.doi for r in rows}
    assert "10.1021/ja8036096" in dois  # Caskey 2008
    assert "10.1021/jacs.6c01686" in dois  # Mancini 2016
    assert "10.1021/acs.langmuir.5c04277" in dois  # Pandey 2025


def test_sites_carry_doi_and_kind() -> None:
    sites = list_site_references()
    assert len(sites) >= 6
    for s in sites:
        assert s.provenance.citation.doi.startswith("10.")
        assert s.site_kind
        for x in s.centroid_frac:
            assert 0.0 <= x < 1.0


def test_load_site_reference_lookup() -> None:
    rows = load_site_reference("Mg-MOF-74", "CO2")
    labels = {r.label for r in rows}
    assert "OMS-A_endon" in labels


def test_validation_thresholds_round_trip() -> None:
    vt = load_validation_thresholds()
    assert isinstance(vt, ValidationThresholds)
    for required in ("v0_4_minimum", "v0_4_strict", "v0_5_broader", "flagship", "broad_screening"):
        assert required in vt.sets


def test_threshold_set_v0_4_minimum_values() -> None:
    ts = load_threshold_set("v0_4_minimum")
    assert ts.KH_relative_error_upper == 0.20
    assert ts.Qads_abs_error_kJmol_upper == 3.0
    assert ts.basin_centroid_max_distance_A == 1.5
    assert ts.convergence_min_insertions_KH == 50000


def test_threshold_set_flagship_is_strictest() -> None:
    flagship = load_threshold_set("flagship")
    v0_4 = load_threshold_set("v0_4_minimum")
    v0_5 = load_threshold_set("v0_5_broader")
    assert flagship.KH_relative_error_upper < v0_5.KH_relative_error_upper < v0_4.KH_relative_error_upper
    assert flagship.basin_centroid_max_distance_A < v0_5.basin_centroid_max_distance_A < v0_4.basin_centroid_max_distance_A


# ---------- schema strictness ----------


def _good_dataset() -> dict:
    return {
        "schema_version": "0.4",
        "name": "Test",
        "kind": "structures",
        "description": "test",
        "primary_url": "https://example.org/x",
        "primary_doi": "10.0000/xyz",
        "license": "CC-BY-4.0",
        "redistribution_status": "open_access_with_attribution",
        "citations": [{"doi": "10.0000/xyz", "source": "Some authors. Some journal (year)."}],
        "content_summary": {"n_materials": 1, "gases": ["CO2"]},
        "file_format": ["CIF"],
    }


def test_dataset_schema_rejects_missing_doi() -> None:
    bad = _good_dataset()
    bad["primary_doi"] = ""
    with pytest.raises(Exception):
        DatasetRegistryEntry.model_validate(bad)


def test_dataset_schema_rejects_missing_license() -> None:
    bad = _good_dataset()
    del bad["license"]
    with pytest.raises(Exception):
        DatasetRegistryEntry.model_validate(bad)


def test_dataset_schema_rejects_missing_redistribution() -> None:
    bad = _good_dataset()
    del bad["redistribution_status"]
    with pytest.raises(Exception):
        DatasetRegistryEntry.model_validate(bad)


def test_dataset_schema_rejects_doi_without_prefix() -> None:
    bad = _good_dataset()
    bad["primary_doi"] = "not_a_doi_or_url"
    with pytest.raises(Exception, match="primary_doi"):
        DatasetRegistryEntry.model_validate(bad)


def test_dataset_schema_rejects_unknown_field() -> None:
    bad = _good_dataset()
    bad["surprise"] = 42
    with pytest.raises(Exception):
        DatasetRegistryEntry.model_validate(bad)


def test_scalar_schema_rejects_zero_temperature() -> None:
    bad = {
        "schema_version": "0.4",
        "material_id": "X",
        "gas": "CO2",
        "temperature_K": 0.0,
        "provenance": {
            "citation": {"doi": "10.0/x", "source": "x"},
            "measurement_method": "experimental_isotherm",
            "redistribution_status": "open_access_with_attribution",
        },
    }
    with pytest.raises(Exception):
        ScalarReferenceEntry.model_validate(bad)


def test_site_schema_rejects_out_of_range_centroid() -> None:
    bad = {
        "schema_version": "0.4",
        "material_id": "X",
        "gas": "CO2",
        "label": "test",
        "centroid_frac": [1.2, 0.5, 0.5],
        "site_kind": "cage_centre",
        "provenance": {
            "citation": {"doi": "10.0/x", "source": "x"},
            "measurement_method": "experimental_isotherm",
            "redistribution_status": "open_access_with_attribution",
        },
    }
    with pytest.raises(Exception, match="centroid_frac"):
        SiteReferenceEntry.model_validate(bad)


# ---------- dataset_status ----------


def test_dataset_status_missing_when_no_cache(tmp_path: Path) -> None:
    crafted = load_dataset("CRAFTED")
    st = dataset_status(crafted, repo_root=tmp_path)
    assert st["present"] is False
    assert st["verified"] is False
    assert "missing" in st["note"].lower()


def test_dataset_status_present_directory(tmp_path: Path) -> None:
    crafted = load_dataset("CRAFTED")
    assert crafted.cache_path is not None
    cache_dir = tmp_path / crafted.cache_path
    cache_dir.mkdir(parents=True)
    (cache_dir / "stub.csv").write_text("hello\n")
    st = dataset_status(crafted, repo_root=tmp_path)
    assert st["present"] is True
    assert "directory" in st["note"]


# ---------- v0.4 follow-up additions ----------


def test_predecessor_doi_recorded_for_crafted_and_core_mof_2019() -> None:
    """v0.4 mandate: CRAFTED + CoRE-MOF-2019 carry their predecessor DOIs for citation traceability."""
    crafted = load_dataset("CRAFTED")
    assert crafted.primary_doi == "10.5281/zenodo.10120180"
    assert crafted.predecessor_doi == "10.5281/zenodo.7689919"

    core = load_dataset("CoRE-MOF-2019")
    assert core.primary_doi == "10.5281/zenodo.14184621"
    assert core.predecessor_doi == "10.5281/zenodo.3677685"


def test_core_mof_2024_split_into_four_entries() -> None:
    """CoRE-MOF 2024 is split into structures / TSA / water / mofid-v2 per the v0.4 mandate."""
    names = {d.name for d in list_datasets()}
    expected = {
        "CoRE-MOF-2024-structures",
        "CoRE-MOF-2024-TSA-stability",
        "CoRE-MOF-2024-water",
        "CoRE-MOF-2024-mofid-v2",
    }
    assert expected <= names

    tsa = load_dataset("CoRE-MOF-2024-TSA-stability")
    assert tsa.kind == "stability_metadata"
    water = load_dataset("CoRE-MOF-2024-water")
    assert water.kind == "subset"
    mofid = load_dataset("CoRE-MOF-2024-mofid-v2")
    assert mofid.kind == "topology_metadata"


def test_odac23_carries_md5_per_file_table() -> None:
    """ODAC23 ships per-file MD5s, not a single sha256.

    Test reads checksums *from the registry YAML* rather than hard-coding
    them in the test source — this avoids tripping the secrets-scan
    regex_candidate heuristic (32-char hex strings) on the test file.
    """
    odac = load_dataset("ODAC23")
    assert odac.expected_sha256 is None
    assert odac.expected_md5 is None
    expected_files = {"odac23_s2ef.tar.gz", "ddec.tar.gz", "odac23_is2r.tar.gz",
                      "extxyz_train.tar.gz", "extxyz_val.tar.gz"}
    assert expected_files <= set(odac.expected_md5s)
    for fname, md5 in odac.expected_md5s.items():
        assert isinstance(md5, str)
        assert len(md5) == 32, f"{fname}: MD5 must be 32 hex chars; got {len(md5)}"
        assert all(c in "0123456789abcdef" for c in md5.lower())


def test_mofxdb_provenance_kind_classification() -> None:
    """MOFX-DB sub-database → provenance kind mapping matches the verified API response."""
    from widom_atlas.data_registry import mofxdb_provenance_kind_from_database

    assert mofxdb_provenance_kind_from_database("CoREMOF 2014") == "experimental"
    assert mofxdb_provenance_kind_from_database("CoREMOF 2019") == "experimental"
    assert mofxdb_provenance_kind_from_database("hMOF") == "hypothetical"
    assert mofxdb_provenance_kind_from_database("Tobacco") == "hypothetical"
    assert mofxdb_provenance_kind_from_database("IZA") == "zeolite"
    assert mofxdb_provenance_kind_from_database("CSD") == "placeholder"
    assert mofxdb_provenance_kind_from_database("PCOD-syn") == "experimental_or_synthetic"
    assert mofxdb_provenance_kind_from_database("UnknownSubDb") == "unknown"


def test_mofxdb_force_field_enum_covers_ids_1_to_9() -> None:
    """Closed enum for the 9 MOFX-DB force-field IDs verified live."""
    from widom_atlas.data_registry import MOFXDB_FORCE_FIELD_ID_TO_NAME

    assert set(MOFXDB_FORCE_FIELD_ID_TO_NAME) == set(range(1, 10))
    assert MOFXDB_FORCE_FIELD_ID_TO_NAME[1] == "UFF"
    assert MOFXDB_FORCE_FIELD_ID_TO_NAME[2] == "TraPPE"
    assert MOFXDB_FORCE_FIELD_ID_TO_NAME[7] == "TraPPE-zeo"


def test_dataset_status_md5_only_verification(tmp_path: Path) -> None:
    """A dataset that declares only MD5 (e.g. ODAC23) verifies via MD5."""
    import hashlib

    from widom_atlas.data_registry.registry import dataset_status
    from widom_atlas.data_registry.schema import DatasetRegistryEntry

    payload = b"fake odac23_s2ef.tar.gz contents"
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    cache_dir = tmp_path / "benchmarks/cache/odac23_test"
    cache_dir.mkdir(parents=True)
    archive = cache_dir / "test.tar.gz"
    archive.write_bytes(payload)

    entry = DatasetRegistryEntry.model_validate(
        {
            "schema_version": "0.4",
            "name": "test_odac23_md5",
            "kind": "ml_force_field",
            "description": "test",
            "primary_url": "https://example.org/x",
            "primary_doi": "10.0/x",
            "license": "CC-BY-4.0",
            "redistribution_status": "open_access_with_attribution",
            "citations": [{"doi": "10.0/x", "source": "test"}],
            "content_summary": {},
            "expected_md5": md5,
            "cache_path": "benchmarks/cache/odac23_test/test.tar.gz",
            "warnings": [],
        }
    )
    st = dataset_status(entry, repo_root=tmp_path)
    assert st["present"] is True
    assert st["verified"] is True
    assert st["actual_md5"] == md5


def test_dataset_status_directory_md5_per_file(tmp_path: Path) -> None:
    """ODAC23-style: cache_path is a directory; each declared file gets MD5-checked."""
    import hashlib

    from widom_atlas.data_registry.registry import dataset_status
    from widom_atlas.data_registry.schema import DatasetRegistryEntry

    cache_dir = tmp_path / "benchmarks/cache/odac23_dir"
    cache_dir.mkdir(parents=True)
    files = {"odac23_s2ef.tar.gz": b"a" * 1000, "ddec.tar.gz": b"b" * 1000}
    md5_table = {fname: hashlib.md5(content, usedforsecurity=False).hexdigest() for fname, content in files.items()}
    for fname, content in files.items():
        (cache_dir / fname).write_bytes(content)

    entry = DatasetRegistryEntry.model_validate(
        {
            "schema_version": "0.4",
            "name": "test_odac23_dir",
            "kind": "ml_force_field",
            "description": "test",
            "primary_url": "https://example.org/x",
            "primary_doi": "10.0/x",
            "license": "CC-BY-4.0",
            "redistribution_status": "open_access_with_attribution",
            "citations": [{"doi": "10.0/x", "source": "test"}],
            "content_summary": {},
            "expected_md5s": md5_table,
            "cache_path": "benchmarks/cache/odac23_dir",
            "warnings": [],
        }
    )
    st = dataset_status(entry, repo_root=tmp_path)
    assert st["present"] is True
    assert st["verified"] is True
