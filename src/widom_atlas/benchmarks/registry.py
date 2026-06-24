"""Curated small benchmark set declared as immutable :class:`BenchmarkMaterial` records.

Source identifiers and license tags follow ``implementation-verdict.txt`` §6
and §13.E. CSD-derived structures and IZA bulk-redistribution files are
explicitly excluded from this registry per §12.
"""

from __future__ import annotations

from typing import Final

from widom_atlas.core.benchmark_models import BenchmarkMaterial


def _m(
    material_id: str,
    formula: str,
    *,
    source: str,
    citation: str,
    license_: str = "CC BY 4.0",
    space_group: str | None = None,
    pore_class: str | None = None,
    source_identifier: str | None = None,
    core_mof_dataset: str | None = None,
) -> BenchmarkMaterial:
    return BenchmarkMaterial(
        material_id=material_id,
        source=source,  # type: ignore[arg-type]
        source_identifier=source_identifier,
        formula=formula,
        space_group=space_group,
        cif_path=None,
        cif_sha256=None,
        license=license_,
        citation=citation,
        pore_class=pore_class,  # type: ignore[arg-type]
        core_mof_dataset=core_mof_dataset,
    )


# CSD refcodes verified against the bundled CoRE-MOF 2019-ASR dataset.
# Replacements / additions need a fresh `CoRE_MOF.list_structures('2019-ASR')` lookup.
SMALL_BENCHMARK_SET: Final[tuple[BenchmarkMaterial, ...]] = (
    _m(
        "Mg-MOF-74",
        "C12H6Mg2O8",
        source="core_mof",
        citation="Britt 2009; Bao 2011 (CoRE MOF 2019 refcode VOGTIV)",
        space_group="R-3",
        pore_class="open_metal_site",
        source_identifier="VOGTIV_clean_h",
        core_mof_dataset="2019-ASR",
    ),
    _m(
        "UiO-66",
        "C48H28O32Zr6",
        source="core_mof",
        citation="Cavka 2008 (CoRE MOF 2019 refcode RUBTAK)",
        space_group="Fm-3m",
        pore_class="standard",
        source_identifier="RUBTAK04_clean",
        core_mof_dataset="2019-ASR",
    ),
    _m(
        "ZIF-8",
        "C8H10N4Zn",
        source="core_mof",
        citation="Park 2006 (CoRE MOF 2019 refcode OFERUN)",
        space_group="I-43m",
        pore_class="standard",
        source_identifier="OFERUN_clean",
        core_mof_dataset="2019-ASR",
    ),
    _m(
        "MOF-5",
        "C24H12O13Zn4",
        source="core_mof",
        citation="Li 1999 (CoRE MOF 2019 refcode EDUSIF)",
        space_group="Fm-3m",
        pore_class="standard",
        source_identifier="EDUSIF_clean",
        core_mof_dataset="2019-ASR",
    ),
    _m(
        "MFI",
        "Si96O192",
        source="manual",
        citation="IZA framework code MFI — drop CIF into tests/fixtures/real_structures/MFI.cif",
        space_group="Pnma",
        pore_class="narrow",
    ),
    _m(
        "CHA",
        "Si36O72",
        source="manual",
        citation="IZA framework code CHA — drop CIF into tests/fixtures/real_structures/CHA.cif",
        space_group="R-3m",
        pore_class="narrow",
    ),
    _m(
        "CoRE_MOF_narrow_example",
        "C8H4Cu2O8",
        source="core_mof",
        citation="Chung 2019 (CoRE MOF 2019, CC BY 4.0); narrow-pore CoRE entry",
        pore_class="narrow",
        source_identifier="ABAVIJ_clean",
        core_mof_dataset="2019-ASR",
    ),
    _m(
        "CoRE_MOF_oms_example",
        "C18H12Cu3O18",
        source="core_mof",
        citation="Chung 2019 (CoRE MOF 2019, CC BY 4.0); open-metal-site CoRE entry",
        pore_class="open_metal_site",
        source_identifier="ACOCOM_clean",
        core_mof_dataset="2019-ASR",
    ),
)


_BENCHMARK_SETS: Final[dict[str, tuple[BenchmarkMaterial, ...]]] = {
    "small": SMALL_BENCHMARK_SET,
}


def get_benchmark_set(name: str) -> tuple[BenchmarkMaterial, ...]:
    """Return the named benchmark set; raises ``ValueError`` on unknown name."""
    if name not in _BENCHMARK_SETS:
        raise ValueError(f"unknown benchmark set: {name!r}; available: {sorted(_BENCHMARK_SETS)}")
    return _BENCHMARK_SETS[name]


__all__ = ["SMALL_BENCHMARK_SET", "get_benchmark_set"]
