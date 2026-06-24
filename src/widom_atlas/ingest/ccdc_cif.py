"""Operator-supplied gas-loaded CIF site extractor.

Use case: the operator obtained a gas-loaded CIF from a CCDC entry (e.g.
Mg-MOF-74 + CO2 from Queen 2014) and wants to register it as a
``SiteReferenceEntry`` in the registry. This ingester reads the CIF via
ASE, extracts the gas atoms (operator provides the gas formula), maps the
gas centre-of-mass position into fractional coordinates of the host cell,
and emits a ``SiteReferenceEntry``-shaped JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GasLoadedCifSiteRecord:
    material_id: str
    gas: str
    label: str
    centroid_frac: tuple[float, float, float]
    coordination_distance_A: float | None
    cif_source_path: str
    notes: str


def extract_gas_centroid_from_cif(
    cif_path: Path,
    *,
    material_id: str,
    gas: str,
    site_label: str,
    gas_atom_indices: list[int] | None = None,
    gas_element_set: set[str] | None = None,
) -> GasLoadedCifSiteRecord:
    """Read a gas-loaded CIF and extract a single (CoM-fractional) site.

    Either ``gas_atom_indices`` or ``gas_element_set`` must be supplied.
    For CO2 a typical gas_element_set is ``{"C", "O"}`` plus an explicit
    coordination_distance check; if multiple distinct guests are present
    the operator must pre-edit the CIF.
    """
    from ase import Atoms
    from ase.io import read as ase_read

    raw = ase_read(str(cif_path))
    if isinstance(raw, list):
        if not raw:
            raise ValueError(f"empty CIF (no Atoms objects) in {cif_path}")
        atoms = raw[0]
    else:
        atoms = raw
    if not isinstance(atoms, Atoms):
        raise TypeError(f"ASE returned non-Atoms object for {cif_path}: {type(atoms).__name__}")
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    if gas_atom_indices is None and gas_element_set is None:
        raise ValueError("provide either gas_atom_indices or gas_element_set")
    if gas_atom_indices is None:
        gas_atom_indices = [i for i, s in enumerate(symbols) if s in (gas_element_set or set())]
    if not gas_atom_indices:
        raise ValueError(f"no gas atoms found in {cif_path}")
    com = positions[gas_atom_indices].mean(axis=0)
    frac_array = atoms.cell.scaled_positions(com[None, :])[0]
    frac: tuple[float, float, float] = (
        float(frac_array[0] % 1.0),
        float(frac_array[1] % 1.0),
        float(frac_array[2] % 1.0),
    )
    return GasLoadedCifSiteRecord(
        material_id=material_id,
        gas=gas,
        label=site_label,
        centroid_frac=frac,
        coordination_distance_A=None,
        cif_source_path=str(cif_path),
        notes=f"CoM of {len(gas_atom_indices)} guest atoms ({gas_element_set or gas_atom_indices})",
    )


def to_site_reference_entry_dict(rec: GasLoadedCifSiteRecord, *, source_doi: str) -> dict[str, object]:
    return {
        "schema_version": "0.4",
        "material_id": rec.material_id,
        "gas": rec.gas,
        "label": rec.label,
        "centroid_frac": list(rec.centroid_frac),
        "coordination_distance_A": rec.coordination_distance_A,
        "site_kind": "open_metal_site" if "OMS" in rec.label.upper() else "other_see_notes",
        "notes": rec.notes,
        "provenance": {
            "citation": {"doi": source_doi, "source": f"operator-supplied gas-loaded CIF: {rec.cif_source_path}"},
            "measurement_method": "experimental_neutron",
            "redistribution_status": "user_supplied_not_redistributed",
            "notes": "site extracted from operator-supplied CIF; not bundled by widom-atlas",
        },
    }


def write_site_reference_entry(
    rec: GasLoadedCifSiteRecord, *, source_doi: str, out_path: Path
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(to_site_reference_entry_dict(rec, source_doi=source_doi), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out_path


__all__ = [
    "GasLoadedCifSiteRecord",
    "extract_gas_centroid_from_cif",
    "to_site_reference_entry_dict",
    "write_site_reference_entry",
]
