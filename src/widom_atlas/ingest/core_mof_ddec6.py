"""CoRE-MOF-DFT-2014 + DDEC6 charge-table ingester.

Operator drops ``CoRE-MOF-1.0-DFT-minimized.tar.gz`` (4.4 MB, CC-BY-4.0)
under ``benchmarks/cache/core_mof_dft_ddec6/`` and runs the ingester.
For each CIF inside, parses the per-atom DDEC6 charge from the
``_atom_site_charge`` column (when present) and emits a
``UserParameterFile.framework_atom_types``-shaped per-MOF charge JSON.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Ddec6ChargeTable:
    refcode: str
    cif_path: str
    n_atoms: int
    elements: list[str]
    charges_e: list[float]
    notes: str = ""


def unpack_core_mof_dft_ddec6(archive_path: Path, dest_root: Path) -> Path:
    archive_path = Path(archive_path)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.name.startswith("/") or ".." in m.name.split("/"):
                raise ValueError(f"unsafe tar member: {m.name!r}")
        tar.extractall(dest_root, filter="data")
    return dest_root


def parse_ddec6_cif(cif_path: Path) -> Ddec6ChargeTable:
    """Extract per-atom (element, charge) from a CoRE-MOF DDEC6 CIF.

    The CoRE-MOF DDEC6 release ships CIFs with an ``_atom_site_charge`` column
    populated. We use ASE's CIF reader to read elements + positions, then a
    lightweight regex pass over the file to lift the charge column.
    """
    text = cif_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    in_loop = False
    headers: list[str] = []
    body: list[list[str]] = []
    for line in lines:
        s = line.strip()
        if s.startswith("loop_"):
            if in_loop and headers and any(h.startswith("_atom_site") for h in headers):
                break
            in_loop = True
            headers = []
            body = []
            continue
        if not in_loop:
            continue
        if s.startswith("_atom_site"):
            headers.append(s.split(maxsplit=1)[0])
            continue
        if s.startswith("_") or s.startswith("data_") or s == "":
            if body:
                break
            continue
        body.append(s.split())

    if not headers or not body:
        return Ddec6ChargeTable(
            refcode=cif_path.stem,
            cif_path=str(cif_path),
            n_atoms=0,
            elements=[],
            charges_e=[],
            notes="no _atom_site_* loop found",
        )

    try:
        elem_idx = headers.index("_atom_site_type_symbol")
    except ValueError:
        try:
            elem_idx = headers.index("_atom_site_label")
        except ValueError:
            elem_idx = 0

    charge_idx = None
    for cand in ("_atom_site_charge", "_atom_site_partial_charge", "_atom_site_DDEC6_charge"):
        if cand in headers:
            charge_idx = headers.index(cand)
            break

    elements: list[str] = []
    charges: list[float] = []
    for row in body:
        if len(row) <= max(elem_idx, charge_idx if charge_idx is not None else 0):
            continue
        e = row[elem_idx]
        elements.append("".join(c for c in e if c.isalpha()) or e)
        if charge_idx is not None:
            try:
                charges.append(float(row[charge_idx]))
            except ValueError:
                charges.append(float("nan"))

    has_charges = bool(charges) and not all(np.isnan(c) for c in charges)
    return Ddec6ChargeTable(
        refcode=cif_path.stem,
        cif_path=str(cif_path),
        n_atoms=len(elements),
        elements=elements,
        charges_e=charges if has_charges else [],
        notes="ok" if has_charges else "_atom_site_charge column not populated",
    )


def to_user_parameter_file_dict(
    charges: Ddec6ChargeTable,
) -> dict[str, list[dict[str, object]] | list[str]]:
    """Project DDEC6 charges to a UserParameterFile.framework_atom_types-shaped fragment.

    Note: charges only — operator must merge with σ/ε from a separate FF source.
    """
    by_element: dict[str, list[float]] = {}
    for e, q in zip(charges.elements, charges.charges_e, strict=True):
        by_element.setdefault(e, []).append(q)
    fw_types = []
    for elem, qs in by_element.items():
        if not qs:
            continue
        fw_types.append({
            "label": elem,
            "atom_type": f"{elem}_DDEC6_avg",
            "charge_e": float(np.mean(qs)),
            "sigma_A": 3.0,  # placeholder; operator must override with FF-specific sigma
            "epsilon_K": 0.0,  # placeholder; operator must override
            "source": "DDEC6 average per element from CoRE-MOF-DFT-2014",
            "doi": "10.5281/zenodo.3986569",
        })
    return {"framework_atom_types": fw_types, "_warnings": [
        "sigma_A / epsilon_K are placeholders. Merge with a published FF (UFF, UFF4MOF, DREIDING, …) before running.",
    ]}


def write_charge_table(table: Ddec6ChargeTable, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(table.__dict__, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_path


__all__ = [
    "Ddec6ChargeTable",
    "parse_ddec6_cif",
    "to_user_parameter_file_dict",
    "unpack_core_mof_dft_ddec6",
    "write_charge_table",
]
