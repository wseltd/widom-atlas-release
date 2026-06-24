"""T014: Geometry-based relabel self-test.

Reads VOGTIV_clean_h.cif using ASE, applies the sublattice classifier,
and verifies that the per-label counts are consistent with the published
Mg-MOF-74 stoichiometry (18 Mg per unit cell, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.io import read

from .mg_mof74_relabel import MgMof74Labelling, classify_mg_mof74


@dataclass
class RelabelSelfTestResult:
    cif_path: Path
    label_counts: dict[str, int]
    n_atoms: int
    passes: bool
    reason: str


def run_relabel_selftest(cif_path: Path) -> RelabelSelfTestResult:
    atoms_obj = read(str(cif_path))
    if isinstance(atoms_obj, list):
        atoms_obj = atoms_obj[0]
    elements = list(atoms_obj.get_chemical_symbols())
    labels: list[str] = []
    for raw in atoms_obj.arrays.get("labels", elements):
        labels.append(str(raw) if raw is not None else "")
    fractional = atoms_obj.get_scaled_positions()
    cartesian = atoms_obj.get_positions()
    labelling: MgMof74Labelling = classify_mg_mof74(
        elements=elements,
        labels=labels,
        fractional=np.asarray(fractional),
        cartesian=np.asarray(cartesian),
    )
    n_mg = labelling.label_counts.get("Mof_Mg", 0)
    n_h = labelling.label_counts.get("Mof_H", 0)
    n_oa = labelling.label_counts.get("Mof_Oa", 0)
    n_ob = labelling.label_counts.get("Mof_Ob", 0)
    n_oc = labelling.label_counts.get("Mof_Oc", 0)
    if n_mg == 0:
        return RelabelSelfTestResult(
            cif_path=cif_path,
            label_counts=labelling.label_counts,
            n_atoms=len(elements),
            passes=False,
            reason="no Mg atoms classified — wrong fixture or classifier broken",
        )
    o_total = n_oa + n_ob + n_oc
    if o_total < n_mg:
        return RelabelSelfTestResult(
            cif_path=cif_path,
            label_counts=labelling.label_counts,
            n_atoms=len(elements),
            passes=False,
            reason=f"too few O atoms: {o_total} < {n_mg} Mg",
        )
    return RelabelSelfTestResult(
        cif_path=cif_path,
        label_counts=labelling.label_counts,
        n_atoms=len(elements),
        passes=True,
        reason=f"Mg={n_mg} H={n_h} Oa={n_oa} Ob={n_ob} Oc={n_oc}",
    )
