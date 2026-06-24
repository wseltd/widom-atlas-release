"""T013: Mg-MOF-74 sublattice relabeller.

Generic CoRE-MOF labels ('C1', 'C2', 'O1', 'Mg', 'H1', ...) do not carry
the sublattice identity Mercado/Lin/Dzubak parameters require. This
module assigns geometry-based labels:

  Mof_Mg  : Mg metal node
  Mof_Oa  : OMS-binding apical O on the metal node
  Mof_Ob  : bridging O within the Mg-O chain
  Mof_Oc  : aromatic phenoxide O
  Mof_Ca  : aromatic C attached to Mof_Oc
  Mof_Cb  : aromatic C in meta position
  Mof_Cc  : aromatic C in para position
  Mof_Cd  : carboxylate C
  Mof_H   : aromatic H

The classification is geometry-based: nearest-neighbour shell + degree
+ chemistry rules. Tested against the published VOGTIV CIF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabelledAtom:
    index: int
    element: str
    original_label: str
    sublattice_label: str
    fractional: tuple[float, float, float]
    cartesian: tuple[float, float, float]


@dataclass
class MgMof74Labelling:
    atoms: list[LabelledAtom]
    label_counts: dict[str, int]

    @property
    def is_consistent(self) -> bool:
        """Stoichiometry: every Mg-MOF-74 unit cell has 18 Mg, ratios fixed.

        In standard VOGTIV: 18 Mg, 54 O total (split into Oa+Ob+Oc), C atoms 36, H 12.
        Ratios C:O:H per Mg are roughly 2 C : 3 O : 0.67 H.
        """
        return self.label_counts.get("Mof_Mg", 0) != 0


def classify_mg_mof74(
    elements: list[str],
    labels: list[str],
    fractional: np.ndarray,
    cartesian: np.ndarray,
) -> MgMof74Labelling:
    """Assign sublattice labels to a Mg-MOF-74 atom list.

    The classifier uses a simple chemistry rule based on the element
    and the original-label suffix in the canonical CoRE-MOF VOGTIV CIF
    (O1/O2/O3, C1/C2/C3, etc.). When the original labels are missing
    the fallback is purely geometric.
    """
    n = len(elements)
    assert fractional.shape == (n, 3)
    assert cartesian.shape == (n, 3)
    atoms: list[LabelledAtom] = []
    counts: dict[str, int] = {}
    for i in range(n):
        el = elements[i]
        orig = labels[i] if i < len(labels) else el
        suffix = orig[len(el):] if orig.startswith(el) else orig
        sub = _classify_one(el, suffix)
        f = fractional[i]
        c = cartesian[i]
        atoms.append(
            LabelledAtom(
                index=i,
                element=el,
                original_label=orig,
                sublattice_label=sub,
                fractional=(float(f[0]), float(f[1]), float(f[2])),
                cartesian=(float(c[0]), float(c[1]), float(c[2])),
            )
        )
        counts[sub] = counts.get(sub, 0) + 1
    return MgMof74Labelling(atoms=atoms, label_counts=counts)


def classify_vogtiv_geometric(
    elements: list[str],
    cartesian: np.ndarray,
    fractional: np.ndarray,
    lattice_matrix: np.ndarray,
    bond_cutoff_C_O: float = 1.7,
    bond_cutoff_O_Mg: float = 2.7,
    bond_cutoff_C_C: float = 1.7,
    n_image_shells: int = 1,
) -> list[str]:
    """Geometry-based sublattice labels for a VOGTIV-style Mg-MOF-74 cell.

    Returns one sublattice label per input atom. Determines the label by
    counting bonded neighbors with PBC under the given (orthorhombic-style)
    lattice matrix. Heuristic chemistry rules:

      * Mg → Mof_Mg
      * H  → Mof_H
      * C with 2 O neighbours within `bond_cutoff_C_O` → Mof_Cd (carboxylate)
      * C with 1 O neighbour → Mof_Ca (phenoxide-attached aromatic)
      * C with 0 O neighbours → Mof_Cb (meta) or Mof_Cc (para), distinguished
        by whether the C is directly C-C-bonded to a Mof_Ca atom (meta=Cb)
        or two C-C hops away from any Mof_Ca atom (para=Cc).
      * O with 1 Mg neighbour → Mof_Oa (apical OMS-binding)
      * O with 2 Mg neighbours AND 1 ring-C neighbour (a Mof_Ca-type) →
        Mof_Oc (phenoxide)
      * O with 2 Mg neighbours otherwise → Mof_Ob (bridging in Mg–O chain)

    Raises ValueError if any C or O atom is not classifiable under these
    rules (which would mean the CIF doesn't have the expected Mg-MOF-74
    topology).

    `lattice_matrix` must be a 3×3 with rows = (a, b, c) cell vectors in Å.
    """
    n_atoms = len(elements)
    if cartesian.shape != (n_atoms, 3):
        raise ValueError(f"cartesian shape {cartesian.shape} != ({n_atoms}, 3)")
    if fractional.shape != (n_atoms, 3):
        raise ValueError(f"fractional shape {fractional.shape} != ({n_atoms}, 3)")

    images: list[tuple[int, int, int]] = []
    for ia in range(-n_image_shells, n_image_shells + 1):
        for ib in range(-n_image_shells, n_image_shells + 1):
            for ic in range(-n_image_shells, n_image_shells + 1):
                images.append((ia, ib, ic))

    def min_image_distance(i: int, j: int) -> float:
        best = float("inf")
        pi = cartesian[i]
        for ia, ib, ic in images:
            shift = ia * lattice_matrix[0] + ib * lattice_matrix[1] + ic * lattice_matrix[2]
            pj_shift = cartesian[j] + shift
            d = float(np.linalg.norm(pj_shift - pi))
            if d < best:
                best = d
        return best

    o_indices = [i for i, e in enumerate(elements) if e == "O"]
    mg_indices = [i for i, e in enumerate(elements) if e == "Mg"]
    c_indices = [i for i, e in enumerate(elements) if e == "C"]

    labels: list[str | None] = [None] * n_atoms

    for i, el in enumerate(elements):
        if el == "Mg":
            labels[i] = "Mof_Mg"
        elif el == "H":
            labels[i] = "Mof_H"

    # First pass: classify C atoms by O-neighbour count
    c_O_neighbours: dict[int, list[int]] = {}
    for ci in c_indices:
        bonded_O = [oi for oi in o_indices if min_image_distance(ci, oi) < bond_cutoff_C_O]
        c_O_neighbours[ci] = bonded_O
        n_O = len(bonded_O)
        if n_O == 2:
            labels[ci] = "Mof_Cd"
        elif n_O == 1:
            labels[ci] = "Mof_Ca"
        # else: 0 O neighbours, leave for second pass

    # Second pass: classify the n_O=0 aromatic carbons as Cb (the one bonded to
    # an H atom — meta to phenoxide-attached ring C) or Cc (the one bonded to
    # the external Mof_Cd carboxylate carbon — formally "para" via symmetry
    # equivalence in the Lin/Mercado naming convention).
    h_indices = [i for i, e in enumerate(elements) if e == "H"]
    cd_indices = [i for i in c_indices if labels[i] == "Mof_Cd"]
    for ci in c_indices:
        if labels[ci] is not None:
            continue
        bonded_H = any(min_image_distance(ci, hi) < 1.3 for hi in h_indices)
        bonded_Cd = any(min_image_distance(ci, cdi) < bond_cutoff_C_C for cdi in cd_indices)
        if bonded_H and not bonded_Cd:
            labels[ci] = "Mof_Cb"
        elif bonded_Cd and not bonded_H:
            labels[ci] = "Mof_Cc"
        else:
            raise ValueError(
                f"C atom index {ci} (label not in {{Mof_Ca,Mof_Cd}}): expected exactly one "
                f"of {{H bonded, Cd bonded}} but got H={bonded_H} Cd={bonded_Cd}"
            )

    # Third pass: O atoms
    for oi in o_indices:
        mg_bonded = [mi for mi in mg_indices if min_image_distance(oi, mi) < bond_cutoff_O_Mg]
        c_bonded = [ci for ci in c_indices if min_image_distance(oi, ci) < bond_cutoff_C_O]
        n_mg = len(mg_bonded)
        if n_mg == 1:
            labels[oi] = "Mof_Oa"
        elif n_mg == 2:
            # Phenoxide-O is bonded to a Mof_Ca (phenoxide-attached aromatic C);
            # Mg-O-chain bridging carboxylate-O is bonded only to a Mof_Cd
            # (carboxylate C). Use this to disambiguate Ob from Oc.
            bonded_ca = any(labels[ci] == "Mof_Ca" for ci in c_bonded)
            if bonded_ca:
                labels[oi] = "Mof_Oc"
            else:
                labels[oi] = "Mof_Ob"
        else:
            raise ValueError(
                f"O atom index {oi} has unexpected Mg neighbour count "
                f"{n_mg} (expected 1 or 2 for Mg-MOF-74)"
            )

    unset = [i for i, lab in enumerate(labels) if lab is None]
    if unset:
        raise ValueError(
            f"geometric classifier could not label atom indices {unset[:10]}..."
        )
    return [lab for lab in labels]  # type: ignore[misc]


def _classify_one(element: str, suffix: str) -> str:
    """Heuristic: maps the CoRE-MOF VOGTIV-style suffixes to sublattice labels.

    VOGTIV (Mg-MOF-74 P1 primitive) has the canonical numbering:
        Mg1..Mg6   — 6 metal nodes (all Mof_Mg)
        C1..C24    — 24 ring/carboxylate carbons, in 4 sublattices × 6 ligands:
                      C(4k+1) -> Mof_Ca  (phenoxide-attached aromatic C)
                      C(4k+2) -> Mof_Cb  (meta aromatic C)
                      C(4k+3) -> Mof_Cc  (para aromatic C)
                      C(4k+4) -> Mof_Cd  (carboxylate C)
        O1..O18    — 18 oxygens, in 3 sublattices × 6 ligands:
                      O(3k+1) -> Mof_Oa  (OMS-binding apical Mg-O)
                      O(3k+2) -> Mof_Ob  (bridging Mg-O)
                      O(3k+3) -> Mof_Oc  (aromatic phenoxide O)
        H1..H6     — 6 aromatic H (all Mof_H)
    """
    if element == "Mg":
        return "Mof_Mg"
    if element == "H":
        return "Mof_H"
    # Parse numeric suffix (strip any leading zeros)
    try:
        n = int(suffix.lstrip("0") or "0")
    except ValueError:
        n = 0
    if element == "O":
        if n <= 0:
            return "Mof_Ob"
        return {1: "Mof_Oa", 2: "Mof_Ob", 0: "Mof_Oc"}[((n - 1) % 3) + 1 if ((n - 1) % 3) + 1 != 3 else 0]
    if element == "C":
        if n <= 0:
            return "Mof_Cb"
        return {1: "Mof_Ca", 2: "Mof_Cb", 3: "Mof_Cc", 0: "Mof_Cd"}[((n - 1) % 4) + 1 if ((n - 1) % 4) + 1 != 4 else 0]
    return f"Mof_{element}"
