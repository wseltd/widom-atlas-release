"""Charge electroneutrality helper.

Used for Na-Rho 5b where the YAML lists explicit Si/O/Na charges but does not
specify the Al charge. Operator directive: do NOT invent silently — derive
from declared composition and record in evidence.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElectroneutralityDerived:
    element: str
    charge: float
    composition: dict[str, float]
    explicit_charges: dict[str, float]
    derivation_text: str


def derive_charge_neutrality(
    target_element: str,
    composition: dict[str, float],
    explicit_charges: dict[str, float],
) -> ElectroneutralityDerived:
    """Return charge of `target_element` enforcing sum(c_i · q_i) = 0.

    `composition` is per-unit-cell atom counts (e.g., {"Si": 38.2, "Al": 9.8,
    "O": 96.0, "Na": 9.2}). `explicit_charges` gives the known charges for the
    other elements. Solves for the target element's charge.
    """
    if target_element not in composition:
        raise ValueError(f"target element {target_element} not in composition")
    n_target = composition[target_element]
    if n_target <= 0:
        raise ValueError(f"target element {target_element} has zero count")
    other_total = 0.0
    used = {}
    for el, n in composition.items():
        if el == target_element:
            continue
        if el not in explicit_charges:
            raise ValueError(f"no explicit charge for {el} in composition")
        q = float(explicit_charges[el])
        other_total += float(n) * q
        used[el] = q
    q_target = -other_total / float(n_target)
    text = (
        "sum(c_i * q_i) = 0 with "
        + " + ".join(f"{composition[el]}*{used[el]:+.4f}" for el in used)
        + f" + {n_target}*q_{target_element} = 0 -> q_{target_element} = {q_target:+.6f} e"
    )
    return ElectroneutralityDerived(
        element=target_element,
        charge=q_target,
        composition=composition,
        explicit_charges=used,
        derivation_text=text,
    )
