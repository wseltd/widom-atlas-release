"""T007: Dzubak 2012 two-attraction pair-energy evaluator.

U(r) = A*exp(-B*r) - C5/r^5 - D6/r^6

`A` in K, `B` in 1/Å, `C5` in K·Å^5, `D6` in K·Å^6.

This form is DIFFERENT from Lin/Mercado (single-attraction) and is the
single highest source of confusion in the v0.4 force-field stack.
The unit test verifies that swapping C5/D6 produces a measurably
different curve, ensuring the decoder does not silently corrupt rows.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .terms import DzubakAExpC5D6


def decode_dzubak_row(
    row: Mapping[str, float] | tuple[float, float, float, float],
) -> DzubakAExpC5D6:
    """Decode a Dzubak 2012 SI Table SI4 row.

    Accepts dict with keys A, B, C5, D6 or 4-tuple (A, B, C5, D6).

    Magnitude heuristic (Mg-MOF-74 from Dzubak SI 4):
      A ≈ 4e7 K, B ≈ 4 /Å, C5 ≈ 0 (or small), D6 ≈ 4e5 K·Å^6.
    """
    if isinstance(row, Mapping):
        A = float(row["A"])
        B = float(row["B"])
        C5 = float(row["C5"])
        D6 = float(row["D6"])
    else:
        if len(row) != 4:
            raise ValueError(f"Dzubak row must have 4 entries, got {len(row)}: {row!r}")
        A, B, C5, D6 = (float(x) for x in row)

    if A <= 0:
        raise ValueError(f"Dzubak A must be > 0; got {A}")
    if B <= 0:
        raise ValueError(f"Dzubak B must be > 0; got {B}")
    if D6 < 0:
        raise ValueError(f"Dzubak D6 must be >= 0; got {D6}")
    if C5 < 0:
        raise ValueError(f"Dzubak C5 must be >= 0; got {C5}")
    if B > 100:
        raise ValueError(f"Dzubak B looks wrong: {B}; expected ~1-10 /Å")

    return DzubakAExpC5D6(
        A_K=A,
        B_per_angstrom=B,
        C5_K_angstrom5=C5,
        D6_K_angstrom6=D6,
    )


def dzubak_energy(r: float, term: DzubakAExpC5D6) -> float:
    """Evaluate the Dzubak two-attraction pair energy at distance r (Å). Returns K."""
    return (
        term.A_K * math.exp(-term.B_per_angstrom * r)
        - term.C5_K_angstrom5 / (r ** 5)
        - term.D6_K_angstrom6 / (r ** 6)
    )
