"""T006: Lin/Mercado column-ordering decoder.

The Lin 2014 / Mercado SI Table S7 stores Buckingham parameters in the
column order (A, B, C) where C is the dispersion coefficient ALREADY
multiplied by the per-pair scaling factor S_g.

This decoder is the highest silent-corruption-risk parser in the build:
the wrong column order produces a plausible-looking but physically
wrong Buckingham well. Every row pass through this function must be
verified by the unit test `test_lin_mercado_column_ordering`.
"""
from __future__ import annotations

from collections.abc import Mapping

from .terms import BuckinghamLinMercado


def decode_lin_mercado_row(
    row: Mapping[str, float] | tuple[float, float, float],
    *,
    S_g: float = 1.0,
    C_already_scaled: bool = True,
) -> BuckinghamLinMercado:
    """Decode a Mercado SI row to a BuckinghamLinMercado term.

    Accepts either:
    - dict with keys 'A', 'B', 'C' (case-sensitive, in K, 1/Å, K·Å^6)
    - 3-tuple ordered (A, B, C)

    A swapped ordering (e.g., the parser feeding (C, B, A)) will produce
    wildly wrong magnitudes; the test asserts dispersion (~ 1e5-1e7) > B (~ 1-10).
    """
    if isinstance(row, Mapping):
        A = float(row["A"])
        B = float(row["B"])
        C = float(row["C"])
    else:
        if len(row) != 3:
            raise ValueError(f"Lin/Mercado row must have 3 entries, got {len(row)}: {row!r}")
        A, B, C = float(row[0]), float(row[1]), float(row[2])

    if A <= 0:
        raise ValueError(f"Lin/Mercado A must be > 0 (repulsive prefactor); got {A}")
    if B <= 0:
        raise ValueError(f"Lin/Mercado B must be > 0 (decay rate); got {B}")
    if C <= 0:
        raise ValueError(f"Lin/Mercado C must be > 0 (dispersion); got {C}")
    # Sanity heuristic: A typically 1e4-1e8 K, B typically 2-6 /Å,
    # C typically 1e4-1e8 K·Å^6. If B > 100 or A < 100, columns are swapped.
    if B > 100 or A < 100:
        raise ValueError(
            f"Lin/Mercado column ordering looks swapped: A={A}, B={B}, C={C}. "
            "Expected A ~ 1e4-1e8 K, B ~ 1-10 /Å, C ~ 1e4-1e8 K·Å^6."
        )

    return BuckinghamLinMercado(
        A_K=A,
        B_per_angstrom=B,
        C_K_angstrom6=C,
        S_g=S_g,
        C_already_scaled=C_already_scaled,
    )
