"""Real-material fixture loader used by Layer 2 integration tests.

When a license-clean CIF is committed to ``tests/fixtures/real_structures/``,
this module loads it. Otherwise it falls back to a deterministic
``ase.build.bulk('Si', 'diamond')`` stand-in tagged ``synthetic_stand_in``
in the metadata so reports never claim chemical fidelity from this path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

REAL_BENCHMARK_IDS: Final[tuple[str, ...]] = (
    "Mg-MOF-74",
    "UiO-66",
    "ZIF-8",
    "MOF-5",
    "MFI",
    "CHA",
)

_FIXTURES_DIR_CANDIDATES: Final[tuple[str, ...]] = (
    "tests/fixtures/real_structures",
)


def _resolve_fixtures_dir() -> Path:
    cwd = Path.cwd()
    for rel in _FIXTURES_DIR_CANDIDATES:
        candidate = cwd / rel
        if candidate.exists():
            return candidate
    return cwd / _FIXTURES_DIR_CANDIDATES[0]


def load_real_material(material_id: str) -> tuple[Any, dict[str, Any]]:
    """Return ``(atoms, metadata)`` for the requested material id.

    Uses a committed CIF when available; otherwise a Si-diamond stand-in.
    """
    fixtures = _resolve_fixtures_dir()
    cif_path = fixtures / f"{material_id}.cif"
    metadata: dict[str, Any] = {"requested_material_id": material_id}
    if cif_path.exists():
        from ase.io import read

        atoms = read(str(cif_path))
        if hasattr(atoms, "set_pbc"):
            atoms.set_pbc(True)
        metadata.update(
            {
                "source": "committed_cif",
                "cif_path": str(cif_path),
                "stand_in": False,
            }
        )
        return atoms, metadata

    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_pbc(True)
    metadata.update(
        {
            "source": "synthetic_stand_in",
            "stand_in_kind": "Si_diamond_5.43A",
            "stand_in": True,
            "warning": "synthetic stand-in — NOT real " + material_id,
        }
    )
    return atoms, metadata


__all__ = ["REAL_BENCHMARK_IDS", "load_real_material"]
