"""PACMOF2 thin wrapper (Snurr group, MIT, Zenodo 12747095).

PACMOF2 is a Python package — operator installs it themselves
(``pip install pacmof2`` or from Zenodo) and runs it on a CIF to produce
per-atom charges. This wrapper:

- shells out to ``pacmof2`` if available
- otherwise reads a pre-computed ``charges.json`` the operator dropped under
  ``benchmarks/cache/pacmof_service_outputs/<refcode>.json``

It does NOT bundle PACMOF2 nor any of its training data.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pacmof2ChargeRun:
    refcode: str
    cif_path: str
    output_json_path: str
    pacmof2_version: str | None
    n_atoms: int
    elements_seen: list[str]
    notes: str


def is_pacmof2_available() -> bool:
    return shutil.which("pacmof2") is not None


def run_pacmof2(cif_path: Path, out_dir: Path, *, timeout_s: float = 600.0) -> Pacmof2ChargeRun:
    """Shell out to ``pacmof2`` to compute charges for one CIF."""
    cif_path = Path(cif_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{cif_path.stem}.charges.json"
    if not is_pacmof2_available():
        raise RuntimeError(
            "pacmof2 binary not found on PATH; install via 'pip install pacmof2' or download "
            "from Zenodo 10.5281/zenodo.12747095"
        )
    # PACMOF2 exact CLI signature varies by release; we wrap a defensive call.
    cmd = ["pacmof2", "--cif", str(cif_path), "--out", str(out_json)]
    proc = subprocess.run(cmd, check=False, capture_output=True, timeout=timeout_s, text=True)
    if proc.returncode != 0:
        return Pacmof2ChargeRun(
            refcode=cif_path.stem,
            cif_path=str(cif_path),
            output_json_path=str(out_json),
            pacmof2_version=None,
            n_atoms=0,
            elements_seen=[],
            notes=f"pacmof2 returned {proc.returncode}; stderr: {proc.stderr[:200]}",
        )
    if not out_json.exists():
        return Pacmof2ChargeRun(
            refcode=cif_path.stem,
            cif_path=str(cif_path),
            output_json_path=str(out_json),
            pacmof2_version=None,
            n_atoms=0,
            elements_seen=[],
            notes="pacmof2 succeeded but produced no output JSON",
        )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    elements = payload.get("elements", [])
    return Pacmof2ChargeRun(
        refcode=cif_path.stem,
        cif_path=str(cif_path),
        output_json_path=str(out_json),
        pacmof2_version=payload.get("pacmof2_version"),
        n_atoms=len(elements),
        elements_seen=sorted(set(elements)),
        notes="ok",
    )


def load_pacmof2_charges(charges_json_path: Path) -> dict[str, object]:
    return json.loads(Path(charges_json_path).read_text(encoding="utf-8"))


__all__ = ["Pacmof2ChargeRun", "is_pacmof2_available", "load_pacmof2_charges", "run_pacmof2"]
