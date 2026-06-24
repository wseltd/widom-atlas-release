"""EQeq thin wrapper (numat/EQeq, GPL-2.0).

The operator clones + builds EQeq from GitHub themselves. We do not bundle
its source. This wrapper shells out to the ``eqeq`` binary on PATH; without
it, the function returns a structured error.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EqeqRun:
    refcode: str
    cif_path: str
    output_path: str | None
    eqeq_version: str | None
    n_atoms: int
    notes: str


def is_eqeq_available() -> bool:
    return shutil.which("eqeq") is not None


def run_eqeq(cif_path: Path, out_dir: Path, *, timeout_s: float = 600.0) -> EqeqRun:
    cif_path = Path(cif_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cif_path.stem}.eqeq.json"
    if not is_eqeq_available():
        return EqeqRun(
            refcode=cif_path.stem,
            cif_path=str(cif_path),
            output_path=None,
            eqeq_version=None,
            n_atoms=0,
            notes="eqeq binary not on PATH; clone numat/EQeq, build, and put 'eqeq' on PATH",
        )
    cmd = ["eqeq", str(cif_path), "-o", str(out_path)]
    proc = subprocess.run(cmd, check=False, capture_output=True, timeout=timeout_s, text=True)
    if proc.returncode != 0:
        return EqeqRun(
            refcode=cif_path.stem,
            cif_path=str(cif_path),
            output_path=None,
            eqeq_version=None,
            n_atoms=0,
            notes=f"eqeq returned {proc.returncode}; stderr: {proc.stderr[:200]}",
        )
    return EqeqRun(
        refcode=cif_path.stem,
        cif_path=str(cif_path),
        output_path=str(out_path),
        eqeq_version=None,
        n_atoms=0,
        notes="ok (parse output_path for charges)",
    )


__all__ = ["EqeqRun", "is_eqeq_available", "run_eqeq"]
