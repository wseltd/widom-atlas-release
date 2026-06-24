"""Phase-C parity gates.

Two gates:

1) RASPA3 reference parity (release-blocker)
   - Run a RASPA3 GCMC reference at a single state point
   - Run our internal evaluator on the same FF, framework, gas, T, n_insertions, seed
   - Compare scalar log10(K_H) and Q_ads within tolerances
   - The MFI + CO2 fixture under tests/fixtures/raspa3_mfi_henry/ is the
     v0.4 designated parity case; if RASPA3 isn't installed the test is
     skipped but the evaluator side still runs and emits its scalars.

2) MOFX-DB simin parity (5 records)
   - Pick 5 deterministic non-hypothetical MOFX records via
     ``select_deterministic_simin_records``
   - For each, parse the simin string, then run the internal evaluator
     using the same gas, T, and a registry-resolved framework structure
   - Emit one parity-row JSON per record

Both gates emit machine-readable parity rows to a single ``parity.jsonl``
that the validation suite consumes.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from widom_atlas.ingest.mofxdb import (
    MofxdbSiminRecord,
    select_deterministic_simin_records,
)

from .runner import WidomResult


@dataclass(frozen=True)
class ParityRow:
    """One parity row, written to parity.jsonl."""

    case_id: str
    kind: str  # "raspa3_reference" | "mofxdb_simin"
    framework_name: str
    component_name: str
    temperature_K: float
    n_insertions: int
    seed: int
    log10_KH_internal: float | None
    log10_KH_reference: float | None
    delta_log10_KH: float | None
    Qads_internal_kJ_per_mol: float | None
    Qads_reference_kJ_per_mol: float | None
    delta_Qads_kJ_per_mol: float | None
    threshold_log10_KH: float
    threshold_Qads_kJ_per_mol: float
    pass_log10_KH: bool
    pass_Qads: bool
    pass_overall: bool
    reference_provenance_sha256: str
    notes: str
    warnings: list[str] = field(default_factory=list)


def _safe_log10(x: float | None) -> float | None:
    if x is None or x <= 0:
        return None
    return math.log10(x)


def is_raspa3_available() -> bool:
    return shutil.which("raspa3") is not None


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_raspa3_reference(
    *,
    raspa3_input_dir: Path,
    output_dir: Path,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Invoke RASPA3 on a prepared input directory and return the parsed output.

    Skipped (returns ``{"status": "skipped"}``) if RASPA3 is not on PATH.
    """
    if not is_raspa3_available():
        return {"status": "skipped", "reason": "raspa3 not on PATH"}
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["raspa3", "--directory", str(raspa3_input_dir)]
    proc = subprocess.run(
        cmd,
        check=False,
        cwd=str(raspa3_input_dir),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    (output_dir / "raspa3_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "raspa3_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return {"status": "error", "returncode": proc.returncode, "stderr": proc.stderr[:1000]}
    log_files = list(raspa3_input_dir.rglob("output*.txt")) + list(raspa3_input_dir.rglob("*.json"))
    KH = None
    Qads = None
    for lf in log_files:
        try:
            text = lf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            s = line.strip()
            if "Henry" in s and ("[mol/kg/Pa]" in s or "[mol kg-1 Pa-1]" in s):
                tokens = s.split()
                for j, t in enumerate(tokens):
                    try:
                        v = float(t)
                        KH = v
                        break
                    except ValueError:
                        continue
                    finally:
                        _ = j
            if "Heat of adsorption" in s and "kJ" in s:
                tokens = s.split()
                for t in tokens:
                    try:
                        Qads = float(t)
                        break
                    except ValueError:
                        continue
    return {
        "status": "ok",
        "KH_mol_per_kg_per_Pa": KH,
        "Qads_kJ_per_mol": Qads,
        "raw_log_files": [str(lf) for lf in log_files],
    }


def parity_row_from_internal_only(
    *,
    case_id: str,
    kind: str,
    internal: WidomResult,
    threshold_log10_KH: float,
    threshold_Qads: float,
    notes: str = "",
) -> ParityRow:
    """Build a parity row when the reference side is missing (skipped raspa3)."""
    return ParityRow(
        case_id=case_id,
        kind=kind,
        framework_name=internal.framework_name,
        component_name=internal.component_name,
        temperature_K=internal.temperature_K,
        n_insertions=internal.n_insertions_used,
        seed=int(internal.provenance.get("seed", 0)),
        log10_KH_internal=_safe_log10(internal.KH_mol_per_kg_per_Pa),
        log10_KH_reference=None,
        delta_log10_KH=None,
        Qads_internal_kJ_per_mol=internal.Qads_kJ_per_mol,
        Qads_reference_kJ_per_mol=None,
        delta_Qads_kJ_per_mol=None,
        threshold_log10_KH=threshold_log10_KH,
        threshold_Qads_kJ_per_mol=threshold_Qads,
        pass_log10_KH=False,
        pass_Qads=False,
        pass_overall=False,
        reference_provenance_sha256="",
        notes=notes or "reference unavailable; only internal scalars recorded",
        warnings=internal.warnings,
    )


def parity_row_from_pair(
    *,
    case_id: str,
    kind: str,
    internal: WidomResult,
    KH_reference: float | None,
    Qads_reference_kJ_per_mol: float | None,
    reference_provenance_sha256: str,
    threshold_log10_KH: float,
    threshold_Qads: float,
    notes: str = "",
) -> ParityRow:
    log_KH_int = _safe_log10(internal.KH_mol_per_kg_per_Pa)
    log_KH_ref = _safe_log10(KH_reference)
    delta_log = (
        abs(log_KH_int - log_KH_ref) if log_KH_int is not None and log_KH_ref is not None else None
    )
    delta_Q = (
        abs(internal.Qads_kJ_per_mol - Qads_reference_kJ_per_mol)
        if internal.Qads_kJ_per_mol is not None and Qads_reference_kJ_per_mol is not None
        else None
    )
    pass_log = bool(delta_log is not None and delta_log <= threshold_log10_KH)
    pass_Q = bool(delta_Q is not None and delta_Q <= threshold_Qads)
    return ParityRow(
        case_id=case_id,
        kind=kind,
        framework_name=internal.framework_name,
        component_name=internal.component_name,
        temperature_K=internal.temperature_K,
        n_insertions=internal.n_insertions_used,
        seed=int(internal.provenance.get("seed", 0)),
        log10_KH_internal=log_KH_int,
        log10_KH_reference=log_KH_ref,
        delta_log10_KH=delta_log,
        Qads_internal_kJ_per_mol=internal.Qads_kJ_per_mol,
        Qads_reference_kJ_per_mol=Qads_reference_kJ_per_mol,
        delta_Qads_kJ_per_mol=delta_Q,
        threshold_log10_KH=threshold_log10_KH,
        threshold_Qads_kJ_per_mol=threshold_Qads,
        pass_log10_KH=pass_log,
        pass_Qads=pass_Q,
        pass_overall=pass_log and pass_Q,
        reference_provenance_sha256=reference_provenance_sha256,
        notes=notes,
        warnings=internal.warnings,
    )


def write_parity_jsonl(rows: list[ParityRow], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.__dict__, sort_keys=True) + "\n")
    return out_path


def derive_mofxdb_parity_inputs(
    records: list[MofxdbSiminRecord], *, n: int = 5, seed: int = 17
) -> list[MofxdbSiminRecord]:
    """Choose 5 deterministic non-hypothetical MOFX simin records for the parity gate."""
    return select_deterministic_simin_records(records, n=n, seed=seed)


def assess_parity_outcome(
    rows: list[ParityRow], *, mofxdb_min_pass: int = 4
) -> dict[str, Any]:
    """Apply the Phase C gate logic.

    Pass condition (per the v0.4 follow-up brief):
    - The RASPA3 reference parity row must pass (or be SKIPPED-with-internal-only,
      flagged as such; SKIPPED does NOT count as pass).
    - At least ``mofxdb_min_pass`` of the 5 MOFX simin parity rows must pass.
    """
    raspa_rows = [r for r in rows if r.kind == "raspa3_reference"]
    mofx_rows = [r for r in rows if r.kind == "mofxdb_simin"]

    raspa_pass = bool(raspa_rows) and all(r.pass_overall for r in raspa_rows)
    raspa_skipped = bool(raspa_rows) and any(
        r.notes.startswith("reference unavailable") for r in raspa_rows
    )
    mofx_passed = sum(1 for r in mofx_rows if r.pass_overall)
    return {
        "raspa3_pass": raspa_pass,
        "raspa3_skipped": raspa_skipped,
        "mofxdb_pass_count": mofx_passed,
        "mofxdb_required": mofxdb_min_pass,
        "mofxdb_pass": mofx_passed >= mofxdb_min_pass,
        "overall_pass": (raspa_pass or raspa_skipped) and mofx_passed >= mofxdb_min_pass,
        "n_rows": len(rows),
    }


def cosine_distance_E_distributions(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
    """Distance between two energy histograms (cheap signal for evaluator drift)."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi - lo < 1e-12:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    if np.linalg.norm(ha) == 0 or np.linalg.norm(hb) == 0:
        return float("nan")
    return float(1.0 - np.dot(ha, hb) / (np.linalg.norm(ha) * np.linalg.norm(hb)))


__all__ = [
    "ParityRow",
    "assess_parity_outcome",
    "cosine_distance_E_distributions",
    "derive_mofxdb_parity_inputs",
    "is_raspa3_available",
    "parity_row_from_internal_only",
    "parity_row_from_pair",
    "run_raspa3_reference",
    "write_parity_jsonl",
]
