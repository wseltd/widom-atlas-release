"""RASPA3 → widom-atlas external-samples adapter.

Per ``implementation-verdict-continuation.txt`` §"Implement RASPA3 path"
(and the v0.4 follow-up brief §5 which **corrected** the per-insertion
assumption): RASPA3 is MIT-licensed and the most practical external
Widom engine, but it **cannot** natively export per-insertion XYZ
positions or energies. The internal ``growData`` struct in
``raspakit/mc_moves/component/widom.cpp`` is overwritten every step;
``WriteMoviesEvery`` only tracks the accepted trajectory of the
combined run, not the failed / accessibility-only insertion attempts a
Widom-style atlas analysis needs.

This module therefore supports **scalar mode only** end-to-end. The
``Raspa3PerInsertionResult`` dataclass + ``write_canonical_external_samples``
writer are kept available for operators who patch ``raspakit/mc_moves/
component/widom.cpp`` to stream the per-insertion data themselves; that
fork is out of scope for v0.4 / v0.5. For genuine per-insertion atlas
input the recommended v0.4 path is widom-atlas's own internal NumPy /
JAX energy evaluator driven by a force-field template (e.g. the RASPA3
``examples/basic/12_mc_henry_coefficient_of_co2_n2_methane_in_mfi``
templates registered as ``RASPA3-templates-MFI-henry`` in
``data_registry/data/datasets.yaml``).

Modes
=====

1. **Scalar mode** (always available). Parses per-cycle Widom
   averages from a RASPA3 ``Output/`` directory and writes a small
   JSON sidecar that drives the **scalar comparator only**. No NPZ is
   produced. The atlas pipeline cannot run on this output; the sidecar
   feeds the benchmark / launch-report's scalar comparison rows.

2. **Per-insertion writer** (operator-supplied data). The
   :func:`write_canonical_external_samples` writer accepts (positions,
   energies, accessible) tuples that the operator has obtained through
   their own means — typically a patched ``raspakit`` or an external
   engine — and writes the canonical ``samples.npz`` +
   :class:`~widom_atlas.backends.schema.ExternalSampleManifest` sidecar.
   This path is **not** wired to a vanilla RASPA3 run.

We deliberately keep RASPA3 *out* of the import path: this module reads
RASPA3 output files. If RASPA3 is not installed locally that is fine —
we are an offline parser.

Format references
=================

Output directory layout (RASPA3 v3.x, observed in the wild):

::

   raspa_run/
     simulation.input
     Output/
       System_<N>/
         output_<framework>_<cycle>.data    # text — accumulated averages

The exact file names vary across RASPA3 versions; the parser globs
defensively and records what it actually read in the manifest.

Energies in RASPA's internal output are in **K** (i.e. ``ε / k_B``); we
keep them in K through to the manifest so the
:class:`ExternalSamplesBackend` does the K → eV conversion via
:func:`widom_atlas.backends.units.to_eV`. **No silent unit assumptions.**
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .schema import (
    SAMPLE_FORMAT_VERSION,
    CitationEntry,
    ExternalSampleManifest,
    ForceFieldDescriptor,
)

_LOGGER = logging.getLogger(__name__)

# RASPA3's input grammar uses simple "Key Value" lines; this captures
# the keys we need to populate the manifest. Full RASPA grammar is
# richer but this subset is what's relevant for Widom-style runs.
_INPUT_KEY_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+(.+?)\s*$")


@dataclass(frozen=True)
class Raspa3InputDescriptor:
    """Parsed contents of a RASPA3 ``simulation.input`` file."""

    framework_name: str
    gas: str
    temperature_K: float
    n_insertions: int
    n_initialization_cycles: int
    forcefield: str
    extra: dict[str, str] = field(default_factory=dict)
    raw_path: str = ""


@dataclass(frozen=True)
class Raspa3ScalarResult:
    """Scalar-only result of a RASPA3 run (always available)."""

    framework_name: str
    gas: str
    temperature_K: float
    n_insertions: int
    henry_coefficient_mol_per_kg_per_Pa: float | None
    heat_of_adsorption_kJ_per_mol: float | None
    raspa_version: str
    output_dir: str
    output_files_sha256: dict[str, str]
    parser_warnings: list[str]


def _parse_simulation_input(path: Path) -> Raspa3InputDescriptor:
    """Parse a RASPA3 ``simulation.input`` file. Best-effort, defensive."""
    text = path.read_text(encoding="utf-8", errors="replace")
    keys: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _INPUT_KEY_RE.match(line)
        if m:
            k, v = m.group(1), m.group(2)
            # Keep first occurrence; RASPA accepts repeats but we don't aggregate.
            keys.setdefault(k, v)
    framework_name = (
        keys.get("FrameworkName")
        or keys.get("Framework0")
        or keys.get("frameworkName")
        or keys.get("framework_name")
        or "unknown"
    )
    gas = (
        keys.get("Component0")
        or keys.get("MoleculeName")
        or keys.get("ComponentName0")
        or keys.get("MoleculeName0")
        or "unknown"
    )
    # gas is sometimes "0 MoleculeName CO2" style — normalise:
    gas = gas.split()[-1] if gas != "unknown" else gas
    forcefield = keys.get("Forcefield") or keys.get("ForceField") or keys.get("forcefield") or "unknown"
    try:
        temperature_K = float(keys.get("ExternalTemperature") or keys.get("Temperature") or 0.0)
    except ValueError:
        temperature_K = 0.0
    try:
        n_insertions = int(float(keys.get("NumberOfCycles") or keys.get("NumberOfWidomInsertions") or 0))
    except ValueError:
        n_insertions = 0
    try:
        n_init = int(float(keys.get("NumberOfInitializationCycles") or 0))
    except ValueError:
        n_init = 0

    return Raspa3InputDescriptor(
        framework_name=str(framework_name),
        gas=str(gas),
        temperature_K=float(temperature_K),
        n_insertions=int(n_insertions),
        n_initialization_cycles=int(n_init),
        forcefield=str(forcefield),
        extra=keys,
        raw_path=str(path),
    )


_HENRY_LINE_RE = re.compile(
    r"Henry\s*coeff(icient)?[^\d\-]*([+-]?\d+\.?\d*[eE]?[+-]?\d*)\s*\[?(mol/kg/Pa|mol/kg.Pa|mol kg-1 Pa-1)?",
    re.IGNORECASE,
)
_QADS_LINE_RE = re.compile(
    r"(Heat of adsorption|enthalpy of adsorption|isosteric heat)[^\d\-]*([+-]?\d+\.?\d*[eE]?[+-]?\d*)\s*\[?(K|kJ/mol|kcal/mol)?",
    re.IGNORECASE,
)
_RASPA_VERSION_RE = re.compile(r"RASPA[ \t]*version[ \t:]*([0-9.]+)", re.IGNORECASE)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_raspa3_scalars(raspa_dir: Path) -> Raspa3ScalarResult:
    """Parse the scalar Widom outputs from a RASPA3 run directory.

    Looks for ``Output/System_*/output_*.data`` files, extracts
    ``Henry coefficient`` and ``Heat of adsorption`` from the text, and
    records sha256 of every file consulted. Returns ``None`` for fields
    that could not be parsed; these get echoed as parser_warnings.

    Note: RASPA3 does **not** export per-insertion guest positions /
    energies (the brief §5 confirms this — ``raspakit/mc_moves/component/
    widom.cpp`` overwrites the ``growData`` struct each step). This
    parser therefore returns scalars only; for atlas input use the
    ``write_canonical_external_samples`` writer with operator-supplied
    per-insertion data, or run widom-atlas's own internal energy
    evaluator with a RASPA3 force-field template.
    """
    raspa_dir = Path(raspa_dir)
    sim_input_path = raspa_dir / "simulation.input"
    if not sim_input_path.exists():
        raise FileNotFoundError(f"no simulation.input at {sim_input_path}")
    desc = _parse_simulation_input(sim_input_path)

    output_files_sha: dict[str, str] = {str(sim_input_path): _hash_file(sim_input_path)}
    output_dir = raspa_dir / "Output"
    out_files = sorted(output_dir.glob("System_*/output_*.data")) if output_dir.exists() else []
    if not out_files:
        # Some RASPA3 layouts use Output/System_*/output_*.s*.data — try a wider glob.
        out_files = sorted(output_dir.glob("System_*/output_*.data*")) if output_dir.exists() else []

    parser_warnings: list[str] = []
    raspa_version = "unknown"
    henry: float | None = None
    qads_kJ_per_mol: float | None = None

    if not out_files:
        parser_warnings.append(
            f"no Output/System_*/output_*.data found under {raspa_dir}; "
            "scalar parse will be empty (RASPA3 may not have completed, or layout changed)"
        )
    for of in out_files:
        output_files_sha[str(of)] = _hash_file(of)
        text = of.read_text(encoding="utf-8", errors="replace")
        if raspa_version == "unknown":
            m_v = _RASPA_VERSION_RE.search(text)
            if m_v:
                raspa_version = m_v.group(1)
        for m in _HENRY_LINE_RE.finditer(text):
            try:
                henry = float(m.group(2))
            except (ValueError, IndexError):
                continue
        for m in _QADS_LINE_RE.finditer(text):
            try:
                value = float(m.group(2))
                unit = (m.group(3) or "K").strip()
                if unit == "K":
                    qads_kJ_per_mol = value * 8.314462618 / 1000.0
                elif unit.lower() == "kj/mol":
                    qads_kJ_per_mol = value
                elif unit.lower() == "kcal/mol":
                    qads_kJ_per_mol = value * 4.184
            except (ValueError, IndexError):
                continue

    if henry is None:
        parser_warnings.append("no 'Henry coefficient' line matched — KH unavailable")
    if qads_kJ_per_mol is None:
        parser_warnings.append("no 'Heat of adsorption' / 'enthalpy' / 'isosteric heat' line matched — Q_ads unavailable")

    return Raspa3ScalarResult(
        framework_name=desc.framework_name,
        gas=desc.gas,
        temperature_K=desc.temperature_K,
        n_insertions=desc.n_insertions,
        henry_coefficient_mol_per_kg_per_Pa=henry,
        heat_of_adsorption_kJ_per_mol=qads_kJ_per_mol,
        raspa_version=raspa_version,
        output_dir=str(raspa_dir),
        output_files_sha256=output_files_sha,
        parser_warnings=parser_warnings,
    )


@dataclass(frozen=True)
class Raspa3PerInsertionResult:
    """Per-insertion (positions + energies) result of a RASPA3 Widom run.

    Only available when the operator configured ``WriteMoviesEvery 1`` (or
    equivalent) so per-frame guest coordinates were dumped.
    """

    descriptor: Raspa3InputDescriptor
    raspa_version: str
    positions_cart_A: np.ndarray  # (N, 3)
    energies_K: np.ndarray  # (N,) in K
    accessible: np.ndarray  # (N,) bool
    cell_matrix_A: np.ndarray  # (3, 3)
    parser_warnings: list[str]


def write_canonical_external_samples(
    *,
    out_npz: Path,
    out_manifest: Path,
    framework: str,
    gas: Literal["CO2", "N2", "CH4"],
    temperature_K: float,
    positions_cart_A: np.ndarray,
    energies: np.ndarray,
    energy_unit: Literal["K", "eV", "kJ_mol", "kcal_mol"],
    accessible: np.ndarray,
    cell_matrix_A: np.ndarray,
    n_insertions: int,
    backend_version: str,
    force_field: ForceFieldDescriptor,
    citations: list[CitationEntry],
    redistribution_status: Literal[
        "bundled_safe",
        "user_supplied_not_bundled",
        "user_supplied_not_redistributed",
        "open_access_with_attribution",
        "unknown",
    ],
    warnings: list[str],
    suitable_for_quantitative_interpretation: bool,
    random_seed: int | None = None,
    samples_origin_engine: str = "RASPA3",
) -> ExternalSampleManifest:
    """Write a canonical (samples.npz, samples.npz.manifest.json) pair.

    The npz schema matches :func:`widom_atlas.io.npz.save_samples_npz`
    (positions_cart, positions_frac, energies_eV, accessible, cell_matrix,
    temperature_K, metadata_json) — but with **energies in eV** in the
    npz (converted from the declared unit) so the existing
    :func:`from_npz` loader works unchanged. The manifest sidecar still
    records the original ``energy_unit`` for full provenance.
    """
    from widom_atlas.backends.units import to_eV

    out_npz = Path(out_npz)
    out_manifest = Path(out_manifest)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    energies_eV = to_eV(np.asarray(energies, dtype=np.float64), energy_unit)

    pos_cart = np.asarray(positions_cart_A, dtype=np.float64)
    if pos_cart.ndim != 2 or pos_cart.shape[1] != 3:
        raise ValueError(f"positions_cart_A must be (N, 3); got {pos_cart.shape}")
    if pos_cart.shape[0] != energies_eV.shape[0]:
        raise ValueError("positions_cart_A and energies must have the same N")
    if accessible.shape != energies_eV.shape:
        raise ValueError("accessible and energies must have the same shape")

    # Compute fractional positions for the npz schema — minimal duplication.
    cell = np.asarray(cell_matrix_A, dtype=np.float64)
    if cell.shape != (3, 3):
        raise ValueError(f"cell_matrix_A must be (3, 3); got {cell.shape}")
    frac = np.linalg.solve(cell.T, pos_cart.T).T

    metadata_blob = json.dumps(
        {
            "gas": gas,
            "structure_id": framework,
            "metadata": {"samples_origin_engine": samples_origin_engine},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    np.savez_compressed(
        out_npz,
        positions_cart=pos_cart,
        positions_frac=frac,
        energies_eV=energies_eV,
        accessible=np.asarray(accessible, dtype=bool),
        cell_matrix=cell,
        temperature_K=np.asarray(temperature_K, dtype=np.float64),
        metadata_json=np.asarray(metadata_blob),
    )
    sha_npz = _hash_file(out_npz)

    manifest = ExternalSampleManifest(
        sample_format_version=SAMPLE_FORMAT_VERSION,
        framework=framework,
        gas=gas,
        temperature_K=float(temperature_K),
        backend="raspa3_external",
        backend_version=backend_version,
        n_insertions=int(n_insertions),
        random_seed=random_seed,
        energy_unit=energy_unit,
        parameter_mode="external_samples",
        force_field=force_field,
        citations=citations,
        redistribution_status=redistribution_status,
        warnings=warnings,
        suitable_for_quantitative_interpretation=suitable_for_quantitative_interpretation,
        samples_path=str(out_npz),
        samples_sha256=sha_npz,
    )
    out_manifest.write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_scalar_only_sidecar(
    *,
    out_path: Path,
    scalar_result: Raspa3ScalarResult,
    force_field_label: str,
    framework_charge_source: str,
    gas_model: str,
    citations: list[dict[str, str]],
    warnings: list[str],
) -> Path:
    """Write a scalar-only RASPA3 ingest sidecar (no atlas inputs)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "0.3-scalar-only",
        "framework": scalar_result.framework_name,
        "gas": scalar_result.gas,
        "temperature_K": scalar_result.temperature_K,
        "n_insertions": scalar_result.n_insertions,
        "raspa_version": scalar_result.raspa_version,
        "henry_coefficient_mol_per_kg_per_Pa": scalar_result.henry_coefficient_mol_per_kg_per_Pa,
        "heat_of_adsorption_kJ_per_mol": scalar_result.heat_of_adsorption_kJ_per_mol,
        "force_field_label": force_field_label,
        "framework_charge_source": framework_charge_source,
        "gas_model": gas_model,
        "citations": citations,
        "warnings": warnings + scalar_result.parser_warnings,
        "atlas_input": False,
        "note": (
            "Scalar comparator only. RASPA3 cannot produce atlas samples through this "
            "path because per-insertion guest positions were not exported. To enable "
            "atlas input, configure RASPA3 with WriteMoviesEvery 1 and re-run."
        ),
        "output_files_sha256": scalar_result.output_files_sha256,
    }
    out_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return out_path


__all__ = [
    "Raspa3InputDescriptor",
    "Raspa3PerInsertionResult",
    "Raspa3ScalarResult",
    "parse_raspa3_scalars",
    "write_canonical_external_samples",
    "write_scalar_only_sidecar",
]
