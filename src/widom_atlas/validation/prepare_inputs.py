"""Live-fetch the v0.4 validation inputs from licence-safe public sources.

Used by ``widom-atlas prepare-validation-inputs``. Resolves:

1. The 6 flagship-MOF CIFs from the RASPA2 GitHub repo
   (https://raw.githubusercontent.com/iRASPA/RASPA2/master/structures/...).
2. The RASPA2 ``ExampleMOFsForceField/force_field_mixing_rules.def`` table
   (UFF + DREIDING + García-Sánchez CO2 + TraPPE-N2/CH4 + Martin-Calvo Ar).
3. UserParameterFiles for each (MOF, gas) pair, projected from the FF table
   onto the MOF's element set.
4. 5 deterministic MOFX-DB records (default ids 173866-173870) with their
   inline CIF + simin (the FF spec is parseable from the simin string).

Caches:

- ``benchmarks/cache/structures/<MOF>.cif``
- ``benchmarks/cache/raspa2_ff/ExampleMOFsForceField_mixing_rules.def``
- ``benchmarks/cache/user_parameter_files/<MOF>_<gas>.json``
- ``benchmarks/cache/mofxdb/<id>.json``
- ``benchmarks/cache/mofxdb_cifs/<framework>.cif``

Public-source provenance per file is recorded in
``benchmarks/cache/provenance.json``.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FetchResult:
    target_path: str
    source_url: str
    bytes_fetched: int
    sha256: str
    status: str  # "ok" | "cached" | "error"
    notes: str = ""


@dataclass(frozen=True)
class PrepareInputsReport:
    structures: list[FetchResult]
    force_field: list[FetchResult]
    mofxdb: list[FetchResult]
    user_parameter_files: list[FetchResult]
    blockers: list[dict[str, Any]] = field(default_factory=list)


_FLAGSHIP_RASPA2_CIFS = [
    ("mofs", "MgMOF-74.cif", "MgMOF74.cif"),
    ("mofs", "Cu-BTC.cif", "HKUST-1.cif"),
    ("mofs", "UIO-66.cif", "UiO-66.cif"),
    ("zeolites", "CHA_SI.cif", "CHA.cif"),
    ("zeolites", "LTA_SI.cif", "LTA_NaK.cif"),
    ("zeolites", "MFI_SI.cif", "MFI.cif"),
]

_RASPA2_BASE = "https://raw.githubusercontent.com/iRASPA/RASPA2/master"
_FF_PATH = "forcefield/ExampleMOFsForceField/force_field_mixing_rules.def"


def _fetch_to_bytes(url: str, *, timeout_s: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "widom-atlas/0.4"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return r.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _db_name(db: Any) -> str:
    if isinstance(db, dict):
        return str(db.get("name") or "unknown")
    if isinstance(db, str):
        return db
    return str(db) if db else "unknown"


def fetch_flagship_structures(out_dir: Path) -> list[FetchResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []
    for kind, src_name, dst_name in _FLAGSHIP_RASPA2_CIFS:
        url = f"{_RASPA2_BASE}/structures/{kind}/cif/{src_name.replace(' ', '%20')}"
        target = out_dir / dst_name
        if target.exists():
            data = target.read_bytes()
            results.append(FetchResult(
                target_path=str(target), source_url=url,
                bytes_fetched=len(data), sha256=_sha256(data),
                status="cached", notes="already on disk; not re-fetched",
            ))
            continue
        try:
            data = _fetch_to_bytes(url)
            target.write_bytes(data)
            results.append(FetchResult(
                target_path=str(target), source_url=url,
                bytes_fetched=len(data), sha256=_sha256(data),
                status="ok", notes="live-fetched from RASPA2 master",
            ))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            results.append(FetchResult(
                target_path=str(target), source_url=url,
                bytes_fetched=0, sha256="",
                status="error", notes=f"fetch failed: {exc}",
            ))
    return results


def fetch_force_field(out_dir: Path) -> list[FetchResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "ExampleMOFsForceField_mixing_rules.def"
    url = f"{_RASPA2_BASE}/{_FF_PATH}"
    if target.exists():
        data = target.read_bytes()
        return [FetchResult(
            target_path=str(target), source_url=url,
            bytes_fetched=len(data), sha256=_sha256(data),
            status="cached", notes="already on disk",
        )]
    try:
        data = _fetch_to_bytes(url)
        target.write_bytes(data)
        return [FetchResult(
            target_path=str(target), source_url=url,
            bytes_fetched=len(data), sha256=_sha256(data),
            status="ok", notes="live-fetched from RASPA2 master",
        )]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return [FetchResult(
            target_path=str(target), source_url=url,
            bytes_fetched=0, sha256="", status="error", notes=f"fetch failed: {exc}",
        )]


def parse_raspa_mixing_rules(path: Path) -> dict[str, dict[str, float]]:
    """Parse the RASPA force_field_mixing_rules.def into {label: {sigma_A, epsilon_K}}.

    The file body has lines of the form ``<label> lennard-jones <eps_K> <sigma_A>``.
    Pseudo-atoms with ``none`` interaction (eg N_com virtual COM) are stored
    with sigma=0, epsilon=0 so the caller can detect and treat them specially.
    """
    table: dict[str, dict[str, float]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if len(toks) >= 2 and toks[1].lower() == "none":
            table[toks[0]] = {"sigma_A": 0.0, "epsilon_K": 0.0}
            continue
        if len(toks) >= 4 and toks[1].lower() in ("lennard-jones", "lj"):
            try:
                eps = float(toks[2])
                sig = float(toks[3])
            except ValueError:
                continue
            table[toks[0]] = {"sigma_A": sig, "epsilon_K": eps}
    if not table:
        raise RuntimeError(f"no FF entries parsed from {path}")
    return table


def _gas_template(gas: str, ff_table: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    if gas == "CO2":
        return [
            {"label": "C_co2", "sigma_A": ff_table["C_co2"]["sigma_A"],
             "epsilon_K": ff_table["C_co2"]["epsilon_K"], "charge_e": 0.6512,
             "source": "Garcia-Sanchez 2009", "doi": "10.1021/jp9035802"},
            {"label": "O_co2", "sigma_A": ff_table["O_co2"]["sigma_A"],
             "epsilon_K": ff_table["O_co2"]["epsilon_K"], "charge_e": -0.3256,
             "source": "Garcia-Sanchez 2009", "doi": "10.1021/jp9035802"},
        ]
    if gas == "N2":
        return [
            {"label": "N_n2", "sigma_A": ff_table["N_n2"]["sigma_A"],
             "epsilon_K": ff_table["N_n2"]["epsilon_K"], "charge_e": -0.482,
             "source": "TraPPE-N2 (Potoff-Siepmann 2001)", "doi": "10.1002/aic.690470719"},
            {"label": "N_com", "sigma_A": 1.0, "epsilon_K": 0.0, "charge_e": 0.964,
             "source": "TraPPE-N2 virtual COM site", "doi": "10.1002/aic.690470719"},
        ]
    if gas == "CH4":
        return [
            {"label": "CH4_ua", "sigma_A": ff_table["CH4"]["sigma_A"],
             "epsilon_K": ff_table["CH4"]["epsilon_K"], "charge_e": 0.0,
             "source": "TraPPE-UA (Martin-Siepmann 1998)", "doi": "10.1021/jp972543+"},
        ]
    if gas == "Ar":
        return [
            {"label": "Ar", "sigma_A": ff_table["Ar"]["sigma_A"],
             "epsilon_K": ff_table["Ar"]["epsilon_K"], "charge_e": 0.0,
             "source": "Martin-Calvo 2011", "doi": "10.1039/c1cp20531e"},
        ]
    raise ValueError(f"no gas template for {gas!r}")


def build_user_parameter_files(
    *,
    cif_dir: Path,
    ff_table: dict[str, dict[str, float]],
    pairs: list[tuple[str, str]],  # [(cif_filename, gas), ...]
    out_dir: Path,
) -> list[FetchResult]:
    """Build one UPF per (cif, gas) pair from the FF table + cif element set.

    Element-to-label mapping uses the RASPA `<element>_` UFF/DREIDING convention
    (e.g. ``Mg`` → ``Mg_``). Frameworks with charges defaulted to 0 (UFF
    baseline). Operator should override with DDEC6/PACMAN before expecting
    flagship-tier accuracy on OMS systems — recorded as a ``hybrid_warning``.
    """
    from ase.io import read as ase_read

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []
    for cif_name, gas in pairs:
        cif_path = cif_dir / cif_name
        if not cif_path.exists():
            results.append(FetchResult(
                target_path=str(out_dir / f"{cif_path.stem}_{gas}.json"),
                source_url=f"derived from {cif_path}",
                bytes_fetched=0, sha256="", status="error",
                notes=f"CIF not on disk: {cif_path}",
            ))
            continue
        atoms = ase_read(str(cif_path))
        if isinstance(atoms, list):
            atoms = atoms[0]
        elements = sorted(set(atoms.get_chemical_symbols()))

        fw_entries: list[dict[str, Any]] = []
        missing_elements: list[str] = []
        for elem in elements:
            label = f"{elem}_"
            ff = ff_table.get(label) or ff_table.get(elem)
            if ff is None:
                missing_elements.append(elem)
                continue
            fw_entries.append({
                "label": elem,
                "sigma_A": ff["sigma_A"],
                "epsilon_K": ff["epsilon_K"],
                "charge_e": 0.0,
                "source": "RASPA2/ExampleMOFsForceField (UFF + DREIDING)",
                "doi": "https://github.com/iRASPA/RASPA2",
            })

        target = out_dir / f"{cif_path.stem}_{gas}.json"
        if missing_elements:
            results.append(FetchResult(
                target_path=str(target),
                source_url=f"derived from {cif_path}",
                bytes_fetched=0, sha256="", status="error",
                notes=f"FF missing for elements {missing_elements}; UFF/DREIDING table does not cover them.",
            ))
            continue

        try:
            gas_sites = _gas_template(gas, ff_table)
        except (KeyError, ValueError) as exc:
            results.append(FetchResult(
                target_path=str(target),
                source_url=f"derived from {cif_path}",
                bytes_fetched=0, sha256="", status="error",
                notes=f"gas template failed: {exc}",
            ))
            continue

        payload = {
            "framework_atom_types": fw_entries,
            "gas_sites": gas_sites,
            "mixing_rules": "Lorentz-Berthelot",
            "electrostatics": "Wolf",
            "redistribution_status": "open_access_with_attribution",
            "hybrid_warning": (
                "Framework LJ params from RASPA2 ExampleMOFsForceField (UFF/DREIDING); "
                "framework charges defaulted to 0 (UFF baseline). For OMS-bearing MOFs "
                "(MgMOF74/HKUST-1) operator should override with DDEC6/PACMAN charges "
                "before expecting flagship-tier parity in absolute K_H magnitude."
            ),
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        target.write_text(text, encoding="utf-8")
        data = text.encode("utf-8")
        results.append(FetchResult(
            target_path=str(target),
            source_url=f"derived from {cif_path}",
            bytes_fetched=len(data), sha256=_sha256(data),
            status="ok",
            notes=f"projected {len(fw_entries)} fw types + {len(gas_sites)} gas sites",
        ))
    return results


def fetch_mofxdb_records(
    *, ids: list[int], cache_dir: Path, cif_dir: Path
) -> list[FetchResult]:
    """Live-fetch the deterministic MOFX-DB records and split out their inline CIFs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cif_dir.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []
    for mid in ids:
        url = f"https://mof.tech.northwestern.edu/mofs/{mid}.json"
        target = cache_dir / f"{mid}.json"
        try:
            if target.exists():
                data = target.read_bytes()
                status = "cached"
            else:
                data = _fetch_to_bytes(url)
                target.write_bytes(data)
                status = "ok"
            payload = json.loads(data)
            cif_text = payload.get("cif") or ""
            if cif_text:
                cif_path = cif_dir / f"{payload.get('name', f'mof_{mid}')}.cif"
                cif_path.write_text(cif_text, encoding="utf-8")
            results.append(FetchResult(
                target_path=str(target), source_url=url,
                bytes_fetched=len(data), sha256=_sha256(data),
                status=status,
                notes=(
                    f"MOFX record name={payload.get('name')} "
                    f"db={_db_name(payload.get('database'))}; "
                    f"cif_chars={len(cif_text)}; "
                    f"n_isotherms={len(payload.get('isotherms') or [])} "
                    f"n_heats={len(payload.get('heats') or [])}"
                ),
            ))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            results.append(FetchResult(
                target_path=str(target), source_url=url,
                bytes_fetched=0, sha256="", status="error",
                notes=f"fetch/parse failed: {exc}",
            ))
    return results


def write_provenance(report: PrepareInputsReport, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "structures": [asdict(r) for r in report.structures],
        "force_field": [asdict(r) for r in report.force_field],
        "mofxdb": [asdict(r) for r in report.mofxdb],
        "user_parameter_files": [asdict(r) for r in report.user_parameter_files],
        "blockers": report.blockers,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def prepare_validation_inputs(
    *,
    cache_root: Path,
    flagship_pairs: list[tuple[str, str]] | None = None,
    mofxdb_ids: list[int] | None = None,
) -> PrepareInputsReport:
    """End-to-end input preparation; called by the CLI."""
    structures_dir = cache_root / "structures"
    ff_dir = cache_root / "raspa2_ff"
    upf_dir = cache_root / "user_parameter_files"
    mofxdb_cache = cache_root / "mofxdb"
    mofxdb_cifs = cache_root / "mofxdb_cifs"

    structures = fetch_flagship_structures(structures_dir)
    ff = fetch_force_field(ff_dir)

    upfs: list[FetchResult] = []
    blockers: list[dict[str, Any]] = []
    if ff and ff[0].status in ("ok", "cached"):
        ff_table = parse_raspa_mixing_rules(Path(ff[0].target_path))
        if flagship_pairs is None:
            flagship_pairs = [
                ("MgMOF74.cif", "CO2"),
                ("HKUST-1.cif", "CO2"),
                ("UiO-66.cif", "CO2"),
                ("CHA.cif", "CO2"),
                ("LTA_NaK.cif", "CO2"),
                ("MFI.cif", "CH4"),
                ("MFI.cif", "CO2"),
                ("MFI.cif", "N2"),
            ]
        upfs = build_user_parameter_files(
            cif_dir=structures_dir, ff_table=ff_table,
            pairs=flagship_pairs, out_dir=upf_dir,
        )
        for u in upfs:
            if u.status == "error":
                blockers.append({
                    "kind": "upf_build",
                    "target_path": u.target_path,
                    "missing_item": "force-field parameters or gas template",
                    "needed_path": u.source_url,
                    "reason": u.notes,
                    "source_url": _RASPA2_BASE + "/" + _FF_PATH,
                    "licence_status": "MIT (RASPA2 redistributable)",
                })

    if mofxdb_ids is None:
        mofxdb_ids = [173866, 173867, 173868, 173869, 173870]
    mofx = fetch_mofxdb_records(
        ids=mofxdb_ids, cache_dir=mofxdb_cache, cif_dir=mofxdb_cifs,
    )

    report = PrepareInputsReport(
        structures=structures, force_field=ff, mofxdb=mofx,
        user_parameter_files=upfs, blockers=blockers,
    )
    write_provenance(report, cache_root / "provenance.json")
    return report


__all__ = [
    "FetchResult",
    "PrepareInputsReport",
    "build_user_parameter_files",
    "fetch_flagship_structures",
    "fetch_force_field",
    "fetch_mofxdb_records",
    "parse_raspa_mixing_rules",
    "prepare_validation_inputs",
    "write_provenance",
]
