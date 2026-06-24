"""T025: RASPA3 v3.0.29 Ewald subprocess runner.

Produces fresh evidence per invocation. NEVER caches outputs. Records:
- RASPA3 binary path + sha + version
- input file sha256s
- stdout + stderr verbatim
- run timestamp
- exit code
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..raspa_binary import DEFAULT_RASPA3_PATH, verify_raspa3_binary
from .input_writer import RaspaInputBundle


@dataclass(frozen=True)
class RaspaRunResult:
    work_dir: Path
    exit_code: int
    duration_s: float
    stdout_path: Path
    stderr_path: Path
    output_txt_path: Path | None
    raspa3_version: str
    raspa3_sha256: str


def run_raspa3(
    bundle: RaspaInputBundle,
    raspa3_binary: Path | None = None,
    timeout_s: int = 1800,
) -> RaspaRunResult:
    """Execute RASPA3 in `bundle.work_dir`. Hard-fails if binary unverified."""
    binary = verify_raspa3_binary(
        path=raspa3_binary if raspa3_binary is not None else DEFAULT_RASPA3_PATH
    )
    t0 = time.time()
    started = _dt.datetime.now(_dt.UTC).isoformat()
    stdout_path = bundle.work_dir / "raspa3.stdout.txt"
    stderr_path = bundle.work_dir / "raspa3.stderr.txt"
    proc = subprocess.run(
        [str(binary.path)],
        cwd=bundle.work_dir,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    duration = time.time() - t0
    stdout_path.write_text(proc.stdout or "")
    stderr_path.write_text(proc.stderr or "")
    # RASPA3 v3.0.29 writes its output to output/output_<T>_0.s0.txt
    output_dir = bundle.work_dir / "output"
    output_txt: Path | None = None
    if output_dir.exists():
        candidates = sorted(output_dir.glob("output_*_0.s0.txt"))
        if candidates:
            output_txt = candidates[0]
    # Record evidence manifest
    manifest = {
        "started_utc": started,
        "duration_s": duration,
        "exit_code": proc.returncode,
        "raspa3_version": binary.version,
        "raspa3_sha256": binary.sha256,
        "raspa3_upstream_commit": binary.upstream_commit,
        "input_sha256": bundle.sha256,
        "stdout_path": str(stdout_path.relative_to(bundle.work_dir)),
        "stderr_path": str(stderr_path.relative_to(bundle.work_dir)),
        "output_txt_path": str(output_txt.relative_to(bundle.work_dir)) if output_txt else None,
    }
    (bundle.work_dir / "raspa3_run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return RaspaRunResult(
        work_dir=bundle.work_dir,
        exit_code=proc.returncode,
        duration_s=duration,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output_txt_path=output_txt,
        raspa3_version=binary.version,
        raspa3_sha256=binary.sha256,
    )
