"""RASPA2 subprocess runner.

Invokes RASPA2 in the working directory with RASPA_DIR set so the
molecule definitions are reachable. Captures stdout/stderr and locates
the RASPA2 output file under Output/System_0/.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .input_writer import Raspa2InputBundle
from .raspa2_binary import DEFAULT_RASPA2_BIN, RaspaB2inary, verify_raspa2_binary


@dataclass(frozen=True)
class Raspa2RunResult:
    work_dir: Path
    exit_code: int
    duration_s: float
    stdout_path: Path
    stderr_path: Path
    output_data_path: Path | None
    raspa2_version: str
    raspa2_bin_sha256: str
    raspa2_lib_sha256: str


def run_raspa2(
    bundle: Raspa2InputBundle,
    raspa2_share_dir: Path,
    raspa2_bin: Path = DEFAULT_RASPA2_BIN,
    timeout_s: int = 1800,
    extra_env: dict[str, str] | None = None,
) -> Raspa2RunResult:
    """Run RASPA2 in `bundle.work_dir`. Hard-fails if binary fails verification."""
    binary: RaspaB2inary = verify_raspa2_binary(
        bin_path=raspa2_bin,
        share_dir=raspa2_share_dir,
    )
    started = _dt.datetime.now(_dt.UTC).isoformat()
    t0 = time.time()
    stdout_path = bundle.work_dir / "raspa2.stdout.txt"
    stderr_path = bundle.work_dir / "raspa2.stderr.txt"

    env = dict(os.environ)
    # RASPA2 reads $RASPA_DIR/share/raspa/{forcefield,structures,molecules}/.
    # Point it at the working dir so the per-branch relabeled framework + FF + CO2 are picked up.
    env["RASPA_DIR"] = str(bundle.work_dir)
    env["DYLD_LIBRARY_PATH"] = str(binary.lib_path.parent)
    env["LD_LIBRARY_PATH"] = str(binary.lib_path.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        [str(binary.bin_path)],
        cwd=str(bundle.work_dir),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    duration = time.time() - t0
    stdout_path.write_text(proc.stdout or "")
    stderr_path.write_text(proc.stderr or "")

    # RASPA2 writes Output/System_0/output_*.data
    output_dir = bundle.work_dir / "Output" / "System_0"
    output_data: Path | None = None
    if output_dir.exists():
        candidates = sorted(output_dir.glob("output_*.data"))
        if candidates:
            output_data = candidates[0]

    manifest = {
        "started_utc": started,
        "duration_s": duration,
        "exit_code": proc.returncode,
        "raspa2_version": binary.version,
        "raspa2_bin_sha256": binary.bin_sha256,
        "raspa2_lib_sha256": binary.lib_sha256,
        "input_sha256": bundle.sha256,
        "stdout_path": str(stdout_path.relative_to(bundle.work_dir)),
        "stderr_path": str(stderr_path.relative_to(bundle.work_dir)),
        "output_data_path": str(output_data.relative_to(bundle.work_dir)) if output_data else None,
    }
    (bundle.work_dir / "raspa2_run_manifest.json").write_text(json.dumps(manifest, indent=2))

    return Raspa2RunResult(
        work_dir=bundle.work_dir,
        exit_code=proc.returncode,
        duration_s=duration,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output_data_path=output_data,
        raspa2_version=binary.version,
        raspa2_bin_sha256=binary.bin_sha256,
        raspa2_lib_sha256=binary.lib_sha256,
    )
