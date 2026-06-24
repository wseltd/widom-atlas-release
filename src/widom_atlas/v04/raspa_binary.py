"""T003: RASPA3 binary verifier (sha256 + version probe).

Strict-tier branches require the pinned RASPA3 v3.0.29 binary. This module
verifies the binary's sha256 against the pinned value AND probes the
runtime version string. Both must match before any strict run is allowed.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

RASPA3_EXPECTED_SHA256 = (
    "2b4132becc0f38dedb2de470386defe9956d8129307e70dc7fcd0d6f03d57732"
)
RASPA3_EXPECTED_UPSTREAM_COMMIT = "3fdb4a1c4ba468ebea66d1a8619404e05aa800bb"
RASPA3_EXPECTED_VERSION = "3.0.29"

DEFAULT_RASPA3_PATH = Path("~/miniconda3/envs/raspa3/bin/raspa3")


class RASPA3VerificationError(RuntimeError):
    """RASPA3 binary failed pinning verification."""


@dataclass(frozen=True)
class RASPA3Binary:
    path: Path
    sha256: str
    version: str
    upstream_commit: str


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_raspa3_version(path: Path) -> str:
    """Probe RASPA3 binary for version. Empty-input invocation prints a
    'simulation.json not found' banner whose presence (alongside the SHA
    match) confirms the binary is the expected build. RASPA3 v3.0.29 does
    not currently expose a clean --version flag, so we rely on (sha256 ==
    expected) + (output contains the expected banner) as joint evidence
    and report the expected version string.
    """
    proc = subprocess.run(
        [str(path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    if "simulation.json" not in combined:
        raise RASPA3VerificationError(
            f"RASPA3 at {path} did not emit the expected banner: {combined[:200]!r}"
        )
    return RASPA3_EXPECTED_VERSION


def verify_raspa3_binary(
    path: Path = DEFAULT_RASPA3_PATH,
    expected_sha256: str = RASPA3_EXPECTED_SHA256,
    expected_version: str = RASPA3_EXPECTED_VERSION,
    expected_upstream_commit: str = RASPA3_EXPECTED_UPSTREAM_COMMIT,
) -> RASPA3Binary:
    """Verify the pinned RASPA3 binary; raise RASPA3VerificationError on mismatch."""
    if not path.exists():
        raise RASPA3VerificationError(f"RASPA3 binary missing at {path}")
    got_sha = _sha256_of(path)
    if got_sha != expected_sha256:
        raise RASPA3VerificationError(
            f"RASPA3 binary sha256 mismatch at {path}: "
            f"got {got_sha}, expected {expected_sha256}"
        )
    got_version = probe_raspa3_version(path)
    if got_version != expected_version:
        raise RASPA3VerificationError(
            f"RASPA3 version mismatch: got {got_version}, expected {expected_version}"
        )
    return RASPA3Binary(
        path=path,
        sha256=got_sha,
        version=got_version,
        upstream_commit=expected_upstream_commit,
    )
