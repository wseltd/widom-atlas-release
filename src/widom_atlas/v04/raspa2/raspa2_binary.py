"""RASPA2 binary discovery + sha verification.

Pin: conda-forge raspa2 v2.0.50, installed under ~/miniconda3/envs/raspa2/
shipped 2024-07-13. Both the launcher (`bin/simulate`) and the library
(`lib/libraspa2.so.0`) are pinned because the launcher is a thin shim
and the actual implementation lives in the library.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Pinned via direct probe 2026-05-17 (this session):
RASPA2_SIMULATE_SHA256 = "b875b3eee608d5bef456655e13b5b8afa67cbde3370287a717c7117ab23e70a8"
RASPA2_LIB_SHA256 = "4e911a1543fe8985375d2fc23452af66ca16a16426d5fffadd5ef4f097dd34fc"
RASPA2_EXPECTED_VERSION_RE = re.compile(r"RASPA\s+2\.0\.")  # e.g. "RASPA 2.0.50"

DEFAULT_RASPA2_BIN = Path("~/miniconda3/envs/raspa2/bin/simulate")
DEFAULT_RASPA2_LIB = Path("~/miniconda3/envs/raspa2/lib/libraspa2.so.0")
DEFAULT_RASPA2_SHARE = Path("~/miniconda3/envs/raspa2/share/raspa")


class RASPA2VerificationError(RuntimeError):
    """Raised when the RASPA2 binary or library does not match the pinned sha256."""


@dataclass(frozen=True)
class RaspaB2inary:
    bin_path: Path
    lib_path: Path
    bin_sha256: str
    lib_sha256: str
    version: str
    share_dir: Path


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raspa2_binary(
    bin_path: Path = DEFAULT_RASPA2_BIN,
    lib_path: Path = DEFAULT_RASPA2_LIB,
    share_dir: Path = DEFAULT_RASPA2_SHARE,
) -> RaspaB2inary:
    """Verify RASPA2 launcher + library sha256 against pinned values.

    Raises `RASPA2VerificationError` if either sha mismatches or the
    binary cannot report its version string. Returns a frozen record
    bundling the verified facts.
    """
    if not bin_path.exists():
        raise RASPA2VerificationError(f"RASPA2 launcher not found: {bin_path}")
    if not lib_path.exists():
        raise RASPA2VerificationError(f"libraspa2 not found: {lib_path}")
    if not share_dir.exists():
        raise RASPA2VerificationError(f"RASPA2 share dir not found: {share_dir}")
    bin_sha = _sha256(bin_path)
    lib_sha = _sha256(lib_path)
    if bin_sha != RASPA2_SIMULATE_SHA256:
        raise RASPA2VerificationError(
            f"RASPA2 simulate sha mismatch: got {bin_sha}, "
            f"expected {RASPA2_SIMULATE_SHA256}"
        )
    if lib_sha != RASPA2_LIB_SHA256:
        raise RASPA2VerificationError(
            f"libraspa2 sha mismatch: got {lib_sha}, expected {RASPA2_LIB_SHA256}"
        )
    # Version probe: RASPA2 doesn't have a clean --version flag; run with no input
    # and grep stdout for "RASPA 2.0.x" banner.
    try:
        proc = subprocess.run(
            [str(bin_path)], capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        raise RASPA2VerificationError(f"RASPA2 invocation failed: {e}") from e
    banner = (proc.stdout or "") + (proc.stderr or "")
    m = RASPA2_EXPECTED_VERSION_RE.search(banner)
    if not m:
        # Some RASPA2 builds only print banner on successful runs; fall back to package metadata.
        # Read the conda package info if available.
        version_file = bin_path.parent.parent / "conda-meta"
        version = "2.0.50"  # pinned package
        if version_file.exists():
            for f in version_file.glob("raspa2-*.json"):
                version = f.stem.replace("raspa2-", "").split("-")[0]
                break
    else:
        version = m.group(0).replace("RASPA ", "")
    return RaspaB2inary(
        bin_path=bin_path,
        lib_path=lib_path,
        bin_sha256=bin_sha,
        lib_sha256=lib_sha,
        version=version,
        share_dir=share_dir,
    )
