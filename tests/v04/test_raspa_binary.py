"""Tests for T003 raspa_binary verifier."""
from __future__ import annotations

from pathlib import Path

import pytest

from widom_atlas.v04 import raspa_binary as rb


def test_pinned_raspa3_binary_verifies() -> None:
    """The pinned binary at DEFAULT_RASPA3_PATH must verify cleanly."""
    if not rb.DEFAULT_RASPA3_PATH.exists():
        pytest.skip(f"RASPA3 binary not installed at {rb.DEFAULT_RASPA3_PATH}")
    b = rb.verify_raspa3_binary()
    assert b.sha256 == rb.RASPA3_EXPECTED_SHA256
    assert b.version == rb.RASPA3_EXPECTED_VERSION
    assert b.upstream_commit == rb.RASPA3_EXPECTED_UPSTREAM_COMMIT


def test_raspa3_verification_raises_on_missing(tmp_path: Path) -> None:
    nonexistent = tmp_path / "no_such_binary"
    with pytest.raises(rb.RASPA3VerificationError, match="missing"):
        rb.verify_raspa3_binary(path=nonexistent)


def test_raspa3_verification_raises_on_sha_mismatch(tmp_path: Path) -> None:
    fake = tmp_path / "fake_raspa3"
    fake.write_bytes(b"not a real raspa3 binary")
    fake.chmod(0o755)
    with pytest.raises(rb.RASPA3VerificationError, match="sha256 mismatch"):
        rb.verify_raspa3_binary(path=fake)


def test_probe_raspa3_version_succeeds_on_pinned_binary() -> None:
    if not rb.DEFAULT_RASPA3_PATH.exists():
        pytest.skip(f"RASPA3 binary not installed at {rb.DEFAULT_RASPA3_PATH}")
    v = rb.probe_raspa3_version(rb.DEFAULT_RASPA3_PATH)
    assert v == rb.RASPA3_EXPECTED_VERSION
