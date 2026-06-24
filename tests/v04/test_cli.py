"""Tests for T001 CLI scaffolding."""
from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

from widom_atlas.v04 import cli


def test_verify_spec_runs() -> None:
    rc = cli.main(["verify-spec"])
    assert rc == 0


def test_verify_binary_runs() -> None:
    from widom_atlas.v04 import raspa_binary as rb
    if not rb.DEFAULT_RASPA3_PATH.exists():
        pytest.skip(f"RASPA3 binary not installed at {rb.DEFAULT_RASPA3_PATH}")
    rc = cli.main(["verify-binary"])
    assert rc == 0


def test_list_cases_emits_all_14_branches() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["list-cases"])
    assert rc == 0
    out = buf.getvalue()
    for bid in ("1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b",
                "5a", "5b", "6a", "6b", "6c", "6d"):
        assert f"  {bid:<5}" in out, f"missing branch {bid} in CLI output"


def test_unknown_command_errors_cleanly() -> None:
    err = io.StringIO()
    with redirect_stderr(err), pytest.raises(SystemExit):
        cli.main(["nonexistent-subcommand"])
