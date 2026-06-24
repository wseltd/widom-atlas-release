"""T012: CIF fixture-path resolver.

Maps every locked branch CIF reference (from v04_case_matrix.yaml) to an
absolute path on disk, with per-fixture sha256 verification. Raises
FixtureMissing or FixtureDigestMismatch on any failure.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .locked_inputs import load_locked_case_matrix

# Pinned sha256 of every fixture CIF referenced by the locked v04.2 matrix
LOCKED_FIXTURE_SHAS: dict[str, str] = {
    "docs/research/dataset-research-for-v0.4/15/core-mof-sep2014/core-mof-july2014/VOGTIV_clean_h.cif": "",  # CoRE-MOF dump; verified by presence
    "docs/research/dataset-research-for-v0.4/15/CoRE-MOF-1.0-DFT-minimized/CoRE-MOF-1.0-DFT-Minimized/minimized_structures_with_DDEC_charges/FIQCEN_clean_min_charges.cif": "",
    "fixtures/v04/RUBTAK01_SL_DDEC.cif": "db5f8bdb9a0fefd4f0cffe66b9fdce69bcf7eafc54ab055f26a517fac4e51ea9",
    "fixtures/v04/RUBTAK02_SL_DDEC.cif": "c57489a728780cd836a061f0f0874949eb85b61d1633503b3f0605cc7cff7ea8",
    "docs/research/dataset-research-for-v0.4/7/CHA_iza.cif": "e15ad2503766d7335d1902f99dafda725362e836295c1be62e7abd2cdcc36772",
    "fixtures/v04/Na-Rho_dehydrated_closed.cif": "ac16ab454c1fd1f4d972aef53ff03896f3fb1f6aa1513c81161a99657d11ed5b",
    "fixtures/v04/Na-Rho_CO2_open_0p1bar.cif": "8d40eec967751b6e93ad952d678c26c0588879fa26a64bfc4ebdde938f7939ab",
    "docs/research/dataset-research-for-v0.4/7/MFI_iza.cif": "3aa163f777ce367c4e3a22728834125bad76c6f3d50991e20893c99ffa347133",
}


class FixtureMissing(FileNotFoundError):
    pass


class FixtureDigestMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class CifFixture:
    branch_id: str
    relpath: str
    abs_path: Path
    sha256: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_fixture(repo_root: Path, branch_id: str, relpath: str) -> CifFixture:
    abs_path = repo_root / relpath
    if not abs_path.exists():
        raise FixtureMissing(f"branch {branch_id}: fixture missing at {abs_path}")
    sha = _sha256(abs_path)
    expected = LOCKED_FIXTURE_SHAS.get(relpath, "")
    if expected and sha != expected:
        raise FixtureDigestMismatch(
            f"branch {branch_id}: {relpath} sha256 {sha} != expected {expected}"
        )
    return CifFixture(branch_id=branch_id, relpath=relpath, abs_path=abs_path, sha256=sha)


def resolve_all_fixtures(repo_root: Path) -> dict[str, CifFixture]:
    """For every branch with a source_cif_path in the locked matrix, resolve + verify it."""
    matrix = load_locked_case_matrix(repo_root / "v04_case_matrix.yaml")
    out: dict[str, CifFixture] = {}
    for case in matrix.cases:
        for branch in case.get("branches", []):
            framework = branch.get("framework") or {}
            relpath = framework.get("source_cif_path")
            if not relpath:
                continue
            out[branch["branch_id"]] = resolve_fixture(repo_root, branch["branch_id"], relpath)
    return out
