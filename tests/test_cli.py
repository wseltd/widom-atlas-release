"""Tests for widom_atlas.cli (T049 / T062)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from typer.testing import CliRunner

from widom_atlas.cli import app
from widom_atlas.io.from_arrays import from_arrays
from widom_atlas.io.npz import save_samples_npz

runner = CliRunner()


def _seed_samples(tmp_path: Path) -> tuple[Path, Path]:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)
    cif = tmp_path / "structure.cif"
    ase_write(str(cif), atoms)
    rng = np.random.default_rng(0)
    n = 60
    target = np.array([0.3, 0.5, 0.5])
    frac = (rng.normal(target, 0.01, (n, 3))) % 1.0
    e = rng.normal(-0.5, 0.01, n)
    ai = from_arrays(
        structure=atoms,
        positions_frac=frac,
        energies_eV=e,
        temperature_K=298.15,
        gas="CO2",
        metadata={"src": "cli-test"},
    )
    samples = tmp_path / "samples.npz"
    save_samples_npz(ai, samples)
    return samples, cif


def test_cli_info_succeeds() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "widom-atlas" in result.output


def test_cli_analyse_samples_runs(tmp_path: Path) -> None:
    samples, cif = _seed_samples(tmp_path)
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "analyse-samples",
            str(samples),
            "--structure",
            str(cif),
            "--gas",
            "CO2",
            "--temperature",
            "298.15",
            "--out",
            str(out),
            "--n-grid",
            "8",
            "--eps-A",
            "0.5",
            "--min-samples",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "manifest.json").exists()


def test_cli_analyse_samples_missing_input_fails_with_message(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "analyse-samples",
            str(tmp_path / "nope.npz"),
            "--structure",
            str(tmp_path / "nope.cif"),
            "--gas",
            "CO2",
            "--temperature",
            "298.15",
            "--out",
            str(tmp_path / "run"),
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_cli_strain_isotropic_writes_cif(tmp_path: Path) -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 10.0, pbc=True)
    cif = tmp_path / "in.cif"
    ase_write(str(cif), atoms)
    out = tmp_path / "out.cif"
    result = runner.invoke(
        app,
        ["strain", str(cif), "--mode", "isotropic", "--value", "0.01", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cli_compare_writes_robustness_report(tmp_path: Path) -> None:
    samples, cif = _seed_samples(tmp_path)
    pri = tmp_path / "pri"
    per = tmp_path / "per"
    runner.invoke(
        app,
        [
            "analyse-samples", str(samples),
            "--structure", str(cif), "--gas", "CO2", "--temperature", "298.15",
            "--out", str(pri), "--n-grid", "8", "--eps-A", "0.5", "--min-samples", "4",
        ],
    )
    runner.invoke(
        app,
        [
            "analyse-samples", str(samples),
            "--structure", str(cif), "--gas", "CO2", "--temperature", "298.15",
            "--out", str(per), "--n-grid", "8", "--eps-A", "0.5", "--min-samples", "4",
        ],
    )
    # Inject perturbation_spec + cell_matrix into perturbed manifest for compare
    manifest_path = per / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["perturbation_spec"] = {"kind": "isotropic", "magnitude": 0.01, "label": "iso1"}
    manifest["cell_matrix"] = (np.eye(3) * 10.0).tolist()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "cmp"
    result = runner.invoke(app, ["compare", str(pri), str(per), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "robustness_report.json").exists()


def test_cli_help_lists_required_flags() -> None:
    for cmd, flags in (
        ("analyse-samples", ["--structure", "--gas", "--temperature", "--out"]),
        ("strain", ["--mode", "--value", "--out"]),
        ("compare", ["--out"]),
        ("benchmark", ["--set", "--download", "--gas", "--out"]),
    ):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0
        for flag in flags:
            assert flag in result.output, f"{cmd}: missing flag {flag}"
