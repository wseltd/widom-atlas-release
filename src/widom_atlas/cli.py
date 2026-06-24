"""Typer CLI for widom-atlas: analyse-samples, strain, compare, benchmark, info."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console

from widom_atlas import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True, help="widom-atlas — Widom insertion atlas + robustness analysis")
_console = Console(stderr=True)


def _fail(message: str, code: int = 2) -> None:
    _console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=code)


@app.command("analyse-samples")
def analyse_samples(
    samples_path: Path = typer.Argument(..., help="Path to insertion-samples .npz file"),
    structure: Path = typer.Option(..., "--structure", help="Path to host-structure CIF"),
    gas: str = typer.Option(..., "--gas", help="Adsorbate gas (CO2, N2, or CH4)"),
    temperature: float = typer.Option(..., "--temperature", help="Temperature in Kelvin"),
    out: Path = typer.Option(..., "--out", help="Output run directory"),
    n_grid: int = typer.Option(48, "--n-grid", help="Density grid edge length"),
    eps_A: float = typer.Option(1.5, "--eps-A", help="DBSCAN eps in Angstrom"),
    min_samples: int = typer.Option(8, "--min-samples", help="DBSCAN minimum samples"),
) -> None:
    """Run the atlas pipeline: density → basins → symmetry → reports."""
    if not samples_path.exists():
        _fail(f"samples not found: {samples_path}")
    if not structure.exists():
        _fail(f"structure not found: {structure}")
    try:
        from ase.io import read

        atoms = read(str(structure))
        if hasattr(atoms, "set_pbc"):
            atoms.set_pbc(True)
    except Exception as exc:
        _fail(f"could not read structure {structure}: {exc}")

    try:
        from widom_atlas.core.pipeline import PipelineParams, run_atlas
        from widom_atlas.io.npz import from_npz

        atlas_input = from_npz(samples_path, structure=atoms)
        if atlas_input.gas != gas:
            _fail(f"--gas={gas} does not match samples gas={atlas_input.gas}")
        if abs(atlas_input.temperature_K - temperature) > 1e-6:
            _fail(f"--temperature={temperature} does not match samples T={atlas_input.temperature_K}")
        params = PipelineParams(n_grid=(n_grid, n_grid, n_grid), dbscan_eps_A=eps_A, min_samples=min_samples)
        run_atlas(atlas_input, params, out, structure=atoms)
        _console.print(f"[green]ok[/green] wrote run to {out}")
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"pipeline failed: {exc}")


@app.command("strain")
def strain(
    structure: Path = typer.Argument(..., help="Path to input CIF"),
    mode: str = typer.Option(..., "--mode", help="isotropic | uniaxial | affine | volume_preserving"),
    value: str = typer.Option(..., "--value", help="Strain magnitude (float) or 3x3 matrix path"),
    axis: str | None = typer.Option(None, "--axis", help="Axis (a/b/c) for uniaxial mode"),
    out: Path = typer.Option(..., "--out", help="Output CIF path"),
) -> None:
    """Apply strain to a structure and write the result as a CIF."""
    if not structure.exists():
        _fail(f"structure not found: {structure}")
    try:
        from ase.io import read, write

        atoms = read(str(structure))
        if hasattr(atoms, "set_pbc"):
            atoms.set_pbc(True)
        from widom_atlas.perturb.strain import apply_strain

        if mode == "affine":
            matrix = np.loadtxt(value)
            new_atoms = apply_strain(atoms, mode="affine", value=matrix)
        elif mode in {"isotropic", "uniaxial", "volume_preserving"}:
            new_atoms = apply_strain(atoms, mode=mode, value=float(value), axis=axis)  # type: ignore[arg-type]
        else:
            _fail(f"unknown strain mode: {mode!r}")
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        write(str(out), new_atoms)
        _console.print(f"[green]ok[/green] wrote strained structure to {out}")
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"strain failed: {exc}")


@app.command("compare")
def compare(
    pristine_run: Path = typer.Argument(..., help="Pristine run directory"),
    perturbed_run: Path = typer.Argument(..., help="Perturbed run directory"),
    out: Path = typer.Option(..., "--out", help="Comparison output directory"),
    match_tol_A: float = typer.Option(0.35, "--match-tol-A"),
) -> None:
    """Build a robustness report from two run directories."""
    if not pristine_run.exists():
        _fail(f"pristine run not found: {pristine_run}")
    if not perturbed_run.exists():
        _fail(f"perturbed run not found: {perturbed_run}")
    try:
        import json

        from widom_atlas.robustness.compare import build_robustness_report

        report = build_robustness_report(pristine_run, perturbed_run, match_tol_A=match_tol_A)
        out.mkdir(parents=True, exist_ok=True)
        (out / "robustness_report.json").write_text(
            json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        _console.print(f"[green]ok[/green] wrote robustness report to {out}/robustness_report.json")
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"compare failed: {exc}")


@app.command("benchmark")
def benchmark(
    set_name: str = typer.Option("small", "--set", help="Benchmark set name"),
    download: bool = typer.Option(False, "--download", help="Allow network downloads"),
    gas: str = typer.Option("CO2", "--gas", help="Adsorbate gas"),
    out: Path = typer.Option(..., "--out", help="Output directory"),
    structures_dir: Path = typer.Option(Path("tests/fixtures/real_structures"), "--structures-dir"),
    cache_dir: Path = typer.Option(Path("benchmarks/cache"), "--cache-dir"),
    temperature: float = typer.Option(298.15, "--temperature"),
    n_samples: int = typer.Option(500, "--n-samples"),
    backend: str = typer.Option(
        "parameterised_lj",
        "--backend",
        help="Backend: parameterised_lj (TraPPE+UFF) | user_parameterised_coulomb_lj | toy_lj | external_samples",
    ),
    external_samples: Path | None = typer.Option(
        None,
        "--external-samples",
        help="NPZ of pre-computed Widom samples (required when --backend external_samples)",
    ),
    external_manifest: Path | None = typer.Option(
        None,
        "--external-manifest",
        help="JSON sidecar (ExternalSampleManifest); auto-detected as <samples>.manifest.json when omitted",
    ),
    params: Path | None = typer.Option(
        None,
        "--params",
        help="JSON parameter file for user_parameterised_coulomb_lj (see widom_atlas.backends.user_parameterised.UserParameterFile)",
    ),
    allow_neutral_fallback: bool = typer.Option(
        False,
        "--allow-neutral-fallback",
        help="Allow user_parameterised_coulomb_lj to run without charges (NOT recommended; warning will be stamped into the manifest)",
    ),
) -> None:
    """Run the public-dataset benchmark suite (Layer 3)."""
    try:
        from widom_atlas.benchmarks.launch_report import write_launch_report
        from widom_atlas.benchmarks.runner import run_benchmark_set
        from widom_atlas.benchmarks.scalar_compare import compare_scalars

        out.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        summary = run_benchmark_set(
            set_name=set_name,
            gas=gas,
            structures_dir=structures_dir,
            samples_dir=None,
            output_dir=out,
            cache_dir=cache_dir,
            download=download,
            temperature_K=temperature,
            n_samples=n_samples,
            backend_name=backend,  # type: ignore[arg-type]
            external_samples_path=external_samples,
            external_manifest_path=external_manifest,
            user_parameter_file=params,
            allow_neutral_fallback=allow_neutral_fallback,
        )
        compare_scalars(summary.benchmark_run_path, cache_dir, out / "scalar_comparison")
        write_launch_report(summary.benchmark_run_path, out / "scalar_comparison" / "scalar_comparison.json", out)
        _console.print(f"[green]ok[/green] wrote benchmark + launch report to {out} (backend={backend})")
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"benchmark failed: {exc}")


@app.command("convergence")
def convergence(
    material: str = typer.Option(..., "--material", help="Benchmark material_id (e.g. UiO-66)"),
    gas: str = typer.Option("CO2", "--gas", help="Adsorbate gas (CO2, N2, or CH4)"),
    insertions: str = typer.Option(
        "100,1000,10000",
        "--insertions",
        help="Comma-separated insertion counts to sweep over",
    ),
    out: Path = typer.Option(..., "--out", help="Output directory for the convergence sweep"),
    cache_dir: Path = typer.Option(Path("benchmarks/cache"), "--cache-dir"),
    structures_dir: Path = typer.Option(
        Path("tests/fixtures/real_structures"), "--structures-dir"
    ),
    temperature: float = typer.Option(298.15, "--temperature"),
    seed: int = typer.Option(0, "--seed"),
    rel_kh: float = typer.Option(0.30, "--rel-kh-threshold"),
    drift_a: float = typer.Option(0.5, "--centroid-drift-threshold-A"),
    weight_change: float = typer.Option(0.05, "--dominant-weight-change-threshold"),
    backend: str = typer.Option(
        "parameterised_lj",
        "--backend",
        help="Backend: parameterised_lj | user_parameterised_coulomb_lj | toy_lj | external_samples",
    ),
    external_samples: Path | None = typer.Option(
        None,
        "--external-samples",
        help="NPZ of pre-computed Widom samples (required when --backend external_samples)",
    ),
    external_manifest: Path | None = typer.Option(
        None,
        "--external-manifest",
        help="JSON sidecar (ExternalSampleManifest); auto-detected when omitted",
    ),
    params: Path | None = typer.Option(
        None,
        "--params",
        help="JSON parameter file for user_parameterised_coulomb_lj",
    ),
    allow_neutral_fallback: bool = typer.Option(
        False,
        "--allow-neutral-fallback",
        help="Allow user_parameterised_coulomb_lj to run without charges (NOT recommended)",
    ),
) -> None:
    """Run a Widom convergence sweep on one material + gas across multiple insertion counts."""
    try:
        ns = [int(x.strip()) for x in insertions.split(",") if x.strip()]
    except ValueError as exc:
        _fail(f"--insertions parse error: {exc}")
        return
    if not ns:
        _fail("--insertions must list at least one positive integer")
        return
    try:
        from widom_atlas.convergence import run_convergence_study

        out.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        summary = run_convergence_study(
            material_id=material,
            gas=gas,
            insertion_counts=ns,
            output_dir=out,
            cache_dir=cache_dir,
            structures_dir=structures_dir,
            temperature_K=temperature,
            seed=seed,
            rel_kh_uncertainty_threshold=rel_kh,
            centroid_drift_threshold_A=drift_a,
            dominant_weight_change_threshold=weight_change,
            backend_name=backend,  # type: ignore[arg-type]
            external_samples_path=external_samples,
            external_manifest_path=external_manifest,
            user_parameter_file=params,
            allow_neutral_fallback=allow_neutral_fallback,
        )
        verdict = summary["verdict"]["overall"]
        _console.print(
            f"[green]ok[/green] convergence sweep written to {out}; verdict={verdict}; backend={backend}"
        )
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"convergence failed: {exc}")


@app.command("info")
def info() -> None:
    """Print package info."""
    typer.echo(f"widom-atlas {__version__}")
    typer.echo("Pure Python companion to CuspAI Widom — adsorption basins, symmetry, robustness.")
    typer.echo("Source of truth: implementation-verdict.txt")


# ----------------------------------------------------------------------
# external-samples sub-app — schema validation + RASPA3 → canonical NPZ
# ----------------------------------------------------------------------

external_samples_app = typer.Typer(
    no_args_is_help=True,
    help="External-samples manifest validation + RASPA3 → canonical NPZ adapter (v0.3).",
)
app.add_typer(external_samples_app, name="external-samples")


@external_samples_app.command("validate")
def external_samples_validate(
    manifest: Path = typer.Argument(..., help="Path to <samples>.manifest.json"),
) -> None:
    """Parse a v0.3 ExternalSampleManifest JSON and report its provenance."""
    if not manifest.exists():
        _fail(f"manifest not found: {manifest}")
    try:
        import json as _json

        from widom_atlas.backends.schema import ExternalSampleManifest

        m = ExternalSampleManifest.model_validate(_json.loads(manifest.read_text(encoding="utf-8")))
        typer.echo(f"sample_format_version : {m.sample_format_version}")
        typer.echo(f"framework             : {m.framework}")
        typer.echo(f"gas                   : {m.gas}")
        typer.echo(f"temperature_K         : {m.temperature_K}")
        typer.echo(f"backend               : {m.backend} (version {m.backend_version})")
        typer.echo(f"n_insertions          : {m.n_insertions}")
        typer.echo(f"random_seed           : {m.random_seed}")
        typer.echo(f"energy_unit           : {m.energy_unit}")
        typer.echo(f"parameter_mode        : {m.parameter_mode}")
        typer.echo(
            f"force_field           : LJ={m.force_field.framework_lj} | charges={m.force_field.framework_charges} | "
            f"gas={m.force_field.gas_model} | mixing={m.force_field.mixing_rules} | electrostatics={m.force_field.electrostatics}"
        )
        typer.echo(f"redistribution_status : {m.redistribution_status}")
        typer.echo(f"citations             : {len(m.citations)}")
        for c in m.citations:
            typer.echo(f"  - role={c.role:14s}  doi={c.doi}  source={c.source[:80]}")
        typer.echo(f"warnings              : {len(m.warnings)}")
        for w in m.warnings:
            typer.echo(f"  - {w}")
        typer.echo(f"suitable_for_quantitative_interpretation : {m.suitable_for_quantitative_interpretation}")
        _console.print("[green]ok[/green] manifest valid")
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"manifest invalid: {exc}")


@external_samples_app.command("convert-raspa3")
def external_samples_convert_raspa3(
    raspa_input: Path = typer.Option(..., "--input", help="RASPA3 run directory containing simulation.input + Output/"),
    structure: Path = typer.Option(..., "--structure", help="Framework CIF (used to set the canonical cell matrix)"),
    gas: str = typer.Option(..., "--gas", help="Gas (CO2 / N2 / CH4) — must match RASPA's input"),
    temperature: float = typer.Option(..., "--temperature", help="Temperature in Kelvin"),
    out: Path = typer.Option(..., "--out", help="Output sidecar path (e.g. samples.scalar.json or samples.npz)"),
    framework_charge_source: str = typer.Option(
        "user_supplied",
        "--framework-charge-source",
        help="Origin of the framework charges used in the RASPA run (DDEC | Qeq | EQeq | PACMOF | user_supplied | none | unknown)",
    ),
    gas_model: str = typer.Option("TraPPE-CO2", "--gas-model", help="Gas model label as configured in RASPA"),
    citation_doi: str | None = typer.Option(None, "--citation-doi", help="DOI of the FF source paper"),
    citation_source: str | None = typer.Option(None, "--citation-source", help="Plain-text citation"),
) -> None:
    """Parse a RASPA3 output dir and emit either a scalar-only sidecar or a canonical NPZ pair.

    By default this is **scalar-only**: writes a small JSON sidecar with KH /
    Q_ads + provenance that the scalar comparator consumes. Per-insertion
    NPZ output requires WriteMoviesEvery 1 in the RASPA input — when no
    Movies/ subdir is found the command falls back to scalar-only and
    documents that in the sidecar's `note` field.
    """
    if not raspa_input.exists():
        _fail(f"RASPA3 directory not found: {raspa_input}")
    try:
        from widom_atlas.backends.raspa3_ingest import (
            parse_raspa3_scalars,
            write_scalar_only_sidecar,
        )

        scalar = parse_raspa3_scalars(raspa_input)
        if scalar.gas != gas and gas != "auto":
            _console.print(
                f"[yellow]warn[/yellow] declared gas={gas!r} != RASPA gas={scalar.gas!r}; "
                "the sidecar records the RASPA value"
            )
        if abs(scalar.temperature_K - float(temperature)) > 1e-3 and scalar.temperature_K > 0:
            _console.print(
                f"[yellow]warn[/yellow] declared T={temperature}K != RASPA T={scalar.temperature_K}K; "
                "the sidecar records the RASPA value"
            )
        citations: list[dict[str, str]] = []
        if citation_doi and citation_source:
            citations.append({"role": "framework_charges", "doi": citation_doi, "source": citation_source})
        sidecar = write_scalar_only_sidecar(
            out_path=out,
            scalar_result=scalar,
            force_field_label="UFF" if "UFF" in scalar.output_files_sha256 else "see-RASPA-input",
            framework_charge_source=framework_charge_source,
            gas_model=gas_model,
            citations=citations,
            warnings=[],
        )
        _console.print(f"[green]ok[/green] wrote scalar-only sidecar to {sidecar}")
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"convert-raspa3 failed: {exc}")


@app.command("compare-backends")
def compare_backends(
    sources: list[Path] = typer.Argument(
        ...,
        help="One or more run directories or scalar-only sidecar JSONs to compare",
    ),
    out: Path = typer.Option(..., "--out", help="Output directory for backend_comparison.{json,md}"),
) -> None:
    """Compare KH / Q_ads / basins across multiple backends on the same material(s)."""
    try:
        from widom_atlas.backends.comparison import write_comparison_report

        if not sources:
            _fail("compare-backends requires at least one source")
        out.mkdir(parents=True, exist_ok=True)
        json_path, md_path = write_comparison_report(sources, out)
        _console.print(
            f"[green]ok[/green] wrote backend comparison report to {json_path} and {md_path}"
        )
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"compare-backends failed: {exc}")


# ----------------------------------------------------------------------
# data sub-app — registry list / show / validate / status
# ----------------------------------------------------------------------

data_app = typer.Typer(
    no_args_is_help=True,
    help="v0.4 dataset registry: list, show, validate, status (no auto-download).",
)
app.add_typer(data_app, name="data")


@data_app.command("list")
def data_list() -> None:
    """List every registered dataset, scalar reference, and site reference."""
    from widom_atlas.data_registry import (
        list_datasets,
        list_scalar_references,
        list_site_references,
        load_validation_thresholds,
    )

    typer.echo("=== Datasets ===")
    typer.echo(f"{'name':25s}  {'kind':22s}  {'license':18s}  {'redistribution':32s}  cache_path")
    for d in list_datasets():
        typer.echo(
            f"{d.name:25s}  {d.kind:22s}  {d.license:18s}  {d.redistribution_status:32s}  {d.cache_path or '—'}"
        )

    typer.echo("\n=== Scalar references ===")
    typer.echo(f"{'material':18s} {'gas':5s} {'T_K':>7s}  {'KH':>10s}  {'Qads':>8s}  doi")
    for s in list_scalar_references():
        kh = "—" if s.KH_value is None else format(s.KH_value, ".2e")
        q = "—" if s.Qads_value is None else format(s.Qads_value, ".1f")
        typer.echo(
            f"{s.material_id:18s} {s.gas:5s} {s.temperature_K:7.2f}  {kh:>10s}  {q:>8s}  {s.provenance.citation.doi}"
        )

    typer.echo("\n=== Site references ===")
    typer.echo(f"{'material':14s} {'gas':5s} {'label':32s} {'kind':18s} {'doi':30s}")
    for sr in list_site_references():
        typer.echo(
            f"{sr.material_id:14s} {sr.gas:5s} {sr.label:32s} {sr.site_kind:18s} {sr.provenance.citation.doi:30s}"
        )

    typer.echo("\n=== Validation threshold sets ===")
    vt = load_validation_thresholds()
    for name, ts in vt.sets.items():
        typer.echo(
            f"  {name:20s} KH±{ts.KH_relative_error_upper * 100:.0f}%  "
            f"Qads±{ts.Qads_abs_error_kJmol_upper:.1f}kJ/mol  "
            f"basin<{ts.basin_centroid_max_distance_A:.1f}Å  "
            f"N≥{ts.convergence_min_insertions_KH}"
        )


@data_app.command("show")
def data_show(
    name: str = typer.Argument(..., help="Dataset name (e.g. CRAFTED) or 'thresholds:<set>'"),
) -> None:
    """Print one registry entry as YAML-style key/value text."""
    import json as _json

    from widom_atlas.data_registry import load_dataset
    from widom_atlas.data_registry.registry import load_threshold_set

    try:
        if name.startswith("thresholds:"):
            ts = load_threshold_set(name.split(":", 1)[1])
            typer.echo(_json.dumps(ts.model_dump(), sort_keys=True, indent=2))
            return
        d = load_dataset(name)
        typer.echo(_json.dumps(d.model_dump(), sort_keys=True, indent=2))
    except KeyError as exc:
        _fail(str(exc))


@data_app.command("validate")
def data_validate(
    path: Path = typer.Argument(..., help="Path to a registry-shaped YAML to validate"),
    kind: str = typer.Option(
        ...,
        "--kind",
        help="One of {dataset, scalar, site, thresholds}",
    ),
) -> None:
    """Validate a registry-shaped YAML file against the v0.4 schema."""
    import yaml as _yaml

    if not path.exists():
        _fail(f"file not found: {path}")
    raw = _yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        if kind == "dataset":
            from widom_atlas.data_registry.schema import DatasetRegistryEntry

            entries = raw.get("datasets") if isinstance(raw, dict) else raw
            if isinstance(entries, list):
                for e in entries:
                    DatasetRegistryEntry.model_validate(e)
                typer.echo(f"[green]ok[/green] {len(entries)} dataset entries valid")
            else:
                DatasetRegistryEntry.model_validate(raw)
                typer.echo("[green]ok[/green] single dataset entry valid")
        elif kind == "scalar":
            from widom_atlas.data_registry.schema import ScalarReferenceEntry

            entries = raw.get("scalars") if isinstance(raw, dict) else raw
            if isinstance(entries, list):
                for e in entries:
                    ScalarReferenceEntry.model_validate(e)
                typer.echo(f"[green]ok[/green] {len(entries)} scalar entries valid")
            else:
                ScalarReferenceEntry.model_validate(raw)
                typer.echo("[green]ok[/green] single scalar entry valid")
        elif kind == "site":
            from widom_atlas.data_registry.schema import SiteReferenceEntry

            entries = raw.get("sites") if isinstance(raw, dict) else raw
            if isinstance(entries, list):
                for e in entries:
                    SiteReferenceEntry.model_validate(e)
                typer.echo(f"[green]ok[/green] {len(entries)} site entries valid")
            else:
                SiteReferenceEntry.model_validate(raw)
                typer.echo("[green]ok[/green] single site entry valid")
        elif kind == "thresholds":
            from widom_atlas.data_registry.schema import ValidationThresholds

            ValidationThresholds.model_validate(raw)
            typer.echo("[green]ok[/green] threshold table valid")
        else:
            _fail(f"unknown --kind {kind!r}; choose dataset|scalar|site|thresholds")
    except Exception as exc:
        _fail(f"invalid: {exc}")


@data_app.command("status")
def data_status(
    repo_root: Path = typer.Option(Path("."), "--repo-root", help="Where to look for cache_path-relative paths"),
) -> None:
    """Report which datasets are present locally and verified."""
    from widom_atlas.data_registry import list_datasets
    from widom_atlas.data_registry.registry import dataset_status

    typer.echo(f"{'name':25s}  {'present':>8s}  {'verified':>9s}  {'cache_path':40s}  note")
    for d in list_datasets():
        st = dataset_status(d, repo_root=repo_root)
        present = "yes" if st["present"] else "no"
        verified = "yes" if st["verified"] else "no"
        cp = (st["cache_path"] or "—")
        if len(cp) > 40:
            cp = "…" + cp[-39:]
        note = st.get("note") or ""
        typer.echo(f"{d.name:25s}  {present:>8s}  {verified:>9s}  {cp:40s}  {note}")


ingest_app = typer.Typer(no_args_is_help=True, help="External-source ingesters (v0.4)")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("raspa3-ff")
def ingest_raspa3_ff(
    input_dir: Path = typer.Argument(..., help="Directory with simulation.json + force_field.json + <gas>.json"),
    gas: str = typer.Option(..., "--gas", help="Gas component label (matches <gas>.json)"),
    out: Path = typer.Option(..., "--out", help="Output UserParameterFile JSON path"),
) -> None:
    """Parse a RASPA3 input directory and emit a UserParameterFile JSON."""
    from .ingest.raspa3_ff import parse_raspa3_input_directory, to_user_parameter_file

    component_paths: dict[str, Path] = {}
    gas_json = input_dir / f"{gas}.json"
    if gas_json.exists():
        component_paths[gas] = gas_json
    parsed = parse_raspa3_input_directory(
        force_field_path=input_dir / "force_field.json",
        simulation_path=input_dir / "simulation.json",
        component_paths=component_paths or None,
    )
    upf = to_user_parameter_file(parsed, gas_name=gas)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(upf.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"wrote {out}")


@ingest_app.command("mofxdb")
def ingest_mofxdb(
    record_id: int = typer.Argument(..., help="MOFX-DB integer record ID"),
    out_dir: Path = typer.Option(Path("benchmarks/cache/mofxdb"), "--out-dir"),
) -> None:
    """Live-fetch a MOFX-DB record and cache the JSON + simin records."""
    from .ingest.mofxdb import (
        fetch_mofxdb_record,
        parse_mofxdb_record,
        write_record_to_cache,
    )

    payload = fetch_mofxdb_record(record_id)
    cached = write_record_to_cache(payload, out_dir)
    distilled = parse_mofxdb_record(payload)
    typer.echo(f"cached raw: {cached}")
    typer.echo(f"simin records distilled: {len(distilled.get('simin_records') or [])}")


@ingest_app.command("nist-isodb")
def ingest_nist_isodb(
    filename: str = typer.Argument(..., help="NIST ISODB isotherm filename, e.g. '10.1021bk-2010-1056.ch013.isotherm0'"),
    out_dir: Path = typer.Option(Path("benchmarks/cache/nist_isodb"), "--out-dir"),
) -> None:
    """Live-fetch one NIST ISODB isotherm record."""
    from .ingest.nist_isodb import fetch_nist_isotherm, parse_nist_isotherm, write_record_to_cache

    payload = fetch_nist_isotherm(filename)
    cached = write_record_to_cache(payload, out_dir)
    scalar = parse_nist_isotherm(payload)
    typer.echo(f"cached raw: {cached}")
    typer.echo(
        f"scalar: gas={scalar.gas} T={scalar.temperature_K} K K_H={scalar.KH_estimator_mol_per_kg_per_Pa}"
    )


@ingest_app.command("crafted")
def ingest_crafted(
    archive: Path = typer.Argument(..., help="Path to CRAFTED-2.0.1.tar.xz"),
    dest: Path = typer.Option(Path("benchmarks/cache/crafted"), "--dest"),
) -> None:
    """Unpack and summarise a CRAFTED v2 archive."""
    from .ingest.crafted import (
        summarise_unpacked_crafted,
        unpack_crafted_archive,
        write_summary_to_cache,
    )

    extracted = unpack_crafted_archive(archive, dest)
    summary = summarise_unpacked_crafted(extracted)
    out = dest / "summary.json"
    write_summary_to_cache(summary, out)
    typer.echo(f"wrote {out}: {summary.n_csv_files} csv, {summary.materials_seen} materials, gases={summary.gases_seen}")


@ingest_app.command("core-mof")
def ingest_core_mof(
    archive: Path = typer.Argument(..., help="Path to CoRE-MOF zip (2019 v2 or 2024 structures)"),
    dest: Path = typer.Option(Path("benchmarks/cache/core_mof"), "--dest"),
) -> None:
    """Unpack a CoRE-MOF zip and emit a summary."""
    from .ingest.core_mof import unpack_core_mof_zip, write_summary

    result = unpack_core_mof_zip(archive, dest)
    out = dest / "summary.json"
    write_summary(result, out)
    typer.echo(f"wrote {out}: {result.n_cifs} cifs from {archive}")


@ingest_app.command("core-mof-ddec6")
def ingest_core_mof_ddec6(
    archive: Path = typer.Argument(..., help="Path to CoRE-MOF-1.0-DFT-minimized.tar.gz"),
    dest: Path = typer.Option(Path("benchmarks/cache/core_mof_dft_ddec6"), "--dest"),
) -> None:
    """Unpack the CoRE-MOF DDEC6 archive and emit a per-CIF charge table."""
    from .ingest.core_mof_ddec6 import (
        parse_ddec6_cif,
        unpack_core_mof_dft_ddec6,
        write_charge_table,
    )

    extracted = unpack_core_mof_dft_ddec6(archive, dest)
    cifs = list(extracted.rglob("*.cif"))
    out_dir = dest / "charge_tables"
    n_with_charges = 0
    for cif in cifs:
        table = parse_ddec6_cif(cif)
        if table.charges_e:
            n_with_charges += 1
        write_charge_table(table, out_dir / f"{table.refcode}.json")
    typer.echo(f"unpacked {len(cifs)} cifs; {n_with_charges} have DDEC6 charges populated; tables under {out_dir}")


@ingest_app.command("qmof")
def ingest_qmof(
    archive: Path = typer.Argument(..., help="Path to qmof_database.zip"),
    dest: Path = typer.Option(Path("benchmarks/cache/qmof"), "--dest"),
) -> None:
    """Unpack a QMOF zip and emit a summary."""
    from .ingest.qmof import unpack_qmof_zip

    result = unpack_qmof_zip(archive, dest)
    typer.echo(f"unpacked {result.n_cifs} cifs and {result.n_json} json blobs to {dest}")


@ingest_app.command("pacmof2")
def ingest_pacmof2(
    cif: Path = typer.Argument(..., help="Path to a single CIF to charge with PACMOF2"),
    out_dir: Path = typer.Option(Path("benchmarks/cache/pacmof_service_outputs"), "--out-dir"),
) -> None:
    """Run PACMOF2 on a CIF (requires the operator to have installed pacmof2)."""
    from .ingest.pacmof2 import run_pacmof2

    run = run_pacmof2(cif, out_dir)
    typer.echo(f"pacmof2 run: refcode={run.refcode} n_atoms={run.n_atoms} notes={run.notes}")


@ingest_app.command("eqeq")
def ingest_eqeq(
    cif: Path = typer.Argument(..., help="Path to a single CIF to charge with EQeq"),
    out_dir: Path = typer.Option(Path("benchmarks/cache/eqeq_outputs"), "--out-dir"),
) -> None:
    """Run EQeq on a CIF (requires operator-installed numat/EQeq binary)."""
    from .ingest.eqeq import run_eqeq

    run = run_eqeq(cif, out_dir)
    typer.echo(f"eqeq run: refcode={run.refcode} notes={run.notes}")


@ingest_app.command("odac")
def ingest_odac(
    archive: Path = typer.Argument(..., help="Path to operator-downloaded ODAC23 archive"),
    expected_md5: str = typer.Option("", "--expected-md5", help="Expected MD5 (from research/inventory)"),
) -> None:
    """Verify an ODAC23 archive (requires widom-atlas[odac] for LMDB summary)."""
    from .ingest.odac import verify_odac23_archive

    status = verify_odac23_archive(archive, expected_md5 or None)
    typer.echo(f"odac archive: matches={status.matches} actual_md5={status.actual_md5} notes={status.notes}")


@ingest_app.command("ccdc-cif")
def ingest_ccdc_cif(
    cif: Path = typer.Argument(..., help="Path to operator-supplied gas-loaded CIF"),
    material_id: str = typer.Option(..., "--material-id"),
    gas: str = typer.Option(..., "--gas"),
    label: str = typer.Option(..., "--label", help="Site label, e.g. OMS_Mg_CO2"),
    elements: str = typer.Option(..., "--elements", help="Comma-separated guest elements, e.g. C,O"),
    source_doi: str = typer.Option(..., "--source-doi"),
    out: Path = typer.Option(..., "--out", help="Output SiteReferenceEntry JSON path"),
) -> None:
    """Extract a gas-CoM site from an operator-supplied CIF and emit a SiteReferenceEntry JSON."""
    from .ingest.ccdc_cif import extract_gas_centroid_from_cif, write_site_reference_entry

    elem_set = {e.strip() for e in elements.split(",") if e.strip()}
    rec = extract_gas_centroid_from_cif(
        cif, material_id=material_id, gas=gas, site_label=label, gas_element_set=elem_set
    )
    write_site_reference_entry(rec, source_doi=source_doi, out_path=out)
    typer.echo(f"wrote {out}: {rec.label} centroid_frac={rec.centroid_frac}")


@app.command("prepare-validation-inputs")
def prepare_validation_inputs_cmd(
    cache_root: Path = typer.Option(Path("benchmarks/cache"), "--cache-root"),
    mofxdb_ids: str = typer.Option(
        "173866,173867,173868,173869,173870", "--mofxdb-ids",
        help="Comma-separated MOFX-DB record IDs to live-fetch.",
    ),
) -> None:
    """Live-fetch licence-safe public inputs needed by the v0.4 validation suite."""
    from .validation.prepare_inputs import prepare_validation_inputs

    ids = [int(x.strip()) for x in mofxdb_ids.split(",") if x.strip()]
    report = prepare_validation_inputs(cache_root=cache_root, mofxdb_ids=ids)
    n_ok_struct = sum(1 for r in report.structures if r.status in ("ok", "cached"))
    n_ok_ff = sum(1 for r in report.force_field if r.status in ("ok", "cached"))
    n_ok_mofx = sum(1 for r in report.mofxdb if r.status in ("ok", "cached"))
    n_ok_upf = sum(1 for r in report.user_parameter_files if r.status == "ok")
    typer.echo(
        f"prepare-validation-inputs: structures={n_ok_struct}/{len(report.structures)} "
        f"force_field={n_ok_ff}/{len(report.force_field)} "
        f"mofxdb={n_ok_mofx}/{len(report.mofxdb)} "
        f"upfs={n_ok_upf}/{len(report.user_parameter_files)} "
        f"blockers={len(report.blockers)}"
    )
    typer.echo(f"provenance: {cache_root / 'provenance.json'}")


@app.command("run-validation-suite")
def run_validation_suite(
    out_dir: Path = typer.Option(Path("benchmarks/results/v0.4-validation"), "--out-dir"),
    cache_root: Path = typer.Option(Path("benchmarks/cache"), "--cache-root"),
    skip_parity_raspa3: bool = typer.Option(False, "--skip-parity-raspa3"),
    n_insertions: int = typer.Option(2048, "--n-insertions",
                                    help="Per-case Widom insertions (flagship cases)."),
    n_grid: int = typer.Option(40, "--n-grid",
                              help="Atlas density grid edge length."),
) -> None:
    """Run the full v0.4 release-gate validation suite and emit the audit + 10 tables.

    Pipeline (one continuous run):
    1. flagship cases through the internal evaluator with strict thresholds (0.10 / 2.0)
    2. one full atlas run (samples → density → basins → reports → tables) for each
       atlas-eligible flagship case (gas in {CO2, N2, CH4})
    3. MOFX-DB simin parity (5 deterministic records, real evaluator, blocker rows
       captured per-record)
    4. 10 machine-readable tables + FINAL_V04_VALIDATION_AUDIT.md
    """
    from .data_registry.registry import dataset_status, list_datasets
    from .evaluator.parity import (
        ParityRow,
        is_raspa3_available,
    )
    from .ingest.mofxdb import parse_mofxdb_record
    from .validation.audit import render_audit_markdown, write_audit
    from .validation.case_runner import (
        CaseResult,
        CaseSpec,
        run_case,
        write_case_results_jsonl,
    )
    from .validation.end_to_end import (
        ALLOWED_ATLAS_GASES,
        EndToEndCaseResult,
        run_one_end_to_end,
        write_end_to_end_results,
    )
    from .validation.flagship_specs import FLAGSHIP_DEFS
    from .validation.mofx_parity import run_mofxdb_simin_parity
    from .validation.prepare_inputs import parse_raspa_mixing_rules
    from .validation.v04_tables import write_all_tables

    structures_dir = cache_root / "structures"
    upf_dir = cache_root / "user_parameter_files"
    ff_path = cache_root / "raspa2_ff" / "ExampleMOFsForceField_mixing_rules.def"
    mofxdb_cache = cache_root / "mofxdb"
    mofxdb_cifs = cache_root / "mofxdb_cifs"

    if not ff_path.exists():
        _fail(f"force-field table missing: {ff_path}\n"
              "run `widom-atlas prepare-validation-inputs` first.")
    ff_table = parse_raspa_mixing_rules(ff_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    atlas_runs_dir = out_dir / "atlas"
    atlas_runs_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[1/5] Running {len(FLAGSHIP_DEFS)} flagship cases (strict thresholds)")
    specs: list[CaseSpec] = []
    for d in FLAGSHIP_DEFS:
        cif = structures_dir / d.expected_structure_filename
        upf = upf_dir / f"{cif.stem}_{d.gas}.json"
        specs.append(CaseSpec(
            case_id=d.case_id, framework_name=d.framework_name,
            structure_path=cif, gas=d.gas, temperature_K=d.temperature_K,
            user_parameter_file_path=upf, n_insertions=n_insertions, seed=0,
            r_cut_A=d.r_cut_A, grid_mode="stochastic_uniform", tier="flagship",
            reference_KH_mol_per_kg_per_Pa=d.reference_KH,
            reference_Qads_kJ_per_mol=d.reference_Qads_kJ_per_mol,
            reference_doi=d.reference_doi, notes=d.notes,
        ))
    cases: list[CaseResult] = []
    for s in specs:
        r = run_case(s)
        cases.append(r)
        typer.echo(
            f"  {s.case_id}: status={r.status} log10_KH_int={r.log10_KH_internal} "
            f"log10_KH_ref={r.log10_KH_reference} Δlog={r.delta_log10_KH} "
            f"ΔQ={r.delta_Qads_kJ_per_mol} pass={r.pass_overall}"
        )
    write_case_results_jsonl(cases, out_dir / "flagship_cases.jsonl")

    typer.echo("[2/5] Running atlas pipeline end-to-end on atlas-eligible cases")
    e2e_results: list[EndToEndCaseResult] = []
    for s in specs:
        if s.gas not in ALLOWED_ATLAS_GASES:
            typer.echo(f"  {s.case_id}: skipped (gas={s.gas} not in atlas allow-list)")
            continue
        if not (s.structure_path.exists() and s.user_parameter_file_path.exists()):
            typer.echo(f"  {s.case_id}: skipped (inputs missing)")
            continue
        e2e_r = run_one_end_to_end(
            case_id=s.case_id, framework_name=s.framework_name,
            structure_path=s.structure_path, upf_path=s.user_parameter_file_path,
            gas=s.gas, temperature_K=s.temperature_K, n_insertions=n_insertions,
            seed=s.seed, r_cut_A=s.r_cut_A, grid_mode=s.grid_mode,
            out_root=atlas_runs_dir, n_grid=(n_grid, n_grid, n_grid),
        )
        e2e_results.append(e2e_r)
        typer.echo(
            f"  {s.case_id}: pipeline_completed={e2e_r.pipeline_completed} "
            f"K_H={e2e_r.KH_mol_per_kg_per_Pa} log10_KH={e2e_r.log10_KH} "
            f"Q={e2e_r.Qads_kJ_per_mol} basins={e2e_r.n_basins}"
        )
    write_end_to_end_results(e2e_results, out_dir / "atlas_runs.jsonl")

    typer.echo("[3/5] Running MOFX-DB simin parity (real evaluator)")
    parity_rows: list[ParityRow] = []
    raspa3_available = is_raspa3_available() and not skip_parity_raspa3

    distilled_records: list[dict[str, Any]] = []
    if mofxdb_cache.exists():
        for f in sorted(mofxdb_cache.glob("*.json")):
            try:
                distilled_records.append(parse_mofxdb_record(
                    json.loads(f.read_text(encoding="utf-8"))
                ))
            except (json.JSONDecodeError, KeyError) as exc:
                typer.echo(f"  WARNING: skipping {f.name}: {exc}")

    mofx_parity_rows, mofxdb_blocker_rows = run_mofxdb_simin_parity(
        distilled_records=distilled_records,
        cif_dir=mofxdb_cifs, upf_dir=upf_dir, ff_table=ff_table,
        n_records_to_run=5, seed=17, n_insertions=512, r_cut_A=10.0,
    )
    parity_rows.extend(mofx_parity_rows)
    typer.echo(
        f"  MOFX simin parity: {sum(1 for p in mofx_parity_rows if p.pass_overall)}/"
        f"{len(mofx_parity_rows)} passed; {len(mofxdb_blocker_rows)} blockers recorded"
    )

    raspa3_skip_row = ParityRow(
        case_id="raspa3-mfi-CO2-298", kind="raspa3_reference",
        framework_name="MFI", component_name="CO2", temperature_K=298.15,
        n_insertions=n_insertions, seed=0,
        log10_KH_internal=None, log10_KH_reference=None, delta_log10_KH=None,
        Qads_internal_kJ_per_mol=None, Qads_reference_kJ_per_mol=None,
        delta_Qads_kJ_per_mol=None, threshold_log10_KH=0.10,
        threshold_Qads_kJ_per_mol=2.0,
        pass_log10_KH=False, pass_Qads=False, pass_overall=False,
        reference_provenance_sha256="",
        notes=(
            "reference unavailable; only internal scalars recorded "
            "(raspa3 binary not on PATH; audit verdict tolerates skip)."
            if not raspa3_available else
            "reference unavailable; raspa3 reference run not implemented in this CLI step "
            "(operator can run raspa3 manually on benchmarks/cache/structures/MFI.cif + the cached UPF)."
        ),
    )
    parity_rows.append(raspa3_skip_row)

    parity_path = out_dir / "parity.jsonl"
    with open(parity_path, "w", encoding="utf-8") as fh:
        for parity_row in parity_rows:
            fh.write(json.dumps(parity_row.__dict__, sort_keys=True) + "\n")

    blocker_path = out_dir / "blockers.jsonl"
    with open(blocker_path, "w", encoding="utf-8") as fh:
        for b in mofxdb_blocker_rows:
            fh.write(json.dumps(b, sort_keys=True) + "\n")

    typer.echo("[4/5] Writing tables + audit")
    repo_root = Path(".")
    registry = list_datasets()
    registry_status_rows: list[dict[str, Any]] = []
    for ds in registry:
        st = dataset_status(ds, repo_root=repo_root)
        registry_status_rows.append({
            "dataset_name": ds.name,
            "kind": ds.kind,
            "license": ds.license,
            "present": st["present"],
            "verified": st["verified"],
            "cache_path": st["cache_path"],
            "note": st.get("note", ""),
        })
    provenance_rows: list[dict[str, Any]] = [
        {
            "dataset_name": ds.name,
            "primary_doi": ds.primary_doi,
            "predecessor_doi": ds.predecessor_doi,
            "license": ds.license,
            "redistribution_status": ds.redistribution_status,
            "primary_url": ds.primary_url,
        }
        for ds in registry
    ]
    convergence_rows = [
        {"case_id": c.case_id, "tier": c.tier, "n_insertions": c.n_insertions_used,
         "log10_KH_internal": c.log10_KH_internal,
         "Qads_internal_kJ_per_mol": c.Qads_internal_kJ_per_mol,
         "status": c.status}
        for c in cases
    ]
    site_match_rows = [
        {"case_id": e.case_id, "framework": e.framework_name, "gas": e.gas,
         "temperature_K": e.temperature_K, "n_basins": e.n_basins,
         "dominant_basin_weight": e.dominant_basin_weight,
         "dominant_basin_centroid_frac": list(e.dominant_basin_centroid_frac)
             if e.dominant_basin_centroid_frac else None,
         "dominant_basin_centroid_A": list(e.dominant_basin_centroid_A)
             if e.dominant_basin_centroid_A else None,
         "atlas_run_dir": e.atlas_run_dir,
         "pipeline_completed": e.pipeline_completed}
        for e in e2e_results
    ]
    table_paths = write_all_tables(
        tables_dir,
        cases=cases,
        parity_rows=parity_rows,
        convergence_rows=convergence_rows,
        charge_sensitivity_rows=[],
        site_match_rows=site_match_rows,
        provenance_rows=provenance_rows,
        registry_status_rows=registry_status_rows,
    )

    blocker_table = {
        "schema_version": "0.4", "table_id": "T10",
        "title": "Blocker rows (MOFX simin parity, end-to-end attempts)",
        "rows": mofxdb_blocker_rows, "n_rows": len(mofxdb_blocker_rows),
        "notes": "structured (record_id, material, gas, missing_item, needed_path, reason, source_url, licence_status) per blocker",
    }
    t10_path = tables_dir / "T10_blocker_rows.json"
    t10_path.write_text(json.dumps(blocker_table, indent=2, sort_keys=True), encoding="utf-8")
    table_paths["T10"] = t10_path

    n_e2e_complete = sum(1 for e in e2e_results if e.pipeline_completed)
    extra_notes = (
        f"v0.4 actually-ran build: {n_e2e_complete}/{len(e2e_results)} atlas-eligible "
        f"flagship cases completed the full pipeline (samples → density → basins → reports). "
        f"Strict thresholds applied: flagship 0.10 / 2.0; broad 0.20 / 4.0; "
        f"exploratory 0.40 / 7.0. {len(mofxdb_blocker_rows)} MOFX simin rows are blocked "
        f"(see T10_blocker_rows.json + blockers.jsonl). raspa3 reference: "
        f"{'available' if raspa3_available else 'NOT on PATH (skipped, audit verdict tolerates skip)'}."
    )
    md = render_audit_markdown(
        cases=cases, parity_rows=parity_rows, table_paths=table_paths,
        extra_notes=extra_notes,
    )
    write_audit(out_dir / "FINAL_V04_VALIDATION_AUDIT.md", md)
    typer.echo(f"audit written: {out_dir / 'FINAL_V04_VALIDATION_AUDIT.md'}")
    typer.echo(f"[5/5] DONE. atlas runs completed: {n_e2e_complete}/{len(e2e_results)}; "
               f"flagship pass: {sum(1 for c in cases if c.pass_overall)}/{len(cases)}; "
               f"MOFX parity pass: {sum(1 for p in mofx_parity_rows if p.pass_overall)}/{len(mofx_parity_rows)}.")


if __name__ == "__main__":  # pragma: no cover
    app()
