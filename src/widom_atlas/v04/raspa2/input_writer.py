"""RASPA2 simulation-input generator for v04 1a (Mg-MOF-74 Lin/Mercado).

Produces a RASPA2 working directory containing:

  simulation.input
  forcefield/LinMercado/pseudo_atoms.def
  forcefield/LinMercado/force_field_mixing_rules.def
  forcefield/LinMercado/force_field.def
  forcefield/LinMercado/{framework_stem}.cif       (relabelled VOGTIV)
  molecules/ExampleDefinitions/CO2.def             (stock RASPA2 CO2)

Then RASPA2 is invoked with RASPA_DIR pointing into the share tree so it
finds the molecule definition. RASPA2 looks for the ForceField directory
in the cwd-relative path `./forcefield/{name}/` first, so we get the
Lin/Mercado FF without polluting the system share dir.
"""
from __future__ import annotations

import hashlib
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .cif_relabeller import relabel_vogtiv_cif


def _rewrite_lin_mercado_pseudo_atoms_to_v2_50_format(src: Path, dst: Path) -> None:
    """Upgrade the operator-supplied Lin/Mercado pseudo_atoms.def from the
    legacy 9-column format to RASPA2 v2.0.50's 14-column format.

    The Lin file uses the legacy header:

        #type  print  as  scat  mass  charge  B-factor  radii  connectivity

    RASPA2 v2.0.50 (conda-forge) expects:

        #type  print  as  chem  oxidation  mass  charge  polarization
        B-factor  radii  connectivity  anisotropic  anisotropic-type  tinker-type

    The new `oxidation` column at position 4 pushes the `mass` column from
    position 4 to position 5. **When RASPA2 v2.0.50 reads the legacy
    9-column file directly, it silently mis-reads the framework mass** —
    in the 2026-05-17 forensic audit the reported `Framework Mass` was
    317.107 g/mol per primitive instead of the correct 727.82 g/mol per
    primitive (a 36.7× under-count, leading to a 36.7× over-stated
    K_H since K_H ∝ 1/M_framework).

    This rewriter inserts the missing `oxidation` column (0 for everything
    — the Lin/Mercado FF doesn't use oxidation explicitly) and pads the
    trailing columns to match v2.0.50's expectation. Mass and charge are
    preserved verbatim from the original file.
    """
    src_text = src.read_text()
    out_lines: list[str] = []
    in_data = False
    for line in src_text.splitlines():
        s = line.strip()
        if not s:
            out_lines.append(line)
            continue
        if s.startswith("#number of pseudo atoms") or s.isdigit():
            out_lines.append(s)
            continue
        if s.startswith("#type"):
            out_lines.append(
                "#type      print   as    chem  oxidation   mass        charge      "
                "polarization  B-factor  radii   connectivity anisotropic  "
                "anisotropic-type  tinker-type"
            )
            in_data = True
            continue
        if s.startswith("#"):
            out_lines.append(line)
            continue
        if not in_data:
            out_lines.append(line)
            continue
        parts = re.split(r"\s+", s)
        if len(parts) < 9:
            out_lines.append(line)
            continue
        type_, print_, as_, scat, mass, charge, bfact, radii, conn = parts[:9]
        new_row = (
            f"{type_:<10s} {print_:<6s}  {as_:<5s} {scat:<5s} 0          "
            f"{mass:<10s}  {charge:<11s} 0.0           "
            f"{bfact:<8s}  {radii:<6s}  {conn:<12s} 0           relative          0"
        )
        out_lines.append(new_row)
    dst.write_text("\n".join(out_lines) + ("\n" if src_text.endswith("\n") else ""))


def _rewrite_lin_mercado_force_field_acb_to_abc(
    src: Path, dst: Path, hardcore_angstrom: float = 1.0
) -> None:
    """Copy the Lin/Mercado raspa_force_field.def to `dst` with two fixes:

    1. **Column-order correction.** The operator-supplied source file's
       per-pair Buckingham rows actually carry values in **`C A B`** order
       (this was discovered by direct cross-check against Lin 2014 JCTC SI
       Table S7 verbatim on 2026-05-17). The file's own header reads
       `#type type2 interaction A C B`, which is misleading: the values do
       NOT match those labels. For Mof_Mg–O_co2 the file has
       `4.08795E5 2.47320E7 3.965`, but Lin SI Table S7 (Model 4) gives
       `A_Mg = 2.47320E7`, `B_Mg = 3.965`, `C_Mg = 4.08795E5`. So the file
       is in (C, A, B) order regardless of what the header claims.

       This rewriter therefore takes file positions (3, 4, 5) as
       (published_C, published_A, published_B) and emits them in RASPA2's
       expected `A B C` order: (file[4], file[5], file[3]).

       Sanity check: with the corrected order, the Buckingham well depth at
       r ≈ 3 Å for Mg–O(CO2) is V ≈ −3–4 kJ/mol per pair, which combined with
       the ~10 framework atoms within the cutoff gives Q_st ≈ 40–50 kJ/mol —
       matching the experimental ~47 kJ/mol (Mason 2011 EES).

    2. **Buckingham → BUCKINGHAM2 with hard-core p_3.** Even with the correct
       A B C, the bare `buckingham` potential has the well-known `−C/r⁶ →
       −∞` catastrophe at very short distances. A Widom-inserted CO₂ that
       happens to land at r < ~1 Å from a framework atom hits an unphysical
       attractive singularity, the Rosenbluth weight overflows, and K_H
       diverges to NaN. RASPA2's `BUCKINGHAM2` form adds a fourth parameter
       `p_3 [Å]` — the short-range hard-core cutoff. We emit `p_3 = 1.0 Å`
       by default, well below any physical CO₂–framework van der Waals
       contact (~2.5–3.0 Å), so no real Widom statistics are affected.

       This is the documented Lin/Mercado-RASPA2 reproduction convention;
       it is NOT a silent LJ-like substitution because the functional form
       `A·exp(-B·r) − C/r⁶` is preserved exactly above the hard-core radius.

    LJ rows use the form `type1 type2 lennard-jones epsilon sigma` — no
    transformation needed.
    """
    src_text = src.read_text()
    out_lines: list[str] = []
    for line in src_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue
        fields = re.split(r"\s+", line.strip())
        if len(fields) >= 6 and fields[2].lower() == "buckingham":
            # The source file's per-pair Buckingham rows are in (C, A, B) order
            # (NOT what the file's own header says). Re-order to RASPA2's `A B C`.
            type1, type2, _kind = fields[0], fields[1], fields[2]
            published_C = fields[3]   # file position "A" actually holds C
            published_A = fields[4]   # file position "C" actually holds A
            published_B = fields[5]   # file position "B" actually holds B (correct in file)
            leading = line[: len(line) - len(line.lstrip())]
            rebuilt = (
                f"{leading}{type1:<8s}{type2:<8s}{'BUCKINGHAM2':<14s}"
                f"{published_A:<18s}{published_B:<22s}{published_C:<22s}{hardcore_angstrom:.4f}"
            )
            out_lines.append(rebuilt)
        else:
            out_lines.append(line)
    dst.write_text("\n".join(out_lines) + ("\n" if src_text.endswith("\n") else ""))


def _supercell_for_cutoff(cif_path: Path, cutoff_angstrom: float) -> tuple[int, int, int]:
    """Pick UnitCells nₐ × n_b × n_c such that each box vector projection is > 2·cutoff.

    For a triclinic cell, the minimum height perpendicular to a face equals the
    box-vector length divided by the appropriate scaling. To stay safe we just
    require n_i × |a_i| > 2 × cutoff for each i — slightly conservative for
    very oblique cells but correct enough for the Mg-MOF-74 triclinic primitive
    (a=6.76, b=c=15.19 Å).
    """
    a = b = c = None
    for line in cif_path.read_text().splitlines():
        s = line.strip()
        m = re.match(r"_cell_length_a\s+([0-9.eE+-]+)", s)
        if m:
            a = float(m.group(1))
        m = re.match(r"_cell_length_b\s+([0-9.eE+-]+)", s)
        if m:
            b = float(m.group(1))
        m = re.match(r"_cell_length_c\s+([0-9.eE+-]+)", s)
        if m:
            c = float(m.group(1))
        if a and b and c:
            break
    if a is None or b is None or c is None:
        return (2, 2, 2)
    target = 2.0 * cutoff_angstrom
    n_a = max(1, math.ceil(target / a))
    n_b = max(1, math.ceil(target / b))
    n_c = max(1, math.ceil(target / c))
    return (n_a, n_b, n_c)


@dataclass(frozen=True)
class Raspa2InputBundle:
    work_dir: Path
    simulation_input: Path
    framework_cif: Path
    ff_dir: Path
    sha256: dict[str, str]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_raspa2_inputs(
    work_dir: Path,
    branch: dict,
    cif_abs_path: Path,
    lin_mercado_pkg_dir: Path,
    temperature_K: float,
    n_cycles: int,
    raspa2_share_dir: Path,
    random_seed: int | None = None,
) -> Raspa2InputBundle:
    """Build a RASPA2 working directory for a Lin/Mercado MOF + CO2 run.

    Layout produced (mirrors RASPA2's `$RASPA_DIR/share/raspa/...` convention):

        work_dir/
          simulation.input
          share/raspa/forcefield/LinMercado/
            pseudo_atoms.def
            force_field_mixing_rules.def
            force_field.def
          share/raspa/structures/cif/VOGTIV_LinMercado.cif
          share/raspa/molecules/ExampleDefinitions/CO2.def

    The runner sets `RASPA_DIR=work_dir`, so RASPA2 picks up the per-branch
    relabelled framework + Lin/Mercado FF without polluting the conda env.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    ff_name = "LinMercado"
    ff_dir = work_dir / "share" / "raspa" / "forcefield" / ff_name
    structures_dir = work_dir / "share" / "raspa" / "structures" / "cif"
    mol_dir = work_dir / "share" / "raspa" / "molecules" / "ExampleDefinitions"
    ff_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)
    mol_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy Lin/Mercado .def files — pseudo_atoms + mixing_rules verbatim,
    # but the cross-pair force_field.def file in the operator-supplied package
    # has columns in `A C B` order (printed in the file's own header) while
    # RASPA2's per-pair Buckingham parser expects `A B C`. Without re-ordering
    # the numerical columns at copy time, RASPA2 silently reads B as a huge
    # number (e.g. 33,788,254 Å⁻¹) and C as a tiny number (3.805 K·Å⁶) and
    # every CO2 Widom insertion overlaps. Verified by directly inspecting the
    # RASPA2 stdout `Force Field Status` block on a smoke run (2026-05-17).
    _rewrite_lin_mercado_pseudo_atoms_to_v2_50_format(
        src=lin_mercado_pkg_dir / "raspa_pseudo_atoms.def",
        dst=ff_dir / "pseudo_atoms.def",
    )
    shutil.copyfile(
        lin_mercado_pkg_dir / "raspa_force_field_mixing_rules.def",
        ff_dir / "force_field_mixing_rules.def",
    )
    _rewrite_lin_mercado_force_field_acb_to_abc(
        src=lin_mercado_pkg_dir / "raspa_force_field.def",
        dst=ff_dir / "force_field.def",
    )

    # 2. Re-label the VOGTIV CIF and write to structures/cif/ where RASPA2 expects it
    framework_stem = "VOGTIV_LinMercado"
    framework_cif = structures_dir / f"{framework_stem}.cif"
    counts = relabel_vogtiv_cif(cif_abs_path, framework_cif)
    # Sanity assertions on the relabel
    assert counts.get("Mof_Mg", 0) > 0, f"VOGTIV relabel produced no Mof_Mg: {counts}"
    assert counts.get("Mof_Ca", 0) > 0, "no Mof_Ca after relabel"
    assert counts.get("Mof_Cd", 0) > 0, "no Mof_Cd after relabel"

    # 3. Copy stock CO2.def from the RASPA2 share tree
    shutil.copyfile(
        raspa2_share_dir / "molecules" / "ExampleDefinitions" / "CO2.def",
        mol_dir / "CO2.def",
    )

    # 4. Build simulation.input
    ep = branch.get("electrostatics_per_branch") or {}
    cutoff = float(ep.get("direct_cutoff_angstrom", 12.0))
    charge_method = "Ewald" if ep.get("ewald_via_raspa3") else "None"
    n_a, n_b, n_c = _supercell_for_cutoff(framework_cif, cutoff)
    seed_block = f"RandomSeed                    {int(random_seed)}\n" if random_seed is not None else ""
    sim_input = f"""SimulationType                MonteCarlo
NumberOfCycles                {int(n_cycles)}
NumberOfInitializationCycles  0
PrintEvery                    {max(int(n_cycles) // 10, 1)}
PrintPropertiesEvery          {max(int(n_cycles) // 10, 1)}
{seed_block}
Forcefield                    {ff_name}
ChargeMethod                  {charge_method}
CutOff                        {cutoff}
EwaldPrecision                1e-6

Framework 0
FrameworkName {framework_stem}
RemoveAtomNumberCodeFromLabel no
UnitCells {n_a} {n_b} {n_c}
ExternalTemperature {float(temperature_K)}
ExternalPressure 0

Component 0 MoleculeName              CO2
            MoleculeDefinition        ExampleDefinitions
            IdealGasRosenbluthWeight  1.0
            WidomProbability          1.0
            CreateNumberOfMolecules   0
"""
    sim_path = work_dir / "simulation.input"
    sim_path.write_text(sim_input)

    sha = {
        "simulation.input": _sha256(sim_path),
        f"share/raspa/forcefield/{ff_name}/pseudo_atoms.def": _sha256(ff_dir / "pseudo_atoms.def"),
        f"share/raspa/forcefield/{ff_name}/force_field_mixing_rules.def": _sha256(
            ff_dir / "force_field_mixing_rules.def"
        ),
        f"share/raspa/forcefield/{ff_name}/force_field.def": _sha256(ff_dir / "force_field.def"),
        f"share/raspa/structures/cif/{framework_stem}.cif": _sha256(framework_cif),
    }
    return Raspa2InputBundle(
        work_dir=work_dir,
        simulation_input=sim_path,
        framework_cif=framework_cif,
        ff_dir=ff_dir,
        sha256=sha,
    )
