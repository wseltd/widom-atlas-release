"""Smoke tests for the v0.4 internal Widom evaluator."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.cell import Cell


def _toy_si_o_mfi_like() -> Atoms:
    a = 13.4
    cell = Cell.fromcellpar([a, a, a, 90.0, 90.0, 90.0])
    positions = []
    symbols = []
    nx = 3
    for i in range(nx):
        for j in range(nx):
            for k in range(nx):
                u = (i + 0.25) / nx * a
                v = (j + 0.25) / nx * a
                w = (k + 0.25) / nx * a
                positions.append([u, v, w])
                symbols.append("Si")
                positions.append([u + 0.6, v, w])
                positions.append([u - 0.6, v, w])
                symbols.extend(["O", "O"])
    return Atoms(symbols=symbols, positions=np.array(positions), cell=cell, pbc=True)


def _toy_user_parameter_file() -> Path:
    """Write a minimal UPF JSON to /tmp and return the path."""
    import json
    import tempfile

    payload = {
        "framework_atom_types": [
            {"label": "Si", "sigma_A": 2.30, "epsilon_K": 22.0, "charge_e": 1.5, "source": "test"},
            {"label": "O", "sigma_A": 3.30, "epsilon_K": 53.0, "charge_e": -0.75, "source": "test"},
        ],
        "gas_sites": [
            {"label": "C_co2", "sigma_A": 2.745, "epsilon_K": 29.933, "charge_e": 0.6512, "source": "test"},
            {"label": "O_co2", "sigma_A": 3.017, "epsilon_K": 85.671, "charge_e": -0.3256, "source": "test"},
        ],
        "mixing_rules": "Lorentz-Berthelot",
        "electrostatics": "Wolf",
        "redistribution_status": "user_supplied_not_bundled",
    }
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        fp = Path(tf.name)
    fp.write_text(json.dumps(payload), encoding="utf-8")
    return fp


def test_component_co2_geometry_is_neutral_and_3site() -> None:
    from widom_atlas.evaluator.component import co2_garcia_sanchez_2009

    co2 = co2_garcia_sanchez_2009()
    assert co2.n_sites == 3
    assert abs(float(co2.site_charges.sum())) < 1e-6
    # C-O distance from C to O (site 1) ≈ 1.149
    d = float(np.linalg.norm(co2.site_offsets[1] - co2.site_offsets[0]))
    assert abs(d - 1.149) < 1e-6


def test_component_n2_trappe_is_neutral() -> None:
    from widom_atlas.evaluator.component import n2_trappe

    n2 = n2_trappe()
    assert n2.n_sites == 3  # 2 N + 1 virtual COM
    assert abs(float(n2.site_charges.sum())) < 1e-6


def test_random_rotation_matrix_is_orthonormal() -> None:
    from widom_atlas.evaluator.component import random_rotation_matrix

    rng = np.random.default_rng(0)
    R = random_rotation_matrix(rng)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
    assert abs(float(np.linalg.det(R)) - 1.0) < 1e-10


def test_grid_deterministic_returns_n_per_axis_cubed_points() -> None:
    from widom_atlas.evaluator.grid import deterministic_uniform_grid

    cell = np.diag([10.0, 10.0, 10.0])
    pts = deterministic_uniform_grid(cell, n_per_axis=4)
    assert pts.shape == (64, 3)
    # all inside cell
    assert (pts >= 0).all() and (pts < 10.0 + 1e-9).all()


def test_lj_energy_is_repulsive_at_short_distance() -> None:
    from widom_atlas.evaluator.energy import lj_energy_K

    r = np.array([1.5, 2.0, 5.0])
    sig = np.array([3.0, 3.0, 3.0])
    eps = np.array([100.0, 100.0, 100.0])
    e = lj_energy_K(r, sig, eps, r_cut_A=12.0)
    assert e > 0  # net repulsive (1.5 << sigma)


def test_wolf_coulomb_zero_for_zero_charges() -> None:
    from widom_atlas.evaluator.energy import wolf_coulomb_energy_K

    r = np.array([3.0, 5.0, 8.0])
    q_pair = np.array([0.0, 0.0, 0.0])
    e = wolf_coulomb_energy_K(r, q_pair, r_cut_A=12.0)
    assert abs(e) < 1e-12


def test_wolf_coulomb_positive_for_like_charges() -> None:
    from widom_atlas.evaluator.energy import wolf_coulomb_energy_K

    r = np.array([3.0])
    q_pair = np.array([0.5 * 0.5])  # like charges → repulsive
    e = wolf_coulomb_energy_K(r, q_pair, r_cut_A=12.0)
    assert e > 0


def test_run_widom_evaluator_unresolved_framework_atoms() -> None:
    from widom_atlas.evaluator.component import co2_garcia_sanchez_2009
    from widom_atlas.evaluator.ff_loader import load_user_parameter_file
    from widom_atlas.evaluator.runner import run_widom_evaluator

    a = 12.0
    atoms = Atoms(
        symbols=["W"],  # tungsten — not in our toy FF
        positions=[[0.0, 0.0, 0.0]],
        cell=Cell.fromcellpar([a, a, a, 90.0, 90.0, 90.0]),
        pbc=True,
    )
    upf = load_user_parameter_file(_toy_user_parameter_file())
    res = run_widom_evaluator(
        atoms=atoms,
        framework_name="ToyW",
        user_parameter_file=upf,
        component=co2_garcia_sanchez_2009(),
        temperature_K=298.0,
        n_insertions=10,
        seed=0,
    )
    assert res.status == "unresolved_framework_atoms"
    assert res.KH_mol_per_kg_per_Pa is None


def test_run_widom_evaluator_smoke_si_o() -> None:
    from widom_atlas.evaluator.component import co2_garcia_sanchez_2009
    from widom_atlas.evaluator.ff_loader import load_user_parameter_file
    from widom_atlas.evaluator.runner import run_widom_evaluator

    atoms = _toy_si_o_mfi_like()
    upf = load_user_parameter_file(_toy_user_parameter_file())
    res = run_widom_evaluator(
        atoms=atoms,
        framework_name="ToyMFI",
        user_parameter_file=upf,
        component=co2_garcia_sanchez_2009(),
        temperature_K=298.0,
        n_insertions=64,
        seed=42,
        grid_mode="deterministic_uniform",
        r_cut_A=8.0,
    )
    assert res.status == "ok"
    assert res.n_insertions_used > 0
    assert res.KH_mol_per_kg_per_Pa is None or res.KH_mol_per_kg_per_Pa >= 0
    assert math.isfinite(float(res.e_lj_K.mean()))


def test_parity_assess_passes_with_fake_rows() -> None:
    from widom_atlas.evaluator.parity import ParityRow, assess_parity_outcome

    rows = [
        ParityRow(
            case_id="raspa3-mfi",
            kind="raspa3_reference",
            framework_name="MFI",
            component_name="CO2",
            temperature_K=298.0,
            n_insertions=64,
            seed=0,
            log10_KH_internal=-5.0,
            log10_KH_reference=-5.05,
            delta_log10_KH=0.05,
            Qads_internal_kJ_per_mol=22.0,
            Qads_reference_kJ_per_mol=22.5,
            delta_Qads_kJ_per_mol=0.5,
            threshold_log10_KH=0.10,
            threshold_Qads_kJ_per_mol=2.0,
            pass_log10_KH=True,
            pass_Qads=True,
            pass_overall=True,
            reference_provenance_sha256="sha",
            notes="ok",
        ),
        *[
            ParityRow(
                case_id=f"mofx-{i}",
                kind="mofxdb_simin",
                framework_name=f"MOF-{i}",
                component_name="CO2",
                temperature_K=298.0,
                n_insertions=64,
                seed=i,
                log10_KH_internal=-5.0,
                log10_KH_reference=-5.0,
                delta_log10_KH=0.0,
                Qads_internal_kJ_per_mol=22.0,
                Qads_reference_kJ_per_mol=22.0,
                delta_Qads_kJ_per_mol=0.0,
                threshold_log10_KH=0.10,
                threshold_Qads_kJ_per_mol=2.0,
                pass_log10_KH=True,
                pass_Qads=True,
                pass_overall=True,
                reference_provenance_sha256="sha",
                notes="ok",
            )
            for i in range(5)
        ],
    ]
    outcome = assess_parity_outcome(rows)
    assert outcome["raspa3_pass"] is True
    assert outcome["mofxdb_pass_count"] == 5
    assert outcome["overall_pass"] is True


def test_parity_skipped_raspa_with_4_mofx_passes() -> None:
    from widom_atlas.evaluator.parity import ParityRow, assess_parity_outcome

    rows = [
        ParityRow(
            case_id="raspa3-skip",
            kind="raspa3_reference",
            framework_name="MFI",
            component_name="CO2",
            temperature_K=298.0,
            n_insertions=64,
            seed=0,
            log10_KH_internal=-5.0,
            log10_KH_reference=None,
            delta_log10_KH=None,
            Qads_internal_kJ_per_mol=22.0,
            Qads_reference_kJ_per_mol=None,
            delta_Qads_kJ_per_mol=None,
            threshold_log10_KH=0.10,
            threshold_Qads_kJ_per_mol=2.0,
            pass_log10_KH=False,
            pass_Qads=False,
            pass_overall=False,
            reference_provenance_sha256="",
            notes="reference unavailable; only internal scalars recorded",
        ),
        *[
            ParityRow(
                case_id=f"mofx-{i}",
                kind="mofxdb_simin",
                framework_name=f"MOF-{i}",
                component_name="CO2",
                temperature_K=298.0,
                n_insertions=64,
                seed=i,
                log10_KH_internal=-5.0,
                log10_KH_reference=-5.0,
                delta_log10_KH=0.0,
                Qads_internal_kJ_per_mol=22.0,
                Qads_reference_kJ_per_mol=22.0,
                delta_Qads_kJ_per_mol=0.0,
                threshold_log10_KH=0.10,
                threshold_Qads_kJ_per_mol=2.0,
                pass_log10_KH=True,
                pass_Qads=True,
                pass_overall=True,
                reference_provenance_sha256="",
                notes="",
            )
            for i in range(4)
        ],
    ]
    outcome = assess_parity_outcome(rows)
    assert outcome["raspa3_skipped"] is True
    assert outcome["mofxdb_pass_count"] == 4
    assert outcome["overall_pass"] is True


def test_run_widom_evaluator_grows_supercell_for_small_cell() -> None:
    from widom_atlas.evaluator.component import ch4_trappe_ua
    from widom_atlas.evaluator.ff_loader import load_user_parameter_file
    from widom_atlas.evaluator.runner import run_widom_evaluator

    pytest.importorskip("ase")
    a = 5.0
    atoms = Atoms(
        symbols=["Si", "Si"],
        positions=[[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]],
        cell=Cell.fromcellpar([a, a, a, 90.0, 90.0, 90.0]),
        pbc=True,
    )
    upf = load_user_parameter_file(_toy_user_parameter_file())
    upf2 = upf.model_copy(
        update={
            "gas_sites": [
                upf.gas_sites[0].model_copy(update={"label": "CH4_ua"}),
            ],
        },
    )
    res = run_widom_evaluator(
        atoms=atoms,
        framework_name="ToySi",
        user_parameter_file=upf2,
        component=ch4_trappe_ua(),
        temperature_K=298.0,
        n_insertions=8,
        seed=0,
        r_cut_A=8.0,
        grid_mode="stochastic_uniform",
    )
    assert res.provenance["supercell_multiplier"] != [1, 1, 1]
