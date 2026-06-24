"""Tests for PerturbationSpec discriminated union (T009)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from widom_atlas.core.models import PerturbationSpec


def test_perturbation_spec_isotropic_requires_magnitude() -> None:
    PerturbationSpec(kind="isotropic", magnitude=0.01, label="iso1")
    with pytest.raises(ValidationError):
        PerturbationSpec(kind="isotropic", label="iso-no-mag")


def test_perturbation_spec_uniaxial_requires_axis() -> None:
    PerturbationSpec(kind="uniaxial", magnitude=0.01, axis="a", label="ux")
    with pytest.raises(ValidationError):
        PerturbationSpec(kind="uniaxial", magnitude=0.01, label="ux-no-axis")


def test_perturbation_spec_affine_strain_matrix_shape() -> None:
    PerturbationSpec(
        kind="affine",
        strain_matrix=[[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        label="aff",
    )
    with pytest.raises(ValidationError):
        PerturbationSpec(kind="affine", strain_matrix=[[0.01, 0.0], [0.0, 0.0]], label="bad")
    with pytest.raises(ValidationError):
        PerturbationSpec(kind="affine", label="affine-no-matrix")


def test_perturbation_spec_atom_removal_indices_unique_nonneg() -> None:
    PerturbationSpec(kind="atom_removal", removed_atom_indices=[3, 7], label="rm")
    with pytest.raises(ValidationError):
        PerturbationSpec(kind="atom_removal", removed_atom_indices=[3, 3], label="dup")
    with pytest.raises(ValidationError):
        PerturbationSpec(kind="atom_removal", removed_atom_indices=[-1, 2], label="neg")
    with pytest.raises(ValidationError):
        PerturbationSpec(kind="atom_removal", removed_atom_indices=[], label="empty")


def test_perturbation_spec_rejects_mixed_fields() -> None:
    with pytest.raises(ValidationError):
        PerturbationSpec(
            kind="isotropic",
            magnitude=0.01,
            strain_matrix=[[0.0] * 3] * 3,
            label="mix",
        )
    with pytest.raises(ValidationError):
        PerturbationSpec(
            kind="atom_removal",
            removed_atom_indices=[1, 2],
            magnitude=0.01,
            label="mix",
        )


def test_perturbation_spec_roundtrip_json() -> None:
    s = PerturbationSpec(kind="uniaxial", magnitude=0.02, axis="b", label="ux", notes="check")
    js = s.model_dump_json()
    reloaded = PerturbationSpec.model_validate_json(js)
    assert reloaded == s
