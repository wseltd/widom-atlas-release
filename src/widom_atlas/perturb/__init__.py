"""Perturbation module: affine/isotropic/uniaxial strain + curated atom-removal defects."""

from widom_atlas.perturb.api import apply_perturbation
from widom_atlas.perturb.defects import DefectRecord, remove_atoms
from widom_atlas.perturb.strain import apply_strain

__all__ = [
    "DefectRecord",
    "apply_perturbation",
    "apply_strain",
    "remove_atoms",
]
