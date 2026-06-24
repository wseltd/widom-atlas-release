"""Lightweight container for spglib-derived host symmetry data.

Kept separate from :class:`widom_atlas.core.models.SymmetryGroup` because the
latter is a basin-grouping record while this dataclass captures the host
framework's space-group operations themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Confidence = Literal["high", "medium", "low", "uncertain"]


@dataclass(frozen=True)
class FrameworkSymmetry:
    """Space-group operations detected on the host framework."""

    space_group_number: int
    international_symbol: str
    hall_number: int
    rotations: np.ndarray
    translations: np.ndarray
    n_operations: int
    confidence: Confidence
    is_low_symmetry: bool
    is_triclinic: bool
    symprec: float
    angle_tolerance_deg: float
    notes: str = ""
    extra: dict[str, float] = field(default_factory=dict)
