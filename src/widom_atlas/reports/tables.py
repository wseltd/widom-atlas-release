"""CSV + JSON serialisation of basin and symmetry-group tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from widom_atlas.core.models import Basin, SymmetryGroup

BASIN_CSV_COLUMNS = (
    "basin_id",
    "sample_count",
    "weight",
    "frac_centroid_a",
    "frac_centroid_b",
    "frac_centroid_c",
    "cart_centroid_x",
    "cart_centroid_y",
    "cart_centroid_z",
    "mean_energy_eV",
    "std_energy_eV",
    "accessible_fraction",
    "spread_A",
)


def _basin_to_csv_row(b: Basin) -> dict[str, object]:
    return {
        "basin_id": int(b.basin_id),
        "sample_count": int(b.count),
        "weight": float(b.weight),
        "frac_centroid_a": float(b.centroid_frac[0]),
        "frac_centroid_b": float(b.centroid_frac[1]),
        "frac_centroid_c": float(b.centroid_frac[2]),
        "cart_centroid_x": float(b.centroid_cart_A[0]),
        "cart_centroid_y": float(b.centroid_cart_A[1]),
        "cart_centroid_z": float(b.centroid_cart_A[2]),
        "mean_energy_eV": float(b.mean_energy_eV),
        "std_energy_eV": float(b.std_energy_eV),
        "accessible_fraction": float(b.accessible_fraction),
        "spread_A": float(b.spread_A),
    }


def write_basins_csv(basins: list[Basin], path: Path) -> None:
    """Write ``basins.csv`` with stable column ordering by ascending ``basin_id``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sorted_basins = sorted(basins, key=lambda b: int(b.basin_id))
    rows = [_basin_to_csv_row(b) for b in sorted_basins]
    df = pd.DataFrame(rows, columns=list(BASIN_CSV_COLUMNS))
    df.to_csv(p, index=False)


def write_basins_json(basins: list[Basin], path: Path) -> None:
    """Write ``basins.json`` with deterministic ordering."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sorted_basins = sorted(basins, key=lambda b: int(b.basin_id))
    payload = {"basins": [b.model_dump(mode="json") for b in sorted_basins]}
    text = json.dumps(payload, sort_keys=True, indent=2)
    p.write_text(text + "\n", encoding="utf-8")


def write_symmetry_groups_json(groups: list[SymmetryGroup], path: Path) -> None:
    """Write ``symmetry_groups.json`` with deterministic ordering."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sorted_groups = sorted(groups, key=lambda g: int(g.group_id))
    payload = {"symmetry_groups": [g.model_dump(mode="json") for g in sorted_groups]}
    text = json.dumps(payload, sort_keys=True, indent=2)
    p.write_text(text + "\n", encoding="utf-8")


__all__ = ["BASIN_CSV_COLUMNS", "write_basins_csv", "write_basins_json", "write_symmetry_groups_json"]
