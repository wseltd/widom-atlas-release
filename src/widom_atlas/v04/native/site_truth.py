"""Site-truth verdict extractor.

Takes the lowest-energy Widom insertion found by `run_native_widom`
(`StrongestInsertion`) and the per-branch `site_truth.target_geometry`
block from `v04_case_matrix.yaml`, and produces a verdict-ready dict:

    {
      "passes_site_truth": bool,
      "tolerance_angstrom": float,
      "tolerance_degrees": float | None,
      "atlas_strongest_U_K": float,
      "distances": [
        {
          "label": "Mg_O_CO2_distance_angstrom",
          "target": 2.27,
          "atlas": 2.65,
          "delta": 0.38,
          "passes": False,
          "method": "min-image closest framework atom of type",
          "probe_atom_type": "O_co2",
          "framework_atom_type": "Mof_Mg",
        },
        ...
      ],
    }

Per-branch target_geometry keys understood:

  * Mg_O_CO2_distance_angstrom   → Mof_Mg vs O_co2 closest pair
  * Cu_O_CO2_distance_angstrom   → Cu vs O_co2 closest pair
  * Na_OC1_distance_angstrom     → Na vs O_co2 closest pair
  * Na_OC3_distance_angstrom     → Na vs O_co2 SECOND-closest pair
                                   (Wang 2009 reports a bridging
                                   configuration with two distinct Na-O
                                   distances; OC3 is the longer one)

If a branch carries `tolerance_angstrom: null` or `enabled: false`, the
extractor returns `{"passes_site_truth": None, "skipped": True, ...}`.
"""
from __future__ import annotations

from typing import Any

from .runner import StrongestInsertion


# Per-target-geometry key → (probe_atom_type, framework_atom_type, rank)
# rank=0 is the closest pair; rank=1 is the second closest of the same type-pair.
_TARGET_KEY_SPECS: dict[str, tuple[str, str, int]] = {
    # 1a, 1b — Mg-MOF-74 OMS-bound CO₂ (Wu 2010 / Queen 2014)
    "Mg_O_CO2_distance_angstrom": ("O_co2", "Mof_Mg", 0),
    # 2a — HKUST-1 open-Cu site I (Wu 2010 SI Tables S2/S3; canonical Cu-O(CO₂) ≈ 2.43 Å)
    # The 2a YAML target_geometry uses Wyckoff fractional positions; this
    # extractor reduces it to a single closest Cu-O(CO₂) distance for the
    # site-truth verdict.
    "Cu_O_CO2_distance_angstrom": ("O_co2", "Cu1", 0),
    # 2a uses the `target_` prefix in its target_geometry block — accept both.
    "target_Cu_O_CO2_distance_angstrom": ("O_co2", "Cu1", 0),
    # 5b — Na-Rho Lozinska 2012 atom positions: two distinct Na-O(CO₂) sites
    # The YAML uses `site_A_Na_O_distance_angstrom` (8MR end-on) and
    # `site_B_Na_O_distance_angstrom` (α-cage S6R end-on). Extractor maps both
    # to the closest and second-closest Na-O(CO₂) pair in the strongest insertion.
    "site_A_Na_O_distance_angstrom": ("O_co2", "Na", 0),
    "site_B_Na_O_distance_angstrom": ("O_co2", "Na", 1),
    # Legacy / alternative naming for the same physical 5b targets (kept for backward compatibility):
    "Na_OC1_distance_angstrom": ("O_co2", "Na", 0),
    "Na_OC3_distance_angstrom": ("O_co2", "Na", 1),
}


def _closest_pair_distance(
    strongest: StrongestInsertion,
    probe_atom_type: str,
    framework_atom_type: str,
    rank: int,
) -> tuple[int, int, float] | None:
    """Return (probe_idx, framework_idx, distance_A) for the `rank`-th-closest
    pair of (probe_atom_type, framework_atom_type) under PBC.

    Pairs are enumerated across every probe atom of the requested probe_atom_type
    (e.g. CO₂ has TWO O_co2 atoms — both are candidates; the rank=0 result is
    the lowest distance across all (probe_O_co2_atom × framework_Mg_atom) pairs).
    """
    candidates: list[tuple[int, int, float]] = []
    for pi, ptype in enumerate(strongest.probe_types):
        if ptype != probe_atom_type:
            continue
        for fi, ftype in enumerate(strongest.framework_types):
            if ftype != framework_atom_type:
                continue
            d = strongest.min_image_distance(
                strongest.probe_cartesian_angstrom[pi],
                strongest.framework_cartesian_angstrom[fi],
            )
            candidates.append((pi, fi, d))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[2])
    if rank >= len(candidates):
        return None
    return candidates[rank]


def extract_site_truth_verdict(
    strongest: StrongestInsertion | None,
    branch_site_truth_block: dict[str, Any],
) -> dict[str, Any]:
    """Convert a lowest-U insertion + the YAML site_truth block into a verdict dict.

    If the YAML disables site_truth for this branch (`enabled: false`) or no
    strongest insertion is provided, returns a `skipped: True` payload.
    """
    if not branch_site_truth_block or branch_site_truth_block.get("enabled") is False:
        return {"passes_site_truth": None, "skipped": True, "reason": "site_truth.enabled is False"}
    if strongest is None:
        return {"passes_site_truth": None, "skipped": True, "reason": "no strongest insertion available"}

    target_geom = branch_site_truth_block.get("target_geometry") or {}
    tolerance_A = branch_site_truth_block.get("tolerance_angstrom")
    distances: list[dict[str, Any]] = []
    overall_pass = True

    n_pass = 0
    n_fail = 0
    for key, target_val in target_geom.items():
        spec = _TARGET_KEY_SPECS.get(key)
        if spec is None:
            # Non-distance target-geometry keys (Wyckoff labels, fractional
            # coordinates, occupancies, target_loading) are recorded but do not
            # participate in the pass/fail decision. The native site-truth
            # extractor compares scalar pair distances only.
            distances.append({
                "label": key,
                "target": target_val,
                "atlas": None,
                "delta": None,
                "passes": None,
                "skipped": True,
                "reason": "no scalar-pair-distance extractor registered (descriptive key)",
            })
            continue
        probe_atom_type, framework_atom_type, rank = spec
        try:
            target_distance = float(target_val)
        except (TypeError, ValueError):
            distances.append({
                "label": key,
                "target": target_val,
                "atlas": None,
                "delta": None,
                "passes": None,
                "skipped": True,
                "reason": "non-numeric target distance",
            })
            continue
        result = _closest_pair_distance(strongest, probe_atom_type, framework_atom_type, rank)
        if result is None:
            distances.append({
                "label": key,
                "target": target_distance,
                "atlas": None,
                "delta": None,
                "passes": None,
                "skipped": True,
                "reason": (
                    f"no framework atom of type {framework_atom_type!r} or no probe "
                    f"atom of type {probe_atom_type!r} (rank {rank}) in the strongest "
                    f"insertion"
                ),
            })
            overall_pass = False
            continue
        _, _, atlas_distance = result
        delta = atlas_distance - target_distance
        tol = float(tolerance_A) if tolerance_A is not None else None
        passes_axis = (tol is not None) and (abs(delta) <= tol)
        if tol is not None:
            if passes_axis:
                n_pass += 1
            else:
                n_fail += 1
                overall_pass = False
        distances.append({
            "label": key,
            "target_angstrom": target_distance,
            "atlas_angstrom": atlas_distance,
            "delta_angstrom": delta,
            "tolerance_angstrom": tol,
            "passes": passes_axis,
            "method": "min-image closest probe-framework pair via native Widom strongest-U insertion",
            "probe_atom_type": probe_atom_type,
            "framework_atom_type": framework_atom_type,
            "rank": rank,
        })

    has_any_scalar_distance = (n_pass + n_fail) > 0
    return {
        "passes_site_truth": (overall_pass if has_any_scalar_distance else None),
        "n_scalar_distances_pass": n_pass,
        "n_scalar_distances_fail": n_fail,
        "tolerance_angstrom": tolerance_A,
        "tolerance_degrees": branch_site_truth_block.get("tolerance_degrees"),
        "atlas_strongest_U_K": strongest.U_K,
        "atlas_strongest_seed": strongest.seed,
        "atlas_strongest_temperature_K": strongest.temperature_K,
        "distances": distances,
        "reference_source_primary": (branch_site_truth_block.get("reference") or {}).get("primary"),
        "reference_doi_primary": (branch_site_truth_block.get("reference") or {}).get("primary_doi"),
    }
