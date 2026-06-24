"""Tests for ``widom_atlas.sites`` — literature site references + basin matching."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from widom_atlas.core.models import Basin
from widom_atlas.sites import (
    DEFAULT_SITE_MATCH_TOL_A,
    EXPECTED_SITES,
    BasinSiteMatch,
    match_basins_to_expected_sites,
    site_match_summary,
)


def _basin(idx: int, frac: tuple[float, float, float]) -> Basin:
    return Basin(
        basin_id=idx, count=10, weight=0.5, centroid_frac=frac,
        centroid_cart_A=(frac[0]*10.0, frac[1]*10.0, frac[2]*10.0),
        mean_energy_eV=-0.5, std_energy_eV=0.01,
        accessible_fraction=1.0, spread_A=0.1,
    )


def test_expected_sites_are_doi_referenced() -> None:
    assert EXPECTED_SITES, "EXPECTED_SITES must be non-empty"
    for s in EXPECTED_SITES:
        assert s.doi.startswith("10."), f"{s.material_id}/{s.gas}/{s.label}: doi must look like '10.xxxx/yyyy', got {s.doi!r}"
        assert s.citation, f"missing citation for {s.material_id}/{s.gas}/{s.label}"
        for x in s.centroid_frac:
            assert 0.0 <= x < 1.0


def test_match_returns_one_entry_per_basin() -> None:
    atoms = Atoms("H", positions=[[0,0,0]], cell=np.eye(3) * 10.0, pbc=True)
    basins = [_basin(0, (0.5, 0.5, 0.5)), _basin(1, (0.25, 0.25, 0.25))]
    out = match_basins_to_expected_sites(basins, atoms, "UiO-66", "CO2")
    assert len(out) == len(basins)
    assert all(isinstance(m, BasinSiteMatch) for m in out)


def test_match_high_confidence_when_basin_is_near_expected() -> None:
    atoms = Atoms("H", positions=[[0,0,0]], cell=np.eye(3) * 10.0, pbc=True)
    # Octahedral cage centre at frac (0.5, 0.5, 0.5) — basin near it.
    basins = [_basin(0, (0.495, 0.5, 0.5))]  # 0.05 Å away in cubic 10 Å
    out = match_basins_to_expected_sites(basins, atoms, "UiO-66", "CO2", site_match_tol_A=1.0)
    assert out[0].site_match_confidence == "high"
    assert out[0].nearest_site_label == "octahedral cage centre"
    assert out[0].distance_A is not None
    assert out[0].distance_A < 1.0


def test_match_falls_to_none_when_far() -> None:
    atoms = Atoms("H", positions=[[0,0,0]], cell=np.eye(3) * 10.0, pbc=True)
    basins = [_basin(0, (0.0, 0.0, 0.5))]  # > 4 Å from any UiO-66 site
    out = match_basins_to_expected_sites(basins, atoms, "UiO-66", "CO2", site_match_tol_A=1.0)
    assert out[0].site_match_confidence == "none"


def test_match_returns_none_label_when_no_expected_site_for_pair() -> None:
    atoms = Atoms("H", positions=[[0,0,0]], cell=np.eye(3) * 10.0, pbc=True)
    basins = [_basin(0, (0.5, 0.5, 0.5))]
    out = match_basins_to_expected_sites(basins, atoms, "Mg-MOF-74", "CH4")  # not registered
    assert out[0].site_match_confidence == "none"
    assert out[0].nearest_site_label is None
    assert out[0].nearest_site_doi is None
    assert "no expected site" in out[0].notes


def test_match_uses_minimum_image_distance() -> None:
    """In a cubic 10 Å cell with UiO-66 sites at (1/2,1/2,1/2) and (1/4,1/4,1/4),
    a basin at frac (0.05, 0.5, 0.5) lands closer to the tetrahedral site:
    d = sqrt(0.20² + 0.25² + 0.25²) × 10 ≈ 4.062 Å (tetrahedral),
    vs.   sqrt(0.45² + 0.00² + 0.00²) × 10 = 4.500 Å (octahedral).
    Both are > 4 × 1.0 Å tol, so confidence is 'none'."""
    atoms = Atoms("H", positions=[[0,0,0]], cell=np.eye(3) * 10.0, pbc=True)
    out = match_basins_to_expected_sites([_basin(0, (0.05, 0.5, 0.5))], atoms, "UiO-66", "CO2")
    assert out[0].site_match_confidence == "none"
    assert out[0].nearest_site_label == "tetrahedral cage centre"
    assert out[0].distance_A is not None
    assert abs(out[0].distance_A - 4.062) < 0.01


def test_site_match_summary_tallies_all_levels() -> None:
    matches = [
        BasinSiteMatch(0, (0.5,0.5,0.5), "site_a", "10.x/y", 0.5, "high"),
        BasinSiteMatch(1, (0.5,0.5,0.5), "site_a", "10.x/y", 1.5, "medium"),
        BasinSiteMatch(2, (0.5,0.5,0.5), "site_a", "10.x/y", 3.5, "low"),
        BasinSiteMatch(3, (0.5,0.5,0.5), "site_a", "10.x/y", 9.0, "none"),
    ]
    s = site_match_summary(matches)
    assert s == {"high": 1, "medium": 1, "low": 1, "none": 1}


def test_default_tol_documented() -> None:
    assert DEFAULT_SITE_MATCH_TOL_A == 1.0
