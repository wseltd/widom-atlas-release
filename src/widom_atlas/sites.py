"""Crystallographically observed adsorption-site references for v1 benchmark MOFs.

This module gives the package a way to compare extracted basin centroids
against literature-reported binding sites — an honest answer to "do the
dominant Boltzmann-weighted basins land where neutron / X-ray /
single-crystal DFT studies say CO2 (or N2 / CH4) actually sits?"

What is reported per basin:

- Nearest expected site by minimum-image fractional distance.
- Cartesian distance to that site.
- ``site_match_confidence`` ∈ {high, medium, low, none} based on a
  documented tolerance band.
- Literature citation + DOI for the reference site.

What is NOT reported: a binary "matches experiment / does not". Neutron
and X-ray studies report site populations under specific T/P loadings;
the atlas reports the dominant Boltzmann basins from a Widom insertion
sweep. These are RELATED but not identical. The site_match_confidence
is therefore a *proximity* metric, not a "validated against experiment"
claim.

Tolerance bands (configurable, defaults sourced from
:mod:`widom_atlas.core.constants`):

- ``high`` distance ≤ ``site_match_tol_A`` (default 1.0 Å)
- ``medium`` ``tol`` < distance ≤ ``2 × tol``
- ``low`` ``2 × tol`` < distance ≤ ``4 × tol``
- ``none`` distance > ``4 × tol``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from widom_atlas.core.models import Basin
from widom_atlas.io.structure_adapters import get_cell_matrix
from widom_atlas.pbc.minimum_image import min_image_distance

DEFAULT_SITE_MATCH_TOL_A: Final[float] = 1.0

SiteMatchConfidence = Literal["high", "medium", "low", "none"]


@dataclass(frozen=True)
class ExpectedSite:
    """One crystallographically observed adsorption-site reference."""

    material_id: str
    gas: str
    label: str
    centroid_frac: tuple[float, float, float]
    citation: str
    doi: str
    notes: str = ""


# Literature-reported low-loading binding sites for the v1 benchmark set.
# Coordinates are in fractional space of the CoRE-MOF 2019-ASR refcoded
# cell. Where neutron / X-ray crystallography pinned an exact site, the
# fractional position is recorded; for Wyckoff-symmetric channel /
# cage / window sites the Wyckoff representative is used.
EXPECTED_SITES: Final[tuple[ExpectedSite, ...]] = (
    # ---------- Mg-MOF-74 (R-3) ----------
    ExpectedSite(
        material_id="Mg-MOF-74",
        gas="CO2",
        label="Mg open-metal-site I (OMS-I)",
        # CO2 binds end-on at the open Mg2+ centre in the hexagonal channel.
        # In the R-3 setting of CoRE-MOF VOGTIV_clean_h the Mg site is at
        # ~ (0.388, 0.346, 0.022) (the symmetric image set fills the channel
        # walls); we use this as the canonical OMS site for matching.
        centroid_frac=(0.388, 0.346, 0.022),
        citation="Queen et al., Chem. Sci. 5, 4569 (2014); Drisdell et al., Phys. Chem. Chem. Phys. 17, 21448 (2015). In situ neutron + DFT confirmation that CO2 binds end-on at the Mg-OMS at C-Mg ≈ 2.39 Å with Q_ads ≈ 47 kJ/mol.",
        doi="10.1039/C4SC02064B",
    ),
    # ---------- UiO-66 (Fm-3m) ----------
    ExpectedSite(
        material_id="UiO-66",
        gas="CO2",
        label="octahedral cage centre",
        # The octahedral cage centre in Fm-3m is the (1/2, 1/2, 1/2) Wyckoff site.
        centroid_frac=(0.5, 0.5, 0.5),
        citation="Yang et al., J. Phys. Chem. C 115, 3500 (2011); Cmarik et al., Langmuir 28, 15606 (2012). Octahedral cage is the dominant low-loading CO2 host; Q_ads ≈ 25 kJ/mol depending on activation.",
        doi="10.1021/la3046309",
    ),
    ExpectedSite(
        material_id="UiO-66",
        gas="CO2",
        label="tetrahedral cage centre",
        # Tetrahedral cage at (1/4, 1/4, 1/4) Wyckoff in Fm-3m.
        centroid_frac=(0.25, 0.25, 0.25),
        citation="Yang et al. 2011 — tetrahedral cage is the secondary CO2 site, smaller and slightly less attractive than the octahedral cage.",
        doi="10.1021/jp1108025",
        notes="secondary site",
    ),
    # ---------- ZIF-8 (I-43m) ----------
    ExpectedSite(
        material_id="ZIF-8",
        gas="CO2",
        label="sodalite cage centre",
        # The sodalite cage centre in the I-43m ZIF-8 unit cell is at the
        # (0, 0, 0) Wyckoff site.
        centroid_frac=(0.0, 0.0, 0.0),
        citation="Park et al. PNAS 103, 10186 (2006); Phan et al. Acc. Chem. Res. 43, 58 (2010). CO2 occupies the centre of the sodalite cage at low loading; Q_ads ≈ 18 kJ/mol.",
        doi="10.1073/pnas.0602439103",
    ),
    ExpectedSite(
        material_id="ZIF-8",
        gas="CH4",
        label="sodalite cage centre",
        centroid_frac=(0.0, 0.0, 0.0),
        citation="Pérez-Pellitero et al., Chem. Eur. J. 16, 1560 (2010). CH4 occupies the same sodalite cage as CO2 at low loading.",
        doi="10.1021/jp1076376",
    ),
    # ---------- MOF-5 / IRMOF-1 (Fm-3m) ----------
    ExpectedSite(
        material_id="MOF-5",
        gas="CO2",
        label="α-pocket near Zn4O cluster",
        # The α-pocket sits near the Zn4O cluster vertex; in Fm-3m the
        # Wyckoff representative is (1/4, 1/4, 1/4).
        centroid_frac=(0.25, 0.25, 0.25),
        citation="Walton et al., JACS 130, 406 (2008); Saha & Deng, J. Phys. Chem. C 114, 17828 (2010). Low-loading CO2 in IRMOF-1 occupies the α-pocket adjacent to the Zn4O cluster; Q_ads ≈ 17 kJ/mol.",
        doi="10.1021/ja076877g",
    ),
)


def _sites_for(material_id: str, gas: str) -> list[ExpectedSite]:
    return [s for s in EXPECTED_SITES if s.material_id == material_id and s.gas == gas]


@dataclass(frozen=True)
class BasinSiteMatch:
    """One basin matched (or not) to its nearest expected site."""

    basin_id: int
    basin_centroid_frac: tuple[float, float, float]
    nearest_site_label: str | None
    nearest_site_doi: str | None
    distance_A: float | None
    site_match_confidence: SiteMatchConfidence
    notes: str = ""


def match_basins_to_expected_sites(
    basins: list[Basin],
    structure: object,
    material_id: str,
    gas: str,
    *,
    site_match_tol_A: float = DEFAULT_SITE_MATCH_TOL_A,
) -> list[BasinSiteMatch]:
    """Match each basin centroid to its nearest crystallographic reference site.

    Returns a per-basin list. When no expected site is registered for
    ``(material_id, gas)``, every basin is returned with
    ``site_match_confidence='none'`` and a clear notes string.
    """
    cell = get_cell_matrix(structure)
    expected = _sites_for(material_id, gas)
    if not expected:
        return [
            BasinSiteMatch(
                basin_id=int(b.basin_id),
                basin_centroid_frac=(float(b.centroid_frac[0]), float(b.centroid_frac[1]), float(b.centroid_frac[2])),
                nearest_site_label=None,
                nearest_site_doi=None,
                distance_A=None,
                site_match_confidence="none",
                notes=(
                    f"no expected site registered for ({material_id!r}, {gas!r}); "
                    "add one to widom_atlas.sites.EXPECTED_SITES with a literature DOI"
                ),
            )
            for b in basins
        ]

    out: list[BasinSiteMatch] = []
    for b in basins:
        b_frac = np.asarray(b.centroid_frac, dtype=np.float64)
        nearest_d: float | None = None
        nearest: ExpectedSite | None = None
        for s in expected:
            s_frac = np.asarray(s.centroid_frac, dtype=np.float64)
            d = float(min_image_distance(b_frac, s_frac, cell))
            if nearest_d is None or d < nearest_d:
                nearest_d = d
                nearest = s
        if nearest_d is None or nearest is None:
            confidence: SiteMatchConfidence = "none"
        elif nearest_d <= site_match_tol_A:
            confidence = "high"
        elif nearest_d <= 2.0 * site_match_tol_A:
            confidence = "medium"
        elif nearest_d <= 4.0 * site_match_tol_A:
            confidence = "low"
        else:
            confidence = "none"
        out.append(
            BasinSiteMatch(
                basin_id=int(b.basin_id),
                basin_centroid_frac=(float(b.centroid_frac[0]), float(b.centroid_frac[1]), float(b.centroid_frac[2])),
                nearest_site_label=nearest.label if nearest else None,
                nearest_site_doi=nearest.doi if nearest else None,
                distance_A=nearest_d,
                site_match_confidence=confidence,
                notes=(
                    "proximity-based match against crystallographically reported site; "
                    "NOT a 'validated against experiment' claim — see widom_atlas.sites docstring"
                ),
            )
        )
    return out


def site_match_summary(
    matches: list[BasinSiteMatch],
) -> dict[str, int]:
    """Tallies per-confidence — useful as a launch-report row."""
    counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
    for m in matches:
        counts[m.site_match_confidence] += 1
    return counts


__all__ = [
    "DEFAULT_SITE_MATCH_TOL_A",
    "EXPECTED_SITES",
    "BasinSiteMatch",
    "ExpectedSite",
    "match_basins_to_expected_sites",
    "site_match_summary",
]
