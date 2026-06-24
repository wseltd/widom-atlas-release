"""T002: locked spec + case-matrix loader with SHA-256 verification.

Verifies V04_LOCKED_SPEC.md and v04_case_matrix.yaml against pinned digests
before any consumer reads them. Raises LockedDigestMismatch if either file
has been mutated since v04.2 was locked.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SPEC_SHA256 = "2c7bbf48cc53e8c9dadd9ec3cf266afe0be7d3d4fe6e0e81383f096c8f508720"
# Case matrix sha rolled forward through the v04 reference-erratum sequence:
#   v04.2 (locked 2026-05-11): b21ae86709e387a4983ecf7250cc56c7fb00dc0fc03f68e072007e7d3fa22de5
#   erratum 2026-05-14 (6c MFI+Ar K_H: 2.003 → 0.224 unit-correction, V04_UNIT_AUDIT.md):
#                     46d9f23f8c404bc978e77a47971928aefbe96adfcb3302068ef7f417ef16d2e6
#   erratum 2026-05-17 (5b Na-Rho scalar K_H + Q_st reclassified reference_blocked,
#                       V04_REFERENCE_AUDIT.md §9.1):
#                     404a0af5cd82edf2a542c40463cfb0e8e1efcb36ad6bcf354c44c0e4f4ac7ba1
#   erratum 2026-05-17 (6b MFI+Kr scalar reclassified reference_blocked_pending_primary,
#                       Talu-Myers + Golden-Sircar PDFs paywalled, V04_REFERENCE_AUDIT.md §9.2):
#                     439c199165409ac990766097d3a3857f1b4620989b924ab4cf5dcf61aa2cd646
#   erratum 2026-05-17 R2 (6b unit-corrected per operator-supplied Talu Table 4 verbatim:
#                          K_H 8.064 → 0.806 mol/(kg·bar) at 298.15 K, factor-of-10 unit
#                          correction; Q_st = 16.02 reclassified as
#                          fitted_van_t_Hoff_from_Talu_Table_4, not_calorimetric).
#                          No simulation/CIF/FF/site_truth change.
#   2026-05-19 addendum: 4c (Si-CHA refined-zeolite-FF, deferred) and 6e (MFI+CH4
#                        alternative-FF, deferred) added per operator priority 5 — branch
#                        count 15 → 17. Both refined-FF sibling branches reserve the slot
#                        but stay deferred until a primary source is in repo.
#   2026-05-19 pass-2 R3-audit-trail erratum: 6b Q_st classification upgraded from
#                       "reference_blocked_pending_golden_sircar_1994" to
#                       "reference_blocked_secondary_heat_value_found_zero_coverage_method_unresolved";
#                       independent ReadKong verification of Talu Table 4 Kr K_H rows
#                       recorded; Ads@UC secondary corroboration -ΔH = 16.39 kJ/mol at
#                       q = 0.58 mol/kg recorded as finite-loading evidence (NOT zero
#                       coverage). VALUE-UNCHANGED audit-trail erratum: Q_st still
#                       REFERENCE_BLOCKED on the strict axis; K_H still 0.806 mol/(kg·bar).
#   2026-05-19 pass-3 R4 erratum: 6b Q_st PROMOTED from REFERENCE_BLOCKED to
#                       reference_anchored_secondary per operator directive. Value =
#                       16.39 kJ/mol (Ads@UC). Provenance: secondary_anchored / cross_cited
#                       (NOT primary_pdf_verbatim). Acceptance window re-centered:
#                       [14.0, 18.0] → [14.39, 18.39]. 6b now contributes a strict
#                       PASS/FAIL verdict on the Q_st axis (no longer REFERENCE_BLOCKED).
#   2026-05-19 pass-4 R5 erratum: 1c PROMOTED from deferred_not_in_v04_verdict to
#                       locked_strict_executed per operator directive. Becker 2018
#                       (DOI 10.1021/acs.jpcc.8b08639) reduced-LJ NON-POLARIZABLE
#                       approximation activated via native_widom_v04 backend. Becker
#                       Table S3 framework LJ + charges + Table S7 CO2 LJ + charges
#                       embedded verbatim; CO2 polarizabilities recorded but NOT used in
#                       energy (no induced-dipole support in native backend). 1c shares
#                       Mason DSL K_H = 381 / Q_st = 42 reference with 1a/1b. Strict
#                       denominator: 9 → 10.
#   2026-05-19 pass-5 R6 erratum: 1d ADDED as new locked_strict_executed branch per
#                       operator directive ("Lin/Mercado Model 4 table is also fully
#                       available"). Mercado, Vlaisavljevich, Lin et al. 2016 JPC C
#                       (DOI 10.1021/acs.jpcc.6b03393, github.com/rociomer/DFT-derived-force-field)
#                       Model 4 DFT-derived FF: mixed Buckingham (8 O_CO2-framework pairs)
#                       + LJ (9 C_CO2-framework + O_CO2-H pairs), EPM2 CO2, global scaling
#                       Sg = 1.7 already applied. Activated via native_widom_v04 backend.
#                       Strict denominator: 10 → 11.
#   2026-05-19 pass-6 R7 erratum: 2b PROMOTED from blocked_pending_Ongari_SI_coefficients
#                       to locked_strict_executed per operator directive ("HKUST-1 2b is
#                       now actionable"). Ongari 2017 SI section 4 coefficients now in
#                       repo (docs/research/jp7b02302_si_001.pdf): V(r) = A*exp(-B*r) -
#                       C6/r^6 - C8/r^8 RASPA generic potential for Cu-O(CO2) ONLY
#                       (A=1.0e8 K, B=4.19 1/A, C6=3.196e4 K*A^6, C8=5.0e6 K*A^8, hard
#                       core r<1.8 A); UFF base LJ + L-B for all other cross-pairs;
#                       REPEAT charges Table S12; TraPPE-CO2; 13 A truncated cutoff; no
#                       tail; Ewald. Strict denominator: 11 → 12.
#   2026-05-19 pass-7 forensic audit: 2b forensic verified — OMS axial-min reproduces
#                       Ongari target -27.3 kJ/mol within +2.21 kJ/mol; Widom rank-4
#                       lowest at Cu-O = 2.25 A gives -27.42 kJ/mol matching target to
#                       0.1 kJ/mol. Implementation correct. NO VALUE CHANGE.
#   2026-05-19 pass-8 source-provenance update: bibliography clean-up — TraPPE-zeo
#                       record verified as DOI 10.1021/jp4074224 (pages 24375-24387) per
#                       Siepmann group / Experts@Minnesota; pre-pass-8 YAML used wrong
#                       DOIs 10.1021/jp4063088 (4c) and 10.1021/jp4099262 (6e) which were
#                       packet-flagged as citation-data inconsistencies. Yang 2011 UiO-66
#                       charge provenance corrected to DOI 10.1021/jp202633t (JPC C 115,
#                       13768) instead of Yang/Maurin Chem Eur J 17, 8882. Maia 2023
#                       UiO-66 partial Table 2 (4 atom rows Zr/O1/O25/O29) ingested into
#                       3b as partial-data-accessible. RASPA generic eq 3.94 full form
#                       documented in OngariAExpC6C8 class docstring + YAML 2b
#                       raspa_generic_eq_3p94_pass_8_provenance field. NO VERDICT CHANGES.
#                       Europe PMC supplementaryFiles ZIP for PMC5523115 confirmed as
#                       independent verification path for Ongari SI (R7 values match
#                       SI exactly).
#   2026-05-28 5b finalisation packet: Lozinska 2012 SI + CIF + Brandani 2021 review +
#                       Brandani Excel archived under docs/research/.../5b_na_rho/.
#                       5b K_H + Q_st FINALISED as method_blocked_or_reference_blocked
#                       after inspection of SI/CIF/Brandani review/Brandani Excel
#                       confirmed NO admissible Henry-regime scalar for pure Na9.8-Rho.
#                       Site-truth stays active using Lozinska CO2-loaded structures.
#                       Lozinska main JACS PDF (purchased ACS) and operator Pasted text
#                       file NOT YET in repo; reserved paths documented in manifest.
#                       Liang 2021 RHO PSA archived under operator_packets/ as
#                       REJECTED-as-5b-scalar audit-trail context. v0.5 candidate
#                       replacement-scalar branches (5c_13X_CO2, 5d_Na_ZK5_CO2)
#                       documented in YAML — these are REPLACEMENTS, NOT 5b validation.
#                       NO VERDICT CHANGE: 5b stays in atlas, scalar blocked, site-truth
#                       active. v0.4 strict tally unchanged.
#   2026-05-28 addendum: Lozinska 2012 JACS main paper PDF now archived at
#                       docs/research/.../5b_na_rho/lozinska_2012_jacs/
#                       lozinska_2012_jacs_main.pdf, SHA-256
#                       1373dff266e6592adbefed8c3a83aa993476f0e59d903710bea4111bd73079dd.
#                       Title + authors + journal verbatim verified from page 1.
#                       Marked private-provenance, do-not-redistribute-publicly.
#                       Abstract confirms operator's evidence summary: target composition
#                       Na9.8Al9.8Si38.2O96; finite uptake 3.07 mmol/g @ 0.1 bar; ZLC
#                       desorption kinetics quantified; NO equilibrium Henry scalar.
#                       5b verdict UNCHANGED — scalar still method_blocked_or_reference_blocked.
#   2026-05-28 pass-9 new directive: operator clarified paper priorities (Maia 2023 +
#                       Bai 2013 actionable IF tables in repo; 5b is the only HARD
#                       missing literature source; Brandani Excel + Liang 2021 are
#                       provenance/comparison only). Operator authorised recording
#                       Liang 2021 Table 3 K = q_m*b = 5.93 mol/(kg·bar) at 298 K as
#                       COMPARISON-ONLY sanity check (NOT 5b validation; 4 caveats
#                       still hold: 78% Na-exchanged, PSA 0-10 bar not Henry, no
#                       trapdoor state specified, no zero-coverage Q_st). Operator
#                       authorised PERMANENT 5b scalar METHOD_BLOCKED classification
#                       and 5c replacement-scalar branch planning with 3-category
#                       priority list (A Liang Na-RHO comparison; B rigid cationic
#                       zeolite Na-ZSM-5/NaY/NaA/NaX13X/Na-ZK5; C rigid MFI/CHA).
#                       Maia 2023 + Bai 2013 PDFs NOT in repo as of pass-9; 3b/4c/6e
#                       execution conditional NOT met — stays deferred. NO VERDICT
#                       CHANGE. Strict tally unchanged.
#   2026-06-01 final pivot v04.3: 3b PROMOTED to locked_strict_executed_multi_variant
#                       (UA/UAq/EHq) via native_widom_v04 + Maia 2023 archived PDF.
#                       4c and 6e RECLASSIFIED status:
#                       blocked_pending_Bai_main_parameter_tables (with
#                       exact_missing_artefact field; do_not_invent_parameters: true).
#                       4 new 5c reference-audited branches added: 5c_NaZK5_CO2_303K
#                       (PRIMARY), 5c_Zeolite5A_CaA_CO2_298K, 5c_Zeolite13X_NaX_CO2_273K,
#                       5c_Zeolite4A_NaA_CO2_273K — all reference_audited with K_H
#                       sensitivity ranges + Q_st cited; atlas execution pending
#                       cation-CIF + FF lock (cation-containing zeolite CIFs are NOT
#                       in the v0.4 repo). NONE is a validation of 5b. 5b stays
#                       METHOD_BLOCKED scalar / site-truth active. Shah 2015 MFI+CH4
#                       smoke_test_or_parity K_H ≈ 0.60 recorded on 6e branch as
#                       cross-check ONLY. Strict denominator: 12 → 13 (3b UA counted).
#   2026-06-01 v04.3 ADDENDUM (post deep-research): added 5b ensemble_mismatch_known
#                       _open_problem block per deep-research conclusion that Na-Rho's
#                       closed-vs-open-state mismatch is NOT a widom-atlas bug.
#                       Documents principled paths (FH-MC via FEASST, Witman 2018;
#                       open-state GCMC) as research-project-post-v0.5 scope. Two-
#                       tier threshold system landed in src/widom_atlas/v04/thresholds.py:
#                       Tier A = historical ±0.10/±2.0 (internal regression gate);
#                       Tier B = per-system literature-scatter bands (±0.40/±8.0 for
#                       OMS/defective MOFs, ±0.20/±4.0 for Si-CHA, ±0.15/±3.0 for MFI).
#                       Two-tier re-verdict tally: 2 PHYSICAL_PASS (6c MFI+Ar control,
#                       3b_EHq UiO-66 Maia explicit-H, both Δlog10 < 0.05 of reference
#                       within ±0.40 band) + 12 PHYSICAL_FAIL. Bayesian log-space
#                       Z-score classification added per branch. NO CIF/FF/SCALAR
#                       VALUE CHANGES — only the threshold-band reclassification.
#   2026-06-01 v04.3 4c + 6e BAI 2013 EXECUTED: PROMOTED both branches from
#                       blocked_pending_Bai_main_parameter_tables to locked_strict_executed.
#                       Bai 2013 TraPPE-zeo Si/O LJ + framework charges
#                       (Si: epsilon=22.0 K, sigma=2.3 A, q=+2.05 e;
#                        O:  epsilon=53.0 K, sigma=3.3 A, q=-1.025 e)
#                       verified PRIMARY-ANCHORED via RASPA3 v3.0.29 conda-forge
#                       bundled force-field JSONs (Snurr/Dubbeldam distribution
#                       with explicit Bai 2013 DOI source citations on every entry).
#                       Paywalled main paper PDF NOT required for this provenance
#                       path. EXECUTION: 4c K_H = 2.22 ± 0.14 vs Maghsoudi 2013 ref 2.43
#                       -> Δlog10 = -0.039 STRICT PASS; Q_st = 22.99 ± 0.33 vs 21.0
#                       -> ΔQ = +1.99 STRICT PASS. FIRST ALL-SILICA ZEOLITE + CO2
#                       REFINED-FF BRANCH PASSING THE ORIGINAL STRICT THRESHOLDS.
#                       6e K_H = 0.45 ± 0.006 vs Hufton 1993 ref 0.89 -> Δlog10 = -0.296
#                       FAIL; Q_st = 18.76 ± 0.03 vs Dunne 1996 ref 20.9 -> ΔQ = -2.14
#                       FAIL; but Bayesian |Z| = 1.48 (AGREEMENT_WITHIN_2_SIGMA) and
#                       cross-checks Shah 2015 simulated K_H ~ 0.60 (atlas / Shah ratio
#                       = 0.75) -- DOCUMENTED TraPPE-zeo + TraPPE-UA CH4 under-binding,
#                       NOT a widom-atlas implementation bug. Tail correction was
#                       LOAD-BEARING on 4c PASS (without tail K_H = 1.98 FAILs strict;
#                       with tail K_H = 2.22 PASSES). Strict denominator: 13 -> 15.
#                       Headline tally: 2 Tier A PASS (6c MFI+Ar, 4c Si-CHA+CO2 NEW);
#                       3 Tier B PHYSICAL_PASS (6c, 3b_EHq, 4c).
#   2026-06-01 6e/4c audit-trail amendment: documented full chronology of the
#                       deep researcher's 4 sequential claims about Bai 2013 framework
#                       charges (+1.20/-0.60 -> +1.30/-0.65 -> +1.50/-0.75 -> fictitious
#                       file path), all rejected. Three independent RASPA-distributed
#                       sources (RASPA3 v3.0.29 JSON, RASPA2 v2.0.50 .def, RASPA2 MFI_SI.cif
#                       Dubbeldam 2011) agree on +2.05/-1.025 with explicit Bai 2013
#                       citations. Experimental falsification test (K_H = 2.22 vs
#                       Maghsoudi 2.43 PASS) confirms RASPA-bundled values correct.
#   2026-06-18 v0.4.1: 6c REBUILT CLEAN. The locked 6c double-counted silicon (Talu Ar-O 93.0 K
#                       already folds Si into O -- Talu Table 1 O-O 72.2, Ar-O 93.0 = LB(Ar, O);
#                       the recipe then added an explicit Ar-Si from an active TraPPE-zeo Si
#                       self-parameter, 22.0 K) AND compared to a mis-derived reference 0.224.
#                       Rebuilt single-source pure-Talu OXYGEN-ONLY (Si LJ = 0), converged 24 A +
#                       tail, on Talu's own (Olson) geometry -> reproduces the CORRECTED reference
#                       0.200 exactly (native 0.200 / RASPA3 0.199, parity 0.5%; Delta-log10 ~ 0.000).
#                       Reference 0.224 -> 0.200 (Dunne 1996 B = 4.35 cm^3/g, K_H = B/kT at isotherm T;
#                       0.224 used the STP molar volume + mis-cited B to Talu Table 3). 6c stays a
#                       Tier-A strict pass (genuine). Old hybrid evidence kept, marked SUPERSEDED.
#                       v0.4 strict tally UNCHANGED (2/15). New file hash below.
# v0.4.2 (2026-06-21): 2a/3a genuine-UFF rebuild. A generator bug stamped source="UFF" on MOF
#                       framework C/H/O that were actually Harris-Yung-C / TraPPE-zeo-O / wrong-H;
#                       fixed (input_writer._uff_lj_K, fail-loud; zeolite path byte-identical).
#                       Rebuilt RASPA3: 3a K_H 6.27 / Q_st 25.6 -> Tier-A STRICT (was -0.802);
#                       2a K_H 8.3-10.8 / Q_st 33.9 -> Tier-B (was -0.759). 4c demoted Tier-A->Tier-B
#                       (Q_st engine-marginal; native<->RASPA3 ~9% on K_H). Native charged-MOF Ewald
#                       disclosed-unreliable (oblique-cell minimum-image; not fixed). 1c Becker defect
#                       disclosed (framework = plain UFF + Mg placeholder). Parity claims corrected.
#                       Tally: strict 2/15 (6c,3a); Tier-B 5/15 (6c,3a,2a,4c,3b_EHq). New hash below.
CASE_MATRIX_SHA256 = "b0571780a4794be0fabb2a7d47a093fcb8dab07c4355467bb2fabe1a6a085664"

DEFAULT_SPEC_PATH = Path("V04_LOCKED_SPEC.md")
DEFAULT_CASE_MATRIX_PATH = Path("v04_case_matrix.yaml")


class LockedDigestMismatch(RuntimeError):
    """A locked input file's sha256 does not match the pinned digest."""


@dataclass(frozen=True)
class LockedSpec:
    path: Path
    sha256: str
    text: str


@dataclass(frozen=True)
class LockedCaseMatrix:
    path: Path
    sha256: str
    version: str
    cases: list[dict[str, Any]]
    raw: dict[str, Any]


def verify_digest(path: Path, expected_sha256: str) -> str:
    """Return the file's actual sha256; raise LockedDigestMismatch if it differs."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected_sha256:
        raise LockedDigestMismatch(
            f"{path}: sha256 mismatch (got {got}, expected {expected_sha256})"
        )
    return got


def load_locked_spec(path: Path = DEFAULT_SPEC_PATH) -> LockedSpec:
    """Load V04_LOCKED_SPEC.md after verifying its sha256."""
    got = verify_digest(path, SPEC_SHA256)
    return LockedSpec(path=path, sha256=got, text=path.read_text())


def load_locked_case_matrix(
    path: Path = DEFAULT_CASE_MATRIX_PATH,
) -> LockedCaseMatrix:
    """Load v04_case_matrix.yaml after verifying its sha256."""
    got = verify_digest(path, CASE_MATRIX_SHA256)
    with path.open() as fp:
        raw = yaml.safe_load(fp)
    version = str(raw.get("schema_version", "")) or "unknown"
    cases = list(raw.get("cases", []))
    return LockedCaseMatrix(
        path=path, sha256=got, version=version, cases=cases, raw=raw
    )
