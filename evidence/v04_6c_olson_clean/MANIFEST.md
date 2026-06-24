# 6c clean-recipe evidence (v0.4.1 rebuild, 2026-06-18)

Promoted from validated scratch runs per operator approval (TASK 7, reconciliation A). This carries
equivalent rigour to a v0.4-audit-format run: a documented single-source recipe + structure +
convention, run on **two independent engines with cross-engine parity**.

## Recipe (single-source, oxygen-only — the silicon double-count removed)
- Guest–framework: **Ar–O 93.0 K / 3.335 Å** (Talu-Myers 2001; LB of Ar 119.8 & O 72.2, the
  silicon-inclusive effective oxygen — oxygen-only by construction).
- **Silicon: LJ = 0** (no Ar–Si). An active Si would double-count silicon (Talu already folds it into O).
- Convention: **24 Å cutoff, unshifted, analytical tail correction ON** (converged; matches 4c).
- Temperature: 298.15 K.

## Structure
- `structures/silicalite/MFI_SI_Olson.cif`, SHA-256 `18aea8dfab2c4e1b356cbc5b4da68eb6f7055c87aa4821c7500b7bc59d8e418c`.
- Provenance (accurate, not upgraded): **RASPA-curated, all-silica adaptation CITING Olson 1981**
  (RASPA2 zeolite library), cell 20.07/19.92/13.42 Å Pnma, clean Si96 O192 — NOT a paper-verified
  copy of Olson's coordinate table. Olson 1981 is the geometry Talu calibrated Ar–O 93.0 on.
- Corroboration: van Koningsveld 1987 (Acta Cryst. B43, 127), primary IUCr refinement — native 0.205 / RASPA3 0.205.

## Result (vs corrected reference K_H 0.200 [0.159, 0.252]; Q_st 15.7 [13.5, 17.9])
| engine | K_H (mol/kg/bar) | files |
|---|---|---|
| native (3 seeds, 2e5) | **0.200** (0.1992 / 0.1985 / 0.2032) | `olson_native_seed{0,1,2}.json` |
| RASPA3 3.0.29 (5000 cyc, tail on) | **0.199** | `raspa3_olson_tailon.{json,txt}` |
| **parity** | **0.5%** | |

Q_st (native) = 14.97 kJ/mol. Δlog₁₀K_H = 0.000, ΔQ_st = −0.73 → **Tier-A strict PASS (genuine)**.
Convergence (native, cut 12→18→24): 0.193 → 0.198 → 0.202 (plateaued). K_H/⟨W⟩ = 0.0219.
Inputs: `force_field.json` (oxygen-only), `simulation.json`. Structure sensitivity: Olson 0.200 / vK 0.205 (both PASS) / IZA-idealized 0.154 (outlier).
