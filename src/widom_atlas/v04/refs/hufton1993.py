"""T029: Hufton 1993 MFI+CH4 Henry-coefficient reference loader.

Reads the NIST isodb JSON for the pure-CH4 Hufton 1993 isotherm
(`fixtures/v04/hufton_1993/Isotherm3.json`) and computes K_H by a
Langmuir + Henry-virial joint fit.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Hufton1993Result:
    file_path: Path
    file_sha256: str
    n_points: int
    K_H_langmuir_mol_per_kg_per_bar: float
    K_H_virial_mol_per_kg_per_bar: float
    K_H_consensus_mol_per_kg_per_bar: float


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hufton1993_methane_298K(path: Path) -> Hufton1993Result:
    raw = json.loads(path.read_text())
    if raw.get("temperature") != 298:
        raise ValueError(f"Expected T=298K, got {raw.get('temperature')} in {path}")
    data = raw["isotherm_data"]
    pressures = [p["pressure"] for p in data]
    loadings = [p["total_adsorption"] for p in data]
    if not data:
        raise ValueError(f"empty isotherm in {path}")
    # Langmuir fit through (P, n)
    P = pressures
    n = loadings
    # Brent's method on K (positive) to minimise residual
    # Closed-form analytical 2-pt would be unstable; do a coarse + refined grid search
    best: tuple[float, float | None, float | None] = (math.inf, None, None)
    for K_guess in [0.05 * i for i in range(1, 200)]:
        # q_sat from least-squares given K: q = sum_i n_i*(1+K*P_i)/K*P_i  / N
        denom = sum((K_guess * p / (1.0 + K_guess * p)) ** 2 for p in P)
        numer = sum(ni * (K_guess * p / (1.0 + K_guess * p)) for ni, p in zip(n, P, strict=False))
        if denom <= 0:
            continue
        q_sat = numer / denom
        resid = sum((ni - q_sat * K_guess * p / (1.0 + K_guess * p)) ** 2
                    for ni, p in zip(n, P, strict=False))
        if resid < best[0]:
            best = (resid, K_guess, q_sat)
    _, K_lang, q_sat = best
    if K_lang is None or q_sat is None:
        raise RuntimeError("Langmuir fit failed")
    K_H_lang = q_sat * K_lang  # mol/(kg*bar)
    # Henry-virial fit: n = K_H*P + B*P^2 over the low-P subset (P<0.6 bar)
    P_lo = [p for p in P if p < 0.6]
    n_lo = [ni for ni, pp in zip(n, P, strict=False) if pp < 0.6]
    if len(P_lo) < 4:
        K_H_vir = K_H_lang
    else:
        # Least squares for K_H, B
        sum_p2 = sum(p ** 2 for p in P_lo)
        sum_p3 = sum(p ** 3 for p in P_lo)
        sum_p4 = sum(p ** 4 for p in P_lo)
        sum_np = sum(ni * pp for ni, pp in zip(n_lo, P_lo, strict=False))
        sum_np2 = sum(ni * pp ** 2 for ni, pp in zip(n_lo, P_lo, strict=False))
        det = sum_p2 * sum_p4 - sum_p3 ** 2
        K_H_vir = K_H_lang if abs(det) < 1e-20 else (sum_p4 * sum_np - sum_p3 * sum_np2) / det
    consensus = 0.5 * (K_H_lang + K_H_vir)
    return Hufton1993Result(
        file_path=path,
        file_sha256=_sha256(path),
        n_points=len(data),
        K_H_langmuir_mol_per_kg_per_bar=K_H_lang,
        K_H_virial_mol_per_kg_per_bar=K_H_vir,
        K_H_consensus_mol_per_kg_per_bar=consensus,
    )
