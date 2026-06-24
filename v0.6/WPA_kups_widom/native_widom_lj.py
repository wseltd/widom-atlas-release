#!/usr/bin/env python3
"""Independent native LJ Widom for the cross-engine parity (WPA-A2).
Computes <exp(-beta U)> and K_H for a single neutral guest atom in an all-silica
zeolite, O-only LJ cross-pair, shifted-truncated at cutoff, no tail. Matches the
locked 6c convention (Ar-O_zeo). Used to validate kUPS's own Widom against an
independent implementation on the SAME potential. Pure numpy + ase (CPU)."""
import argparse, json, math
import numpy as np
from ase.io import read
from widom_atlas.v04.native.widom import WidomAccumulator

ap = argparse.ArgumentParser()
ap.add_argument("--cif", required=True)
ap.add_argument("--eps_K", type=float, required=True)      # guest-O cross epsilon (K)
ap.add_argument("--sigma_A", type=float, required=True)    # guest-O cross sigma (A)
ap.add_argument("--guest_mass", type=float, default=39.948)
ap.add_argument("--T", type=float, default=298.15)
ap.add_argument("--cutoff", type=float, default=12.0)
ap.add_argument("--N", type=int, default=2_000_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tail", action="store_true")  # analytical LJ tail (RASPA 6c uses FALSE)
ap.add_argument("--no_shift", action="store_true")  # truncated-not-shifted (mimic kUPS plain LJ)
ap.add_argument("--si_eps_K", type=float, default=0.0)    # Ar-Si cross eps (K); 0 = O-only
ap.add_argument("--si_sigma_A", type=float, default=0.0)  # Ar-Si cross sigma (A)
ap.add_argument("--out", required=True)
a = ap.parse_args()

KB_K = 1.0  # work in Kelvin for U; beta=1/T
frame = read(a.cif)
reps = [max(1, int(np.ceil(2 * a.cutoff / np.linalg.norm(frame.cell[i])))) for i in range(3)]
frame = frame.repeat(reps)
cell = np.array(frame.cell); inv = np.linalg.inv(cell)
sym = np.array(frame.get_chemical_symbols())
opos = frame.get_positions()[sym == "O"]          # O sites (explicit Talu-Myers cross-pair)
sipos = frame.get_positions()[sym == "Si"]        # Si sites (LB cross-pair; 0 eps => skip)
M_kg = sum(frame.get_masses()) * 1.66053906660e-27
V_m3 = abs(np.dot(cell[0], np.cross(cell[1], cell[2]))) * 1e-30
cut2 = a.cutoff ** 2
sr6c = (a.sigma_A / a.cutoff) ** 6
ushift_K = 4.0 * a.eps_K * (sr6c * sr6c - sr6c)   # O shift at cutoff
sr6c_si = (a.si_sigma_A / a.cutoff) ** 6 if a.si_sigma_A > 0 else 0.0
ushift_si_K = 4.0 * a.si_eps_K * (sr6c_si * sr6c_si - sr6c_si) if a.si_eps_K > 0 else 0.0
# analytical isotropic LJ tail for one guest in a uniform O background (RASPA convention):
# u_tail = (8/3) pi rho_O eps sigma^3 [ (1/3)(sig/rc)^9 - (sig/rc)^3 ]  (Kelvin)
rho_O = len(opos) / (V_m3 * 1e30)                  # O number density, A^-3
s3 = (a.sigma_A / a.cutoff) ** 3
u_tail_K = (8.0 / 3.0) * math.pi * rho_O * a.eps_K * a.sigma_A ** 3 * ((1.0 / 3.0) * s3 ** 3 - s3)
if not a.tail:
    u_tail_K = 0.0
if a.no_shift:
    ushift_K = 0.0; ushift_si_K = 0.0

rng = np.random.default_rng(a.seed)
energies_K = np.empty(a.N)
BATCH = 20000
done = 0
while done < a.N:
    n = min(BATCH, a.N - done)
    fr = rng.random((n, 3))
    cart = fr @ cell
    for i in range(n):
        d = opos - cart[i]
        fd = d @ inv; fd -= np.round(fd); dd = fd @ cell
        r2 = np.einsum("ij,ij->i", dd, dd)
        within = r2 < cut2
        if not np.any(within):
            energies_K[done + i] = 0.0; continue
        sr6 = (a.sigma_A ** 2 / r2[within]) ** 3
        u = np.sum(4.0 * a.eps_K * (sr6 * sr6 - sr6) - ushift_K)
        if a.si_eps_K > 0:
            dsi = sipos - cart[i]; fdsi = dsi @ inv; fdsi -= np.round(fdsi); ddsi = fdsi @ cell
            r2si = np.einsum("ij,ij->i", ddsi, ddsi); wsi = r2si < cut2
            if np.any(wsi):
                sr6s = (a.si_sigma_A ** 2 / r2si[wsi]) ** 3
                u += np.sum(4.0 * a.si_eps_K * (sr6s * sr6s - sr6s) - ushift_si_K)
        energies_K[done + i] = u + u_tail_K
    done += n

acc = WidomAccumulator(); acc.update(energies_K, beta_inv_K=a.T)
W = float(np.mean(np.exp(np.clip(-energies_K / a.T, -50, 700))))
KH_Pa = acc.K_H_mol_per_kg_per_Pa(T_K=a.T, M_framework_kg=M_kg, V_supercell_m3=V_m3)
out = {"engine": "native_lj_widom", "cif": a.cif, "N": a.N, "seed": a.seed,
       "eps_K": a.eps_K, "sigma_A": a.sigma_A, "cutoff": a.cutoff, "T": a.T,
       "tail_correction": bool(a.tail), "u_tail_K": float(u_tail_K),
       "si_eps_K": a.si_eps_K, "si_sigma_A": a.si_sigma_A,
       "mean_exp_negbU": W, "K_H_mol_per_kg_per_Pa": KH_Pa,
       "K_H_mol_per_kg_per_bar": KH_Pa * 1e5,
       "Q_st_kJ_per_mol": acc.Q_st_kJ_per_mol(T_K=a.T),
       "supercell_reps": reps, "V_supercell_A3": V_m3 * 1e30}
json.dump(out, open(a.out, "w"), indent=2)
print(json.dumps(out, indent=2))
