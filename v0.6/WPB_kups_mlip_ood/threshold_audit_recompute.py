import json, os, math, glob
import numpy as np
from ase.io import read

REPO = "/home/onur/projects/widom-atlas-release"
WPB = os.path.join(REPO, "v0.6/WPB_kups_mlip_ood")
CIF = os.path.join(REPO, "v0.6/figures/CHA_iza.cif")

GUEST, T_K, SEED, CUTOFF, N = "Ar", 298.15, 0, 6.0, 120
KB_EV, EV_TO_KJ = 8.617333262e-5, 96.48533212
RT = KB_EV * EV_TO_KJ * T_K
TWO16 = 2.0 ** (1.0 / 6.0)
KCAL_TO_K = 503.2195

AS_CODED = {"Si": (202.29, 3.826), "O": (30.19, 3.500), "Ar": (185.0, 3.405)}

UFF_XD = {"Si": (3.826, 0.402), "O": (3.500, 0.060), "Ar": (3.868, 0.185)}
GENUINE = {}
for el, xd in UFF_XD.items():
    GENUINE[el] = (xd[1] * KCAL_TO_K, xd[0] / TWO16)

print("=== parameter comparison (eps in K, sigma in A) ===")
for el in ("Si", "O", "Ar"):
    a, b = AS_CODED[el], GENUINE[el]
    print("  %-3s as-coded eps=%8.2f sig=%7.4f   genuine-UFF eps=%8.2f sig=%7.4f" % (el, a[0], a[1], b[0], b[1]))
print()

frame = read(CIF)
reps = [max(1, int(np.ceil(2 * CUTOFF / np.linalg.norm(frame.cell[i])))) for i in range(3)]
frame = frame.repeat(reps)
cell = np.array(frame.cell)
inv = np.linalg.inv(cell)
fpos = frame.get_positions()
fsym = np.array(frame.get_chemical_symbols())


def lj(by_elem, params):
    eps_g, sig_g = params[GUEST]
    u = 0.0
    for elem, dists in by_elem.items():
        eps_f, sig_f = params.get(elem, (0.0, 3.5))
        eps = (eps_f * eps_g) ** 0.5
        sig = 0.5 * (sig_f + sig_g)
        d = np.asarray(dists)
        d = d[(d > 1e-6) & (d <= CUTOFF)]
        sr6 = (sig / d) ** 6
        u += float(np.sum(4.0 * eps * (sr6 * sr6 - sr6)))
    return u


rng = np.random.default_rng(SEED)
recomputed = []
for i in range(N):
    cart = rng.random(3) @ cell
    dfrac = (fpos - cart) @ inv
    dfrac -= np.round(dfrac)
    dist = np.linalg.norm(dfrac @ cell, axis=1)
    md = float(dist.min())
    within = dist <= CUTOFF
    by = {}
    for s, r in zip(fsym[within], dist[within]):
        by.setdefault(s, []).append(r)
    recomputed.append({
        "i": i, "min_dist_A": round(md, 4),
        "U_as_coded": lj(by, AS_CODED) * KB_EV * EV_TO_KJ,
        "U_genuine": lj(by, GENUINE) * KB_EV * EV_TO_KJ,
    })

stored = json.load(open(os.path.join(WPB, "wpb_medium-mpa-0_per_insertion.json")))
dmax = max(abs(recomputed[k]["min_dist_A"] - stored[k]["min_dist_A"]) for k in range(N))
umax = max(abs(recomputed[k]["U_as_coded"] - stored[k]["U_classical_kJ"]) for k in range(N))
print("=== reproduction check against committed evidence ===")
print("  max |dmin difference|      = %.6f A" % dmax)
print("  max |U_classical difference| = %.4f kJ/mol   (as-coded params)" % umax)
print("  geometries reproduce exactly:", dmax < 1e-3)
print()

HARD_AS = 0.80 * 0.5 * (AS_CODED["O"][1] + AS_CODED["Ar"][1])
HARD_GEN = 0.80 * 0.5 * (GENUINE["O"][1] + GENUINE["Ar"][1])
print("=== overlap threshold ===")
print("  as-coded  0.80 * 0.5*(3.500 + 3.405) = %.4f A   [mixes UFF r_min with Talu sigma]" % HARD_AS)
print("  genuine   0.80 * 0.5*(%.4f + %.4f) = %.4f A" % (GENUINE["O"][1], GENUINE["Ar"][1], HARD_GEN))
print()

print("=== classical baseline at the four open sites (C1 comparison) ===")
far = sorted([r for r in recomputed if r["min_dist_A"] > 3.5], key=lambda r: r["min_dist_A"])
print("  dist(A)   U_as_coded   U_genuine")
for r in far:
    print("   %6.3f   %10.2f   %10.2f" % (r["min_dist_A"], r["U_as_coded"], r["U_genuine"]))
print()

FLOOR = -25.0


def analyse(tag, path, hard, params_label):
    rows = json.load(open(path))
    n = len(rows)
    ws, flagged_w, tot_w = [], 0.0, 0.0
    n_flag = 0
    clean = []
    for r in rows:
        u = r.get("U_mace_kJ")
        if not isinstance(u, (int, float)) or not math.isfinite(u):
            continue
        hard_f = r["min_dist_A"] < hard
        anom_f = (u < FLOOR) or (u < r.get("U_classical_kJ", 0.0) - 50.0)
        w = math.exp(min(-u / RT, 700.0))
        tot_w += w
        if hard_f:
            flagged_w += w
        if hard_f or anom_f:
            n_flag += 1
        else:
            clean.append(u)
    fw = flagged_w / tot_w if tot_w else float("nan")
    return n, n_flag, len(clean), fw, clean


print("=== flagged fraction and CLEAN sample size, as-coded vs corrected threshold ===")
for path in sorted(glob.glob(os.path.join(WPB, "wpb_*_per_insertion.json"))):
    nm = os.path.basename(path).replace("wpb_", "").replace("_per_insertion.json", "")
    n, nf_a, nc_a, fw_a, cl_a = analyse(nm, path, HARD_AS, "as")
    n, nf_g, nc_g, fw_g, cl_g = analyse(nm, path, HARD_GEN, "gen")
    print("  %-20s N=%3d" % (nm, n))
    print("       as-coded : flagged %3d, clean %3d, flagged-weight %.4f" % (nf_a, nc_a, fw_a))
    print("       corrected: flagged %3d, clean %3d, flagged-weight %.4f" % (nf_g, nc_g, fw_g))
    if cl_g:
        arr = np.array(cl_g)
        boot = [float(np.min(np.random.default_rng(s).choice(arr, size=len(arr), replace=True))) for s in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print("       clean min U = %.2f kJ/mol  bootstrap 95%% CI [%.2f, %.2f]  (n_clean=%d)"
              % (float(arr.min()), lo, hi, len(arr)))
    print()
