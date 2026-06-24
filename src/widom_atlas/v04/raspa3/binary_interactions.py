"""Per-branch BinaryInteractions cross-LJ overrides for RASPA3 force_field.json.

RASPA3 v3.0.29 JSON FF accepts only `lennard-jones`, `morse`, `none` types.
We use BinaryInteractions to override Lorentz-Berthelot mixing where the
literature force field specifies an explicit cross-pair (Talu-Myers,
García-Sánchez 2009 for Na-CO2, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrossPair:
    a: str
    b: str
    epsilon_K: float
    sigma_angstrom: float
    source: str


# Talu-Myers 2001 Colloids Surf A 187-188 83-93 (DOI 10.1016/S0927-7757(01)00628-8).
# Per-pair LJ for noble gas + silicalite (all-silica MFI) O_zeo cross.
TALU_MYERS_2001: dict[str, CrossPair] = {
    "Kr-O": CrossPair("Kr", "O", 109.6, 3.450, "Talu_Myers_2001_Table_4_Kr_in_silicalite"),
    "Ar-O": CrossPair("Ar", "O", 93.0, 3.335, "Talu_Myers_2001_Table_3_Ar_in_silicalite"),
}

# García-Sánchez 2009 J. Phys. Chem. C 113 8814-8820 — CO2 in Na-zeolites.
# Per-pair LJ for Na-CO2 and framework-O-CO2 cross.
GARCIA_SANCHEZ_2009: dict[str, CrossPair] = {
    "Na-C_co2": CrossPair("Na", "C_co2", 362.292, 3.32, "Garcia_Sanchez_2009_Na_C_co2"),
    "Na-O_co2": CrossPair("Na", "O_co2", 200.831, 2.758, "Garcia_Sanchez_2009_Na_O_co2"),
    "O-C_co2": CrossPair("O", "C_co2", 37.595, 3.511, "Garcia_Sanchez_2009_O_zeo_C_co2"),
    "O-O_co2": CrossPair("O", "O_co2", 78.98, 3.237, "Garcia_Sanchez_2009_O_zeo_O_co2"),
}


def binary_interactions_for_branch(
    branch_id: str, present_framework_elements: set[str], gas_species: str
) -> list[dict]:
    """Return BinaryInteractions entries for the given branch.

    Only emits entries when the framework actually contains the relevant
    elements (e.g., do not emit Na-CO2 cross for a branch without Na).
    """
    out: list[dict] = []
    pairs: list[CrossPair] = []
    if branch_id == "5b":
        # Na-Rho + CO2: García-Sánchez 2009 cross-pairs
        for key in ("O-O_co2", "O-C_co2", "Na-C_co2", "Na-O_co2"):
            if key in GARCIA_SANCHEZ_2009:
                cp = GARCIA_SANCHEZ_2009[key]
                if cp.a in present_framework_elements or cp.a in ("C_co2", "O_co2"):
                    pairs.append(cp)
    if branch_id in ("6b",) and gas_species == "Kr" and "O" in present_framework_elements:
        pairs.append(TALU_MYERS_2001["Kr-O"])
    if branch_id in ("6c", "6d") and gas_species == "Ar" and "O" in present_framework_elements:
        pairs.append(TALU_MYERS_2001["Ar-O"])
    for cp in pairs:
        out.append({
            "names": [cp.a, cp.b],
            "type": "lennard-jones",
            "parameters": [cp.epsilon_K, cp.sigma_angstrom],
            "source": cp.source,
        })
    return out
