"""Native widom-atlas Widom evaluator.

Parallel to `raspa2/` and `raspa3/`, this module implements a self-contained
Widom-insertion engine in Python that supports functional forms RASPA3
v3.0.29 and RASPA2 v2.0.50 cannot represent directly:

  LJ_12_6              : V(r) = 4ε [(σ/r)^12 - (σ/r)^6]
  BUCKINGHAM_A_EXP_C6  : V(r) = A·exp(-B·r) - C/r^6
  DZUBAK_A_EXP_C5_D6   : V(r) = A·exp(-B·r) - C/r^5 - D/r^6

The Dzubak form is the reason this module exists at all — neither RASPA3
nor RASPA2 ships a tabulated/custom pair-potential path that fits the
A·exp − C/r^5 − D/r^6 functional form Dzubak 2012 uses for 1b
Mg-MOF-74 + CO2.

Before being used for any verdict-affecting branch (currently 1b), this
evaluator must pass the validation suite V1-V4:

  V1  native LJ vs RASPA3 on 6c MFI + Ar
  V2  native LJ vs RASPA3 on 6a MFI + CH4
  V3  native LJ/charged branch vs RASPA3 (electrostatics check)
  V4  native Buckingham vs RASPA2 on 1a Mg-MOF-74 + CO2

The full convention here matches v04 units throughout:
  - distances in Å
  - energies in K (k_B·T units) for the internal accumulator
  - K_H emitted in mol/(kg·Pa)
  - Q_st emitted in kJ/mol (positive exothermic)
"""
