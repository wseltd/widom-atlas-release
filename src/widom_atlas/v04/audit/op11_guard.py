"""T036: OP11 guard for case 6a.

Validates that case 6a uses Hufton 1993 K_H + Dunne 1996 Q_st as the
EXPERIMENTAL reference, NOT the RASPA3 MFI+CH4 example output. Raises
OP11GuardViolation if the YAML drifts to simulator-as-truth.
"""
from __future__ import annotations


class OP11GuardViolation(RuntimeError):
    pass


HUFTON_DOI = "10.1002/aic.690390605"
DUNNE_DOI = "10.1021/la960495z"


def enforce_op11_guard(branch_6a: dict) -> None:
    refs = branch_6a.get("references") or {}
    kh = refs.get("K_H") or {}
    qst = refs.get("Q_st") or {}
    kh_source = str(kh.get("source", ""))
    qst_source = str(qst.get("source", ""))
    kh_doi = str(kh.get("primary_doi") or kh.get("source_doi", ""))
    qst_doi = str(qst.get("primary_doi") or qst.get("source_doi", ""))
    if "Hufton" not in kh_source and HUFTON_DOI not in kh_doi:
        raise OP11GuardViolation(
            f"6a K_H source must reference Hufton 1993 ({HUFTON_DOI}); "
            f"got source='{kh_source}' doi='{kh_doi}'"
        )
    if "Dunne" not in qst_source and DUNNE_DOI not in qst_doi:
        raise OP11GuardViolation(
            f"6a Q_st source must reference Dunne 1996 ({DUNNE_DOI}); "
            f"got source='{qst_source}' doi='{qst_doi}'"
        )
    # Verify the RASPA3 example output is labelled smoke_test_or_parity
    smoke = refs.get("smoke_test_or_parity") or {}
    if smoke:
        klass = str(smoke.get("classification", ""))
        if klass != "smoke_test_or_parity":
            raise OP11GuardViolation(
                f"6a smoke_test_or_parity block must classify as 'smoke_test_or_parity'; got '{klass}'"
            )
