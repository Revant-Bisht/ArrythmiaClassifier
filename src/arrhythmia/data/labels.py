"""PTB-XL label schema: SCP code → 5-class superdiagnostic mapping."""

from __future__ import annotations

from enum import Enum


class Superclass(str, Enum):
    """Five superdiagnostic classes used in the PTB-XL benchmark."""

    NORM = "NORM"  # Normal ECG
    MI = "MI"  # Myocardial Infarction
    STTC = "STTC"  # ST/T-wave Change
    CD = "CD"  # Conduction Disturbance
    HYP = "HYP"  # Hypertrophy

    @property
    def index(self) -> int:
        """Integer index — stable, matches the order declared above."""
        return list(Superclass).index(self)

    @property
    def full_name(self) -> str:
        return _FULL_NAMES[self]

    @property
    def clinical_description(self) -> str:
        return _CLINICAL_DESCRIPTIONS[self]

    def __str__(self) -> str:  # keeps str(Superclass.MI) == "MI"
        return self.value


_FULL_NAMES: dict[Superclass, str] = {
    Superclass.NORM: "Normal",
    Superclass.MI: "Myocardial Infarction",
    Superclass.STTC: "ST/T-wave Change",
    Superclass.CD: "Conduction Disturbance",
    Superclass.HYP: "Hypertrophy",
}

_CLINICAL_DESCRIPTIONS: dict[Superclass, str] = {
    Superclass.NORM: "No significant abnormality detected.",
    Superclass.MI: "Evidence of myocardial infarction — abnormal Q-waves or ST elevation "
    "consistent with ischaemic injury.",
    Superclass.STTC: "Non-specific ST-segment or T-wave changes — may indicate ischaemia, "
    "electrolyte disturbance, or medication effect.",
    Superclass.CD: "Abnormal conduction: bundle branch block, AV block, or accessory pathway.",
    Superclass.HYP: "Ventricular or atrial hypertrophy — increased voltage or axis deviation.",
}

_SUPERCLASS_CODES: dict[Superclass, frozenset[str]] = {
    Superclass.NORM: frozenset(
        {
            "NORM",
        }
    ),
    Superclass.MI: frozenset(
        {
            "AMI",
            "IMI",
            "ILMI",
            "ALMI",
            "LMI",
            "IPMI",
            "IPLMI",
            "PMI",
            "INJAS",
            "INJAL",
            "INJIN",
            "INJLA",
            "INJIL",
        }
    ),
    Superclass.STTC: frozenset(
        {
            "STD_",
            "STE_",
            "NST_",
            "INVT",
            "ANEUR",
            "EL",
            "LNGQT",
            "ISC_",
            "ISCAN",
            "ISCAL",
            "ISCIN",
            "ISCIL",
            "ISCAS",
            "ISCLA",
        }
    ),
    Superclass.CD: frozenset(
        {
            "LBBB",
            "RBBB",
            "IRBBB",
            "ILBBB",
            "CLBBB",
            "LAFB",
            "LPFB",
            "IVCD",
            "1AVB",
            "2AVB",
            "3AVB",
            "WPW",
        }
    ),
    Superclass.HYP: frozenset(
        {
            "LVH",
            "RVH",
            "SEHYP",
            "LAO/LAE",
            "RAO/RAE",
        }
    ),
}

SCP_CODE_MAP: dict[str, Superclass] = {
    code: superclass for superclass, codes in _SUPERCLASS_CODES.items() for code in codes
}

SUPERCLASSES: list[Superclass] = list(Superclass)

NUM_CLASSES: int = len(SUPERCLASSES)
