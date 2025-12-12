"""PTB-XL label schema: maps SCP codes → 5 superdiagnostic classes."""

from __future__ import annotations

SUPERCLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Maps each PTB-XL SCP code to its superdiagnostic class.
# Source: ptbxl_database.csv `diagnostic_superclass` field + PTB-XL paper Table 1.
LABEL_MAP: dict[str, str] = {
    # Normal
    "NORM": "NORM",
    # Myocardial Infarction
    "AMI": "MI",
    "IMI": "MI",
    "ILMI": "MI",
    "ALMI": "MI",
    "INJAS": "MI",
    "LMI": "MI",
    "INJAL": "MI",
    "IPLMI": "MI",
    "IPMI": "MI",
    "INJIN": "MI",
    "INJLA": "MI",
    "PMI": "MI",
    "INJIL": "MI",
    # ST/T Change
    "STD_": "STTC",
    "ISCAL": "STTC",
    "ISCIN": "STTC",
    "ISCIL": "STTC",
    "ISCAS": "STTC",
    "ISCLA": "STTC",
    "ISC_": "STTC",
    "ISCAN": "STTC",
    "STE_": "STTC",
    "ANEUR": "STTC",
    "EL": "STTC",
    "NST_": "STTC",
    "INVT": "STTC",
    "LNGQT": "STTC",
    # Conduction Disturbance
    "LAFB": "CD",
    "IRBBB": "CD",
    "1AVB": "CD",
    "IVCD": "CD",
    "RBBB": "CD",
    "2AVB": "CD",
    "LBBB": "CD",
    "3AVB": "CD",
    "LPFB": "CD",
    "WPW": "CD",
    "ILBBB": "CD",
    "CLBBB": "CD",
    # Hypertrophy
    "LVH": "HYP",
    "LAO/LAE": "HYP",
    "RVH": "HYP",
    "RAO/RAE": "HYP",
    "SEHYP": "HYP",
}

SUPERCLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(SUPERCLASS_NAMES)}
