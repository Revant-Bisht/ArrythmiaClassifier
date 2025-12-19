"""Tests for the label mapping and Superclass enum."""

import pytest

from arrhythmia.data.labels import (
    _SUPERCLASS_CODES,
    NUM_CLASSES,
    SCP_CODE_MAP,
    SUPERCLASSES,
    Superclass,
)


def test_num_classes():
    assert NUM_CLASSES == 5


def test_superclass_is_str():
    """Superclass must serialise as a plain string (for JSON/YAML compatibility)."""
    assert str(Superclass.MI) == "MI"
    assert Superclass.NORM == "NORM"


def test_superclass_index_stable():
    """Indices must be 0–4 in declaration order."""
    assert [cls.index for cls in Superclass] == list(range(5))


def test_superclass_has_metadata():
    for cls in Superclass:
        assert cls.full_name
        assert cls.clinical_description


def test_scp_codes_no_overlap():
    """Each SCP code must belong to exactly one superclass."""
    seen: dict[str, Superclass] = {}
    for superclass, codes in _SUPERCLASS_CODES.items():
        for code in codes:
            assert (
                code not in seen
            ), f"SCP code '{code}' appears in both {seen[code]} and {superclass}"
            seen[code] = superclass


def test_scp_code_map_covers_all_groups():
    """Derived SCP_CODE_MAP must contain every code from every group."""
    for superclass, codes in _SUPERCLASS_CODES.items():
        for code in codes:
            assert code in SCP_CODE_MAP
            assert SCP_CODE_MAP[code] is superclass


def test_superclasses_list_matches_enum():
    assert list(Superclass) == SUPERCLASSES


@pytest.mark.parametrize(
    "code,expected",
    [
        ("NORM", Superclass.NORM),
        ("AMI", Superclass.MI),
        ("LBBB", Superclass.CD),
        ("STD_", Superclass.STTC),
        ("LVH", Superclass.HYP),
    ],
)
def test_known_mappings(code, expected):
    assert SCP_CODE_MAP[code] is expected
