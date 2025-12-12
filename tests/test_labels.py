"""Tests for the label mapping logic."""

import torch
from arrhythmia.data.labels import LABEL_MAP, SUPERCLASS_INDEX, SUPERCLASS_NAMES


def test_superclass_names_count():
    assert len(SUPERCLASS_NAMES) == 5


def test_all_label_map_values_valid():
    for code, superclass in LABEL_MAP.items():
        assert superclass in SUPERCLASS_NAMES, f"{code} → unknown superclass {superclass}"


def test_superclass_index_complete():
    assert set(SUPERCLASS_INDEX.keys()) == set(SUPERCLASS_NAMES)
    assert list(SUPERCLASS_INDEX.values()) == list(range(5))
